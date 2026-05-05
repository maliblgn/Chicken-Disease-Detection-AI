import os
import sys

# tools/ klasöründen çalıştırıldığında proje kökünü ayarla
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Ayarlar
EGITIM_YOLU = "Spektrogramlar/Veri_Egitim"
TEST_YOLU = "Spektrogramlar/Veri_Test"
GORSEL_KLASORU = "Gorseller"

BATCH_SIZE = 32
IMG_SIZE = (128, 128)
EPOCHS = 50 

if not os.path.exists(GORSEL_KLASORU):
    os.makedirs(GORSEL_KLASORU)

print("1. V2 (3-KANALLI) SPECTROGRAMLAR YÜKLENİYOR...\n" + "="*50)

try:
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        EGITIM_YOLU,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode='int'
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_YOLU,
        shuffle=False,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode='int'
    )
except Exception as e:
    print(f"Veri Yükleme Hatası: {e}\nLütfen 'spektrogram_olusturucu.py' dosyasının başarıyla bittiğinden emin olun.")
    exit()

class_names = train_dataset.class_names
print(f"Tanınan Sınıflar: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("\n2. V2 AKADEMİK CSCNN MİMARİSİ İNŞA EDİLİYOR...\n" + "="*50)

l2_reg = regularizers.l2(0.001)

# V2: Batch Normalization ile Güçlendirilmiş Çok Kanallı Mimari
model = models.Sequential([
    # Giriş (128x128 Piksel, 3 Kanal: Mel, Delta, Delta2)
    layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # 1. Evrişim Bloğu
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(), # YENİ: Veri dengesini sağlar, hızlı öğrenir
    layers.MaxPooling2D((2, 2)),
    
    # 2. Evrişim Bloğu
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # 3. Evrişim Bloğu
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # 4. Evrişim Bloğu
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    # Dual Dropout 1
    layers.Dropout(0.3),
    
    layers.Dense(128, activation='relu', kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    
    # Dual Dropout 2
    layers.Dropout(0.5),
    
    layers.Dense(len(class_names), activation='softmax')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8, # V2 verisi zorlu olduğu için sabrı biraz artırdık
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

print("\n3. V2 AKILLI EĞİTİM (CSCNN TRAINING) BAŞLATILIYOR...\n" + "="*50)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

print("\n4. V2 MODEL DEĞERLENDİRMESİ (ZİRVE AĞIRLIKLARLA)\n" + "="*50)
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print(f"\n🚀 YENİ CSCNN GERÇEK DOĞRULUK SKORU (TEST SETİ): % {test_acc * 100:.2f}")

print("\n5. V2 KARMAŞIKLIK MATRİSİ ÇİZİLİYOR...\n" + "="*50)

y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

y_test = np.concatenate([y for x, y in test_dataset], axis=0)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title(f'V2 3-Kanallı CSCNN Matrisi (Accuracy: %{test_acc*100:.1f})')
plt.ylabel('Gerçek Sınıflar')
plt.xlabel('Tahmin Edilen Sınıflar')
plt.tight_layout()

cm_path = os.path.join(GORSEL_KLASORU, "Karmasiklik_Matrisi_CNN_V2.png")
plt.savefig(cm_path)
plt.close()

print(f" -> V2 CNN Matrisi {cm_path} konumuna kaydedildi.")

print("\n6. V2 CSCNN MODELİ ARAYÜZ İÇİN DİSKE KAYDEDİLİYOR...\n" + "="*50)
kayit_yolu = os.path.join("models", "tavuk_cnn_modeli.keras")
model.save(kayit_yolu)
print(f"BAŞARILI: Hız ve İvmeyi Okuyabilen Ağır Siklet Model '{kayit_yolu}' diske yazıldı!")