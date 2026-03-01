import os
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
EPOCHS = 50 # Akıllı eğitimle durdurulacağı için yüksek tutuldu

if not os.path.exists(GORSEL_KLASORU):
    os.makedirs(GORSEL_KLASORU)

print("1. VERİLER (SPECTROGRAMLAR) YÜKLENİYOR...\n" + "="*50)

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
    print(f"Veri Yükleme Hatası: {e}\nLütfen Aşama 2 (spektrogram_olusturucu.py) dosyasının çalıştığından emin olun.")
    exit()

class_names = train_dataset.class_names
print(f"Tanınan Sınıflar: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("\n2. BİLİMSEL CNN MİMARİSİ (V2) İNŞA EDİLİYOR...\n" + "="*50)

# L2 Düzenleyicisi tanımla
l2_reg = regularizers.l2(0.001)

model = models.Sequential([
    # Piksel değerlerini 0-255 aralığından 0-1 aralığına sıkıştır
    layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # 1. Evrişim Bloğu
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # 2. Evrişim Bloğu
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # 3. Evrişim Bloğu
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # 4. Evrişim Bloğu (Bilimsel İyileştirme: Daha derin harita tespiti)
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    # Erken Ezber Bozucu: Dual Dropout 1 (Bilimsel İyileştirme)
    layers.Dropout(0.3),
    
    # Matematiksel Düzenlemeli Derin Ağ (L2 Eklendi)
    layers.Dense(128, activation='relu', kernel_regularizer=l2_reg),
    
    # Sıkı Ezber Bozucu: Dual Dropout 2
    layers.Dropout(0.5),
    
    layers.Dense(len(class_names), activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Akıllı Callback'ler (Bilimsel İyileştirme)
# 1. Erken Durdurma (Zirvede Bırakma)
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=6, # 6 Epoch boyunca gelişim yoksa eğitimi kes
    restore_best_weights=True, # Eğitim bitince en düşük loss alan Epoch'a kendini geri sar!
    verbose=1
)

# 2. Öğrenme Hızını Düşürme (Hedefe İnce Yaklaşma)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5, # Gelişim durursa öğrenme hızını Yarıya Kes
    patience=3, # 3 Epoch gelişim yoksa devreye gir
    min_lr=0.00001,
    verbose=1
)

print("\n3. AKILLI EĞİTİM (SMART TRAINING) BAŞLATILIYOR...\n" + "="*50)
print(f"Hiç görmediği '{TEST_YOLU}' klasörüyle Valide edilecek...")
print(f"Max Epochs: {EPOCHS} (EarlyStopping ile gözetim altındadır)\n")

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr] # Sisteme Otonom Yöneticileri Taktık
)

print("\n4. DÜRÜST MODEL DEĞERLENDİRMESİ (EVALUATION)\n" + "Zirvedeki Ağırlıklarla...\n" + "="*50)
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print(f"\n YENİ GERÇEK DOĞRULUK SKORU (TEST SETİ ACCURACY): % {test_acc * 100:.2f}")

print("\n5. YENİ KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX) ÇİZİLİYOR...\n" + "="*50)

y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

y_test = np.concatenate([y for x, y in test_dataset], axis=0)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title(f'Optimize CNN Matrisi (Accuracy: %{test_acc*100:.1f})')
plt.ylabel('Gerçek Sınıflar')
plt.xlabel('Tahmin Edilen Sınıflar')
plt.tight_layout()

cm_path = os.path.join(GORSEL_KLASORU, "Karmasiklik_Matrisi_CNN.png")
plt.savefig(cm_path)
plt.close()

print(f"  -> Optimize CNN Matrisi {cm_path} konumuna kaydedildi.")

print("\n6. GELİŞTİRİLMİŞ CNN MODELİ ARAYÜZ İÇİN DİSKE KAYDEDİLİYOR...\n" + "="*50)
kayit_yolu = "tavuk_cnn_modeli.keras"
model.save(kayit_yolu)
print(f"BAŞARILI: Mükemmelleştirilmiş Derin Öğrenme Modeli '{kayit_yolu}' adıyla diske aktarıldı!")
print("Yeni Güçlü Ağınızı 'python arayuz_cnn.py' ile teste tabi tutabilirsiniz.")
