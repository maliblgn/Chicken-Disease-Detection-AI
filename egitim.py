import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

GORSEL_KLASORU = "Gorseller"
if not os.path.exists(GORSEL_KLASORU):
    os.makedirs(GORSEL_KLASORU)

# Özellik Çıkarma
def ozellik_cikar(dosya_yolu):
    ses, fs = librosa.load(dosya_yolu, sr=None)
    mfccs = librosa.feature.mfcc(y=ses, sr=fs, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# YENİ HEDEF KLASÖRLER (Eğitim ve Test Fiziksel Olarak Ayrı)
# Veri_Egitim içindeki klasör isimleri Healthy, Unhealthy, Noise
klasorler = {
    "Healthy": 0, 
    "Unhealthy": 1, 
    "Noise": 2
}

def veri_yukle(ana_klasor):
    X = []
    y = []
    print(f"\n{ana_klasor} KLASÖRÜNDEN VERİLER OKUNUYOR...")
    
    if not os.path.exists(ana_klasor):
        print(f"UYARI: '{ana_klasor}' klasörü bulunamadı. Lütfen önce veri_cogaltici.py çalıştırın.")
        return np.array(X), np.array(y)
        
    for klasor, etiket in klasorler.items():
        tam_klasor = os.path.join(ana_klasor, klasor)
        if os.path.exists(tam_klasor):
            print(f"[{tam_klasor}] klasörü işleniyor...")
            for dosya in os.listdir(tam_klasor):
                if dosya.endswith(".wav"):
                    tam_yol = os.path.join(tam_klasor, dosya)
                    ozellik = ozellik_cikar(tam_yol)
                    X.append(ozellik)
                    y.append(etiket)
        else:
            print(f"UYARI: '{tam_klasor}' bulunamadı!")
            
    return np.array(X), np.array(y)

# 1. Eğitim verilerini yükle
X_train, y_train = veri_yukle("Veri_Egitim")
print(f"TOPLAM {len(X_train)} ADET EĞİTİM SESİ YÜKLENDİ!")

# 2. Test verilerini yükle
X_test, y_test = veri_yukle("Veri_Test")
print(f"TOPLAM {len(X_test)} ADET TEST SESİ YÜKLENDİ!")

if len(X_train) == 0 or len(X_test) == 0:
    print("\nEKSİK VERİ! Lütfen önce veri_cogaltici.py dosyasını çalıştırarak verileri üretin.")
    exit()

model = RandomForestClassifier(n_estimators=100, random_state=42)
print("\nYapay zeka modeli, hiç görmediği verilerle (Data Leakage olmadan) eğitime başladı...")
model.fit(X_train, y_train)

# MATRİS VE SKOR İŞLEMLERİ
y_pred = model.predict(X_test)
skor = accuracy_score(y_test, y_pred)
print(f"\nEĞİTİM TAMAMLANDI! Modelin Gerçekçi Doğruluk Skoru: % {skor * 100:.2f}")

cm = confusion_matrix(y_test, y_pred)
sinif_isimleri = ["Sağlıklı", "Hasta", "Gürültü"]
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sinif_isimleri, yticklabels=sinif_isimleri)
plt.title('Yapay Zeka Karmaşıklık Matrisi (Data Leakage Giderildi)')
plt.ylabel('Gerçek Sınıflar')
plt.xlabel('Tahmin Edilen Sınıflar')
plt.tight_layout()
plt.savefig(os.path.join(GORSEL_KLASORU, "Karmasiklik_Matrisi_Yeni.png"))
plt.close()

# MODELİ DİSKE KAYDETME
kayit_yolu = 'tavuk_modeli.pkl'
joblib.dump(model, kayit_yolu)
print(f"\nBAŞARILI: Veri sızıntısından arındırılmış GÜVENİLİR model '{kayit_yolu}' adıyla diske kaydedildi!")
print("Artık 'arayuz.py' dosyasını güvenle kullanabilirsiniz.")