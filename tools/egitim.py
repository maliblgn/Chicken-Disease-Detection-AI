import os
import sys

# tools/ klasöründen çalıştırıldığında proje kökünü ayarla
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

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

# --- V2: AKADEMİK ÖZNİTELİK ÇIKARMA (DELTA & DELTA-DELTA) ---
def ozellik_cikar_v2(dosya_yolu):
    """
    Sadece temel MFCC değil, sesin hızını (Delta) ve ivmesini (Delta-Delta) 
    de hesaplayarak 120 boyutlu devasa bir matematiksel parmak izi çıkarır.
    """
    # 1. Sesi yükle
    ses, fs = librosa.load(dosya_yolu, sr=None)
    
    # 2. Temel MFCC (Katsayı 13'ten 20'ye çıkarıldı - Daha fazla detay)
    mfccs = librosa.feature.mfcc(y=ses, sr=fs, n_mfcc=20)
    
    # 3. Delta (Hız) ve Delta-Delta (İvme)
    delta_mfcc = librosa.feature.delta(mfccs)
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    
    # 4. İstatistiksel Haritalama (Ortalama ve Standart Sapma birleşimi)
    # Numpy hstack ile tüm değerleri tek bir uzun vektörde (Excel satırı gibi) birleştiriyoruz
    ozellik_vektoru = np.hstack([
        np.mean(mfccs, axis=1),       # 20 değer
        np.std(mfccs, axis=1),        # 20 değer
        np.mean(delta_mfcc, axis=1),  # 20 değer
        np.std(delta_mfcc, axis=1),   # 20 değer
        np.mean(delta2_mfcc, axis=1), # 20 değer
        np.std(delta2_mfcc, axis=1)   # 20 değer (Toplam: 120 boyutlu matematiksel özellik!)
    ])
    
    return ozellik_vektoru

# YENİ HEDEF KLASÖRLER (Eğitim ve Test Fiziksel Olarak Ayrı)
klasorler = {
    "Healthy": 0, 
    "Unhealthy": 1, 
    "Noise": 2
}

def veri_yukle(ana_klasor):
    X = []
    y = []
    print(f"\n{ana_klasor} KLASÖRÜNDEN V2 ÖZNİTELİKLER ÇIKARTILIYOR...")
    
    if not os.path.exists(ana_klasor):
        print(f"UYARI: '{ana_klasor}' klasörü bulunamadı.")
        return np.array(X), np.array(y)
        
    for klasor, etiket in klasorler.items():
        tam_klasor = os.path.join(ana_klasor, klasor)
        if os.path.exists(tam_klasor):
            print(f"[{tam_klasor}] analiz ediliyor (Delta/Delta-Delta hesaplanıyor)...")
            dosyalar = [d for d in os.listdir(tam_klasor) if d.endswith(".wav")]
            for dosya in dosyalar:
                tam_yol = os.path.join(tam_klasor, dosya)
                # Yeni V2 fonksiyonumuzu kullanıyoruz
                ozellik = ozellik_cikar_v2(tam_yol)
                X.append(ozellik)
                y.append(etiket)
        else:
            print(f"UYARI: '{tam_klasor}' bulunamadı!")
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("KLASİK MODEL V2 (AKADEMİK DELTA-DELTA MİMARİSİ) EĞİTİMİ BAŞLIYOR\n" + "="*60)
    
    # 1. Eğitim ve Test verilerini yükle
    X_train, y_train = veri_yukle("Veri_Egitim")
    print(f"TOPLAM {len(X_train)} ADET EĞİTİM SESİNİN MATEMATİKSEL İZİ ÇIKARTILDI!")

    X_test, y_test = veri_yukle("Veri_Test")
    print(f"TOPLAM {len(X_test)} ADET TEST SESİNİN MATEMATİKSEL İZİ ÇIKARTILDI!")

    if len(X_train) == 0 or len(X_test) == 0:
        print("\nEKSİK VERİ! Lütfen önce verileri kontrol edin.")
        exit()

    # V2: Model parametrelerini biraz daha güçlendirdik (100 ağaçtan 200 ağaca çıkardık)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    print("\nYapay zeka (V2), 120 boyutlu Delta özellikleri ile eğitime başladı...")
    model.fit(X_train, y_train)

    # MATRİS VE SKOR İŞLEMLERİ
    y_pred = model.predict(X_test)
    skor = accuracy_score(y_test, y_pred)
    print(f"\nEĞİTİM TAMAMLANDI! Modelin V2 Doğruluk Skoru (Accuracy): % {skor * 100:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    sinif_isimleri = ["Sağlıklı", "Hasta", "Gürültü"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sinif_isimleri, yticklabels=sinif_isimleri)
    plt.title(f'V2 Delta-Delta Random Forest Matrisi (Accuracy: %{skor*100:.1f})')
    plt.ylabel('Gerçek Sınıflar')
    plt.xlabel('Tahmin Edilen Sınıflar')
    plt.tight_layout()
    plt.savefig(os.path.join(GORSEL_KLASORU, "Karmasiklik_Matrisi_Yeni_V2.png"))
    plt.close()

    # MODELİ DİSKE KAYDETME
    kayit_yolu = os.path.join('models', 'tavuk_modeli.pkl')
    joblib.dump(model, kayit_yolu)
    print("\n" + "="*60)
    print(f"BAŞARILI: Yüksek Kapasiteli V2 Modeli '{kayit_yolu}' adıyla diske kaydedildi!")
    print("Artık eski modelin çok daha ötesinde (İvmeleri dahi okuyan) bir aklımız var.")