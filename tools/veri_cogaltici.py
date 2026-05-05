import os
import sys

# tools/ klasöründen çalıştırıldığında proje kökünü ayarla
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import shutil
import librosa
import numpy as np
import soundfile as sf
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- KLASÖR İSİMLENDİRME VE HARİTALAMA ---
kategoriler = {
    "Healthy": {"orjinal": "Kayıtlar/Healthy", "egitim": "Veri_Egitim/Healthy", "test": "Veri_Test/Healthy"},
    "Unhealthy": {"orjinal": "Kayıtlar/Unhealthy", "egitim": "Veri_Egitim/Unhealthy", "test": "Veri_Test/Unhealthy"},
    "Noise": {"orjinal": "Kayıtlar/Noise", "egitim": "Veri_Egitim/Noise", "test": "Veri_Test/Noise"}
}

HEDEF_SAYI = 1000

# --- SES ÇOĞALTMA (AUGMENTATION) FONKSİYONLARI (V2.0 AKADEMİK GÜNCELLEME) ---

def gercekci_kumes_gurultusu_ekle(veri):
    """V2: Matematiksel olarak Kümes Fanı ve Hava Akımı (Pink/Brown Noise) simüle eder."""
    uzunluk = len(veri)
    
    # 1. Havalandırma / Rüzgar (White Noise)
    hava_akimi = np.random.normal(0, 1, uzunluk)
    
    # 2. Fan / Jeneratör Uğultusu (Düşük Frekanslı Brown Noise Simülasyonu)
    fan_ugultusu = np.cumsum(np.random.normal(0, 0.1, uzunluk))
    fan_ugultusu = fan_ugultusu / (np.max(np.abs(fan_ugultusu)) + 1e-9) # Normalize
    
    # Mikser: %60 Hava akımı, %40 Fan uğultusu
    kumes_gurultusu = (0.6 * hava_akimi) + (0.4 * fan_ugultusu)
    
    # Sese çok hafif oranda (orijinal sesin max genliğinin %1 - %3'ü kadar) bindir
    katsayi = random.uniform(0.01, 0.03) * np.amax(np.abs(veri))
    return veri + (katsayi * kumes_gurultusu)

def pitch_kaydir(veri, fs):
    """Sesin tonunu bozmadan hafifçe kalınlaştırır veya inceltir."""
    adim = random.choice([-2, -1, 1, 2])
    return librosa.effects.pitch_shift(y=veri, sr=fs, n_steps=adim)

def zaman_esnet(veri):
    """Sesin frekansını bozmadan hafifçe hızlandırır veya yavaşlatır."""
    hiz = random.choice([0.8, 0.9, 1.1, 1.2])
    return librosa.effects.time_stretch(y=veri, rate=hiz)

def kombo_zor_efekt(veri, fs):
    """V2: Hem zamanı esnetir hem de gerçekçi kümes gürültüsü ekler (Modeli zorlar)."""
    yeni_ses = zaman_esnet(veri)
    return gercekci_kumes_gurultusu_ekle(yeni_ses)

# --- İŞÇİ FONKSİYON ---
def uret_ve_kaydet(hedef_yol, secilen_isim, ses_verisi, fs, i):
    """Tek bir ses dosyasını çoğaltıp diske kaydeden paralel fonksiyon"""
    islem_turu = random.choice([1, 2, 3, 4]) # 4. Seçenek (Kombo) eklendi!
    
    if islem_turu == 1:
        yeni_ses = gercekci_kumes_gurultusu_ekle(ses_verisi)
        ek = "kumes_gurultulu"
    elif islem_turu == 2:
        yeni_ses = pitch_kaydir(ses_verisi, fs)
        ek = "ton_degisik"
    elif islem_turu == 3:
        yeni_ses = zaman_esnet(ses_verisi)
        ek = "esnetilmis"
    else:
        yeni_ses = kombo_zor_efekt(ses_verisi, fs)
        ek = "kombo_zorlu"
        
    yeni_isim = f"uretilen_{i}_{ek}_{secilen_isim}"
    tam_yol = os.path.join(hedef_yol, yeni_isim)
    sf.write(tam_yol, yeni_ses, fs)
    return True

# --- ANA İŞLEM ---
if __name__ == '__main__':
    print("VERİ ÇOĞALTMA FABRİKASI (V2 - AKADEMİK GÜRÜLTÜ SİMÜLASYONU) ÇALIŞIYOR\n" + "="*80)

    for ingilizce_isim, yollar in kategoriler.items():
        orj_klasor = yollar["orjinal"]
        egitim_klasor = yollar["egitim"]
        test_klasor = yollar["test"]
        
        os.makedirs(egitim_klasor, exist_ok=True)
        os.makedirs(test_klasor, exist_ok=True)
        
        if not os.path.exists(orj_klasor):
            print(f"HATA: Orijinal klasör bulunamadı -> {orj_klasor}")
            continue
            
        orijinal_dosyalar = [f for f in os.listdir(orj_klasor) if f.endswith('.wav')]
        mevcut_sayi = len(orijinal_dosyalar)
        
        if mevcut_sayi == 0:
            print(f"UYARI: {orj_klasor} içinde hiç ses yok, atlanıyor.")
            continue
            
        print(f"\nKategori: {ingilizce_isim} | Orijinal Dosya: {mevcut_sayi}")
        
        # 1. ADIM: %80 Eğitim ve %20 Test (Data Leakage Koruması)
        train_dosyalar, test_dosyalar = train_test_split(orijinal_dosyalar, test_size=0.2, random_state=42)
        
        for dosya in test_dosyalar:
            shutil.copy2(os.path.join(orj_klasor, dosya), os.path.join(test_klasor, dosya))
            
        print(f"Test Setine Ayrılan (Ayrık): {len(test_dosyalar)}")
        
        for dosya in train_dosyalar:
            shutil.copy2(os.path.join(orj_klasor, dosya), os.path.join(egitim_klasor, dosya))
            
        Uretilecek_Sayi = HEDEF_SAYI - len(train_dosyalar)
        
        if Uretilecek_Sayi <= 0:
            print(f"Zaten {HEDEF_SAYI} veya daha fazla eğitim dosyası var. Çoğaltmaya gerek yok.")
            continue
            
        # 2. ADIM: Preloading (RAM'e Alma)
        print("Eğitim sesleri belleğe (RAM) yükleniyor...")
        bellek_sesler = []
        for dosya in tqdm(train_dosyalar, desc="Belleğe Alma", leave=False):
            dosya_yolu = os.path.join(orj_klasor, dosya)
            ses_verisi, fs = librosa.load(dosya_yolu, sr=None)
            bellek_sesler.append({"isim": dosya, "veri": ses_verisi, "fs": fs})
            
        print(f"Üretilecek Yeni Eğitim Ses Sayısı: {Uretilecek_Sayi}")
        
        # 3. ADIM: Multiprocessing ile Paralel Üretim
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            gelecekler = []
            for i in range(1, Uretilecek_Sayi + 1):
                secilen = random.choice(bellek_sesler)
                gelecekler.append(
                    executor.submit(uret_ve_kaydet, egitim_klasor, secilen["isim"], secilen["veri"], secilen["fs"], i)
                )
            
            for _ in tqdm(as_completed(gelecekler), total=Uretilecek_Sayi, desc="Paralel Çoğaltım İşlemi"):
                pass
                
        print(f"BAŞARILI: {ingilizce_isim} kategorisi (Eğitim) toplam {HEDEF_SAYI} adet sese ulaştı!")

    print("\n" + "="*80)
    print("V2 İŞLEMLERİ BİTTİ! Artık CNN için spektrogram oluşturmaya hazırız.")