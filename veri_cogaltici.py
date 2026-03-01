import os
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

# --- SES ÇOĞALTMA (AUGMENTATION) FONKSİYONLARI ---
def gurultu_ekle(veri):
    """Sese çok hafif arka plan cızırtısı/rüzgar sesi ekler."""
    gurultu_katsayisi = 0.005 * np.random.uniform() * np.amax(veri)
    return veri + gurultu_katsayisi * np.random.normal(size=veri.shape[0])

def pitch_kaydir(veri, fs):
    """Sesin tonunu bozmadan hafifçe kalınlaştırır veya inceltir."""
    adim = random.choice([-2, -1, 1, 2]) # Yarım nota kaydırmaları
    return librosa.effects.pitch_shift(y=veri, sr=fs, n_steps=adim)

def zaman_esnet(veri):
    """Sesin frekansını bozmadan hafifçe hızlandırır veya yavaşlatır."""
    hiz = random.choice([0.8, 0.9, 1.1, 1.2])
    return librosa.effects.time_stretch(y=veri, rate=hiz)

# --- İŞÇİ FONKSİYON ---
def uret_ve_kaydet(hedef_yol, secilen_isim, ses_verisi, fs, i):
    """Tek bir ses dosyasını çoğaltıp diske kaydeden paralel fonksiyon"""
    islem_turu = random.choice([1, 2, 3])
    if islem_turu == 1:
        yeni_ses = gurultu_ekle(ses_verisi)
        ek = "gurultulu"
    elif islem_turu == 2:
        yeni_ses = pitch_kaydir(ses_verisi, fs)
        ek = "ton_degisik"
    else:
        yeni_ses = zaman_esnet(ses_verisi)
        ek = "esnetilmis"
        
    yeni_isim = f"uretilen_{i}_{ek}_{secilen_isim}"
    tam_yol = os.path.join(hedef_yol, yeni_isim)
    sf.write(tam_yol, yeni_ses, fs)
    return True

# --- ANA İŞLEM ---
if __name__ == '__main__':
    print("VERİ ÇOĞALTMA FABRİKASI ÇALIŞIYOR (Data Leakage Kapatıldı) - HIZLANDIRILMIŞ SÜRÜM\n" + "="*80)

    for ingilizce_isim, yollar in kategoriler.items():
        orj_klasor = yollar["orjinal"]
        egitim_klasor = yollar["egitim"]
        test_klasor = yollar["test"]
        
        # Klasörleri oluştur
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
            
        print(f"\nKategori: {ingilizce_isim}")
        print(f"Orijinal Dosya Sayısı: {mevcut_sayi}")
        
        # 1. ADIM: %80 Eğitim ve %20 Test olarak böl
        train_dosyalar, test_dosyalar = train_test_split(orijinal_dosyalar, test_size=0.2, random_state=42)
        
        # Test dosyalarını sadece Veri_Test klasörüne kopyala ve BIRAK (Asla dokunma)
        for dosya in test_dosyalar:
            shutil.copy2(os.path.join(orj_klasor, dosya), os.path.join(test_klasor, dosya))
            
        print(f"Test Setine Ayrılan (Ayrık): {len(test_dosyalar)}")
        print(f"Eğitim Setine Ayrılan (Çoğaltılacak): {len(train_dosyalar)}")
        
        # Eğitim dosyalarını Veri_Egitim klasörüne kopyala
        for dosya in train_dosyalar:
            shutil.copy2(os.path.join(orj_klasor, dosya), os.path.join(egitim_klasor, dosya))
            
        Uretilecek_Sayi = HEDEF_SAYI - len(train_dosyalar)
        
        if Uretilecek_Sayi <= 0:
            print(f"Zaten {HEDEF_SAYI} veya daha fazla eğitim dosyası var. Çoğaltmaya gerek yok.")
            continue
            
        # 2. ADIM: Preloading (Sadece Eğitim dosyalarını RAM'e al)
        print("Eğitim sesleri belleğe (RAM) yükleniyor... Bu işlem disk okumalarını yok ederek muazzam hız kazandıracak.")
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
            
            # Üretim İlerleme Çubuğu (Progress Bar)
            for _ in tqdm(as_completed(gelecekler), total=Uretilecek_Sayi, desc="Paralel Çoğaltım İşlemi"):
                pass
                
        print(f"BAŞARILI: {ingilizce_isim} kategorisi (Eğitim) toplam {HEDEF_SAYI} adet sese ulaştı!")

    print("\n" + "="*80 + "\nTÜM İŞLEMLER BİTTİ! Artık egitim.py dosyasını çalıştırabilirsiniz.")