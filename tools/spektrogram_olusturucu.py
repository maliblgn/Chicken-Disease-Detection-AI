import os
import sys

# tools/ klasöründen çalıştırıldığında proje kökünü ayarla
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ayarlar ve Klasörler
GIRIS_KLASORLERI = ["Veri_Egitim", "Veri_Test"]
CIKIS_KLASORU = "Spektrogramlar"
SINIFLAR = ["Healthy", "Unhealthy", "Noise"]

def normalize_matrix(matrix):
    """Matrisi 0-255 arası piksel değerlerine (RGB formatına) dönüştürür."""
    mn = np.min(matrix)
    mx = np.max(matrix)
    if mx - mn == 0:
        return np.zeros_like(matrix, dtype=np.uint8)
    return ((matrix - mn) / (mx - mn) * 255).astype(np.uint8)

def spektrogram_kaydet(hedef_yol, dosya_yolu):
    """V2 (Akademik): Sesin Mel, Delta ve Delta-Delta'sını RGB Resme (Tensör) çevirir."""
    try:
        # 1. Sesi yükle
        y, sr = librosa.load(dosya_yolu, sr=None)
        
        # 2. Üç temel özelliği (Hız ve İvme dahil) hesapla
        # fmax=8000 (Tavuk sesleri için akademik eşik)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        delta = librosa.feature.delta(S_dB)
        delta2 = librosa.feature.delta(S_dB, order=2)
        
        # 3. Her bir özelliği bir Renk Kanalına (R, G, B) ata
        R = normalize_matrix(S_dB)    # Kırmızı: Saf Frekans Gücü (Mel)
        G = normalize_matrix(delta)   # Yeşil: Frekans Değişim Hızı
        B = normalize_matrix(delta2)  # Mavi: Frekans Değişim İvmesi
        
        # 4. Kanalları birleştir (Boyut: 128, Zaman, 3)
        img_array = np.stack([R, G, B], axis=-1)
        
        # Alt frekansları görselin altına almak için matrisi dikeyde çevir (Standart Spektrogram görünümü)
        img_array = np.flipud(img_array)
        
        # 5. Matplotlib YERİNE süper hızlı PIL kütüphanesi ile kaydet
        img = Image.fromarray(img_array, 'RGB')
        
        # Modeli yormamak için görseli tam 128x128 boyutuna kilitle
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        img.save(hedef_yol)
        
        return True
    
    except Exception as e:
        print(f"\nHATA ({os.path.basename(dosya_yolu)}): {e}")
        return False

if __name__ == '__main__':
    print("V2 ÜÇ KANALLI (RGB) ÖZNİTELİK HARİTASI OLUŞTURUCU BAŞLATILDI\n" + "="*65)
    
    islem_listesi = []
    
    # 1. Adım: Hiyerarşiyi oluştur ve dönüştürülecek dosyaları tespit et
    for ana_klasor in GIRIS_KLASORLERI:
        if not os.path.exists(ana_klasor):
            print(f"UYARI: {ana_klasor} klasörü bulunamadı.")
            continue
            
        for sinif in SINIFLAR:
            kaynak_dizin = os.path.join(ana_klasor, sinif)
            hedef_dizin = os.path.join(CIKIS_KLASORU, ana_klasor, sinif)
            
            if not os.path.exists(kaynak_dizin):
                continue
                
            os.makedirs(hedef_dizin, exist_ok=True)
            
            dosyalar = [f for f in os.listdir(kaynak_dizin) if f.endswith('.wav')]
            for dosya in dosyalar:
                kaynak_yol = os.path.join(kaynak_dizin, dosya)
                yeni_isim = dosya.rsplit('.', 1)[0] + ".png"
                hedef_yol = os.path.join(hedef_dizin, yeni_isim)
                
                # Sadece olmayanları ekle (Zaman tasarrufu)
                if not os.path.exists(hedef_yol):
                    islem_listesi.append({"hedef": hedef_yol, "kaynak": kaynak_yol})

    toplam_islem = len(islem_listesi)
    if toplam_islem == 0:
        print("\nOluşturulacak yeni Harita bulunamadı (Tüm veriler zaten hazır).")
        exit()
        
    print(f"Toplam {toplam_islem} adet Ses dosyası '3 Boyutlu RGB Tensöre' dönüştürülecek...")
    print("Matplotlib devreden çıkarıldı. Süper Hızlı (PIL) dönüşüm başlıyor...\n")
    
    # 2. Adım: Multiprocessing ile paralel işleme başla
    basarili = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        gelecekler = [executor.submit(spektrogram_kaydet, islem["hedef"], islem["kaynak"]) for islem in islem_listesi]
        
        for gelecek in tqdm(as_completed(gelecekler), total=toplam_islem, desc="RGB Resimler Üretiliyor"):
            if gelecek.result():
                basarili += 1
                
    print("\n" + "="*65)
    print(f"BAŞARILI: {basarili}/{toplam_islem} adet Akademik 3-Kanallı Görsel üretildi!")
    print(f"Artık 'egitim_cnn.py' dosyasını bu yeni üst düzey verilerle çalıştırabiliriz.")