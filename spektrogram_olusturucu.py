import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ayarlar ve Klasörler
GIRIS_KLASORLERI = ["Veri_Egitim", "Veri_Test"]
CIKIS_KLASORU = "Spektrogramlar"
SINIFLAR = ["Healthy", "Unhealthy", "Noise"]

def spektrogram_kaydet(hedef_yol, dosya_yolu):
    """Tek bir ses dosyasının Mel-Spectrogram'ını saf görüntü (heatmap) olarak kaydeder."""
    try:
        # 1. Sesi yükle
        y, sr = librosa.load(dosya_yolu, sr=None)
        
        # 2. Mel-spectrogram oluştur
        # n_mels=128 standart yüksek çözünürlük
        # fmax=8000 (tavuk sesleri 8kHz üzerine çok çıkmaz, detayı artırır)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        
        # 3. Yüksek kontrast için Log (Decibel) dönüşümü
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 4. Çizim İşlemleri (Tüm boşluklar kapatılıyor!)
        fig, ax = plt.subplots(figsize=(4, 4)) # 4x4 inç boyut (Kare görüntü makine için idealdir)
        
        # Sadece resmi çizdir
        librosa.display.specshow(S_dB, sr=sr, fmax=8000, ax=ax, cmap='magma') # magma renk paleti CNN'ler için iyidir
        
        # Eksenleri, yazıları ve boşlukları tamamen sil
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        
        # Resmi PNG olarak kaydet (pad_inches=0 en önemli kısımdır, beyaz kanatları keser)
        plt.savefig(hedef_yol, bbox_inches='tight', pad_inches=0)
        
        # Bellek sızıntısını önlemek için figürü kapat
        plt.close(fig)
        
        return True
    
    except Exception as e:
        print(f"\nHATA ({os.path.basename(dosya_yolu)}): {e}")
        return False

if __name__ == '__main__':
    print("MEL-SPECTROGRAM GÖRÜNTÜ OLUŞTURUCU BAŞLATILDI\n" + "="*60)
    
    islem_listesi = []
    
    # 1. Adım: Hiyerarşiyi oluştur ve dönüştürülecek dosyaları tespit et
    for ana_klasor in GIRIS_KLASORLERI:
        if not os.path.exists(ana_klasor):
            print(f"UYARI: {ana_klasor} klasörü bulunamadı. Lütfen önce Aşama 1'i tamamlayın.")
            continue
            
        for sinif in SINIFLAR:
            kaynak_dizin = os.path.join(ana_klasor, sinif)
            hedef_dizin = os.path.join(CIKIS_KLASORU, ana_klasor, sinif)
            
            # Eğer kaynak dizini boşsa (veri yoksa) atla
            if not os.path.exists(kaynak_dizin):
                continue
                
            # Spektrogramların kaydedileceği test/train dizinlerini otomatik aç
            os.makedirs(hedef_dizin, exist_ok=True)
            
            dosyalar = [f for f in os.listdir(kaynak_dizin) if f.endswith('.wav')]
            for dosya in dosyalar:
                kaynak_yol = os.path.join(kaynak_dizin, dosya)
                # Uzantıyı .wav yerine .png yapacağız
                yeni_isim = dosya.rsplit('.', 1)[0] + ".png"
                hedef_yol = os.path.join(hedef_dizin, yeni_isim)
                
                # Eğer daha önce oluşturulmuşsa atla, vakit kazandırır
                if not os.path.exists(hedef_yol):
                    islem_listesi.append({"hedef": hedef_yol, "kaynak": kaynak_yol})

    toplam_islem = len(islem_listesi)
    if toplam_islem == 0:
        print("\nOluşturulacak yeni Spectrogram bulunamadı (Tüm veriler zaten dönüştürülmüş).")
        exit()
        
    print(f"Toplam {toplam_islem} adet Ses dosyası 'Isı Haritası' görseline (PNG) dönüştürülecek...")
    print("Derin Öğrenme (CNN) altyapısı hazırlanıyor, lütfen bekleyin...\n")
    
    # 2. Adım: Multiprocessing ile paralel işleme başla
    basarili = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        gelecekler = [executor.submit(spektrogram_kaydet, islem["hedef"], islem["kaynak"]) for islem in islem_listesi]
        
        for gelecek in tqdm(as_completed(gelecekler), total=toplam_islem, desc="Png Dönüştürülüyor"):
            if gelecek.result():
                basarili += 1
                
    print("\n" + "="*60)
    print(f"BAŞARILI: {basarili}/{toplam_islem} adet görüntü üretildi!")
    print(f"Görselleri {CIKIS_KLASORU}/ klasöründen inceleyebilirsiniz.")
