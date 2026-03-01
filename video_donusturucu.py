import os
from moviepy import VideoFileClip

# --- KLASÖR AYARLARI ---
GIRIS_KLASORU = "Videolar"
CIKIS_KLASORU = "Donusturulen_Sesler"

# Klasörler yoksa otomatik oluştur
os.makedirs(GIRIS_KLASORU, exist_ok=True)
os.makedirs(CIKIS_KLASORU, exist_ok=True)

def mp4_ten_wav_yap():
    print("VİDEO -> SES DÖNÜŞTÜRÜCÜ BAŞLATILDI...\n" + "="*40)
    
    # Giriş klasöründeki mp4 dosyalarını bul
    videolar = [f for f in os.listdir(GIRIS_KLASORU) if f.lower().endswith('.mp4')]
    
    if len(videolar) == 0:
        print(f"UYARI: '{GIRIS_KLASORU}' klasöründe dönüştürülecek MP4 dosyası bulunamadı.")
        print("Lütfen videolarınızı bu klasöre atıp tekrar çalıştırın.")
        return
        
    print(f"Toplam {len(videolar)} adet video bulundu. Dönüştürme başlıyor...\n")
    
    basarili_sayisi = 0
    
    for dosya_adi in videolar:
        video_yolu = os.path.join(GIRIS_KLASORU, dosya_adi)
        # Yeni dosyanın adını .mp4 yerine .wav yap
        yeni_dosya_adi = dosya_adi.rsplit('.', 1)[0] + ".wav"
        wav_yolu = os.path.join(CIKIS_KLASORU, yeni_dosya_adi)
        
        try:
            print(f"İşleniyor: {dosya_adi} -> {yeni_dosya_adi}")
            
            # Videoyu yükle ve sesi çıkar
            video = VideoFileClip(video_yolu)
            ses = video.audio
            
            # Sesi 44100 Hz (standart kalite) ile kaydet
            # logger=None parametresi terminali gereksiz yazılarla doldurmayı engeller
            ses.write_audiofile(wav_yolu, fps=44100, logger=None)
            
            # Belleği temizle
            video.close()
            ses.close()
            
            basarili_sayisi += 1
            
        except Exception as e:
            print(f"HATA ({dosya_adi}): İşlem başarısız oldu. Detay: {e}")
            
    print("\n" + "="*40)
    print(f"İŞLEM TAMAMLANDI! {basarili_sayisi}/{len(videolar)} video başarıyla sese dönüştürüldü.")
    print(f"Ses dosyalarını '{CIKIS_KLASORU}' klasöründe bulabilirsiniz.")

# Fonksiyonu çalıştır
if __name__ == "__main__":
    mp4_ten_wav_yap()