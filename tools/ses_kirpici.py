import os
import sys

# tools/ klasöründen çalıştırıldığında proje kökünü ayarla
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import librosa
import soundfile as sf
import math
import numpy as np

# --- AYARLAR ---
GIRDI_KLASORU = "Kesilecek_Sesler"
CIKTI_KLASORU = "Kirpilmis_Sesler"
PARCA_SURESI_SANIYE = 5
GURULTU_ESIGI = 0.01  # YENİ (V2): Bu değerin altındaki sesler (sessizlik/sadece fan) silinecek

def klasorleri_hazirla():
    """Gerekli klasörleri oluşturur."""
    if not os.path.exists(GIRDI_KLASORU):
        os.makedirs(GIRDI_KLASORU)
        print(f"[*] '{GIRDI_KLASORU}' klasörü oluşturuldu. Lütfen kesilecek ses dosyalarını buraya atın.")
    else:
        print(f"[*] '{GIRDI_KLASORU}' klasörü hazır.")

    if not os.path.exists(CIKTI_KLASORU):
        os.makedirs(CIKTI_KLASORU)
        print(f"[*] '{CIKTI_KLASORU}' klasörü oluşturuldu. Kesilmiş dosyalar buraya kaydedilecek.")
    else:
        print(f"[*] '{CIKTI_KLASORU}' klasörü hazır.")

def sesleri_kes():
    """Girdi klasöründeki sesleri okur, gürültü filtresinden geçirir ve kaydeder."""
    gecerli_uzantilar = {".wav", ".mp3", ".ogg", ".flac"}
    
    dosyalar = [dosya for dosya in os.listdir(GIRDI_KLASORU) 
                if os.path.splitext(dosya.lower())[1] in gecerli_uzantilar]

    if not dosyalar:
        print(f"[!] '{GIRDI_KLASORU}' klasöründe ses dosyası bulunamadı.")
        return

    print(f"[*] Toplam {len(dosyalar)} ses dosyası işlenecek...\n")

    atlanan_parca_sayisi = 0
    kaydedilen_parca_sayisi = 0

    for dosya_adi in dosyalar:
        dosya_yolu = os.path.join(GIRDI_KLASORU, dosya_adi)
        isim, uzanti = os.path.splitext(dosya_adi)
        
        try:
            print(f"> {dosya_adi} yükleniyor...")
            y, sr = librosa.load(dosya_yolu, sr=None)
            toplam_sure = len(y) / sr
            
            # Eğer dosya zaten 5 saniyeden kısaysa
            if toplam_sure <= PARCA_SURESI_SANIYE:
                # V2: Gürültü kontrolü (Sessizlik filtresi)
                rms_degeri = np.mean(librosa.feature.rms(y=y))
                if rms_degeri < GURULTU_ESIGI:
                    print(f"  - ATLANDI: {dosya_adi} (Sessiz veya anlamsız gürültü)")
                    atlanan_parca_sayisi += 1
                    continue
                    
                cikis_yolu = os.path.join(CIKTI_KLASORU, f"{isim}_part1.wav")
                sf.write(cikis_yolu, y, sr)
                kaydedilen_parca_sayisi += 1
                continue
            
            # Uzun dosyaları parçalama
            parca_sayisi = math.ceil(toplam_sure / PARCA_SURESI_SANIYE)
            ornek_sayisi_basina = PARCA_SURESI_SANIYE * sr
            
            for i in range(parca_sayisi):
                baslangic_index = int(i * ornek_sayisi_basina)
                bitis_index = int(min((i + 1) * ornek_sayisi_basina, len(y)))
                
                ses_parcasi = y[baslangic_index:bitis_index]
                
                # V2: AKILLI GÜRÜLTÜ KAPISI (NOISE GATE)
                rms_degeri = np.mean(librosa.feature.rms(y=ses_parcasi))
                
                # Eğer parça sessizse veya sadece fan uğultusuysa kaydetme!
                if rms_degeri < GURULTU_ESIGI:
                    atlanan_parca_sayisi += 1
                    continue 
                    
                parca_adi = f"{isim}_part{i+1}.wav"
                cikis_yolu = os.path.join(CIKTI_KLASORU, parca_adi)
                
                sf.write(cikis_yolu, ses_parcasi, sr)
                kaydedilen_parca_sayisi += 1
                print(f"  - Kaydedildi: {parca_adi} (Enerji: {rms_degeri:.3f})")
                
        except Exception as e:
            print(f"[HATA] {dosya_adi} işlenirken bir hata oluştu: {e}")

    # İşlem Özeti
    print("\n" + "="*50)
    print("V2 KESİM VE FİLTRELEME ÖZETİ:")
    print(f"✅ Başarıyla Kaydedilen Anlamlı Parça: {kaydedilen_parca_sayisi}")
    print(f"🗑️ Çöpe Atılan Sessiz/Gürültülü Parça: {atlanan_parca_sayisi}")
    print("="*50)

if __name__ == "__main__":
    print("--- Akıllı Ses Kesme ve Gürültü Filtreleme Sistemi (V2) ---")
    klasorleri_hazirla()
    sesleri_kes()