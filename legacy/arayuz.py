import os
import librosa
import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
from scipy.io.wavfile import write

# --- V2: AKADEMİK ÖZNİTELİK ÇIKARMA (120 Boyutlu Vektör) ---
def ozellik_cikar_v2(dosya_yolu):
    ses, fs = librosa.load(dosya_yolu, sr=None)
    
    mfccs = librosa.feature.mfcc(y=ses, sr=fs, n_mfcc=20)
    delta_mfcc = librosa.feature.delta(mfccs)
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    
    return np.hstack([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(delta_mfcc, axis=1), np.std(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1), np.std(delta2_mfcc, axis=1)
    ])

# EĞİTİLMİŞ MODELİ YÜKLE
MODEL_YOLU = 'tavuk_modeli.pkl'
if not os.path.exists(MODEL_YOLU):
    tk.Tk().withdraw()
    messagebox.showerror("Hata", f"Model dosyası ({MODEL_YOLU}) bulunamadı!\nLütfen önce egitim.py dosyasını çalıştırın.")
    exit()

model = joblib.load(MODEL_YOLU)

# --- ANALİZ VE ARAYÜZ GÜNCELLEME FONKSİYONU ---
def analizi_ekrana_yazdir(dosya_yolu):
    try:
        # V2 Fonksiyonunu çağırıyoruz
        yeni_ozellik = ozellik_cikar_v2(dosya_yolu)
        tahmin = model.predict([yeni_ozellik])[0]
        
        if tahmin == 0:
            sonuc_metni = "SAĞLIKLI (Healthy)"
            renk = "#2ecc71"
        elif tahmin == 1:
            sonuc_metni = "HASTA (Unhealthy)"
            renk = "#e74c3c"
        else:
            sonuc_metni = "GÜRÜLTÜ (Noise)"
            renk = "#f39c12"
            
        lbl_sonuc.config(text=f"SONUÇ: {sonuc_metni}", fg=renk)
    except Exception as e:
        messagebox.showerror("Analiz Hatası", f"Hata oluştu:\n{e}")
        lbl_sonuc.config(text="Analiz Başarısız", fg="black")

# --- BUTONLARIN İŞLEVLERİ ---
def ses_yukle_ve_test_et():
    dosya_yolu = filedialog.askopenfilename(
        title="Ses Dosyası Seçin",
        filetypes=(("WAV Dosyaları", "*.wav"), ("Tüm Dosyalar", "*.*"))
    )
    if dosya_yolu:
        lbl_durum.config(text=f"Seçilen: {os.path.basename(dosya_yolu)}")
        lbl_sonuc.config(text="V2 RF Analiz Ediliyor...", fg="blue")
        pencere.update()
        analizi_ekrana_yazdir(dosya_yolu)

def mikrofondan_kaydet():
    fs = 44100
    saniye = 3
    gecici_dosya = "anlik_kayit.wav"
    
    lbl_durum.config(text="🔴 KAYIT BAŞLADI! Lütfen 3 saniye ses dinletin...", fg="red", font=("Segoe UI", 10, "bold"))
    lbl_sonuc.config(text="Dinleniyor...", fg="gray")
    pencere.update()
    
    try:
        kayit = sd.rec(int(saniye * fs), samplerate=fs, channels=1)
        sd.wait()
        write(gecici_dosya, fs, kayit)
        
        lbl_durum.config(text="✅ Kayıt Tamamlandı! V2 analiz ediliyor...", fg="green", font=("Segoe UI", 10))
        pencere.update()
        analizi_ekrana_yazdir(gecici_dosya)
        
    except Exception as e:
        messagebox.showerror("Mikrofon Hatası", f"Mikrofona erişilemedi.\nSistem hatası: {e}")
        lbl_durum.config(text="Kayıt Başarısız!", fg="red")

# ---------------------------------------------------------
# ARAYÜZ (GUI) TASARIMI
# ---------------------------------------------------------
pencere = tk.Tk()
pencere.title("V2 TÜME - Klasik RF Ses Analiz Sistemi")
pencere.geometry("500x450")
pencere.configure(bg="#f8f9fa")

lbl_baslik = tk.Label(pencere, text="V2 Yapay Zeka Teşhis Merkezi (RF)", font=("Segoe UI", 16, "bold"), bg="#f8f9fa", fg="#2c3e50")
lbl_baslik.pack(pady=15)

lbl_altyazi = tk.Label(pencere, text="120 Boyutlu Delta-Delta Özellikleri ile Çalışır", font=("Segoe UI", 9), bg="#f8f9fa", fg="#95a5a6")
lbl_altyazi.pack()

frame_butonlar = tk.Frame(pencere, bg="#f8f9fa")
frame_butonlar.pack(pady=10)

btn_sec = tk.Button(frame_butonlar, text="📂 Dosya Yükle", command=ses_yukle_ve_test_et, font=("Segoe UI", 11, "bold"), bg="#3498db", fg="white", width=20, pady=8, cursor="hand2", relief="flat")
btn_sec.grid(row=0, column=0, padx=10)

btn_kaydet = tk.Button(frame_butonlar, text="🎙️ Anlık Kayıt (3 Sn)", command=mikrofondan_kaydet, font=("Segoe UI", 11, "bold"), bg="#e74c3c", fg="white", width=20, pady=8, cursor="hand2", relief="flat")
btn_kaydet.grid(row=0, column=1, padx=10)

lbl_durum = tk.Label(pencere, text="Sistem Hazır. Dosya seçin veya kayıt yapın.", font=("Segoe UI", 10), bg="#f8f9fa", fg="#7f8c8d")
lbl_durum.pack(pady=15)

lbl_sonuc = tk.Label(pencere, text="SONUÇ: Bekleniyor...", font=("Segoe UI", 18, "bold"), bg="#f8f9fa", fg="#34495e")
lbl_sonuc.pack(pady=20)

pencere.mainloop()