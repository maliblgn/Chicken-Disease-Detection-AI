import os
import librosa
import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
from scipy.io.wavfile import write

# --- GEREKLİ FONKSİYONLAR ---
def ozellik_cikar(dosya_yolu):
    ses, fs = librosa.load(dosya_yolu, sr=None)
    mfccs = librosa.feature.mfcc(y=ses, sr=fs, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

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
        yeni_ozellik = ozellik_cikar(dosya_yolu)
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

# --- BUTON 1: DOSYA SEÇME ---
def ses_yukle_ve_test_et():
    dosya_yolu = filedialog.askopenfilename(
        title="Ses Dosyası Seçin",
        filetypes=(("WAV Dosyaları", "*.wav"), ("Tüm Dosyalar", "*.*"))
    )
    if dosya_yolu:
        lbl_durum.config(text=f"Seçilen: {os.path.basename(dosya_yolu)}")
        lbl_sonuc.config(text="Analiz Ediliyor...", fg="blue")
        pencere.update()
        analizi_ekrana_yazdir(dosya_yolu)

# --- BUTON 2: MİKROFONDAN KAYIT ---
def mikrofondan_kaydet():
    fs = 44100  # Standart ses örnekleme frekansı
    saniye = 3  # Kaç saniye kayıt alınacağı
    gecici_dosya = "anlik_kayit.wav"
    
    lbl_durum.config(text="🔴 KAYIT BAŞLADI! Lütfen 3 saniye ses dinletin...", fg="red", font=("Segoe UI", 10, "bold"))
    lbl_sonuc.config(text="Dinleniyor...", fg="gray")
    pencere.update() # Arayüzü dondurmamak için güncelliyoruz
    
    try:
        # Mikrofonu aç ve kaydet
        kayit = sd.rec(int(saniye * fs), samplerate=fs, channels=1)
        sd.wait()  # Kaydın bitmesini bekle
        
        # Geçici bir wav dosyası olarak diske kaydet
        write(gecici_dosya, fs, kayit)
        
        lbl_durum.config(text="✅ Kayıt Tamamlandı! Analiz ediliyor...", fg="green", font=("Segoe UI", 10))
        pencere.update()
        
        # Kaydedilen dosyayı modele gönder
        analizi_ekrana_yazdir(gecici_dosya)
        
    except Exception as e:
        messagebox.showerror("Mikrofon Hatası", f"Mikrofona erişilemedi.\nSistem hatası: {e}")
        lbl_durum.config(text="Kayıt Başarısız!", fg="red")

# ---------------------------------------------------------
# ARAYÜZ (GUI) TASARIMI
# ---------------------------------------------------------
pencere = tk.Tk()
pencere.title("TÜME - Tavuk Ses Analiz Sistemi")
pencere.geometry("500x450") # Biraz daha büyüttük
pencere.configure(bg="#f8f9fa")

lbl_baslik = tk.Label(pencere, text="Yapay Zeka Hastalık Tespit Merkezi", font=("Segoe UI", 16, "bold"), bg="#f8f9fa", fg="#2c3e50")
lbl_baslik.pack(pady=15)

# --- BUTONLAR ---
frame_butonlar = tk.Frame(pencere, bg="#f8f9fa")
frame_butonlar.pack(pady=10)

btn_sec = tk.Button(frame_butonlar, text="📂 Dosya Yükle", command=ses_yukle_ve_test_et, font=("Segoe UI", 11, "bold"), bg="#3498db", fg="white", width=20, pady=8, cursor="hand2", relief="flat")
btn_sec.grid(row=0, column=0, padx=10)

btn_kaydet = tk.Button(frame_butonlar, text="🎙️ Anlık Kayıt (3 Sn)", command=mikrofondan_kaydet, font=("Segoe UI", 11, "bold"), bg="#e74c3c", fg="white", width=20, pady=8, cursor="hand2", relief="flat")
btn_kaydet.grid(row=0, column=1, padx=10)

# --- BİLGİ VE SONUÇ EKRANI ---
lbl_durum = tk.Label(pencere, text="Sistem Hazır. Dosya seçin veya kayıt yapın.", font=("Segoe UI", 10), bg="#f8f9fa", fg="#7f8c8d")
lbl_durum.pack(pady=15)

lbl_sonuc = tk.Label(pencere, text="SONUÇ: Bekleniyor...", font=("Segoe UI", 18, "bold"), bg="#f8f9fa", fg="#34495e")
lbl_sonuc.pack(pady=20)

pencere.mainloop()