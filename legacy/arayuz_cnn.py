import os
import librosa
import numpy as np
import tensorflow as tf
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
from scipy.io.wavfile import write

# Model Yolu
MODEL_YOLU = 'tavuk_cnn_modeli.keras'

if not os.path.exists(MODEL_YOLU):
    tk.Tk().withdraw()
    messagebox.showerror("Hata", f"Model dosyası ({MODEL_YOLU}) bulunamadı!\nLütfen önce egitim_cnn.py dosyasını çalıştırın.")
    exit()

print("V2 Yapay Zeka (CSCNN) Modeli Belleğe Alınıyor...")
model = tf.keras.models.load_model(MODEL_YOLU)

# Tensorflow varsayılan sınıf sırası
SINIFLAR = ['Healthy', 'Noise', 'Unhealthy']
IMG_SIZE = (128, 128)

def normalize_matrix(matrix):
    """Matrisi 0-255 piksel aralığına sıkıştırır."""
    mn = np.min(matrix)
    mx = np.max(matrix)
    if mx - mn == 0:
        return np.zeros_like(matrix, dtype=np.uint8)
    return ((matrix - mn) / (mx - mn) * 255).astype(np.uint8)

def sesi_spectrograma_cevir_ve_tahmin_et(dosya_yolu):
    gecici_png = "temp_v2_spectrogram.png"
    try:
        # 1. Sesi yükle ve 3-Kanallı (RGB) Tensör oluştur (V2 Akademik Metot)
        y, sr = librosa.load(dosya_yolu, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        delta = librosa.feature.delta(S_dB)
        delta2 = librosa.feature.delta(S_dB, order=2)
        
        R = normalize_matrix(S_dB)
        G = normalize_matrix(delta)
        B = normalize_matrix(delta2)
        
        img_array = np.stack([R, G, B], axis=-1)
        img_array = np.flipud(img_array)
        
        img = Image.fromarray(img_array, 'RGB')
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        img.save(gecici_png)
        
        # 2. Keras için yükle ve boyutlandır
        img_keras = tf.keras.utils.load_img(gecici_png, target_size=IMG_SIZE)
        img_keras_array = tf.keras.utils.img_to_array(img_keras)
        img_keras_array = tf.expand_dims(img_keras_array, 0) 
        
        # 3. Modele tahmin ettir!
        tahmin_olasiliklari = model.predict(img_keras_array)
        tahmin_edilen_sinif_index = np.argmax(tahmin_olasiliklari[0])
        
        tahmin_edilen_sinif_ismi = SINIFLAR[tahmin_edilen_sinif_index]
        guven_orani = np.max(tahmin_olasiliklari[0]) * 100
        
        # 4. Temizlik
        if os.path.exists(gecici_png):
            os.remove(gecici_png)
            
        return tahmin_edilen_sinif_ismi, guven_orani
        
    except Exception as e:
        print(f"Hata detayı: {e}")
        if os.path.exists(gecici_png):
            os.remove(gecici_png)
        return None, 0

def analizi_ekrana_yazdir(dosya_yolu):
    sonuc_metni, guven = sesi_spectrograma_cevir_ve_tahmin_et(dosya_yolu)
    
    if sonuc_metni:
        if sonuc_metni == 'Healthy':
            gosterim = "SAĞLIKLI (Healthy)"
            renk = "#2ecc71"
        elif sonuc_metni == 'Unhealthy':
            gosterim = "HASTA (Unhealthy)"
            renk = "#e74c3c"
        else:
            gosterim = "GÜRÜLTÜ (Noise)"
            renk = "#f39c12"
            
        # Güven skoruna göre metin ekle
        guven_durumu = "Yüksek Güven" if guven > 85 else "Orta Güven (Dinletin)"
        lbl_sonuc.config(text=f"SONUÇ: {gosterim}\n(%{guven:.1f} - {guven_durumu})", fg=renk)
    else:
        messagebox.showerror("Analiz Hatası", "Ses işlenirken bir hata oluştu.")
        lbl_sonuc.config(text="Analiz Başarısız", fg="black")

def ses_yukle_ve_test_et():
    dosya_yolu = filedialog.askopenfilename(
        title="Ses Dosyası Seçin",
        filetypes=(("WAV Dosyaları", "*.wav"), ("Tüm Dosyalar", "*.*"))
    )
    if dosya_yolu:
        lbl_durum.config(text=f"Seçilen: {os.path.basename(dosya_yolu)}")
        lbl_sonuc.config(text="V2 CNN Tensor üretiyor...", fg="blue")
        pencere.update()
        analizi_ekrana_yazdir(dosya_yolu)

def mikrofondan_kaydet():
    fs = 44100
    saniye = 3
    gecici_dosya = "anlik_kayit.wav"
    
    lbl_durum.config(text="🔴 MİKROFON AÇIK! Lütfen 3 saniye ses dinletin...", fg="red", font=("Segoe UI", 10, "bold"))
    lbl_sonuc.config(text="Dinleniyor...", fg="gray")
    pencere.update()
    
    try:
        kayit = sd.rec(int(saniye * fs), samplerate=fs, channels=1)
        sd.wait()
        write(gecici_dosya, fs, kayit)
        
        lbl_durum.config(text="✅ Kayıt Tamamlandı! CSCNN analiz ediyor...", fg="green", font=("Segoe UI", 10))
        pencere.update()
        
        analizi_ekrana_yazdir(gecici_dosya)
        
        if os.path.exists(gecici_dosya):
            os.remove(gecici_dosya)
            
    except Exception as e:
        messagebox.showerror("Mikrofon Hatası", f"Mikrofona erişilemedi.\nSistem hatası: {e}")
        lbl_durum.config(text="Kayıt Başarısız!", fg="red")

# Arayüz (GUI) Tasarımı
pencere = tk.Tk()
pencere.title("V2 Derin Öğrenme (CSCNN) Tavuk Ses Analizi")
pencere.geometry("550x450")
pencere.configure(bg="#f8f9fa")

lbl_baslik = tk.Label(pencere, text="V2 Derin Öğrenme (CSCNN) Merkezi", font=("Segoe UI", 16, "bold"), bg="#f8f9fa", fg="#2c3e50")
lbl_baslik.pack(pady=15)

lbl_altyazi = tk.Label(pencere, text="Model: tavuk_cnn_modeli.keras (3-Kanallı RGB Tensor)", font=("Segoe UI", 8), bg="#f8f9fa", fg="#95a5a6")
lbl_altyazi.pack()

frame_butonlar = tk.Frame(pencere, bg="#f8f9fa")
frame_butonlar.pack(pady=10)

btn_sec = tk.Button(frame_butonlar, text="📂 Dosya Yükle", command=ses_yukle_ve_test_et, font=("Segoe UI", 11, "bold"), bg="#3498db", fg="white", width=20, pady=8, cursor="hand2", relief="flat")
btn_sec.grid(row=0, column=0, padx=10)

btn_kaydet = tk.Button(frame_butonlar, text="🎙️ Anlık Kayıt (3 Sn)", command=mikrofondan_kaydet, font=("Segoe UI", 11, "bold"), bg="#e74c3c", fg="white", width=20, pady=8, cursor="hand2", relief="flat")
btn_kaydet.grid(row=0, column=1, padx=10)

lbl_durum = tk.Label(pencere, text="Sistem Hazır. Dosya seçin veya mikrofonla kayıt yapın.", font=("Segoe UI", 10), bg="#f8f9fa", fg="#7f8c8d")
lbl_durum.pack(pady=15)

lbl_sonuc = tk.Label(pencere, text="SONUÇ: Bekleniyor...", font=("Segoe UI", 18, "bold"), bg="#f8f9fa", fg="#34495e")
lbl_sonuc.pack(pady=20)

print("V2 GUI Yüklemesi Tamamlandı.")
pencere.mainloop()