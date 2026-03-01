import os
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

# Eğitilmiş Keras Modelini Yükle
print("Yapay Zeka (CNN) Modeli Belleğe Alınıyor...")
model = tf.keras.models.load_model(MODEL_YOLU)

# Tensorflow 'image_dataset_from_directory' varsayılan alfabetik sınıf sırası
SINIFLAR = ['Healthy', 'Noise', 'Unhealthy']
IMG_SIZE = (128, 128)

def sesi_spectrograma_cevir_ve_tahmin_et(dosya_yolu):
    gecici_png = "temp_spectrogram.png"
    try:
        # 1. Sesi yükle ve spectrogram oluştur (spektrogram_olusturucu.py ile birebir aynı!)
        y, sr = librosa.load(dosya_yolu, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 2. Resmi çiz ve kaydet (Eksensiz, boşluksuz saf ısı haritası)
        fig, ax = plt.subplots(figsize=(4, 4))
        librosa.display.specshow(S_dB, sr=sr, fmax=8000, ax=ax, cmap='magma')
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        plt.savefig(gecici_png, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # 3. Resmi Keras için yükle ve boyutlandır (Eğitimdeki boyut: 128x128)
        img = tf.keras.utils.load_img(gecici_png, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        
        # CNN eğitim sırasında Batch formatı(32, 128, 128, 3) beklentisindedir. 
        # Biz 1 resim yolladığımız için Expand ederek (1, 128, 128, 3) boyutuna çekiyoruz.
        img_array = tf.expand_dims(img_array, 0) 
        
        # Sıkıştırmayı (Rescaling 1./255) modelimizin ilk katmanına (layers.Rescaling) entegre etmiştik.
        # Bu nedenle veriyi manuel olarak /255'e bölmeye gerek YOKTUR.
        
        # 4. Modele tahmin ettir!
        tahmin_olasiliklari = model.predict(img_array)
        tahmin_edilen_sinif_index = np.argmax(tahmin_olasiliklari[0])
        
        tahmin_edilen_sinif_ismi = SINIFLAR[tahmin_edilen_sinif_index]
        guven_orani = np.max(tahmin_olasiliklari[0]) * 100
        
        # 5. Temizlik (Geçici izleri sil)
        if os.path.exists(gecici_png):
            os.remove(gecici_png)
            
        return tahmin_edilen_sinif_ismi, guven_orani
        
    except Exception as e:
        print(f"Hata detayı: {e}")
        # Temizliği hata anında da yapmayı dene
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
        else: # Noise
            gosterim = "GÜRÜLTÜ (Noise)"
            renk = "#f39c12"
            
        lbl_sonuc.config(text=f"SONUÇ: {gosterim}\n(Gizli Özgüven: %{guven:.1f})", fg=renk)
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
        lbl_sonuc.config(text="CNN Spektrogramı çiziyor...", fg="blue")
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
        
        lbl_durum.config(text="✅ Kayıt Tamamlandı! CNN analiz ediyor...", fg="green", font=("Segoe UI", 10))
        pencere.update()
        
        analizi_ekrana_yazdir(gecici_dosya)
        
        if os.path.exists(gecici_dosya):
            os.remove(gecici_dosya)
            
    except Exception as e:
        messagebox.showerror("Mikrofon Hatası", f"Mikrofona erişilemedi.\nSistem hatası: {e}")
        lbl_durum.config(text="Kayıt Başarısız!", fg="red")

# Arayüz (GUI) Tasarımı (Eski Tasarıma Sadık Kalındı, Başlıklar Güncellendi)
pencere = tk.Tk()
pencere.title("Derin Öğrenme (CNN) Tavuk Ses Analizi")
pencere.geometry("550x450")
pencere.configure(bg="#f8f9fa")

lbl_baslik = tk.Label(pencere, text="Derin Öğrenme (CNN) Tavuk Ses Analizi", font=("Segoe UI", 16, "bold"), bg="#f8f9fa", fg="#2c3e50")
lbl_baslik.pack(pady=15)

# Biraz teknik detay gösterelim
lbl_altyazi = tk.Label(pencere, text="Model: tavuk_cnn_modeli.keras (128x128 Px Mel-Spectrogram)", font=("Segoe UI", 8), bg="#f8f9fa", fg="#95a5a6")
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

print("GUI Yüklemesi Tamamlandı.")
pencere.mainloop()
