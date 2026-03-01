import os
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import gradio as gr

# Model Yolları
RF_MODEL_YOLU = 'tavuk_modeli.pkl'
CNN_MODEL_YOLU = 'tavuk_cnn_modeli.keras'

# Modelleri Yükle
print("Modeller yükleniyor...")
try:
    rf_model = joblib.load(RF_MODEL_YOLU)
    cnn_model = tf.keras.models.load_model(CNN_MODEL_YOLU)
    print("Modeller başarıyla yüklendi!")
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    rf_model = None
    cnn_model = None

# Sınıf İsimleri
RF_SINIFLAR = {0: "Healthy", 1: "Unhealthy", 2: "Noise"}
CNN_SINIFLAR = ["Healthy", "Noise", "Unhealthy"]
IMG_SIZE = (128, 128)

def analyze_audio(audio_path):
    if not audio_path:
        return "Lütfen bir ses dosyası yükleyin.", "Lütfen bir ses dosyası yükleyin."
    
    if rf_model is None or cnn_model is None:
        return "Modeller yüklenemedi!", "Modeller yüklenemedi!"

    try:
        # Sesi ortak olarak yükle
        y, sr = librosa.load(audio_path, sr=None)
        
        # ---------------------------------------------------------
        # 1. KLASİK MODEL (Random Forest) İÇİN ÇIKARIM (MFCC)
        # ---------------------------------------------------------
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        rf_tahmin_idx = rf_model.predict([mfccs_mean])[0]
        rf_sonuc = RF_SINIFLAR.get(rf_tahmin_idx, "Bilinmeyen")
        
        # ---------------------------------------------------------
        # 2. DERİN ÖĞRENME (CNN) İÇİN ÇIKARIM (Mel-Spectrogram)
        # ---------------------------------------------------------
        gecici_png = "temp_hf_spectrogram.png"
        
        # Spectrogram oluştur
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig, ax = plt.subplots(figsize=(4, 4))
        librosa.display.specshow(S_dB, sr=sr, fmax=8000, ax=ax, cmap='magma')
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        plt.savefig(gecici_png, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # Görüntüyü modele ver
        img = tf.keras.utils.load_img(gecici_png, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) 
        
        cnn_tahmin_olasiliklari = cnn_model.predict(img_array)
        cnn_tahmin_idx = np.argmax(cnn_tahmin_olasiliklari[0])
        cnn_guven = np.max(cnn_tahmin_olasiliklari[0]) * 100
        cnn_sonuc = f"{CNN_SINIFLAR[cnn_tahmin_idx]} (%{cnn_guven:.1f})"
        
        # Geçici dosyayı temizle
        if os.path.exists(gecici_png):
            os.remove(gecici_png)
            
        return rf_sonuc, cnn_sonuc

    except Exception as e:
        return f"Hata: {str(e)}", f"Hata: {str(e)}"

# Gradio Arayüzü (A/B Test Formatında)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐔 CIK-SOR: Tavuk Ses Analizi (Canlı A/B Testi)")
    gr.Markdown("Bu platform Hugging Face Spaces üzerinde canlı test için tasarlanmıştır. "
                "Bir ses dosyası yükleyin veya mikrofondan kaydedin, iki farklı yapay zeka jenerasyonunun kararını canlı karşılaştırın.")
    
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Ses Girdisi (Mikrofon / Dosya)")
        
    with gr.Row():
        btn = gr.Button("Analizi Başlat 🚀", variant="primary")
        
    with gr.Row():
        rf_output = gr.Textbox(label="🌳 Klasik Makine Öğrenmesi (Random Forest)")
        cnn_output = gr.Textbox(label="🧠 Derin Öğrenme (CNN)")
        
    btn.click(fn=analyze_audio, inputs=audio_input, outputs=[rf_output, cnn_output])

if __name__ == "__main__":
    demo.launch()
