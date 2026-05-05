"""
CIK-SOR V2 — TensorFlow Lite Model Donusturucu
================================================
Egitilmis Keras (.keras) modelini, Post-Training Quantization
uygulayarak hafif .tflite formatina cevirir.

Kullanim:
    python tflite_donusturucu.py
"""

import os
import sys

# tools/ klasöründen çalıştırıldığında proje kökünü ayarla
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf

# --- AYARLAR ---
KERAS_MODEL_YOLU = os.path.join("models", "tavuk_cnn_modeli.keras")
TFLITE_CIKTI_YOLU = os.path.join("models", "tavuk_edge_model.tflite")


def donustur():
    # 1. Egitilmis Keras modelini yukle
    if not os.path.exists(KERAS_MODEL_YOLU):
        print(f"HATA: '{KERAS_MODEL_YOLU}' bulunamadi.")
        print("Lutfen once egitim_cnn.py dosyasini calistirin.")
        return

    print(f"Keras modeli yukleniyor: {KERAS_MODEL_YOLU}")
    model = tf.keras.models.load_model(KERAS_MODEL_YOLU)

    # Orijinal model boyutunu kaydet (karsilastirma icin)
    orijinal_boyut = os.path.getsize(KERAS_MODEL_YOLU)

    # 2. TFLite Converter olustur
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 3. Post-Training Quantization (Egitim Sonrasi Niceleme)
    #    DEFAULT: float32 agirliklarini dynamic-range int8'e sikistirir
    #    Model boyutunu ~4x kucultir, CPU inference'i hizlandirir
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 4. Donusturme islemi
    print("TFLite donusumu baslatiliyor (Post-Training Quantization)...")
    tflite_model = converter.convert()

    # 5. Diske kaydet
    with open(TFLITE_CIKTI_YOLU, "wb") as f:
        f.write(tflite_model)

    tflite_boyut = os.path.getsize(TFLITE_CIKTI_YOLU)
    oran = (1 - tflite_boyut / orijinal_boyut) * 100

    print("\n" + "=" * 55)
    print("DONUSUM TAMAMLANDI")
    print("=" * 55)
    print(f"  Orijinal (.keras) : {orijinal_boyut / 1024 / 1024:.2f} MB")
    print(f"  TFLite (.tflite)  : {tflite_boyut / 1024 / 1024:.2f} MB")
    print(f"  Sikistirma orani  : %{oran:.1f} kuculme")
    print(f"  Cikti dosyasi     : {TFLITE_CIKTI_YOLU}")


if __name__ == "__main__":
    donustur()
