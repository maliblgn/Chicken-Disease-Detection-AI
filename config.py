"""
CİK-SÖR V1 Demo — Merkezi Konfigürasyon
========================================
Tüm sabitler, yollar, eşikler ve sınıf tanımları.
"""

import os

# ═══════════════════════════════════════════════════════════
#  PROJE KÖK DİZİNİ
# ═══════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════
#  SES İŞLEME SABİTLERİ
# ═══════════════════════════════════════════════════════════
SR = 22050                      # Örnekleme hızı (Hz)
SEGMENT_DURATION = 5.0          # Segment uzunluğu (saniye)
MIN_RMS_THRESHOLD = 0.01        # Sessizlik eşiği
NOISE_REDUCTION_RATIO = 0.85    # Gürültü azaltma oranı
NOISE_PROFILE_SECONDS = 0.5     # Gürültü profili süresi
N_FFT = 1024                    # FFT pencere boyutu
HOP_LENGTH = 256                # Hop uzunluğu

# ═══════════════════════════════════════════════════════════
#  ÖZELLİK ÇIKARIMI
# ═══════════════════════════════════════════════════════════
N_MFCC = 20                     # MFCC katsayı sayısı
IMG_SIZE = (128, 128)           # Spektrogram görsel boyutu
N_MELS = 128                    # Mel filtre sayısı
FMAX = 8000                     # Maksimum frekans (Hz)

# ═══════════════════════════════════════════════════════════
#  MODEL DOSYA YOLLARI
# ═══════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "tavuk_modeli.pkl")
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "tavuk_cnn_modeli.keras")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "tavuk_edge_model.tflite")

# ═══════════════════════════════════════════════════════════
#  VERİTABANI
# ═══════════════════════════════════════════════════════════
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
DB_PATH = os.path.join(LOGS_DIR, "ciksor.sqlite")

# ═══════════════════════════════════════════════════════════
#  SINIF TANIMLARI
# ═══════════════════════════════════════════════════════════
CLASSES = ["Healthy", "Noise", "Unhealthy"]  # CNN sınıf sırası (alfabetik)
RF_CLASSES = {0: "Healthy", 1: "Unhealthy", 2: "Noise"}

DISPLAY_LABELS = {
    "Healthy": "Normal Vokalizasyon",
    "Unhealthy": "Anormal / Hastalık Şüphesi Taşıyan Vokalizasyon",
    "Noise": "Gürültü / Analiz Dışı Ses",
}

DISPLAY_COLORS = {
    "Healthy": "#2ecc71",
    "Unhealthy": "#e74c3c",
    "Noise": "#f39c12",
}

# ═══════════════════════════════════════════════════════════
#  KARAR EŞİKLERİ
# ═══════════════════════════════════════════════════════════
CONFIDENCE_HIGH = 0.75          # Güçlü tahmin eşiği
CONFIDENCE_LOW = 0.50           # Düşük güvenli tahmin eşiği
ALERT_THRESHOLD = 0.75          # Uyarı üretme eşiği (Unhealthy için)

# ═══════════════════════════════════════════════════════════
#  MİKROFON KAYDI
# ═══════════════════════════════════════════════════════════
MIC_RECORD_SECONDS = 5          # Mikrofon kayıt süresi
MIC_CHANNELS = 1                # Mono

# ═══════════════════════════════════════════════════════════
#  MEVCUT VERİ KLASÖRLERİ (EĞİTİM İÇİN — DEĞİŞTİRME)
# ═══════════════════════════════════════════════════════════
TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "Veri_Egitim")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "Veri_Test")
RECORDS_DIR = os.path.join(PROJECT_ROOT, "Kayıtlar")
