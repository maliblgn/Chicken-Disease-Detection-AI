"""
CİK-SÖR V1 Demo — Model Yükleme ve Tahmin Modülü
==================================================
RF, CNN ve TFLite modellerini yükler.
Model yoksa deterministik mock mode ile çalışır.
"""

import os
import hashlib
import time
import numpy as np
import logging

import config

logger = logging.getLogger("ciksor.predictor")


class ModelManager:
    """RF, CNN ve TFLite modellerini yönetir."""

    def __init__(self):
        self.rf_model = None
        self.cnn_model = None
        self.tflite_interpreter = None
        self._tflite_input = None
        self._tflite_output = None
        self.mock_mode = False
        self._load_models()

    def _load_models(self):
        """Tüm modelleri yüklemeye çalışır."""
        loaded_any = False

        # RF Model
        if os.path.exists(config.RF_MODEL_PATH):
            try:
                import joblib
                self.rf_model = joblib.load(config.RF_MODEL_PATH)
                logger.info(f"RF modeli yüklendi: {config.RF_MODEL_PATH}")
                loaded_any = True
            except Exception as e:
                logger.error(f"RF model yükleme hatası: {e}")
        else:
            logger.warning("RF model dosyası bulunamadı.")

        # CNN Model
        if os.path.exists(config.CNN_MODEL_PATH):
            try:
                import tensorflow as tf
                self.cnn_model = tf.keras.models.load_model(config.CNN_MODEL_PATH)
                logger.info(f"CNN modeli yüklendi: {config.CNN_MODEL_PATH}")
                loaded_any = True
            except Exception as e:
                logger.error(f"CNN model yükleme hatası: {e}")
        else:
            logger.warning("CNN model dosyası bulunamadı.")

        # TFLite Model (opsiyonel)
        if os.path.exists(config.TFLITE_MODEL_PATH):
            try:
                try:
                    from tflite_runtime.interpreter import Interpreter
                except ImportError:
                    from tensorflow.lite.python.interpreter import Interpreter

                interp = Interpreter(model_path=config.TFLITE_MODEL_PATH)
                interp.allocate_tensors()
                self._tflite_input = interp.get_input_details()[0]
                self._tflite_output = interp.get_output_details()[0]
                self.tflite_interpreter = interp
                logger.info(f"TFLite modeli yüklendi: {config.TFLITE_MODEL_PATH}")
                loaded_any = True
            except Exception as e:
                logger.error(f"TFLite model yükleme hatası: {e}")
        else:
            logger.info("TFLite model dosyası bulunamadı (opsiyonel).")

        if not loaded_any:
            self.mock_mode = True
            logger.warning("HİÇ MODEL YÜKLENEMEDİ — Demo/Mock mode aktif.")

    # ─── TAHMİN FONKSİYONLARI ───

    def predict_rf(self, features: np.ndarray):
        """
        RF modeli ile tahmin.
        Returns: (sınıf_adı, güven_skoru) veya None
        """
        if self.rf_model is None:
            return None

        try:
            idx = self.rf_model.predict([features])[0]
            class_name = config.RF_CLASSES.get(idx, "Noise")

            # RF probabilities (destekliyorsa)
            try:
                probs = self.rf_model.predict_proba([features])[0]
                confidence = float(np.max(probs))
            except Exception:
                confidence = 0.80  # RF varsayılan güven

            return class_name, confidence
        except Exception as e:
            logger.error(f"RF tahmin hatası: {e}")
            return None

    def predict_cnn(self, spectrogram: np.ndarray):
        """
        CNN modeli ile tahmin.
        Returns: (sınıf_adı, güven_skoru) veya None
        """
        if self.cnn_model is None:
            return None

        try:
            inp = np.expand_dims(spectrogram, 0)
            probs = self.cnn_model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(probs))
            class_name = config.CLASSES[idx]
            confidence = float(np.max(probs))
            return class_name, confidence
        except Exception as e:
            logger.error(f"CNN tahmin hatası: {e}")
            return None

    def predict_tflite(self, spectrogram: np.ndarray):
        """
        TFLite modeli ile tahmin.
        Returns: (sınıf_adı, güven_skoru, latency_ms) veya None
        """
        if self.tflite_interpreter is None:
            return None

        try:
            arr = spectrogram.copy()
            if arr.ndim == 3:
                arr = np.expand_dims(arr, 0)
            if arr.dtype != self._tflite_input["dtype"]:
                arr = arr.astype(self._tflite_input["dtype"])

            t0 = time.perf_counter()
            self.tflite_interpreter.set_tensor(self._tflite_input["index"], arr)
            self.tflite_interpreter.invoke()
            output = self.tflite_interpreter.get_tensor(self._tflite_output["index"])
            latency = (time.perf_counter() - t0) * 1000

            probs = output[0]
            idx = int(np.argmax(probs))
            class_name = config.CLASSES[idx]
            confidence = float(np.max(probs))
            return class_name, confidence, latency
        except Exception as e:
            logger.error(f"TFLite tahmin hatası: {e}")
            return None

    def predict_mock(self, y: np.ndarray):
        """
        Deterministik mock tahmin. Aynı girdiye aynı sonucu verir.
        """
        # Ses verisinin hash'inden deterministik sonuç üret
        data_hash = hashlib.md5(y[:1000].tobytes()).hexdigest()
        hash_int = int(data_hash[:8], 16)
        idx = hash_int % 3
        class_name = config.CLASSES[idx]
        # Deterministik güven skoru (0.55 - 0.95 arası)
        confidence = 0.55 + (hash_int % 40) / 100.0
        return class_name, confidence

    # ─── DURUM KONTROLÜ ───

    def get_status(self):
        """Model durumlarını döndürür."""
        return {
            "rf": self.rf_model is not None,
            "cnn": self.cnn_model is not None,
            "tflite": self.tflite_interpreter is not None,
            "mock_mode": self.mock_mode,
        }
