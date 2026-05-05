"""
CIK-SOR V2 — TFLite Edge Inference Motoru
==========================================
Ağır TensorFlow/Keras yerine sadece TFLite Interpreter kullanarak
tahmin yapar. Raspberry Pi ve benzeri uç cihazlar için optimize.

Pipeline entegrasyonu:
    from tflite_motoru import EdgePredictor
    motor = EdgePredictor("tavuk_edge_model.tflite")
    sinif, skor = motor.predict(spektrogram_128x128x3)
"""

import os
import numpy as np

# TFLite runtime varsa onu kullan (Raspberry Pi'de sadece ~1 MB),
# yoksa tam TensorFlow icindeki interpreter'a dus
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


class EdgePredictor:
    """
    TFLite tabanli hafif cikarim (inference) motoru.
    Keras veya TensorFlow model.predict() KULLANMAZ.
    Tum islem TFLite Interpreter uzerinden yurutur.
    """

    # Sinif isimleri (egitim sirasina sadik — alfabetik)
    SINIFLAR = ["Healthy (Saglikli)", "Noise (Gurultu)", "Unhealthy (Hasta)"]

    def __init__(self, model_yolu: str = "tavuk_edge_model.tflite"):
        """
        TFLite modelini belleğe alır ve tensor bilgilerini okur.

        Args:
            model_yolu: .tflite dosyasinin yolu.
        """
        if not os.path.exists(model_yolu):
            raise FileNotFoundError(
                f"TFLite modeli bulunamadi: {model_yolu}\n"
                "Lutfen once tflite_donusturucu.py dosyasini calistirin."
            )

        # Interpreter'i baslat ve tensor bellegini ayir
        self._interpreter = Interpreter(model_path=model_yolu)
        self._interpreter.allocate_tensors()

        # Giris/cikis tensor meta bilgilerini bir kez oku (performans)
        self._giris_detay = self._interpreter.get_input_details()[0]
        self._cikis_detay = self._interpreter.get_output_details()[0]

        # Beklenen giris boyutu (ornegin [1, 128, 128, 3])
        self._giris_shape = self._giris_detay["shape"]
        self._giris_dtype = self._giris_detay["dtype"]

    def predict(self, spektrogram: np.ndarray) -> tuple[str, float]:
        """
        3 kanalli spektrogram gorselini (numpy) modele verir,
        sinif ismi ve guven skorunu dondurur.

        Args:
            spektrogram: (128, 128, 3) veya (1, 128, 128, 3) boyutlu numpy dizisi.
                         Piksel degerleri 0-255 (uint8) veya 0.0-255.0 (float) olabilir.

        Returns:
            (sinif_ismi, guven_skoru_yuzde)
            Ornek: ("Unhealthy (Hasta)", 92.4)
        """
        # --- Giris hazirlik ---
        giris = self._girisi_hazirla(spektrogram)

        # --- TFLite Inference (Keras KULLANILMIYOR) ---
        self._interpreter.set_tensor(self._giris_detay["index"], giris)
        self._interpreter.invoke()
        cikis = self._interpreter.get_tensor(self._cikis_detay["index"])

        # --- Sonucu yorumla ---
        olasiliklar = cikis[0]
        tahmin_idx = int(np.argmax(olasiliklar))
        guven = float(np.max(olasiliklar)) * 100.0

        sinif_ismi = self.SINIFLAR[tahmin_idx] if tahmin_idx < len(self.SINIFLAR) else "Bilinmeyen"

        return sinif_ismi, round(guven, 2)

    def _girisi_hazirla(self, spektrogram: np.ndarray) -> np.ndarray:
        """
        Ham numpy dizisini interpreter'in bekledigiBatch + boyut + dtype
        formatina donusturur.
        """
        arr = spektrogram.copy()

        # Batch boyutu yoksa ekle: (128,128,3) -> (1,128,128,3)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)

        # dtype uyumu (quantized model int8 bekleyebilir)
        if arr.dtype != self._giris_dtype:
            arr = arr.astype(self._giris_dtype)

        return arr

    @property
    def giris_boyutu(self) -> tuple:
        """Modelin beklediği giriş tensör boyutunu döndürür."""
        return tuple(self._giris_shape)


# =====================================================================
#  BAGIMSIZ TEST  (python tflite_motoru.py)
# =====================================================================

if __name__ == "__main__":
    TFLITE_YOLU = "tavuk_edge_model.tflite"

    if not os.path.exists(TFLITE_YOLU):
        print(f"Test icin '{TFLITE_YOLU}' gerekli.")
        print("Once tflite_donusturucu.py dosyasini calistirin.")
        exit(1)

    motor = EdgePredictor(TFLITE_YOLU)

    print("=" * 55)
    print("CIK-SOR TFLite Edge Motoru - Bagimsiz Test")
    print("=" * 55)
    print(f"  Model          : {TFLITE_YOLU}")
    print(f"  Giris boyutu   : {motor.giris_boyutu}")
    print(f"  Giris dtype    : {motor._giris_dtype}")
    print(f"  Siniflar       : {motor.SINIFLAR}")

    # Rastgele sahte spektrogram ile inference testi
    # Gercek kullanımda buraya 128x128x3 RGB spektrogram gelecek
    sahte_giris = np.random.rand(128, 128, 3).astype(np.float32) * 255.0
    sinif, skor = motor.predict(sahte_giris)

    print(f"\n  [TEST] Rastgele giris ile tahmin:")
    print(f"         Sinif : {sinif}")
    print(f"         Skor  : %{skor}")

    # Hiz testi (10 ardisik inference)
    import time
    basla = time.perf_counter()
    for _ in range(10):
        motor.predict(sahte_giris)
    gecen = time.perf_counter() - basla

    print(f"\n  [HIZ]  10 inference suresi : {gecen * 1000:.1f} ms")
    print(f"         Ortalama latency   : {gecen / 10 * 1000:.1f} ms/tahmin")
    print("\n  Tum testler basarili.")
