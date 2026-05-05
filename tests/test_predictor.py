"""CİK-SÖR — Predictor Testleri"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.predictor import ModelManager


def test_mock_deterministic():
    """Aynı girdiye aynı sonuç vermeli."""
    mm = ModelManager.__new__(ModelManager)
    mm.rf_model = None
    mm.cnn_model = None
    mm.tflite_interpreter = None
    mm.mock_mode = True

    y = np.random.randn(1000).astype(np.float32)
    r1 = mm.predict_mock(y)
    r2 = mm.predict_mock(y)
    assert r1 == r2, "Mock tahmin deterministik değil!"
    assert r1[0] in config.CLASSES
    assert 0.0 <= r1[1] <= 1.0
    print("[PASS] Mock deterministik tahmin testi")


def test_model_status():
    """get_status doğru döndürmeli."""
    mm = ModelManager.__new__(ModelManager)
    mm.rf_model = None
    mm.cnn_model = None
    mm.tflite_interpreter = None
    mm.mock_mode = True

    status = mm.get_status()
    assert status["mock_mode"] is True
    assert status["rf"] is False
    print("[PASS] Model durum testi")


if __name__ == "__main__":
    test_mock_deterministic()
    test_model_status()
    print("\nTüm predictor testleri tamamlandı.")
