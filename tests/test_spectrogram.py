"""CİK-SÖR — Spektrogram Testleri"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.spectrogram import create_spectrogram


def test_spectrogram_shape():
    y = np.random.randn(config.SR * 5).astype(np.float32) * 0.1
    result = create_spectrogram(y, config.SR)
    assert result.shape == (128, 128, 3), f"Beklenen (128,128,3), alınan {result.shape}"
    assert result.dtype == np.float32
    print("[PASS] Spektrogram boyut testi")


def test_spectrogram_range():
    y = np.random.randn(config.SR * 5).astype(np.float32) * 0.1
    result = create_spectrogram(y, config.SR)
    assert np.min(result) >= 0
    assert np.max(result) <= 255
    print("[PASS] Spektrogram değer aralığı testi")


if __name__ == "__main__":
    test_spectrogram_shape()
    test_spectrogram_range()
    print("\nTüm spectrogram testleri tamamlandı.")
