"""CİK-SÖR — Ön İşleme Testleri"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing import preprocess, pad_or_trim


def test_pad_short_audio():
    y = np.zeros(1000, dtype=np.float32)
    result = pad_or_trim(y, config.SR)
    expected_len = int(config.SR * config.SEGMENT_DURATION)
    assert len(result) == expected_len
    print("[PASS] Kısa ses padding testi")


def test_trim_long_audio():
    y = np.zeros(config.SR * 10, dtype=np.float32)
    result = pad_or_trim(y, config.SR)
    expected_len = int(config.SR * config.SEGMENT_DURATION)
    assert len(result) == expected_len
    print("[PASS] Uzun ses trim testi")


def test_preprocess_normalizes():
    y = np.random.randn(config.SR).astype(np.float32) * 0.5
    result = preprocess(y, config.SR)
    rms = np.sqrt(np.mean(result ** 2))
    assert abs(rms - 0.1) < 0.01
    print("[PASS] RMS normalize testi")


if __name__ == "__main__":
    test_pad_short_audio()
    test_trim_long_audio()
    test_preprocess_normalizes()
    print("\nTüm preprocessing testleri tamamlandı.")
