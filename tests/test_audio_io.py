"""CİK-SÖR — Ses I/O Testleri"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.audio_io import load_audio


def test_load_nonexistent_file():
    y, sr, err = load_audio("bu_dosya_yok.wav")
    assert y is None
    assert err != ""
    print("[PASS] Olmayan dosya testi")


def test_load_audio_returns_correct_sr():
    # Veri_Test klasöründen bir dosya varsa test et
    test_dir = os.path.join(config.TEST_DATA_DIR, "Healthy")
    if not os.path.exists(test_dir):
        print("[SKIP] Test veri klasörü yok")
        return

    files = [f for f in os.listdir(test_dir) if f.endswith(".wav")]
    if not files:
        print("[SKIP] Test ses dosyası yok")
        return

    y, sr, err = load_audio(os.path.join(test_dir, files[0]))
    assert err == ""
    assert sr == config.SR
    assert len(y) > 0
    print("[PASS] Ses yükleme testi")


if __name__ == "__main__":
    test_load_nonexistent_file()
    test_load_audio_returns_correct_sr()
    print("\nTüm audio_io testleri tamamlandı.")
