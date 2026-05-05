"""
CİK-SÖR V1 Demo — Spektrogram Üretim Modülü
=============================================
128x128x3 RGB spektrogram: R=Mel, G=Delta, B=Delta-Delta.
"""

import numpy as np
import librosa
from PIL import Image
import logging

import config

logger = logging.getLogger("ciksor.spectrogram")


def _normalize_channel(matrix: np.ndarray) -> np.ndarray:
    """Matrisi 0-255 piksel aralığına sıkıştırır."""
    mn, mx = np.min(matrix), np.max(matrix)
    if mx - mn == 0:
        return np.zeros_like(matrix, dtype=np.uint8)
    return ((matrix - mn) / (mx - mn) * 255).astype(np.uint8)


def create_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """
    128x128x3 RGB spektrogram üretir.

    Returns:
        float32 numpy array (128, 128, 3)
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS, fmax=config.FMAX
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    delta = librosa.feature.delta(S_dB)
    delta2 = librosa.feature.delta(S_dB, order=2)

    R = _normalize_channel(S_dB)
    G = _normalize_channel(delta)
    B = _normalize_channel(delta2)

    img_array = np.stack([R, G, B], axis=-1)
    img_array = np.flipud(img_array)

    img = Image.fromarray(img_array, "RGB")
    img = img.resize(config.IMG_SIZE, Image.Resampling.LANCZOS)

    result = np.array(img, dtype=np.float32)
    logger.debug(f"Spektrogram üretildi: {result.shape}")
    return result


def create_spectrogram_image(y: np.ndarray, sr: int):
    """
    Görselleştirme için PIL Image döndürür.
    """
    spec_array = create_spectrogram(y, sr)
    return Image.fromarray(spec_array.astype(np.uint8), "RGB")
