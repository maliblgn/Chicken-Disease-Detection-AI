"""
CİK-SÖR V1 Demo — Ses Ön İşleme Modülü
========================================
Normalize, mono, padding, trim işlemleri.
"""

import numpy as np
import logging

import config

logger = logging.getLogger("ciksor.preprocessing")


def preprocess(y: np.ndarray, sr: int) -> np.ndarray:
    """Sesi RMS normalize eder."""
    if y is None or len(y) == 0:
        return y

    rms = np.sqrt(np.mean(y ** 2))
    if rms > 1e-6:
        y = y / (rms + 1e-10) * 0.1
        logger.debug(f"RMS normalize: {rms:.4f} → 0.1")
    return y


def pad_or_trim(y: np.ndarray, sr: int, target_duration: float = None) -> np.ndarray:
    """Sesi hedef süreye tamamlar veya keser."""
    if target_duration is None:
        target_duration = config.SEGMENT_DURATION

    target_len = int(sr * target_duration)

    if len(y) < target_len:
        pad_len = target_len - len(y)
        y = np.pad(y, (0, pad_len), mode="constant", constant_values=0)
        logger.debug(f"Padding: {pad_len} örnek eklendi")
    elif len(y) > target_len:
        y = y[:target_len]
        logger.debug(f"Trim: {target_len} örneğe kesildi")

    return y
