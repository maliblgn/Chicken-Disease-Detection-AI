"""
CİK-SÖR V1 Demo — Özellik Çıkarım Modülü
==========================================
RF modeli için 120 boyutlu MFCC + Delta + Delta-Delta vektörü.
"""

import numpy as np
import librosa
import logging

import config

logger = logging.getLogger("ciksor.features")


def extract_rf_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    120 boyutlu RF özellik vektörü çıkarır.
    20 MFCC + 20 Delta + 20 Delta-Delta × (mean + std) = 120 boyut
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)

    features = np.hstack([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1),
        np.mean(delta2, axis=1), np.std(delta2, axis=1),
    ])

    logger.debug(f"RF özellik vektörü: {features.shape[0]} boyut")
    return features
