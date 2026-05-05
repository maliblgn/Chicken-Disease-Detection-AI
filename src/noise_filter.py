"""
CİK-SÖR V1 Demo — Gürültü Azaltma Modülü
==========================================
Spektral çıkarma yöntemiyle sabit arka plan gürültüsü temizleme.
"""

import numpy as np
import logging

import config

logger = logging.getLogger("ciksor.noise_filter")


def reduce_noise(y: np.ndarray, sr: int):
    """
    Sesi gürültüden arındırır.

    Returns:
        (temiz_ses, istatistik_dict)
    """
    if y is None or len(y) == 0:
        return y, {"rms_before": 0, "rms_after": 0, "reduction_pct": 0}

    rms_before = float(np.sqrt(np.mean(y ** 2)))

    try:
        import noisereduce as nr
    except ImportError:
        logger.warning("noisereduce yok, gürültü azaltma atlanıyor.")
        return y, {"rms_before": rms_before, "rms_after": rms_before, "reduction_pct": 0}

    profile_len = int(sr * config.NOISE_PROFILE_SECONDS)
    if len(y) <= profile_len:
        return y, {"rms_before": rms_before, "rms_after": rms_before, "reduction_pct": 0}

    noise_profile = _find_quietest_segment(y, profile_len)

    try:
        clean = nr.reduce_noise(
            y=y,
            sr=sr,
            y_noise=noise_profile,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            prop_decrease=config.NOISE_REDUCTION_RATIO,
            stationary=True,
        )
        clean = clean.astype(y.dtype)
    except Exception as e:
        logger.error(f"Gürültü azaltma hatası: {e}")
        return y, {"rms_before": rms_before, "rms_after": rms_before, "reduction_pct": 0}

    rms_after = float(np.sqrt(np.mean(clean ** 2)))
    reduction = ((rms_before - rms_after) / rms_before * 100) if rms_before > 0 else 0

    logger.info(f"Gürültü azaltma: RMS {rms_before:.4f} → {rms_after:.4f} ({reduction:.1f}%)")
    return clean, {
        "rms_before": rms_before,
        "rms_after": rms_after,
        "reduction_pct": reduction,
    }


def _find_quietest_segment(y: np.ndarray, segment_len: int) -> np.ndarray:
    """En sessiz dilimi bulur (gürültü profili)."""
    min_rms = np.inf
    min_start = 0

    for start in range(0, len(y) - segment_len + 1, segment_len):
        segment = y[start : start + segment_len]
        rms = float(np.sqrt(np.mean(segment ** 2)))
        if rms < min_rms:
            min_rms = rms
            min_start = start

    return y[min_start : min_start + segment_len]
