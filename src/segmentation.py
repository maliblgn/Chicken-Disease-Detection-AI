"""
CİK-SÖR V1 Demo — Ses Segmentleme Modülü
==========================================
Sesi 5 saniyelik segmentlere böler, sessiz segmentleri filtreler.
"""

import numpy as np
import logging

import config

logger = logging.getLogger("ciksor.segmentation")


def segment_audio(y: np.ndarray, sr: int):
    """
    Sesi 5 saniyelik segmentlere böler.

    Returns:
        list[np.ndarray] — Anlamlı segmentler
    """
    segment_len = int(sr * config.SEGMENT_DURATION)

    if len(y) <= segment_len:
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms < config.MIN_RMS_THRESHOLD:
            logger.info(f"Ses çok sessiz (RMS: {rms:.4f}), analiz dışı.")
            return []
        return [y]

    segments = []
    num_segments = int(np.ceil(len(y) / segment_len))

    for i in range(num_segments):
        start = i * segment_len
        end = min((i + 1) * segment_len, len(y))
        seg = y[start:end]

        # Kısa son segmenti padding ile tamamla
        if len(seg) < segment_len:
            seg = np.pad(seg, (0, segment_len - len(seg)), mode="constant")

        # Sessizlik kontrolü
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < config.MIN_RMS_THRESHOLD:
            continue

        segments.append(seg)

    logger.info(f"Segmentleme: {num_segments} parçadan {len(segments)} tanesi anlamlı.")
    return segments
