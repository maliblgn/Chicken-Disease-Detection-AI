"""
CİK-SÖR V1 Demo — Yardımcı Fonksiyonlar
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger("ciksor")


def setup_logging():
    """Uygulama logging yapılandırması."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directories():
    """Gerekli runtime klasörlerini otomatik oluşturur."""
    import config

    dirs = [
        config.MODELS_DIR,
        config.DATA_DIR,
        config.LOGS_DIR,
        os.path.join(config.DATA_DIR, "raw"),
        os.path.join(config.DATA_DIR, "processed"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Klasör yapısı kontrol edildi.")


def timestamp():
    """Şu anki zamanı string olarak döndürür."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
