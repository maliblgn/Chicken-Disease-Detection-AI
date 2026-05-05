"""
CİK-SÖR V1 Demo — Karar Katmanı
=================================
Güven eşikleri, uyarı üretimi, kullanıcı dostu etiketler.
"""

import logging
import config

logger = logging.getLogger("ciksor.decision")


def evaluate_prediction(class_name: str, confidence: float):
    """
    Tahmin sonucunu değerlendirir.

    Returns:
        dict: {
            "class": "Unhealthy",
            "display_label": "Anormal / Hastalık Şüphesi ...",
            "confidence": 0.92,
            "confidence_level": "high" | "low" | "uncertain",
            "alert": True/False,
            "alert_message": "..." veya None,
            "color": "#e74c3c"
        }
    """
    display = config.DISPLAY_LABELS.get(class_name, class_name)
    color = config.DISPLAY_COLORS.get(class_name, "#95a5a6")

    # Güven seviyesi
    if confidence >= config.CONFIDENCE_HIGH:
        level = "high"
    elif confidence >= config.CONFIDENCE_LOW:
        level = "low"
    else:
        level = "uncertain"

    # Uyarı kontrolü
    alert = False
    alert_message = None
    if class_name == "Unhealthy" and confidence >= config.ALERT_THRESHOLD:
        alert = True
        alert_message = (
            f"⚠ Anormal vokalizasyon tespit edildi (Güven: %{confidence*100:.1f}). "
            f"Veteriner kontrolü önerilir. Bu sonuç kesin teşhis değildir; "
            f"erken uyarı amaçlıdır."
        )

    return {
        "class": class_name,
        "display_label": display,
        "confidence": confidence,
        "confidence_level": level,
        "alert": alert,
        "alert_message": alert_message,
        "color": color,
    }


def confidence_text(level: str) -> str:
    """Güven seviyesi metni."""
    return {
        "high": "Güçlü Tahmin",
        "low": "Düşük Güvenli Tahmin",
        "uncertain": "Belirsiz — Tekrar Analiz Gerekli",
    }.get(level, "Bilinmiyor")
