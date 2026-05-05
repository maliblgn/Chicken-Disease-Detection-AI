"""
CİK-SÖR V1 Demo — Ses Giriş/Çıkış Modülü
==========================================
Ses dosyası okuma, video'dan ses çıkarma, mikrofon kaydı.
"""

import os
import logging
import numpy as np
import librosa

import config

logger = logging.getLogger("ciksor.audio_io")


def load_audio(file_path: str):
    """
    Ses veya video dosyasını okur.

    Returns:
        (audio_array, sample_rate, error_message)
        Hata varsa audio_array=None, error_message dolu.
    """
    if not file_path or not os.path.exists(file_path):
        return None, config.SR, "Dosya bulunamadı."

    ext = os.path.splitext(file_path)[1].lower()

    # Video dosyası ise sesi çıkar
    if ext in (".mp4", ".avi", ".mkv", ".mov"):
        return _load_from_video(file_path)

    # Ses dosyası
    try:
        y, sr = librosa.load(file_path, sr=config.SR, mono=True)
        if y is None or len(y) == 0:
            return None, config.SR, "Dosyadan ses verisi okunamadı."
        logger.info(
            f"Ses yüklendi: {os.path.basename(file_path)} | "
            f"{len(y)/sr:.1f}s | SR={sr}"
        )
        return y, sr, ""
    except Exception as e:
        logger.error(f"Ses okuma hatası: {e}")
        return None, config.SR, f"Ses dosyası okunamadı: {e}"


def _load_from_video(file_path: str):
    """Video dosyasından sesi çıkarır."""
    try:
        from moviepy import VideoFileClip
    except ImportError:
        return None, config.SR, "Video desteği için 'moviepy' kütüphanesi gerekli."

    temp_path = os.path.join(config.DATA_DIR, "processed", "_temp_video_audio.wav")
    try:
        video = VideoFileClip(file_path)
        if video.audio is None:
            video.close()
            return None, config.SR, "Videoda ses kanalı bulunamadı."

        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        video.audio.write_audiofile(temp_path, fps=config.SR, logger=None)
        video.close()

        y, sr = librosa.load(temp_path, sr=config.SR, mono=True)
        return y, sr, ""
    except Exception as e:
        logger.error(f"Video ses çıkarma hatası: {e}")
        return None, config.SR, f"Video işlenemedi: {e}"
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def record_from_mic(duration: float = None):
    """
    Mikrofondan kayıt alır.

    Returns:
        (audio_array, sample_rate, error_message)
    """
    if duration is None:
        duration = config.MIC_RECORD_SECONDS

    try:
        import sounddevice as sd
    except ImportError:
        return None, config.SR, "Mikrofon desteği için 'sounddevice' kütüphanesi gerekli."

    try:
        logger.info(f"Mikrofon kaydı başlıyor ({duration}s)...")
        recording = sd.rec(
            int(duration * config.SR),
            samplerate=config.SR,
            channels=config.MIC_CHANNELS,
            dtype="float32",
        )
        sd.wait()
        y = recording[:, 0]  # Mono
        logger.info(f"Mikrofon kaydı tamamlandı: {len(y)} örnek")
        return y, config.SR, ""
    except Exception as e:
        logger.error(f"Mikrofon hatası: {e}")
        return None, config.SR, f"Mikrofon erişilemedi: {e}"
