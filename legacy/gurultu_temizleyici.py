"""
CIK-SOR V2 — Algoritmik Gürültü İptali (Pre-Processing Katmanı)
=================================================================
Ring Buffer'dan gelen 5 saniyelik ham sesi, yapay zeka modeline
girmeden önce sabit frekanslı arka plan gürültülerinden (fan, uğultu)
arındırır.

Yöntem:
  1. Sinyalin en düşük enerjili 0.5 saniyelik dilimi → gürültü profili
  2. Spektral çıkarma (noisereduce) ile profil tüm sinyalden temizlenir

Pipeline entegrasyonu:
  from gurultu_temizleyici import gurultu_temizle
  temiz = gurultu_temizle(ham_ses, sr)
"""

import numpy as np
import noisereduce as nr


# ─── Performans sabitleri (1 sn'lik analiz döngüsüne sığacak trade-off) ───
_N_FFT = 1024          # Küçük pencere → düşük gecikme, yeterli frekans çözünürlüğü
_HOP_LENGTH = 256      # %75 örtüşme → yumuşak geçiş, artefakt minimumda
_PROFIL_SANIYE = 0.5   # Gürültü profili için ayrılan süre (saniye)


def gurultu_temizle(ham_ses_dizisi: np.ndarray, frekans_hizi: int) -> np.ndarray:
    """
    5 saniyelik ham ses dizisini alır, sabit arka plan gürültüsünü
    çıkarır ve temizlenmiş diziyi döndürür.

    Args:
        ham_ses_dizisi: 1-boyutlu float32/float64 numpy dizisi (mono).
        frekans_hizi:   Örnekleme hızı (Hz), ör. 22050.

    Returns:
        Gürültüden arındırılmış numpy dizisi (aynı uzunluk, aynı dtype).
    """

    # --- 0. Girdi doğrulama ---
    if ham_ses_dizisi is None or len(ham_ses_dizisi) == 0:
        return ham_ses_dizisi

    # --- 1. Dinamik gürültü profili çıkarımı ---
    #     Sinyalin 0.5 saniyelik dilimlerinin RMS'ini karşılaştır
    #     ve en sessiz dilimi "gürültü" kabul et.
    profil_uzunluk = int(frekans_hizi * _PROFIL_SANIYE)

    # Sinyal profilden kısaysa temizleme anlamsız, dokunmadan döndür
    if len(ham_ses_dizisi) <= profil_uzunluk:
        return ham_ses_dizisi

    gurultu_profili = _en_sessiz_dilimi_bul(ham_ses_dizisi, profil_uzunluk)

    # --- 2. Spektral çıkarma (noisereduce) ---
    temiz = nr.reduce_noise(
        y=ham_ses_dizisi,
        sr=frekans_hizi,
        y_noise=gurultu_profili,
        n_fft=_N_FFT,
        hop_length=_HOP_LENGTH,
        prop_decrease=0.85,       # Gürültü bastırma oranı (%85 — agresif ama doğal)
        stationary=True,          # Sabit frekanslı gürültü varsayımı (fan, uğultu)
    )

    return temiz.astype(ham_ses_dizisi.dtype)


def _en_sessiz_dilimi_bul(sinyal: np.ndarray, dilim_uzunluk: int) -> np.ndarray:
    """
    Sinyal boyunca örtüşmeyen dilimler arasından en düşük RMS'li olanı seçer.
    Bu dilim 'gürültü profili' olarak kullanılır.
    """
    en_dusuk_rms = np.inf
    en_sessiz_bas = 0

    # Örtüşmeyen dilimler üzerinde gez (hız için tam dilim kadar atla)
    for bas in range(0, len(sinyal) - dilim_uzunluk + 1, dilim_uzunluk):
        dilim = sinyal[bas : bas + dilim_uzunluk]
        rms = float(np.sqrt(np.mean(dilim ** 2)))

        if rms < en_dusuk_rms:
            en_dusuk_rms = rms
            en_sessiz_bas = bas

    return sinyal[en_sessiz_bas : en_sessiz_bas + dilim_uzunluk]


# =====================================================================
#  BAĞIMSIZ TEST  (python gurultu_temizleyici.py)
# =====================================================================

if __name__ == "__main__":
    SR = 22050
    SURE = 5.0

    # Yapay test sinyali: 440 Hz saf ton + sürekli fan gürültüsü
    t = np.linspace(0, SURE, int(SR * SURE), endpoint=False, dtype=np.float32)
    saf_ton = 0.5 * np.sin(2 * np.pi * 440 * t)                         # Tavuk sesi yerine
    fan_gurultusu = 0.08 * np.random.normal(size=len(t)).astype(np.float32)  # Kümes fanı
    karisik = saf_ton + fan_gurultusu

    temiz = gurultu_temizle(karisik, SR)

    rms_once = float(np.sqrt(np.mean(karisik ** 2)))
    rms_sonra = float(np.sqrt(np.mean(temiz ** 2)))

    print("=" * 55)
    print("CIK-SOR Gürültü Temizleyici — Bağımsız Test")
    print("=" * 55)
    print(f"  Sinyal süresi   : {SURE}s @ {SR} Hz")
    print(f"  Girdi RMS       : {rms_once:.4f}")
    print(f"  Çıktı RMS       : {rms_sonra:.4f}")
    print(f"  Gürültü düşüşü  : {((rms_once - rms_sonra) / rms_once) * 100:.1f}%")
    print(f"  Cikti boyutu    : {len(temiz)} ornek (degismedi OK)" if len(temiz) == len(karisik) else "  BOYUT UYUMSUZLUGU!")
