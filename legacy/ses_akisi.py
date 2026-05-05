"""
CIK-SOR V2 — Kesintisiz Otonom Dinleme Modülü (Ring Buffer / Kayan Pencere)
=============================================================================
Mikrofonu sürekli dinler, RAM'de sabit 5 saniyelik dairesel tampon tutar,
her 1 saniyede bir son 5 saniyelik pencereyi analiz fonksiyonuna iletir.
Hiçbir noktada diske yazma yapılmaz; UI thread'i bloklanmaz.
"""

import threading
import time
import numpy as np
import sounddevice as sd
import librosa


class AudioStreamer:
    """
    Ring Buffer tabanlı kesintisiz ses akışı yöneticisi.

    Kullanım:
        streamer = AudioStreamer(sr=22050, tampon_saniye=5, analiz_araligi=1.0)
        streamer.baslat()   # Otonom dinlemeyi başlatır
        streamer.durdur()   # Durdurur ve kaynakları serbest bırakır
    """

    def __init__(
        self,
        sr: int = 22050,
        tampon_saniye: float = 5.0,
        analiz_araligi: float = 1.0,
        sessizlik_esigi: float = 0.01,
        analiz_callback=None,
    ):
        """
        Args:
            sr:               Örnekleme hızı (Hz). 22050 veya 44100 önerilir.
            tampon_saniye:     Ring buffer kapasitesi (saniye).
            analiz_araligi:   Kayan pencere tetikleme aralığı (saniye).
            sessizlik_esigi:  RMS eşiği; altındaki tamponlar "sessizlik" sayılır.
            analiz_callback:  Her pencere için çağrılacak fonksiyon.
                              İmza: callback(y: np.ndarray, sr: int) -> str | None
                              Verilmezse varsayılan mock çıktı kullanılır.
        """
        self.sr = sr
        self.tampon_saniye = tampon_saniye
        self.analiz_araligi = analiz_araligi
        self.sessizlik_esigi = sessizlik_esigi

        # --- Ring Buffer (sabit boyutlu numpy dizisi) ---
        self._tampon_boyut = int(self.sr * self.tampon_saniye)
        self._ring_buffer = np.zeros(self._tampon_boyut, dtype=np.float32)
        self._yazma_pozisyonu = 0          # Dairesel yazma imleci
        self._toplam_yazilan = 0           # Tampona şimdiye dek yazılan toplam örnek
        self._buffer_kilidi = threading.Lock()

        # --- Kontrol bayrakları ---
        self._aktif = False                # Sistem açık mı?
        self._stream: sd.InputStream | None = None
        self._analiz_thread: threading.Thread | None = None

        # --- Harici analiz fonksiyonu (opsiyonel) ---
        self._analiz_callback = analiz_callback or self._mock_analiz

        # --- Son analiz sonucunu dışarıya açan alan ---
        self.son_sonuc: str = "Bekleniyor..."

    # =================================================================
    #  HERKESE AÇIK API
    # =================================================================

    def baslat(self):
        """Mikrofonu açar ve arka plan analiz döngüsünü başlatır."""
        if self._aktif:
            return  # Zaten çalışıyor

        self._aktif = True
        self._toplam_yazilan = 0
        self._yazma_pozisyonu = 0
        self._ring_buffer[:] = 0.0

        # 1) sounddevice InputStream başlat (callback modeli — UI bloklamaz)
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            callback=self._ses_callback,
            blocksize=1024,
        )
        self._stream.start()

        # 2) Analiz thread'ini başlat (daemon → ana süreç kapanınca otomatik ölür)
        self._analiz_thread = threading.Thread(
            target=self._analiz_dongusu,
            daemon=True,
            name="CIK-SOR-AnalizThread",
        )
        self._analiz_thread.start()

    def durdur(self):
        """Mikrofonu kapatır ve analiz döngüsünü sonlandırır."""
        self._aktif = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Thread doğal olarak _aktif=False ile çıkacak; yine de join ile bekle
        if self._analiz_thread is not None and self._analiz_thread.is_alive():
            self._analiz_thread.join(timeout=3.0)
            self._analiz_thread = None

        self.son_sonuc = "Durduruldu."

    @property
    def dinliyor_mu(self) -> bool:
        return self._aktif

    # =================================================================
    #  DAHİLİ — SES AKIŞI CALLBACK'İ  (sounddevice tarafından çağrılır)
    # =================================================================

    def _ses_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        sounddevice her blok hazır olduğunda bu fonksiyonu çağırır.
        Gelen veriyi ring buffer'a FIFO mantığıyla yazar.
        """
        if status:
            # Örn. buffer overflow — kritik değil, devam et
            pass

        # Mono kanal → düz 1-boyutlu dizi
        mono = indata[:, 0]
        n = len(mono)

        with self._buffer_kilidi:
            # Dairesel yazma: tamponun sonuna kadar yaz, taşarsa başa sar
            bitis = self._yazma_pozisyonu + n

            if bitis <= self._tampon_boyut:
                self._ring_buffer[self._yazma_pozisyonu:bitis] = mono
            else:
                # İki parçaya böl: sonuna kadar + baştan kalan
                ilk_kisim = self._tampon_boyut - self._yazma_pozisyonu
                self._ring_buffer[self._yazma_pozisyonu:] = mono[:ilk_kisim]
                self._ring_buffer[:n - ilk_kisim] = mono[ilk_kisim:]

            self._yazma_pozisyonu = bitis % self._tampon_boyut
            self._toplam_yazilan += n

    # =================================================================
    #  DAHİLİ — ARKA PLAN ANALİZ DÖNGÜSÜ
    # =================================================================

    def _analiz_dongusu(self):
        """
        Her `analiz_araligi` saniyede bir tetiklenen kayan pencere döngüsü.
        Tampon dolmamışsa bekler; sessizse modeli atlar.
        """
        while self._aktif:
            time.sleep(self.analiz_araligi)

            if not self._aktif:
                break

            # Tampon henüz 5 saniye dolmadıysa analiz yapma
            if self._toplam_yazilan < self._tampon_boyut:
                self.son_sonuc = f"Tampon dolduruluyor... ({self._toplam_yazilan}/{self._tampon_boyut})"
                continue

            # --- Tamponun anlık kopyasını al (thread-safe) ---
            with self._buffer_kilidi:
                # Doğrusal sıralı kopya: en eski → en yeni
                kopya = np.empty(self._tampon_boyut, dtype=np.float32)
                okuma_bas = self._yazma_pozisyonu  # En eski örnek burada
                ilk_kisim = self._tampon_boyut - okuma_bas
                kopya[:ilk_kisim] = self._ring_buffer[okuma_bas:]
                kopya[ilk_kisim:] = self._ring_buffer[:okuma_bas]

            # --- Noise Gate: RMS eşik kontrolü ---
            rms = float(np.sqrt(np.mean(kopya ** 2)))
            if rms < self.sessizlik_esigi:
                self.son_sonuc = f"Sessizlik (RMS: {rms:.4f}) — analiz atlandı."
                continue

            # --- Analiz callback'ini çağır ---
            try:
                sonuc = self._analiz_callback(kopya, self.sr)
                if sonuc is not None:
                    self.son_sonuc = sonuc
            except Exception as e:
                self.son_sonuc = f"Analiz hatası: {e}"

    # =================================================================
    #  VARSAYILAN MOCK ANALİZ (Gerçek model bağlanana kadar)
    # =================================================================

    @staticmethod
    def _mock_analiz(y: np.ndarray, sr: int) -> str:
        """Model entegre edilmeden önceki test fonksiyonu."""
        rms = float(np.sqrt(np.mean(y ** 2)))
        sure = len(y) / sr
        return f"[MOCK] {sure:.1f}s pencere | RMS: {rms:.4f} | Pik: {np.max(np.abs(y)):.4f}"


# =====================================================================
#  BAĞIMSIZ ÇALIŞTIRMA TESTİ  (python ses_akisi.py)
# =====================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("CIK-SOR Ring Buffer — Bağımsız Konsol Testi")
    print("Mikrofon açılacak. Durdurmak için Ctrl+C basın.")
    print("=" * 60)

    streamer = AudioStreamer(sr=22050, tampon_saniye=5.0, analiz_araligi=1.0)
    streamer.baslat()

    try:
        son_gosterilen = ""
        while True:
            time.sleep(0.3)
            if streamer.son_sonuc != son_gosterilen:
                son_gosterilen = streamer.son_sonuc
                print(f"  → {son_gosterilen}")
    except KeyboardInterrupt:
        pass
    finally:
        streamer.durdur()
        print("\nDinleme durduruldu. Çıkış yapılıyor.")
        sys.exit(0)
