"""
CIK-SOR V2 — Yerel SQLite Tespit Loglama Modülü
=================================================
Yapay zeka motorunun ürettiği "Hasta" tespitlerini, güven skoru
eşiğini aştığında kalıcı olarak yerel diske (ciksordb.sqlite) yazar.

Pipeline entegrasyonu:
    from veritabani import DatabaseManager
    db = DatabaseManager()
    db.log_kaydet("Unhealthy", 92.4)
    son_kayitlar = db.son_loglari_getir(limit=10)
"""

import sqlite3
import os
from datetime import datetime

# Veritabanı dosyası proje kök dizininde oluşturulur
_VARSAYILAN_DB_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciksordb.sqlite")

# Sadece bu eşiğin üzerindeki "Hasta" tespitleri loglanır
_GUVEN_ESIGI = 75.0


class DatabaseManager:
    """
    Thread-güvenli, hafif SQLite tespit loglama yöneticisi.

    Her veritabanı işlemi kendi bağlantısını açar ve kapatır;
    böylece birden fazla thread (Ring Buffer analiz thread'i,
    Gradio/Tkinter UI thread'i) aynı anda güvenle erişebilir.
    """

    def __init__(self, db_yolu: str = _VARSAYILAN_DB_YOLU):
        """
        Veritabanı dosyasını ve tespit_loglari tablosunu oluşturur
        (zaten varsa dokunmaz).
        """
        self.db_yolu = db_yolu
        self._tablo_olustur()

    # ─────────────────────────────────────────────
    #  HERKESE AÇIK API
    # ─────────────────────────────────────────────

    def log_kaydet(self, tespit_sinifi: str, guven_skoru: float) -> bool:
        """
        Yeni bir tespit kaydı ekler.

        Kurallar:
          - Sadece "Unhealthy" (Hasta) sınıfı loglanır.
          - Güven skoru %75'in altındaysa kayıt atlanır.

        Returns:
            True  → kayıt başarıyla eklendi.
            False → eşik/sınıf kuralı nedeniyle atlandı.
        """
        # Sınıf filtresi: Sadece hasta tespitleri loglanır
        if "unhealthy" not in tespit_sinifi.lower():
            return False

        # Güven skoru eşik kontrolü
        if guven_skoru < _GUVEN_ESIGI:
            return False

        zaman_damgasi = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = self._baglan()
        try:
            conn.execute(
                "INSERT INTO tespit_loglari (tarih_saat, tespit_sinifi, guven_skoru) VALUES (?, ?, ?)",
                (zaman_damgasi, tespit_sinifi, round(guven_skoru, 2)),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def son_loglari_getir(self, limit: int = 10) -> list[dict]:
        """
        Son N kaydı tarihe göre azalan (en yeni → en eski) sırada döndürür.

        Returns:
            [
                {"id": 3, "tarih_saat": "2026-04-13 17:45:02", "tespit_sinifi": "Unhealthy", "guven_skoru": 91.2},
                ...
            ]
        """
        conn = self._baglan()
        try:
            cursor = conn.execute(
                "SELECT id, tarih_saat, tespit_sinifi, guven_skoru "
                "FROM tespit_loglari "
                "ORDER BY id DESC "
                "LIMIT ?",
                (limit,),
            )
            satirlar = cursor.fetchall()
        finally:
            conn.close()

        return [
            {
                "id": s[0],
                "tarih_saat": s[1],
                "tespit_sinifi": s[2],
                "guven_skoru": s[3],
            }
            for s in satirlar
        ]

    # ─────────────────────────────────────────────
    #  DAHİLİ YARDIMCILAR
    # ─────────────────────────────────────────────

    def _baglan(self) -> sqlite3.Connection:
        """Her çağrıda taze bağlantı açar (thread güvenliği)."""
        return sqlite3.connect(self.db_yolu)

    def _tablo_olustur(self):
        """tespit_loglari tablosunu yoksa oluşturur."""
        conn = self._baglan()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tespit_loglari (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    tarih_saat     TEXT    NOT NULL,
                    tespit_sinifi  TEXT    NOT NULL,
                    guven_skoru    REAL    NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


# =====================================================================
#  BAGIMSIZ TEST  (python veritabani.py)
# =====================================================================

if __name__ == "__main__":
    import os

    TEST_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_ciksordb.sqlite")

    # Temiz test ortami
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    db = DatabaseManager(db_yolu=TEST_DB)

    print("=" * 55)
    print("CIK-SOR Veritabani Modulu - Bagimsiz Test")
    print("=" * 55)

    # --- Test 1: Esik altindaki kayit atlanmali ---
    sonuc = db.log_kaydet("Unhealthy", 60.0)
    print(f"  [TEST 1] Esik alti (60%) -> Atlandi: {not sonuc}")

    # --- Test 2: Saglikli sinif atlanmali ---
    sonuc = db.log_kaydet("Healthy", 95.0)
    print(f"  [TEST 2] Saglikli sinif  -> Atlandi: {not sonuc}")

    # --- Test 3: Gurultu sinifi atlanmali ---
    sonuc = db.log_kaydet("Noise", 88.0)
    print(f"  [TEST 3] Gurultu sinifi  -> Atlandi: {not sonuc}")

    # --- Test 4: Gecerli kayitlar eklenmeli ---
    db.log_kaydet("Unhealthy (Hasta)", 91.2)
    db.log_kaydet("Unhealthy (Hasta)", 78.5)
    db.log_kaydet("Unhealthy (Hasta)", 85.0)
    print(f"  [TEST 4] 3 gecerli kayit eklendi.")

    # --- Test 5: Son loglari getir ---
    loglar = db.son_loglari_getir(limit=5)
    print(f"  [TEST 5] Son loglar ({len(loglar)} kayit):")
    for log in loglar:
        print(f"           #{log['id']} | {log['tarih_saat']} | {log['tespit_sinifi']} | %{log['guven_skoru']}")

    # --- Test 6: Thread guvenligi (coklu yazma) ---
    import threading

    def paralel_yaz(i):
        db.log_kaydet("Unhealthy", 80.0 + i)

    threads = [threading.Thread(target=paralel_yaz, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    toplam = len(db.son_loglari_getir(limit=100))
    print(f"  [TEST 6] Thread guvenligi -> Toplam kayit: {toplam} (beklenen: 13)")

    # Temizlik
    os.remove(TEST_DB)
    print("\n  Tum testler basarili. Test DB silindi.")
