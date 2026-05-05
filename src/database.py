"""
CİK-SÖR V1 Demo — Veritabanı / Loglama Modülü
===============================================
Tüm tahminleri SQLite'a kaydeder. Uyarı logları filtrelenebilir.
"""

import sqlite3
import os
import logging
from datetime import datetime

import config

logger = logging.getLogger("ciksor.database")


class DatabaseManager:
    """Thread-güvenli SQLite loglama yöneticisi."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source_type TEXT,
                    file_name TEXT,
                    predicted_class TEXT NOT NULL,
                    confidence REAL,
                    alert INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def log_prediction(
        self,
        predicted_class: str,
        confidence: float,
        source_type: str = "dosya",
        file_name: str = None,
        alert: bool = False,
        notes: str = None,
    ) -> int:
        """
        Tahmin sonucunu loglar. Tüm sınıflar kaydedilir.

        Returns:
            Eklenen kaydın ID'si.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO predictions "
                "(timestamp, source_type, file_name, predicted_class, confidence, alert, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, source_type, file_name, predicted_class, round(confidence, 2),
                 1 if alert else 0, notes),
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.info(f"Log kaydedildi #{row_id}: {predicted_class} (%{confidence*100:.1f})")
            return row_id
        finally:
            conn.close()

    def get_recent_logs(self, limit: int = 20):
        """Son N kaydı döndürür."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT id, timestamp, source_type, file_name, "
                "predicted_class, confidence, alert, notes "
                "FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        return [
            {
                "id": r[0], "timestamp": r[1], "source_type": r[2],
                "file_name": r[3], "predicted_class": r[4],
                "confidence": r[5], "alert": bool(r[6]), "notes": r[7],
            }
            for r in rows
        ]

    def get_alert_logs(self, limit: int = 20):
        """Sadece uyarı loglarını döndürür."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT id, timestamp, source_type, file_name, "
                "predicted_class, confidence, alert, notes "
                "FROM predictions WHERE alert = 1 ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        return [
            {
                "id": r[0], "timestamp": r[1], "source_type": r[2],
                "file_name": r[3], "predicted_class": r[4],
                "confidence": r[5], "alert": bool(r[6]), "notes": r[7],
            }
            for r in rows
        ]

    def get_log_count(self):
        """Toplam ve uyarı log sayılarını döndürür."""
        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            alerts = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE alert = 1"
            ).fetchone()[0]
        finally:
            conn.close()
        return {"total": total, "alerts": alerts}
