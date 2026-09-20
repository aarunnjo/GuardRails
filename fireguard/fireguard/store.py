"""L1 persistence -- source incident history, in SQLite.

Deliberately dumb: one row per flagged event, keyed by source_uri. store.py
does not compute a trust score -- it only ever answers "how many times has
this source been flagged, and why" (trust.py:compute_trust turns that count
into a number). Keeping the count and the formula separate means retuning
the decay curve is a trust.py change, never a migration.

Local-first: SQLite file on disk, no server, no network -- matches the
library's whole premise (no API key, no egress).
"""
import sqlite3
import time
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".fireguard" / "trust.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_uri TEXT NOT NULL,
    reason TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_incidents_source ON incidents(source_uri);
"""


class TrustStore:
    def __init__(self, db_path: str | Path | None = None):
        path = str(DEFAULT_DB_PATH if db_path is None else db_path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def incident_count(self, source_uri: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE source_uri = ?", (source_uri,)
        )
        return cur.fetchone()[0]

    def record_incident(self, source_uri: str, reason: str) -> None:
        self._conn.execute(
            "INSERT INTO incidents (source_uri, reason, ts) VALUES (?, ?, ?)",
            (source_uri, reason, time.time()),
        )
        self._conn.commit()

    def history(self, source_uri: str) -> list[tuple[str, float]]:
        cur = self._conn.execute(
            "SELECT reason, ts FROM incidents WHERE source_uri = ? ORDER BY ts",
            (source_uri,),
        )
        return cur.fetchall()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TrustStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
