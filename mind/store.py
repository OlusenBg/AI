"""The Self store: the single source of truth for who the AI is.

Only the subconscious writes here. The conscious reads snapshots."""
import json
import re
import sqlite3
import time
from dataclasses import dataclass

from .emotions import DEFAULT_BASELINES, EMOTIONS

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (emotion TEXT PRIMARY KEY, value REAL NOT NULL, baseline REAL NOT NULL);
CREATE TABLE IF NOT EXISTS traits (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, importance REAL NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('short','long')), emotions TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS thoughts (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, value TEXT NOT NULL,
    verdict TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


@dataclass
class Memory:
    id: int
    text: str
    importance: float
    kind: str
    created_at: float


class SelfStore:
    def __init__(self, path: str = ":memory:", now=time.time):
        self.now = now
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        for e in EMOTIONS:
            self.db.execute(
                "INSERT OR IGNORE INTO state (emotion, value, baseline) VALUES (?,?,?)",
                (e, DEFAULT_BASELINES[e], DEFAULT_BASELINES[e]),
            )
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('last_tick', ?)", (str(self.now()),))
        self.db.commit()

    # --- emotions ---
    def get_state(self) -> dict[str, float]:
        return {r["emotion"]: r["value"] for r in self.db.execute("SELECT * FROM state")}

    def get_baselines(self) -> dict[str, float]:
        return {r["emotion"]: r["baseline"] for r in self.db.execute("SELECT * FROM state")}

    def set_state(self, state: dict[str, float]) -> None:
        for k, v in state.items():
            self.db.execute("UPDATE state SET value=? WHERE emotion=?", (v, k))
        self.db.commit()

    def set_baseline(self, emotion: str, value: float) -> None:
        self.db.execute("UPDATE state SET baseline=? WHERE emotion=?", (max(0.0, min(1.0, value)), emotion))
        self.db.commit()

    def get_last_tick(self) -> float:
        return float(self.db.execute("SELECT value FROM meta WHERE key='last_tick'").fetchone()[0])

    def set_last_tick(self, t: float) -> None:
        self.db.execute("UPDATE meta SET value=? WHERE key='last_tick'", (str(t),))
        self.db.commit()

    # --- traits (long-term identity) ---
    def get_trait(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM traits WHERE key=?", (normalize_key(key),)).fetchone()
        return row["value"] if row else None

    def get_traits(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.db.execute("SELECT * FROM traits ORDER BY created_at")}

    def add_trait(self, key: str, value: str, source: str = "", confidence: float = 1.0) -> bool:
        """Commit a trait. Existing traits are never overwritten (identity is stable);
        returns False if the key was already decided."""
        key = normalize_key(key)
        if not key or not value.strip():
            return False
        cur = self.db.execute(
            "INSERT OR IGNORE INTO traits VALUES (?,?,?,?,?)", (key, value.strip(), confidence, source, self.now())
        )
        self.db.commit()
        return cur.rowcount > 0

    def revise_trait(self, key: str, value: str, source: str = "drift") -> None:
        """The only way an existing trait changes. Reserved for slow consolidation/drift."""
        self.db.execute(
            "UPDATE traits SET value=?, source=? WHERE key=?", (value.strip(), source, normalize_key(key))
        )
        self.db.commit()

    # --- memories ---
    def add_memory(self, text: str, importance: float, kind: str = "short", emotions: dict | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO memories (text, importance, kind, emotions, created_at) VALUES (?,?,?,?,?)",
            (text, max(0.0, min(1.0, importance)), kind, json.dumps(emotions or {}), self.now()),
        )
        self.db.commit()
        return cur.lastrowid

    def get_memories(self, kind: str | None = None, limit: int = 50) -> list[Memory]:
        q = "SELECT * FROM memories" + (" WHERE kind=?" if kind else "") + " ORDER BY id DESC LIMIT ?"
        args = (kind, limit) if kind else (limit,)
        return [Memory(r["id"], r["text"], r["importance"], r["kind"], r["created_at"]) for r in self.db.execute(q, args)]

    def promote_memory(self, memory_id: int) -> None:
        self.db.execute("UPDATE memories SET kind='long' WHERE id=?", (memory_id,))
        self.db.commit()

    def delete_memory(self, memory_id: int) -> None:
        self.db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.db.commit()

    # --- private thoughts and claims ---
    def log_thought(self, text: str) -> None:
        self.db.execute("INSERT INTO thoughts (text, created_at) VALUES (?,?)", (text, self.now()))
        self.db.commit()

    def get_thoughts(self, limit: int = 20) -> list[str]:
        rows = self.db.execute("SELECT text FROM thoughts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [r["text"] for r in reversed(rows)]

    def log_claim(self, key: str, value: str, verdict: str) -> None:
        self.db.execute(
            "INSERT INTO claims (key, value, verdict, created_at) VALUES (?,?,?,?)",
            (normalize_key(key), value, verdict, self.now()),
        )
        self.db.commit()

    def get_claims(self) -> list[tuple[str, str, str]]:
        return [(r["key"], r["value"], r["verdict"]) for r in self.db.execute("SELECT * FROM claims ORDER BY id")]
