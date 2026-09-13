"""SQLite storage: corpus docs, ledger, feedback, events, gate/pattern logs, re-check log.

One file, no server. The ledger *is* the memory at this scale.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    pmid         TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    abstract     TEXT NOT NULL,
    journal      TEXT,
    pub_year     INTEGER,
    entrez_date  TEXT,
    retrieved_at TEXT NOT NULL,
    snapshot_file TEXT,
    license_tag  TEXT NOT NULL DEFAULT 'pubmed-abstract',
    flags        TEXT
);
CREATE TABLE IF NOT EXISTS doc_species (
    pmid        TEXT NOT NULL,
    species_key TEXT NOT NULL,
    PRIMARY KEY (pmid, species_key)
);
CREATE TABLE IF NOT EXISTS ledger (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    query        TEXT NOT NULL,
    card_json    TEXT NOT NULL,
    dedupe_key   TEXT NOT NULL,
    duplicate_of TEXT,
    state        TEXT NOT NULL,
    groundedness REAL NOT NULL,
    n_verified   INTEGER NOT NULL,
    n_species    INTEGER NOT NULL,
    expanded     INTEGER NOT NULL DEFAULT 0,
    exported     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ledger_dedupe ON ledger(dedupe_key);
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id    TEXT NOT NULL,
    verdict    TEXT NOT NULL,
    reason     TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gate_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id    TEXT,
    n_items    INTEGER NOT NULL,
    n_passed   INTEGER NOT NULL,
    n_failed   INTEGER NOT NULL,
    reasons    TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pattern_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id      TEXT,
    judged       INTEGER NOT NULL,
    judge_model  TEXT,
    supported    TEXT,
    strength     TEXT,
    passed       INTEGER NOT NULL,
    reason       TEXT,
    retried      INTEGER NOT NULL,
    disagreement INTEGER NOT NULL,
    mechanical_flags TEXT,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recheck_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    n_new           INTEGER NOT NULL,
    n_gone          INTEGER NOT NULL,
    n_cards_touched INTEGER NOT NULL,
    n_flags         INTEGER NOT NULL,
    report_json     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS card_updates (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id    TEXT NOT NULL,
    pmid       TEXT NOT NULL,
    reason     TEXT NOT NULL,
    seen       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE (card_id, pmid, reason)
);
"""


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).date().isoformat()


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(docs)").fetchall()}
    if "flags" not in cols:
        conn.execute("ALTER TABLE docs ADD COLUMN flags TEXT")


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open (and initialise) the database. ':memory:' is allowed for tests."""
    target = str(path) if path is not None else str(config.DB_PATH)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


# --- docs -----------------------------------------------------------------

def _row_to_doc(r) -> dict:
    d = dict(r)
    d["species"] = sorted((d.get("species") or "").split(",")) if d.get("species") else []
    d["flags"] = json.loads(d["flags"]) if d.get("flags") else []
    return d


def upsert_doc(conn: sqlite3.Connection, doc: dict, species_key: str | None) -> None:
    conn.execute(
        """INSERT INTO docs (pmid, title, abstract, journal, pub_year, entrez_date, retrieved_at, snapshot_file, flags)
           VALUES (:pmid, :title, :abstract, :journal, :pub_year, :entrez_date, :retrieved_at, :snapshot_file, :flags)
           ON CONFLICT(pmid) DO UPDATE SET
             title=excluded.title, abstract=excluded.abstract, journal=excluded.journal,
             pub_year=excluded.pub_year, entrez_date=excluded.entrez_date,
             retrieved_at=excluded.retrieved_at, snapshot_file=excluded.snapshot_file,
             flags=excluded.flags""",
        {
            "pmid": doc["pmid"],
            "title": doc.get("title") or "",
            "abstract": doc.get("abstract") or "",
            "journal": doc.get("journal"),
            "pub_year": doc.get("pub_year"),
            "entrez_date": doc.get("entrez_date"),
            "retrieved_at": doc.get("retrieved_at") or now_iso(),
            "snapshot_file": doc.get("snapshot_file"),
            "flags": json.dumps(doc.get("flags") or []),
        },
    )
    if species_key:
        conn.execute(
            "INSERT OR IGNORE INTO doc_species (pmid, species_key) VALUES (?, ?)",
            (doc["pmid"], species_key),
        )


def set_doc_flags(conn: sqlite3.Connection, pmid: str, flags: list[str]) -> None:
    conn.execute("UPDATE docs SET flags = ? WHERE pmid = ?", (json.dumps(flags), pmid))
    conn.commit()


def get_docs(conn: sqlite3.Connection, species_keys: list[str] | None = None) -> list[dict]:
    if species_keys:
        marks = ",".join("?" * len(species_keys))
        rows = conn.execute(
            f"""SELECT d.*, GROUP_CONCAT(ds.species_key) AS species
                FROM docs d JOIN doc_species ds ON ds.pmid = d.pmid
                WHERE d.pmid IN (SELECT pmid FROM doc_species WHERE species_key IN ({marks}))
                GROUP BY d.pmid""",
            species_keys,
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT d.*, GROUP_CONCAT(ds.species_key) AS species
               FROM docs d LEFT JOIN doc_species ds ON ds.pmid = d.pmid
               GROUP BY d.pmid"""
        ).fetchall()
    return [_row_to_doc(r) for r in rows]


def docs_by_pmid(conn: sqlite3.Connection, pmids: list[str]) -> dict[str, dict]:
    if not pmids:
        return {}
    marks = ",".join("?" * len(pmids))
    rows = conn.execute(
        f"""SELECT d.*, GROUP_CONCAT(ds.species_key) AS species
            FROM docs d LEFT JOIN doc_species ds ON ds.pmid = d.pmid
            WHERE d.pmid IN ({marks}) GROUP BY d.pmid""",
        list(pmids),
    ).fetchall()
    return {r["pmid"]: _row_to_doc(r) for r in rows}


def count_docs_by_species(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT species_key, COUNT(*) AS n FROM doc_species GROUP BY species_key"
    ).fetchall()
    return {r["species_key"]: r["n"] for r in rows}


# --- events & logs -----------------------------------------------------------

def log_event(conn: sqlite3.Connection, kind: str, payload: dict | None = None) -> None:
    conn.execute(
        "INSERT INTO events (kind, payload, created_at) VALUES (?, ?, ?)",
        (kind, json.dumps(payload) if payload is not None else None, now_iso()),
    )
    conn.commit()


def log_pattern(conn: sqlite3.Connection, card_id: str | None, pc) -> None:
    conn.execute(
        """INSERT INTO pattern_log (card_id, judged, judge_model, supported, strength, passed, reason, retried, disagreement, mechanical_flags, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            card_id, int(pc.judged), pc.judge_model, pc.supported, pc.strength, int(pc.passed), pc.reason,
            int(pc.retried), int(pc.disagreement), json.dumps(pc.mechanical_strong_words), now_iso(),
        ),
    )
    conn.commit()


def log_gate(conn: sqlite3.Connection, card_id: str | None, n_items: int, n_passed: int, reasons: list[str]) -> None:
    conn.execute(
        "INSERT INTO gate_log (card_id, n_items, n_passed, n_failed, reasons, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (card_id, n_items, n_passed, n_items - n_passed, json.dumps(reasons), now_iso()),
    )
    conn.commit()


def log_recheck(conn: sqlite3.Connection, n_new: int, n_gone: int, n_cards_touched: int, n_flags: int, report_json: str) -> None:
    conn.execute(
        "INSERT INTO recheck_log (created_at, n_new, n_gone, n_cards_touched, n_flags, report_json) VALUES (?, ?, ?, ?, ?, ?)",
        (now_iso(), n_new, n_gone, n_cards_touched, n_flags, report_json),
    )
    conn.commit()


# --- card updates (what changed for a card) --------------------------------------

def add_card_update(conn: sqlite3.Connection, card_id: str, pmid: str, reason: str) -> bool:
    """Record a touch; returns False if this exact touch was already recorded."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO card_updates (card_id, pmid, reason, created_at) VALUES (?, ?, ?, ?)",
        (card_id, pmid, reason, now_iso()),
    )
    conn.commit()
    return cur.rowcount == 1


def has_card_update(conn: sqlite3.Connection, card_id: str, pmid: str, reason: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM card_updates WHERE card_id = ? AND pmid = ? AND reason = ? LIMIT 1", (card_id, pmid, reason)
    ).fetchone()
    return row is not None


def card_updates(conn: sqlite3.Connection, card_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT pmid, reason, seen, created_at FROM card_updates WHERE card_id = ? ORDER BY id", (card_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def unseen_update_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT card_id, COUNT(*) AS n FROM card_updates WHERE seen = 0 GROUP BY card_id").fetchall()
    return {r["card_id"]: r["n"] for r in rows}


def mark_updates_seen(conn: sqlite3.Connection, card_id: str) -> None:
    conn.execute("UPDATE card_updates SET seen = 1 WHERE card_id = ?", (card_id,))
    conn.commit()
