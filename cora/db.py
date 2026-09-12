"""SQLite storage: corpus docs, ledger, feedback, events, gate log.

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
    license_tag  TEXT NOT NULL DEFAULT 'pubmed-abstract'
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
"""


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).date().isoformat()


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open (and initialise) the database. ':memory:' is allowed for tests."""
    target = str(path) if path is not None else str(config.DB_PATH)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# --- docs -----------------------------------------------------------------

def upsert_doc(conn: sqlite3.Connection, doc: dict, species_key: str) -> None:
    conn.execute(
        """INSERT INTO docs (pmid, title, abstract, journal, pub_year, entrez_date, retrieved_at, snapshot_file)
           VALUES (:pmid, :title, :abstract, :journal, :pub_year, :entrez_date, :retrieved_at, :snapshot_file)
           ON CONFLICT(pmid) DO UPDATE SET
             title=excluded.title, abstract=excluded.abstract, journal=excluded.journal,
             pub_year=excluded.pub_year, entrez_date=excluded.entrez_date,
             retrieved_at=excluded.retrieved_at, snapshot_file=excluded.snapshot_file""",
        {
            "pmid": doc["pmid"],
            "title": doc.get("title") or "",
            "abstract": doc.get("abstract") or "",
            "journal": doc.get("journal"),
            "pub_year": doc.get("pub_year"),
            "entrez_date": doc.get("entrez_date"),
            "retrieved_at": doc.get("retrieved_at") or now_iso(),
            "snapshot_file": doc.get("snapshot_file"),
        },
    )
    conn.execute(
        "INSERT OR IGNORE INTO doc_species (pmid, species_key) VALUES (?, ?)",
        (doc["pmid"], species_key),
    )


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
    out = []
    for r in rows:
        d = dict(r)
        d["species"] = sorted((d.get("species") or "").split(",")) if d.get("species") else []
        out.append(d)
    return out


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
    out = {}
    for r in rows:
        d = dict(r)
        d["species"] = sorted((d.get("species") or "").split(",")) if d.get("species") else []
        out[d["pmid"]] = d
    return out


def count_docs_by_species(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT species_key, COUNT(*) AS n FROM doc_species GROUP BY species_key"
    ).fetchall()
    return {r["species_key"]: r["n"] for r in rows}


# --- events ---------------------------------------------------------------

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
