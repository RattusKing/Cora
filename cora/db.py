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
    flags        TEXT,
    pub_types    TEXT
);
CREATE TABLE IF NOT EXISTS findings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid          TEXT NOT NULL,
    species_key   TEXT NOT NULL,
    mechanism     TEXT NOT NULL,
    hallmarks     TEXT NOT NULL,
    tier          TEXT NOT NULL,
    direction     TEXT,
    evidence_type TEXT NOT NULL,
    quote         TEXT NOT NULL,
    extractor     TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    UNIQUE (pmid, species_key, mechanism, quote, extractor)
);
CREATE INDEX IF NOT EXISTS findings_mechanism ON findings(mechanism);
CREATE INDEX IF NOT EXISTS findings_species ON findings(species_key);
CREATE TABLE IF NOT EXISTS doc_species (
    pmid        TEXT NOT NULL,
    species_key TEXT NOT NULL,
    PRIMARY KEY (pmid, species_key)
);
CREATE TABLE IF NOT EXISTS doc_topic (
    pmid      TEXT NOT NULL,
    topic_key TEXT NOT NULL,
    PRIMARY KEY (pmid, topic_key)
);
CREATE INDEX IF NOT EXISTS doc_topic_topic ON doc_topic(topic_key);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS custom_topics (
    key        TEXT PRIMARY KEY,
    spec_json  TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingest_state (
    topic_key   TEXT NOT NULL,
    species_key TEXT NOT NULL,
    query       TEXT NOT NULL,
    n_pmids     INTEGER NOT NULL,
    stored      INTEGER NOT NULL,
    first_date  TEXT NOT NULL,
    updated     TEXT NOT NULL,
    PRIMARY KEY (topic_key, species_key)
);
CREATE TABLE IF NOT EXISTS ledger (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    topics       TEXT,
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
CREATE TABLE IF NOT EXISTS species_traits (
    hagrid            TEXT PRIMARY KEY,
    class_            TEXT,
    order_            TEXT,
    family            TEXT,
    genus             TEXT,
    species           TEXT,
    binomial          TEXT,
    common_name       TEXT,
    max_longevity_yrs REAL,
    data_quality      TEXT,
    sample_size       TEXT,
    specimen_origin   TEXT,
    adult_weight_g    REAL,
    body_mass_g       REAL,
    metabolic_rate_w  REAL,
    temperature_k     REAL,
    female_maturity_days REAL,
    growth_rate       REAL,
    source            TEXT,
    loaded_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS species_traits_binomial ON species_traits(binomial);
CREATE INDEX IF NOT EXISTS species_traits_genus ON species_traits(genus);
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
    if "pub_types" not in cols:
        conn.execute("ALTER TABLE docs ADD COLUMN pub_types TEXT")
    led = {r[1] for r in conn.execute("PRAGMA table_info(ledger)").fetchall()}
    if led and "topics" not in led:
        conn.execute("ALTER TABLE ledger ADD COLUMN topics TEXT")
    # A database written before topics existed holds exactly the aging corpus: label it,
    # so its abstracts, cards and findings stay reachable once topics filter everything.
    if conn.execute("SELECT COUNT(*) AS n FROM docs").fetchone()[0] and not conn.execute(
        "SELECT COUNT(*) AS n FROM doc_topic"
    ).fetchone()[0]:
        conn.execute("INSERT OR IGNORE INTO doc_topic (pmid, topic_key) SELECT pmid, 'aging' FROM docs")
        conn.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES ('enabled_topics', '[\"aging\"]')"
        )
        # ... and record those (aging, species) pairs as ingested, so `cora setup` does not
        # re-download a corpus that is already here.
        conn.execute(
            """INSERT OR IGNORE INTO ingest_state (topic_key, species_key, query, n_pmids, stored, first_date, updated)
               SELECT 'aging', species_key, '(migrated from a pre-topics database)', COUNT(*), COUNT(*), ?, ?
               FROM doc_species GROUP BY species_key""",
            (today(), today()),
        )
    _migrate_card_topics(conn)


def _migrate_card_topics(conn: sqlite3.Connection) -> None:
    """Label cards written before topics existed as `aging` - the only topic there was.

    Without this they carry no topic and show under every topic's ledger, because an
    unlabelled card is shown rather than hidden.
    """
    try:
        rows = conn.execute("SELECT id, card_json FROM ledger WHERE topics IS NULL").fetchall()
    except sqlite3.OperationalError:
        return
    for r in rows:
        try:
            card = json.loads(r["card_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not card.get("topics"):
            card["topics"] = ["aging"]
        conn.execute(
            "UPDATE ledger SET topics = ?, card_json = ? WHERE id = ?",
            (json.dumps(card["topics"]), json.dumps(card), r["id"]),
        )


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
    d["topics"] = sorted((d.get("topics") or "").split(",")) if d.get("topics") else []
    d["flags"] = json.loads(d["flags"]) if d.get("flags") else []
    d["pub_types"] = json.loads(d["pub_types"]) if d.get("pub_types") else None
    return d


# Species and topics come from correlated subqueries: two GROUP_CONCAT joins would
# multiply the rows against each other and inflate both lists.
_DOC_SELECT = """SELECT d.*,
       (SELECT GROUP_CONCAT(ds.species_key) FROM doc_species ds WHERE ds.pmid = d.pmid) AS species,
       (SELECT GROUP_CONCAT(dt.topic_key) FROM doc_topic dt WHERE dt.pmid = d.pmid) AS topics
  FROM docs d"""


def upsert_doc(conn: sqlite3.Connection, doc: dict, species_key: str | None, topic_key: str | None = None) -> None:
    conn.execute(
        """INSERT INTO docs (pmid, title, abstract, journal, pub_year, entrez_date, retrieved_at, snapshot_file, flags, pub_types)
           VALUES (:pmid, :title, :abstract, :journal, :pub_year, :entrez_date, :retrieved_at, :snapshot_file, :flags, :pub_types)
           ON CONFLICT(pmid) DO UPDATE SET
             title=excluded.title, abstract=excluded.abstract, journal=excluded.journal,
             pub_year=excluded.pub_year, entrez_date=excluded.entrez_date,
             retrieved_at=excluded.retrieved_at, snapshot_file=excluded.snapshot_file,
             flags=excluded.flags, pub_types=COALESCE(excluded.pub_types, docs.pub_types)""",
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
            "pub_types": json.dumps(doc["pub_types"]) if doc.get("pub_types") is not None else None,
        },
    )
    if species_key:
        conn.execute(
            "INSERT OR IGNORE INTO doc_species (pmid, species_key) VALUES (?, ?)",
            (doc["pmid"], species_key),
        )
    if topic_key:
        conn.execute(
            "INSERT OR IGNORE INTO doc_topic (pmid, topic_key) VALUES (?, ?)",
            (doc["pmid"], topic_key),
        )


def set_doc_flags(conn: sqlite3.Connection, pmid: str, flags: list[str]) -> None:
    conn.execute("UPDATE docs SET flags = ? WHERE pmid = ?", (json.dumps(flags), pmid))
    conn.commit()


def get_docs(
    conn: sqlite3.Connection,
    species_keys: list[str] | None = None,
    topic_keys: list[str] | None = None,
) -> list[dict]:
    """Abstracts, optionally restricted to species and/or topics (an AND of both filters)."""
    where: list[str] = []
    args: list = []
    if species_keys:
        where.append(f"d.pmid IN (SELECT pmid FROM doc_species WHERE species_key IN ({','.join('?' * len(species_keys))}))")
        args += list(species_keys)
    if topic_keys:
        where.append(f"d.pmid IN (SELECT pmid FROM doc_topic WHERE topic_key IN ({','.join('?' * len(topic_keys))}))")
        args += list(topic_keys)
    q = _DOC_SELECT + ((" WHERE " + " AND ".join(where)) if where else "")
    return [_row_to_doc(r) for r in conn.execute(q, args).fetchall()]


def docs_by_pmid(conn: sqlite3.Connection, pmids: list[str]) -> dict[str, dict]:
    if not pmids:
        return {}
    marks = ",".join("?" * len(pmids))
    rows = conn.execute(f"{_DOC_SELECT} WHERE d.pmid IN ({marks})", list(pmids)).fetchall()
    return {r["pmid"]: _row_to_doc(r) for r in rows}


def count_docs(conn: sqlite3.Connection, topic_keys: list[str] | None = None) -> int:
    """Unique abstracts. Summing count_docs_by_species double-counts papers about two species."""
    if topic_keys:
        marks = ",".join("?" * len(topic_keys))
        return conn.execute(
            f"SELECT COUNT(DISTINCT pmid) AS n FROM doc_topic WHERE topic_key IN ({marks})", list(topic_keys)
        ).fetchone()["n"]
    return conn.execute("SELECT COUNT(*) AS n FROM docs").fetchone()["n"]


def count_docs_by_species(conn: sqlite3.Connection, topic_keys: list[str] | None = None) -> dict[str, int]:
    if topic_keys:
        marks = ",".join("?" * len(topic_keys))
        rows = conn.execute(
            f"""SELECT ds.species_key, COUNT(DISTINCT ds.pmid) AS n FROM doc_species ds
                WHERE ds.pmid IN (SELECT pmid FROM doc_topic WHERE topic_key IN ({marks}))
                GROUP BY ds.species_key""",
            list(topic_keys),
        ).fetchall()
    else:
        rows = conn.execute("SELECT species_key, COUNT(*) AS n FROM doc_species GROUP BY species_key").fetchall()
    return {r["species_key"]: r["n"] for r in rows}


def count_docs_by_topic(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT topic_key, COUNT(*) AS n FROM doc_topic GROUP BY topic_key").fetchall()
    return {r["topic_key"]: r["n"] for r in rows}


# --- topics: enabled set, custom topics, per-pair ingest state --------------------

def get_setting(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


def enabled_topics(conn: sqlite3.Connection) -> list[str]:
    raw = get_setting(conn, "enabled_topics")
    if not raw:
        return []
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(x) for x in val] if isinstance(val, list) else []


def set_enabled_topics(conn: sqlite3.Connection, keys: list[str]) -> None:
    set_setting(conn, "enabled_topics", json.dumps(list(keys)))


def custom_topic_rows(conn: sqlite3.Connection) -> list[str]:
    try:
        rows = conn.execute("SELECT spec_json FROM custom_topics ORDER BY created_at, key").fetchall()
    except sqlite3.OperationalError:  # database from before custom topics existed
        return []
    return [r["spec_json"] for r in rows]


def save_custom_topic(conn: sqlite3.Connection, key: str, spec_json: str) -> None:
    conn.execute(
        """INSERT INTO custom_topics (key, spec_json, created_at) VALUES (?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET spec_json = excluded.spec_json""",
        (key, spec_json, now_iso()),
    )
    conn.commit()


def delete_custom_topic(conn: sqlite3.Connection, key: str) -> bool:
    cur = conn.execute("DELETE FROM custom_topics WHERE key = ?", (key,))
    conn.commit()
    return cur.rowcount > 0


def record_ingest(conn: sqlite3.Connection, topic_key: str, species_key: str, query: str, n_pmids: int, stored: int) -> None:
    conn.execute(
        """INSERT INTO ingest_state (topic_key, species_key, query, n_pmids, stored, first_date, updated)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(topic_key, species_key) DO UPDATE SET
             query = excluded.query, n_pmids = excluded.n_pmids,
             stored = ingest_state.stored + excluded.stored, updated = excluded.updated""",
        (topic_key, species_key, query, n_pmids, stored, today(), today()),
    )
    conn.commit()


def ingested_pairs(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    try:
        rows = conn.execute("SELECT topic_key, species_key FROM ingest_state").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {(r["topic_key"], r["species_key"]) for r in rows}


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
