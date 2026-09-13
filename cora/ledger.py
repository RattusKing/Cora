"""The persistent hypothesis ledger. Ranks decay; nothing is ever deleted."""

from __future__ import annotations

import json

from . import db
from .card import Card

VERDICTS = ("accept", "reject:wrong", "reject:uninteresting", "dig_deeper")


def insert(conn, card: Card) -> Card:
    dup = conn.execute(
        "SELECT id FROM ledger WHERE dedupe_key = ? AND duplicate_of IS NULL ORDER BY created_at LIMIT 1",
        (card.dedupe_key,),
    ).fetchone()
    duplicate_of = dup["id"] if dup else None
    conn.execute(
        """INSERT INTO ledger (id, created_at, query, card_json, dedupe_key, duplicate_of, state, groundedness, n_verified, n_species)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            card.id, card.created_at, card.query, card.model_dump_json(), card.dedupe_key, duplicate_of,
            card.status, card.gate.groundedness, card.evidence_band.n_verified_quotes, card.evidence_band.n_species,
        ),
    )
    conn.commit()
    return card


def _row_to_card(row) -> Card:
    return Card.model_validate_json(row["card_json"])


def get(conn, card_id: str) -> Card | None:
    row = conn.execute("SELECT * FROM ledger WHERE id = ?", (card_id,)).fetchone()
    return _row_to_card(row) if row else None


def get_row(conn, card_id: str):
    return conn.execute("SELECT * FROM ledger WHERE id = ?", (card_id,)).fetchone()


def list_cards(conn, include_archived: bool = False, limit: int = 50) -> list[dict]:
    """Ranked: verified species, then groundedness, then recency. Archived entries, cards
    whose pattern failed its check, and duplicates are demoted, never removed. Each row
    carries `new_evidence`: unseen touches from the re-check."""
    rows = conn.execute(
        """SELECT * FROM ledger
           ORDER BY (state = 'archived') ASC,
                    (state IN ('overclaim', 'unsupported', 'ungrounded')) ASC,
                    (duplicate_of IS NOT NULL) ASC,
                    n_species DESC, groundedness DESC, n_verified DESC, created_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    unseen = db.unseen_update_counts(conn)
    out = []
    for r in rows:
        if r["state"] == "archived" and not include_archived:
            continue
        out.append(
            {
                "card": _row_to_card(r), "state": r["state"], "duplicate_of": r["duplicate_of"],
                "expanded": r["expanded"], "exported": r["exported"], "new_evidence": unseen.get(r["id"], 0),
            }
        )
    return out


def feedback(conn, card_id: str, verdict: str, reason: str | None = None) -> None:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    if get_row(conn, card_id) is None:
        raise KeyError(card_id)
    conn.execute(
        "INSERT INTO feedback (card_id, verdict, reason, created_at) VALUES (?, ?, ?, ?)",
        (card_id, verdict, reason, db.now_iso()),
    )
    new_state = {"accept": "accepted", "reject:wrong": "archived", "reject:uninteresting": "archived", "dig_deeper": "dig_deeper"}[verdict]
    conn.execute("UPDATE ledger SET state = ? WHERE id = ?", (new_state, card_id))
    conn.commit()


def mark_expanded(conn, card_id: str) -> None:
    conn.execute("UPDATE ledger SET expanded = 1 WHERE id = ?", (card_id,))
    db.mark_updates_seen(conn, card_id)
    db.log_event(conn, "expanded", {"card_id": card_id})


def mark_exported(conn, card_id: str) -> None:
    conn.execute("UPDATE ledger SET exported = 1 WHERE id = ?", (card_id,))
    db.log_event(conn, "exported", {"card_id": card_id})


def feedback_counts(conn) -> dict[str, int]:
    rows = conn.execute("SELECT verdict, COUNT(*) AS n FROM feedback GROUP BY verdict").fetchall()
    return {r["verdict"]: r["n"] for r in rows}


def dump(conn) -> list[dict]:
    return [json.loads(r["card_json"]) for r in conn.execute("SELECT card_json FROM ledger").fetchall()]
