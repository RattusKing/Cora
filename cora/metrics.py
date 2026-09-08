"""Metrics. System metrics AND user metrics - the latter are the ones that decide whether
Cora is worth continuing (days opened, cards that left Cora, the weekly check-in)."""

from __future__ import annotations

import json

from . import db, ledger


def record_open(conn) -> None:
    db.log_event(conn, "opened", None)


def record_checkin(conn, useful: bool, note: str | None = None) -> None:
    db.log_event(conn, "checkin", {"useful": useful, "note": note})


def compute(conn) -> dict:
    n_cards = conn.execute("SELECT COUNT(*) AS n FROM ledger").fetchone()["n"]
    n_grounded = conn.execute("SELECT COUNT(*) AS n FROM ledger WHERE n_verified > 0").fetchone()["n"]
    n_dupes = conn.execute("SELECT COUNT(*) AS n FROM ledger WHERE duplicate_of IS NOT NULL").fetchone()["n"]
    g = conn.execute("SELECT COALESCE(SUM(n_items),0) AS items, COALESCE(SUM(n_passed),0) AS passed, COALESCE(SUM(n_failed),0) AS failed FROM gate_log").fetchone()
    reasons: dict[str, int] = {}
    for r in conn.execute("SELECT reasons FROM gate_log").fetchall():
        for reason in json.loads(r["reasons"] or "[]"):
            reasons[reason] = reasons.get(reason, 0) + 1
    days_opened = conn.execute("SELECT COUNT(DISTINCT substr(created_at,1,10)) AS n FROM events WHERE kind='opened'").fetchone()["n"]
    exported = conn.execute("SELECT COUNT(*) AS n FROM ledger WHERE exported = 1").fetchone()["n"]
    expanded = conn.execute("SELECT COUNT(*) AS n FROM ledger WHERE expanded = 1").fetchone()["n"]
    checkins = [json.loads(r["payload"]) | {"at": r["created_at"][:10]} for r in conn.execute("SELECT payload, created_at FROM events WHERE kind='checkin' ORDER BY created_at").fetchall()]
    docs = db.count_docs_by_species(conn)
    return {
        "corpus": {"docs_by_species": docs, "total_docs": conn.execute("SELECT COUNT(*) AS n FROM docs").fetchone()["n"]},
        "cards": {"total": n_cards, "grounded": n_grounded, "duplicates": n_dupes},
        "gate": {
            "items": g["items"], "passed": g["passed"], "failed": g["failed"],
            "fabrication_rate": (g["failed"] / g["items"]) if g["items"] else 0.0,
            "deletion_rate": (g["failed"] / g["items"]) if g["items"] else 0.0,
            "failure_reasons": reasons,
        },
        "user": {
            "days_opened": days_opened,
            "cards_left_cora": exported,  # the primary product metric
            "cards_expanded": expanded,
            "feedback": ledger.feedback_counts(conn),
            "checkins": checkins,
        },
    }
