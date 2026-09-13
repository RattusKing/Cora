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
    p = conn.execute(
        """SELECT COUNT(*) AS n, COALESCE(SUM(judged),0) AS judged, COALESCE(SUM(passed),0) AS passed,
                  COALESCE(SUM(retried),0) AS retried, COALESCE(SUM(disagreement),0) AS disagreement
           FROM pattern_log"""
    ).fetchone()
    p_reasons: dict[str, int] = {}
    for r in conn.execute("SELECT reason FROM pattern_log WHERE passed = 0 AND reason IS NOT NULL").fetchall():
        p_reasons[r["reason"]] = p_reasons.get(r["reason"], 0) + 1
    pattern_stats = {
        "checked": p["n"], "judged": p["judged"], "passed": p["passed"], "failed": p["n"] - p["passed"],
        "retried": p["retried"],
        "mechanical_vs_judge_disagreement_rate": (p["disagreement"] / p["judged"]) if p["judged"] else 0.0,
        "failure_reasons": p_reasons,
    }
    rc = conn.execute(
        """SELECT COUNT(*) AS runs, MAX(created_at) AS last_run, COALESCE(SUM(n_new),0) AS new_abstracts,
                  COALESCE(SUM(n_cards_touched),0) AS cards_touched, COALESCE(SUM(n_flags),0) AS flags
           FROM recheck_log"""
    ).fetchone()
    unseen = sum(db.unseen_update_counts(conn).values())
    recheck_stats = {
        "runs": rc["runs"], "last_run": rc["last_run"], "new_abstracts": rc["new_abstracts"],
        "cards_touched": rc["cards_touched"], "source_flags": rc["flags"], "unseen_updates": unseen,
    }
    g_rows = conn.execute("SELECT extractor, COUNT(*) AS n FROM findings GROUP BY extractor").fetchall()
    graph_stats = {"findings": sum(r["n"] for r in g_rows), "by_extractor": {r["extractor"]: r["n"] for r in g_rows}}
    return {
        "pattern_check": pattern_stats,
        "recheck": recheck_stats,
        "graph": graph_stats,
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
