"""Thin web API + the single-page UI. The card is the unit; the gates decide what it says."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import config, db, generate, ledger, metrics, recheck
from .card import export_draft, render_tight
from .llm import get_llm

STATIC = Path(__file__).parent / "static"


class AskBody(BaseModel):
    query: str
    species: list[str] | None = None
    mock: bool = False
    judge: bool = True  # False = mechanical pattern check only
    k: int | None = None


class FeedbackBody(BaseModel):
    verdict: str
    reason: str | None = None


class CheckinBody(BaseModel):
    useful: bool
    note: str | None = None


def _doc_meta(conn, card) -> dict[str, dict]:
    docs = db.docs_by_pmid(conn, [e.pmid for e in card.draft.evidence])
    return {
        p: {"title": d.get("title"), "pub_year": d.get("pub_year"), "species": d.get("species", []), "journal": d.get("journal"), "flags": d.get("flags", [])}
        for p, d in docs.items()
    }


def _live_cards(conn) -> dict:
    return {r["card"].id: r["card"] for r in ledger.list_cards(conn, include_archived=False, limit=10000)}


def create_app(db_path: str | None = None, recheck_runner=None) -> FastAPI:
    app = FastAPI(title="Cora", version="0.0.7")

    def conn():
        return db.connect(db_path)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/species")
    def species():
        from . import anage

        c = conn()
        rows = anage.panel_summary(c)
        c.close()
        return {"default": config.DEFAULT_SPECIES, "species": rows}

    @app.post("/api/ask")
    def ask(body: AskBody):
        c = conn()
        metrics.record_open(c)
        try:
            llm = get_llm(mock=body.mock)
        except Exception as e:
            c.close()
            raise HTTPException(status_code=503, detail=f"could not create model client: {e}")
        try:
            card = generate.ask(c, body.query, species_keys=body.species, llm=llm, k=body.k, use_judge=body.judge)
        except generate.NoEvidence as e:
            c.close()
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            c.close()
            name = type(e).__name__
            status = 401 if ("Authentication" in name or "PermissionDenied" in name) else 502
            raise HTTPException(status_code=status, detail=f"model call failed: {e}")
        docs = db.docs_by_pmid(c, [e.pmid for e in card.draft.evidence])
        out = {"card": card.model_dump(), "tight": render_tight(card, docs), "docs": _doc_meta(c, card)}
        c.close()
        return out

    @app.get("/api/ledger")
    def get_ledger(include_archived: bool = False, limit: int = 50):
        c = conn()
        metrics.record_open(c)
        rows = ledger.list_cards(c, include_archived=include_archived, limit=limit)
        out = []
        for r in rows:
            card = r["card"]
            docs = db.docs_by_pmid(c, [e.pmid for e in card.draft.evidence])
            out.append(
                {
                    "card": card.model_dump(), "state": r["state"], "duplicate_of": r["duplicate_of"], "expanded": r["expanded"],
                    "exported": r["exported"], "new_evidence": r["new_evidence"], "tight": render_tight(card, docs), "docs": _doc_meta(c, card),
                }
            )
        c.close()
        return out

    @app.get("/api/card/{card_id}")
    def get_card(card_id: str):
        c = conn()
        card = ledger.get(c, card_id)
        if card is None:
            c.close()
            raise HTTPException(status_code=404, detail="no such card")
        updates = db.card_updates(c, card_id)
        ledger.mark_expanded(c, card_id)
        out = {"card": card.model_dump(), "docs": _doc_meta(c, card), "updates": updates}
        c.close()
        return out

    @app.post("/api/card/{card_id}/feedback")
    def post_feedback(card_id: str, body: FeedbackBody):
        c = conn()
        try:
            ledger.feedback(c, card_id, body.verdict, body.reason)
        except KeyError:
            c.close()
            raise HTTPException(status_code=404, detail="no such card")
        except ValueError as e:
            c.close()
            raise HTTPException(status_code=400, detail=str(e))
        c.close()
        return {"ok": True}

    @app.post("/api/card/{card_id}/export")
    def post_export(card_id: str):
        c = conn()
        card = ledger.get(c, card_id)
        if card is None:
            c.close()
            raise HTTPException(status_code=404, detail="no such card")
        docs = db.docs_by_pmid(c, [e.pmid for e in card.draft.evidence])
        ledger.mark_exported(c, card_id)
        text = export_draft(card, docs)
        c.close()
        return {"draft": text, "status": card.status}

    @app.get("/api/metrics")
    def get_metrics():
        c = conn()
        out = metrics.compute(c)
        c.close()
        return out

    @app.post("/api/checkin")
    def post_checkin(body: CheckinBody):
        c = conn()
        metrics.record_checkin(c, body.useful, body.note)
        c.close()
        return {"ok": True}

    @app.post("/api/recheck")
    def post_recheck():
        c = conn()
        metrics.record_open(c)
        runner = recheck_runner or (lambda cc: recheck.run(cc, log=lambda *a, **k: None))
        try:
            report = runner(c)
        except Exception as e:
            c.close()
            raise HTTPException(status_code=502, detail=f"re-check failed: {e}")
        out = {"report": recheck.report_as_dict(report), "text": recheck.render_report(report, _live_cards(c))}
        c.close()
        return out

    @app.get("/api/recheck/last")
    def get_last_recheck():
        c = conn()
        report = recheck.last_report(c)
        if report is None:
            c.close()
            return {"report": None, "text": "no re-check has run yet"}
        out = {"report": recheck.report_as_dict(report), "text": recheck.render_report(report, _live_cards(c))}
        c.close()
        return out

    return app
