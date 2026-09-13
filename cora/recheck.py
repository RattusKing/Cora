"""The re-check: what changed since last time, for each card in the ledger.

Cadence is the evidence rate. This panel gains a handful of aging abstracts a week, so the
re-check is meant to run weekly (cron) or on demand. It never calls a model. It:

  1. searches PubMed again for every panel species and fetches only unseen PMIDs;
  2. re-fetches the PMIDs that live cards cite and records retraction / erratum /
     expression-of-concern flags on them;
  3. matches each new abstract to live cards - does it rank for the card's query within
     the card's species, or share the card's pattern terms? - and records the touches;
  4. writes a templated "what changed" report. No free prose: every line is a new PMID,
     a flag, or a count. A run that finds nothing says so in one line.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from . import config, db, ingest, ledger
from .card import Card, pattern_tokens
from .retrieve import Index, tokenize

MIN_SHARED_TERMS = 3


class Touch(BaseModel):
    card_id: str
    pmid: str
    title: str
    pub_year: int | None
    species: list[str]
    reasons: list[str]


class SourceFlag(BaseModel):
    card_id: str
    pmid: str
    flags: list[str]


class Report(BaseModel):
    date: str
    species: dict[str, dict]  # key -> {"n_pmids", "new", "gone", "stored"}
    n_new: int
    n_gone: int
    touches: list[Touch]
    flags: list[SourceFlag]
    n_cards_live: int
    n_cards_touched: int


def match_new_docs(cards: list[Card], new_docs: list[dict], k: int = 5, corpus_docs: list[dict] | None = None) -> list[Touch]:
    """Which new abstracts touch which live cards, and why. Deterministic, no model.

    A new abstract "ranks" for a card only if it lands in the top k for the card's query
    among ALL abstracts in the card's species (`corpus_docs`), not merely among the new
    ones - with one new abstract, ranking first among the new ones means nothing."""
    touches: list[Touch] = []
    index_cache: dict[tuple, Index] = {}
    for card in cards:
        scope = tuple(sorted(card.species_scope))
        candidates = [d for d in new_docs if set(scope) & set(d.get("species", []))]
        if not candidates:
            continue
        if scope not in index_cache:
            pool = [d for d in (corpus_docs or []) if set(scope) & set(d.get("species", []))]
            by_pmid = {d["pmid"]: d for d in pool}
            for d in candidates:
                by_pmid.setdefault(d["pmid"], d)
            index_cache[scope] = Index(list(by_pmid.values()))
        ranked = {d["pmid"]: i + 1 for i, (d, _) in enumerate(index_cache[scope].search(card.query, k=k))}
        ptoks = pattern_tokens(card.draft.pattern)
        for d in candidates:
            reasons = []
            if d["pmid"] in ranked:
                reasons.append(f"ranks #{ranked[d['pmid']]} for this card's query")
            shared = ptoks & set(tokenize(f"{d.get('title', '')} {d.get('abstract', '')}"))
            if len(shared) >= MIN_SHARED_TERMS:
                reasons.append("shares pattern terms: " + ", ".join(sorted(shared)[:5]))
            if reasons:
                touches.append(Touch(card_id=card.id, pmid=d["pmid"], title=d.get("title", ""), pub_year=d.get("pub_year"), species=d.get("species", []), reasons=reasons))
    return touches


def source_flags(conn, cards: list[Card], fetch_fn=None, log=print) -> list[SourceFlag]:
    """Re-fetch the PMIDs live cards cite; record any retraction / erratum / concern flags."""
    cited: dict[str, list[str]] = {}
    for card in cards:
        for e in card.draft.evidence:
            cited.setdefault(e.pmid, []).append(card.id)
    flags = ingest.fetch_flags(sorted(cited), fetch_fn=fetch_fn, log=log)
    out: list[SourceFlag] = []
    for pmid, fl in flags.items():
        db.set_doc_flags(conn, pmid, fl)
        if fl:
            for cid in cited.get(pmid, []):
                out.append(SourceFlag(card_id=cid, pmid=pmid, flags=fl))
    return out


def run(conn, species_keys: list[str] | None = None, k: int = 5, search_fn=None, fetch_fn=None, log=print) -> Report:
    species_keys = list(species_keys or config.DEFAULT_SPECIES)
    manifest = ingest.load_manifest() or ingest._empty_manifest()
    per_species: dict[str, dict] = {}
    new_docs: list[dict] = []
    for key in species_keys:
        r = ingest.update_species(conn, config.SPECIES[key], manifest, log=log, search_fn=search_fn, fetch_fn=fetch_fn)
        per_species[key] = {"n_pmids": len(r["pmids"]), "new": len(r["new"]), "gone": len(r["gone"]), "stored": r["stored"]}
        new_docs.extend(r["docs"])
        prev = manifest["species"].get(key, {})
        manifest["species"][key] = {
            "query": r["query"], "date": prev.get("date", r["date"]), "updated": r["date"],
            "n_pmids": len(r["pmids"]), "stored": prev.get("stored", 0) + r["stored"], "pmids": r["pmids"],
        }
    # a doc can belong to several species; dedupe by pmid and merge species lists from the DB
    if new_docs:
        stored = db.docs_by_pmid(conn, sorted({d["pmid"] for d in new_docs}))
        new_docs = list(stored.values())

    live = [r["card"] for r in ledger.list_cards(conn, include_archived=False, limit=10000)]
    # Only touches and flags not already recorded count as "changed": a flag seen last week
    # is not news this week.
    corpus = db.get_docs(conn, species_keys) if new_docs else []
    touches = [t for t in match_new_docs(live, new_docs, k=k, corpus_docs=corpus) if db.add_card_update(conn, t.card_id, t.pmid, "; ".join(t.reasons))]
    flags = [
        f for f in source_flags(conn, live, fetch_fn=fetch_fn, log=log)
        if db.add_card_update(conn, f.card_id, f.pmid, "source flag: " + ", ".join(f.flags))
    ]

    report = Report(
        date=db.today(),
        species=per_species,
        n_new=sum(s["new"] for s in per_species.values()),
        n_gone=sum(s["gone"] for s in per_species.values()),
        touches=touches,
        flags=flags,
        n_cards_live=len(live),
        n_cards_touched=len({t.card_id for t in touches} | {f.card_id for f in flags}),
    )
    manifest.setdefault("runs", []).append({"date": report.date, "kind": "recheck", "species": species_keys, "new": report.n_new, "gone": report.n_gone, "cards_touched": report.n_cards_touched})
    manifest["runs"] = manifest["runs"][-50:]
    ingest.save_manifest(manifest)
    db.log_recheck(conn, report.n_new, report.n_gone, report.n_cards_touched, len(flags), report.model_dump_json())
    db.log_event(conn, "recheck", {"new": report.n_new, "cards_touched": report.n_cards_touched, "flags": len(flags)})
    return report


def last_report(conn) -> Report | None:
    row = conn.execute("SELECT report_json FROM recheck_log ORDER BY id DESC LIMIT 1").fetchone()
    return Report.model_validate_json(row["report_json"]) if row else None


def render_report(report: Report, cards_by_id: dict[str, Card]) -> str:
    """Templated 'what changed'. Every line is a PMID, a flag, or a count."""
    per = " · ".join(f"{k} +{v['new']}" for k, v in report.species.items())
    head = f"recheck {report.date} · {len(report.species)} species · {report.n_new} new abstract{'s' if report.n_new != 1 else ''} ({per}) · {report.n_gone} no longer returned"
    if not report.touches and not report.flags:
        return head + f"\nnothing changed for the {report.n_cards_live} live cards"
    lines = [head]
    by_card: dict[str, list[Touch]] = {}
    for t in report.touches:
        by_card.setdefault(t.card_id, []).append(t)
    if by_card:
        lines.append(f"cards with new evidence: {len(by_card)}")
        for cid, ts in by_card.items():
            card = cards_by_id.get(cid)
            lines.append(f"  [{cid}] {card.draft.pattern if card else ''}")
            for t in ts:
                lines.append(f"     + PMID {t.pmid} ({t.pub_year or '?'}; {', '.join(t.species)}) {t.title[:80]}")
                for r in t.reasons:
                    lines.append(f"         {r}")
    if report.flags:
        lines.append(f"source flags: {len(report.flags)}")
        for f in report.flags:
            lines.append(f"  [{f.card_id}] PMID {f.pmid} has {', '.join(f.flags)} - re-read before citing")
    untouched = report.n_cards_live - report.n_cards_touched
    lines.append(f"unchanged: {untouched} card{'s' if untouched != 1 else ''}")
    return "\n".join(lines)


def report_as_dict(report: Report) -> dict:
    return json.loads(report.model_dump_json())
