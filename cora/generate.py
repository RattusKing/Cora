"""retrieve -> draft -> gate -> band -> steering -> ledger. Single-shot, on demand."""

from __future__ import annotations

from . import config, db, ledger
from .card import Card, apply_steering, dedupe_key, evidence_band, new_card_id
from .gate import gate_draft
from .llm import LLM, get_llm
from .retrieve import build_index


class NoEvidence(Exception):
    pass


def ask(conn, query: str, species_keys: list[str] | None = None, llm: LLM | None = None, k: int | None = None) -> Card:
    species_keys = list(species_keys or config.DEFAULT_SPECIES)
    index = build_index(conn, species_keys)
    hits = index.search(query, k=k or config.RETRIEVE_K, species_keys=species_keys)
    if not hits:
        raise NoEvidence(f"no abstracts matched {query!r} in {species_keys}; run `cora ingest` first?")
    passages = [
        {"pmid": d["pmid"], "species": d.get("species", []), "title": d.get("title", ""), "abstract": d.get("abstract", "")}
        for d, _ in hits
    ]
    llm = llm or get_llm()
    draft = llm.draft(query, passages, species_keys)

    # The gate checks against the WHOLE corpus: a PMID "resolves" iff it is stored locally.
    cited = [e.pmid for e in draft.evidence] + ([draft.nearest_prior_pmid] if draft.nearest_prior_pmid else [])
    docs_map = db.docs_by_pmid(conn, list({*cited, *(p["pmid"] for p in passages)}))
    filtered, summary = gate_draft(draft, docs_map)

    band = evidence_band(filtered.evidence, docs_map)
    why, flags = apply_steering(band, db.count_docs_by_species(conn), species_keys)
    supported_species = sorted({s.species_key for s in filtered.species_support})
    card = Card(
        id=new_card_id(),
        created_at=db.now_iso(),
        query=query,
        species_scope=species_keys,
        draft=filtered,
        gate=summary,
        evidence_band=band,
        why_seeing_this=why,
        flags=flags,
        dedupe_key=dedupe_key(filtered.pattern, supported_species),
        next_action="desk_check" if filtered.evidence else "none",
        status="gated" if filtered.evidence else "ungrounded",
    )
    ledger.insert(conn, card)
    db.log_gate(conn, card.id, summary.n_items, summary.n_passed, [f["reason"] for f in summary.failed])
    db.log_event(conn, "ask", {"card_id": card.id, "llm": getattr(llm, "name", "?"), "n_passages": len(passages)})
    return card
