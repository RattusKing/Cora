"""retrieve -> draft -> gate quotes -> check pattern -> band -> steering -> ledger. Single-shot."""

from __future__ import annotations

from . import config, db, graph, ledger
from .card import Card, apply_steering, dedupe_key, evidence_band, new_card_id
from .gate import gate_draft
from .llm import LLM, MockLLM, get_llm
from .retrieve import build_index
from .verify import Judge, check_pattern, get_judge


class NoEvidence(Exception):
    pass


def ask(
    conn,
    query: str,
    species_keys: list[str] | None = None,
    llm: LLM | None = None,
    judge: Judge | None = None,
    k: int | None = None,
    use_judge: bool = True,
    retries: int | None = None,
) -> Card:
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

    # 1. Quotes: the mechanical gate checks against the WHOLE corpus.
    cited = [e.pmid for e in draft.evidence] + ([draft.nearest_prior_pmid] if draft.nearest_prior_pmid else [])
    docs_map = db.docs_by_pmid(conn, list({*cited, *(p["pmid"] for p in passages)}))
    filtered, summary = gate_draft(draft, docs_map)

    # 2. Pattern: mechanical strength check, then an independent judge; one rewrite allowed.
    sources = [
        {"pmid": e.pmid, "quote": e.quote, "species": docs_map.get(e.pmid, {}).get("species", [])}
        for e in filtered.evidence
    ]
    if use_judge and judge is None and sources:
        judge = get_judge(mock=isinstance(llm, MockLLM), drafter_model=getattr(llm, "model", None))
    active_judge = judge if use_judge else None
    pc = check_pattern(filtered.pattern, sources, active_judge)
    retries = config.PATTERN_RETRIES if retries is None else retries
    if not pc.passed and pc.reason in ("stronger", "strong_language_no_judge") and retries > 0:
        original = filtered.pattern
        rewritten = llm.rewrite_pattern(original, sources, pc.rationale)
        filtered = filtered.model_copy(update={"pattern": rewritten})
        pc = check_pattern(rewritten, sources, active_judge)
        pc.retried = True
        pc.original_pattern = original

    # 3. Everything below is computed by Cora, never asserted by a model.
    band = evidence_band(filtered.evidence, docs_map)
    why, flags = apply_steering(band, db.count_docs_by_species(conn), species_keys)
    if not filtered.evidence:
        status = "ungrounded"
    elif pc.passed:
        status = "gated"
    elif pc.reason == "unsupported":
        status = "unsupported"
    else:
        status = "overclaim"
    if status not in ("gated", "ungrounded"):
        flags.append(f"pattern check failed: {pc.reason}")
    supported_species = sorted({s.species_key for s in filtered.species_support})
    convergence = None
    if graph.findings_count(conn):
        convergence = graph.card_convergence(conn, filtered.pattern)
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
        next_action="desk_check" if status == "gated" else "none",
        status=status,
        pattern_check=pc,
        convergence=convergence,
    )
    ledger.insert(conn, card)
    db.log_gate(conn, card.id, summary.n_items, summary.n_passed, [f["reason"] for f in summary.failed])
    db.log_pattern(conn, card.id, pc)
    db.log_event(conn, "ask", {"card_id": card.id, "llm": getattr(llm, "name", "?"), "judge": getattr(active_judge, "name", None), "n_passages": len(passages)})
    return card
