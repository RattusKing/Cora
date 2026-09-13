"""The hypothesis card (tight card v2) and its helpers.

`CardDraft` is what the model produces. `Card` is what the ledger stores: the draft with
its evidence *filtered by the mechanical gate*, plus fields Cora computes itself
(evidence band, groundedness, dedupe key). Nothing in the computed fields is model-asserted.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from . import config

NextAction = Literal["export_draft", "share_page", "desk_check", "none"]

_HUMAN_LEVER_NOTE = "animal data only - no human application"


class EvidenceItem(BaseModel):
    pmid: str = Field(description="PubMed ID of the abstract the quote is taken from")
    quote: str = Field(description="A verbatim span copied exactly from that abstract, at least ~25 characters")
    note: str = Field(default="", description="What this quote supports, in a few words")


class SpeciesSupport(BaseModel):
    species_key: str
    pmids: list[str]


class HumanLever(BaseModel):
    gene: str | None = Field(default=None, description="Human gene symbol if one is implicated, else null")
    direction: str | None = Field(default=None, description="'gain' or 'loss' of function/expression, if stated")
    lever_type: str | None = Field(default=None, description="variant | expression | dosage | systemic, if stated")
    note: str = Field(default=_HUMAN_LEVER_NOTE)


class DeskCheck(BaseModel):
    step: str = Field(description="One concrete step doable at a laptop today: a database lookup, a named paper to read, a dataset to open")
    doable_today: bool = True


class CardDraft(BaseModel):
    """What the model returns. Every quote will be checked mechanically against the corpus."""

    pattern: str = Field(description="One line. A pattern the abstracts support - not a finding, not a causal claim")
    evidence: list[EvidenceItem] = Field(description="2-6 verbatim quotes with PMIDs")
    objection: str = Field(description="The single strongest objection to the pattern, one line")
    nearest_prior_pmid: str | None = Field(default=None, description="PMID of the closest existing paper, if any")
    nearest_prior_note: str = Field(default="", description="How the pattern differs from that paper")
    human_lever: HumanLever
    desk_check: DeskCheck
    species_support: list[SpeciesSupport] = Field(description="Which panel species the cited PMIDs come from")


class GateSummary(BaseModel):
    n_items: int
    n_passed: int
    n_failed: int
    failed: list[dict]  # [{pmid, quote, reason}]
    fabrication_rate: float
    groundedness: float


class EvidenceBand(BaseModel):
    n_verified_quotes: int
    n_pmids: int
    n_species: int
    label: str


class PatternCheck(BaseModel):
    """Result of checking the pattern line against the verified quotes (cora/verify.py)."""

    mechanical_strong_words: list[str] = []
    judged: bool = False
    judge_model: str | None = None
    supported: str | None = None  # yes | partial | no
    strength: str | None = None  # weaker | equal | stronger
    evidence_type: str | None = None
    species_match: bool | None = None
    rationale: str = ""
    passed: bool
    reason: str | None = None  # stronger | unsupported | strong_language_no_judge | no_verified_quotes
    retried: bool = False
    original_pattern: str | None = None
    disagreement: bool = False  # mechanical flag and judge verdict disagree


class Card(BaseModel):
    id: str
    created_at: str
    query: str
    species_scope: list[str]
    draft: CardDraft  # evidence already filtered to gate-passing items
    gate: GateSummary
    evidence_band: EvidenceBand
    why_seeing_this: list[str]
    flags: list[str]
    dedupe_key: str
    next_action: NextAction = "desk_check"
    status: str = "gated"  # gated | ungrounded | overclaim | unsupported
    pattern_check: PatternCheck | None = None


# --- helpers ---------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")
_KEY_STOP = {
    "the", "a", "an", "of", "in", "and", "or", "to", "is", "are", "with", "by", "as", "that", "this",
    "for", "on", "at", "from", "than", "not", "be", "its", "it", "may", "across", "between", "among",
}


def dedupe_key(pattern: str, species_keys: list[str]) -> str:
    """Structured-ish key: the salient tokens of the pattern + the species set."""
    toks = sorted({t for t in _WORD.findall(pattern.lower()) if t not in _KEY_STOP and len(t) > 3})
    basis = "|".join(toks[:12]) + "||" + ",".join(sorted(species_keys))
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def evidence_band(passed: list[EvidenceItem], docs_by_pmid: dict[str, dict]) -> EvidenceBand:
    pmids = {e.pmid for e in passed}
    species: set[str] = set()
    for p in pmids:
        species.update(docs_by_pmid.get(p, {}).get("species", []))
    n_q, n_p, n_s = len(passed), len(pmids), len(species)
    if n_q == 0:
        label = "ungrounded - nothing survived the gate"
    else:
        label = f"{n_q} verified quote{'s' if n_q != 1 else ''} · {n_p} abstract{'s' if n_p != 1 else ''} · {n_s} species · abstract-only"
    return EvidenceBand(n_verified_quotes=n_q, n_pmids=n_p, n_species=n_s, label=label)


def apply_steering(band: EvidenceBand, doc_counts: dict[str, int], species_scope: list[str]) -> tuple[list[str], list[str]]:
    """Explicit rules -> (why_seeing_this, flags). They annotate and order; they never verify."""
    why: list[str] = []
    flags: list[str] = []
    for rule in config.STEERING_RULES:
        rid = rule["id"]
        if rid == "min_verified_quotes":
            if band.n_verified_quotes >= rule["min"]:
                why.append(rule["text"])
            else:
                flags.append(f"fewer than {rule['min']} verified quotes")
        elif rid == "min_species":
            if band.n_species >= rule["min"]:
                why.append(rule["text"])
            else:
                flags.append("no verified species support")
        elif rid == "sparse_species_flag":
            sparse = [s for s in species_scope if doc_counts.get(s, 0) < rule["min_docs"]]
            if sparse:
                flags.append("sparse corpus for: " + ", ".join(sparse))
    return why, flags


def new_card_id() -> str:
    return uuid.uuid4().hex[:12]


def render_tight(card: Card, docs_by_pmid: dict[str, dict]) -> str:
    """The ~8-line tight view. Leads with the verifiable things, not the model's confidence."""
    lines = [f"[{card.id}]  {card.draft.pattern}"]
    if card.draft.evidence:
        e = max(card.draft.evidence, key=lambda x: len(x.quote))
        title = docs_by_pmid.get(e.pmid, {}).get("title", "")
        lines.append(f'  quote   "{e.quote}"  - PMID {e.pmid}{(" · " + title[:70]) if title else ""}')
    else:
        lines.append("  quote   (none survived the gate)")
    lines.append(f"  against {card.draft.objection}")
    if card.draft.nearest_prior_pmid:
        lines.append(f"  vs      PMID {card.draft.nearest_prior_pmid}: {card.draft.nearest_prior_note}")
    lines.append(f"  band    {card.evidence_band.label}")
    pc = card.pattern_check
    if pc is not None:
        if pc.passed:
            who = pc.judge_model or "mechanical only"
            detail = f"supported={pc.supported} strength={pc.strength}" if pc.judged else "no strong language beyond the quotes"
            lines.append(f"  check   pass ({who}): {detail}" + (" · pattern rewritten once" if pc.retried else ""))
        else:
            lines.append(f"  check   FAIL [{pc.reason}]: {pc.rationale[:140]}")
    if card.why_seeing_this:
        lines.append("  why     " + "; ".join(card.why_seeing_this))
    if card.flags:
        lines.append("  flags   " + "; ".join(card.flags))
    lines.append(f"  desk    {card.draft.desk_check.step}")
    lines.append(f"  next    {card.next_action}")
    return "\n".join(lines)


def export_draft(card: Card, docs_by_pmid: dict[str, dict]) -> str:
    """A citation-complete paragraph: every sentence carries the PMID of a gate-verified quote."""
    if not card.draft.evidence:
        return f"{card.draft.pattern} [no gate-verified evidence - do not cite]"
    sentences = [f"{card.draft.pattern}"]
    for e in card.draft.evidence:
        d = docs_by_pmid.get(e.pmid, {})
        cite = f"(PMID {e.pmid}"
        if d.get("pub_year"):
            cite += f", {d['pub_year']}"
        cite += ")"
        sentences.append(f'"{e.quote}" {cite}')
    sentences.append(f"Strongest objection: {card.draft.objection}")
    sentences.append(f"Human relevance: {card.draft.human_lever.note}.")
    return " ".join(sentences)
