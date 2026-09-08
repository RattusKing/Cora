"""The mechanical citation gate - the trust core of P0.5. No model opinion is involved.

An evidence item passes only if:
  1. its PMID exists in the local corpus, and
  2. its quote (whitespace/unicode-normalized, case-folded) is an exact substring of that
     abstract (or its title), and is at least MIN_QUOTE_CHARS long.

Everything else is dropped and *counted*: the fabrication rate is failed / total items,
and every deletion is logged, so cutting the hard-to-verify claims cannot quietly improve
the score without showing up in the deletion count.

Canaries are known-good and known-bad items built from real corpus documents; the gate
must catch every known-bad one and pass every known-good one.
"""

from __future__ import annotations

import re
import unicodedata

from . import config
from .card import CardDraft, EvidenceItem, GateSummary, SpeciesSupport

_WS = re.compile(r"\s+")
_MAP = {
    "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-", " ": " ", " ": " ", " ": " ",
}


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    for k, v in _MAP.items():
        s = s.replace(k, v)
    return _WS.sub(" ", s).strip()


def check_item(item: EvidenceItem, docs_by_pmid: dict[str, dict]) -> str | None:
    """Return None if the item passes, otherwise the failure reason."""
    pmid = str(item.pmid).strip()
    doc = docs_by_pmid.get(pmid)
    if doc is None:
        return "pmid_not_in_corpus"
    q = normalize(item.quote)
    if len(q) < config.MIN_QUOTE_CHARS:
        return "quote_too_short"
    hay = normalize(doc.get("abstract", "")).casefold()
    title = normalize(doc.get("title", "")).casefold()
    qf = q.casefold()
    if qf in hay or qf in title:
        return None
    return "quote_not_in_abstract"


def gate_draft(draft: CardDraft, docs_by_pmid: dict[str, dict]) -> tuple[CardDraft, GateSummary]:
    """Filter a draft's evidence through the gate. Species support is *recomputed* from the
    corpus for the surviving PMIDs - it is never taken from the model."""
    passed: list[EvidenceItem] = []
    failed: list[dict] = []
    for item in draft.evidence:
        reason = check_item(item, docs_by_pmid)
        if reason is None:
            passed.append(item)
        else:
            failed.append({"pmid": item.pmid, "quote": item.quote, "reason": reason})
    n = len(draft.evidence)
    summary = GateSummary(
        n_items=n,
        n_passed=len(passed),
        n_failed=len(failed),
        failed=failed,
        fabrication_rate=(len(failed) / n) if n else 0.0,
        groundedness=(len(passed) / n) if n else 0.0,
    )
    by_species: dict[str, list[str]] = {}
    for e in passed:
        for s in docs_by_pmid.get(e.pmid, {}).get("species", []):
            by_species.setdefault(s, [])
            if e.pmid not in by_species[s]:
                by_species[s].append(e.pmid)
    support = [SpeciesSupport(species_key=k, pmids=v) for k, v in sorted(by_species.items())]
    prior = draft.nearest_prior_pmid if draft.nearest_prior_pmid in docs_by_pmid else None
    filtered = draft.model_copy(
        update={
            "evidence": passed,
            "species_support": support,
            "nearest_prior_pmid": prior,
            "nearest_prior_note": draft.nearest_prior_note if prior else "",
        }
    )
    return filtered, summary


# --- canaries -----------------------------------------------------------------

_HEDGES = [
    "possibly ", "possible ", "potentially ", "potential ", "suggesting ", "suggests that ",
    "suggest that ", "may ", "might ", "could ", "likely ", "appears to ", "appear to ",
]
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def first_sentence(text: str, min_chars: int = 40, max_chars: int = 300) -> str:
    """A sentence-ish slice of the ORIGINAL text (so it is guaranteed to be a substring)."""
    text = text or ""
    for sent in _SENT_END.split(text):
        sent = sent.strip()
        if len(sent) >= min_chars:
            return sent[:max_chars]
    return text[:max_chars]


def hedged_sentence(text: str, min_chars: int = 40, max_chars: int = 300) -> tuple[str, str] | None:
    """The first sentence of the ORIGINAL text that contains a hedge, plus the hedge found."""
    for sent in _SENT_END.split(text or ""):
        sent = sent.strip()
        if len(sent) < min_chars:
            continue
        low = sent.lower()
        for h in _HEDGES:
            if h in low:
                return sent[:max_chars], h
    return None


def make_canaries(docs: list[dict], n: int = 10) -> list[dict]:
    """Known-good and known-bad items from real documents.

    Each entry: {"kind", "expect": "pass"|"fail", "item": EvidenceItem}.
    """
    canaries: list[dict] = []
    usable = [d for d in docs if len(d.get("abstract", "")) > 120][: max(n, 2)]
    for i, d in enumerate(usable):
        sent = first_sentence(d["abstract"])
        # positive control - must PASS
        canaries.append({"kind": "verbatim", "expect": "pass", "item": EvidenceItem(pmid=d["pmid"], quote=sent)})
        # fabricated PMID - must FAIL
        canaries.append({"kind": "fabricated_pmid", "expect": "fail", "item": EvidenceItem(pmid=f"0000{i:04d}", quote=sent)})
        # one word altered - must FAIL
        words = sent.split()
        if len(words) > 6:
            words[len(words) // 2] = "REPLACEDWORD"
            canaries.append({"kind": "altered_word", "expect": "fail", "item": EvidenceItem(pmid=d["pmid"], quote=" ".join(words))})
        # hedge stripped - must FAIL (uses whichever sentence of the abstract carries a hedge)
        hs = hedged_sentence(d["abstract"])
        if hs is not None:
            hsent, h = hs
            idx = hsent.lower().find(h)
            stripped = hsent[:idx] + hsent[idx + len(h):]
            canaries.append({"kind": "hedge_stripped", "expect": "fail", "item": EvidenceItem(pmid=d["pmid"], quote=stripped)})
        # right quote, wrong PMID (from another doc) - must FAIL
        other = usable[(i + 1) % len(usable)]
        if other["pmid"] != d["pmid"]:
            canaries.append({"kind": "wrong_pmid", "expect": "fail", "item": EvidenceItem(pmid=other["pmid"], quote=sent)})
        # trivially short "quote" - must FAIL
        canaries.append({"kind": "too_short", "expect": "fail", "item": EvidenceItem(pmid=d["pmid"], quote=sent[:10])})
    return canaries


def run_canaries(docs_by_pmid: dict[str, dict], canaries: list[dict]) -> dict:
    """Returns detection statistics. A surviving known-bad canary is the alarm."""
    caught = missed = passed_ok = passed_bad = 0
    misses: list[dict] = []
    for c in canaries:
        reason = check_item(c["item"], docs_by_pmid)
        if c["expect"] == "fail":
            if reason is None:
                missed += 1
                misses.append({"kind": c["kind"], "pmid": c["item"].pmid, "quote": c["item"].quote[:80]})
            else:
                caught += 1
        else:
            if reason is None:
                passed_ok += 1
            else:
                passed_bad += 1
                misses.append({"kind": c["kind"] + "_wrongly_failed", "pmid": c["item"].pmid, "reason": reason})
    n_bad = caught + missed
    n_good = passed_ok + passed_bad
    return {
        "n_canaries": len(canaries),
        "known_bad": n_bad,
        "known_bad_caught": caught,
        "detection_rate": (caught / n_bad) if n_bad else 1.0,
        "known_good": n_good,
        "known_good_passed": passed_ok,
        "false_reject_rate": (passed_bad / n_good) if n_good else 0.0,
        "misses": misses,
        "ok": missed == 0 and passed_bad == 0,
    }
