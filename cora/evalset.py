"""Build the verifier gold set *candidates* - unlabeled pairs for a human to label.

Why this exists: the red-team's most convergent trust finding was that a verifier must be
measured against human labels before it is trusted, and before any generation code ships.
P0.5's verifier is mechanical, but P1 will add model-judged entailment; this file produces
the (claim, passage) pairs it will be measured against, including hard negatives:

  verbatim          - a real sentence from the abstract          (expected: supports)
  hedge_stripped    - same sentence with its hedge removed        (expected: not_supports*)
  strength_inflated - "suggest" -> "demonstrate", etc.            (expected: not_supports*)
  altered_word      - one content word replaced                   (expected: not_supports)
  wrong_pmid        - a real sentence attributed to another paper (expected: not_supports)

(*) "expected" is what the *generator* of the pair intends; the human label is the truth.
A hedge-stripped sentence can occasionally still be supported by the rest of the abstract -
that is exactly why a human labels it.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from . import db
from .gate import check_item, first_sentence
from .card import EvidenceItem

_HEDGE_PATTERNS = [
    (r"\bmay\s", ""), (r"\bmight\s", ""), (r"\bcould\s", ""), (r"\bpossibl[ey]\s", ""),
    (r"\bpotential(?:ly)?\s", ""), (r"\bsuggest(?:s|ing)?\s+that\s", ""), (r"\bappears?\s+to\s", ""),
]
_INFLATE = [
    (r"\bsuggest(?:s|ed|ing)?\b", "demonstrate"), (r"\bassociated with\b", "causes"),
    (r"\bmay contribute to\b", "drives"), (r"\bcorrelat(?:es|ed) with\b", "causes"),
    (r"\bcould\b", "does"), (r"\bmight\b", "does"),
]
_WORD = re.compile(r"[A-Za-z]{6,}")


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _apply(patterns, text) -> str | None:
    for pat, rep in patterns:
        new = re.sub(pat, rep, text, count=1, flags=re.IGNORECASE)
        if new != text:
            return new
    return None


def _apply_anywhere(patterns, abstract: str, min_chars: int = 60) -> tuple[str, str] | None:
    """Find the first sentence of the abstract that a pattern changes; return (original, changed).
    First sentences rarely hedge - the hedges live mid-abstract, so scan all of them."""
    for sent in _SENT_END.split(abstract or ""):
        sent = sent.strip()
        if len(sent) < min_chars:
            continue
        changed = _apply(patterns, sent)
        if changed:
            return sent, changed
    return None


def build_verifier_candidates(docs: list[dict], n: int = 60, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    usable = [d for d in docs if len(d.get("abstract", "")) > 200]
    rng.shuffle(usable)
    out: list[dict] = []
    i = 0
    for d in usable:
        if len(out) >= n:
            break
        sent = first_sentence(d["abstract"], min_chars=60)
        base = {"pmid": d["pmid"], "passage": d["abstract"], "label": None, "notes": ""}
        out.append({"id": f"v{i:04d}", "kind": "verbatim", "expected": "supports", "claim": sent, **base}); i += 1
        hs = _apply_anywhere(_HEDGE_PATTERNS, d["abstract"])
        if hs:
            out.append({"id": f"v{i:04d}", "kind": "hedge_stripped", "expected": "not_supports", "claim": hs[1], "original": hs[0], **base}); i += 1
        inf = _apply_anywhere(_INFLATE, d["abstract"])
        if inf:
            out.append({"id": f"v{i:04d}", "kind": "strength_inflated", "expected": "not_supports", "claim": inf[1], "original": inf[0], **base}); i += 1
        words = _WORD.findall(sent)
        if len(words) >= 3:
            w = rng.choice(words[1:])
            out.append({"id": f"v{i:04d}", "kind": "altered_word", "expected": "not_supports", "claim": sent.replace(w, "unrelated", 1), **base}); i += 1
        other = rng.choice(usable)
        if other["pmid"] != d["pmid"]:
            out.append({"id": f"v{i:04d}", "kind": "wrong_pmid", "expected": "not_supports", "claim": sent, "pmid": other["pmid"], "passage": other["abstract"], "label": None, "notes": ""}); i += 1
    return out[:n]


def write_verifier_candidates(conn, n: int = 60, out: str | Path = "eval/verifier_gold_candidates.jsonl") -> Path:
    docs = db.get_docs(conn)
    if not docs:
        raise SystemExit("no documents in the corpus - run `cora ingest` first")
    rows = build_verifier_candidates(docs, n=n)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def score_mechanical_gate(labeled_path: str | Path) -> dict:
    """Baseline: how the P0.5 mechanical gate does against a hand-labeled set.

    The mechanical gate can only say 'supports' for verbatim spans, so it will have perfect
    precision and poor recall on paraphrases - that is the number P1's verifier must beat.
    """
    rows = [json.loads(line) for line in Path(labeled_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("label") in ("supports", "not_supports")]
    if not rows:
        return {"n_labeled": 0}
    tp = fp = fn = tn = 0
    for r in rows:
        doc = {r["pmid"]: {"pmid": r["pmid"], "abstract": r["passage"], "title": ""}}
        verdict = "supports" if check_item(EvidenceItem(pmid=r["pmid"], quote=r["claim"]), doc) is None else "not_supports"
        if r["label"] == "supports":
            tp += verdict == "supports"
            fn += verdict != "supports"
        else:
            fp += verdict == "supports"
            tn += verdict != "supports"
    return {
        "n_labeled": len(rows),
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "false_negative_rate": fn / (tp + fn) if (tp + fn) else None,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else None,
    }
