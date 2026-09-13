"""Blind labeling loop for the verifier gold set, plus scoring.

`cora label` walks the candidate pairs in eval/verifier_gold_candidates.jsonl and records a
human label per pair. It is *blind*: the generator's `kind` and `expected` are hidden while
labeling (pass --reveal to see them after each answer). Progress is saved after every
answer, so it can be done in several sittings.

Scoring: `score_judge` runs a judge over the labeled pairs and reports precision, recall
and the false-negative rate - the number the Phase 1 gate is set on. `score_mechanical_gate`
(in cora/evalset.py) is the baseline the judge must beat.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from .evalset import score_mechanical_gate  # noqa: F401  (re-exported for the CLI)
from .verify import Judge

LABELS = ("supports", "not_supports")


def load_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run `cora evalset` first")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_rows(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def label_loop(path: str | Path, ask=input, say=print, reveal: bool = False, only_unlabeled: bool = True) -> dict:
    """Interactive loop. `ask`/`say` are injectable for tests. Saves after each answer."""
    rows = load_rows(path)
    todo = [i for i, r in enumerate(rows) if not (only_unlabeled and r.get("label") in LABELS)]
    answered = skipped = 0
    say(f"{len(todo)} pairs to label ({len(rows) - len(todo)} already labeled). s=supports  n=not supports  u=unsure/skip  q=quit")
    for n, i in enumerate(todo, 1):
        r = rows[i]
        say("")
        say(f"[{n}/{len(todo)}]  PMID {r['pmid']}")
        say(textwrap.fill(r["passage"], width=100, initial_indent="  ", subsequent_indent="  "))
        say("")
        say(textwrap.fill("CLAIM: " + r["claim"], width=100, initial_indent="  ", subsequent_indent="         "))
        while True:
            ans = (ask("  supports? [s/n/u/q] > ") or "").strip().lower()
            if ans in ("s", "supports"):
                r["label"] = "supports"
            elif ans in ("n", "not", "not_supports"):
                r["label"] = "not_supports"
            elif ans in ("u", "skip", ""):
                skipped += 1
                break
            elif ans in ("q", "quit"):
                save_rows(path, rows)
                say(f"saved. labeled {answered}, skipped {skipped}, remaining {len(todo) - n + 1}")
                return {"labeled": answered, "skipped": skipped, "remaining": len(todo) - n + 1}
            else:
                say("  please answer s, n, u or q")
                continue
            answered += 1
            save_rows(path, rows)
            if reveal:
                say(f"  (generator kind: {r.get('kind')}, generator expected: {r.get('expected')})")
            break
    save_rows(path, rows)
    say(f"done. labeled {answered}, skipped {skipped}")
    return {"labeled": answered, "skipped": skipped, "remaining": 0}


def stats(rows: list[dict]) -> dict:
    by_label: dict[str, int] = {}
    by_kind: dict[str, dict[str, int]] = {}
    agree = disagree = 0
    for r in rows:
        lab = r.get("label") or "unlabeled"
        by_label[lab] = by_label.get(lab, 0) + 1
        k = r.get("kind", "?")
        by_kind.setdefault(k, {})
        by_kind[k][lab] = by_kind[k].get(lab, 0) + 1
        if lab in LABELS and r.get("expected") in LABELS:
            if lab == r["expected"]:
                agree += 1
            else:
                disagree += 1
    return {
        "n": len(rows),
        "by_label": by_label,
        "by_kind": by_kind,
        "human_vs_generator": {"agree": agree, "disagree": disagree},
    }


def score_judge(rows: list[dict], judge: Judge) -> dict:
    """Judge each labeled pair with the whole passage as the single source.

    predicted 'supports' iff supported == 'yes' and claim_strength_vs_source != 'stronger'.
    """
    labeled = [r for r in rows if r.get("label") in LABELS]
    if not labeled:
        return {"n_labeled": 0}
    tp = fp = fn = tn = 0
    by_kind: dict[str, dict[str, int]] = {}
    for r in labeled:
        v = judge.judge(r["claim"], [{"pmid": r["pmid"], "quote": r["passage"], "species": []}])
        pred = "supports" if (v.supported == "yes" and v.claim_strength_vs_source != "stronger") else "not_supports"
        k = r.get("kind", "?")
        by_kind.setdefault(k, {"correct": 0, "wrong": 0})
        by_kind[k]["correct" if pred == r["label"] else "wrong"] += 1
        if r["label"] == "supports":
            tp += pred == "supports"
            fn += pred != "supports"
        else:
            fp += pred == "supports"
            tn += pred != "supports"
    return {
        "n_labeled": len(labeled),
        "judge": getattr(judge, "name", "?"),
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "false_negative_rate": fn / (tp + fn) if (tp + fn) else None,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else None,
        "by_kind": by_kind,
    }
