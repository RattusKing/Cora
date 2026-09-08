# Cora eval harness (P0.5)

Two things the red-team said must exist **before** any generation code is trusted:

## 1. Verifier gold set — human-labeled (claim, passage) pairs

```
cora ingest                 # real corpus first
cora evalset --n 150        # writes eval/verifier_gold_candidates.jsonl (UNLABELED)
```

Each line is `{id, kind, expected, claim, pmid, passage, label: null, notes}`.
**A human fills `label`** with `supports` or `not_supports` after reading the passage.
`expected` is only what the generator intended (it built the hard negatives by stripping
hedges, inflating claim strength, altering a word, or attributing a real sentence to the
wrong paper) — the human label is the truth, and the two will sometimes disagree. That is
the point.

P0.5's verifier is the **mechanical gate** (exact substring). Score it as the baseline any
model-judged verifier in P1 must beat:

```python
from cora.evalset import score_mechanical_gate
score_mechanical_gate("eval/verifier_gold_candidates.jsonl")   # precision / recall / FNR
```

Gate autonomy on **verifier false-negative rate**, never on the fabrication rate the system
computes about itself.

## 2. Closed-book baseline — the contamination control

`eval/gold_questions.jsonl` lists discovery questions with known answers. Before any
"Cora rediscovered X" claim, run the same questions against the model with **no passages**:

```
python eval/closed_book.py            # needs model credentials
```

If the bare model names the mechanism from memory, that item is **void** as evidence of
discovery — it measures recall, not reasoning. The four classic items (rockfish DNA repair,
naked mole-rat hyaluronan, bowhead ERCC1/PCNA, *Turritopsis*) are kept as
**contamination positive controls**: a frontier model will almost certainly name them.
Real gold must be published after the model's training cutoff and re-selected on every
model upgrade.

## 3. Canaries (run any time)

```
cora canary --n 20
```

Builds known-good and known-bad citations from the real corpus and asserts the gate
catches every bad one and passes every good one. A surviving known-bad canary is the alarm
that pauses everything (the future circuit breaker's absolute tripwire).
