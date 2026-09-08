"""Closed-book baseline: ask the model each gold question with NO corpus.

This is the contamination control the red-team required. Any gold item the bare model
reproduces from memory cannot count as a Cora "discovery".

    python eval/closed_book.py [--gold eval/gold_questions.jsonl] [--out eval/closed_book_results.jsonl]

Requires model credentials (ANTHROPIC_API_KEY or `ant auth login`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SYSTEM = (
    "You are being tested for prior knowledge. You have NO documents. Answer from memory only. "
    "For the question, (1) name the mechanism, genes or pathways you believe are associated, "
    "(2) say whether you are recalling a specific publication and, if so, which, and "
    "(3) rate your confidence low/medium/high. Be brief."
)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold_questions.jsonl")
    ap.add_argument("--out", default="eval/closed_book_results.jsonl")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)

    try:
        import anthropic
        from cora import config
    except ImportError as e:
        sys.exit(f"missing dependency: {e}")

    client = anthropic.Anthropic()
    model = args.model or config.DEFAULT_MODEL
    gold = [json.loads(l) for l in Path(args.gold).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for g in gold:
            resp = client.messages.create(
                model=model,
                max_tokens=2000,
                system=SYSTEM,
                messages=[{"role": "user", "content": g["question"]}],
            )
            if resp.stop_reason == "refusal":
                answer = "<refused>"
            else:
                answer = "".join(b.text for b in resp.content if b.type == "text")
            rec = {"id": g["id"], "question": g["question"], "known_answer": g.get("known_answer_summary"), "model": model, "answer": answer}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{g['id']}] {g['question']}\n  -> {answer[:300]}\n")
    print(f"wrote {out}. Mark any item the bare model reproduced as VOID for discovery evals.")


if __name__ == "__main__":
    main()
