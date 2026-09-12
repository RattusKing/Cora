from cora import label as lab
from cora.verify import MockJudge

PASSAGE = "Naked mole-rats show remarkable longevity relative to body size. Hyaluronan may contribute to cancer resistance in this species."


def _rows(labels=(None, None, None)):
    base = [
        {"id": "v0", "kind": "verbatim", "expected": "supports", "claim": "Naked mole-rats show remarkable longevity relative to body size."},
        {"id": "v1", "kind": "hedge_stripped", "expected": "not_supports", "claim": "Hyaluronan contributes to cancer resistance in this species."},
        {"id": "v2", "kind": "strength_inflated", "expected": "not_supports", "claim": "Hyaluronan causes cancer resistance in this species."},
    ]
    return [dict(r, pmid="1", passage=PASSAGE, label=lab_, notes="") for r, lab_ in zip(base, labels)]


def test_label_loop_is_blind_and_saves_after_each_answer(tmp_path):
    path = tmp_path / "c.jsonl"
    lab.save_rows(path, _rows())
    shown = []
    answers = iter(["s", "n", "q"])
    result = lab.label_loop(path, ask=lambda prompt: next(answers), say=shown.append)
    text = "\n".join(shown).lower()
    assert "kind" not in text and "expected" not in text and "hedge_stripped" not in text
    rows = lab.load_rows(path)
    assert [r["label"] for r in rows] == ["supports", "not_supports", None]
    assert result == {"labeled": 2, "skipped": 0, "remaining": 1}


def test_label_loop_skip_reveal_and_resume(tmp_path):
    path = tmp_path / "c.jsonl"
    lab.save_rows(path, _rows())
    shown = []
    answers = iter(["u", "x", "s", "n"])  # skip, invalid, supports, not
    result = lab.label_loop(path, ask=lambda prompt: next(answers), say=shown.append, reveal=True)
    assert result == {"labeled": 2, "skipped": 1, "remaining": 0}
    assert any("generator kind: hedge_stripped" in s for s in shown)
    # a second pass only visits the unlabeled pair
    answers2 = iter(["n"])
    result2 = lab.label_loop(path, ask=lambda prompt: next(answers2), say=lambda s: None)
    assert result2["labeled"] == 1
    assert [r["label"] for r in lab.load_rows(path)] == ["not_supports", "supports", "not_supports"]


def test_stats_crosstab_and_agreement():
    s = lab.stats(_rows(("supports", "not_supports", "supports")))
    assert s["n"] == 3 and s["by_label"] == {"supports": 2, "not_supports": 1}
    assert s["by_kind"]["verbatim"] == {"supports": 1}
    assert s["human_vs_generator"] == {"agree": 2, "disagree": 1}


def test_score_judge_with_mock_reports_the_numbers():
    rows = _rows(("supports", "not_supports", "not_supports"))
    r = lab.score_judge(rows, MockJudge())
    # The mock judge catches causal language but not a stripped hedge: 1 TP, 1 FP, 1 TN.
    assert r["n_labeled"] == 3 and r["judge"] == "mock-judge"
    assert r["precision"] == 0.5 and r["recall"] == 1.0
    assert r["false_negative_rate"] == 0.0 and r["false_positive_rate"] == 0.5
    assert r["by_kind"]["hedge_stripped"] == {"correct": 0, "wrong": 1}


def test_mechanical_gate_baseline_on_labeled_rows(tmp_path):
    path = tmp_path / "c.jsonl"
    lab.save_rows(path, _rows(("supports", "not_supports", "not_supports")))
    r = lab.score_mechanical_gate(path)
    # exact-substring gate: perfect on these three (verbatim passes, the two edits fail)
    assert r["n_labeled"] == 3 and r["precision"] == 1.0 and r["recall"] == 1.0 and r["false_negative_rate"] == 0.0
