import pytest

from cora import config
from cora.verify import MockJudge, PatternVerdict, check_pattern, default_judge_model, get_judge, lexical_strength_check

Q = [
    "high-molecular-mass hyaluronan may contribute to their resistance to cancer",
    "Comparative analyses point to DNA repair pathways and immune gene copy number as candidates associated with extreme longevity",
]
S = [
    {"pmid": "1000001", "quote": Q[0], "species": ["naked_mole_rat"]},
    {"pmid": "3000001", "quote": Q[1], "species": ["rockfish"]},
]


def _verdict(supported="yes", strength="equal"):
    return PatternVerdict(supported=supported, claim_strength_vs_source=strength, evidence_type="unclear", species_match=True, rationale="forced")


def test_lexical_flags_strong_words_absent_from_quotes():
    assert lexical_strength_check("Hyaluronan causes cancer resistance and DNA repair drives longevity", Q) == ["causes", "drives"]


def test_lexical_ignores_strong_words_the_quotes_use():
    assert lexical_strength_check("this demonstrates X", ["the study demonstrates X"]) == []


def test_lexical_multiword_phrase():
    assert "required for" in lexical_strength_check("DNA repair is required for longevity", Q)


def test_mock_judge_equal_when_grounded():
    v = MockJudge().judge("high-molecular-mass hyaluronan may contribute to resistance to cancer in naked mole-rats", S)
    assert v.claim_strength_vs_source == "equal" and v.supported == "yes"


def test_mock_judge_stronger_on_causal_language():
    v = MockJudge().judge("hyaluronan causes cancer resistance", S)
    assert v.claim_strength_vs_source == "stronger"


def test_check_pattern_without_sources_cannot_pass():
    pc = check_pattern("anything at all", [], MockJudge())
    assert not pc.passed and pc.reason == "no_verified_quotes" and not pc.judged


def test_check_pattern_mechanical_only():
    ok = check_pattern("hyaluronan may contribute to cancer resistance", S, None)
    assert ok.passed and not ok.judged and ok.reason is None
    bad = check_pattern("hyaluronan causes cancer resistance", S, None)
    assert not bad.passed and bad.reason == "strong_language_no_judge" and bad.mechanical_strong_words == ["causes"]


def test_judge_stronger_fails_and_agrees_with_mechanical():
    pc = check_pattern("hyaluronan causes cancer resistance", S, MockJudge())
    assert not pc.passed and pc.reason == "stronger" and pc.judged and pc.judge_model == "mock-judge"
    assert pc.disagreement is False


def test_judge_unsupported_fails():
    pc = check_pattern("hyaluronan may contribute to cancer resistance", S, MockJudge(force=_verdict("no", "equal")))
    assert not pc.passed and pc.reason == "unsupported"


def test_partial_support_at_equal_strength_passes():
    pc = check_pattern("hyaluronan may contribute to cancer resistance", S, MockJudge(force=_verdict("partial", "weaker")))
    assert pc.passed and pc.supported == "partial"


def test_disagreement_between_mechanical_and_judge_is_recorded():
    pc = check_pattern("hyaluronan causes cancer resistance", S, MockJudge(force=_verdict("yes", "equal")))
    assert pc.passed and pc.disagreement and pc.mechanical_strong_words == ["causes"]


def test_default_judge_differs_from_drafter(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MODEL_EXPLICIT", False)
    monkeypatch.setattr(config, "JUDGE_MODEL", "claude-sonnet-5")
    assert default_judge_model("claude-opus-5") == "claude-sonnet-5"
    assert default_judge_model("claude-sonnet-5") == "claude-opus-5"


def test_get_judge_rejects_same_model_as_drafter():
    with pytest.raises(ValueError):
        get_judge(drafter_model="claude-sonnet-5", model="claude-sonnet-5")


def test_get_judge_mock():
    assert isinstance(get_judge(mock=True), MockJudge)
