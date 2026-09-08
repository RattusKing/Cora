from cora import config
from cora.card import CardDraft, DeskCheck, EvidenceItem, HumanLever, SpeciesSupport
from cora.gate import check_item, gate_draft, make_canaries, normalize, run_canaries


def test_exact_quote_passes(docs_map):
    item = EvidenceItem(pmid="1000001", quote="high-molecular-mass hyaluronan may contribute to their resistance to cancer")
    assert check_item(item, docs_map) is None


def test_whitespace_and_case_are_normalized(docs_map):
    item = EvidenceItem(pmid="3000001", quote="  Comparative   analyses point TO dna repair pathways ")
    assert check_item(item, docs_map) is None


def test_title_counts_as_source_text(docs_map):
    item = EvidenceItem(pmid="3000002", quote="Otolith-based age validation in deep-dwelling Sebastes")
    assert check_item(item, docs_map) is None


def test_unknown_pmid_fails(docs_map):
    item = EvidenceItem(pmid="9999999", quote="Naked mole-rats (Heterocephalus glaber) show remarkable longevity")
    assert check_item(item, docs_map) == "pmid_not_in_corpus"


def test_altered_quote_fails(docs_map):
    item = EvidenceItem(pmid="1000001", quote="high-molecular-mass hyaluronan definitely causes their resistance to cancer")
    assert check_item(item, docs_map) == "quote_not_in_abstract"


def test_hedge_stripped_quote_fails(docs_map):
    # the abstract says "may contribute" - dropping "may" is a strength inflation and must not pass
    item = EvidenceItem(pmid="1000001", quote="high-molecular-mass hyaluronan contribute to their resistance to cancer")
    assert check_item(item, docs_map) == "quote_not_in_abstract"


def test_right_quote_wrong_pmid_fails(docs_map):
    item = EvidenceItem(pmid="2000001", quote="Rockfishes of the genus Sebastes vary in maximum lifespan")
    assert check_item(item, docs_map) == "quote_not_in_abstract"


def test_too_short_fails(docs_map):
    item = EvidenceItem(pmid="1000001", quote="cancer")
    assert check_item(item, docs_map) == "quote_too_short"
    assert len("cancer") < config.MIN_QUOTE_CHARS


def test_normalize_unicode_quotes_and_dashes():
    assert normalize("mole‑rat “quote” — x") == normalize('mole‑rat "quote" - x')
    assert normalize("a  \n  b") == "a b"


def _draft(items, prior=None):
    return CardDraft(
        pattern="test pattern", evidence=items, objection="none",
        nearest_prior_pmid=prior, nearest_prior_note="close",
        human_lever=HumanLever(), desk_check=DeskCheck(step="look it up"),
        species_support=[SpeciesSupport(species_key="MODEL_ASSERTED_WRONG", pmids=["1000001"])],
    )


def test_gate_draft_filters_and_recomputes_species(docs_map):
    good = EvidenceItem(pmid="1000001", quote="Naked mole-rats (Heterocephalus glaber) show remarkable longevity relative to body size")
    bad = EvidenceItem(pmid="0000001", quote="This citation does not exist anywhere in the corpus at all")
    filtered, summary = gate_draft(_draft([good, bad], prior="9999999"), docs_map)
    assert summary.n_items == 2 and summary.n_passed == 1 and summary.n_failed == 1
    assert summary.fabrication_rate == 0.5 and summary.groundedness == 0.5
    assert [e.pmid for e in filtered.evidence] == ["1000001"]
    # species support comes from the corpus, never from the model
    assert [s.species_key for s in filtered.species_support] == ["naked_mole_rat"]
    # a nearest-prior PMID that is not in the corpus is nulled
    assert filtered.nearest_prior_pmid is None and filtered.nearest_prior_note == ""


def test_canaries_all_caught(docs_map):
    canaries = make_canaries(list(docs_map.values()), n=6)
    kinds = {c["kind"] for c in canaries}
    assert {"verbatim", "fabricated_pmid", "altered_word", "wrong_pmid", "too_short"} <= kinds
    assert "hedge_stripped" in kinds  # the synthetic corpus has hedges
    result = run_canaries(docs_map, canaries)
    assert result["ok"], result["misses"]
    assert result["detection_rate"] == 1.0
    assert result["false_reject_rate"] == 0.0
