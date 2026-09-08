from cora.card import (
    Card, CardDraft, DeskCheck, EvidenceBand, EvidenceItem, GateSummary, HumanLever, SpeciesSupport,
    apply_steering, dedupe_key, evidence_band, export_draft, render_tight,
)


def test_dedupe_key_is_stable_and_species_sensitive():
    a = dedupe_key("DNA repair pathways associated with extreme longevity", ["rockfish", "naked_mole_rat"])
    b = dedupe_key("DNA repair pathways associated with extreme longevity", ["naked_mole_rat", "rockfish"])
    c = dedupe_key("DNA repair pathways associated with extreme longevity", ["rockfish"])
    d = dedupe_key("Proteostasis and oxidative resistance", ["rockfish", "naked_mole_rat"])
    assert a == b and a != c and a != d


def test_evidence_band_counts_species_from_corpus(docs_map):
    passed = [EvidenceItem(pmid="1000001", quote="x" * 30), EvidenceItem(pmid="1000002", quote="y" * 30), EvidenceItem(pmid="3000001", quote="z" * 30)]
    band = evidence_band(passed, docs_map)
    assert band.n_verified_quotes == 3 and band.n_pmids == 3 and band.n_species == 2
    assert "abstract-only" in band.label
    assert evidence_band([], docs_map).label.startswith("ungrounded")


def test_apply_steering_flags_without_hiding():
    band = EvidenceBand(n_verified_quotes=1, n_pmids=1, n_species=1, label="1 verified quote")
    why, flags = apply_steering(band, {"rockfish": 3, "naked_mole_rat": 50}, ["rockfish", "naked_mole_rat"])
    assert any("fewer than 2" in f for f in flags)
    assert any("sparse corpus for: rockfish" in f for f in flags)
    assert any("at least 1" in w for w in why)


def _card(evidence):
    draft = CardDraft(
        pattern="Oxidative resistance recurs across long-lived panel species",
        evidence=evidence, objection="abstract-only support",
        nearest_prior_pmid="2000001", nearest_prior_note="already states most of it",
        human_lever=HumanLever(), desk_check=DeskCheck(step="open PMID 2000001"),
        species_support=[SpeciesSupport(species_key="ocean_quahog", pmids=["2000001"])],
    )
    return Card(
        id="abc123", created_at="2026-09-08T00:00:00+00:00", query="oxidative", species_scope=["ocean_quahog"],
        draft=draft, gate=GateSummary(n_items=1, n_passed=1, n_failed=0, failed=[], fabrication_rate=0.0, groundedness=1.0),
        evidence_band=EvidenceBand(n_verified_quotes=1, n_pmids=1, n_species=1, label="1 verified quote · 1 abstract · 1 species · abstract-only"),
        why_seeing_this=["has verified support"], flags=[], dedupe_key="k", next_action="desk_check",
    )


def test_render_tight_and_export(docs_map):
    card = _card([EvidenceItem(pmid="2000001", quote="Proteins from long-lived individuals appear to resist oxidative damage")])
    tight = render_tight(card, docs_map)
    assert "Oxidative resistance recurs" in tight and "PMID 2000001" in tight and "desk" in tight
    draft = export_draft(card, docs_map)
    assert "(PMID 2000001, 2011)" in draft and "animal data only" in draft


def test_export_of_ungrounded_card_is_marked(docs_map):
    card = _card([])
    assert "do not cite" in export_draft(card, docs_map)
    assert "none survived" in render_tight(card, docs_map)
