from cora import db, generate, graph
from cora.card import EvidenceItem
from cora.gate import check_item
from cora.llm import MockLLM


def _doc(pmid, species, abstract, pub_types=None, title="t"):
    return {"pmid": pmid, "species": species, "abstract": abstract, "title": title, "pub_types": pub_types}


def test_lexicon_tiers_directions_and_review_type():
    ex = graph.LexiconExtractor()
    fs = ex.extract(_doc("1", ["naked_mole_rat"], "We found that high-molecular-mass hyaluronan may contribute to their resistance to cancer, suggesting a possible role for the extracellular matrix in protection."))
    mechs = {f.mechanism: f for f in fs}
    assert {"hyaluronan", "cancer_resistance", "extracellular_matrix"} <= set(mechs)
    assert mechs["hyaluronan"].tier == "association" and mechs["hyaluronan"].direction is None
    assert mechs["hyaluronan"].evidence_type == "unknown"  # no pub_types on this doc

    fs = ex.extract(_doc("2", ["killifish"], "SIRT6 overexpression extended lifespan in male fish by twenty percent.", pub_types=["Journal Article"]))
    f = next(x for x in fs if x.mechanism == "nutrient_sensing")
    assert f.tier == "intervention" and f.direction == "up" and f.evidence_type == "primary"

    fs = ex.extract(_doc("3", ["hydra"], "There was no association between telomere length and age in polyps.", pub_types=["Journal Article", "Review"]))
    f = next(x for x in fs if x.mechanism == "telomere")
    assert f.tier == "tested_negative" and f.evidence_type == "review"

    # a short sentence is never a finding; quotes are always substrings of the abstract
    doc = _doc("4", ["rockfish"], "DNA repair. Comparative analyses point to DNA repair pathways as candidates associated with extreme longevity.")
    fs = ex.extract(doc)
    assert all(len(f.quote) >= 25 for f in fs)
    assert all(check_item(EvidenceItem(pmid="4", quote=f.quote), {"4": doc}) is None for f in fs)


def test_rather_than_is_not_a_negation():
    # a real bowhead sentence: a strong positive for DNA repair that names a rejected alternative
    ex = graph.LexiconExtractor()
    sent = "These results indicate that rather than possessing additional tumor suppressor genes as barriers to oncogenesis, the bowhead whale relies on more accurate and efficient DNA repair to preserve genome integrity."
    f = next(x for x in ex.extract(_doc("9", ["bowhead_whale"], sent, pub_types=["Journal Article"])) if x.mechanism == "dna_repair")
    assert f.tier == "association" and f.direction == "up"
    # real negations still count as negative
    g = next(x for x in ex.extract(_doc("10", ["hydra"], "We found no association between DNA repair capacity and lifespan in Hydra polyps.")) if x.mechanism == "dna_repair")
    assert g.tier == "tested_negative"


def test_build_stores_gated_findings(conn):
    stats = graph.build(conn, graph.LexiconExtractor(), log=lambda *a, **k: None)
    assert stats["findings"] > 0 and stats["dropped_by_gate"] == 0 and stats["extractor"] == "lexicon-v1"
    assert graph.findings_count(conn) == stats["findings"]
    # rebuilding replaces rather than duplicates
    again = graph.build(conn, graph.LexiconExtractor(), log=lambda *a, **k: None)
    assert graph.findings_count(conn) == again["findings"] == stats["findings"]


def test_convergence_counts_lineages_not_species(conn):
    graph.build(conn, graph.LexiconExtractor(), log=lambda *a, **k: None)
    rows = {r["mechanism"]: r for r in graph.converge(conn, perms=50)}
    # hypoxia/anoxia: naked mole-rat (Mammalia) + ocean quahog (Bivalvia) -> 2 species, 2 lineages
    hyp = rows["hypoxia"]
    assert set(hyp["supporting_species"]) == {"naked_mole_rat", "ocean_quahog"}
    assert hyp["n_species"] == 2 and hyp["n_lineages"] == 2 and set(hyp["lineages"]) == {"Bivalvia", "Mammalia"}
    assert 0 < hyp["p_perm"] <= 1 and hyp["perms"] == 50
    assert hyp["species"]["naked_mole_rat"]["support_rate"] == 0.5  # 1 of 2 NMR docs
    assert hyp["species"]["naked_mole_rat"]["best"]["pmid"] == "1000002"
    # oxidative stress is quahog-only -> 1 lineage
    assert rows["oxidative_stress"]["n_lineages"] == 1
    # ranking: more lineages first
    ordered = graph.converge(conn, perms=10)
    assert ordered[0]["n_lineages"] >= ordered[-1]["n_lineages"]


def test_same_class_species_are_one_lineage(conn):
    # add a bowhead doc mentioning hypoxia: Mammalia again -> species 3, lineages still 2
    db.upsert_doc(conn, {"pmid": "4000001", "title": "Bowhead diving", "abstract": "Bowhead whales tolerate hypoxia during prolonged dives, and hypoxia tolerance is associated with high myoglobin.", "retrieved_at": "x", "pub_types": ["Journal Article"]}, "bowhead_whale")
    conn.commit()
    graph.build(conn, graph.LexiconExtractor(), log=lambda *a, **k: None)
    hyp = {r["mechanism"]: r for r in graph.converge(conn, perms=20)}["hypoxia"]
    assert hyp["n_species"] == 3 and hyp["n_lineages"] == 2


def test_reviews_and_negatives_do_not_count_as_support(conn):
    db.upsert_doc(conn, {"pmid": "5000001", "title": "A review", "abstract": "This review summarizes DNA repair in the naked mole-rat and its possible link to longevity.", "retrieved_at": "x", "pub_types": ["Review"]}, "naked_mole_rat")
    db.upsert_doc(conn, {"pmid": "5000002", "title": "Null result", "abstract": "We found no association between DNA repair capacity and lifespan in Hydra polyps.", "retrieved_at": "x", "pub_types": ["Journal Article"]}, "hydra")
    conn.commit()
    graph.build(conn, graph.LexiconExtractor(), log=lambda *a, **k: None)
    dna = {r["mechanism"]: r for r in graph.converge(conn, perms=20)}["dna_repair"]
    assert dna["supporting_species"] == ["rockfish"]  # the review (NMR) and the negative (hydra) add no support
    assert dna["n_review_mentions"] == 1 and dna["n_negative"] == 1
    assert dna["species"]["naked_mole_rat"]["review_pmids"] == ["5000001"] and dna["species"]["hydra"]["negative_pmids"] == ["5000002"]


def test_min_tier_filters_mentions(conn):
    db.upsert_doc(conn, {"pmid": "6000001", "title": "Mention only", "abstract": "Telomeres were measured in the Greenland shark as part of a broader survey of tissues.", "retrieved_at": "x", "pub_types": ["Journal Article"]}, "greenland_shark")
    conn.commit()
    graph.build(conn, graph.LexiconExtractor(), log=lambda *a, **k: None)
    tel_mention = {r["mechanism"]: r for r in graph.converge(conn, min_tier="mention", perms=0)}["telomere"]
    assert tel_mention["n_species"] == 1 and tel_mention["p_perm"] is None
    rows_assoc = {r["mechanism"]: r for r in graph.converge(conn, min_tier="association", perms=0)}
    assert rows_assoc["telomere"]["n_species"] == 0


def test_permutation_is_deterministic(conn):
    graph.build(conn, graph.LexiconExtractor(), log=lambda *a, **k: None)
    a = {r["mechanism"]: r["p_perm"] for r in graph.converge(conn, perms=100, seed=3)}
    b = {r["mechanism"]: r["p_perm"] for r in graph.converge(conn, perms=100, seed=3)}
    assert a == b


def test_mechanism_detail_and_render(conn):
    graph.build(conn, graph.LexiconExtractor(), log=lambda *a, **k: None)
    detail = graph.mechanism_detail(conn, "hypoxia")
    assert {d["species_key"] for d in detail} == {"naked_mole_rat", "ocean_quahog"} and all(d["quote"] for d in detail)
    text = graph.render_convergence(graph.converge(conn, perms=10), min_lineages=2)
    assert "hypoxia" in text and "lineage = taxonomic class" in text
    assert "no mechanism reaches 5" in graph.render_convergence(graph.converge(conn, perms=10), min_lineages=5)


def test_card_gets_convergence_for_mechanisms_it_names(conn):
    graph.build(conn, graph.LexiconExtractor(), log=lambda *a, **k: None)
    rows = graph.card_convergence(conn, "Hypoxia tolerance via metabolic rewiring recurs across the panel")
    assert [r["mechanism"] for r in rows] and "hypoxia" in [r["mechanism"] for r in rows]
    assert rows[0]["lineage_proxy"].startswith("taxonomic class")
    assert graph.card_convergence(conn, "nothing mechanistic here") == []
    card = generate.ask(conn, "hypoxia tolerance", species_keys=["naked_mole_rat", "ocean_quahog"], llm=MockLLM())
    assert isinstance(card.convergence, list)


def test_no_findings_means_no_convergence_field(conn):
    card = generate.ask(conn, "hypoxia tolerance", species_keys=["naked_mole_rat"], llm=MockLLM())
    assert card.convergence is None
