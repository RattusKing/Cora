import json

import pytest

from cora import config, db, graph, ingest, topics


def test_every_builtin_topic_is_internally_consistent():
    for key, t in topics.BUILTIN.items():
        assert t.key == key and t.name and t.question and t.terms
        assert t.species, f"{key} has no species"
        assert t.mechanisms, f"{key} has no mechanisms"
        assert t.lineage_rank in topics.LINEAGE_RANKS
        assert t.status in ("ready", "beta")
        unknown_species = [s for s in t.species if s not in config.SPECIES]
        assert unknown_species == [], f"{key}: unknown species {unknown_species}"
        unknown_mechs = [m for m in t.mechanisms if m not in graph.MECHANISMS]
        assert unknown_mechs == [], f"{key}: unknown mechanisms {unknown_mechs}"
        for s in t.species:
            assert config.SPECIES[s].class_ and config.SPECIES[s].phylum, f"{s} has no lineage"


def test_queries_pair_species_with_topic_terms():
    nmr = config.SPECIES["naked_mole_rat"]
    aging = ingest.build_query(nmr, topics.BUILTIN["aging"])
    cancer = ingest.build_query(nmr, topics.BUILTIN["cancer_resistance"])
    assert "Heterocephalus glaber" in aging and "longevity" in aging
    assert "Heterocephalus glaber" in cancer and "neoplas*" in cancer
    assert aging != cancer  # the same species, two different corpora
    # binomials, never bare ambiguous common names
    assert "quahog" not in ingest.build_query(config.SPECIES["ocean_quahog"], topics.BUILTIN["aging"]).lower()
    bats = config.SPECIES["bats"].query
    assert "Chiroptera[MeSH Terms]" in bats and '"bats"' not in bats  # "bats" maps to BAT


def test_a_species_in_two_topics_is_queried_once_per_topic():
    assert "naked_mole_rat" in topics.BUILTIN["aging"].species
    assert "naked_mole_rat" in topics.BUILTIN["hypoxia"].species
    pairs = topics.pairs(["aging", "hypoxia"])
    assert ("aging", "naked_mole_rat") in pairs and ("hypoxia", "naked_mole_rat") in pairs
    # species_for de-duplicates, pairs do not
    assert topics.species_for(["aging", "hypoxia"]).count("naked_mole_rat") == 1


def test_narrowing_applies_to_one_species_only():
    regen = topics.BUILTIN["regeneration"]
    zf = ingest.build_query(config.SPECIES["zebrafish"], regen)
    ax = ingest.build_query(config.SPECIES["axolotl"], regen)
    assert "regenerat*[Title]" in zf and "regenerat*[Title]" not in ax


def test_lineage_rank_of_a_mixed_run_is_the_coarsest(conn):
    """Combining topics must never report MORE independent lineages than one alone."""
    assert topics.lineage_rank_for(["aging"], conn) == "class"
    coarse = topics.custom_topic(
        key="phylum_topic", name="n", question="q", terms="x[Title/Abstract]",
        species=["hydra"], mechanisms=["stem_cells"], lineage_rank="phylum",
    )
    db.save_custom_topic(conn, coarse.key, topics.to_json(coarse))
    assert topics.lineage_rank_for(["phylum_topic"], conn) == "phylum"
    assert topics.lineage_rank_for(["aging", "phylum_topic"], conn) == "phylum"  # coarsest wins
    # and the coarser rank really does collapse lineages
    assert graph._lineage("rockfish", "class") != graph._lineage("greenland_shark", "class")
    assert graph._lineage("rockfish", "phylum") == graph._lineage("greenland_shark", "phylum")


def test_resolve_falls_back_to_enabled_then_default(conn):
    assert topics.resolve(None, conn) == ["aging"]  # seeded by the fixture via doc_topic
    db.set_enabled_topics(conn, ["hypoxia", "hibernation"])
    assert topics.resolve(None, conn) == ["hypoxia", "hibernation"]
    assert topics.resolve(["aging"], conn) == ["aging"]  # an explicit request wins
    assert topics.resolve(["aging", "aging"], conn) == ["aging"]  # de-duplicated
    with pytest.raises(topics.TopicError):
        topics.resolve(["nope"], conn)


def test_enabled_topics_survive_a_reopen(db_file):
    c = db.connect(db_file)
    db.set_enabled_topics(c, ["regeneration"])
    c.close()
    c2 = db.connect(db_file)
    assert db.enabled_topics(c2) == ["regeneration"]
    c2.close()


def test_a_pre_topics_database_is_migrated_to_aging(tmp_path):
    """An existing install must keep its corpus, its cards and its ingest state."""
    path = tmp_path / "legacy.db"
    c = db.connect(path)
    # simulate the pre-topics state: docs and doc_species, no doc_topic rows
    c.execute("DELETE FROM doc_topic")
    c.execute("DELETE FROM settings")
    c.execute("DELETE FROM ingest_state")
    db.upsert_doc(c, {"pmid": "9000001", "title": "t", "abstract": "a" * 40, "retrieved_at": "x"}, "naked_mole_rat")
    c.execute("DELETE FROM doc_topic")
    c.commit()
    c.close()

    c2 = db.connect(path)  # the migration runs here
    assert db.count_docs_by_topic(c2) == {"aging": 1}
    assert db.enabled_topics(c2) == ["aging"]
    assert ("aging", "naked_mole_rat") in db.ingested_pairs(c2)  # not re-downloaded
    assert db.get_docs(c2, topic_keys=["aging"])[0]["pmid"] == "9000001"
    c2.close()


def test_custom_topic_roundtrip_and_validation(conn):
    t = topics.custom_topic(
        key="venom_analgesia", name="Venom-derived painkillers",
        question="Which venom peptides block pain without opioids?",
        terms='(venom[Title/Abstract] OR analgesi*[Title/Abstract])',
        species=["bats"], mechanisms=["ion_channel_arrest", "neuroprotection"],
    )
    assert t.builtin is False and t.status == "beta" and t.trait_source is None
    blob = topics.to_json(t)
    assert topics.from_json(blob) == t

    db.save_custom_topic(conn, t.key, blob)
    registry = topics.all_topics(conn)
    assert registry["venom_analgesia"].name == t.name
    assert topics.resolve(["venom_analgesia"], conn) == ["venom_analgesia"]
    assert topics.get("venom_analgesia", conn).terms == t.terms
    assert db.delete_custom_topic(conn, "venom_analgesia") is True
    assert "venom_analgesia" not in topics.all_topics(conn)


def test_custom_topic_rejects_bad_input():
    bad = [
        {"key": "Aging"},                      # uppercase
        {"key": "a"},                          # too short
        {"key": "aging"},                      # collides with a built-in
        {"key": "ok_key", "name": ""},         # no name
        {"key": "ok_key", "terms": ""},        # no terms
        {"key": "ok_key", "species": []},      # no species
        {"key": "ok_key", "lineage_rank": "order"},  # unknown rank
    ]
    for override in bad:
        kwargs = dict(
            key="ok_key", name="n", question="q", terms="x[Title/Abstract]",
            species=["hydra"], mechanisms=["stem_cells"],
        )
        kwargs.update(override)
        with pytest.raises(topics.TopicError):
            topics.custom_topic(**kwargs)
    with pytest.raises(topics.TopicError):
        topics.from_json(json.dumps({"key": "ok_key", "name": "n"}))  # missing fields


def test_a_malformed_stored_topic_does_not_break_the_registry(conn):
    db.save_custom_topic(conn, "broken", '{"key": "broken"}')
    registry = topics.all_topics(conn)
    assert "broken" not in registry and "aging" in registry


def test_picker_rendering_and_selection():
    registry = topics.BUILTIN
    text = topics.render_picker(registry, enabled=["aging"], counts={"aging": 654})
    assert "1. Aging & longevity" in text and "654 abstracts" in text
    assert "[x] 1." in text and "[ ] 2." in text  # enabled is marked
    assert "beta lexicon" in text  # the four new topics are honest about it
    keys = list(registry)
    assert topics.parse_selection("1", registry) == [keys[0]]
    assert topics.parse_selection("1,3", registry) == [keys[0], keys[2]]
    assert topics.parse_selection(" 2 4 ", registry) == [keys[1], keys[3]]
    assert topics.parse_selection("all", registry) == keys
    assert topics.parse_selection("regeneration", registry) == ["regeneration"]
    assert topics.parse_selection("1,1", registry) == [keys[0]]
    for bad in ("", "0", "99", "nope", "   "):
        with pytest.raises(topics.TopicError):
            topics.parse_selection(bad, registry)


def test_topics_cli_is_wired():
    from cora.cli import build_parser

    p = build_parser()
    a = p.parse_args(["topics"])
    assert a.cmd == "topics" and a.set is None and not a.add
    assert p.parse_args(["topics", "--set", "aging", "hypoxia"]).set == ["aging", "hypoxia"]
    assert p.parse_args(["topics", "--remove", "mine"]).remove == ["mine"]
    assert p.parse_args(["topics", "--export", "aging"]).export == "aging"
    assert p.parse_args(["topics", "-i"]).interactive is True
