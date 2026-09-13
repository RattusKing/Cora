import json

import pytest

from cora import config, db, generate, ingest, ledger, metrics, recheck
from cora.llm import MockLLM

NEW_NMR = {
    "pmid": "1000009", "pub_year": 2026,
    "title": "Hyaluronan synthase activity and cancer resistance in naked mole-rat fibroblasts",
    "abstract": "We measured hyaluronan production in naked mole-rat fibroblasts and found that high-molecular-mass hyaluronan is associated with resistance to malignant transformation. Cancer resistance in this species may depend on extracellular matrix composition.",
}
NEW_ROCKFISH = {
    "pmid": "3000009", "pub_year": 2026,
    "title": "Otolith chemistry of deep-dwelling Sebastes",
    "abstract": "Otolith trace-element chemistry in Sebastes species tracks depth and temperature across the lifespan. These signatures allow reconstruction of habitat use in long-lived rockfishes.",
}
SPECIES_PMIDS = {"naked_mole_rat": ["1000001", "1000002"], "ocean_quahog": ["2000001", "2000002"], "rockfish": ["3000001", "3000002"]}


class Fake:
    """Injectable PubMed: `search` returns PMIDs per species term; `fetch` returns docs/flags."""

    def __init__(self, new=None, flags=None):
        self.new = new or {}  # species_key -> [doc]
        self.flags = flags or {}  # pmid -> [flag]
        self.fetched: list[list[str]] = []

    def search(self, term):
        for key, sp in config.SPECIES.items():
            if sp.query in term:
                return SPECIES_PMIDS.get(key, []) + [d["pmid"] for d in self.new.get(key, [])]
        return []

    def fetch(self, pmids, tag, keep_empty=False, log=print):
        self.fetched.append(list(pmids))
        if tag == "flags":
            return [{"pmid": p, "title": "", "abstract": "", "flags": self.flags.get(p, [])} for p in pmids]
        docs = [dict(d) for d in self.new.get(tag, []) if d["pmid"] in pmids]
        for d in docs:
            d["retrieved_at"] = "2026-09-20T00:00:00+00:00"
            d["flags"] = self.flags.get(d["pmid"], [])
        return docs


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path / "snapshots")
    return tmp_path


def _card(conn, query="hyaluronan cancer resistance", species=("naked_mole_rat",)):
    return generate.ask(conn, query, species_keys=list(species), llm=MockLLM())


def test_first_recheck_finds_new_abstracts_touches_cards_and_flags_sources(conn, paths):
    card = _card(conn)
    other = _card(conn, query="otolith age validation", species=("rockfish",))
    fake = Fake(new={"naked_mole_rat": [NEW_NMR], "rockfish": [NEW_ROCKFISH]}, flags={"1000001": ["ErratumIn:1000099"]})

    report = recheck.run(conn, search_fn=fake.search, fetch_fn=fake.fetch, log=lambda *a, **k: None)

    assert report.n_new == 2 and report.n_gone == 0
    assert report.species["naked_mole_rat"]["new"] == 1 and report.species["rockfish"]["new"] == 1
    # the new NMR abstract touches the NMR card, not the rockfish card; the rockfish one touches the rockfish card
    touched = {(t.card_id, t.pmid) for t in report.touches}
    assert (card.id, "1000009") in touched and (other.id, "1000009") not in touched
    assert (other.id, "3000009") in touched
    nmr_touch = next(t for t in report.touches if t.card_id == card.id)
    assert any(r.startswith("ranks #") for r in nmr_touch.reasons)
    # the new docs are stored with their species
    assert db.docs_by_pmid(conn, ["1000009"])["1000009"]["species"] == ["naked_mole_rat"]
    # the erratum on a cited source is flagged for the card that cites it
    assert any(f.card_id == card.id and f.pmid == "1000001" and f.flags == ["ErratumIn:1000099"] for f in report.flags)
    assert db.docs_by_pmid(conn, ["1000001"])["1000001"]["flags"] == ["ErratumIn:1000099"]
    # only cited PMIDs were re-fetched for flags
    assert any(set(batch) <= {"1000001", "1000002", "3000001", "3000002"} and "1000001" in batch for batch in fake.fetched)
    # touches persist as unseen updates and the ledger shows them
    rows = {r["card"].id: r for r in ledger.list_cards(conn)}
    assert rows[card.id]["new_evidence"] == 2  # new abstract + source flag
    # manifest and log updated
    m = json.loads((paths / "manifest.json").read_text())
    assert "1000009" in m["species"]["naked_mole_rat"]["pmids"] and m["runs"][-1]["kind"] == "recheck"
    assert recheck.last_report(conn).n_new == 2
    assert metrics.compute(conn)["recheck"]["runs"] == 1


def test_second_recheck_with_nothing_new_reports_nothing_changed(conn, paths):
    card = _card(conn)
    fake = Fake(new={"naked_mole_rat": [NEW_NMR]}, flags={"1000001": ["ErratumIn:1000099"]})
    recheck.run(conn, search_fn=fake.search, fetch_fn=fake.fetch, log=lambda *a, **k: None)
    second = recheck.run(conn, search_fn=fake.search, fetch_fn=fake.fetch, log=lambda *a, **k: None)
    assert second.n_new == 0 and second.touches == [] and second.flags == []  # the same flag is not re-reported
    text = recheck.render_report(second, {card.id: card})
    assert "nothing changed" in text and "0 new abstracts" in text


def test_render_report_is_templated(conn, paths):
    card = _card(conn)
    fake = Fake(new={"naked_mole_rat": [NEW_NMR]}, flags={"1000001": ["RetractionIn:1000098"]})
    report = recheck.run(conn, search_fn=fake.search, fetch_fn=fake.fetch, log=lambda *a, **k: None)
    text = recheck.render_report(report, {card.id: card})
    assert text.startswith("recheck ") and "1 new abstract (" in text
    assert "cards with new evidence: 1" in text and "+ PMID 1000009 (2026; naked_mole_rat)" in text
    assert "source flags: 1" in text and "RetractionIn:1000098 - re-read before citing" in text
    assert "unchanged: 0 cards" in text


def test_gone_pmids_are_counted_not_deleted(conn, paths):
    _card(conn)

    def search_missing_one(term):
        pm = Fake().search(term)
        return [p for p in pm if p != "1000002"]

    report = recheck.run(conn, search_fn=search_missing_one, fetch_fn=Fake().fetch, log=lambda *a, **k: None)
    assert report.n_gone == 1 and report.species["naked_mole_rat"]["gone"] == 1
    assert db.docs_by_pmid(conn, ["1000002"])  # still stored


def test_match_by_pattern_terms_without_query_rank():
    from cora.card import Card, CardDraft, DeskCheck, EvidenceBand, GateSummary, HumanLever

    draft = CardDraft(
        pattern="High-molecular-mass hyaluronan is associated with cancer resistance in naked mole-rat fibroblasts",
        evidence=[], objection="-", human_lever=HumanLever(), desk_check=DeskCheck(step="-"), species_support=[],
    )
    card = Card(
        id="c1", created_at="x", query="zzzz qqqq", species_scope=["naked_mole_rat"], draft=draft,
        gate=GateSummary(n_items=0, n_passed=0, n_failed=0, failed=[], fabrication_rate=0.0, groundedness=0.0),
        evidence_band=EvidenceBand(n_verified_quotes=0, n_pmids=0, n_species=0, label="-"),
        why_seeing_this=[], flags=[], dedupe_key="k",
    )
    doc = dict(NEW_NMR, species=["naked_mole_rat"])
    touches = recheck.match_new_docs([card], [doc])
    assert len(touches) == 1 and len(touches[0].reasons) == 1
    assert touches[0].reasons[0].startswith("shares pattern terms: ") and "hyaluronan" in touches[0].reasons[0]
    # out-of-scope species never matches
    assert recheck.match_new_docs([card], [dict(NEW_ROCKFISH, species=["rockfish"])]) == []


def test_rank_is_against_the_whole_species_corpus_not_just_new_docs(conn):
    from cora.card import Card, CardDraft, DeskCheck, EvidenceBand, GateSummary, HumanLever

    draft = CardDraft(pattern="x y z", evidence=[], objection="-", human_lever=HumanLever(), desk_check=DeskCheck(step="-"), species_support=[])
    card = Card(
        id="c2", created_at="x", query="hyaluronan cancer resistance", species_scope=["naked_mole_rat"], draft=draft,
        gate=GateSummary(n_items=0, n_passed=0, n_failed=0, failed=[], fabrication_rate=0.0, groundedness=0.0),
        evidence_band=EvidenceBand(n_verified_quotes=0, n_pmids=0, n_species=0, label="-"), why_seeing_this=[], flags=[], dedupe_key="k",
    )
    # a new abstract with one weak overlapping term ("resistance") must not rank when the
    # corpus already holds a far better match for the query
    weak = {"pmid": "1000010", "title": "Thermal resistance of burrow air", "abstract": "Burrow air temperature and humidity were recorded across seasons; resistance to heat loss was estimated.", "species": ["naked_mole_rat"]}
    corpus = db.get_docs(conn, ["naked_mole_rat"]) + [weak]
    assert recheck.match_new_docs([card], [weak], k=1, corpus_docs=corpus) == []
    # ... but a strong match does rank
    strong = dict(NEW_NMR, species=["naked_mole_rat"])
    corpus = db.get_docs(conn, ["naked_mole_rat"]) + [strong]
    touches = recheck.match_new_docs([card], [strong], k=2, corpus_docs=corpus)
    assert touches and any(r.startswith("ranks #") for r in touches[0].reasons)


def test_known_pmids_falls_back_to_db_without_manifest(conn):
    assert ingest.known_pmids(conn, "rockfish", None) == {"3000001", "3000002"}
    assert ingest.known_pmids(conn, "rockfish", {"species": {"rockfish": {"pmids": ["x"]}}}) == {"x"}


def test_show_marks_updates_seen(conn, paths):
    card = _card(conn)
    fake = Fake(new={"naked_mole_rat": [NEW_NMR]})
    recheck.run(conn, search_fn=fake.search, fetch_fn=fake.fetch, log=lambda *a, **k: None)
    assert ledger.list_cards(conn)[0]["new_evidence"] == 1
    ledger.mark_expanded(conn, card.id)
    assert ledger.list_cards(conn)[0]["new_evidence"] == 0
    assert db.card_updates(conn, card.id)[0]["seen"] == 1
