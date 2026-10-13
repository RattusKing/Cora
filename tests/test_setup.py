import pytest

from cora import anage, config, db, graph, setup

ABSTRACTS = {
    "naked_mole_rat": [{"pmid": "8100001", "title": "NMR DNA repair", "abstract": "Naked mole-rat cells show enhanced DNA repair capacity associated with their exceptional longevity. These results suggest that genome maintenance may contribute to cancer resistance.", "pub_types": ["Journal Article"]}],
    "ocean_quahog": [{"pmid": "8200001", "title": "Quahog proteostasis", "abstract": "Arctica islandica proteins resist oxidative damage, and proteostasis is associated with extreme longevity in this bivalve. Sample sizes were modest.", "pub_types": ["Journal Article"]}],
}
ANAGE_TXT = (
    "HAGRID\tKingdom\tPhylum\tClass\tOrder\tFamily\tGenus\tSpecies\tCommon name\tMaximum longevity (yrs)\tData quality\tSample size\tSpecimen origin\tBody mass (g)\n"
    "00001\tAnimalia\tChordata\tMammalia\tRodentia\tBathyergidae\tHeterocephalus\tglaber\tNaked mole-rat\t31\thigh\tlarge\tcaptivity\t35\n"
    "00002\tAnimalia\tMollusca\tBivalvia\tVeneroida\tArcticidae\tArctica\tislandica\tOcean quahog\t507\tacceptable\tmedium\twild\t\n"
)


class FakePubMed:
    def __init__(self, fail=False):
        self.fail = fail
        self.searched: list[str] = []

    def search(self, term):
        if self.fail:
            raise RuntimeError("E-utilities unreachable")
        for key, sp in config.SPECIES.items():
            if sp.query in term:
                self.searched.append(key)
                return [d["pmid"] for d in ABSTRACTS.get(key, [])]
        return []

    def fetch(self, pmids, tag, keep_empty=False, log=print):
        return [dict(d, retrieved_at="2026-10-13T00:00:00+00:00", flags=[]) for d in ABSTRACTS.get(tag, []) if d["pmid"] in pmids]


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "cora.db")
    conn = db.connect(tmp_path / "cora.db")
    yield conn, tmp_path
    conn.close()


def _anage_file(tmp_path):
    p = tmp_path / "anage_data.txt"
    p.write_text(ANAGE_TXT, encoding="utf-8")
    return lambda: p


def _statuses(res):
    return {s.name: s.status for s in res.steps}


def test_fresh_install_runs_every_step(fresh):
    conn, tmp = fresh
    pm = FakePubMed()
    keys = ["naked_mole_rat", "ocean_quahog"]
    res = setup.run(conn, keys, search_fn=pm.search, fetch_fn=pm.fetch, anage_download=_anage_file(tmp), log=lambda *a, **k: None)
    assert res.ok and _statuses(res) == {"init": "ok", "ingest": "ok", "anage": "ok", "graph": "ok", "canary": "ok"}
    assert db.count_docs_by_species(conn) == {"naked_mole_rat": 1, "ocean_quahog": 1}
    assert anage.loaded_count(conn) == 2
    assert graph.findings_count(conn) > 0
    assert "2 of 8 panel species matched" in next(s.detail for s in res.steps if s.name == "anage")
    text = setup.render(res, conn)
    assert "ready. try:" in text and "corpus: 2 unique abstracts across 2 species" in text
    assert "2 new unique abstracts" in next(s.detail for s in res.steps if s.name == "ingest")


def test_paper_about_two_species_is_counted_once(fresh):
    conn, tmp = fresh
    shared = {"pmid": "8300001", "title": "Two species", "abstract": "Naked mole-rats and bowhead whales both show enhanced DNA repair associated with longevity, and these shared traits may contribute to cancer resistance.", "pub_types": ["Journal Article"]}

    class Shared(FakePubMed):
        def search(self, term):
            return ["8300001"] if (config.SPECIES["naked_mole_rat"].query in term or config.SPECIES["bowhead_whale"].query in term) else []

        def fetch(self, pmids, tag, keep_empty=False, log=print):
            return [dict(shared, retrieved_at="x", flags=[])]

    pm = Shared()
    res = setup.run(conn, ["naked_mole_rat", "bowhead_whale"], skip={"anage"}, search_fn=pm.search, fetch_fn=pm.fetch, log=lambda *a, **k: None)
    assert db.count_docs_by_species(conn) == {"naked_mole_rat": 1, "bowhead_whale": 1}
    assert db.count_docs(conn) == 1
    assert "1 new unique abstracts" in next(s.detail for s in res.steps if s.name == "ingest")
    assert "corpus: 1 unique abstracts across 2 species" in setup.render(res, conn)


def test_rerun_skips_what_is_already_there(fresh):
    conn, tmp = fresh
    pm = FakePubMed()
    keys = ["naked_mole_rat", "ocean_quahog"]
    setup.run(conn, keys, search_fn=pm.search, fetch_fn=pm.fetch, anage_download=_anage_file(tmp), log=lambda *a, **k: None)
    pm.searched.clear()

    def no_download():
        raise AssertionError("AnAge must not be re-downloaded on a re-run")

    res = setup.run(conn, keys, search_fn=pm.search, fetch_fn=pm.fetch, anage_download=no_download, log=lambda *a, **k: None)
    assert res.ok and _statuses(res)["ingest"] == "skipped" and _statuses(res)["anage"] == "skipped"
    assert pm.searched == []  # no PubMed calls
    assert _statuses(res)["graph"] == "ok" and _statuses(res)["canary"] == "ok"  # always re-run


def test_refresh_and_partial_ingest(fresh):
    conn, tmp = fresh
    pm = FakePubMed()
    setup.run(conn, ["naked_mole_rat"], search_fn=pm.search, fetch_fn=pm.fetch, anage_download=_anage_file(tmp), log=lambda *a, **k: None)
    pm.searched.clear()
    # adding a species ingests only the new one
    setup.run(conn, ["naked_mole_rat", "ocean_quahog"], search_fn=pm.search, fetch_fn=pm.fetch, anage_download=_anage_file(tmp), log=lambda *a, **k: None)
    assert pm.searched == ["ocean_quahog"]
    pm.searched.clear()
    setup.run(conn, ["naked_mole_rat", "ocean_quahog"], refresh=True, search_fn=pm.search, fetch_fn=pm.fetch, anage_download=_anage_file(tmp), log=lambda *a, **k: None)
    assert pm.searched == ["naked_mole_rat", "ocean_quahog"]


def test_network_failure_is_reported_and_dependent_steps_skip(fresh):
    conn, tmp = fresh

    def broken_download():
        raise OSError("HAGR unreachable")

    pm = FakePubMed(fail=True)
    res = setup.run(conn, ["naked_mole_rat"], search_fn=pm.search, fetch_fn=pm.fetch, anage_download=broken_download, log=lambda *a, **k: None)
    st = _statuses(res)
    assert not res.ok and st["ingest"] == "failed" and st["anage"] == "failed"
    assert st["graph"] == "skipped" and st["canary"] == "skipped"
    assert "E-utilities unreachable" in next(s.detail for s in res.steps if s.name == "ingest")
    assert "re-run `cora setup`" in setup.render(res, conn)


def test_skip_flags(fresh):
    conn, tmp = fresh
    pm = FakePubMed()
    res = setup.run(conn, ["naked_mole_rat"], skip={"anage", "graph"}, search_fn=pm.search, fetch_fn=pm.fetch, log=lambda *a, **k: None)
    st = _statuses(res)
    assert res.ok and st["anage"] == "skipped" and st["graph"] == "skipped" and st["ingest"] == "ok"
    assert anage.loaded_count(conn) == 0 and graph.findings_count(conn) == 0


def test_cli_setup_is_wired():
    from cora.cli import build_parser

    args = build_parser().parse_args(["setup", "--species", "naked_mole_rat", "--skip", "anage", "--refresh"])
    assert args.cmd == "setup" and args.species == ["naked_mole_rat"] and args.skip == ["anage"] and args.refresh
