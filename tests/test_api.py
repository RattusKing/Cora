import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from cora.api import create_app  # noqa: E402


@pytest.fixture
def client(db_file):
    return TestClient(create_app(db_path=str(db_file)))


def test_index_and_species(client):
    assert client.get("/").status_code == 200
    s = client.get("/api/species").json()
    assert {x["key"] for x in s["species"]} >= {"naked_mole_rat", "ocean_quahog", "rockfish"}
    assert s["species"][0]["docs"] >= 0


def test_ask_ledger_feedback_export_metrics(client):
    r = client.post("/api/ask", json={"query": "longevity lifespan", "mock": True})
    assert r.status_code == 200, r.text
    card = r.json()["card"]
    assert card["gate"]["groundedness"] == 1.0
    assert card["pattern_check"]["passed"] and card["pattern_check"]["judged"]

    r2 = client.post("/api/ask", json={"query": "longevity lifespan", "mock": True, "judge": False})
    assert r2.status_code == 200 and not r2.json()["card"]["pattern_check"]["judged"]

    rows = client.get("/api/ledger").json()
    assert rows and rows[0]["card"]["id"] == card["id"] and "tight" in rows[0]

    assert client.get(f"/api/card/{card['id']}").status_code == 200
    assert client.post(f"/api/card/{card['id']}/feedback", json={"verdict": "dig_deeper"}).json() == {"ok": True}
    assert client.post(f"/api/card/{card['id']}/feedback", json={"verdict": "bogus"}).status_code == 400
    draft = client.post(f"/api/card/{card['id']}/export").json()["draft"]
    assert "PMID" in draft

    m = client.get("/api/metrics").json()
    assert m["user"]["cards_left_cora"] == 1 and m["user"]["cards_expanded"] == 1
    assert client.post("/api/checkin", json={"useful": False}).json() == {"ok": True}


def test_recheck_endpoints_with_injected_runner(db_file):
    from cora import recheck

    calls = []

    def fake_runner(conn):
        calls.append(1)
        report = recheck.Report(date="2026-09-20", species={"naked_mole_rat": {"n_pmids": 2, "new": 0, "gone": 0, "stored": 0}}, n_new=0, n_gone=0, touches=[], flags=[], n_cards_live=0, n_cards_touched=0)
        from cora import db as _db
        _db.log_recheck(conn, 0, 0, 0, 0, report.model_dump_json())
        return report

    client = TestClient(create_app(db_path=str(db_file), recheck_runner=fake_runner))
    assert client.get("/api/recheck/last").json()["report"] is None
    r = client.post("/api/recheck")
    assert r.status_code == 200 and calls == [1] and "nothing changed" in r.json()["text"]
    last = client.get("/api/recheck/last").json()
    assert last["report"]["date"] == "2026-09-20" and "nothing changed" in last["text"]
    assert client.get("/api/metrics").json()["recheck"]["runs"] == 1


def test_ask_no_evidence_is_404(client):
    r = client.post("/api/ask", json={"query": "zzzz qqqq", "mock": True})
    assert r.status_code == 404


def test_topics_endpoint_lists_and_sets(client):
    t = client.get("/api/topics").json()
    assert t["enabled"] == ["aging"]
    keys = {x["key"] for x in t["topics"]}
    assert {"aging", "regeneration", "cancer_resistance", "hypoxia", "hibernation"} <= keys
    aging = next(x for x in t["topics"] if x["key"] == "aging")
    assert aging["enabled"] and aging["docs"] == 6 and aging["status"] == "ready"
    regen = next(x for x in t["topics"] if x["key"] == "regeneration")
    assert not regen["enabled"] and regen["docs"] == 0 and regen["ingested_pairs"] == 0

    r = client.post("/api/topics", json={"topics": ["hypoxia"]})
    assert r.status_code == 200 and r.json()["enabled"] == ["hypoxia"]
    assert r.json()["not_ingested"] and all("hypoxia/" in p for p in r.json()["not_ingested"])
    assert client.get("/api/topics").json()["enabled"] == ["hypoxia"]
    assert client.post("/api/topics", json={"topics": ["nope"]}).status_code == 400


def test_species_and_ledger_follow_the_topic(client):
    s = client.get("/api/species?topic=aging").json()
    assert s["topics"] == ["aging"] and {x["key"] for x in s["species"]} == set(s["default"])
    assert len(s["species"]) == 8
    hib = client.get("/api/species?topic=hibernation").json()
    assert "ground_squirrel" in {x["key"] for x in hib["species"]}
    assert client.get("/api/species?topic=nope").status_code == 400

    card = client.post("/api/ask", json={"query": "longevity lifespan", "mock": True}).json()["card"]
    assert card["topics"] == ["aging"]
    assert [r["card"]["id"] for r in client.get("/api/ledger?topic=aging").json()] == [card["id"]]
    assert client.get("/api/ledger?topic=hibernation").json() == []


def test_ask_in_a_topic_with_no_corpus_is_404(client):
    r = client.post("/api/ask", json={"query": "blastema", "topics": ["regeneration"], "mock": True})
    assert r.status_code == 404 and "regeneration" in r.json()["detail"]


def test_metrics_corpus_follows_the_topic(client):
    m = client.get("/api/metrics").json()
    assert m["corpus"]["topics"] == ["aging"]
    assert m["corpus"]["total_docs"] == 6 == m["corpus"]["total_docs_all_topics"]
    assert m["corpus"]["docs_by_topic"] == {"aging": 6}
    scoped = client.get("/api/metrics?topic=hibernation").json()
    assert scoped["corpus"]["total_docs"] == 0 and scoped["corpus"]["total_docs_all_topics"] == 6
    assert scoped["corpus"]["docs_by_species"] == {}
    # behaviour metrics stay global
    assert scoped["gate"]["items"] == m["gate"]["items"]
