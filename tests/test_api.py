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


def test_ask_no_evidence_is_404(client):
    r = client.post("/api/ask", json={"query": "zzzz qqqq", "mock": True})
    assert r.status_code == 404
