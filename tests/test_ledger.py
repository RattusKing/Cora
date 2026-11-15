import pytest

from cora import generate, ledger, metrics
from cora.llm import MockLLM


def test_feedback_updates_state(conn):
    card = generate.ask(conn, "longevity", llm=MockLLM())
    ledger.feedback(conn, card.id, "accept")
    assert ledger.get_row(conn, card.id)["state"] == "accepted"
    ledger.feedback(conn, card.id, "reject:wrong", reason="quote misread")
    assert ledger.get_row(conn, card.id)["state"] == "archived"
    assert ledger.feedback_counts(conn) == {"accept": 1, "reject:wrong": 1}


def test_archived_cards_are_hidden_not_deleted(conn):
    card = generate.ask(conn, "longevity", llm=MockLLM())
    ledger.feedback(conn, card.id, "reject:uninteresting")
    assert all(r["card"].id != card.id for r in ledger.list_cards(conn))
    assert any(r["card"].id == card.id for r in ledger.list_cards(conn, include_archived=True))
    assert ledger.get(conn, card.id) is not None  # still retrievable


def test_invalid_verdict_and_missing_card(conn):
    card = generate.ask(conn, "longevity", llm=MockLLM())
    with pytest.raises(ValueError):
        ledger.feedback(conn, card.id, "meh")
    with pytest.raises(KeyError):
        ledger.feedback(conn, "nope", "accept")


def test_export_counts_as_leaving_cora(conn):
    card = generate.ask(conn, "longevity", llm=MockLLM())
    assert metrics.compute(conn)["user"]["cards_left_cora"] == 0
    ledger.mark_exported(conn, card.id)
    ledger.mark_expanded(conn, card.id)
    m = metrics.compute(conn)["user"]
    assert m["cards_left_cora"] == 1 and m["cards_expanded"] == 1


def test_days_opened_and_checkin(conn):
    assert metrics.compute(conn)["user"]["days_opened"] == 0
    metrics.record_open(conn)
    metrics.record_open(conn)
    metrics.record_checkin(conn, True, "used a card in a draft")
    m = metrics.compute(conn)["user"]
    assert m["days_opened"] == 1 and m["checkins"][0]["useful"] is True


def test_the_topic_filter_is_applied_before_the_limit(conn):
    """Ranking first and filtering second would hide a topic's cards behind other topics'."""
    from cora import db as _db
    from cora.llm import MockLLM

    # six aging cards, then one hibernation card ranked below them
    for i in range(6):
        generate.ask(conn, f"longevity lifespan {i}", llm=MockLLM(max_items=2))
    _db.upsert_doc(conn, {"pmid": "6600001", "title": "Bear muscle",
                          "abstract": "Muscle mass is preserved through hibernation in bears, and torpor is associated with metabolic depression.",
                          "retrieved_at": "x", "pub_types": ["Journal Article"]}, "bear", topic_key="hibernation")
    conn.commit()
    hib = generate.ask(conn, "muscle preserved through torpor", llm=MockLLM(), topic_keys=["hibernation"])

    assert len(ledger.list_cards(conn, limit=3)) == 3  # the limit still limits
    rows = ledger.list_cards(conn, limit=3, topic_keys=["hibernation"])
    assert [r["card"].id for r in rows] == [hib.id], "the hibernation card fell outside the global top 3"
