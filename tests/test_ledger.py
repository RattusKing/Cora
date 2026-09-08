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
