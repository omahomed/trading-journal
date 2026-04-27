"""Tests for holdings_source plumbing through /api/journal/edit.

Same architecture as nlv_source (migration 013): the endpoint reads the
field from the request body, validates against the allowed enum-equivalent
values, and forwards it to db.save_journal_entry. These tests stub
save_journal_entry to capture the dict it would have written and assert
the field values + independence from nlv_source.

Real-DB persistence (the trip through psycopg into the trading_journal
table with the CHECK constraint) is exercised by the existing
DATABASE_URL-gated integration tests in tests/test_manual_price_override.py
that share the same skipif decorator.
"""
from __future__ import annotations

from typing import Any

import pytest


_TEST_SECRET = "test-secret-key-for-pytest-only-not-prod"
_TEST_USER_ID = "00000000-0000-4000-8000-000000000000"


def _make_auth_headers() -> dict:
    """Generate a valid bearer header for the test JWT secret."""
    import jwt
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient + the auth shim already documented in test_ibkr_nav."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    from fastapi.testclient import TestClient
    import api.main as main

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)
    if hasattr(main.limiter, "enabled"):
        original = main.limiter.enabled
        main.limiter.enabled = False
    else:
        original = None

    c = TestClient(main.app, headers=_make_auth_headers())
    try:
        yield c
    finally:
        if original is not None:
            main.limiter.enabled = original


@pytest.fixture
def captured_save(monkeypatch):
    """Stub db.save_journal_entry and capture the journal_entry dict.

    The endpoint also reads existing rows via db.load_journal — return an
    empty DataFrame so the merge path treats every test as a fresh insert,
    which keeps the captured dict purely a function of the request body.
    """
    import api.main as main
    import pandas as pd

    captured: dict[str, Any] = {}

    def fake_save(journal_entry):
        captured.update(journal_entry)
        return 42  # any plausible row id

    monkeypatch.setattr(main.db, "save_journal_entry", fake_save)
    monkeypatch.setattr(main.db, "load_journal", lambda *a, **kw: pd.DataFrame())
    # Stub auto-compute helpers so they don't reach for real DB / yfinance
    monkeypatch.setattr(main, "_compute_cycle_state", lambda *a, **kw: "")
    monkeypatch.setattr(main, "_compute_ticker_atr_pct", lambda *a, **kw: 0.0)
    monkeypatch.setattr(main, "_compute_portfolio_heat", lambda *a, **kw: 0.0)
    return captured


def _post_journal_edit(client, **overrides):
    """Send a minimal valid /api/journal/edit body, with overrides merged in."""
    body = {
        "portfolio": "CanSlim",
        "day": "2026-04-24",
        "end_nlv": 486630.39,
        "beg_nlv": 487703.85,
        "cash_change": 0,
        "pct_invested": 0,
        "spy": 715.14,
        "nasdaq": 24880.21,
        "score": 5,
        "highlights": "{}",
        "mistakes": "",
        "market_window": "",
        **overrides,
    }
    return client.post("/api/journal/edit", json=body)


# ---------------------------------------------------------------------------
# Default behavior — pre-IBKR clients keep working
# ---------------------------------------------------------------------------


def test_holdings_source_defaults_to_manual_when_omitted(client, captured_save):
    """Older clients that don't send holdings_source must save as 'manual'.
    Same backwards-compat contract as nlv_source did at migration 013."""
    r = _post_journal_edit(client)  # no holdings_source / nlv_source
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert captured_save["holdings_source"] == "manual"
    assert captured_save["nlv_source"] == "manual"


def test_holdings_source_invalid_value_collapses_to_manual(client, captured_save):
    """Anything outside the three allowed values must be coerced to 'manual'
    rather than reaching the DB and tripping the CHECK constraint."""
    r = _post_journal_edit(client, holdings_source="garbage_value")
    assert r.status_code == 200
    assert captured_save["holdings_source"] == "manual"


# ---------------------------------------------------------------------------
# Forwarding — the IBKR auto-fill flow
# ---------------------------------------------------------------------------


def test_holdings_source_ibkr_auto_round_trips(client, captured_save):
    """Frontend sets holdings_source='ibkr_auto' on a clean IBKR pull.
    Endpoint must persist that value verbatim."""
    r = _post_journal_edit(client, holdings_source="ibkr_auto", nlv_source="ibkr_auto")
    assert r.status_code == 200
    assert captured_save["holdings_source"] == "ibkr_auto"
    assert captured_save["nlv_source"] == "ibkr_auto"


def test_holdings_source_ibkr_override_round_trips(client, captured_save):
    """User edited the auto-filled value before save — frontend sends
    'ibkr_override'. That tag is the diagnostic we'll query later when
    investigating "why is this row's number different from the broker's"."""
    r = _post_journal_edit(client, holdings_source="ibkr_override")
    assert r.status_code == 200
    assert captured_save["holdings_source"] == "ibkr_override"


# ---------------------------------------------------------------------------
# Independence — nlv_source and holdings_source don't have to agree
# ---------------------------------------------------------------------------


def test_nlv_auto_with_holdings_override_persists_independently(client, captured_save):
    """User accepts IBKR's NLV but overrides Holdings. Each tag stands
    alone — there's no hidden coupling between them."""
    r = _post_journal_edit(client,
                           nlv_source="ibkr_auto",
                           holdings_source="ibkr_override")
    assert r.status_code == 200
    assert captured_save["nlv_source"] == "ibkr_auto"
    assert captured_save["holdings_source"] == "ibkr_override"


def test_nlv_override_with_holdings_auto_persists_independently(client, captured_save):
    """The reverse: user trusts IBKR's Holdings but typed over the NLV
    (e.g. accounting for a margin call adjustment IBKR hasn't applied yet)."""
    r = _post_journal_edit(client,
                           nlv_source="ibkr_override",
                           holdings_source="ibkr_auto")
    assert r.status_code == 200
    assert captured_save["nlv_source"] == "ibkr_override"
    assert captured_save["holdings_source"] == "ibkr_auto"
