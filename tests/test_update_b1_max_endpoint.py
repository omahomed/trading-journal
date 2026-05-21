"""Tests for POST /api/trades/{trade_id}/update-b1-max (migration 036).

The endpoint persists the running peak B1 return % per OPEN campaign.
The CORRECTNESS-CRITICAL property is monotonic non-decrease: lower /
equal values must be no-ops. The SQL guard inside
db.update_b1_max_return_pct enforces this; these tests stub that helper
to confirm the endpoint plumbs args through correctly and shapes the
response per spec.

Pattern mirrors tests/test_log_buy_strategy.py — TestClient, stubbed
db_layer helpers, AUTH_SECRET overridden, rate-limiter disabled.
"""
from __future__ import annotations

from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def stubbed(monkeypatch):
    """Yield (state, client). Tests set state['db_response'] to control
    what the stubbed db.update_b1_max_return_pct returns.
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    import db_layer

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "db_response": None,
        "calls": [],
    }

    def fake_update(portfolio_name, trade_id, new_max_pct):
        state["calls"].append((portfolio_name, trade_id, new_max_pct))
        return state["db_response"]
    monkeypatch.setattr(db_layer, "update_b1_max_return_pct", fake_update)

    # load_summary.clear() is a cache invalidation that's harmless under stub.
    if hasattr(db_layer.load_summary, "clear"):
        monkeypatch.setattr(db_layer.load_summary, "clear", lambda: None)

    original_enabled = getattr(main.limiter, "enabled", True)
    if hasattr(main.limiter, "enabled"):
        main.limiter.enabled = False

    client = TestClient(main.app, headers=_auth_headers())
    try:
        yield state, client
    finally:
        if hasattr(main.limiter, "enabled"):
            main.limiter.enabled = original_enabled


def test_first_observation_persists(stubbed):
    """NULL stored → write new value, was_updated=True."""
    state, client = stubbed
    state["db_response"] = {"stored_max_pct": 12.5, "was_updated": True}

    resp = client.post(
        "/api/trades/202604-001/update-b1-max",
        json={"portfolio": "CanSlim", "new_max_pct": 12.5},
    )
    assert resp.status_code == 200
    assert resp.json() == {"stored_max_pct": 12.5, "was_updated": True}
    assert state["calls"] == [("CanSlim", "202604-001", 12.5)]


def test_higher_value_updates(stubbed):
    """Stored 30 → POST 55 → was_updated=True with new value returned."""
    state, client = stubbed
    state["db_response"] = {"stored_max_pct": 55.0, "was_updated": True}

    resp = client.post(
        "/api/trades/202604-001/update-b1-max",
        json={"portfolio": "CanSlim", "new_max_pct": 55.0},
    )
    assert resp.status_code == 200
    assert resp.json() == {"stored_max_pct": 55.0, "was_updated": True}


def test_equal_value_noop(stubbed):
    """Stored 60 → POST 60 → was_updated=False; stored unchanged."""
    state, client = stubbed
    state["db_response"] = {"stored_max_pct": 60.0, "was_updated": False}

    resp = client.post(
        "/api/trades/202604-001/update-b1-max",
        json={"portfolio": "CanSlim", "new_max_pct": 60.0},
    )
    assert resp.status_code == 200
    assert resp.json() == {"stored_max_pct": 60.0, "was_updated": False}


def test_lower_value_noop(stubbed):
    """Stored 70 → POST 30 (pullback case) → was_updated=False;
    stored stays at 70. Confirms the monotonic guard."""
    state, client = stubbed
    state["db_response"] = {"stored_max_pct": 70.0, "was_updated": False}

    resp = client.post(
        "/api/trades/202604-001/update-b1-max",
        json={"portfolio": "CanSlim", "new_max_pct": 30.0},
    )
    assert resp.status_code == 200
    assert resp.json() == {"stored_max_pct": 70.0, "was_updated": False}


def test_missing_trade_returns_404(stubbed):
    state, client = stubbed
    state["db_response"] = None  # db helper signals "not found"

    resp = client.post(
        "/api/trades/999999-999/update-b1-max",
        json={"portfolio": "CanSlim", "new_max_pct": 12.5},
    )
    assert resp.status_code == 404


def test_missing_new_max_pct_returns_422(stubbed):
    _, client = stubbed
    resp = client.post(
        "/api/trades/202604-001/update-b1-max",
        json={"portfolio": "CanSlim"},
    )
    assert resp.status_code == 422


def test_non_numeric_new_max_pct_returns_422(stubbed):
    _, client = stubbed
    resp = client.post(
        "/api/trades/202604-001/update-b1-max",
        json={"portfolio": "CanSlim", "new_max_pct": "not-a-number"},
    )
    assert resp.status_code == 422


def test_infinity_new_max_pct_returns_422(stubbed):
    """Inf is a valid float in Python but not a valid B1 return %.
    The endpoint must reject it so a poisoned payload can't corrupt
    the stored max with a value that breaks the SQL guard's < operator.
    Sent as a string so TestClient's JSON serializer (which rejects
    non-finite floats outright) doesn't intercept it before the server
    validator runs."""
    _, client = stubbed
    resp = client.post(
        "/api/trades/202604-001/update-b1-max",
        json={"portfolio": "CanSlim", "new_max_pct": "Infinity"},
    )
    assert resp.status_code == 422
    assert "finite" in resp.json().get("detail", "").lower()


def test_default_portfolio_is_canslim(stubbed):
    """Omitted portfolio defaults to CanSlim."""
    state, client = stubbed
    state["db_response"] = {"stored_max_pct": 5.0, "was_updated": True}

    resp = client.post(
        "/api/trades/202604-001/update-b1-max",
        json={"new_max_pct": 5.0},
    )
    assert resp.status_code == 200
    assert state["calls"][0][0] == "CanSlim"
