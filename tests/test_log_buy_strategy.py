"""Tests for the strategy field on POST /api/trades/buy (Migration 019).

Verifies that:
  - Omitted strategy defaults to 'CanSlim' (matches the DB column DEFAULT)
  - Explicit valid strategy is persisted
  - Unknown strategy is rejected with a 200 + {error: ...} payload (the
    endpoint's existing error contract — no HTTP error code, callers read
    response body)
  - GET /api/strategies returns the seeded rows in created_at ASC order

Test invocation mirrors tests/test_exercise_option.py: stubbed db_layer
helpers, FastAPI TestClient, rate-limiter disabled per-session.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import jwt
import pandas as pd
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def stubbed(monkeypatch):
    """Yield (state, client). Tests mutate state to configure DB stubs.

    state knobs:
      strategies          — list of dicts returned by db.load_strategies
      summary_df          — what db.load_summary returns (post-normalize)
      details_df          — what db.load_details returns (post-normalize)

    state observations:
      saved_summaries     — list of summary_row dicts passed to save_summary_row
      saved_details       — list of detail_row dicts passed to save_detail_row
      audit_logs          — list of (action, trade_id, ticker, details) tuples
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    import db_layer

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "strategies": [
            {"name": "CanSlim",     "description": "primary",   "color": "#6366f1",
             "is_active": True, "created_at": datetime(2026, 1, 1, 0, 0, 0)},
            {"name": "StockTalk",   "description": "small-cap", "color": "#d97706",
             "is_active": True, "created_at": datetime(2026, 1, 1, 0, 0, 1)},
            {"name": "21eStrategy", "description": "21 EMA",    "color": "#0d9488",
             "is_active": True, "created_at": datetime(2026, 1, 1, 0, 0, 2)},
        ],
        "summary_df": pd.DataFrame(),
        "details_df": pd.DataFrame(),
        "saved_summaries": [],
        "saved_details": [],
        "audit_logs": [],
    }

    monkeypatch.setattr(db_layer, "load_strategies",
                        lambda active_only=True: [
                            s for s in state["strategies"]
                            if not active_only or s["is_active"]
                        ])

    # Wrap load_summary/load_details to bypass the @ttl_cache decorator —
    # tests need to swap the dataframe between calls without hitting cache.
    monkeypatch.setattr(db_layer, "load_summary",
                        lambda *a, **kw: state["summary_df"])
    monkeypatch.setattr(db_layer, "load_details",
                        lambda *a, **kw: state["details_df"])

    # _normalize_trades is permissive — pass dataframes through as-is so
    # tests can supply already-normalized snake_case columns.
    monkeypatch.setattr(main, "_normalize_trades", lambda df: df)

    summary_id_counter = {"v": 100}
    def fake_save_summary(portfolio_name, row_dict):
        summary_id_counter["v"] += 1
        state["saved_summaries"].append(dict(row_dict))
        return summary_id_counter["v"]
    monkeypatch.setattr(db_layer, "save_summary_row", fake_save_summary)

    detail_id_counter = {"v": 0}
    def fake_save_detail(portfolio_name, row_dict):
        detail_id_counter["v"] += 1
        state["saved_details"].append(dict(row_dict))
        return detail_id_counter["v"]
    monkeypatch.setattr(db_layer, "save_detail_row", fake_save_detail)

    monkeypatch.setattr(db_layer, "generate_unique_trx_id",
                        lambda portfolio, trade_id, prefix: f"{prefix}1")

    def fake_log_audit(*args, **kwargs):
        state["audit_logs"].append((args, kwargs))
    monkeypatch.setattr(db_layer, "log_audit", fake_log_audit)

    # Disable rate limiting so back-to-back posts in one test never 429.
    original_enabled = getattr(main.limiter, "enabled", True)
    if hasattr(main.limiter, "enabled"):
        main.limiter.enabled = False

    client = TestClient(main.app, headers=_auth_headers())
    try:
        yield state, client
    finally:
        if hasattr(main.limiter, "enabled"):
            main.limiter.enabled = original_enabled


# ---------------------------------------------------------------------------
# POST /api/trades/buy — strategy field
# ---------------------------------------------------------------------------


def _buy_body(**overrides) -> dict:
    """Minimal valid Log Buy payload. Tests override one or two fields."""
    base = {
        "portfolio": "CanSlim",
        "action_type": "new",
        "ticker": "AAPL",
        "trade_id": "202605-001",
        "shares": 100,
        "price": 200.0,
        "stop_loss": 190.0,
        "rule": "br1.1 Consolidation",
        "notes": "test",
        "date": "2026-05-08",
        "time": "09:30",
    }
    base.update(overrides)
    return base


def test_log_buy_defaults_strategy_to_canslim(stubbed):
    """Body without `strategy` → row persisted with Strategy='CanSlim'."""
    state, client = stubbed

    r = client.post("/api/trades/buy", json=_buy_body())

    assert r.status_code == 200
    assert r.json().get("status") == "ok", r.json()
    assert len(state["saved_summaries"]) == 1
    assert state["saved_summaries"][0]["Strategy"] == "CanSlim"


def test_log_buy_persists_explicit_strategy(stubbed):
    """Body with strategy='StockTalk' → persisted as-is."""
    state, client = stubbed

    r = client.post("/api/trades/buy", json=_buy_body(strategy="StockTalk"))

    assert r.status_code == 200
    assert r.json().get("status") == "ok", r.json()
    assert state["saved_summaries"][0]["Strategy"] == "StockTalk"


def test_log_buy_rejects_unknown_strategy(stubbed):
    """Body with strategy='Bogus' → error response, no row written."""
    state, client = stubbed

    r = client.post("/api/trades/buy", json=_buy_body(strategy="Bogus"))

    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert "Bogus" in body["error"]
    assert state["saved_summaries"] == []
    assert state["saved_details"] == []


def test_log_buy_scalein_inherits_parent_strategy(stubbed):
    """Scale-in ignores body strategy, inherits from the parent campaign.

    This is the defense-in-depth case: even if a malicious / buggy client
    submits a different strategy on a scale-in, the backend uses the
    parent's value. Mirrors how instrument_type is inherited on scale-in.
    """
    state, client = stubbed
    state["summary_df"] = pd.DataFrame([{
        "trade_id": "202605-001", "ticker": "AAPL", "status": "OPEN",
        "instrument_type": "STOCK", "multiplier": 1,
        "shares": 100, "avg_entry": 200.0, "total_cost": 20000.0,
        "open_date": "2026-04-01", "stop_loss": 190.0, "rule": "br1.1",
        "buy_notes": "", "risk_budget": 1000.0,
        "strategy": "21eStrategy",  # parent strategy, not CanSlim
    }])
    state["details_df"] = pd.DataFrame([{
        "trade_id": "202605-001", "ticker": "AAPL", "action": "BUY",
    }])

    r = client.post("/api/trades/buy", json=_buy_body(
        action_type="scalein",
        # Body says StockTalk, but parent is 21eStrategy — parent wins.
        strategy="StockTalk",
    ))

    assert r.status_code == 200
    assert r.json().get("status") == "ok", r.json()
    assert state["saved_summaries"][0]["Strategy"] == "21eStrategy"


# ---------------------------------------------------------------------------
# GET /api/strategies
# ---------------------------------------------------------------------------


def test_strategies_endpoint_returns_seeded_rows_in_order(stubbed):
    """GET /api/strategies → CanSlim, StockTalk, 21eStrategy in that order."""
    _, client = stubbed

    r = client.get("/api/strategies")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert [s["name"] for s in body] == ["CanSlim", "StockTalk", "21eStrategy"]
    # CanSlim color must match the dashboard primary accent (--color-g-dash).
    assert body[0]["color"] == "#6366f1"


def test_strategies_endpoint_filters_active_by_default(stubbed):
    """A disabled strategy is hidden by default but visible with ?active=false."""
    state, client = stubbed
    state["strategies"].append({
        "name": "Retired",
        "description": "old",
        "color": "#888888",
        "is_active": False,
        "created_at": datetime(2026, 1, 1, 0, 0, 3),
    })

    active_only = client.get("/api/strategies").json()
    assert "Retired" not in [s["name"] for s in active_only]

    all_rows = client.get("/api/strategies?active=false").json()
    assert "Retired" in [s["name"] for s in all_rows]
