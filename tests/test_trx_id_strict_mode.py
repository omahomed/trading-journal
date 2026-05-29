"""Strict-mode guards on client-supplied trx_id (Phase 1a hardening).

Three endpoints used to honor or partially honor a client-supplied
``trx_id`` field in the request body:

  * POST /api/trades/buy   — passed `given_trx_id` through to
    `_save_detail_with_unique_trx_id`, which preferred client value
    over the algorithm.
  * POST /api/trades/sell  — same pattern.
  * PUT  /api/trades/edit-transaction — allowed `trx_id` change as long
    as it didn't collide with another row in the same trade. A
    non-colliding change (e.g. A2 → A99) was silently accepted.

Strict mode rejects all three:

  * Buy + Sell: any non-empty client `trx_id` → 422.
  * Edit: a client `trx_id` that differs from the existing row value
    → 422. Matching value is tolerated because the frontend's
    read-only edit field echoes the existing value back to the server.

These tests pin the new behavior. They are paired with no frontend
change — every UI caller already either omits the field or echoes
the existing value (verified by grep across log-buy.tsx, log-sell.tsx,
trade-manager.tsx, import-trades.tsx, api.ts).
"""
from __future__ import annotations

from typing import Any

import jwt
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import db_layer


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def _summary_row(trade_id="202605-001", ticker="AAPL", shares=100.0, status="OPEN"):
    return {
        "trade_id": trade_id, "ticker": ticker, "status": status,
        "open_date": pd.Timestamp("2026-05-01"), "closed_date": None,
        "shares": shares, "avg_entry": 200.0, "avg_exit": 0.0,
        "total_cost": shares * 200.0, "realized_pl": 0.0,
        "unrealized_pl": 0.0, "return_pct": 0.0,
        "rule": "br1.1", "buy_notes": "", "sell_rule": None, "sell_notes": None,
        "risk_budget": 500.0, "stop_loss": 195.0,
        "instrument_type": "STOCK", "multiplier": 1.0, "strategy": "CanSlim",
    }


def _detail_row(detail_id=1, trade_id="202605-001", trx_id="B1",
                action="BUY", shares=100.0, amount=200.0):
    return {
        "detail_id": detail_id, "trade_id": trade_id, "ticker": "AAPL",
        "action": action, "date": pd.Timestamp("2026-05-01 09:30:00"),
        "shares": shares, "amount": amount, "value": shares * amount,
        "rule": "br1.1", "notes": "", "realized_pl": 0,
        "stop_loss": 195.0, "trx_id": trx_id,
        "instrument_type": "STOCK", "multiplier": 1.0,
    }


@pytest.fixture
def stubbed(monkeypatch):
    """FastAPI TestClient with db_layer stubbed enough for the three
    endpoints to reach the strict-mode reject. The actual DB writes are
    no-ops; we assert on the HTTP response, not persistence.
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "summary_df": pd.DataFrame([_summary_row()]),
        "details_df": pd.DataFrame([_detail_row()]),
    }

    monkeypatch.setattr(db_layer, "load_summary",
                        lambda *a, **kw: state["summary_df"])
    monkeypatch.setattr(db_layer, "load_details",
                        lambda *a, **kw: state["details_df"])
    monkeypatch.setattr(main, "_normalize_trades", lambda df: df)

    # Stubs sufficient to let log_buy / log_sell / edit reach (or skip past)
    # the strict check without exploding on downstream DB calls.
    monkeypatch.setattr(db_layer, "load_strategies",
                        lambda active_only=True, portfolio_name=None: [
                            {"name": "CanSlim", "is_active": True}])
    monkeypatch.setattr(db_layer, "save_summary_row", lambda *a, **kw: 1)
    monkeypatch.setattr(db_layer, "save_detail_row", lambda *a, **kw: 1)
    monkeypatch.setattr(db_layer, "update_detail_row", lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "save_summary_with_closures",
                        lambda *a, **kw: 1)
    monkeypatch.setattr(db_layer, "mirror_detail_edit_to_summary",
                        lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "generate_unique_trx_id",
                        lambda p, t, prefix: f"{prefix}99")
    monkeypatch.setattr(db_layer, "log_audit", lambda *a, **kw: None)
    monkeypatch.setattr(main, "validate_post_edit_matching",
                        lambda *a, **kw: None)

    if hasattr(main.limiter, "enabled"):
        original_enabled = main.limiter.enabled
        main.limiter.enabled = False
    else:
        original_enabled = None

    client = TestClient(main.app, headers=_auth_headers())
    try:
        yield state, client
    finally:
        if original_enabled is not None:
            main.limiter.enabled = original_enabled


# ---------------------------------------------------------------------------
# POST /api/trades/buy
# ---------------------------------------------------------------------------


def _buy_body(**overrides) -> dict:
    base = {
        "portfolio": "CanSlim", "action_type": "new",
        "ticker": "AAPL", "trade_id": "202605-001",
        "shares": 100, "price": 200.0, "stop_loss": 195.0,
        "rule": "br1.1", "notes": "", "date": "2026-05-08", "time": "09:30",
    }
    base.update(overrides)
    return base


def test_log_buy_rejects_client_trx_id(stubbed):
    """Buy with `trx_id` in body returns 422 + 'server-assigned' note."""
    _state, client = stubbed
    r = client.post("/api/trades/buy", json=_buy_body(trx_id="B7"))
    assert r.status_code == 422, r.text
    assert "server-assigned" in r.json()["detail"]
    assert "buy" in r.json()["detail"]


def test_log_buy_accepts_omitted_trx_id(stubbed):
    """Smoke: omitted trx_id still succeeds (server assigns)."""
    _state, client = stubbed
    r = client.post("/api/trades/buy", json=_buy_body())
    assert r.status_code == 200, r.text
    assert r.json().get("trx_id"), r.json()


# ---------------------------------------------------------------------------
# POST /api/trades/sell
# ---------------------------------------------------------------------------


def _sell_body(**overrides) -> dict:
    base = {
        "portfolio": "CanSlim", "trade_id": "202605-001",
        "shares": 50, "price": 220.0,
        "rule": "sr1.1", "notes": "",
        "date": "2026-05-08", "time": "10:00",
    }
    base.update(overrides)
    return base


def test_log_sell_rejects_client_trx_id(stubbed):
    """Sell with `trx_id` in body returns 422 + 'server-assigned' note."""
    _state, client = stubbed
    r = client.post("/api/trades/sell", json=_sell_body(trx_id="S7"))
    assert r.status_code == 422, r.text
    assert "server-assigned" in r.json()["detail"]
    assert "sell" in r.json()["detail"]


def test_log_sell_accepts_omitted_trx_id(stubbed):
    """Smoke: omitted trx_id still succeeds."""
    _state, client = stubbed
    r = client.post("/api/trades/sell", json=_sell_body())
    assert r.status_code == 200, r.text
    assert r.json().get("trx_id"), r.json()


# ---------------------------------------------------------------------------
# PUT /api/trades/edit-transaction
# ---------------------------------------------------------------------------


def _edit_body(**overrides) -> dict:
    base = {
        "detail_id": 1, "portfolio": "CanSlim", "trade_id": "202605-001",
        "ticker": "AAPL", "action": "BUY",
        "date": "2026-05-01 09:30", "shares": 100, "amount": 200.0,
        "rule": "br1.1 (edited)", "notes": "", "stop_loss": 195.0,
    }
    base.update(overrides)
    return base


def test_edit_rejects_trx_id_change(stubbed):
    """Edit with `trx_id` different from existing returns 422 + 'immutable'."""
    # Existing row has trx_id="B1"; client sends "B99".
    _state, client = stubbed
    r = client.put("/api/trades/edit-transaction", json=_edit_body(trx_id="B99"))
    assert r.status_code == 422, r.text
    assert "immutable" in r.json()["detail"]
    assert "B1" in r.json()["detail"]
    assert "B99" in r.json()["detail"]


def test_edit_tolerates_matching_trx_id(stubbed):
    """Edit with `trx_id` matching existing succeeds (frontend echo path)."""
    _state, client = stubbed
    r = client.put("/api/trades/edit-transaction", json=_edit_body(trx_id="B1"))
    assert r.status_code == 200, r.text
    assert "error" not in r.json(), r.json()


def test_edit_tolerates_omitted_trx_id(stubbed):
    """Edit with no `trx_id` field in body succeeds (treated as no-change)."""
    _state, client = stubbed
    r = client.put("/api/trades/edit-transaction", json=_edit_body())
    assert r.status_code == 200, r.text
    assert "error" not in r.json(), r.json()
