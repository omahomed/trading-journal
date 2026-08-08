"""Regression test for the 2026-08-07 Trade Manager Stop Loss Adjustment
bug — /api/trades/update-stops was updating stop_loss on EVERY BUY row of
a campaign, including fully-LIFO-consumed lots from months ago. The fix
routes the request through compute_open_inventory to filter to lots that
still have remaining shares, then passes the trx_id whitelist down to
db.update_trade_stops. Closed lots keep their historical stops.

Coverage:
  * Fresh campaign (only B1, no sells) → all lots update (baseline).
  * Partial-close (B1 100 sh, sell 30) → B1 (70 remaining) updates.
  * Add-on chain with LIFO closures — A11 fully consumed via a later
    SELL, later A12/A13/A15 still open → only the open trx_ids reach
    the DB helper. The specific DELL-inspired scenario from the bug
    report.
"""
from __future__ import annotations

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


def _make_summary(trade_id: str, ticker: str, avg_entry: float) -> pd.DataFrame:
    """Minimal summary shape load_summary would return."""
    return pd.DataFrame([{
        "Trade_ID": trade_id, "Ticker": ticker, "Status": "OPEN",
        "Open_Date": "2026-04-06", "Shares": 100.0, "Avg_Entry": avg_entry,
        "Stop_Loss": None, "Multiplier": 1.0, "Instrument_Type": "STOCK",
    }])


def _make_details(rows: list[dict]) -> pd.DataFrame:
    """Minimal detail-rows shape load_details would return. Rows are
    passed in insertion order; the LIFO walker sorts by date + type."""
    return pd.DataFrame(rows)


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    import db_layer as db

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "summary": None,
        "details": None,
        "update_calls": [],
        "audit_calls": [],
    }

    def fake_load_summary(*_a, **_kw):
        return state["summary"]

    def fake_load_details(*_a, **_kw):
        return state["details"]

    def fake_update_trade_stops(
        portfolio_name, trade_id, new_stop,
        be_applied=False, be_cleared=False, open_lot_trx_ids=None,
    ):
        state["update_calls"].append({
            "portfolio_name": portfolio_name,
            "trade_id": trade_id,
            "new_stop": new_stop,
            "be_applied": be_applied,
            "be_cleared": be_cleared,
            "open_lot_trx_ids": (
                list(open_lot_trx_ids) if open_lot_trx_ids is not None else None
            ),
        })
        return len(state["update_calls"][-1]["open_lot_trx_ids"] or [])

    def fake_log_audit(*args, **kwargs):
        state["audit_calls"].append((args, kwargs))

    monkeypatch.setattr(db, "load_summary", fake_load_summary)
    monkeypatch.setattr(db, "load_details", fake_load_details)
    monkeypatch.setattr(db, "update_trade_stops", fake_update_trade_stops)
    monkeypatch.setattr(db, "log_audit", fake_log_audit)

    # Disable rate limiter (borrowed pattern from other endpoint tests).
    from slowapi import Limiter

    def _noop_limit(*_a, **_kw):
        def _decorator(func):
            return func
        return _decorator

    monkeypatch.setattr(main.limiter, "limit", _noop_limit)

    return state, TestClient(main.app)


def _post(client: TestClient, body: dict):
    return client.put(
        "/api/trades/update-stops", json=body, headers=_auth_headers(),
    )


def test_fresh_campaign_all_lots_update(stubbed):
    """Only B1, no sells — the sole open lot's trx_id should be passed."""
    state, client = stubbed
    state["summary"] = _make_summary("202604-001", "TEST", 100.0)
    state["details"] = _make_details([
        {"Trade_ID": "202604-001", "Ticker": "TEST", "Action": "BUY",
         "Date": "2026-04-06 10:00:00", "Shares": 100.0, "Amount": 100.0,
         "Stop_Loss": 90.0, "Trx_ID": "B1", "Match_Method": None,
         "Instrument_Type": "STOCK", "Multiplier": 1.0},
    ])
    r = _post(client, {"portfolio": "CanSlim", "trade_id": "202604-001",
                       "new_stop": 95.0})
    assert r.status_code == 200
    assert state["update_calls"][-1]["open_lot_trx_ids"] == ["B1"]


def test_partial_close_leaves_b1_open(stubbed):
    """B1 100 sh, sell 30 → B1 has 70 remaining, updates."""
    state, client = stubbed
    state["summary"] = _make_summary("202604-002", "TEST", 100.0)
    state["details"] = _make_details([
        {"Trade_ID": "202604-002", "Ticker": "TEST", "Action": "BUY",
         "Date": "2026-04-06 10:00:00", "Shares": 100.0, "Amount": 100.0,
         "Stop_Loss": 90.0, "Trx_ID": "B1", "Match_Method": None,
         "Instrument_Type": "STOCK", "Multiplier": 1.0},
        {"Trade_ID": "202604-002", "Ticker": "TEST", "Action": "SELL",
         "Date": "2026-04-20 10:00:00", "Shares": 30.0, "Amount": 110.0,
         "Stop_Loss": 0.0, "Trx_ID": "S1", "Match_Method": "LIFO",
         "Instrument_Type": "STOCK", "Multiplier": 1.0},
    ])
    r = _post(client, {"portfolio": "CanSlim", "trade_id": "202604-002",
                       "new_stop": 108.0})
    assert r.status_code == 200
    assert state["update_calls"][-1]["open_lot_trx_ids"] == ["B1"]


def test_lifo_chain_excludes_fully_consumed_lots(stubbed):
    """DELL-inspired scenario: B1 + A1 + A2 + A3 with intervening sells;
    A1 and A2 fully consumed by S1 (LIFO), A3 still open, B1 partial."""
    state, client = stubbed
    state["summary"] = _make_summary("202604-013", "DELL", 200.0)
    state["details"] = _make_details([
        # B1: 100 sh @ $176.21
        {"Trade_ID": "202604-013", "Ticker": "DELL", "Action": "BUY",
         "Date": "2026-04-06 10:00:00", "Shares": 100.0, "Amount": 176.21,
         "Stop_Loss": 160.0, "Trx_ID": "B1", "Match_Method": None,
         "Instrument_Type": "STOCK", "Multiplier": 1.0},
        # A1: 40 sh @ $200
        {"Trade_ID": "202604-013", "Ticker": "DELL", "Action": "BUY",
         "Date": "2026-04-14 10:00:00", "Shares": 40.0, "Amount": 200.0,
         "Stop_Loss": 180.0, "Trx_ID": "A1", "Match_Method": None,
         "Instrument_Type": "STOCK", "Multiplier": 1.0},
        # A2: 30 sh @ $220
        {"Trade_ID": "202604-013", "Ticker": "DELL", "Action": "BUY",
         "Date": "2026-04-28 10:00:00", "Shares": 30.0, "Amount": 220.0,
         "Stop_Loss": 200.0, "Trx_ID": "A2", "Match_Method": None,
         "Instrument_Type": "STOCK", "Multiplier": 1.0},
        # S1: sells 70 (LIFO consumes A2 fully + A1 fully = 70 shares)
        {"Trade_ID": "202604-013", "Ticker": "DELL", "Action": "SELL",
         "Date": "2026-05-19 10:00:00", "Shares": 70.0, "Amount": 250.0,
         "Stop_Loss": 0.0, "Trx_ID": "S1", "Match_Method": "LIFO",
         "Instrument_Type": "STOCK", "Multiplier": 1.0},
        # A3: 50 sh @ $270 (fresh add after the sell)
        {"Trade_ID": "202604-013", "Ticker": "DELL", "Action": "BUY",
         "Date": "2026-06-01 10:00:00", "Shares": 50.0, "Amount": 270.0,
         "Stop_Loss": 240.0, "Trx_ID": "A3", "Match_Method": None,
         "Instrument_Type": "STOCK", "Multiplier": 1.0},
    ])
    # Post-walk open inventory (LIFO): B1 100 sh remaining + A3 50 sh
    # remaining. A1 and A2 fully consumed by S1.
    r = _post(client, {"portfolio": "CanSlim", "trade_id": "202604-013",
                       "new_stop": 260.0})
    assert r.status_code == 200
    ids = state["update_calls"][-1]["open_lot_trx_ids"]
    assert set(ids) == {"B1", "A3"}, (
        f"expected only open lots {{'B1', 'A3'}}, got {ids} — closed lot "
        f"stop_loss would be overwritten again (regression to pre-2026-08-07)"
    )
    # Order doesn't matter for the SET behavior but keep the assertion
    # deterministic — LIFO walker preserves arrival_seq for remaining lots.
    # (Both come from _walk_inventory's final inventory list.)
