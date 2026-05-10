"""Buy-event Trade Risk $ recompute — log_buy integration tests.

Group 7-1 reframes trades_summary.risk_budget so it's recomputed on every
buy event (new campaign or scale-in) via compute_trade_risk over the full
post-insert BUY set, with each lot's stop applied per-lot. This kills the
prior additive scale-in bug: existing_risk_budget + new_lot_risk_budget
inflated risk every time and never reflected lots whose stops had moved
up. The recompute now walks the inventory holistically.

These tests exercise log_buy's POST /api/trades/buy endpoint end-to-end
(stubbed at the DB boundary), asserting that the value persisted to
summary.risk_budget reflects the post-insert inventory at current stops.

Scope intentionally narrow: log_buy is the *only* path that updates
risk_budget under the design. Stop changes, sells, edits, and deletes
preserve risk_budget — that contract is guarded by
test_recompute_preserves_metadata.py.
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


def _existing_summary_row(risk_budget: float = 500.0) -> dict[str, Any]:
    return {
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "status": "OPEN",
        "open_date": pd.Timestamp("2026-05-01"),
        "closed_date": None,
        "shares": 100.0,
        "avg_entry": 200.0,
        "avg_exit": 0.0,
        "total_cost": 20000.0,
        "realized_pl": 0.0,
        "unrealized_pl": 0.0,
        "return_pct": 0.0,
        "rule": "br1.3 Cup w/o Handle",
        "buy_notes": "Initial entry on breakout",
        "sell_rule": None,
        "sell_notes": None,
        "notes": "General notes on this campaign",
        "risk_budget": risk_budget,
        "stop_loss": 200.0,
        "instrument_type": "STOCK",
        "multiplier": 1.0,
        "strategy": "CanSlim",
    }


def _buy_detail_row(
    detail_id: int = 1, trx_id: str = "B1",
    shares: float = 100.0, amount: float = 200.0, stop_loss: float = 200.0,
    date: str = "2026-05-01 09:30:00",
) -> dict[str, Any]:
    return {
        "detail_id": detail_id,
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "action": "BUY",
        "date": pd.Timestamp(date),
        "shares": shares,
        "amount": amount,
        "value": shares * amount,
        "rule": "br1.3 Cup w/o Handle",
        "notes": "",
        "realized_pl": 0,
        "stop_loss": stop_loss,
        "trx_id": trx_id,
        "instrument_type": "STOCK",
        "multiplier": 1.0,
    }


@pytest.fixture
def stubbed(monkeypatch):
    """Mirror of the preservation-test fixture, scoped to log_buy assertions.

    state knobs:
      summary_df  — what db.load_summary returns
      details_df  — what db.load_details returns

    state observations:
      saved_summaries — list of dicts passed to save_summary_row (log_buy's
                        canonical write under Group 7-1 design)
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    import db_layer

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "summary_df": pd.DataFrame(),
        "details_df": pd.DataFrame(),
        "saved_summaries": [],
    }

    monkeypatch.setattr(db_layer, "load_summary",
                        lambda *a, **kw: state["summary_df"])
    monkeypatch.setattr(db_layer, "load_details",
                        lambda *a, **kw: state["details_df"])
    monkeypatch.setattr(main, "_normalize_trades", lambda df: df)

    def fake_save_summary_row(portfolio, row_dict):
        state["saved_summaries"].append({
            "portfolio": portfolio,
            "row": dict(row_dict),
        })
        return 1
    monkeypatch.setattr(db_layer, "save_summary_row", fake_save_summary_row)

    detail_id_counter = {"v": 1}
    def fake_save_detail(portfolio_name, row_dict):
        detail_id_counter["v"] += 1
        return detail_id_counter["v"]
    monkeypatch.setattr(db_layer, "save_detail_row", fake_save_detail)

    monkeypatch.setattr(db_layer, "generate_unique_trx_id",
                        lambda portfolio, trade_id, prefix: f"{prefix}1")
    monkeypatch.setattr(db_layer, "log_audit", lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "load_strategies",
                        lambda *a, **kw: [{"name": "CanSlim", "color": "#000",
                                            "description": None, "is_active": True,
                                            "created_at": "2026-01-01"}])

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


def test_new_campaign_open_writes_initial_risk(stubbed):
    """A fresh campaign's risk_budget = shares × (entry − stop) × multiplier.

    The "new" branch routes through compute_trade_risk on a single-BUY
    synthesized DataFrame — mathematically equivalent to the legacy
    calc_risk_budget call, but uses the canonical helper.
    """
    state, client = stubbed
    # No existing summary or details — clean open.

    r = client.post("/api/trades/buy", json={
        "portfolio": "CanSlim",
        "action_type": "new",
        "ticker": "TEST",
        "trade_id": "202605-100",
        "shares": 100,
        "price": 50.0,
        "stop_loss": 47.0,
        "rule": "br1.1 Consolidation",
        "notes": "",
        "date": "2026-05-15",
        "time": "10:00",
        "strategy": "CanSlim",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body

    assert state["saved_summaries"], "Expected log_buy to save summary"
    saved = state["saved_summaries"][-1]["row"]
    # 100 × (50 − 47) × 1 = $300
    assert saved.get("Risk_Budget") == 300.0


def test_scale_in_with_lot1_at_BE_does_not_inflate_risk(stubbed):
    """The exact additive-bug scenario the Group 7 audit called out.

    Pre-Group 7-1: log_buy scale-in stored existing_risk_budget + new_lot_risk
    regardless of whether the existing lot's stop had moved up. So lot 1 at
    BE (stop = entry, risk = 0) + lot 2 with a normal stop would persist as
    "lot 1's frozen historical risk + lot 2" — inflating the headline risk
    figure.

    After Group 7-1: log_buy walks current inventory via compute_trade_risk
    over the post-insert BUY set with each lot's stop applied per-lot, so
    lot 1 at BE contributes 0 and only lot 2 counts.

    Setup: existing summary's risk_budget is 99999 (nonsense / inflated by
    the old bug). Existing details: lot 1 (100 shs @ $200, stop $200 = BE).
    Scale-in adds lot 2 (50 shs @ $210, stop $205, risk $250). After
    log_buy persists the recompute, the written Risk_Budget should equal
    $250 — NOT 99999, NOT 0 + 750, NOT the existing value carried forward.
    """
    state, client = stubbed
    bumped = _existing_summary_row(risk_budget=99999.0)
    state["summary_df"] = pd.DataFrame([bumped])
    state["details_df"] = pd.DataFrame([
        _buy_detail_row(detail_id=1, trx_id="B1",
                        shares=100.0, amount=200.0, stop_loss=200.0,
                        date="2026-05-01 09:30:00"),
    ])

    r = client.post("/api/trades/buy", json={
        "portfolio": "CanSlim",
        "action_type": "scalein",
        "ticker": "AAPL",
        "trade_id": "202605-001",
        "shares": 50,
        "price": 210.0,
        "stop_loss": 205.0,
        "rule": "br1.3 Cup w/o Handle",
        "notes": "Scale-in on continuation",
        "date": "2026-05-08",
        "time": "10:00",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body

    assert state["saved_summaries"], "Expected log_buy to save summary"
    saved = state["saved_summaries"][-1]["row"]
    # Lot 1: 100 × max(0, 200 − 200) = 0   (BE)
    # Lot 2:  50 × max(0, 210 − 205) = 250
    # Total: 250 — NOT 99999 (preserved old value), NOT 750 (only lot 2's
    # naive risk added without lot 1 at BE), NOT 99999 + 250 (additive).
    assert saved.get("Risk_Budget") == 250.0, \
        f"Risk_Budget should equal lot 2's contribution alone " \
        f"(lot 1 at BE contributes 0): got {saved.get('Risk_Budget')!r}, " \
        f"expected 250.0"


def test_scale_in_both_lots_active_sums_risk(stubbed):
    """Both lots have active risk (stops below entry, not at BE).
    Holistic recompute sums them.

    Existing: 100 shs @ $200, stop $195 → 500 risk.
    Scale-in:  50 shs @ $210, stop $205 → 250 risk.
    Total: 750 — NOT 500 (frozen at original), NOT 250 (only new lot).
    """
    state, client = stubbed
    bumped = _existing_summary_row(risk_budget=500.0)
    bumped["stop_loss"] = 195.0
    state["summary_df"] = pd.DataFrame([bumped])
    state["details_df"] = pd.DataFrame([
        _buy_detail_row(detail_id=1, trx_id="B1",
                        shares=100.0, amount=200.0, stop_loss=195.0,
                        date="2026-05-01 09:30:00"),
    ])

    r = client.post("/api/trades/buy", json={
        "portfolio": "CanSlim",
        "action_type": "scalein",
        "ticker": "AAPL",
        "trade_id": "202605-001",
        "shares": 50,
        "price": 210.0,
        "stop_loss": 205.0,
        "rule": "br1.3 Cup w/o Handle",
        "notes": "Scale-in",
        "date": "2026-05-08",
        "time": "10:00",
    })
    assert r.status_code == 200, r.text

    saved = state["saved_summaries"][-1]["row"]
    assert saved.get("Risk_Budget") == 750.0, \
        f"Both lots active → risks sum: got {saved.get('Risk_Budget')!r}, expected 750.0"
