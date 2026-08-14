"""Tests for GET /api/risk/levels — Migration 068 L-series composed view.

Verifies that the endpoint correctly maps:
  * MCT engine state (violation_21 / consec_below_21 / consec_below_50 /
    violation_50) into L2 / L3 / L4 status.
  * cycle_reference row + current NLV into L1 status.
  * Level priority: deepest cap wins for active_level.
  * excess_dollars_to_sell math from (exposure - cap) × NLV.

Stubs nlv_service.dashboard_metrics, db.list_portfolios,
db.load_active_cycle_reference, and api.mct_endpoint_adapter.run_engine
so tests never touch Postgres or the real MCT engine.
"""
from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from unittest.mock import MagicMock

import jwt
import pandas as pd
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": "test-user"}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@dataclass
class _FakeEngineResult:
    """Minimal shape mimicking api.mct_endpoint_adapter.EngineResult."""
    final_state: dict[str, Any]
    bars: pd.DataFrame


@pytest.fixture
def risk_client(monkeypatch):
    """TestClient with all downstream dependencies stubbed. Tests configure
    per-call return values via the state dict."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    import nlv_service
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "portfolios": [{"id": 1, "name": "CanSlim"}],
        "cycle_ref": None,
        "metrics": {
            "nlv": 100_000.0,
            "exposure_pct": 50.0,
            "total_holdings": 50_000.0,
            "drawdown_peak_nlv": 100_000.0,
            "drawdown_current_pct": 0.0,
        },
        "engine_state": {
            "consec_below_21": 0,
            "consec_below_50": 0,
            "violation_21_fired": False,
            "violation_50_fired": False,
        },
        "entry_exposure": 100,
    }

    monkeypatch.setattr(main.db, "list_portfolios",
                        lambda: state["portfolios"])
    monkeypatch.setattr(main.db, "load_active_cycle_reference",
                        lambda name: state["cycle_ref"])
    monkeypatch.setattr(nlv_service, "dashboard_metrics",
                        lambda pid, name: state["metrics"])

    def fake_run_engine(*args, **kwargs):
        return _FakeEngineResult(
            final_state=dict(state["engine_state"]),
            bars=pd.DataFrame({"trade_date": ["2026-08-13"], "close": [24000.0]}),
        )
    from api import mct_endpoint_adapter
    monkeypatch.setattr(mct_endpoint_adapter, "run_engine", fake_run_engine)

    def fake_to_rally(_result):
        return {"entry_exposure": state["entry_exposure"]}
    monkeypatch.setattr(mct_endpoint_adapter,
                        "to_rally_prefix_response", fake_to_rally)

    # Force-correction override — return None so the endpoint takes the
    # no-override path.
    monkeypatch.setattr(main, "_current_override_date", lambda: None)

    tc = TestClient(main.app, headers=_auth_headers())
    tc.state = state  # type: ignore[attr-defined]
    return tc


def _get(tc, portfolio="CanSlim"):
    r = tc.get(f"/api/risk/levels?portfolio={portfolio}")
    assert r.status_code == 200, r.text
    return r.json()


# ── Base cases ──────────────────────────────────────────────────────

def test_unknown_portfolio_returns_error(risk_client):
    """Non-existent portfolio → {error}, not 500."""
    res = _get(risk_client, portfolio="Ghost")
    assert "error" in res


def test_empty_state_all_clear(risk_client):
    """No cycle_ref, no engine breaches → all levels CLEAR, active_level null."""
    res = _get(risk_client)
    assert res["portfolio"] == "CanSlim"
    assert res["cycle_reference"] is None
    assert res["active_level"] is None
    assert res["effective_cap_pct"] is None
    assert res["excess_dollars_to_sell"] == 0.0

    keys = [lvl["key"] for lvl in res["levels_state"]]
    assert keys == ["L1", "L2", "L3", "L4"]
    for lvl in res["levels_state"]:
        assert lvl["status"] == "CLEAR"


# ── L1 — cycle-reference drawdown ─────────────────────────────────

def test_l1_fires_at_7_5_pct_below_cycle_reference(risk_client):
    """current_nlv <= ratcheted × 0.925 → L1 FIRED, cap 80%."""
    risk_client.state["cycle_ref"] = {
        "id": 1, "portfolio_id": 1, "flip_date": "2026-08-07",
        "initial_nlv": 100_000, "ratcheted_nlv": 100_000,
        "ratcheted_on_date": "2026-08-07",
        "is_frozen": False, "frozen_at_date": None,
    }
    # 92,500 = 100k × 0.925 — exactly on threshold, fires.
    risk_client.state["metrics"]["nlv"] = 92_500.0
    risk_client.state["metrics"]["exposure_pct"] = 100.0
    res = _get(risk_client)

    l1 = next(lv for lv in res["levels_state"] if lv["key"] == "L1")
    assert l1["status"] == "FIRED"
    assert l1["threshold_nlv"] == 92_500.0
    assert res["active_level"] == "L1"
    assert res["effective_cap_pct"] == 80
    # exposure 100% > cap 80% → 20% × 92500 = 18500 to sell
    assert abs(res["excess_dollars_to_sell"] - 18_500.0) < 1


def test_l1_clear_when_above_threshold(risk_client):
    """current_nlv > ratcheted × 0.925 → L1 CLEAR even if drawdown exists."""
    risk_client.state["cycle_ref"] = {
        "id": 1, "portfolio_id": 1, "flip_date": "2026-08-07",
        "initial_nlv": 100_000, "ratcheted_nlv": 100_000,
        "ratcheted_on_date": "2026-08-07",
        "is_frozen": False, "frozen_at_date": None,
    }
    # 95,000 = -5% from cycle ref — inside the buffer.
    risk_client.state["metrics"]["nlv"] = 95_000.0
    res = _get(risk_client)

    l1 = next(lv for lv in res["levels_state"] if lv["key"] == "L1")
    assert l1["status"] == "CLEAR"
    assert res["active_level"] is None


# ── L2/L3/L4 — MCT engine state ───────────────────────────────────

def test_l2_fires_on_violation_21(risk_client):
    """violation_21_fired without consec>=2 → L2 FIRED, cap 60%."""
    risk_client.state["engine_state"]["violation_21_fired"] = True
    risk_client.state["metrics"]["exposure_pct"] = 90.0
    res = _get(risk_client)

    l2 = next(lv for lv in res["levels_state"] if lv["key"] == "L2")
    assert l2["status"] == "FIRED"
    assert res["active_level"] == "L2"
    assert res["effective_cap_pct"] == 60
    # 90% - 60% = 30% × 100k = 30k
    assert abs(res["excess_dollars_to_sell"] - 30_000.0) < 1


def test_l3_fires_on_2_consec_below_21(risk_client):
    """consec_below_21 >= 2 → L3 FIRED, cap 40%. Supersedes L2."""
    risk_client.state["engine_state"]["consec_below_21"] = 2
    risk_client.state["engine_state"]["violation_21_fired"] = True
    res = _get(risk_client)

    l3 = next(lv for lv in res["levels_state"] if lv["key"] == "L3")
    assert l3["status"] == "FIRED"
    # L2 is NOT independently FIRED — L3 supersedes on the same signal path.
    l2 = next(lv for lv in res["levels_state"] if lv["key"] == "L2")
    assert l2["status"] == "CLEAR"
    assert res["active_level"] == "L3"
    assert res["effective_cap_pct"] == 40


def test_l4_fires_on_2_consec_below_50(risk_client):
    """consec_below_50 >= 2 → L4 FIRED, cap 20%. Deepest — wins over
    any other level that also fires."""
    risk_client.state["engine_state"]["consec_below_50"] = 2
    risk_client.state["engine_state"]["consec_below_21"] = 5
    risk_client.state["engine_state"]["violation_21_fired"] = True
    res = _get(risk_client)

    l4 = next(lv for lv in res["levels_state"] if lv["key"] == "L4")
    assert l4["status"] == "FIRED"
    assert res["active_level"] == "L4"
    assert res["effective_cap_pct"] == 20


def test_l4_fires_on_violation_50(risk_client):
    """violation_50_fired → L4 FIRED (independent of consec_below_50)."""
    risk_client.state["engine_state"]["violation_50_fired"] = True
    res = _get(risk_client)

    l4 = next(lv for lv in res["levels_state"] if lv["key"] == "L4")
    assert l4["status"] == "FIRED"
    assert res["active_level"] == "L4"


def test_l2_armed_on_1_close_below_21(risk_client):
    """consec_below_21 == 1 → L2 ARMED (not FIRED). No cap applied."""
    risk_client.state["engine_state"]["consec_below_21"] = 1
    res = _get(risk_client)

    l2 = next(lv for lv in res["levels_state"] if lv["key"] == "L2")
    assert l2["status"] == "ARMED"
    assert res["active_level"] is None
    assert res["effective_cap_pct"] is None


def test_l4_armed_on_1_close_below_50(risk_client):
    """consec_below_50 == 1 without violation_50 → L4 ARMED."""
    risk_client.state["engine_state"]["consec_below_50"] = 1
    res = _get(risk_client)

    l4 = next(lv for lv in res["levels_state"] if lv["key"] == "L4")
    assert l4["status"] == "ARMED"
    assert res["active_level"] is None


# ── Level priority ────────────────────────────────────────────────

def test_deepest_cap_wins_when_multiple_levels_fire(risk_client):
    """L1 (80%) + L3 (40%) + L4 (20%) all fire → active_level = L4."""
    risk_client.state["cycle_ref"] = {
        "id": 1, "portfolio_id": 1, "flip_date": "2026-08-07",
        "initial_nlv": 100_000, "ratcheted_nlv": 100_000,
        "ratcheted_on_date": "2026-08-07",
        "is_frozen": False, "frozen_at_date": None,
    }
    risk_client.state["metrics"]["nlv"] = 80_000.0  # -20% from ref, L1 fires
    risk_client.state["engine_state"]["consec_below_21"] = 2  # L3 fires
    risk_client.state["engine_state"]["consec_below_50"] = 2  # L4 fires
    res = _get(risk_client)

    assert res["active_level"] == "L4"
    assert res["effective_cap_pct"] == 20


# ── Excess-dollars-to-sell edge cases ─────────────────────────────

def test_zero_excess_when_exposure_below_cap(risk_client):
    """Exposure already inside cap → excess_dollars_to_sell = 0."""
    risk_client.state["cycle_ref"] = {
        "id": 1, "portfolio_id": 1, "flip_date": "2026-08-07",
        "initial_nlv": 100_000, "ratcheted_nlv": 100_000,
        "ratcheted_on_date": "2026-08-07",
        "is_frozen": False, "frozen_at_date": None,
    }
    risk_client.state["metrics"]["nlv"] = 90_000.0  # -10% from ref, L1 fires
    risk_client.state["metrics"]["exposure_pct"] = 50.0  # inside 80% cap
    res = _get(risk_client)

    assert res["active_level"] == "L1"
    assert res["excess_dollars_to_sell"] == 0.0


# ── ATH-drawdown surfaces (informational only) ────────────────────

def test_ath_fields_are_passed_through(risk_client):
    """drawdown_peak_nlv + drawdown_current_pct come from dashboard_metrics
    and appear as ath_hwm / ath_drawdown_pct — informational only."""
    risk_client.state["metrics"]["drawdown_peak_nlv"] = 940_294.0
    risk_client.state["metrics"]["drawdown_current_pct"] = -35.62
    risk_client.state["metrics"]["nlv"] = 605_337.0
    res = _get(risk_client)

    assert res["ath_hwm"] == 940_294.0
    assert res["ath_drawdown_pct"] == -35.62
    # Should NOT populate active_level from ATH drawdown alone.
    assert res["active_level"] is None
