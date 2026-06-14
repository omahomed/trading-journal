"""Unit tests for mors.monitor.analyze() — the SR8-weekly conform path.

Mocks mors_backtest.run() to return a synthetic engine result so the
test exercises analyze()'s downstream math (floor schedule, Green-no-sell,
trim_dollars / trim_shares, current_tier surface) without spinning up
the real cascade engine or hitting yfinance.

Coverage:
  1. force_weekly=True is passed through to run()              — always-weekly contract
  2. TIER_NLV_FLOORS schedule: GREEN=15 / QUICK=10 / QS=5 / GD=0 — spec floors
  3. GREEN never sells (delta_shares=0 regardless of position size)
  4. QUICK trims down to 10% NLV target; delta_shares is positive
  5. QUICKSAND trims down to 5% NLV target
  6. GD exits to 0; terminated flag is set
  7. current_tier is surfaced in the return dict (distinct from last_signal)
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def _engine_result(
    *,
    ticker: str = "TKR",
    exit_px: float = 100.0,
    current_tier_label: str = "GREEN",
    last_signal: str = "GREEN",
) -> dict[str, Any]:
    """Build the shape mors_backtest.run() returns. Only the fields
    analyze() reads are populated; the rest are filler so analyze()'s
    DataFrame indexing doesn't NaN-out."""
    log = pd.DataFrame([{
        "Date": pd.Timestamp("2026-06-12").date(),
        "Phase": 2,
        "TF": "weekly",
        "Signal": last_signal,
        "Unit%": 100.0,
    }])
    return {
        "ticker": ticker,
        "entry_date": pd.Timestamp("2026-01-01").date(),
        "entry_px": 50.0,
        "exit_date": pd.Timestamp("2026-06-12").date(),
        "exit_px": exit_px,
        "log": log,
        "current_tier_idx": 0,
        "current_tier_label": current_tier_label,
        "current_tier_seeded": True,
    }


@pytest.fixture
def patched_run(monkeypatch):
    """Replace mors_backtest.run with a controllable spy. Tests set the
    desired engine result via state["result"]; the spy captures the
    kwargs passed by analyze() so we can assert force_weekly etc."""
    import mors.monitor as monitor_mod
    state: dict[str, Any] = {"result": _engine_result(), "calls": []}

    def fake_run(*args, **kwargs):
        state["calls"].append({"args": args, "kwargs": kwargs})
        return state["result"]

    monkeypatch.setattr(monitor_mod, "run", fake_run)
    return state


def _pos(ticker: str = "TKR", *, shares_held: float = 100.0, avg_price: float = 50.0):
    return {
        "ticker": ticker,
        "b1_date": "2026-01-01",
        "b1_price": 50.0,
        "shares_held": shares_held,
        "avg_price": avg_price,
    }


# ─────────────────────────────────────────────────────────────────────
# 1. force_weekly contract
# ─────────────────────────────────────────────────────────────────────

def test_analyze_forces_weekly_cascade(patched_run):
    """analyze() must call run() with force_weekly=True so the engine
    starts in Phase 2 (skipping the daily 21/34/50 cascade entirely)."""
    from mors.monitor import analyze
    analyze(_pos(), nlv=500_000.0)
    assert len(patched_run["calls"]) == 1
    assert patched_run["calls"][0]["kwargs"].get("force_weekly") is True
    assert patched_run["calls"][0]["kwargs"].get("mode") == "terminate"


# ─────────────────────────────────────────────────────────────────────
# 2-3. Floor schedule + Green-no-sell
# ─────────────────────────────────────────────────────────────────────

def test_green_tier_never_sells(patched_run):
    """GREEN positions never trim — delta_shares is 0 even when the
    position currently sits well above the 15% NLV rebuild target.
    Rebuild (BUY back up to 15%) is intentionally deferred."""
    patched_run["result"] = _engine_result(current_tier_label="GREEN", exit_px=100.0)
    from mors.monitor import analyze
    # 200 sh × $100 = $20,000 = 20% of $100K NLV → well above the 15% floor.
    r = analyze(_pos(shares_held=200.0), nlv=100_000.0)
    assert r["current_tier"] == "GREEN"
    assert r["tier_pct_nlv"] == 15.00
    assert r["delta_dollars"] == 0.0
    assert r["delta_shares"] == 0
    assert r["current_pct_nlv"] == pytest.approx(20.0, abs=0.01)


def test_quick_tier_trims_to_10pct_nlv(patched_run):
    """QUICK floor = 10% NLV. A 20%-of-NLV position trims half of it."""
    patched_run["result"] = _engine_result(current_tier_label="QUICK", exit_px=100.0)
    from mors.monitor import analyze
    # 200 sh × $100 = $20,000 = 20% of $100K NLV. Target = 10% = $10K.
    # delta = $10K → at $100/sh → 100 shares.
    r = analyze(_pos(shares_held=200.0), nlv=100_000.0)
    assert r["current_tier"] == "QUICK"
    assert r["tier_pct_nlv"] == 10.00
    assert r["target_dollars"] == pytest.approx(10_000.0, abs=0.01)
    assert r["delta_dollars"] == pytest.approx(10_000.0, abs=0.01)
    assert r["delta_shares"] == 100


def test_quicksand_tier_trims_to_5pct_nlv(patched_run):
    """QUICKSAND floor = 5% NLV. A 20%-of-NLV position trims 75% of it."""
    patched_run["result"] = _engine_result(current_tier_label="QUICKSAND", exit_px=100.0)
    from mors.monitor import analyze
    r = analyze(_pos(shares_held=200.0), nlv=100_000.0)
    assert r["current_tier"] == "QUICKSAND"
    assert r["tier_pct_nlv"] == 5.00
    assert r["target_dollars"] == pytest.approx(5_000.0, abs=0.01)
    assert r["delta_dollars"] == pytest.approx(15_000.0, abs=0.01)
    assert r["delta_shares"] == 150


def test_gd_tier_exits_to_zero(patched_run):
    """GD floor = 0 — full exit. terminated flag must be True."""
    patched_run["result"] = _engine_result(current_tier_label="GD", exit_px=100.0)
    from mors.monitor import analyze
    r = analyze(_pos(shares_held=200.0), nlv=100_000.0)
    assert r["current_tier"] == "GD"
    assert r["tier_pct_nlv"] == 0.00
    assert r["target_dollars"] == 0.0
    assert r["delta_dollars"] == pytest.approx(20_000.0, abs=0.01)
    assert r["delta_shares"] == 200
    assert r["terminated"] is True


# ─────────────────────────────────────────────────────────────────────
# 4. Live tier is surfaced (distinct from last log emission)
# ─────────────────────────────────────────────────────────────────────

def test_current_tier_surfaced_even_when_log_says_entry(patched_run):
    """A newly-entered SR8 position whose log still ends at ENTRY must
    still expose the live cascade tier (default GREEN) — the frontend
    binds the Signal badge to current_tier, not last_signal."""
    # Engine result: log emission is "ENTRY", but the ratchet's live tier
    # is GREEN (the seeded state in Phase 2 for an above-all-MAs position).
    log = pd.DataFrame([{
        "Date": pd.Timestamp("2026-06-12").date(),
        "Phase": 2, "TF": "weekly",
        "Signal": "ENTRY", "Unit%": 100.0,
    }])
    patched_run["result"] = {
        "ticker": "SNDK",
        "exit_px": 100.0,
        "exit_date": pd.Timestamp("2026-06-12").date(),
        "log": log,
        "current_tier_label": "GREEN",
        "current_tier_idx": 0,
        "current_tier_seeded": True,
    }
    from mors.monitor import analyze
    r = analyze(_pos(ticker="SNDK"), nlv=100_000.0)
    assert r["last_signal"] == "ENTRY"   # log emission unchanged
    assert r["current_tier"] == "GREEN"  # live tier — what the UI badge reads
