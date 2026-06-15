"""Endpoint integration tests for MCT V11.

Calls the FastAPI endpoint functions directly (skipping the HTTP layer +
JWT middleware). Verifies the V11 response shape for /api/market/rally-prefix
plus pure adapter helpers.

The legacy IBD-style market-school endpoint and its frontend component were
deleted in Phase 4; the V11 frontend consumes /api/market/rally-prefix and
/api/market/signals directly. See tests/test_mct_state_by_date_range.py for
the Phase 4 endpoint tests.

DB tests skip when DATABASE_URL is unset.
"""

from __future__ import annotations

import os

import pytest


requires_db = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping endpoint tests",
)


# ---------------------------------------------------------------------------
# /api/market/rally-prefix
# ---------------------------------------------------------------------------

@requires_db
def test_rally_prefix_returns_v11_state(canonical_dependencies):
    from api.main import rally_prefix
    response = rally_prefix()
    assert "state" in response
    assert response["state"] in ("CORRECTION", "RALLY MODE", "UPTREND", "POWERTREND")


@requires_db
def test_rally_prefix_response_has_legacy_fields(canonical_dependencies):
    """Every field the frontend reads must be present in the response."""
    from api.main import rally_prefix
    response = rally_prefix()
    expected = {
        "prefix", "day_num", "state", "entry_step", "entry_exposure",
        "price", "ema8", "ema21", "sma50", "sma200",
        "reference_high", "reference_high_date", "drawdown_pct",
        "consecutive_below_21", "active_exits",
        "low_above_21_streak", "low_above_50_streak",
        "stack_8_21", "stack_21_50", "stack_50_200",
        "entry_ladder", "ftd_date", "data_as_of", "power_trend_on_since",
        "cap_at_100", "cycle_start_date",
    }
    missing = expected - set(response.keys())
    assert not missing, f"Missing fields: {missing}"


@requires_db
def test_rally_prefix_entry_ladder_has_9_steps_with_legacy_exposures(canonical_dependencies):
    from api.main import rally_prefix
    response = rally_prefix()
    ladder = response["entry_ladder"]
    assert len(ladder) == 9
    # Legacy exposure ladder: 20/40/60/80/100/120/140/160/200
    assert [step["exposure"] for step in ladder] == [20, 40, 60, 80, 100, 120, 140, 160, 200]


@requires_db
def test_rally_prefix_historical_as_of_date(canonical_dependencies):
    """Passing as_of_date slices history. 12/17/2025 is mid-correction
    so state should be CORRECTION/RALLY MODE depending on rally state."""
    from api.main import rally_prefix
    response = rally_prefix(as_of_date="2025-12-17")
    assert response["state"] in ("CORRECTION", "RALLY MODE", "UPTREND", "POWERTREND")
    # data_as_of should be on or before 2025-12-17
    assert response["data_as_of"] <= "2025-12-17"


# ---------------------------------------------------------------------------
# market_window deprecation (static checks — don't pollute the journal table)
# ---------------------------------------------------------------------------

# Phase 3b deletes the deprecated helper entirely from api/main.py — any
# attempt to re-introduce it would require redefining the symbol, which the
# verification grep would catch immediately.


# ---------------------------------------------------------------------------
# Adapter-level pure tests (no DB needed)
# ---------------------------------------------------------------------------

def test_state_name_powertrend_takes_precedence():
    from api.mct_endpoint_adapter import _state_name
    assert _state_name({"power_trend": True}) == "POWERTREND"


def test_state_name_uptrend_when_step4_done_outside_correction():
    from api.mct_endpoint_adapter import _state_name
    s = {"power_trend": False, "step4_done": True, "in_correction": False}
    assert _state_name(s) == "UPTREND"


def test_state_name_rally_mode_during_rally_hunt():
    from api.mct_endpoint_adapter import _state_name
    s = {"power_trend": False, "step4_done": False, "rally_active": True,
         "step0_done": True, "in_correction": True}
    assert _state_name(s) == "RALLY MODE"


def test_state_name_correction_default():
    from api.mct_endpoint_adapter import _state_name
    assert _state_name({"power_trend": False}) == "CORRECTION"


def test_entry_ladder_has_9_steps_with_legacy_exposures():
    """Legacy fallback (no signals/cycle_start_date passed) reads live state flags."""
    from api.mct_endpoint_adapter import _entry_ladder
    ladder = _entry_ladder({"step0_done": True, "step1_done": True, "power_trend": False})
    assert len(ladder) == 9
    assert ladder[0]["achieved"] is True
    assert ladder[0]["exposure"] == 20
    assert ladder[8]["achieved"] is False
    assert ladder[8]["exposure"] == 200


def test_entry_ladder_latched_vs_live_contract():
    """Under the sum-of-valid-steps exposure model, the entry ladder's
    per-step achievement comes from a mixed contract:

      Step 0      — latched: state["step0_done"]
      Step 1      — latched: state["step1_done"]
      Steps 2-7   — LIVE: state["live_step_valid"][s], recomputed against
                    the current bar's close/low and MA stack every bar
                    by _phase_exposure_recompute. Old "STEP_N fired
                    in this cycle" signal-log semantics no longer apply.
      Step 8      — latched: state["power_trend"]

    Failure mode this guards: an earlier design read achievement from
    the signal log, which kept STEP_5/6/7 "achieved" forever after a
    single firing in the cycle. Under the live contract, a stack
    inversion immediately drops the corresponding step's checkmark.
    """
    from api.mct_endpoint_adapter import _entry_ladder

    # Mid-cycle synthetic state: step0/step1 latched True, PT off, and a
    # representative live_step_valid mix — 2/3/6 live True; 4/5/7 live
    # False. The ladder must mirror this exactly: 0/1/2/3/6 achieved;
    # 4/5/7/8 not achieved (irrespective of any step{N}_done value).
    state = {
        "step0_done": True,  "step1_done": True,
        # Latched step{N}_done for 2-7 intentionally set to the OPPOSITE
        # of the live flag to prove the adapter reads live, not latched.
        "step2_done": False, "step3_done": False, "step4_done": True,
        "step5_done": True,  "step6_done": False, "step7_done": True,
        "power_trend": False,
        "live_step_valid": {2: True, 3: True, 4: False,
                            5: False, 6: True, 7: False},
    }

    ladder = _entry_ladder(state)
    achieved = [ladder[i]["achieved"] for i in range(9)]
    assert achieved == [True, True, True, True, False,
                        False, True, False, False], achieved


def test_entry_ladder_cycle_active_gate_via_step0_done():
    """The user-facing "cycle active?" gate is now step0_done — not a
    signal-log cycle_start_date lookup. When step0_done is False, the
    rally cycle hasn't started (or has been formally invalidated /
    declared into a correction) and the engine's exposure recompute
    short-circuits to 0. The ladder reflects that gate directly:
    Step 0 reads False, and step1_done / live conditions / power_trend
    keep reading whatever the engine currently has — they're not
    masked by the gate at the ladder layer.
    """
    from api.mct_endpoint_adapter import _entry_ladder

    # No active cycle: step0_done False. Engine would zero exposure;
    # the ladder should show Step 0 unchecked. Other flags pass through.
    state = {
        "step0_done": False, "step1_done": False,
        "step2_done": False, "step3_done": False, "step4_done": False,
        "step5_done": False, "step6_done": False, "step7_done": False,
        "power_trend": False,
        "live_step_valid": {2: False, 3: False, 4: False,
                            5: False, 6: False, 7: False},
    }
    ladder = _entry_ladder(state)
    assert all(not ladder[i]["achieved"] for i in range(9))

    # An active cycle (step0_done=True) but no other progress yet: only
    # Step 0 lights up. Same shape the engine produces on the bar that
    # STEP_0_RALLY_DAY fires (e.g. 2026-03-31).
    state["step0_done"] = True
    ladder = _entry_ladder(state)
    assert ladder[0]["achieved"] is True
    assert all(not ladder[i]["achieved"] for i in range(1, 9))


def test_market_state_for_journal_excludes_market_window():
    """Phase 3a deprecation: market_window MUST NOT appear in this snapshot."""
    from api.mct_engine import EngineResult
    from api.mct_endpoint_adapter import to_market_state_for_journal

    # Synthetic minimal final_state
    fake_state = {
        "power_trend": False, "step4_done": True, "in_correction": False,
        "exposure": 100, "cap_at_100": False,
    }
    import pandas as pd
    fake = EngineResult(bars=pd.DataFrame(), signals=[], final_state=fake_state)
    out = to_market_state_for_journal(fake)
    assert "market_window" not in out
    assert out["market_state"] == "UPTREND"
    assert out["exposure_ceiling"] == 100


# ---------------------------------------------------------------------------
# Shared fixture — DB endpoint tests reuse a single engine run for speed.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def canonical_dependencies():
    """Smoke check that market_data has rows; if not, skip the whole module's
    DB-dependent tests."""
    from api.market_data_repo import get_latest_date
    if get_latest_date("^IXIC") is None:
        pytest.skip("market_data has no ^IXIC bars; run scripts/backfill_market_data.py")
    return True
