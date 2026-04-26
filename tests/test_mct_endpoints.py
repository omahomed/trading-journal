"""Endpoint integration tests for MCT V11.

Calls the FastAPI endpoint functions directly (skipping the HTTP layer +
JWT middleware). Verifies the V11 response shape for /api/market/rally-prefix
plus pure adapter helpers.

The /api/market/ibd endpoint and the IBDMarketSchool frontend component were
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
    from api.mct_endpoint_adapter import _entry_ladder
    ladder = _entry_ladder({"step0_done": True, "step1_done": True, "power_trend": False})
    assert len(ladder) == 9
    assert ladder[0]["achieved"] is True
    assert ladder[0]["exposure"] == 20
    assert ladder[8]["achieved"] is False
    assert ladder[8]["exposure"] == 200


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
