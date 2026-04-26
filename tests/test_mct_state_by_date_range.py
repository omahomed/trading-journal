"""Phase 4 backend endpoint tests.

Covers two endpoints introduced in Phase 4:
  - /api/journal/mct-state-by-date-range  (per-day V11 state for journal join)
  - /api/market/signals                   (V11 signal log for MCT page)

Calls the FastAPI endpoint functions directly (bypassing HTTP + JWT).
DB tests skip when DATABASE_URL is unset. Both endpoints rely on
real seeded market_data history; the four canonical assertions below
are anchored on real historical bars (^IXIC) from the staging DB.
"""

from __future__ import annotations

import os

import pytest


requires_db = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping endpoint tests",
)


def _make_request():
    """Minimal real starlette Request — slowapi rejects non-Request objects."""
    from starlette.requests import Request
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/api/market/signals",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 0),
    })


# ---------------------------------------------------------------------------
# /api/journal/mct-state-by-date-range
# ---------------------------------------------------------------------------

@requires_db
def test_mct_state_range_returns_states_list(canonical_dependencies):
    from api.main import journal_mct_state_by_date_range
    response = journal_mct_state_by_date_range("2026-04-20", "2026-04-24")
    assert "states" in response
    assert isinstance(response["states"], list)
    assert len(response["states"]) > 0


@requires_db
def test_mct_state_range_row_shape(canonical_dependencies):
    """Every row must carry the V11 fields the journal UI consumes."""
    from api.main import journal_mct_state_by_date_range
    response = journal_mct_state_by_date_range("2026-04-22", "2026-04-24")
    expected = {
        "trade_date", "state", "exposure_ceiling", "cap_at_100",
        "cycle_day", "in_correction", "correction_active", "power_trend",
    }
    for row in response["states"]:
        missing = expected - set(row.keys())
        assert not missing, f"Missing fields in row: {missing}"


@requires_db
def test_mct_state_range_2026_04_24_is_powertrend_day_18(canonical_dependencies):
    """Anchor: 2026-04-24 is mid-Power-Trend, cycle Day 18."""
    from api.main import journal_mct_state_by_date_range
    response = journal_mct_state_by_date_range("2026-04-24", "2026-04-24")
    assert response["states"], "no row returned for 2026-04-24"
    row = response["states"][0]
    assert row["trade_date"] == "2026-04-24"
    assert row["state"] == "POWERTREND"
    assert row["cycle_day"] == 18


@requires_db
def test_mct_state_range_2026_04_16_is_uptrend(canonical_dependencies):
    """Anchor: 2026-04-16 — correction just nullified, Step 8 (PT-ON) hasn't
    fired yet, so the state is UPTREND not POWERTREND."""
    from api.main import journal_mct_state_by_date_range
    response = journal_mct_state_by_date_range("2026-04-16", "2026-04-16")
    assert response["states"], "no row returned for 2026-04-16"
    row = response["states"][0]
    assert row["state"] == "UPTREND"


@requires_db
def test_mct_state_range_2025_12_17_has_cap_at_100(canonical_dependencies):
    """Anchor: 2025-12-17 — correction-active period with a fired Violation,
    so cap_at_100 must be True."""
    from api.main import journal_mct_state_by_date_range
    response = journal_mct_state_by_date_range("2025-12-17", "2025-12-17")
    assert response["states"], "no row returned for 2025-12-17"
    row = response["states"][0]
    assert row["cap_at_100"] is True


@requires_db
def test_mct_state_range_2025_11_21_is_rally_mode(canonical_dependencies):
    """Anchor: 2025-11-21 — STEP_0 just fired, rally hunt active."""
    from api.main import journal_mct_state_by_date_range
    response = journal_mct_state_by_date_range("2025-11-21", "2025-11-21")
    assert response["states"], "no row returned for 2025-11-21"
    row = response["states"][0]
    assert row["state"] == "RALLY MODE"


@requires_db
def test_mct_state_range_inverted_dates_returns_error(canonical_dependencies):
    from api.main import journal_mct_state_by_date_range
    response = journal_mct_state_by_date_range("2026-04-24", "2026-04-20")
    assert "error" in response
    assert response["states"] == []


def test_mct_state_range_bad_date_returns_error():
    """No DB needed — input validation runs before the engine call."""
    from api.main import journal_mct_state_by_date_range
    response = journal_mct_state_by_date_range("not-a-date", "2026-04-24")
    assert "error" in response
    assert response["states"] == []


# ---------------------------------------------------------------------------
# /api/market/signals
# ---------------------------------------------------------------------------

@requires_db
def test_market_signals_returns_signals_list(canonical_dependencies):
    from api.main import get_recent_market_signals
    response = get_recent_market_signals(_make_request(), days=30)
    assert "signals" in response
    assert isinstance(response["signals"], list)


@requires_db
def test_market_signals_row_shape(canonical_dependencies):
    """Every row must carry the V11 fields the MCT signal log displays."""
    from api.main import get_recent_market_signals
    response = get_recent_market_signals(_make_request(), days=365)
    if not response["signals"]:
        pytest.skip("market_signals empty for last 365d — backfill not yet run")
    expected = {
        "trade_date", "signal_type", "signal_label",
        "exposure_before", "exposure_after",
        "state_before", "state_after", "meta",
    }
    for row in response["signals"]:
        missing = expected - set(row.keys())
        assert not missing, f"Missing fields in signal row: {missing}"


@requires_db
def test_market_signals_sorted_desc_by_date(canonical_dependencies):
    from api.main import get_recent_market_signals
    response = get_recent_market_signals(_make_request(), days=365)
    signals = response["signals"]
    if len(signals) < 2:
        pytest.skip("need ≥2 signals to verify sort")
    dates = [s["trade_date"] for s in signals]
    assert dates == sorted(dates, reverse=True), (
        "signals must be sorted desc by trade_date"
    )


@requires_db
def test_market_signals_signal_type_filter(canonical_dependencies):
    """Passing signal_type narrows the result set to that type only."""
    from api.main import get_recent_market_signals
    response = get_recent_market_signals(
        _make_request(), days=365, signal_type="STEP_1_FTD"
    )
    if not response["signals"]:
        pytest.skip("no STEP_1_FTD signals in last 365d")
    for row in response["signals"]:
        assert row["signal_type"] == "STEP_1_FTD"


# ---------------------------------------------------------------------------
# Shared fixture — DB endpoint tests reuse a single engine run for speed.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def canonical_dependencies():
    from api.market_data_repo import get_latest_date
    if get_latest_date("^IXIC") is None:
        pytest.skip("market_data has no ^IXIC bars; run scripts/backfill_market_data.py")
    return True
