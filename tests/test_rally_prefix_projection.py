"""Tests for _project_rally_prefix_for_data_lag — the helper that bumps
day_num forward when the requested date sits past the latest ingested
market_data bar.

These tests exercise the helper as a pure function with canned response
dicts, so they run without a database (no DATABASE_URL gating). The
endpoint-level tests in test_mct_endpoints.py still cover the live path.
"""
from __future__ import annotations

from datetime import date

import pytest

from api.main import (
    _project_rally_prefix_for_data_lag,
    _RALLY_PREFIX_MAX_PROJECTION_DAYS,
)


def _resp(**overrides) -> dict:
    """Build a representative rally-prefix response. Defaults match the
    user's prod state on 2026-04-24: POWERTREND, Day 18, anchored to
    3/31/2026 STEP_0."""
    base = {
        "prefix": "Day 18: ",
        "day_num": 18,
        "state": "POWERTREND",
        "data_as_of": "2026-04-24",
        "cap_at_100": False,
        "cycle_start_date": "2026-03-31",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path — the bug we're fixing
# ---------------------------------------------------------------------------


def test_projects_one_trading_day_forward(monkeypatch):
    """User opens routine on Mon 4/27 before today's bar ingests. Engine's
    latest is Fri 4/24 = Day 18 → projection bumps to Day 19 for Monday."""
    out = _project_rally_prefix_for_data_lag(_resp(), date(2026, 4, 27))
    assert out["day_num"] == 19
    assert out["prefix"] == "Day 19: "
    assert out["day_num_projected"] is True
    assert out["day_num_projection_offset"] == 1


def test_projects_multiple_trading_days(monkeypatch):
    """Data lag of two trading days — Mon picked, latest is Thu 4/23.
    Friday + Monday = 2 trading days projected."""
    out = _project_rally_prefix_for_data_lag(
        _resp(data_as_of="2026-04-23", day_num=17, prefix="Day 17: "),
        date(2026, 4, 27),
    )
    assert out["day_num"] == 19  # 17 + Fri + Mon
    assert out["day_num_projection_offset"] == 2


def test_does_not_count_weekend_days():
    """Saturday/Sunday are not trading days. Latest bar Fri 4/24, picking
    Saturday 4/25 → no offset, day_num unchanged."""
    out = _project_rally_prefix_for_data_lag(_resp(), date(2026, 4, 25))
    assert out["day_num"] == 18  # unchanged
    assert "day_num_projected" not in out


def test_does_not_count_sunday():
    out = _project_rally_prefix_for_data_lag(_resp(), date(2026, 4, 26))
    assert out["day_num"] == 18


# ---------------------------------------------------------------------------
# No-op paths — must not mutate the response
# ---------------------------------------------------------------------------


def test_no_projection_when_requested_date_is_none():
    """Default endpoint call (no date param) — no projection."""
    out = _project_rally_prefix_for_data_lag(_resp(), None)
    assert out["day_num"] == 18
    assert "day_num_projected" not in out


def test_no_projection_when_requested_date_equals_data_date():
    """Engine's latest bar already matches what user asked for."""
    out = _project_rally_prefix_for_data_lag(_resp(), date(2026, 4, 24))
    assert out["day_num"] == 18
    assert "day_num_projected" not in out


def test_no_projection_when_requested_date_is_in_the_past():
    """Historical query — engine has already sliced its bars to that date,
    day_num is correct as-is."""
    out = _project_rally_prefix_for_data_lag(_resp(), date(2026, 4, 20))
    assert out["day_num"] == 18  # whatever engine returned, untouched
    assert "day_num_projected" not in out


def test_no_projection_during_correction():
    """day_num is meaningless during corrections — projecting forward would
    fabricate a number with no semantic basis. Skip."""
    resp = _resp(state="CORRECTION", day_num=0, prefix="CORRECTION: ")
    out = _project_rally_prefix_for_data_lag(resp, date(2026, 4, 27))
    assert out["day_num"] == 0
    assert out["prefix"] == "CORRECTION: "
    assert "day_num_projected" not in out


def test_no_projection_when_day_num_is_zero():
    """Even in an active state, day_num=0 means the engine couldn't
    establish a cycle — projecting 0+1=1 would be meaningless."""
    out = _project_rally_prefix_for_data_lag(
        _resp(day_num=0, prefix=""), date(2026, 4, 27),
    )
    assert out["day_num"] == 0
    assert "day_num_projected" not in out


def test_no_projection_when_data_as_of_missing():
    """Defensive — if the response is malformed, leave it alone rather
    than guess."""
    resp = _resp()
    del resp["data_as_of"]
    out = _project_rally_prefix_for_data_lag(resp, date(2026, 4, 27))
    assert out["day_num"] == 18


def test_invalid_data_as_of_format_does_not_crash():
    """Defensive — junk in data_as_of shouldn't blow up the endpoint."""
    out = _project_rally_prefix_for_data_lag(
        _resp(data_as_of="not-a-date"), date(2026, 4, 27),
    )
    assert out["day_num"] == 18


# ---------------------------------------------------------------------------
# Bound — projection capped so a stuck cron can't inflate numbers
# ---------------------------------------------------------------------------


def test_projection_capped_at_max_days():
    """7 trading days requested — capped at MAX_PROJECTION_DAYS (5).
    Better to under-claim than over-claim when the data feed is broken."""
    # Latest bar 2026-04-17 (Fri); ask for 2026-04-28 (Tue). Trading days
    # in that window: 4/20, 4/21, 4/22, 4/23, 4/24, 4/27, 4/28 = 7 days.
    # Cap should clamp at 5.
    out = _project_rally_prefix_for_data_lag(
        _resp(data_as_of="2026-04-17", day_num=14, prefix="Day 14: "),
        date(2026, 4, 28),
    )
    assert out["day_num_projection_offset"] == _RALLY_PREFIX_MAX_PROJECTION_DAYS
    assert out["day_num"] == 14 + _RALLY_PREFIX_MAX_PROJECTION_DAYS  # 19


# ---------------------------------------------------------------------------
# Prefix shape — only standard "Day N: " gets rebuilt
# ---------------------------------------------------------------------------


def test_non_standard_prefix_passes_through_untouched():
    """Custom prefix strings (e.g. test fixtures, unusual states) keep
    their shape — only the canonical 'Day N: ' form is rebuilt."""
    out = _project_rally_prefix_for_data_lag(
        _resp(prefix="POWERTREND - 18 days in: "),
        date(2026, 4, 27),
    )
    # day_num still bumps so the diagnostic flags are useful
    assert out["day_num"] == 19
    # but the prefix string is left alone
    assert out["prefix"] == "POWERTREND - 18 days in: "


def test_uptrend_state_also_projects():
    """All three rally-active states project; only CORRECTION skips."""
    for state in ("UPTREND", "POWERTREND", "RALLY MODE"):
        resp = _resp(state=state)
        out = _project_rally_prefix_for_data_lag(resp, date(2026, 4, 27))
        assert out["day_num"] == 19, f"projection failed for state={state}"
        assert out["day_num_projected"] is True
