"""Tests for nlv_service.compute_twr_returns and the /twr-returns endpoint.

Pure-math tests construct synthetic journal DataFrames so the daily-return /
cumprod logic can be verified without a database. Endpoint integration tests
hit the FastAPI handler directly (skipping HTTP + JWT) and require
DATABASE_URL — they're skipped automatically otherwise.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from nlv_service import _compute_twr_from_journal_df


requires_db = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping endpoint tests",
)


# ---------------------------------------------------------------------------
# Pure-math tests for _compute_twr_from_journal_df
# ---------------------------------------------------------------------------


class TestTwrPureMath:
    def test_empty_df_returns_zeros_with_unavailable_ytd(self) -> None:
        out = _compute_twr_from_journal_df(pd.DataFrame())
        assert out["twr_ltd_pct"] == 0.0
        assert out["twr_ytd_pct"] is None
        assert out["twr_ytd_available"] is False

    def test_chained_daily_returns_no_cash_flow(self) -> None:
        # Two days of +10% returns: (1.1 * 1.1) - 1 = 0.21 = 21%
        today = pd.Timestamp.now().normalize()
        df = pd.DataFrame([
            {"day": today - pd.Timedelta(days=2), "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
            {"day": today - pd.Timedelta(days=1), "beg_nlv": 110.0, "end_nlv": 121.0, "cash_change": 0.0},
        ])
        out = _compute_twr_from_journal_df(df)
        assert abs(out["twr_ltd_pct"] - 21.0) < 0.001

    def test_cash_flow_does_not_distort_twr(self) -> None:
        # Day 1: +10% with no flow.
        # Day 2: deposit $10 at start; portfolio still earns +10% on the
        # adjusted base of 120. End_NLV = 120 * 1.10 = 132.
        # TWR LTD must equal Day-1-only TWR (~21%), unaffected by the deposit.
        today = pd.Timestamp.now().normalize()
        df = pd.DataFrame([
            {"day": today - pd.Timedelta(days=2), "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
            {"day": today - pd.Timedelta(days=1), "beg_nlv": 110.0, "end_nlv": 132.0, "cash_change": 10.0},
        ])
        out = _compute_twr_from_journal_df(df)
        assert abs(out["twr_ltd_pct"] - 21.0) < 0.001

    def test_ytd_filters_to_current_year(self) -> None:
        today = pd.Timestamp.now().normalize()
        last_year_jan = pd.Timestamp(year=today.year - 1, month=1, day=15)
        last_year_jul = pd.Timestamp(year=today.year - 1, month=7, day=1)
        this_year_jan = pd.Timestamp(year=today.year, month=1, day=15)
        df = pd.DataFrame([
            # Last year: +100% then +50% → cumulative *3.0
            {"day": last_year_jan, "beg_nlv": 100.0, "end_nlv": 200.0, "cash_change": 0.0},
            {"day": last_year_jul, "beg_nlv": 200.0, "end_nlv": 300.0, "cash_change": 0.0},
            # This year: +10%
            {"day": this_year_jan, "beg_nlv": 300.0, "end_nlv": 330.0, "cash_change": 0.0},
        ])
        out = _compute_twr_from_journal_df(df)
        # LTD = (2.0 * 1.5 * 1.1) - 1 = 2.30 = 230%
        assert abs(out["twr_ltd_pct"] - 230.0) < 0.001
        # YTD = +10% from a single current-year row
        assert out["twr_ytd_available"] is True
        assert abs(out["twr_ytd_pct"] - 10.0) < 0.001

    def test_ytd_unavailable_when_no_current_year_rows(self) -> None:
        today = pd.Timestamp.now().normalize()
        last_year = pd.Timestamp(year=today.year - 1, month=6, day=1)
        df = pd.DataFrame([
            {"day": last_year, "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
        ])
        out = _compute_twr_from_journal_df(df)
        assert out["twr_ytd_available"] is False
        assert out["twr_ytd_pct"] is None
        assert abs(out["twr_ltd_pct"] - 10.0) < 0.001

    def test_zero_or_negative_adjusted_beg_does_not_distort(self) -> None:
        # First row has beg_nlv=0 (pre-funding entry) — it should contribute
        # daily_return=0 so the cumprod stays at 1 for that step.
        today = pd.Timestamp.now().normalize()
        df = pd.DataFrame([
            {"day": today - pd.Timedelta(days=2), "beg_nlv": 0.0, "end_nlv": 100.0, "cash_change": 0.0},
            {"day": today - pd.Timedelta(days=1), "beg_nlv": 100.0, "end_nlv": 120.0, "cash_change": 0.0},
        ])
        out = _compute_twr_from_journal_df(df)
        # Only day 2 contributes: +20%
        assert abs(out["twr_ltd_pct"] - 20.0) < 0.001

    def test_unsorted_input_is_sorted_by_day(self) -> None:
        today = pd.Timestamp.now().normalize()
        df = pd.DataFrame([
            {"day": today - pd.Timedelta(days=1), "beg_nlv": 110.0, "end_nlv": 121.0, "cash_change": 0.0},
            {"day": today - pd.Timedelta(days=2), "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
        ])
        out = _compute_twr_from_journal_df(df)
        assert abs(out["twr_ltd_pct"] - 21.0) < 0.001

    def test_missing_cash_change_column_treated_as_zero(self) -> None:
        today = pd.Timestamp.now().normalize()
        df = pd.DataFrame([
            {"day": today - pd.Timedelta(days=1), "beg_nlv": 100.0, "end_nlv": 110.0},
        ])
        out = _compute_twr_from_journal_df(df)
        assert abs(out["twr_ltd_pct"] - 10.0) < 0.001


# ---------------------------------------------------------------------------
# Endpoint integration test — requires DATABASE_URL with CanSlim journal data
# ---------------------------------------------------------------------------


@requires_db
def test_compute_twr_returns_canslim_value() -> None:
    """Call the service-layer function directly (skipping the rate-limited
    endpoint wrapper) and assert response shape + that the LTD value falls
    in a sensible range for CanSlim's full history.

    The audit found the correct TWR for CanSlim is approximately 292%. We
    assert a wider band (200%–350%) because real journal data drifts daily;
    a formula regression would push it far outside this range (e.g., the
    old snapshot ratio sat at 228%).
    """
    import db_layer as db
    from nlv_service import compute_twr_returns

    try:
        portfolios = db.list_portfolios()
    except Exception:
        pytest.skip("Could not list portfolios from DB")

    canslim = next((p for p in portfolios if p.get("name") == "CanSlim"), None)
    if canslim is None:
        pytest.skip("CanSlim portfolio not present in this database")

    response = compute_twr_returns(canslim["name"])

    assert "twr_ltd_pct" in response
    assert "twr_ytd_pct" in response
    assert "twr_ytd_available" in response
    assert "as_of" in response

    ltd = response["twr_ltd_pct"]
    assert isinstance(ltd, (int, float))
    assert 200.0 < ltd < 350.0, (
        f"twr_ltd_pct {ltd} outside expected band — "
        "either the formula regressed or CanSlim's data shifted significantly"
    )

    if response["twr_ytd_available"]:
        ytd = response["twr_ytd_pct"]
        assert isinstance(ytd, (int, float))
        assert 0.0 < ytd < 200.0, f"twr_ytd_pct {ytd} outside sensible bounds"
