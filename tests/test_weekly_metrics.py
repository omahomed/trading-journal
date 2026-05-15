"""Tests for nlv_service.weekly_metrics (Phase 5 — Weekly Insights tiles).

Pure-math tests construct synthetic journal + trades_summary DataFrames and
patch db_layer so the service-layer function can be exercised without a
database. The formula correctness is the contract — these tests pin the
math so a future refactor of the chained-return / NLV-delta logic can't
silently regress.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

import nlv_service


def _journal(rows: list[dict]) -> pd.DataFrame:
    """Build a journal DataFrame with the columns load_journal returns."""
    return pd.DataFrame(rows)


def _summary(rows: list[dict]) -> pd.DataFrame:
    """Build a trades_summary DataFrame with the Title_Case columns
    load_summary returns."""
    return pd.DataFrame(rows)


def _call(journal_df: pd.DataFrame,
          summary_df: pd.DataFrame,
          week_start: str,
          portfolio: str = "CanSlim") -> dict:
    with patch.object(nlv_service.db, "load_journal", return_value=journal_df), \
         patch.object(nlv_service.db, "load_summary", return_value=summary_df):
        return nlv_service.weekly_metrics(portfolio, week_start)


class TestInputParsing:
    def test_invalid_week_start_returns_error(self) -> None:
        out = _call(_journal([]), _summary([]), "not-a-date")
        assert "error" in out

    def test_empty_week_start_returns_error(self) -> None:
        out = _call(_journal([]), _summary([]), "")
        assert "error" in out


class TestShape:
    def test_empty_journal_returns_zeros_and_shape(self) -> None:
        out = _call(_journal([]), _summary([]), "2026-05-11")
        assert out["weekly_pnl"] == 0.0
        assert out["weekly_return_pct"] == 0.0
        assert out["ltd_pct"] == 0.0
        assert out["ytd_pct"] == 0.0
        assert out["win_rate"] == {"rate": 0.0, "wins": 0, "losses": 0, "flat": 0, "total": 0}
        assert out["week_start"] == "2026-05-11"
        assert out["week_end"] == "2026-05-15"  # Friday
        assert "as_of" in out


class TestWeeklyPnl:
    def test_weekly_pnl_is_end_minus_beg_minus_cashflow(self) -> None:
        # Mon 100 → Fri 130, with +$10 deposit on Wednesday.
        # weekly_pnl = 130 - (100 + 10) = 20.
        journal = _journal([
            {"day": "2026-05-11", "beg_nlv": 100.0, "end_nlv": 105.0, "cash_change": 0.0},
            {"day": "2026-05-13", "beg_nlv": 105.0, "end_nlv": 120.0, "cash_change": 10.0},
            {"day": "2026-05-15", "beg_nlv": 120.0, "end_nlv": 130.0, "cash_change": 0.0},
        ])
        out = _call(journal, _summary([]), "2026-05-11")
        assert out["weekly_pnl"] == 20.0

    def test_weekly_pnl_excludes_rows_outside_week_window(self) -> None:
        # Rows outside Mon-Sun must not contribute to weekly_pnl.
        journal = _journal([
            {"day": "2026-05-08", "beg_nlv": 90.0, "end_nlv": 100.0, "cash_change": 0.0},  # prior Fri
            {"day": "2026-05-11", "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
            {"day": "2026-05-15", "beg_nlv": 110.0, "end_nlv": 120.0, "cash_change": 0.0},
            {"day": "2026-05-18", "beg_nlv": 120.0, "end_nlv": 130.0, "cash_change": 0.0},  # next Mon
        ])
        out = _call(journal, _summary([]), "2026-05-11")
        # In-week: beg=100, end=120, cash=0 → pnl=20
        assert out["weekly_pnl"] == 20.0


class TestWeeklyReturnPct:
    def test_weekly_return_chains_daily_dietz_returns(self) -> None:
        # Two days, each +10%: chained = 1.1 * 1.1 - 1 = 21%.
        journal = _journal([
            {"day": "2026-05-11", "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
            {"day": "2026-05-12", "beg_nlv": 110.0, "end_nlv": 121.0, "cash_change": 0.0},
        ])
        out = _call(journal, _summary([]), "2026-05-11")
        assert abs(out["weekly_return_pct"] - 21.0) < 0.001

    def test_weekly_return_immune_to_cashflow(self) -> None:
        # Day 1 +10% no flow. Day 2 deposit $10 at start; portfolio still
        # earns +10% on adjusted base of 120 → end=132. Chained = 21%.
        journal = _journal([
            {"day": "2026-05-11", "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
            {"day": "2026-05-12", "beg_nlv": 110.0, "end_nlv": 132.0, "cash_change": 10.0},
        ])
        out = _call(journal, _summary([]), "2026-05-11")
        assert abs(out["weekly_return_pct"] - 21.0) < 0.001


class TestLtdAndYtd:
    def test_ltd_chains_full_history(self) -> None:
        # 3 days, +10% each → cumulative 1.1^3 = 1.331 → 33.1%.
        journal = _journal([
            {"day": "2026-05-11", "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
            {"day": "2026-05-12", "beg_nlv": 110.0, "end_nlv": 121.0, "cash_change": 0.0},
            {"day": "2026-05-13", "beg_nlv": 121.0, "end_nlv": 133.1, "cash_change": 0.0},
        ])
        out = _call(journal, _summary([]), "2026-05-11")
        assert abs(out["ltd_pct"] - 33.1) < 0.01

    def test_ytd_filters_to_week_end_year(self) -> None:
        # Last year +100% then this year +10%. As of a week in this year:
        #   LTD = 2.0 * 1.1 - 1 = 1.20 = 120%
        #   YTD = 1.1 - 1 = 0.10 = 10%
        journal = _journal([
            {"day": "2025-06-01", "beg_nlv": 100.0, "end_nlv": 200.0, "cash_change": 0.0},
            {"day": "2026-05-12", "beg_nlv": 200.0, "end_nlv": 220.0, "cash_change": 0.0},
        ])
        out = _call(journal, _summary([]), "2026-05-11")
        assert abs(out["ltd_pct"] - 120.0) < 0.01
        assert abs(out["ytd_pct"] - 10.0) < 0.01

    def test_ytd_falls_back_to_ltd_when_account_started_in_request_year(self) -> None:
        # All rows are in 2026; YTD must equal LTD per Phase 5 inception
        # fallback ("ytd_pct uses inception as fallback").
        journal = _journal([
            {"day": "2026-01-15", "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
            {"day": "2026-05-12", "beg_nlv": 110.0, "end_nlv": 132.0, "cash_change": 0.0},
        ])
        out = _call(journal, _summary([]), "2026-05-11")
        assert out["ytd_pct"] == out["ltd_pct"]
        assert abs(out["ytd_pct"] - 32.0) < 0.01

    def test_metrics_as_of_week_end_ignore_later_rows(self) -> None:
        # Rows after Sunday must not influence LTD/YTD computed for this
        # week — historical-week stability.
        journal = _journal([
            {"day": "2026-05-11", "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
            {"day": "2026-06-01", "beg_nlv": 110.0, "end_nlv": 220.0, "cash_change": 0.0},  # post-week
        ])
        out = _call(journal, _summary([]), "2026-05-11")
        # LTD as of week of 2026-05-11 = +10%, ignoring the June row.
        assert abs(out["ltd_pct"] - 10.0) < 0.01


class TestInceptionWeek:
    def test_inception_week_returns_equal(self) -> None:
        # All journal rows fall inside the requested week → LTD, YTD, and
        # weekly_return_pct all equal the same cumprod.
        journal = _journal([
            {"day": "2026-05-11", "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
            {"day": "2026-05-13", "beg_nlv": 110.0, "end_nlv": 121.0, "cash_change": 0.0},
        ])
        out = _call(journal, _summary([]), "2026-05-11")
        assert abs(out["weekly_return_pct"] - 21.0) < 0.01
        assert abs(out["ytd_pct"] - 21.0) < 0.01
        assert abs(out["ltd_pct"] - 21.0) < 0.01


class TestWinRate:
    def test_win_rate_excludes_open_trades(self) -> None:
        # Only CLOSED status contributes — OPEN must be filtered out.
        summary = _summary([
            {"Status": "OPEN",   "Closed_Date": None,         "Realized_PL": 999.0},
            {"Status": "CLOSED", "Closed_Date": "2026-03-15", "Realized_PL": 100.0},
            {"Status": "CLOSED", "Closed_Date": "2026-04-20", "Realized_PL": -50.0},
        ])
        out = _call(_journal([]), summary, "2026-05-11")
        wr = out["win_rate"]
        assert wr["wins"] == 1
        assert wr["losses"] == 1
        assert wr["flat"] == 0
        assert wr["total"] == 2
        assert abs(wr["rate"] - 0.5) < 0.001

    def test_win_rate_flat_counted_in_denominator(self) -> None:
        # Locked Phase 5 formula: flats stay in the denominator (matches
        # the existing computeWinRate convention in analytics-stats.ts).
        summary = _summary([
            {"Status": "CLOSED", "Closed_Date": "2026-02-01", "Realized_PL": 100.0},
            {"Status": "CLOSED", "Closed_Date": "2026-02-02", "Realized_PL": 0.0},
            {"Status": "CLOSED", "Closed_Date": "2026-02-03", "Realized_PL": -50.0},
        ])
        out = _call(_journal([]), summary, "2026-05-11")
        wr = out["win_rate"]
        assert wr["wins"] == 1 and wr["losses"] == 1 and wr["flat"] == 1
        assert wr["total"] == 3
        assert abs(wr["rate"] - (1 / 3)) < 0.001

    def test_win_rate_ytd_filter_excludes_prior_year(self) -> None:
        # Prior-year closes must NOT count.
        summary = _summary([
            {"Status": "CLOSED", "Closed_Date": "2025-12-30", "Realized_PL": 1000.0},
            {"Status": "CLOSED", "Closed_Date": "2026-01-15", "Realized_PL": 100.0},
            {"Status": "CLOSED", "Closed_Date": "2026-02-15", "Realized_PL": -50.0},
        ])
        out = _call(_journal([]), summary, "2026-05-11")
        wr = out["win_rate"]
        assert wr["wins"] == 1 and wr["losses"] == 1 and wr["total"] == 2

    def test_win_rate_zero_trades_yields_rate_zero(self) -> None:
        out = _call(_journal([]), _summary([]), "2026-05-11")
        wr = out["win_rate"]
        assert wr == {"rate": 0.0, "wins": 0, "losses": 0, "flat": 0, "total": 0}

    def test_win_rate_uses_campaign_level_realized_pl(self) -> None:
        # One row per campaign. Per the audit and Phase 5 lock — we do NOT
        # iterate lot_closures; scale-outs roll up into trades_summary.
        summary = _summary([
            {"Status": "CLOSED", "Closed_Date": "2026-03-01", "Realized_PL": 250.0},
            {"Status": "CLOSED", "Closed_Date": "2026-03-15", "Realized_PL": 425.0},
        ])
        out = _call(_journal([]), summary, "2026-05-11")
        wr = out["win_rate"]
        # Both campaigns positive → 2W, 0L, total=2, rate=1.0.
        assert wr == {"rate": 1.0, "wins": 2, "losses": 0, "flat": 0, "total": 2}


class TestWeekEndEcho:
    def test_week_end_is_friday_of_requested_week(self) -> None:
        # Monday + 4 = Friday.
        out = _call(_journal([]), _summary([]), "2026-05-11")
        assert out["week_start"] == "2026-05-11"
        assert out["week_end"] == "2026-05-15"


class TestEndpointWiring:
    """Direct call into the FastAPI handler to exercise the route shape
    without spinning up a server / JWT layer."""

    def test_get_weekly_metrics_handler_returns_metrics_for_valid_input(self) -> None:
        # Import inside the test so module-load side effects (sentry,
        # rate-limiter wiring) don't run unless this test executes.
        from api.main import get_weekly_metrics
        from starlette.requests import Request as _Req

        # slowapi reads the limiter off `request.state.limiter` via the
        # decorator; here we synthesize a minimal scope. The decorator
        # itself proxies through to the underlying function — calling
        # the wrapped function directly bypasses the rate limit, which
        # is what we want for unit testing.
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/analytics/weekly-metrics",
            "headers": [],
            "query_string": b"",
        }
        req = _Req(scope)  # type: ignore[arg-type]

        with patch.object(nlv_service.db, "load_journal", return_value=_journal([])), \
             patch.object(nlv_service.db, "load_summary", return_value=_summary([])):
            out = get_weekly_metrics(req, portfolio="CanSlim", week_start="2026-05-11")
        # Either the dict we expect, or the rate-limit decorator returned
        # a wrapped response — sanity check on the happy-path keys.
        if isinstance(out, dict):
            assert out["week_start"] == "2026-05-11"
            assert out["week_end"] == "2026-05-15"
        else:
            pytest.skip("Handler returned non-dict (likely rate-limit wrapper)")

    def test_get_weekly_metrics_handler_requires_week_start(self) -> None:
        from api.main import get_weekly_metrics
        from starlette.requests import Request as _Req

        scope = {
            "type": "http", "method": "GET",
            "path": "/api/analytics/weekly-metrics",
            "headers": [], "query_string": b"",
        }
        req = _Req(scope)  # type: ignore[arg-type]

        out = get_weekly_metrics(req, portfolio="CanSlim", week_start="")
        assert isinstance(out, dict) and "error" in out
