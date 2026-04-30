"""Unit tests for pure trade calculation helpers in trade_calc.py.

These exercise the core accounting logic (LIFO matching, risk budgeting,
journal column normalization) without a database or FastAPI app.
"""
from __future__ import annotations

import pandas as pd
import pytest

from trade_calc import (
    calc_risk_budget,
    compute_lifo_summary,
    is_option_ticker,
    multiplier_for_ticker,
    normalize_journal_columns,
)


class TestCalcRiskBudget:
    def test_standard(self) -> None:
        assert calc_risk_budget(100, 50, 48) == 200.0

    def test_zero_shares(self) -> None:
        assert calc_risk_budget(0, 50, 48) == 0.0

    def test_zero_stop(self) -> None:
        assert calc_risk_budget(100, 50, 0) == 0.0

    def test_negative_stop_treated_as_missing(self) -> None:
        assert calc_risk_budget(100, 50, -1) == 0.0

    def test_stop_above_entry(self) -> None:
        """Stop at/above entry means no meaningful long-side risk — return 0."""
        assert calc_risk_budget(100, 50, 55) == 0.0

    def test_stop_equals_entry(self) -> None:
        assert calc_risk_budget(100, 50, 50) == 0.0

    def test_fractional_shares(self) -> None:
        assert calc_risk_budget(12.5, 100, 95) == 62.5

    def test_rounds_to_cents(self) -> None:
        # 100 * (50.123 - 48.456) = 166.7 exactly
        assert calc_risk_budget(100, 50.123, 48.456) == 166.7

    def test_rounds_sub_cent_to_cents(self) -> None:
        assert calc_risk_budget(3, 10.1234, 10.0) == 0.37


class TestNormalizeJournalColumns:
    def test_renames_known_columns(self) -> None:
        df = pd.DataFrame({"Beg NLV": [10000], "End NLV": [11000]})
        result = normalize_journal_columns(df)
        assert list(result.columns) == ["beg_nlv", "end_nlv"]

    def test_unknown_columns_pass_through(self) -> None:
        df = pd.DataFrame({"custom_field": [1], "Beg NLV": [10000]})
        result = normalize_journal_columns(df)
        assert "custom_field" in result.columns
        assert "beg_nlv" in result.columns

    def test_already_normalized_columns_unchanged(self) -> None:
        df = pd.DataFrame({"beg_nlv": [10000], "end_nlv": [11000]})
        result = normalize_journal_columns(df)
        assert list(result.columns) == ["beg_nlv", "end_nlv"]

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        result = normalize_journal_columns(df)
        assert result.empty

    def test_preserves_values(self) -> None:
        df = pd.DataFrame({"Beg NLV": [10000.5], "End NLV": [11000.25]})
        result = normalize_journal_columns(df)
        assert result["beg_nlv"].iloc[0] == 10000.5
        assert result["end_nlv"].iloc[0] == 11000.25


class TestComputeLifoSummary:
    @staticmethod
    def _txns(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_empty_df_returns_none(self) -> None:
        df = pd.DataFrame(columns=["date", "action", "shares", "amount"])
        assert compute_lifo_summary(df, "T1", "AAPL") is None

    def test_all_null_dates_returns_none(self) -> None:
        df = self._txns([
            {"date": None, "action": "BUY", "shares": 100, "amount": 50.0},
        ])
        assert compute_lifo_summary(df, "T1", "AAPL") is None

    def test_single_buy_opens_position(self) -> None:
        df = self._txns([
            {"date": "2026-01-15", "action": "BUY", "shares": 100, "amount": 50.0},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        assert result is not None
        assert result["Status"] == "OPEN"
        assert result["Shares"] == 100.0
        assert result["Avg_Entry"] == 50.0
        assert result["Realized_PL"] == 0.0
        assert result["Closed_Date"] is None
        assert result["Open_Date"] == "2026-01-15"
        assert result["Trade_ID"] == "T1"
        assert result["Ticker"] == "AAPL"

    def test_full_sell_closes_trade(self) -> None:
        df = self._txns([
            {"date": "2026-01-15", "action": "BUY", "shares": 100, "amount": 50.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 100, "amount": 60.0},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        assert result["Status"] == "CLOSED"
        assert result["Realized_PL"] == 1000.0
        assert result["Closed_Date"] == "2026-01-20"
        assert result["Return_Pct"] == pytest.approx(20.0)
        # Closed trades report total bought shares, not remaining (0)
        assert result["Shares"] == 100.0

    def test_lifo_matches_newest_lot_first(self) -> None:
        """LIFO: a sell eats the most recent buy lot before older ones."""
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY", "shares": 100, "amount": 50.0},
            {"date": "2026-01-15", "action": "BUY", "shares": 50, "amount": 55.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 50, "amount": 60.0},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        # Sell takes the 50-share lot @ 55 → realized = 50 * (60 - 55) = 250
        assert result["Realized_PL"] == 250.0
        # Remaining: 100 @ 50 (the older lot)
        assert result["Shares"] == 100.0
        assert result["Avg_Entry"] == 50.0
        assert result["Status"] == "OPEN"

    def test_sell_spans_two_lots(self) -> None:
        """Sell of 80 consumes the 50-share newest lot and 30 of the oldest."""
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY", "shares": 100, "amount": 50.0},
            {"date": "2026-01-15", "action": "BUY", "shares": 50, "amount": 55.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 80, "amount": 60.0},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        # 50 * (60 - 55) = 250 + 30 * (60 - 50) = 300 → 550
        assert result["Realized_PL"] == 550.0
        # Remaining: 70 shares from the oldest lot, still @ 50
        assert result["Shares"] == 70.0
        assert result["Avg_Entry"] == 50.0

    def test_scale_in_weights_avg_entry(self) -> None:
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY", "shares": 100, "amount": 50.0},
            {"date": "2026-01-15", "action": "BUY", "shares": 100, "amount": 60.0},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        assert result["Shares"] == 200.0
        assert result["Avg_Entry"] == 55.0
        assert result["Status"] == "OPEN"

    def test_losing_trade_realized_pl_negative(self) -> None:
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY", "shares": 100, "amount": 50.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 100, "amount": 45.0},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        assert result["Status"] == "CLOSED"
        assert result["Realized_PL"] == -500.0
        assert result["Return_Pct"] == pytest.approx(-10.0)

    def test_out_of_order_dates_are_sorted(self) -> None:
        """Transactions sort by date before LIFO, regardless of input order."""
        df = self._txns([
            {"date": "2026-01-20", "action": "SELL", "shares": 50, "amount": 60.0},
            {"date": "2026-01-10", "action": "BUY", "shares": 100, "amount": 50.0},
            {"date": "2026-01-15", "action": "BUY", "shares": 50, "amount": 55.0},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        # After sorting: buy 100@50, buy 50@55, sell 50@60
        # LIFO: sell takes the 50@55 → realized = 250
        assert result["Realized_PL"] == 250.0
        assert result["Shares"] == 100.0

    def test_partial_sell_then_second_buy(self) -> None:
        """After a partial sell, a new buy creates a fresh lot on top of LIFO stack."""
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY", "shares": 100, "amount": 50.0},
            {"date": "2026-01-15", "action": "SELL", "shares": 40, "amount": 55.0},
            {"date": "2026-01-20", "action": "BUY", "shares": 30, "amount": 52.0},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        # Realized from the partial sell: 40 * (55 - 50) = 200
        assert result["Realized_PL"] == 200.0
        # Remaining: 60 @ 50 + 30 @ 52 = 90 shares, total cost 3000 + 1560 = 4560, avg 50.6667
        assert result["Shares"] == 90.0
        assert result["Avg_Entry"] == pytest.approx(50.6667, abs=1e-4)
        assert result["Status"] == "OPEN"


class TestOptionsMultiplier:
    """Equity options use multiplier=100; cost basis and P&L scale by it,
    return % stays invariant. Mirrors the RKLB 260618 $80C case from the
    bug report (6 contracts × $12.08 → $7.78)."""

    def test_is_option_ticker_recognizes_readable_format(self) -> None:
        assert is_option_ticker("RKLB 260618 $80C") is True
        assert is_option_ticker("ARM 260618 $175C") is True
        assert is_option_ticker("LUMN 260717 $8.5P") is True

    def test_is_option_ticker_rejects_stocks(self) -> None:
        assert is_option_ticker("AAPL") is False
        assert is_option_ticker("BRK.B") is False
        assert is_option_ticker("") is False
        assert is_option_ticker(None) is False

    def test_multiplier_for_ticker_routes_options_to_100(self) -> None:
        assert multiplier_for_ticker("RKLB 260618 $80C") == 100.0
        assert multiplier_for_ticker("AAPL") == 1.0

    def test_calc_risk_budget_scales_by_multiplier(self) -> None:
        # 6 contracts × ($12 entry - $6 stop) × 100 = $3,600 at risk
        assert calc_risk_budget(6, 12.0, 6.0, multiplier=100) == 3600.0

    def test_lifo_full_loss_scales_realized_pl(self) -> None:
        # The RKLB scenario: 6 contracts buy @ 12.08, sell @ 7.78
        df = pd.DataFrame([
            {"date": "2026-04-27", "action": "BUY", "shares": 6, "amount": 12.08},
            {"date": "2026-04-29", "action": "SELL", "shares": 6, "amount": 7.78},
        ])
        result = compute_lifo_summary(df, "T1", "RKLB 260618 $80C", multiplier=100)
        assert result["Status"] == "CLOSED"
        assert result["Realized_PL"] == pytest.approx(-2580.0)
        assert result["Total_Cost"] == pytest.approx(7248.0)
        assert result["Return_Pct"] == pytest.approx(-35.5960, abs=1e-3)
        assert result["Avg_Entry"] == 12.08
        assert result["Avg_Exit"] == 7.78

    def test_lifo_open_option_position_scales_total_cost(self) -> None:
        df = pd.DataFrame([
            {"date": "2026-04-27", "action": "BUY", "shares": 13, "amount": 3.15},
        ])
        result = compute_lifo_summary(df, "T1", "LUMN 270115 $7C", multiplier=100)
        assert result["Status"] == "OPEN"
        assert result["Total_Cost"] == pytest.approx(4095.0)
        assert result["Shares"] == 13.0
        assert result["Avg_Entry"] == 3.15

    def test_lifo_default_multiplier_unchanged(self) -> None:
        """Stocks (multiplier=1, the default) keep the original behavior."""
        df = pd.DataFrame([
            {"date": "2026-01-10", "action": "BUY", "shares": 100, "amount": 50.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 100, "amount": 60.0},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        assert result["Realized_PL"] == 1000.0
        assert result["Total_Cost"] == 5000.0
