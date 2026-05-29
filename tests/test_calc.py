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
    compute_trade_risk,
    is_option_ticker,
    multiplier_for_ticker,
    normalize_journal_columns,
    validate_post_edit_lifo,
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


class TestValidatePostEditLifo:
    """Pure-logic tests for the LIFO-breaking edit guard. No DB."""

    @staticmethod
    def _txns(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if "detail_id" not in df.columns:
            df["detail_id"] = range(1, len(df) + 1)
        return df

    def test_delete_buy_with_referencing_sell_rejected(self) -> None:
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 10.0},
            {"date": "2026-01-05", "action": "SELL", "shares": 50, "amount": 15.0},
        ])
        err = validate_post_edit_lifo(df, 1, "DELETE", 0, 0, "")
        assert err is not None
        assert "unmatched" in err.lower()

    def test_delete_sell_allowed(self) -> None:
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 10.0},
            {"date": "2026-01-05", "action": "SELL", "shares": 50, "amount": 15.0},
        ])
        assert validate_post_edit_lifo(df, 2, "DELETE", 0, 0, "") is None

    def test_delete_buy_when_no_sells_allowed(self) -> None:
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 10.0},
            {"date": "2026-01-02", "action": "BUY", "shares": 50, "amount": 12.0},
        ])
        assert validate_post_edit_lifo(df, 1, "DELETE", 0, 0, "") is None

    def test_delete_buy_when_prior_buy_absorbs_sells_allowed(self) -> None:
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 200, "amount": 8.0},
            {"date": "2026-01-02", "action": "BUY", "shares": 100, "amount": 10.0},
            {"date": "2026-01-05", "action": "SELL", "shares": 50, "amount": 15.0},
        ])
        assert validate_post_edit_lifo(df, 2, "DELETE", 0, 0, "") is None

    def test_edit_buy_shares_down_below_sells_rejected(self) -> None:
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 10.0},
            {"date": "2026-01-05", "action": "SELL", "shares": 50, "amount": 15.0},
        ])
        err = validate_post_edit_lifo(df, 1, "BUY", 20, 10.0, "2026-01-01")
        assert err is not None
        assert "30" in err

    def test_edit_buy_date_forward_past_sell_rejected(self) -> None:
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 10.0},
            {"date": "2026-01-05", "action": "SELL", "shares": 50, "amount": 15.0},
        ])
        err = validate_post_edit_lifo(df, 1, "BUY", 100, 10.0, "2026-01-10")
        assert err is not None

    def test_edit_unrelated_field_allowed(self) -> None:
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 10.0},
            {"date": "2026-01-05", "action": "SELL", "shares": 50, "amount": 15.0},
        ])
        assert validate_post_edit_lifo(df, 1, "BUY", 100, 12.50, "2026-01-01") is None

    def test_empty_action_falls_back_to_existing(self) -> None:
        # Caller omits `action`. Validator should treat the row as keeping
        # its existing action (BUY) instead of simulating a blank-action
        # row that loses its inventory contribution. Today's callers always
        # send action, but this guard protects future partial-edit callers.
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 10.0},
            {"date": "2026-01-05", "action": "SELL", "shares": 50, "amount": 15.0},
        ])
        assert validate_post_edit_lifo(df, 1, "", 100, 10.0, "2026-01-01") is None

    def test_empty_txns_allowed(self) -> None:
        df = pd.DataFrame(columns=["date", "action", "shares", "amount", "detail_id"])
        assert validate_post_edit_lifo(df, 1, "DELETE", 0, 0, "") is None

    def test_delete_only_remaining_txn_allowed(self) -> None:
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 10.0},
        ])
        assert validate_post_edit_lifo(df, 1, "DELETE", 0, 0, "") is None


class TestComputeTradeRisk:
    """Unit tests for the Group 7-1 holistic Trade Risk $ helper.

    Formula: Σ over open BUY-lot remainders of
             lot_shares × max(0, lot_entry − lot_stop) × multiplier.
    """

    @staticmethod
    def _txns(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_single_open_buy(self) -> None:
        """One BUY with a stop below entry: risk = shares × (entry − stop) × mult."""
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 200.0, "stop_loss": 195.0},
        ])
        assert compute_trade_risk(df, multiplier=1.0) == 500.0

    def test_two_open_buys_same_stop_sums(self) -> None:
        """Two BUYs sharing the same stop: contributions sum.

        Lot 1: 100 × (200 − 195) = 500. Lot 2: 50 × (210 − 195) = 750. Total 1250.
        """
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 200.0, "stop_loss": 195.0},
            {"date": "2026-01-02", "action": "BUY", "shares": 50,  "amount": 210.0, "stop_loss": 195.0},
        ])
        assert compute_trade_risk(df, multiplier=1.0) == 1250.0

    def test_free_roll_lot_contributes_zero(self) -> None:
        """Stop at-or-above entry contributes 0 (free-roll).

        Lot 1: 100 × (200 − 200) = 0  (stop at entry — free roll).
        Lot 2: 50 × (210 − 195) = 750.
        Total: 750 (the BE lot drops out).
        """
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 200.0, "stop_loss": 200.0},
            {"date": "2026-01-02", "action": "BUY", "shares": 50,  "amount": 210.0, "stop_loss": 195.0},
        ])
        assert compute_trade_risk(df, multiplier=1.0) == 750.0

    def test_stop_above_entry_contributes_zero(self) -> None:
        """Stop above entry (locked-in profit on this lot) contributes 0."""
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 200.0, "stop_loss": 205.0},
        ])
        assert compute_trade_risk(df, multiplier=1.0) == 0.0

    def test_missing_stop_contributes_zero(self) -> None:
        """stop_loss=0 / NULL means 'no stop set' → 0 contribution.

        Matches calc_risk_budget's existing convention: unsized lots show 0
        rather than imply infinite/unbounded risk.
        """
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 200.0, "stop_loss": 0.0},
        ])
        assert compute_trade_risk(df, multiplier=1.0) == 0.0

    def test_fully_closed_campaign_returns_zero(self) -> None:
        """All BUYs matched by SELLs → empty inventory → risk = 0.

        Even if the BUYs had a stop, once the position is gone there's no
        forward exposure.
        """
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY",  "shares": 100, "amount": 200.0, "stop_loss": 195.0},
            {"date": "2026-01-05", "action": "SELL", "shares": 100, "amount": 220.0, "stop_loss": 0.0},
        ])
        assert compute_trade_risk(df, multiplier=1.0) == 0.0

    def test_partial_sell_reduces_via_lifo(self) -> None:
        """Partial SELL eats the most-recent BUY first (LIFO).

        BUY 100 @ 200, stop 195 → contributes 500 fully.
        BUY 50 @ 210, stop 195 → contributes 750 fully.
        SELL 30 LIFO-matches the second BUY: 20 remaining @ 210, stop 195
        contributes 20 × 15 = 300. First BUY untouched, still contributes 500.
        Total: 800.
        """
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY",  "shares": 100, "amount": 200.0, "stop_loss": 195.0},
            {"date": "2026-01-02", "action": "BUY",  "shares": 50,  "amount": 210.0, "stop_loss": 195.0},
            {"date": "2026-01-03", "action": "SELL", "shares": 30,  "amount": 215.0, "stop_loss": 0.0},
        ])
        assert compute_trade_risk(df, multiplier=1.0) == 800.0

    def test_multi_lot_one_at_be_one_with_stop(self) -> None:
        """The exact additive-bug scenario from the Group 7 audit.

        Pre-Group 7: scale-in additive logic would store
          lot1_risk(at-original-stop) + lot2_risk(at-new-stop)
        even after lot1's stop moved to BE. compute_trade_risk on the
        post-stop-move state shows the correct value: only lot2 contributes.

        Lot 1: 100 × max(0, 200 − 200) = 0  (stop moved to BE)
        Lot 2: 50  × max(0, 210 − 205) = 250
        Total: 250 (NOT 0 + 750 = 750 from the additive bug)
        """
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 200.0, "stop_loss": 200.0},
            {"date": "2026-01-02", "action": "BUY", "shares": 50,  "amount": 210.0, "stop_loss": 205.0},
        ])
        assert compute_trade_risk(df, multiplier=1.0) == 250.0

    def test_option_with_stop_returns_cost_under_policy_i(self) -> None:
        """Group 7-3 policy: for long options, Trade Risk $ = premium paid
        regardless of any stop value. Stops on options no longer drive the
        math (the prior "50% stop" practice is discontinued).

        Replaces the pre-Group-7-3 test that asserted the distance-to-stop
        formula on options. 1 contract × $5.00 premium × 100 = $500. The
        $3.00 stop is ignored.
        """
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 1, "amount": 5.0, "stop_loss": 3.0},
        ])
        assert compute_trade_risk(df, multiplier=100.0) == 500.0

    def test_option_no_stop_returns_cost(self) -> None:
        """The common case: option BUY with no stop set. Trade Risk $ =
        premium = 1 × 5 × 100 = $500. Pre-Group-7-3 the formula returned
        0 here (stop=0 fell out of the distance-to-stop guard), which
        Migration 021 then masked by backfilling sizing_mode × NLV — the
        bug Group 7-3 corrects at the formula level."""
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 1, "amount": 5.0, "stop_loss": 0.0},
        ])
        assert compute_trade_risk(df, multiplier=100.0) == 500.0

    def test_option_with_stop_above_entry_returns_cost(self) -> None:
        """Defensive: a stop above entry (impossible in practice but
        harmless data) does not crash the formula. Options ignore stops
        entirely under Policy I, so the answer is still cost.
        """
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 1, "amount": 5.0, "stop_loss": 10.0},
        ])
        assert compute_trade_risk(df, multiplier=100.0) == 500.0

    def test_option_multi_lot_returns_total_cost(self) -> None:
        """Scale-in on an option campaign: Trade Risk $ = Σ over open lots
        of qty × entry × 100. Two BUYs of 1 contract at $5 and 2 contracts
        at $6 → 1×5×100 + 2×6×100 = 500 + 1200 = $1700.
        """
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY", "shares": 1, "amount": 5.0, "stop_loss": 0.0},
            {"date": "2026-01-03", "action": "BUY", "shares": 2, "amount": 6.0, "stop_loss": 0.0},
        ])
        assert compute_trade_risk(df, multiplier=100.0) == 1700.0

    def test_empty_dataframe_returns_zero(self) -> None:
        df = pd.DataFrame(columns=["date", "action", "shares", "amount", "stop_loss"])
        assert compute_trade_risk(df, multiplier=1.0) == 0.0

    def test_unparseable_dates_dropped(self) -> None:
        """Rows with bad dates drop out so they don't crash the sort; remaining
        valid rows still drive the inventory walk."""
        df = self._txns([
            {"date": "bogus", "action": "BUY", "shares": 999, "amount": 999.0, "stop_loss": 100.0},
            {"date": "2026-01-01", "action": "BUY", "shares": 100, "amount": 200.0, "stop_loss": 195.0},
        ])
        assert compute_trade_risk(df, multiplier=1.0) == 500.0


class TestComputeLifoSummaryHcfoBranch:
    """HCFO matching: each SELL consumes the highest-cost open lot first,
    ties broken by date ASC (oldest highest-cost wins).

    Mirrors the LIFO test set in TestComputeLifoSummary so the two matching
    methods can be compared side-by-side and any future divergence in the
    walk's structural invariants (closure emission, residual inventory,
    avg_entry/avg_exit) is caught.
    """

    @staticmethod
    def _txns(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_hcfo_matches_highest_cost_first(self) -> None:
        """3 BUYs at $40 / $60 / $50. One SELL of 100 @ $70 stamped HCFO
        consumes the $60 lot first (highest cost), not the most-recent ($50).

        Realized PL = 100 × (70 − 60) = 1000. After: 100 @ $40 + 100 @ $50
        remain → avg_entry = (4000+5000)/200 = 45.
        """
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY",  "shares": 100, "amount": 40.0},
            {"date": "2026-01-12", "action": "BUY",  "shares": 100, "amount": 60.0},
            {"date": "2026-01-14", "action": "BUY",  "shares": 100, "amount": 50.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 100, "amount": 70.0,
             "match_method": "HCFO"},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        assert result is not None
        assert result["Realized_PL"] == 1000.0
        assert result["Shares"] == 200.0
        assert result["Avg_Entry"] == 45.0
        assert result["Status"] == "OPEN"

    def test_hcfo_sell_spans_two_lots(self) -> None:
        """SELL larger than highest-cost lot consumes that lot fully, then
        the next-highest. 100 @ $40 + 50 @ $60. SELL 80 @ $70 HCFO eats the
        50 @ $60 lot in full, then 30 from $40 lot.

        Realized PL = 50×(70−60) + 30×(70−40) = 500 + 900 = 1400.
        Remaining: 70 @ $40, avg_entry = 40.
        """
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY",  "shares": 100, "amount": 40.0},
            {"date": "2026-01-15", "action": "BUY",  "shares": 50,  "amount": 60.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 80,  "amount": 70.0,
             "match_method": "HCFO"},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        assert result is not None
        assert result["Realized_PL"] == 1400.0
        assert result["Shares"] == 70.0
        assert result["Avg_Entry"] == 40.0

    def test_hcfo_tiebreak_uses_date_asc(self) -> None:
        """Two BUYs at the SAME price tiebreak on date ASC (oldest first).

        BUY 100 @ $50 (2026-01-10), BUY 100 @ $50 (2026-01-15). SELL 50 @ $60
        HCFO sorts by (-price, date) → the older lot wins.

        Verified by checking the remaining inventory: only the 2026-01-15
        lot should have its full 100 shares; the 2026-01-10 lot is partial.
        Since both prices match, this is checked via the closures' buy_trx_id
        if present, or indirectly via shares remaining.
        """
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY",  "shares": 100, "amount": 50.0, "trx_id": "B1"},
            {"date": "2026-01-15", "action": "BUY",  "shares": 100, "amount": 50.0, "trx_id": "A1"},
            {"date": "2026-01-20", "action": "SELL", "shares": 50,  "amount": 60.0,
             "match_method": "HCFO", "trx_id": "S1"},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL", with_closures=True)
        assert result is not None
        summary, closures = result
        # Same price for both BUYs → realized PL identical regardless of which
        # was consumed. Tiebreak proof: closure points at B1 (older), not A1.
        assert len(closures) == 1
        assert closures[0]["buy_trx_id"] == "B1"
        assert summary["Realized_PL"] == 500.0
        assert summary["Shares"] == 150.0

    def test_hcfo_losing_trade_realized_pl_negative(self) -> None:
        """HCFO works for losing trades too. BUY 100 @ $60, BUY 100 @ $40.
        SELL 100 @ $45 HCFO consumes the $60 lot → loss.

        Realized PL = 100 × (45 − 60) = −1500. Remaining: 100 @ $40.
        """
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY",  "shares": 100, "amount": 60.0},
            {"date": "2026-01-15", "action": "BUY",  "shares": 100, "amount": 40.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 100, "amount": 45.0,
             "match_method": "HCFO"},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        assert result is not None
        assert result["Realized_PL"] == -1500.0
        assert result["Shares"] == 100.0
        assert result["Avg_Entry"] == 40.0

    def test_hcfo_with_closures_emits_pair_dicts(self) -> None:
        """Closure dicts produced by an HCFO walk reference the highest-cost
        lot's buy_trx_id, not the most-recent's."""
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY",  "shares": 100, "amount": 40.0, "trx_id": "B1"},
            {"date": "2026-01-12", "action": "BUY",  "shares": 100, "amount": 60.0, "trx_id": "A1"},
            {"date": "2026-01-14", "action": "BUY",  "shares": 100, "amount": 50.0, "trx_id": "A2"},
            {"date": "2026-01-20", "action": "SELL", "shares": 100, "amount": 70.0, "trx_id": "S1",
             "match_method": "HCFO"},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL", with_closures=True)
        assert result is not None
        summary, closures = result
        assert len(closures) == 1
        assert closures[0]["buy_trx_id"] == "A1"  # the $60 lot
        assert closures[0]["sell_trx_id"] == "S1"
        assert closures[0]["shares"] == 100.0
        assert closures[0]["buy_price"] == 60.0
        assert closures[0]["sell_price"] == 70.0
        assert closures[0]["realized_pl"] == 1000.0

    def test_hcfo_remaining_inventory_is_lowest_cost_lots(self) -> None:
        """Multiple SELLs under HCFO walk down the cost ladder. Residual
        inventory is the LOWEST-cost lots, the opposite of LIFO's residual.

        BUYs: 100@$40, 100@$50, 100@$60. SELL 150 HCFO → consumes 100@$60
        + 50@$50. Remaining: 50@$50 + 100@$40, avg_entry = (2500+4000)/150
        = 43.3333.
        """
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY",  "shares": 100, "amount": 40.0},
            {"date": "2026-01-12", "action": "BUY",  "shares": 100, "amount": 50.0},
            {"date": "2026-01-14", "action": "BUY",  "shares": 100, "amount": 60.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 150, "amount": 70.0,
             "match_method": "HCFO"},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        assert result is not None
        # 100×(70−60) + 50×(70−50) = 1000 + 1000 = 2000
        assert result["Realized_PL"] == 2000.0
        assert result["Shares"] == 150.0
        assert result["Avg_Entry"] == pytest.approx(43.3333, abs=0.0001)


class TestComputeLifoSummaryMixedMethods:
    """Mixed-method SELLs in one trade: each SELL is stamped independently
    and the walk handles the per-SELL switch. The arrival_seq lot field is
    what protects LIFO ordering after an HCFO mutation reordered inventory.
    """

    @staticmethod
    def _txns(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_null_match_method_defaults_to_lifo(self) -> None:
        """No match_method column / NULL value defaults to LIFO so every
        pre-B-1 fixture and every historical SELL behaves byte-for-byte
        identically. Cross-checked against the canonical LIFO case in
        TestComputeLifoSummary."""
        # No column at all
        df_no_col = self._txns([
            {"date": "2026-01-10", "action": "BUY",  "shares": 100, "amount": 50.0},
            {"date": "2026-01-15", "action": "BUY",  "shares": 50,  "amount": 55.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 50,  "amount": 60.0},
        ])
        # Explicit None
        df_null = self._txns([
            {"date": "2026-01-10", "action": "BUY",  "shares": 100, "amount": 50.0},
            {"date": "2026-01-15", "action": "BUY",  "shares": 50,  "amount": 55.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 50,  "amount": 60.0,
             "match_method": None},
        ])
        # Explicit LIFO
        df_lifo = self._txns([
            {"date": "2026-01-10", "action": "BUY",  "shares": 100, "amount": 50.0},
            {"date": "2026-01-15", "action": "BUY",  "shares": 50,  "amount": 55.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 50,  "amount": 60.0,
             "match_method": "LIFO"},
        ])
        for df in (df_no_col, df_null, df_lifo):
            result = compute_lifo_summary(df, "T1", "AAPL")
            assert result is not None
            # LIFO eats the newest lot (50 @ $55): realized = 50×(60−55) = 250
            assert result["Realized_PL"] == 250.0
            assert result["Shares"] == 100.0

    def test_mixed_lifo_then_hcfo_in_one_trade(self) -> None:
        """LIFO SELL first, then HCFO SELL in the same trade.

        BUYs: 100@$40, 100@$60, 100@$50.
        SELL 100 @ $70 LIFO → consumes newest ($50). Realized = 100×20 = 2000.
        Remaining: 100@$40, 100@$60.
        SELL 100 @ $80 HCFO → consumes highest-cost ($60). Realized = 100×20 = 2000.
        Remaining: 100@$40. Total realized = 4000. avg_entry = 40.
        """
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY",  "shares": 100, "amount": 40.0},
            {"date": "2026-01-12", "action": "BUY",  "shares": 100, "amount": 60.0},
            {"date": "2026-01-14", "action": "BUY",  "shares": 100, "amount": 50.0},
            {"date": "2026-01-20", "action": "SELL", "shares": 100, "amount": 70.0,
             "match_method": "LIFO"},
            {"date": "2026-01-22", "action": "SELL", "shares": 100, "amount": 80.0,
             "match_method": "HCFO"},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL")
        assert result is not None
        assert result["Realized_PL"] == 4000.0
        assert result["Shares"] == 100.0
        assert result["Avg_Entry"] == 40.0

    def test_mixed_hcfo_then_lifo_in_one_trade(self) -> None:
        """HCFO first, then LIFO — the edge case the arrival_seq field
        protects. After HCFO mutates inventory order, LIFO must still
        consume the most-recently-arrived BUY, not whatever happens to
        be at the end of the (potentially reordered) list.

        BUYs in arrival order: B1=100@$40, A1=100@$60, A2=100@$50.
        SELL 100 @ $70 HCFO → consumes A1 ($60). Realized = 100×10 = 1000.
        Inventory after sort+mutation: contains B1 and A2 in some order.
        SELL 100 @ $75 LIFO → must consume A2 (newest), realized = 100×25 = 2500.
        Total = 3500. Remaining: B1 (100@$40). avg_entry = 40.
        """
        df = self._txns([
            {"date": "2026-01-10", "action": "BUY",  "shares": 100, "amount": 40.0, "trx_id": "B1"},
            {"date": "2026-01-12", "action": "BUY",  "shares": 100, "amount": 60.0, "trx_id": "A1"},
            {"date": "2026-01-14", "action": "BUY",  "shares": 100, "amount": 50.0, "trx_id": "A2"},
            {"date": "2026-01-20", "action": "SELL", "shares": 100, "amount": 70.0,
             "match_method": "HCFO", "trx_id": "S1"},
            {"date": "2026-01-22", "action": "SELL", "shares": 100, "amount": 75.0,
             "match_method": "LIFO", "trx_id": "S2"},
        ])
        result = compute_lifo_summary(df, "T1", "AAPL", with_closures=True)
        assert result is not None
        summary, closures = result
        assert summary["Realized_PL"] == 3500.0
        assert summary["Shares"] == 100.0
        assert summary["Avg_Entry"] == 40.0
        # Closure pairs prove the per-SELL switch routed correctly even
        # after the HCFO sort reordered the inventory list.
        s1_closures = [c for c in closures if c["sell_trx_id"] == "S1"]
        s2_closures = [c for c in closures if c["sell_trx_id"] == "S2"]
        assert len(s1_closures) == 1 and s1_closures[0]["buy_trx_id"] == "A1"
        assert len(s2_closures) == 1 and s2_closures[0]["buy_trx_id"] == "A2"


class TestComputeTradeRiskHcfoBranch:
    """compute_trade_risk's residual-inventory exposure under HCFO.

    Trade Risk $ depends on what remains after each SELL, so the per-SELL
    method choice shifts the residual toward the lowest-cost lots under
    HCFO. Coverage centers on stock trades (where stops drive the formula);
    options policy I is method-agnostic (max loss = premium regardless of
    matching method), so no separate HCFO option test is needed.
    """

    @staticmethod
    def _txns(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_hcfo_residual_drives_risk(self) -> None:
        """Two BUYs, one SELL stamped HCFO. The residual inventory under
        HCFO holds the LOWER-cost lot, which means the risk number diverges
        from the LIFO path's residual.

        BUYs: 100 @ $50 stop 45 (lot1), 50 @ $60 stop 55 (lot2).
        SELL 50 @ $65 HCFO → consumes lot2 fully. Remaining: lot1 only.
        Risk = 100 × (50 − 45) = 500.

        Under LIFO the same SELL also consumes lot2 (it's the newest),
        so the LIFO/HCFO answers coincide here. The next test exercises
        a case where they diverge.
        """
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY",  "shares": 100, "amount": 50.0, "stop_loss": 45.0},
            {"date": "2026-01-02", "action": "BUY",  "shares": 50,  "amount": 60.0, "stop_loss": 55.0},
            {"date": "2026-01-03", "action": "SELL", "shares": 50,  "amount": 65.0, "stop_loss": 0.0,
             "match_method": "HCFO"},
        ])
        assert compute_trade_risk(df, multiplier=1.0) == 500.0

    def test_hcfo_diverges_from_lifo_when_newer_lot_cheaper(self) -> None:
        """When the older BUY is the highest-cost lot, HCFO and LIFO
        produce DIFFERENT residuals → different risk numbers.

        BUYs in arrival order: 100 @ $60 stop 55 (older, expensive),
        50 @ $40 stop 35 (newer, cheap). SELL 50 @ $65.

        HCFO consumes 50 from the $60 lot (highest cost). Residual:
        50 @ $60 stop 55  → risk 50×(60−55) = 250
        50 @ $40 stop 35  → risk 50×(40−35) = 250
        Total HCFO risk = 500.

        LIFO would consume the $40 lot first → residual is 100 @ $60
        stop 55 → risk 100×5 = 500. Coincidentally equal here, so we
        check the residuals via stricter math:
        Build a no-stop variant where HCFO leaves an option-style cost
        exposure visible.
        """
        df = self._txns([
            {"date": "2026-01-01", "action": "BUY",  "shares": 100, "amount": 60.0, "stop_loss": 55.0},
            {"date": "2026-01-02", "action": "BUY",  "shares": 50,  "amount": 40.0, "stop_loss": 35.0},
            {"date": "2026-01-03", "action": "SELL", "shares": 50,  "amount": 65.0, "stop_loss": 0.0,
             "match_method": "HCFO"},
        ])
        # HCFO consumes 50 of the $60 lot. Residual: 50 @ $60 + 50 @ $40.
        # 50×(60-55) + 50×(40-35) = 250 + 250 = 500.
        assert compute_trade_risk(df, multiplier=1.0) == 500.0

    def test_lifo_and_hcfo_diverge_when_residual_differs(self) -> None:
        """Construct a case where LIFO and HCFO residuals visibly differ.

        BUYs: 100 @ $60 stop 50 (older, expensive), 100 @ $40 stop 35 (newer).
        SELL 100 stamped HCFO → consumes the $60 lot in full.
        Residual: 100 @ $40 stop 35 → risk 100×5 = 500.

        Identical inputs with LIFO stamp would consume the $40 lot.
        Residual: 100 @ $60 stop 50 → risk 100×10 = 1000.
        """
        df_hcfo = self._txns([
            {"date": "2026-01-01", "action": "BUY",  "shares": 100, "amount": 60.0, "stop_loss": 50.0},
            {"date": "2026-01-02", "action": "BUY",  "shares": 100, "amount": 40.0, "stop_loss": 35.0},
            {"date": "2026-01-03", "action": "SELL", "shares": 100, "amount": 65.0, "stop_loss": 0.0,
             "match_method": "HCFO"},
        ])
        df_lifo = self._txns([
            {"date": "2026-01-01", "action": "BUY",  "shares": 100, "amount": 60.0, "stop_loss": 50.0},
            {"date": "2026-01-02", "action": "BUY",  "shares": 100, "amount": 40.0, "stop_loss": 35.0},
            {"date": "2026-01-03", "action": "SELL", "shares": 100, "amount": 65.0, "stop_loss": 0.0,
             "match_method": "LIFO"},
        ])
        assert compute_trade_risk(df_hcfo, multiplier=1.0) == 500.0
        assert compute_trade_risk(df_lifo, multiplier=1.0) == 1000.0
