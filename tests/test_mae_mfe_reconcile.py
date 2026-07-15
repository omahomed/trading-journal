"""Tests for api/mae_mfe_reconcile.py — pure math only.

The DB write path (_update_row) and yfinance download (_download_history)
are I/O; they're kept thin so the two pure math functions can be tested
in isolation without a live DB or the network. Coverage:

  * compute_excursions_from_frame
      - straight-up MAE / MFE / days-to on a walk-forward fixture
      - same-day entry (single-bar frame) → mae/mfe/days = 0
      - max_retrace_pct distinct from mae_pct on peak-then-pullback
      - up-only run → mae_pct = 0, max_retrace_pct = 0
      - down-only run → mfe_pct = 0
      - empty frame / non-positive entry price → None

  * compute_atr21_from_frame
      - 21-bar canonical case matches SMA(TR,21)/SMA(Low,21)×100
      - <21 bars → None
      - flat series (sma_low = 0 edge case) → None
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.mae_mfe_reconcile import (  # noqa: E402
    compute_atr21_from_frame,
    compute_excursions_from_frame,
)


# ─────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────


def _bars(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    """rows = [(low, high, close), ...] in date order."""
    return pd.DataFrame({
        "Low":   [r[0] for r in rows],
        "High":  [r[1] for r in rows],
        "Close": [r[2] for r in rows],
    })


# ═══════════════════════════════════════════════════════════════════════
# compute_excursions_from_frame
# ═══════════════════════════════════════════════════════════════════════


class TestExcursions:
    def test_walk_forward_canonical(self):
        # Entry $100. Walk: bar 0 low 99 high 102 (small dip + rally)
        # bar 1 low 97 high 103; bar 2 low 96 high 108 (new MAE at bar
        # 2, new MFE at bar 2); bar 3 low 102 high 110 (new MFE);
        # bar 4 low 104 high 109. MAE = 96, MFE = 110.
        df = _bars([
            (99, 102, 101),
            (97, 103, 100),
            (96, 108, 107),
            (102, 110, 108),
            (104, 109, 107),
        ])
        r = compute_excursions_from_frame(df, entry_price=100.0)
        assert r is not None
        # MAE % = (96 - 100) / 100 = -4%
        assert r["mae_pct"] == pytest.approx(-4.0)
        # MFE % = (110 - 100) / 100 = 10%
        assert r["mfe_pct"] == pytest.approx(10.0)
        # 0-based day indexing: MAE on bar 2 → 2; MFE on bar 3 → 3
        assert r["days_to_mae"] == 2
        assert r["days_to_mfe"] == 3

    def test_same_day_entry_no_sell_yields_zero_excursions(self):
        # Same-day entry with NO SELL activity: bar 0's OHLC extremes
        # are deliberately excluded (reversal-candle rule). Without
        # any post-entry price data, MAE and MFE are both 0. This
        # replaces the previous test that used bar 0 low/high directly.
        df = _bars([(98.5, 101.5, 100)])
        r = compute_excursions_from_frame(df, entry_price=100.0)
        assert r is not None
        assert r["mae_pct"] == pytest.approx(0.0)
        assert r["mfe_pct"] == pytest.approx(0.0)
        assert r["days_to_mae"] == 0
        assert r["days_to_mfe"] == 0

    def test_reversal_candle_entry_no_phantom_mae_from_bar0_low(self):
        # The motivating case from the fix: an "Upside Reversal" candle
        # prints a low of $73 during an intraday washout, then closes
        # at $79 where the user buys. Under the OLD rule bar 0's low
        # would attribute a phantom -7.6% MAE to the trader; under
        # the NEW rule bar 0 is skipped and MAE tracks from bar 1.
        # Later bars flat → MAE = 0, not -7.6%.
        df = _bars([
            (73.0, 80.0, 79.0),   # bar 0: reversal candle
            (79.5, 82.0, 81.0),   # bar 1: continuation up
            (80.0, 83.5, 82.5),
        ])
        r = compute_excursions_from_frame(df, entry_price=79.0)
        assert r is not None
        # No phantom -7.6% from bar 0's $73 low.
        assert r["mae_pct"] == pytest.approx(0.0)
        # MFE reflects the day-1/day-2 rally.
        # High 83.5 vs entry 79 → +5.7%.
        assert r["mfe_pct"] == pytest.approx(5.6962, abs=1e-3)
        assert r["days_to_mfe"] == 2

    def test_max_retrace_distinct_from_mae_on_peak_then_pullback(self):
        # Trade opens flat, rips to 120, then bleeds to 108 before
        # closing. MAE off entry = 0 (never below 100). Max retrace
        # off the running peak (120) is (108 - 120) / 120 = -10%.
        df = _bars([
            (100, 102, 101),
            (105, 115, 112),
            (110, 120, 118),   # peak at 120
            (108, 116, 111),   # pullback low 108 vs peak 120
        ])
        r = compute_excursions_from_frame(df, entry_price=100.0)
        assert r is not None
        assert r["mae_pct"] == pytest.approx(0.0)   # low never below entry
        assert r["mfe_pct"] == pytest.approx(20.0)  # high 120 vs 100
        # (108 - 120) / 120 × 100 = -10%
        assert r["max_retrace_pct"] == pytest.approx(-10.0)

    def test_up_only_run_has_zero_mae_and_zero_retrace(self):
        # Monotonic gap-up: each bar's low > prior bar's high, so no
        # bar's low ever falls below any prior peak. MAE off entry
        # stays at 0 (entry-day low IS the entry price in this
        # stylized case); max retrace stays 0 because no lower low ever
        # follows a higher high.
        df = _bars([
            (100, 102, 101),
            (103, 105, 104),   # gap up: low 103 > prev high 102
            (106, 108, 107),   # gap up: low 106 > prev high 105
            (109, 112, 110),   # gap up: low 109 > prev high 108
        ])
        r = compute_excursions_from_frame(df, entry_price=100.0)
        assert r is not None
        assert r["mae_pct"] == pytest.approx(0.0)
        assert r["mfe_pct"] == pytest.approx(12.0)
        assert r["max_retrace_pct"] == pytest.approx(0.0)

    def test_down_only_run_has_zero_mfe(self):
        # Monotonic down: mfe off entry stays at 0. mae hits the low.
        df = _bars([
            (98, 100, 98),
            (95, 99, 96),
            (92, 96, 94),
        ])
        r = compute_excursions_from_frame(df, entry_price=100.0)
        assert r is not None
        assert r["mae_pct"] == pytest.approx(-8.0)   # low 92
        assert r["mfe_pct"] == pytest.approx(0.0)    # high never above entry

    def test_max_retrace_captures_deeper_second_pullback(self):
        # Peak 120 → pullback to 110 (-8.3%). New peak 130 →
        # pullback to 115 (-11.5%). Running max_retrace should be
        # the deeper of the two, from the LATER peak.
        df = _bars([
            (100, 120, 118),
            (110, 118, 115),   # pullback #1: -8.3% off 120
            (114, 130, 128),   # new peak 130
            (115, 122, 118),   # pullback #2: -11.5% off 130
        ])
        r = compute_excursions_from_frame(df, entry_price=100.0)
        assert r is not None
        # (115 - 130) / 130 × 100 = -11.538%
        assert r["max_retrace_pct"] == pytest.approx(-11.5385, abs=1e-3)

    def test_empty_frame_returns_none(self):
        assert compute_excursions_from_frame(pd.DataFrame(), 100.0) is None

    def test_non_positive_entry_price_returns_none(self):
        df = _bars([(99, 101, 100)])
        assert compute_excursions_from_frame(df, 0.0) is None
        assert compute_excursions_from_frame(df, -5.0) is None

    # ────────────────────────────────────────────────────────────────
    # Same-day sell activity (post-entry-day-fix)
    # ────────────────────────────────────────────────────────────────

    def test_same_day_sell_below_entry_becomes_bar0_mae(self):
        # Same-day stop-out: bought at $100, stopped out at $96 same
        # day. Bar 0 low is $95 (below the stop), which we do NOT
        # count — the trader exited at $96, not $95. days_to_mae = 0
        # because the exit was on entry day.
        df = _bars([(95.0, 101.0, 96.5)])
        r = compute_excursions_from_frame(
            df, entry_price=100.0,
            same_day_low_exit_price=96.0,
        )
        assert r is not None
        assert r["mae_pct"] == pytest.approx(-4.0)  # (96-100)/100
        assert r["mfe_pct"] == pytest.approx(0.0)
        assert r["days_to_mae"] == 0
        assert r["days_to_mfe"] == 0

    def test_same_day_sell_above_entry_becomes_bar0_mfe(self):
        # Same-day partial scalp winner: bought at $100, sold half at
        # $103 same day. Bar 0 high is $104 which we do NOT count —
        # the trader captured $103, not $104. Symmetric to the MAE case.
        df = _bars([(99.5, 104.0, 102.5)])
        r = compute_excursions_from_frame(
            df, entry_price=100.0,
            same_day_high_exit_price=103.0,
        )
        assert r is not None
        assert r["mae_pct"] == pytest.approx(0.0)
        assert r["mfe_pct"] == pytest.approx(3.0)   # (103-100)/100
        assert r["days_to_mae"] == 0
        assert r["days_to_mfe"] == 0

    def test_later_bar_low_overrides_same_day_sell_price(self):
        # Same-day partial sell at $97 (-3%), then user keeps holding
        # and the position tanks further to $94 on day 2. Later bar's
        # low IS a valid MAE candidate — the same-day sell only seeds
        # bar 0; later bars can (and here do) print a lower low.
        df = _bars([
            (96.0, 101.0, 97.5),   # bar 0: same-day sell at 97
            (95.0, 98.0,  96.0),   # bar 1
            (93.5, 96.0,  94.5),   # bar 2: new low
        ])
        r = compute_excursions_from_frame(
            df, entry_price=100.0,
            same_day_low_exit_price=97.0,
        )
        assert r is not None
        # (93.5 - 100) / 100 = -6.5%. NOT -3% (same-day sell) — bar 2
        # went lower and wins.
        assert r["mae_pct"] == pytest.approx(-6.5)
        assert r["days_to_mae"] == 2

    def test_same_day_sell_ignored_when_price_is_zero_or_none(self):
        # None and 0 both mean "no same-day activity". Bar 0 is skipped
        # entirely; same-day sell inputs are treated as absent even
        # when the caller passes them defensively.
        df = _bars([
            (95.0, 105.0, 96.0),  # bar 0 has wide range, both skipped
            (94.0, 104.0, 100.0), # bar 1: real post-entry data
        ])
        r_none = compute_excursions_from_frame(
            df, entry_price=100.0,
            same_day_low_exit_price=None,
            same_day_high_exit_price=None,
        )
        r_zero = compute_excursions_from_frame(
            df, entry_price=100.0,
            same_day_low_exit_price=0.0,
            same_day_high_exit_price=0.0,
        )
        # Both cases: bar 0 skipped. Bar 1 sets MAE and MFE.
        # (94-100)/100 = -6%; (104-100)/100 = +4%.
        for r in (r_none, r_zero):
            assert r is not None
            assert r["mae_pct"] == pytest.approx(-6.0)
            assert r["mfe_pct"] == pytest.approx(4.0)
            assert r["days_to_mae"] == 1
            assert r["days_to_mfe"] == 1


class TestExitDaySkip:
    """Symmetric rule to entry-day: when a trade is CLOSED, the exit
    bar's OHLC is ignored and the exit-day sell prices are the only
    known-touched levels for that bar. Motivating case: stopped-out
    trades where price kept falling after the exit — the trader is out
    and that further move is not part of their position's excursion."""

    def test_gev_style_stopped_out_before_day_low(self):
        # Entry $1161 on bar 0. Bar 1 (exit day) prints a low of $1027
        # but the trader stopped out at $1071 — the true intraday low
        # happened after they were out. MAE should be (1071-1161)/1161
        # = -7.75%, NOT the full -11.55% the day's low would show.
        df = _bars([
            (1150, 1170, 1161),   # bar 0: entry-day
            (1027, 1085, 1050),   # bar 1: exit-day low irrelevant post-exit
        ])
        r = compute_excursions_from_frame(
            df, entry_price=1161.0,
            exit_low_price=1071.0,
            exit_high_price=1071.0,
        )
        assert r is not None
        # (1071 - 1161) / 1161 = -7.7519%
        assert r["mae_pct"] == pytest.approx(-7.7519, abs=1e-3)
        assert r["days_to_mae"] == 1
        # MFE stays at 0 — exit price 1071 < entry 1161, no MFE.
        assert r["mfe_pct"] == pytest.approx(0.0)

    def test_exit_day_high_seeds_mfe_but_low_after_ignored(self):
        # Scale-out at $110 on bar 1 (a good print), bar 1 low went to
        # $95 AFTER the trader was out. MFE = +10%; MAE unchanged by
        # bar 1's low (still whatever bar 0 seeded).
        df = _bars([
            (99, 101, 100),        # bar 0: entry
            (95, 112, 108),        # bar 1: exit-day — low IGNORED
        ])
        r = compute_excursions_from_frame(
            df, entry_price=100.0,
            exit_low_price=110.0,
            exit_high_price=110.0,
        )
        assert r is not None
        assert r["mfe_pct"] == pytest.approx(10.0)
        assert r["days_to_mfe"] == 1
        # MAE stays at 0 — bar 0 skipped, bar 1's $95 low not counted.
        assert r["mae_pct"] == pytest.approx(0.0)

    def test_middle_bar_low_still_counted(self):
        # 3-bar trade: entry bar 0, middle bar 1 (walked normally),
        # exit bar 2 (skipped). Middle-bar low is a real experienced
        # excursion; only the exit bar's OHLC is skipped.
        df = _bars([
            (99, 101, 100),        # bar 0: entry
            (85, 102, 88),         # bar 1: MIDDLE — low counts
            (70, 90, 88),          # bar 2: exit — OHLC IGNORED
        ])
        r = compute_excursions_from_frame(
            df, entry_price=100.0,
            exit_low_price=88.0,
            exit_high_price=88.0,
        )
        assert r is not None
        # MAE from bar 1's $85 low → -15%.
        # Exit at $88 (-12%) is BETTER, so mae_low stays at $85.
        assert r["mae_pct"] == pytest.approx(-15.0)
        assert r["days_to_mae"] == 1

    def test_min_and_max_of_multiple_exit_day_sells(self):
        # Scale-out: sells at $105 and $95 on exit day. Higher seeds MFE,
        # lower seeds MAE. Bar 2's OHLC ignored either way.
        df = _bars([
            (99, 101, 100),
            (98, 103, 102),        # middle bar — small chop
            (80, 108, 100),        # exit bar — OHLC skipped
        ])
        r = compute_excursions_from_frame(
            df, entry_price=100.0,
            exit_low_price=95.0,
            exit_high_price=105.0,
        )
        assert r is not None
        # MAE = -5% from the $95 sell (bar 1's -2% doesn't beat it).
        assert r["mae_pct"] == pytest.approx(-5.0)
        assert r["days_to_mae"] == 2
        # MFE = +5% from the $105 sell (bar 1's +3% doesn't beat it).
        assert r["mfe_pct"] == pytest.approx(5.0)
        assert r["days_to_mfe"] == 2

    def test_open_trade_no_exit_skip(self):
        # No exit_*_price provided → last bar walks normally with full
        # OHLC. Ensures OPEN trades keep their current behavior.
        df = _bars([
            (99, 101, 100),
            (85, 100, 90),
        ])
        r = compute_excursions_from_frame(df, entry_price=100.0)
        assert r is not None
        assert r["mae_pct"] == pytest.approx(-15.0)   # bar 1's low counts

    def test_same_day_round_trip_falls_through_to_entry_day_rule(self):
        # n == 1: bar 0 IS bar N-1. Exit-day args are IGNORED (bar 0
        # would be already handled via same_day_*_exit_price). This
        # test verifies exit_*_price on n==1 doesn't double-count.
        df = _bars([(90, 105, 95)])   # single bar
        # Same-day sell at $92 (below entry) — the entry-day rule
        # should catch it via same_day_low_exit_price. Passing
        # exit_low_price=92 as well should NOT change the answer.
        r_entry_only = compute_excursions_from_frame(
            df, entry_price=100.0,
            same_day_low_exit_price=92.0,
        )
        r_with_exit = compute_excursions_from_frame(
            df, entry_price=100.0,
            same_day_low_exit_price=92.0,
            exit_low_price=92.0,
            exit_high_price=92.0,
        )
        assert r_entry_only is not None
        assert r_with_exit is not None
        # Same MAE either way — no double-count.
        assert r_entry_only["mae_pct"] == pytest.approx(-8.0)
        assert r_with_exit["mae_pct"] == pytest.approx(-8.0)

    def test_retrace_updated_when_exit_undercuts_running_peak(self):
        # Trade rips to a $120 running peak on bar 1, then trader
        # exits at $110 on bar 2. Retrace = (110-120)/120 = -8.33%.
        # (Bar 2 low of $70 is skipped.)
        df = _bars([
            (100, 102, 101),
            (105, 120, 118),
            (70, 115, 110),        # exit bar — OHLC skipped
        ])
        r = compute_excursions_from_frame(
            df, entry_price=100.0,
            exit_low_price=110.0,
            exit_high_price=110.0,
        )
        assert r is not None
        # Peak = 120 (bar 1's high). Exit @ 110 → retrace -8.33%.
        assert r["max_retrace_pct"] == pytest.approx(-8.3333, abs=1e-3)


# ═══════════════════════════════════════════════════════════════════════
# compute_atr21_from_frame
# ═══════════════════════════════════════════════════════════════════════


class TestAtr21Snapshot:
    def _synth_bars(self, n: int, base: float = 100.0,
                    daily_range: float = 2.0) -> pd.DataFrame:
        """n bars with constant intraday range = daily_range and slight
        drift so the pandas shift() doesn't collapse to zero on TR."""
        rows = []
        close_prev = base
        for i in range(n):
            close = base + i * 0.5
            low = close - daily_range / 2
            high = close + daily_range / 2
            rows.append((low, high, close))
            close_prev = close
        return _bars(rows)

    def test_21_bar_case_matches_reference_formula(self):
        df = self._synth_bars(21)
        r = compute_atr21_from_frame(df)
        assert r is not None
        # Reference computation on the same DataFrame — same formula
        # api/main.py:_compute_ticker_atr_pct uses.
        high = df["High"]; low = df["Low"]; close = df["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        expected = round((tr.tail(21).mean() / low.tail(21).mean()) * 100, 4)
        assert r == pytest.approx(expected, abs=1e-4)

    def test_below_21_bars_returns_none(self):
        assert compute_atr21_from_frame(self._synth_bars(10)) is None
        assert compute_atr21_from_frame(self._synth_bars(20)) is None

    def test_flat_low_series_returns_none(self):
        # SMA(Low) = 0 → we can't divide. compute_atr21_from_frame guards
        # this so a bad symbol / edge case returns None instead of inf.
        df = pd.DataFrame({
            "Low":   [0.0] * 21,
            "High":  [1.0] * 21,
            "Close": [0.5] * 21,
        })
        assert compute_atr21_from_frame(df) is None

    def test_empty_frame_returns_none(self):
        assert compute_atr21_from_frame(pd.DataFrame()) is None
