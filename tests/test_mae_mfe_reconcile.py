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

    def test_same_day_entry_yields_single_bar_excursions(self):
        # Same-day entry: only bar 0. mae from that day's low, mfe from
        # its high. days_to_mae = days_to_mfe = 0 (0-based spec).
        df = _bars([(98.5, 101.5, 100)])
        r = compute_excursions_from_frame(df, entry_price=100.0)
        assert r is not None
        assert r["mae_pct"] == pytest.approx(-1.5)
        assert r["mfe_pct"] == pytest.approx(1.5)
        assert r["days_to_mae"] == 0
        assert r["days_to_mfe"] == 0

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
