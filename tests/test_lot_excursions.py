"""Tests for api/lot_excursions.py.

The excursion math itself is covered by tests/test_mae_mfe_reconcile.py
(compute_excursions_from_frame is imported and re-used). This file
covers the per-lot grain — the piece that's new in api/lot_excursions:

  * _compute_one_lot slices df_all to THIS lot's window and anchors on
    THIS lot's fill_price (not B1's) — so A1's MAE should reflect
    A1's own drawdown from its own fill, independent of B1.
  * ATR21 at fill computed on the 21 bars ending the day BEFORE the
    lot's fill_date — so B1 / A1 / A2 each get their own atr21_at_fill
    even though the bar series is one shared fetch.
  * Grouping: compute_lot_excursions_for_campaign issues ONE download
    per campaign, then slices per lot (verified via a call counter on
    a monkeypatched _download_history).
  * Error rows: empty download → all lots marked "no_bars"; missing
    fill_price → "bad_lot".
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api import lot_excursions  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Bar-frame builder — DatetimeIndex mirrors what yfinance returns.
# ─────────────────────────────────────────────────────────────────────


def _bars(dated_rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    """rows = [(iso_date, low, high, close), ...]."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in dated_rows])
    return pd.DataFrame({
        "Low":   [r[1] for r in dated_rows],
        "High":  [r[2] for r in dated_rows],
        "Close": [r[3] for r in dated_rows],
    }, index=idx)


def _lot(**kwargs) -> dict:
    """Sensible defaults so tests only override what they care about."""
    base = {
        "trade_id":                 "202601-001",
        "portfolio_name":           "TestPort",
        "portfolio_id":             1,
        "ticker":                   "DELL",
        "status":                   "OPEN",
        "closed_date":              None,
        # Per-LOT P&L (from lot_closures SUM by buy_trx_id) vs the
        # campaign-wide total (trades_summary.realized_pl). Distinct
        # signals; both flow through _base_row.
        "realized_pl":              None,
        "campaign_realized_pl":     None,
        "shares_closed":            None,
        "add_exempt_reason":        None,
        "trx_id":                   "B1",
        "fill_date":                date(2026, 1, 10),
        "fill_price":               100.0,
        "shares":                   50,
        "same_day_low_exit_price":  None,
        "same_day_high_exit_price": None,
        "close_day_low_exit_price": None,
        "close_day_high_exit_price": None,
    }
    base.update(kwargs)
    return base


# ═══════════════════════════════════════════════════════════════════════
# _compute_one_lot — window slicing + ATR anchoring
# ═══════════════════════════════════════════════════════════════════════


class TestComputeOneLot:
    def test_a1_anchors_on_its_own_fill_price_not_b1(self):
        """B1 filled $100 on 01-10; A1 filled $105 on 01-15. Bars show
        a 96 low on 01-11 (before A1) and a 102 low on 01-18 (after
        A1). B1's MAE would count the 96 (−4%). A1's MAE should IGNORE
        the 96 (before its fill) and count 102 (−2.86% from $105).
        """
        df_all = _bars([
            # Pre-B1 padding for ATR context (21 bars).
            *[(f"2025-12-{d:02d}", 98.0 + d*0.1, 99.5 + d*0.1, 99.0 + d*0.1) for d in range(10, 31)],
            ("2026-01-10", 99.0, 101.0, 100.0),  # B1 day
            ("2026-01-11",  96.0,  99.0,  97.0),  # deep low BEFORE A1 fills
            ("2026-01-12",  97.0, 103.0, 102.0),
            ("2026-01-13", 101.0, 106.0, 105.0),
            ("2026-01-14", 104.0, 108.0, 107.0),
            ("2026-01-15", 105.0, 109.0, 108.0),  # A1 fill day
            ("2026-01-16", 106.0, 110.0, 109.0),
            ("2026-01-17", 107.0, 111.0, 110.0),
            ("2026-01-18", 102.0, 111.0, 108.0),  # A1's deepest low from its fill
            ("2026-01-19", 108.0, 114.0, 113.0),
            ("2026-01-20", 110.0, 116.0, 115.0),
        ])

        a1 = _lot(trx_id="A1", fill_date=date(2026, 1, 15), fill_price=105.0, shares=25)
        window_end = date(2026, 1, 20)

        row = lot_excursions._compute_one_lot(
            a1, df_all,
            closed_date=None,
            window_end_date=window_end,
            status="OPEN",
        )

        # A1's MAE = (102 − 105) / 105 × 100 ≈ −2.857
        assert row["error"] is None
        assert row["mae_pct"] == -2.857 or abs(row["mae_pct"] - (-2.857)) < 0.01
        # Days from fill (2026-01-15) to MAE low (2026-01-18) = 3 bars,
        # 0-indexed so index 3 (bar 0 is fill day, skipped).
        assert row["days_to_mae"] == 3
        # MFE = (116 − 105) / 105 × 100 ≈ 10.476
        assert abs(row["mfe_pct"] - 10.476) < 0.01

    def test_b1_uses_full_window_and_deeper_mae(self):
        """B1 on same fixture should see the 96 low (bar 1 after fill) —
        its MAE is deeper than A1's because it anchors on $100 and
        includes the pre-A1 drawdown.
        """
        df_all = _bars([
            *[(f"2025-12-{d:02d}", 98.0 + d*0.1, 99.5 + d*0.1, 99.0 + d*0.1) for d in range(10, 31)],
            ("2026-01-10", 99.0, 101.0, 100.0),
            ("2026-01-11",  96.0,  99.0,  97.0),
            ("2026-01-12",  97.0, 103.0, 102.0),
            ("2026-01-13", 101.0, 106.0, 105.0),
            ("2026-01-14", 104.0, 108.0, 107.0),
            ("2026-01-15", 105.0, 109.0, 108.0),
            ("2026-01-16", 106.0, 110.0, 109.0),
            ("2026-01-17", 107.0, 111.0, 110.0),
            ("2026-01-18", 102.0, 111.0, 108.0),
            ("2026-01-19", 108.0, 114.0, 113.0),
            ("2026-01-20", 110.0, 116.0, 115.0),
        ])
        b1 = _lot()  # defaults = B1 at $100 on 2026-01-10
        row = lot_excursions._compute_one_lot(
            b1, df_all,
            closed_date=None,
            window_end_date=date(2026, 1, 20),
            status="OPEN",
        )
        # B1 MAE = (96 − 100) / 100 = −4.0
        assert row["error"] is None
        assert abs(row["mae_pct"] - (-4.0)) < 0.01
        assert row["days_to_mae"] == 1
        # Days_held = calendar days from fill to window_end.
        assert row["days_held"] == 10

    def test_atr21_anchor_at_fill_uses_only_pre_fill_bars(self):
        """ATR21 must come from ~21 bars ENDING the day BEFORE the fill.
        Feed a low-vol pre-B1 stretch + a hot mid-window stretch — the
        ATR21 should reflect the pre-fill regime, not the post-fill one.
        """
        # Calm pre-B1: 1-point range per bar.
        calm = [(f"2025-12-{d:02d}", 99.0, 100.0, 99.5) for d in range(1, 32) if d <= 31]
        # Hot post-B1: 10-point ranges — should NOT bleed into ATR anchor.
        hot = [
            ("2026-01-10",  99.0, 101.0, 100.0),  # fill day
            ("2026-01-11",  90.0, 110.0, 100.0),
            ("2026-01-12",  90.0, 110.0, 100.0),
        ]
        df_all = _bars(calm + hot)

        row = lot_excursions._compute_one_lot(
            _lot(), df_all,
            closed_date=None,
            window_end_date=date(2026, 1, 12),
            status="OPEN",
        )
        # ATR from calm bars: ~1.0/99.5 × 100 ≈ 1.005. The exact figure
        # comes from api.mae_mfe_reconcile.compute_atr21_from_frame; we
        # just assert it's the calm regime, not the hot one (which would
        # be several percent).
        assert row["atr21_at_fill_pct"] is not None
        assert row["atr21_at_fill_pct"] < 2.0, (
            f"ATR21 anchor bled hot post-fill bars: {row['atr21_at_fill_pct']}"
        )

    def test_error_row_when_fill_price_missing(self):
        df_all = _bars([("2026-01-10", 99.0, 101.0, 100.0)])
        bad = _lot(fill_price=0)
        row = lot_excursions._compute_one_lot(
            bad, df_all,
            closed_date=None,
            window_end_date=date(2026, 1, 10),
            status="OPEN",
        )
        assert row["error"] == "bad_lot"

    def test_error_row_when_window_has_no_bars(self):
        df_all = _bars([("2025-12-01", 99.0, 101.0, 100.0)])  # all pre-fill
        row = lot_excursions._compute_one_lot(
            _lot(), df_all,
            closed_date=None,
            window_end_date=date(2026, 1, 10),
            status="OPEN",
        )
        assert row["error"] == "no_bars_in_window"


# ═══════════════════════════════════════════════════════════════════════
# compute_lot_excursions_for_campaign — one fetch, sliced per lot
# ═══════════════════════════════════════════════════════════════════════


class TestCampaignLevel:
    def test_single_download_serves_all_lots_of_a_campaign(self, monkeypatch):
        """3-lot campaign should trigger exactly ONE _download_history
        call — the widest window covering all lots' pre-B1 ATR context
        through the campaign end."""
        counter = {"calls": 0, "spans": []}

        def fake_download(ticker, start, end):
            counter["calls"] += 1
            counter["spans"].append((ticker, start, end))
            return _bars([
                *[(f"2025-12-{d:02d}", 98.0, 100.0, 99.0) for d in range(1, 32)],
                ("2026-01-10",  99.0, 101.0, 100.0),
                ("2026-01-15", 100.0, 106.0, 105.0),
                ("2026-01-20", 105.0, 116.0, 115.0),
                ("2026-01-25", 108.0, 118.0, 117.0),
            ])

        monkeypatch.setattr(lot_excursions, "_download_history", fake_download)

        lots = [
            _lot(trx_id="B1", fill_date=date(2026, 1, 10), fill_price=100.0),
            _lot(trx_id="A1", fill_date=date(2026, 1, 15), fill_price=105.0),
            _lot(trx_id="A2", fill_date=date(2026, 1, 20), fill_price=115.0),
        ]
        rows = lot_excursions.compute_lot_excursions_for_campaign(lots)

        assert counter["calls"] == 1, "campaign should fetch bars once, not per-lot"
        assert len(rows) == 3
        assert [r["trx_id"] for r in rows] == ["B1", "A1", "A2"]
        # Each lot got its own row; none errored.
        for r in rows:
            assert r["error"] is None
            assert r["fill_price"] is not None
            assert r["days_to_mfe"] is not None

    def test_empty_yfinance_response_marks_all_lots_no_bars(self, monkeypatch):
        monkeypatch.setattr(lot_excursions, "_download_history",
                            lambda t, s, e: pd.DataFrame())
        lots = [
            _lot(trx_id="B1"),
            _lot(trx_id="A1", fill_date=date(2026, 1, 15), fill_price=105.0),
        ]
        rows = lot_excursions.compute_lot_excursions_for_campaign(lots)
        assert len(rows) == 2
        assert all(r["error"] == "no_bars" for r in rows)

    def test_empty_lots_list_returns_empty(self):
        assert lot_excursions.compute_lot_excursions_for_campaign([]) == []


# ═══════════════════════════════════════════════════════════════════════
# _base_row / _iso — small serialization helpers
# ═══════════════════════════════════════════════════════════════════════


class TestBaseRow:
    def test_iso_from_various_types(self):
        assert lot_excursions._iso(date(2026, 1, 15)) == "2026-01-15"
        assert lot_excursions._iso(pd.Timestamp("2026-01-15 09:30")) == "2026-01-15"
        assert lot_excursions._iso("2026-01-15") == "2026-01-15"
        assert lot_excursions._iso(None) is None

    def test_base_row_includes_all_output_fields(self):
        """Consumers (CSV writer, JSON serializer) rely on every key
        existing even on error rows. Lock the schema."""
        row = lot_excursions._base_row(_lot(), date(2026, 1, 20))
        for key in ["trade_id", "portfolio_name", "ticker", "trx_id",
                    "fill_date", "fill_price", "shares", "shares_closed",
                    "add_exempt_reason",
                    "status", "closed_date", "window_end_date", "days_held",
                    "mae_pct", "mfe_pct", "days_to_mae", "days_to_mfe",
                    "atr21_at_fill_pct", "mae_atr_multiple",
                    "mfe_atr_multiple", "min_low", "min_low_date",
                    "max_high", "max_high_date",
                    "realized_pl", "campaign_realized_pl", "error"]:
            assert key in row, f"missing key: {key}"

    def test_add_exempt_reason_flows_through_base_row(self):
        """Migration 049: §2 Window exempt-reason ('sr8_rebuild' /
        'fresh_base') is carried verbatim from the SQL lot row into
        the output base row so the CSV export can bucket by declared
        override for the 30-add review."""
        lot = _lot(trx_id="A5", add_exempt_reason="sr8_rebuild")
        row = lot_excursions._base_row(lot, date(2026, 1, 20))
        assert row["add_exempt_reason"] == "sr8_rebuild"

        lot_none = _lot(trx_id="B1")  # default is None
        row_none = lot_excursions._base_row(lot_none, date(2026, 1, 20))
        assert row_none["add_exempt_reason"] is None

    def test_per_lot_realized_pl_is_distinct_from_campaign_total(self):
        """Regression guard for the 'realized_pl smeared campaign total
        onto every lot' bug the downstream Claude analyst flagged. The
        two fields must NOT be the same value plumbing — realized_pl is
        the per-lot number from lot_closures, campaign_realized_pl is
        the summary total. If a future refactor accidentally reads the
        wrong column, this test catches it before another export goes
        out with duplicated numbers.
        """
        # SNDK-style multi-lot: B1 closed at +$40k, A1 closed at +$28k,
        # campaign total = $68k. A11/A12/A13 all showing $68k on the
        # export was the flagged bug.
        b1 = _lot(trx_id="B1", realized_pl=40000.0, campaign_realized_pl=68000.0)
        a1 = _lot(trx_id="A1", realized_pl=28000.0, campaign_realized_pl=68000.0)

        b1_row = lot_excursions._base_row(b1, date(2026, 1, 20))
        a1_row = lot_excursions._base_row(a1, date(2026, 1, 20))

        # Per-lot P&L differs; campaign total matches on both rows.
        assert b1_row["realized_pl"] == 40000.0
        assert a1_row["realized_pl"] == 28000.0
        assert b1_row["realized_pl"] != a1_row["realized_pl"], \
            "per-lot P&L must not collapse to campaign total"
        assert b1_row["campaign_realized_pl"] == 68000.0
        assert a1_row["campaign_realized_pl"] == 68000.0

    def test_open_lot_realized_pl_is_null_not_zero(self):
        """A still-open lot has no closures → realized_pl is NULL.
        Distinguishes 'not closed yet' from 'closed at $0'. Analysts
        filtering `realized_pl != 0` would get wrong buckets if we
        collapsed the two cases."""
        open_lot = _lot(trx_id="B1", realized_pl=None, campaign_realized_pl=None)
        row = lot_excursions._base_row(open_lot, date(2026, 1, 20))
        assert row["realized_pl"] is None
        assert row["campaign_realized_pl"] is None
