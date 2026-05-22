"""Tests for scripts/import_daily_journal_xlsx.py.

Pure-function parser + validator tests run without a DB. The DB
writer path is exercised through a minimal FakeCursor that records
INSERT calls and replays the RETURNING shape — same pattern as
test_import_robinhood_csv's _FakeCursor.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import import_daily_journal_xlsx as ij  # noqa: E402

FIXTURE_XLSX = SCRIPTS / "fixtures" / "long_term_growth_journal.xlsx"


# ─────────────────────────────────────────────────────────────────────────────
# Coercion helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestCoerceFloat:
    @pytest.mark.parametrize("v,expected", [
        (1.5, 1.5),
        (0, 0.0),
        ("100", 100.0),
        ("$1,050.46", 1050.46),
        ("", 0.0),
        (None, 0.0),
        (float("nan"), 0.0),
    ])
    def test_cases(self, v, expected):
        assert ij._coerce_float(v) == pytest.approx(expected)

    def test_garbage_returns_default(self):
        assert ij._coerce_float("not a number") == 0.0
        assert ij._coerce_float("garbage", default=-1.0) == -1.0


class TestCoerceDate:
    @pytest.mark.parametrize("v,expected", [
        ("2026-01-02", date(2026, 1, 2)),
        ("1/2/2026", date(2026, 1, 2)),
        ("5/21/26", date(2026, 5, 21)),
    ])
    def test_cases(self, v, expected):
        assert ij._coerce_date(v) == expected

    def test_nan_returns_none(self):
        assert ij._coerce_date(float("nan")) is None

    def test_none_returns_none(self):
        assert ij._coerce_date(None) is None

    def test_passthrough_date(self):
        d = date(2026, 5, 21)
        assert ij._coerce_date(d) == d


# ─────────────────────────────────────────────────────────────────────────────
# read_xlsx + compute_derived
# ─────────────────────────────────────────────────────────────────────────────


class TestReadXlsx:
    def test_fixture_row_count(self):
        rows = ij.read_xlsx(FIXTURE_XLSX)
        # Revised fixture covers 3/31/26 onward (37 trading days).
        assert len(rows) == 37

    def test_fixture_date_range(self):
        rows = ij.read_xlsx(FIXTURE_XLSX)
        days = [r["day"] for r in rows]
        assert min(days) == date(2026, 3, 31)
        assert max(days) == date(2026, 5, 21)

    def test_nan_cash_change_coerced_to_zero(self):
        """Days with no cash flow have NaN cash_change in the source.
        read_xlsx must coerce these to 0.0, not leave NaN floats
        that would crash arithmetic downstream."""
        rows = ij.read_xlsx(FIXTURE_XLSX)
        for r in rows:
            assert isinstance(r["cash_change"], float)
            assert not math.isnan(r["cash_change"])

    def test_header_row_dropped(self):
        """The embedded header 'Date / Beg NLV / ...' must not appear
        as a data row — its date string ('Date') doesn't parse."""
        rows = ij.read_xlsx(FIXTURE_XLSX)
        assert all(isinstance(r["day"], date) for r in rows)


class TestComputeDerived:
    def test_pct_with_prev_end_baseline(self):
        """Standard no-deposit day: prev_end=100, end=102.5, cash=0.
        Dollar = 2.5, pct = 2.5/100 * 100 = 2.5%. The row's xlsx
        beg_nlv is IGNORED and overridden to prev_end."""
        r = ij.compute_derived({
            "day": date(2026, 3, 15),
            "beg_nlv": 999,        # ignored — overridden to prev_end
            "cash_change": 0.0,
            "end_nlv": 102.5,
        }, prev_end_nlv=100.0)
        assert r["beg_nlv"] == 100.0
        assert r["daily_dollar_change"] == pytest.approx(2.5)
        assert r["daily_pct_change"] == pytest.approx(2.5)

    def test_pct_uses_adjusted_baseline(self):
        """$1000 deposit + $50 gain. Divisor = prev_end + cash (post-
        deposit baseline) per app convention. Note: dollar uses
        prev_end (NOT xlsx beg)."""
        r = ij.compute_derived({
            "day": date(2026, 3, 15),
            "beg_nlv": 999,        # ignored
            "cash_change": 1000.0,
            "end_nlv": 11050.0,
        }, prev_end_nlv=10000.0)
        assert r["beg_nlv"] == 10000.0
        assert r["daily_dollar_change"] == pytest.approx(50.0)
        # 50 / (10000 + 1000) * 100 ≈ 0.4545%
        assert r["daily_pct_change"] == pytest.approx(50.0 / 11000.0 * 100)

    def test_zero_adjusted_beg_yields_none_pct(self):
        """When prev_end = 0 AND cash = 0, adjusted_beg = 0 → pct is
        None. Covers the empty-account early-January rows."""
        r = ij.compute_derived({
            "day": date(2026, 1, 5),
            "beg_nlv": 0.0,
            "cash_change": 0.0,
            "end_nlv": 0.0,
        }, prev_end_nlv=0.0)
        assert r["daily_dollar_change"] == 0.0
        assert r["daily_pct_change"] is None

    def test_first_deposit_day_with_zero_prev_end(self):
        """First deposit day after an empty stretch: prev_end=0,
        cash=4850, end=4850. adjusted_beg = 4850; dollar = 0;
        pct = 0.0 (defined math, not None)."""
        r = ij.compute_derived({
            "day": date(2026, 1, 12),
            "beg_nlv": 0.0,
            "cash_change": 4850.0,
            "end_nlv": 4850.0,
        }, prev_end_nlv=0.0)
        assert r["beg_nlv"] == 0.0
        assert r["daily_dollar_change"] == pytest.approx(0.0)
        assert r["daily_pct_change"] == pytest.approx(0.0)

    def test_pct_matches_app_convention(self):
        """Anchor the formula to daily-routine.tsx:255-276:
            portDailyChg = portNlvN - portPrev - portCashN
            portAdj      = portPrev + portCashN
            portDailyPct = (portDailyChg / portAdj) * 100

        Synthetic: prev_end=10000, cash=5000, end=15500.
        Dollar = 500. Pct = 500/15000*100 ≈ 3.333%."""
        r = ij.compute_derived({
            "day": date(2026, 3, 15),
            "beg_nlv": 999,        # ignored
            "cash_change": 5000.0,
            "end_nlv": 15500.0,
        }, prev_end_nlv=10000.0)
        assert r["beg_nlv"] == 10000.0
        assert r["daily_dollar_change"] == pytest.approx(500.0)
        assert r["daily_pct_change"] == pytest.approx(500.0 / 15000.0 * 100)

    def test_first_row_returns_none_pct(self):
        """First row has no prior baseline (prev_end_nlv=None).
        Contract: daily_dollar_change = 0, daily_pct_change = None,
        beg_nlv preserved from xlsx (not overridden)."""
        r = ij.compute_derived({
            "day": date(2026, 1, 2),
            "beg_nlv": 0.0,
            "cash_change": 0.0,
            "end_nlv": 0.0,
        }, prev_end_nlv=None)
        assert r["beg_nlv"] == 0.0
        assert r["daily_dollar_change"] == 0.0
        assert r["daily_pct_change"] is None

    def test_beg_nlv_overridden_on_subsequent_rows(self):
        """Xlsx beg_nlv (5125.55) is ignored; output uses prev_end
        (4741.92). This is the user's real 2026-01-13 row case."""
        r = ij.compute_derived({
            "day": date(2026, 1, 14),
            "beg_nlv": 5125.55,
            "cash_change": 0.0,
            "end_nlv": 14258.13,
        }, prev_end_nlv=4741.92)
        assert r["beg_nlv"] == 4741.92          # overridden
        # dollar = 14258.13 - 4741.92 - 0 = 9516.21
        assert r["daily_dollar_change"] == pytest.approx(9516.21)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectDuplicates:
    def test_no_duplicates(self):
        rows = [
            {"day": date(2026, 3, 1), "beg_nlv": 0, "cash_change": 0, "end_nlv": 0},
            {"day": date(2026, 3, 2), "beg_nlv": 0, "cash_change": 0, "end_nlv": 0},
        ]
        assert ij.detect_duplicates(rows) == []

    def test_finds_duplicate(self):
        rows = [
            {"day": date(2026, 3, 1), "beg_nlv": 0, "cash_change": 0, "end_nlv": 0},
            {"day": date(2026, 3, 1), "beg_nlv": 1, "cash_change": 0, "end_nlv": 0},
            {"day": date(2026, 3, 2), "beg_nlv": 0, "cash_change": 0, "end_nlv": 0},
        ]
        assert ij.detect_duplicates(rows) == [date(2026, 3, 1)]


class TestDetectNlvContinuityGaps:
    def _r(self, d, beg, end):
        return {"day": d, "beg_nlv": beg, "cash_change": 0, "end_nlv": end}

    def test_perfect_continuity_yields_no_gaps(self):
        rows = [
            self._r(date(2026, 3, 1), 1000, 1100),
            self._r(date(2026, 3, 2), 1100, 1050),
            self._r(date(2026, 3, 3), 1050, 1075),
        ]
        assert ij.detect_nlv_continuity_gaps(rows) == []

    def test_finds_gap(self):
        rows = [
            self._r(date(2026, 3, 1), 1000, 1100),
            self._r(date(2026, 3, 2), 1050, 1075),   # 50 gap from prior
        ]
        gaps = ij.detect_nlv_continuity_gaps(rows)
        assert len(gaps) == 1
        d_a, e, d_b, b, delta = gaps[0]
        assert d_a == date(2026, 3, 1)
        assert e == 1100
        assert d_b == date(2026, 3, 2)
        assert b == 1050
        assert delta == pytest.approx(50.0)

    def test_tolerance_absorbs_rounding_noise(self):
        rows = [
            self._r(date(2026, 3, 1), 1000.00, 1100.00),
            self._r(date(2026, 3, 2), 1100.30, 1075.00),  # 30¢ gap
        ]
        assert ij.detect_nlv_continuity_gaps(rows, tolerance=0.5) == []

    def test_fixture_has_known_gap_count(self):
        """Revised fixture (3/31-onward) has 36 day-pair gaps > 50¢.
        Locks the expected shape so a future re-export with cleaner
        data signals correctly."""
        rows = ij.read_xlsx(FIXTURE_XLSX)
        gaps = ij.detect_nlv_continuity_gaps(rows)
        assert len(gaps) == 36


class TestFilterByDate:
    def test_drops_pre_cutoff(self):
        rows = [
            {"day": date(2025, 12, 31), "beg_nlv": 0, "cash_change": 0, "end_nlv": 0},
            {"day": date(2026, 1, 2), "beg_nlv": 0, "cash_change": 0, "end_nlv": 0},
        ]
        kept, dropped = ij.filter_by_date(rows, date(2026, 1, 1))
        assert len(kept) == 1
        assert dropped == 1


# ─────────────────────────────────────────────────────────────────────────────
# DB writer (mocked cursor)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeCursor:
    """Records INSERT calls and replays the RETURNING (xmax = 0)
    shape so write_rows can be tested without a DB.

    `conflict_rows` is a set of dates that should simulate an existing
    row. For DO NOTHING those return None from fetchone(); for
    DO UPDATE those return (False,) (xmax != 0)."""

    def __init__(self, conflict_rows: set[date] | None = None,
                 overwrite: bool = False):
        self.conflict_rows = conflict_rows or set()
        self.overwrite = overwrite
        self.inserts: list[tuple] = []
        self._last_returned = None

    def execute(self, sql: str, params: tuple) -> None:
        # write_rows passes (portfolio_id, user_id, day, beg_nlv, cash, end, dd, dp)
        if "INSERT INTO trading_journal" in sql:
            self.inserts.append(params)
            day = params[2]
            if day in self.conflict_rows:
                if self.overwrite:
                    self._last_returned = (False,)   # xmax != 0 → updated
                else:
                    self._last_returned = None         # DO NOTHING swallowed it
            else:
                self._last_returned = (True,)         # xmax = 0 → inserted

    def fetchone(self):
        return self._last_returned


class TestWriteRows:
    def _enriched(self, day, beg=1000, cash=0, end=1100):
        return ij.compute_derived({
            "day": day, "beg_nlv": beg, "cash_change": cash, "end_nlv": end,
        })

    def test_all_inserted_no_conflicts(self):
        cur = _FakeCursor()
        rows = [
            self._enriched(date(2026, 3, 1)),
            self._enriched(date(2026, 3, 2)),
        ]
        counts = ij.write_rows(cur, 1, "u", rows, overwrite=False)
        assert counts == {"inserted": 2, "updated": 0, "skipped": 0}
        assert len(cur.inserts) == 2

    def test_do_nothing_skips_on_conflict(self):
        cur = _FakeCursor(conflict_rows={date(2026, 3, 1)})
        rows = [
            self._enriched(date(2026, 3, 1)),
            self._enriched(date(2026, 3, 2)),
        ]
        counts = ij.write_rows(cur, 1, "u", rows, overwrite=False)
        assert counts == {"inserted": 1, "updated": 0, "skipped": 1}

    def test_overwrite_updates_on_conflict(self):
        cur = _FakeCursor(conflict_rows={date(2026, 3, 1)}, overwrite=True)
        rows = [
            self._enriched(date(2026, 3, 1)),
            self._enriched(date(2026, 3, 2)),
        ]
        counts = ij.write_rows(cur, 1, "u", rows, overwrite=True)
        assert counts == {"inserted": 1, "updated": 1, "skipped": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Real-fixture integration
# ─────────────────────────────────────────────────────────────────────────────


class TestRealFixtureIntegration:
    def test_row_count(self):
        rows = ij.read_xlsx(FIXTURE_XLSX)
        # Revised fixture covers 3/31/26 onward (post migration 040).
        assert len(rows) == 37

    def test_date_range(self):
        rows = ij.read_xlsx(FIXTURE_XLSX)
        days = sorted(r["day"] for r in rows)
        assert days[0] == date(2026, 3, 31)
        assert days[-1] == date(2026, 5, 21)

    def test_five_nonzero_cash_entries_totaling_16830(self):
        """Post-reset xlsx has 5 cash injections totalling $16,830
        (the user's deposits between 4/2 and 5/21)."""
        rows = ij.read_xlsx(FIXTURE_XLSX)
        nz = [r for r in rows if r["cash_change"] != 0]
        assert len(nz) == 5
        assert sum(r["cash_change"] for r in nz) == pytest.approx(16830.0)

    def test_chained_pipeline_against_real_fixture(self):
        """Run the full 37-row revised fixture through the chained-
        prev_end loop and anchor known days:
          - First row (2026-03-31): pct = None (no prior baseline);
            beg_nlv = 14681.62 (preserved from xlsx, matches the
            migration-040 starting_capital)
          - 4/2 (first deposit day): dollar ≈ +$399, pct ≈ +2.17%
          - 5/20 (typical day): dollar ≈ +$1144, pct ≈ +3.39%
          - 5/21 (deposit + gain day): dollar ≈ +$4577, pct ≈ +11.05%
        """
        rows = sorted(ij.read_xlsx(FIXTURE_XLSX), key=lambda r: r["day"])
        enriched: list[dict] = []
        prev_end: float | None = None
        for r in rows:
            enriched.append(ij.compute_derived(r, prev_end_nlv=prev_end))
            prev_end = r["end_nlv"]

        by_day = {e["day"]: e for e in enriched}

        first = by_day[date(2026, 3, 31)]
        assert first["daily_pct_change"] is None
        assert first["beg_nlv"] == pytest.approx(14681.62, abs=0.01)

        apr2 = by_day[date(2026, 4, 2)]
        assert apr2["daily_dollar_change"] == pytest.approx(399.11, abs=0.05)
        assert apr2["daily_pct_change"] == pytest.approx(2.17, abs=0.05)

        may20 = by_day[date(2026, 5, 20)]
        assert may20["daily_dollar_change"] == pytest.approx(1143.75, abs=0.5)
        assert may20["daily_pct_change"] == pytest.approx(3.39, abs=0.05)

        may21 = by_day[date(2026, 5, 21)]
        assert may21["daily_dollar_change"] == pytest.approx(4577.26, abs=0.5)
        assert may21["daily_pct_change"] == pytest.approx(11.05, abs=0.05)

        # beg_nlv override: 5/21's xlsx beg was 39506.79 (a snapshot
        # that already included intraday gain), now overridden to 5/20's
        # end_nlv = 34929.53 per app convention.
        assert may21["beg_nlv"] == pytest.approx(34929.53, abs=0.05)
