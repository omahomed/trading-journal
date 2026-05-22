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
        assert len(rows) == 97

    def test_fixture_date_range(self):
        rows = ij.read_xlsx(FIXTURE_XLSX)
        days = [r["day"] for r in rows]
        assert min(days) == date(2026, 1, 2)
        assert max(days) == date(2026, 5, 21)

    def test_nan_cash_change_coerced_to_zero(self):
        """First-week rows have NaN cash_change in the source.
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
    def test_basic_no_cash_change(self):
        r = ij.compute_derived({
            "day": date(2026, 3, 15),
            "beg_nlv": 10000.0,
            "cash_change": 0.0,
            "end_nlv": 10250.0,
        })
        assert r["daily_dollar_change"] == pytest.approx(250.0)
        assert r["daily_pct_change"] == pytest.approx(0.025)

    def test_with_cash_change_excluded_from_pct(self):
        """A $1000 deposit + $50 gain on $10k → daily_dollar=50, not 1050."""
        r = ij.compute_derived({
            "day": date(2026, 3, 15),
            "beg_nlv": 10000.0,
            "cash_change": 1000.0,
            "end_nlv": 11050.0,
        })
        assert r["daily_dollar_change"] == pytest.approx(50.0)
        assert r["daily_pct_change"] == pytest.approx(0.005)

    def test_beg_nlv_zero_yields_none_pct(self):
        """Zero-NLV days (account just opened, no money yet) must
        return None for daily_pct_change — math is undefined."""
        r = ij.compute_derived({
            "day": date(2026, 1, 2),
            "beg_nlv": 0.0,
            "cash_change": 0.0,
            "end_nlv": 0.0,
        })
        assert r["daily_dollar_change"] == 0.0
        assert r["daily_pct_change"] is None

    def test_cash_injection_day_with_zero_beg_nlv(self):
        """The first deposit day — beg_nlv=0, cash=4850, end=4850.
        Should NOT crash; daily_pct_change = None."""
        r = ij.compute_derived({
            "day": date(2026, 1, 12),
            "beg_nlv": 0.0,
            "cash_change": 4850.0,
            "end_nlv": 4850.0,
        })
        assert r["daily_dollar_change"] == pytest.approx(0.0)
        assert r["daily_pct_change"] is None


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
        """Real xlsx fixture has 90 day-pair gaps > 50¢. Locks the
        expected shape so a future re-export with cleaner data
        signals correctly."""
        rows = ij.read_xlsx(FIXTURE_XLSX)
        gaps = ij.detect_nlv_continuity_gaps(rows)
        assert len(gaps) == 90


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
        assert len(rows) == 97

    def test_date_range(self):
        rows = ij.read_xlsx(FIXTURE_XLSX)
        days = sorted(r["day"] for r in rows)
        assert days[0] == date(2026, 1, 2)
        assert days[-1] == date(2026, 5, 21)

    def test_twelve_nonzero_cash_entries_totaling_48420(self):
        rows = ij.read_xlsx(FIXTURE_XLSX)
        nz = [r for r in rows if r["cash_change"] != 0]
        assert len(nz) == 12
        assert sum(r["cash_change"] for r in nz) == pytest.approx(48420.0)

    def test_zero_nlv_rows_resolve_to_none_pct(self):
        """The early-January rows (account empty before first deposit)
        all have beg_nlv = 0. compute_derived must yield None for
        daily_pct_change on each."""
        rows = ij.read_xlsx(FIXTURE_XLSX)
        zero_nlv = [r for r in rows if r["beg_nlv"] == 0]
        enriched = [ij.compute_derived(r) for r in zero_nlv]
        assert all(r["daily_pct_change"] is None for r in enriched)
        # Expected: 7 such rows (Jan 2, 5, 6, 7, 8, 9, 12).
        assert len(zero_nlv) == 7
