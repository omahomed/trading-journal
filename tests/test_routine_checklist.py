"""Tests for the Trading Checklist (Migration 050) derivation helpers.

Covers the pure-function surface in db_layer that runs on every list
read — overdue-days computation and the trading-day (Mon-Fri) elapsed
count. These are the pieces most likely to drift from spec because the
weekday math is fiddly (Fri tick + Mon today = 1 weekday elapsed, not
overdue; Fri tick + Tue today = 2 elapsed, overdue by 1).

No DB access — module-level helpers only.
"""
from __future__ import annotations

from datetime import date

from db_layer import (
    _routine_overdue_days,
    _routine_weekdays_after,
)


# ── _routine_weekdays_after ─────────────────────────────────────────

class TestWeekdaysAfter:
    """Weekdays STRICTLY after `start`, up to and including `end`."""

    def test_same_day_is_zero(self):
        d = date(2026, 7, 20)  # Mon
        assert _routine_weekdays_after(d, d) == 0

    def test_next_weekday_is_one(self):
        # Mon → Tue = 1 weekday elapsed (Tue).
        assert _routine_weekdays_after(date(2026, 7, 20), date(2026, 7, 21)) == 1

    def test_fri_to_sat_is_zero(self):
        # Fri → Sat = 0 weekdays elapsed (Sat is not a weekday).
        assert _routine_weekdays_after(date(2026, 7, 24), date(2026, 7, 25)) == 0

    def test_fri_to_sun_is_zero(self):
        assert _routine_weekdays_after(date(2026, 7, 24), date(2026, 7, 26)) == 0

    def test_fri_to_mon_is_one(self):
        # Fri → Mon = 1 weekday (Mon).
        assert _routine_weekdays_after(date(2026, 7, 24), date(2026, 7, 27)) == 1

    def test_fri_to_tue_is_two(self):
        # Fri → Tue = 2 weekdays (Mon, Tue). This is the "overdue" boundary
        # for daily items (elapsed_weekdays > 1 → overdue).
        assert _routine_weekdays_after(date(2026, 7, 24), date(2026, 7, 28)) == 2

    def test_mon_to_next_mon_is_five(self):
        # Full week: Mon → next Mon = Tue+Wed+Thu+Fri+Mon = 5 weekdays.
        assert _routine_weekdays_after(date(2026, 7, 20), date(2026, 7, 27)) == 5

    def test_two_weeks_apart(self):
        # Mon 2026-07-20 → Mon 2026-08-03 = 10 weekdays.
        assert _routine_weekdays_after(date(2026, 7, 20), date(2026, 8, 3)) == 10

    def test_end_before_start_is_zero(self):
        assert _routine_weekdays_after(date(2026, 7, 25), date(2026, 7, 20)) == 0


# ── _routine_overdue_days ───────────────────────────────────────────

class TestOverdueDaysCounter:
    """Counter items never overdue — no cadence semantics."""

    def test_counter_never_overdue_no_tick(self):
        assert _routine_overdue_days("daily", "counter", None, date(2026, 7, 25)) is None

    def test_counter_never_overdue_stale_tick(self):
        # Ticked six months ago — still None because it's a counter.
        assert _routine_overdue_days("daily", "counter", date(2026, 1, 1), date(2026, 7, 25)) is None


class TestOverdueDaysDaily:
    """Daily tasks: weekday-elapsed > 1 = overdue."""

    def test_never_ticked_is_none(self):
        # Explicit: never-ticked daily items appear neutral, not overdue.
        # Different from the mockup — matches the spec's "overdue thresholds"
        # (a threshold requires a baseline).
        assert _routine_overdue_days("daily", "task", None, date(2026, 7, 25)) is None

    def test_ticked_today_not_overdue(self):
        assert _routine_overdue_days("daily", "task", date(2026, 7, 25), date(2026, 7, 25)) is None

    def test_ticked_yesterday_not_overdue(self):
        # Mon → Tue = 1 weekday elapsed, threshold is > 1, so not overdue.
        assert _routine_overdue_days("daily", "task", date(2026, 7, 20), date(2026, 7, 21)) is None

    def test_ticked_two_days_ago_overdue_by_1(self):
        # Mon → Wed = 2 weekdays elapsed, overdue by 1.
        assert _routine_overdue_days("daily", "task", date(2026, 7, 20), date(2026, 7, 22)) == 1

    def test_fri_tick_mon_check_not_overdue(self):
        # Fri → Mon = 1 weekday, weekend is transparent.
        assert _routine_overdue_days("daily", "task", date(2026, 7, 24), date(2026, 7, 27)) is None

    def test_fri_tick_tue_check_overdue_by_1(self):
        # Fri → Tue = 2 weekdays, overdue by 1.
        assert _routine_overdue_days("daily", "task", date(2026, 7, 24), date(2026, 7, 28)) == 1

    def test_ticked_last_week_overdue_by_4(self):
        # Mon 2026-07-20 → Mon 2026-07-27 = 5 weekdays, overdue by 4.
        assert _routine_overdue_days("daily", "task", date(2026, 7, 20), date(2026, 7, 27)) == 4


class TestOverdueDaysPeriodic:
    """Weekly / monthly / quarterly use calendar days past threshold."""

    def test_weekly_never_ticked_is_none(self):
        assert _routine_overdue_days("weekly", "task", None, date(2026, 7, 25)) is None

    def test_weekly_ticked_7_days_ago_not_overdue(self):
        # Threshold is > 7, so exactly 7 days is fine.
        assert _routine_overdue_days("weekly", "task", date(2026, 7, 18), date(2026, 7, 25)) is None

    def test_weekly_ticked_8_days_ago_overdue_by_1(self):
        assert _routine_overdue_days("weekly", "task", date(2026, 7, 17), date(2026, 7, 25)) == 1

    def test_monthly_ticked_31_days_ago_not_overdue(self):
        assert _routine_overdue_days("monthly", "task", date(2026, 6, 24), date(2026, 7, 25)) is None

    def test_monthly_ticked_45_days_ago_overdue_by_14(self):
        # 45 - 31 = 14.
        assert _routine_overdue_days("monthly", "task", date(2026, 6, 10), date(2026, 7, 25)) == 14

    def test_quarterly_ticked_92_days_ago_not_overdue(self):
        assert _routine_overdue_days("quarterly", "task", date(2026, 4, 24), date(2026, 7, 25)) is None

    def test_quarterly_ticked_104_days_ago_overdue_by_12(self):
        # Matches the mockup ("MAE / MFE study — 104 days overdue").
        # 104 - 92 = 12.
        assert _routine_overdue_days("quarterly", "task", date(2026, 4, 12), date(2026, 7, 25)) == 12

    def test_unknown_frequency_is_none(self):
        # Defensive: an unrecognized frequency shouldn't crash.
        assert _routine_overdue_days("yearly", "task", date(2026, 1, 1), date(2026, 7, 25)) is None


# ── Seed items sanity ──────────────────────────────────────────────

class TestSystemItemsSeed:
    """The seed list is code-owned — a rename lands via git, not migration."""

    def test_seed_has_seven_items(self):
        from db_layer import _ROUTINE_SYSTEM_ITEMS
        assert len(_ROUTINE_SYSTEM_ITEMS) == 7

    def test_seed_includes_discretionary_counter(self):
        from db_layer import _ROUTINE_SYSTEM_ITEMS
        counter_items = [i for i in _ROUTINE_SYSTEM_ITEMS if i[3] == "counter"]
        assert len(counter_items) == 1
        assert "Discretionary action" in counter_items[0][0]

    def test_seed_daily_items_all_after_close(self):
        # If a new daily seed lands in a different slot, it's probably an
        # oversight — the current canon puts every daily seed after close.
        from db_layer import _ROUTINE_SYSTEM_ITEMS
        daily = [i for i in _ROUTINE_SYSTEM_ITEMS if i[1] == "daily"]
        assert all(i[2] == "after_close" for i in daily), daily

    def test_seed_weekly_items_all_weekend(self):
        from db_layer import _ROUTINE_SYSTEM_ITEMS
        weekly = [i for i in _ROUTINE_SYSTEM_ITEMS if i[1] == "weekly"]
        assert all(i[2] == "weekend" for i in weekly), weekly

    def test_seed_uses_valid_enums(self):
        from db_layer import (
            _ROUTINE_SYSTEM_ITEMS,
            _ROUTINE_FREQUENCIES,
            _ROUTINE_SLOTS,
            _ROUTINE_ITEM_TYPES,
        )
        for name, frequency, slot, item_type, sort_order in _ROUTINE_SYSTEM_ITEMS:
            assert frequency in _ROUTINE_FREQUENCIES, name
            assert slot in _ROUTINE_SLOTS, name
            assert item_type in _ROUTINE_ITEM_TYPES, name
            assert isinstance(sort_order, int), name
