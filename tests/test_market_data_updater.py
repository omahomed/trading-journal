"""Unit tests for api.market_data_updater helpers.

Focus: `_last_business_day` post-close gate. Before the 22:00 UTC gate
was added, calls during market hours (or worse — pre-market) would
target today's date, causing update_if_needed to fetch yfinance and
persist an intraday snapshot as the "settled" bar. That snapshot's
close/high/low then flowed into the MCT engine and produced wrong
state (STEP_0 pink-day gate uses position_in_range which the intraday
range mis-reports; violation checks are close-based).

These tests pin the corrected behavior across the weekday boundary,
the post-close threshold, and weekend fall-through.
"""
from __future__ import annotations

from datetime import date, datetime

from api.market_data_updater import _last_business_day


# Reference calendar (all UTC, all 2026):
#   Fri 2026-07-24, Sat 07-25, Sun 07-26, Mon 07-27, Tue 07-28


def test_pre_close_weekday_rolls_back_to_previous_weekday():
    """Monday 8:45 AM CT = 13:45 UTC → still intraday → target is Friday.
    This is the concrete regression that motivated the fix: the 8:45 AM
    ingest was writing today's bar with a stale early-morning snapshot."""
    mon_pre_close = datetime(2026, 7, 27, 13, 45)  # UTC, well before 22:00
    assert _last_business_day(mon_pre_close) == date(2026, 7, 24)


def test_at_close_still_pre_settled():
    """4 PM ET in EDT = 20:00 UTC. That's market close but yfinance may not
    have published the daily bar yet. Under the 22:00 UTC gate we keep
    rolling back — 2 hours of buffer catches yfinance's publish delay."""
    mon_at_close = datetime(2026, 7, 27, 20, 0)
    assert _last_business_day(mon_at_close) == date(2026, 7, 24)


def test_post_close_settles_today():
    """22:00 UTC = 5 PM ET (EDT) / 4 PM ET (EST). At/after this
    threshold, today is considered settled and becomes the target."""
    mon_settled = datetime(2026, 7, 27, 22, 30)
    assert _last_business_day(mon_settled) == date(2026, 7, 27)


def test_saturday_rolls_to_friday_regardless_of_time():
    """Weekend logic is time-independent — Sat/Sun always target the
    prior Friday, so a Sat afternoon page load doesn't try to fetch a
    non-existent weekend bar."""
    sat_afternoon = datetime(2026, 7, 25, 20, 0)
    sat_evening = datetime(2026, 7, 25, 23, 30)
    assert _last_business_day(sat_afternoon) == date(2026, 7, 24)
    assert _last_business_day(sat_evening) == date(2026, 7, 24)


def test_sunday_rolls_to_friday():
    """Same weekend rule for Sunday."""
    sun_morning = datetime(2026, 7, 26, 12, 0)
    sun_evening = datetime(2026, 7, 26, 23, 45)
    assert _last_business_day(sun_morning) == date(2026, 7, 24)
    assert _last_business_day(sun_evening) == date(2026, 7, 24)


def test_monday_pre_close_rolls_to_friday_not_sunday():
    """Regression guard: the weekend fall-through loop must run AFTER the
    intraday roll-back. Monday 8:45 AM CT rolls back one day → Sunday →
    weekend loop rolls Sunday back to Friday. Order matters; if the
    weekend loop ran first we'd stop at Monday (weekday) and then the
    intraday roll-back would land on Sunday (missed weekend logic)."""
    mon_pre_close = datetime(2026, 7, 27, 13, 45)
    assert _last_business_day(mon_pre_close) == date(2026, 7, 24)


def test_tuesday_early_morning_targets_monday_only_after_monday_settled():
    """Tuesday 8 AM UTC (3 AM ET) — before Tuesday close AND after Monday
    close. Should target Monday (the last settled weekday)."""
    tue_early = datetime(2026, 7, 28, 8, 0)
    assert _last_business_day(tue_early) == date(2026, 7, 27)
