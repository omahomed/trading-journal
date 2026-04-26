"""Tests for api/mct_signals_writer.py.

Tests use a sentinel trade_date range far in the past (years 1900–1901) to
avoid colliding with real engine output in the market_signals table. Each
test deletes those sentinel rows on setup and teardown so partial failures
don't leak state.
"""

from __future__ import annotations

import os
from datetime import date

import pytest


SENTINEL_DATE_RANGE = (date(1900, 1, 1), date(1901, 12, 31))


def _has_db() -> bool:
    return bool(os.getenv("DATABASE_URL"))


requires_db = pytest.mark.skipif(
    not _has_db(),
    reason="DATABASE_URL not set",
)


def _delete_sentinel_rows():
    from db_layer import get_db_connection
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM market_signals WHERE trade_date BETWEEN %s AND %s",
                SENTINEL_DATE_RANGE,
            )
        conn.commit()


@pytest.fixture
def clean_sentinel_rows():
    if not _has_db():
        pytest.skip("DATABASE_URL not set")
    _delete_sentinel_rows()
    yield
    _delete_sentinel_rows()


def _make_event(d: date, signal_type: str = "STEP_0_RALLY_DAY"):
    from api.mct_engine import SignalEvent
    return SignalEvent(
        trade_date=d,
        signal_type=signal_type,
        signal_label=f"{signal_type} test",
        exposure_before=0,
        exposure_after=20,
        state_before="CORRECTION",
        state_after="RALLY MODE",
        meta={"test": True},
    )


@requires_db
def test_write_signals_inserts_new_rows(clean_sentinel_rows):
    from api.mct_signals_writer import write_signals
    events = [
        _make_event(date(1900, 1, 2), "STEP_0_RALLY_DAY"),
        _make_event(date(1900, 1, 3), "STEP_1_FTD"),
        _make_event(date(1900, 1, 4), "STEP_2_CLOSE_ABOVE_21EMA"),
    ]
    inserted = write_signals(events)
    assert inserted == 3


@requires_db
def test_write_signals_idempotent_on_conflict(clean_sentinel_rows):
    """Re-writing the same events returns 0 new inserts."""
    from api.mct_signals_writer import write_signals
    events = [
        _make_event(date(1900, 2, 1), "STEP_0_RALLY_DAY"),
        _make_event(date(1900, 2, 2), "STEP_1_FTD"),
    ]
    first = write_signals(events)
    second = write_signals(events)
    assert first == 2
    assert second == 0


@requires_db
def test_write_signals_validates_signal_type(clean_sentinel_rows):
    """Unknown signal_type raises ValueError before any DB writes."""
    from api.mct_engine import SignalEvent
    from api.mct_signals_writer import write_signals
    bad = SignalEvent(
        trade_date=date(1900, 3, 1),
        signal_type="NOT_A_REAL_SIGNAL",
        signal_label="bogus",
        exposure_before=100, exposure_after=100,
        state_before="UPTREND", state_after="UPTREND",
        meta={},
    )
    with pytest.raises(ValueError, match="Unknown signal_type"):
        write_signals([bad])


@requires_db
def test_write_signals_for_date_range_round_trip(clean_sentinel_rows):
    """write_signals_for_date_range over a real range writes the engine output.

    The Feb 2025 range falls inside production market_data, so this test must
    clean up its writes to leave the table in the same state it found it.
    """
    from api.mct_signals_writer import write_signals_for_date_range
    from db_layer import get_db_connection

    test_range = (date(2025, 2, 25), date(2025, 3, 1))

    def _cleanup():
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM market_signals WHERE trade_date BETWEEN %s AND %s",
                    test_range,
                )
            conn.commit()

    _cleanup()
    try:
        summary = write_signals_for_date_range(
            test_range[0], test_range[1],
            initial_reference_high=20118.61,
        )
        # 2/27/2025 fires CORRECTION_DECLARED; the range may emit other signals too.
        assert summary["events_emitted"] >= 1

        # Idempotency: same range, second call should insert 0 new rows.
        summary2 = write_signals_for_date_range(
            test_range[0], test_range[1],
            initial_reference_high=20118.61,
        )
        assert summary2["rows_inserted"] == 0
    finally:
        _cleanup()
