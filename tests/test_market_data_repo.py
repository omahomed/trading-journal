"""Tests for api/market_data_repo.py and the indicator computation contract.

The DB-backed tests connect to whatever database DATABASE_URL points at and
isolate themselves with a sentinel `symbol` (TEST_MCT_SYMBOL) so they cannot
collide with production rows. Each fixture deletes any rows for the sentinel
symbol on setup and teardown, so partial failures cannot leak state.

If DATABASE_URL is not set, the DB-dependent tests skip cleanly. The pure
indicator test runs unconditionally.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import psycopg2
import pytest
from psycopg2.extras import execute_values


TEST_MCT_SYMBOL = "TEST_MCT_PHASE1"


# -----------------------------------------------------------------------------
# Pure indicator-contract test (no DB)
# -----------------------------------------------------------------------------

def test_indicators_match_pandas() -> None:
    """Backfill's indicator math must match a fresh pandas computation."""
    from scripts.backfill_market_data import compute_indicators  # local import

    rng = np.random.default_rng(seed=42)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, size=300))
    base = pd.DataFrame({
        "trade_date": [date(2020, 1, 1) + timedelta(days=i) for i in range(300)],
        "open": closes - 0.1,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": [1_000_000] * 300,
    })

    out = compute_indicators(base.copy())

    expected_ema_21 = base["close"].ewm(span=21, adjust=False).mean()
    expected_sma_50 = base["close"].rolling(window=50).mean()
    expected_sma_200 = base["close"].rolling(window=200).mean()

    pd.testing.assert_series_equal(
        out["ema_21"].reset_index(drop=True),
        expected_ema_21.reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        out["sma_50"].reset_index(drop=True),
        expected_sma_50.reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        out["sma_200"].reset_index(drop=True),
        expected_sma_200.reset_index(drop=True),
        check_names=False,
    )


# -----------------------------------------------------------------------------
# DB-backed tests
# -----------------------------------------------------------------------------

def _has_database_url() -> bool:
    return bool(os.getenv("DATABASE_URL"))


pytestmark_db = pytest.mark.skipif(
    not _has_database_url(),
    reason="DATABASE_URL not set; skipping DB-dependent tests",
)


def _delete_test_rows() -> None:
    from db_layer import get_db_connection
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM market_data WHERE symbol = %s", (TEST_MCT_SYMBOL,))
        conn.commit()


def _insert_test_rows(rows: list[tuple]) -> None:
    """Insert rows formatted as (trade_date, open, high, low, close, volume,
    ema_8, ema_21, sma_50, sma_200)."""
    from db_layer import get_db_connection
    payload = [
        (TEST_MCT_SYMBOL, *row) for row in rows
    ]
    sql = """
        INSERT INTO market_data (
            symbol, trade_date, open, high, low, close, volume,
            ema_8, ema_21, sma_50, sma_200
        ) VALUES %s
        ON CONFLICT (symbol, trade_date) DO NOTHING
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, payload)
        conn.commit()


@pytest.fixture
def clean_test_rows():
    if not _has_database_url():
        pytest.skip("DATABASE_URL not set")
    _delete_test_rows()
    yield TEST_MCT_SYMBOL
    _delete_test_rows()


@pytestmark_db
class TestRepo:
    def test_get_history_returns_correct_range(self, clean_test_rows) -> None:
        from api.market_data_repo import get_history

        rows = [
            (date(2024, 1, 2), 100.0, 101.0, 99.5, 100.5, 1_000_000, 100.5, 100.5, None, None),
            (date(2024, 1, 3), 100.5, 102.0, 100.0, 101.5, 1_100_000, 100.7, 100.6, None, None),
            (date(2024, 1, 4), 101.5, 103.0, 101.0, 102.0, 1_200_000, 100.9, 100.7, None, None),
            (date(2024, 1, 5), 102.0, 102.5, 100.5, 101.0, 1_050_000, 101.0, 100.7, None, None),
        ]
        _insert_test_rows(rows)

        out = get_history(TEST_MCT_SYMBOL, date(2024, 1, 3), date(2024, 1, 4))
        assert len(out) == 2
        assert list(out["trade_date"]) == [date(2024, 1, 3), date(2024, 1, 4)]
        assert out["close"].tolist() == [101.5, 102.0]

    def test_get_recent_returns_n_bars_ascending(self, clean_test_rows) -> None:
        from api.market_data_repo import get_recent

        rows = [
            (date(2024, 1, 1) + timedelta(days=i),
             100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1_000_000,
             None, None, None, None)
            for i in range(10)
        ]
        _insert_test_rows(rows)

        out = get_recent(TEST_MCT_SYMBOL, 5)
        assert len(out) == 5
        # Last 5 rows ascending → trade_date 1/6 through 1/10
        assert list(out["trade_date"]) == [date(2024, 1, 6) + timedelta(days=i) for i in range(5)]

    def test_get_latest_date_returns_most_recent(self, clean_test_rows) -> None:
        from api.market_data_repo import get_latest_date

        rows = [
            (date(2024, 3, 10), 100, 101, 99, 100, 1_000_000, None, None, None, None),
            (date(2024, 3, 12), 101, 102, 100, 101, 1_000_000, None, None, None, None),
            (date(2024, 3, 11), 100, 101, 99, 100, 1_000_000, None, None, None, None),
        ]
        _insert_test_rows(rows)

        assert get_latest_date(TEST_MCT_SYMBOL) == date(2024, 3, 12)

    def test_get_bar_returns_none_for_missing_date(self, clean_test_rows) -> None:
        from api.market_data_repo import get_bar

        # Insert a single row, then query a different date.
        _insert_test_rows([
            (date(2024, 6, 1), 100, 101, 99, 100, 1_000_000, None, None, None, None),
        ])

        assert get_bar(TEST_MCT_SYMBOL, date(2024, 6, 2)) is None

    def test_get_bar_returns_dict_for_existing_date(self, clean_test_rows) -> None:
        from api.market_data_repo import get_bar

        _insert_test_rows([
            (date(2024, 6, 1), 100.25, 101.5, 99.75, 100.75, 1_234_567,
             100.5, 100.4, None, None),
        ])

        bar = get_bar(TEST_MCT_SYMBOL, date(2024, 6, 1))
        assert bar is not None
        assert bar["symbol"] == TEST_MCT_SYMBOL
        assert bar["trade_date"] == date(2024, 6, 1)
        assert float(bar["close"]) == pytest.approx(100.75)
        assert int(bar["volume"]) == 1_234_567
        assert bar["sma_50"] is None
