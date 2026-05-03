"""Tests for db_layer.generate_unique_trx_id (collision-safe trx_id helper).

DB-backed tests connect to whatever DATABASE_URL points at and isolate
themselves with a sentinel trade_id ('TEST-TRX-GEN-001') so they cannot
collide with production trades. Each fixture hard-deletes the sentinel
summary + details + closures on setup AND teardown, so partial failures
cannot leak state.

If DATABASE_URL is not set, the DB-dependent tests skip cleanly. The
concurrency test is additionally skipped under CI because threading
+ Postgres timing can be flaky in shared CI environments.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime

import pytest


pytestmark_db = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping DB-dependent tests",
)


# Sentinels — TEST_TRADE_ID is unique enough that it cannot collide with
# production trade_ids (which follow the YYYYMM-NNN format).
TEST_PORTFOLIO = "CanSlim"  # Real portfolio (must exist); sentinel by trade_id.
TEST_TRADE_ID = "TEST-TRX-GEN-001"
TEST_TICKER = "TESTTRX"
FOUNDER_UUID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"


def _set_user_context() -> None:
    """RLS gate — db_layer reads tenant from a ContextVar."""
    from db_layer import current_user_id
    current_user_id.set(FOUNDER_UUID)


def _get_portfolio_id() -> int | None:
    from db_layer import get_db_connection
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM portfolios WHERE name = %s", (TEST_PORTFOLIO,))
            r = cur.fetchone()
            return r[0] if r else None


def _delete_test_rows() -> None:
    """Hard-delete (NOT soft-delete) every row tied to the sentinel trade.
    Used by the cleanup fixture so partial failures cannot leak state."""
    from db_layer import get_db_connection
    portfolio_id = _get_portfolio_id()
    if portfolio_id is None:
        return
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM lot_closures WHERE portfolio_id = %s AND trade_id = %s",
                (portfolio_id, TEST_TRADE_ID),
            )
            cur.execute(
                "DELETE FROM trades_details WHERE portfolio_id = %s AND trade_id = %s",
                (portfolio_id, TEST_TRADE_ID),
            )
            cur.execute(
                "DELETE FROM trades_summary WHERE portfolio_id = %s AND trade_id = %s",
                (portfolio_id, TEST_TRADE_ID),
            )
        conn.commit()


def _seed_summary() -> None:
    """trades_details has an FK to trades_summary on (portfolio_id, trade_id),
    so we need a summary row before we can insert details."""
    from db_layer import save_summary_row
    summary_row = {
        "Trade_ID": TEST_TRADE_ID, "Ticker": TEST_TICKER, "Status": "OPEN",
        "Open_Date": datetime.now().strftime("%Y-%m-%d"),
        "Shares": 0, "Avg_Entry": 0, "Total_Cost": 0,
    }
    save_summary_row(TEST_PORTFOLIO, summary_row)


def _insert_detail(action: str, trx_id: str) -> None:
    """Insert one detail row to set up a test scenario with pre-existing trx_ids."""
    from db_layer import save_detail_row
    detail_row = {
        "Trade_ID": TEST_TRADE_ID, "Ticker": TEST_TICKER, "Action": action,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Shares": 1, "Amount": 100, "Value": 100,
        "Trx_ID": trx_id,
    }
    save_detail_row(TEST_PORTFOLIO, detail_row)


@pytest.fixture
def clean_test_trade():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")
    _set_user_context()
    _delete_test_rows()
    _seed_summary()
    yield TEST_TRADE_ID
    _delete_test_rows()


@pytestmark_db
class TestSerial:
    """Single-threaded behavior — what the helper guarantees on its own."""

    def test_returns_S1_when_no_existing(self, clean_test_trade) -> None:
        from db_layer import generate_unique_trx_id
        assert generate_unique_trx_id(TEST_PORTFOLIO, TEST_TRADE_ID, "S") == "S1"

    def test_skips_existing_S1_to_S2(self, clean_test_trade) -> None:
        from db_layer import generate_unique_trx_id
        _insert_detail("SELL", "S1")
        assert generate_unique_trx_id(TEST_PORTFOLIO, TEST_TRADE_ID, "S") == "S2"

    def test_fills_gap_in_sequence(self, clean_test_trade) -> None:
        """A deleted (or missing) suffix in the middle should be reused."""
        from db_layer import generate_unique_trx_id
        _insert_detail("SELL", "S1")
        _insert_detail("SELL", "S3")  # gap at S2
        assert generate_unique_trx_id(TEST_PORTFOLIO, TEST_TRADE_ID, "S") == "S2"

    def test_serial_5_calls_yield_S1_through_S5(self, clean_test_trade) -> None:
        """5 sequential calls (each followed by an INSERT of the returned id)
        should yield exactly S1, S2, S3, S4, S5 — strictly increasing, no gaps,
        no duplicates."""
        from db_layer import generate_unique_trx_id
        results: list[str] = []
        for _ in range(5):
            trx = generate_unique_trx_id(TEST_PORTFOLIO, TEST_TRADE_ID, "S")
            results.append(trx)
            _insert_detail("SELL", trx)
        assert results == ["S1", "S2", "S3", "S4", "S5"]

    def test_different_prefixes_are_independent(self, clean_test_trade) -> None:
        """B / A / S sequences don't share a counter — the regex isolates each."""
        from db_layer import generate_unique_trx_id
        _insert_detail("BUY", "B1")
        _insert_detail("BUY", "A1")
        _insert_detail("SELL", "S1")
        assert generate_unique_trx_id(TEST_PORTFOLIO, TEST_TRADE_ID, "B") == "B2"
        assert generate_unique_trx_id(TEST_PORTFOLIO, TEST_TRADE_ID, "A") == "A2"
        assert generate_unique_trx_id(TEST_PORTFOLIO, TEST_TRADE_ID, "S") == "S2"

    def test_legacy_two_letter_prefix_does_not_consume_single_letter_suffix(
        self, clean_test_trade,
    ) -> None:
        """Legacy SA1 / SB1 rows must not block S1. The ^S\\d+$ regex isolates
        the integer-suffix variant — SA1 doesn't match, so S1 is still free."""
        from db_layer import generate_unique_trx_id
        _insert_detail("SELL", "SA1")
        _insert_detail("SELL", "SB1")
        assert generate_unique_trx_id(TEST_PORTFOLIO, TEST_TRADE_ID, "S") == "S1"
        # And asking for SA-prefix returns SA2 (skipping the existing SA1).
        assert generate_unique_trx_id(TEST_PORTFOLIO, TEST_TRADE_ID, "SA") == "SA2"


@pytestmark_db
@pytest.mark.skipif(
    "CI" in os.environ,
    reason="threading + Postgres timing can be flaky in CI",
)
class TestConcurrency:
    """The helper's lock is a contention reducer, not a correctness guarantee.

    Strict per-call uniqueness is the UNIQUE (portfolio_id, trade_id, trx_id)
    constraint's job (migration 018). These tests prove what the helper
    actually does on its own: complete without deadlock or error under
    concurrent load. We do NOT assert per-call uniqueness here because
    the lock auto-releases at the helper's commit BEFORE the caller's
    INSERT — concurrent callers can legitimately receive the same value
    in the small window between this function returning and either
    caller's INSERT landing. See generate_unique_trx_id docstring.
    """

    def test_5_concurrent_helpers_complete_without_error(self, clean_test_trade) -> None:
        from db_layer import generate_unique_trx_id

        results: list[str] = []
        errors: list[Exception] = []
        results_lock = threading.Lock()

        def worker() -> None:
            try:
                trx = generate_unique_trx_id(TEST_PORTFOLIO, TEST_TRADE_ID, "S")
                with results_lock:
                    results.append(trx)
            except Exception as exc:
                with results_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)  # generous — should finish in <1s

        assert not errors, f"Worker errors: {errors}"
        assert len(results) == 5
        # All results are valid trx_ids of the form S<digits>
        assert all(r.startswith("S") and r[1:].isdigit() for r in results), \
            f"Got non-conforming trx_ids: {results}"
