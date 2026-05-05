"""Tests for trade_id generation safety against soft-deleted summaries.

The 5/4/26 incident: user logged two new "Start New Campaign" trades and
both got assigned '202605-004' — a trade_id whose summary was soft-deleted
the previous day. The first new trade attached to the tombstoned summary
(invisible in the UI); the second collided on top of the first.

Two bugs combined:
  1. next_trade_id (api/main.py) computed max(seq)+1 over load_summary,
     which filters deleted_at IS NULL — so soft-deleted ids vanished from
     the calculation and got recycled.
  2. save_summary_row's existence check (db_layer.py) didn't filter
     deleted_at IS NULL — so the recycled id silently UPDATE-ed the
     tombstoned row instead of attempting a fresh INSERT.

These tests pin down both fixes. The schema's unique_trade_per_portfolio
constraint is non-partial, so post-fix INSERT against a tombstoned id
surfaces as IntegrityError — the loud-failure guarantee that "deleted
trade_ids are never reused."

Sentinels live in a future month (209912-NNN) so they cannot collide
with production trade_ids. If DATABASE_URL is not set, all tests skip.
"""
from __future__ import annotations

import os

import psycopg2
import pytest


pytestmark_db = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping DB-dependent tests",
)


TEST_PORTFOLIO = "CanSlim"  # Real portfolio (must exist); sentinel by trade_id prefix.
TEST_YM = "209912"  # Year 2099 / month 12 — far enough out to never collide.
TEST_DATE = "2099-12-15"  # Any date inside TEST_YM works.
TEST_TICKER = "TESTTID"
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
    """Hard-delete (NOT soft-delete) every row whose trade_id starts with
    TEST_YM, so partial failures cannot leak state across tests."""
    from db_layer import get_db_connection
    portfolio_id = _get_portfolio_id()
    if portfolio_id is None:
        return
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM lot_closures WHERE portfolio_id = %s AND trade_id LIKE %s",
                (portfolio_id, f"{TEST_YM}-%"),
            )
            cur.execute(
                "DELETE FROM trades_details WHERE portfolio_id = %s AND trade_id LIKE %s",
                (portfolio_id, f"{TEST_YM}-%"),
            )
            cur.execute(
                "DELETE FROM trades_summary WHERE portfolio_id = %s AND trade_id LIKE %s",
                (portfolio_id, f"{TEST_YM}-%"),
            )
        conn.commit()


def _seed_summary(seq: int, *, soft_deleted: bool = False) -> str:
    """Insert a summary at TEST_YM-<seq:03d> and optionally soft-delete it.
    Returns the full trade_id string."""
    from db_layer import save_summary_row, delete_trade
    trade_id = f"{TEST_YM}-{seq:03d}"
    save_summary_row(TEST_PORTFOLIO, {
        "Trade_ID": trade_id, "Ticker": TEST_TICKER, "Status": "OPEN",
        "Open_Date": TEST_DATE, "Shares": 1, "Avg_Entry": 1, "Total_Cost": 1,
    })
    if soft_deleted:
        delete_trade(TEST_PORTFOLIO, trade_id)
    return trade_id


def _summary_counts(trade_id: str) -> tuple[int, int]:
    """Returns (active_count, deleted_count) for the given trade_id."""
    from db_layer import get_db_connection
    portfolio_id = _get_portfolio_id()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT "
                "  COUNT(*) FILTER (WHERE deleted_at IS NULL), "
                "  COUNT(*) FILTER (WHERE deleted_at IS NOT NULL) "
                "FROM trades_summary WHERE portfolio_id = %s AND trade_id = %s",
                (portfolio_id, trade_id),
            )
            return cur.fetchone()


def _call_next_trade_id() -> str:
    """Invoke the FastAPI handler in-process (no HTTP). Lazy-imports so dry
    test runs don't pull FastAPI."""
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "api"))
    from main import next_trade_id  # type: ignore
    result = next_trade_id(portfolio=TEST_PORTFOLIO, date=TEST_DATE)
    assert "trade_id" in result, f"next_trade_id returned error: {result}"
    return result["trade_id"]


@pytest.fixture
def clean_test_month():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")
    _set_user_context()
    _delete_test_rows()
    yield
    _delete_test_rows()


@pytestmark_db
class TestNextTradeId:
    """next_trade_id must count soft-deleted summaries when computing
    max(seq)+1 so a deleted trade_id is never recycled to a new campaign."""

    def test_next_trade_id_skips_soft_deleted_max(self, clean_test_month) -> None:
        """The reported incident shape: 001..004 active, then 004 gets
        soft-deleted. The next id must be 005 — recycling 004 is what
        caused the silent overwrite."""
        _seed_summary(1)
        _seed_summary(2)
        _seed_summary(3)
        _seed_summary(4, soft_deleted=True)
        assert _call_next_trade_id() == f"{TEST_YM}-005"

    def test_next_trade_id_with_only_deleted_summaries(self, clean_test_month) -> None:
        """All summaries for the month are soft-deleted. The generator must
        still skip every used id — first new campaign of the month gets
        max(seq)+1 relative to the deleted rows, not 001."""
        _seed_summary(1, soft_deleted=True)
        _seed_summary(2, soft_deleted=True)
        assert _call_next_trade_id() == f"{TEST_YM}-003"

    def test_next_trade_id_gaps_from_deletes(self, clean_test_month) -> None:
        """001 active, 002 deleted, 003 active. Generator returns 004
        (max+1), not 002 (gap-fill). Gaps are expected and reserved —
        the missing number is recoverable from the soft-deleted row."""
        _seed_summary(1)
        _seed_summary(2, soft_deleted=True)
        _seed_summary(3)
        assert _call_next_trade_id() == f"{TEST_YM}-004"


@pytestmark_db
class TestSaveSummaryRowSoftDelete:
    """save_summary_row's existence check must filter deleted_at IS NULL so
    a recycled id triggers INSERT (not silent UPDATE of a tombstoned row).
    The schema's unique_trade_per_portfolio constraint then converts the
    duplicate into an IntegrityError — the loud-failure guarantee."""

    def test_save_summary_row_with_soft_deleted_id_does_not_overwrite(
        self, clean_test_month,
    ) -> None:
        from db_layer import save_summary_row

        # Setup: tombstoned summary at 209912-001 with original data.
        trade_id = _seed_summary(1, soft_deleted=True)
        active, deleted = _summary_counts(trade_id)
        assert (active, deleted) == (0, 1), \
            f"Setup expected 0 active + 1 deleted, got {active} + {deleted}"

        # Act: simulate the bug repro — a "new campaign" save lands with
        # the tombstoned id. Post-fix the existence check skips the
        # tombstone, INSERT runs, and unique_trade_per_portfolio raises.
        with pytest.raises((psycopg2.errors.UniqueViolation, psycopg2.IntegrityError)):
            save_summary_row(TEST_PORTFOLIO, {
                "Trade_ID": trade_id, "Ticker": TEST_TICKER, "Status": "OPEN",
                "Open_Date": TEST_DATE, "Shares": 99, "Avg_Entry": 99,
                "Total_Cost": 9999,
            })

        # Assert: tombstoned row is untouched. Pre-fix this would have
        # been UPDATE-ed in place (Shares=99) without clearing deleted_at,
        # leaving an invisible-but-active campaign — exactly the silent
        # data-loss mode of the original incident.
        from db_layer import get_db_connection
        portfolio_id = _get_portfolio_id()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT shares, deleted_at FROM trades_summary "
                    "WHERE portfolio_id = %s AND trade_id = %s",
                    (portfolio_id, trade_id),
                )
                rows = cur.fetchall()
        assert len(rows) == 1, f"Expected 1 row after failed INSERT, got {len(rows)}"
        shares, deleted_at = rows[0]
        assert int(shares) == 1, \
            f"Tombstone should retain original Shares=1, got {shares}"
        assert deleted_at is not None, \
            "Tombstone should still be soft-deleted"
