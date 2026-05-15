"""Tests for db_layer.load_lot_closures.

DB-backed tests connect to whatever DATABASE_URL points at and isolate
themselves with a sentinel trade_id ('TEST-LOT-CLOSURES-001') so they
cannot collide with production trades (which follow YYYYMM-NNN). The
fixture hard-deletes the sentinel summary + details + closures on
setup AND teardown, so partial failures cannot leak state.

If DATABASE_URL is not set, the DB-dependent tests skip cleanly.
"""
from __future__ import annotations

import os
from datetime import datetime

import pytest


pytestmark_db = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping DB-dependent tests",
)


# Sentinels — TEST_TRADE_ID is unique enough that it cannot collide with
# production trade_ids (which follow the YYYYMM-NNN format).
TEST_PORTFOLIO = "CanSlim"
TEST_TRADE_ID_A = "TEST-LOT-CLOSURES-001"
TEST_TRADE_ID_B = "TEST-LOT-CLOSURES-002"
TEST_TICKER_A = "TESTLCA"
TEST_TICKER_B = "TESTLCB"
FOUNDER_UUID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"


def _set_user_context() -> None:
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
    """Hard-delete (NOT soft-delete) every row tied to the sentinel trades."""
    from db_layer import get_db_connection
    portfolio_id = _get_portfolio_id()
    if portfolio_id is None:
        return
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for tid in (TEST_TRADE_ID_A, TEST_TRADE_ID_B):
                cur.execute(
                    "DELETE FROM lot_closures WHERE portfolio_id = %s AND trade_id = %s",
                    (portfolio_id, tid),
                )
                cur.execute(
                    "DELETE FROM trades_details WHERE portfolio_id = %s AND trade_id = %s",
                    (portfolio_id, tid),
                )
                cur.execute(
                    "DELETE FROM trades_summary WHERE portfolio_id = %s AND trade_id = %s",
                    (portfolio_id, tid),
                )
        conn.commit()


def _seed_summary(trade_id: str, ticker: str) -> None:
    """trades_details has an FK to trades_summary on (portfolio_id, trade_id),
    so we need a summary row before we can insert details. Closures don't
    have an FK to summary, but we keep the seed to mirror real-world state."""
    from db_layer import save_summary_row
    summary_row = {
        "Trade_ID": trade_id, "Ticker": ticker, "Status": "OPEN",
        "Open_Date": datetime.now().strftime("%Y-%m-%d"),
        "Shares": 0, "Avg_Entry": 0, "Total_Cost": 0,
    }
    save_summary_row(TEST_PORTFOLIO, summary_row)


def _insert_closure(
    trade_id: str,
    sell_trx_id: str,
    buy_trx_id: str,
    *,
    shares: float = 100.0,
    buy_price: float = 50.0,
    sell_price: float = 60.0,
    multiplier: float = 1.0,
    realized_pl: float | None = None,
) -> None:
    """Insert one lot_closures row for a sentinel trade."""
    from db_layer import get_db_connection
    portfolio_id = _get_portfolio_id()
    if portfolio_id is None:
        raise RuntimeError(f"Portfolio '{TEST_PORTFOLIO}' not found")
    if realized_pl is None:
        realized_pl = (sell_price - buy_price) * shares * multiplier
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lot_closures (
                    portfolio_id, trade_id, sell_trx_id, buy_trx_id,
                    shares, buy_price, sell_price, multiplier,
                    realized_pl, closed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """,
                (portfolio_id, trade_id, sell_trx_id, buy_trx_id,
                 shares, buy_price, sell_price, multiplier, realized_pl),
            )
        conn.commit()


@pytest.fixture
def clean_test_trades():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")
    _set_user_context()
    _delete_test_rows()
    _seed_summary(TEST_TRADE_ID_A, TEST_TICKER_A)
    _seed_summary(TEST_TRADE_ID_B, TEST_TICKER_B)
    yield (TEST_TRADE_ID_A, TEST_TRADE_ID_B)
    _delete_test_rows()


@pytestmark_db
class TestLoadLotClosures:

    def test_empty_when_no_closures(self, clean_test_trades) -> None:
        """Trades exist but no closures → empty DataFrame, correct columns."""
        from db_layer import load_lot_closures
        df = load_lot_closures(TEST_PORTFOLIO, trade_id=TEST_TRADE_ID_A)
        assert df.empty
        # Even when empty, downstream callers iterate columns — make sure the
        # frame is well-shaped so they don't KeyError. (The empty-list-filter
        # branch returns this shape; the SQL-result branch returns whatever
        # cur.description gave. Either way these columns exist.)
        for col in ("trade_id", "buy_trx_id", "sell_trx_id", "shares",
                    "buy_price", "sell_price", "multiplier", "realized_pl",
                    "closed_at"):
            assert col in df.columns, f"missing column: {col}"

    def test_single_row_returned_with_correct_types(self, clean_test_trades) -> None:
        """One closure row → returned with numeric values coerced from Decimal."""
        from db_layer import load_lot_closures
        _insert_closure(
            TEST_TRADE_ID_A, sell_trx_id="S1", buy_trx_id="B1",
            shares=100.0, buy_price=50.0, sell_price=60.0,
        )
        df = load_lot_closures(TEST_PORTFOLIO, trade_id=TEST_TRADE_ID_A)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["trade_id"] == TEST_TRADE_ID_A
        assert row["sell_trx_id"] == "S1"
        assert row["buy_trx_id"] == "B1"
        # Numeric coercion: Decimal -> float so JSON serialization downstream
        # produces numbers, not strings.
        assert isinstance(float(row["shares"]), float)
        assert float(row["shares"]) == 100.0
        assert float(row["buy_price"]) == 50.0
        assert float(row["sell_price"]) == 60.0
        assert float(row["realized_pl"]) == 1000.0  # (60-50) * 100 * 1
        assert float(row["multiplier"]) == 1.0

    def test_trade_id_filter_isolates_one_trade(self, clean_test_trades) -> None:
        """trade_id filter returns only that trade's closures."""
        from db_layer import load_lot_closures
        _insert_closure(TEST_TRADE_ID_A, "S1", "B1")
        _insert_closure(TEST_TRADE_ID_B, "S1", "B1")
        df = load_lot_closures(TEST_PORTFOLIO, trade_id=TEST_TRADE_ID_A)
        assert len(df) == 1
        assert df.iloc[0]["trade_id"] == TEST_TRADE_ID_A

    def test_trade_ids_list_filter(self, clean_test_trades) -> None:
        """trade_ids list filter returns rows for the given subset only."""
        from db_layer import load_lot_closures
        _insert_closure(TEST_TRADE_ID_A, "S1", "B1")
        _insert_closure(TEST_TRADE_ID_A, "S2", "B1")
        _insert_closure(TEST_TRADE_ID_B, "S1", "B1")
        df = load_lot_closures(TEST_PORTFOLIO, trade_ids=[TEST_TRADE_ID_A])
        assert len(df) == 2
        assert set(df["trade_id"].unique()) == {TEST_TRADE_ID_A}

    def test_trade_ids_with_both_returns_all(self, clean_test_trades) -> None:
        """trade_ids = [A, B] returns rows for both."""
        from db_layer import load_lot_closures
        _insert_closure(TEST_TRADE_ID_A, "S1", "B1")
        _insert_closure(TEST_TRADE_ID_B, "S1", "B1")
        df = load_lot_closures(TEST_PORTFOLIO, trade_ids=[TEST_TRADE_ID_A, TEST_TRADE_ID_B])
        assert len(df) == 2
        assert set(df["trade_id"].unique()) == {TEST_TRADE_ID_A, TEST_TRADE_ID_B}

    def test_empty_trade_ids_list_returns_empty(self, clean_test_trades) -> None:
        """Empty list filter short-circuits — no SQL, empty frame with columns."""
        from db_layer import load_lot_closures
        _insert_closure(TEST_TRADE_ID_A, "S1", "B1")
        df = load_lot_closures(TEST_PORTFOLIO, trade_ids=[])
        assert df.empty
        # Columns still present so callers can iterate without KeyError.
        assert "trade_id" in df.columns
        assert "realized_pl" in df.columns

    def test_no_filter_returns_all_portfolio_closures(self, clean_test_trades) -> None:
        """No trade_id and no trade_ids → all closures for the portfolio.
        We assert the sentinel rows are PRESENT (not the exact length) because
        the test DB may have other trades' closures."""
        from db_layer import load_lot_closures
        _insert_closure(TEST_TRADE_ID_A, "S1", "B1")
        _insert_closure(TEST_TRADE_ID_B, "S1", "B1")
        df = load_lot_closures(TEST_PORTFOLIO)
        sentinel_rows = df[df["trade_id"].isin([TEST_TRADE_ID_A, TEST_TRADE_ID_B])]
        assert len(sentinel_rows) == 2

    def test_trade_id_takes_precedence_over_trade_ids(self, clean_test_trades) -> None:
        """When both trade_id and trade_ids are passed, trade_id wins.
        The helper's `if trade_id is not None: ... elif trade_ids is not None`
        branching enforces this; this test pins it down."""
        from db_layer import load_lot_closures
        _insert_closure(TEST_TRADE_ID_A, "S1", "B1")
        _insert_closure(TEST_TRADE_ID_B, "S1", "B1")
        df = load_lot_closures(
            TEST_PORTFOLIO,
            trade_id=TEST_TRADE_ID_A,
            trade_ids=[TEST_TRADE_ID_A, TEST_TRADE_ID_B],
        )
        assert len(df) == 1
        assert df.iloc[0]["trade_id"] == TEST_TRADE_ID_A

    def test_multiple_closures_for_one_trade_ordered_by_closed_at(
        self, clean_test_trades,
    ) -> None:
        """Multiple closures for one trade returned in (trade_id, closed_at, id) order."""
        from db_layer import load_lot_closures
        # Inserted in S2-first order; ORDER BY closed_at + id should still
        # return them in insertion order since closed_at = NOW() for each
        # is monotonically nondecreasing within this test.
        _insert_closure(TEST_TRADE_ID_A, "S1", "B1")
        _insert_closure(TEST_TRADE_ID_A, "S2", "B1")
        _insert_closure(TEST_TRADE_ID_A, "S3", "B2")
        df = load_lot_closures(TEST_PORTFOLIO, trade_id=TEST_TRADE_ID_A)
        assert len(df) == 3
        assert list(df["sell_trx_id"]) == ["S1", "S2", "S3"]

    def test_closed_at_is_datetime(self, clean_test_trades) -> None:
        """closed_at comes back as a pandas Timestamp / datetime, not a string."""
        from db_layer import load_lot_closures
        _insert_closure(TEST_TRADE_ID_A, "S1", "B1")
        df = load_lot_closures(TEST_PORTFOLIO, trade_id=TEST_TRADE_ID_A)
        # The exact type depends on psycopg2 / pandas conversion, but it
        # should be something datetime-y (not a raw string) so downstream
        # code (e.g. _df_to_records) can format it as ISO via .strftime.
        val = df.iloc[0]["closed_at"]
        assert hasattr(val, "strftime"), f"closed_at not datetime-like: {type(val)}"

    def test_unknown_portfolio_raises(self, clean_test_trades) -> None:
        from db_layer import load_lot_closures
        with pytest.raises(
            ValueError,
            match="Portfolio '_DOES_NOT_EXIST_TEST_PORTFOLIO_' not found",
        ):
            load_lot_closures("_DOES_NOT_EXIST_TEST_PORTFOLIO_")


@pytestmark_db
class TestDeleteTradeCascadesToLotClosures:
    """Regression: delete_trade must hard-delete lot_closures for the trade.
    Pre-fix, soft-deleting a trade left orphan lot_closures rows pointing at
    a soft-deleted parent. The fix folds the DELETE into delete_trade's
    transaction; this test would have caught the original miss."""

    def test_delete_trade_cascades_to_lot_closures(self, clean_test_trades) -> None:
        _insert_closure(TEST_TRADE_ID_A, "S1", "B1")
        _insert_closure(TEST_TRADE_ID_A, "S1", "B2")

        from db_layer import delete_trade, load_lot_closures
        delete_trade(TEST_PORTFOLIO, TEST_TRADE_ID_A)

        df = load_lot_closures(TEST_PORTFOLIO, trade_id=TEST_TRADE_ID_A)
        assert df.empty, "lot_closures must be hard-deleted by delete_trade"
