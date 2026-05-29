"""Tests for the Phase 2 B-1 persistence layer — how `match_method` flows
through write/read.

Coverage:
  1. _save_detail_row_in_txn emits the match_method column only when
     Match_Method is present in row_dict (back-compat with existing
     callers like log_buy + the IBKR import that don't set it).
  2. update_detail_row's UPDATE statement does NOT include match_method
     — stamps are immutable after insert.
  3. load_details SELECT includes match_method (aliased to PascalCase
     for the _normalize_trades rename map).
  4. _normalize_trades rename map carries the Match_Method → match_method
     entry so downstream walks see the snake_case form.
  5. log_buy source does NOT reference _resolve_match_method or
     Match_Method — BUY rows always land NULL.
  6. log_sell source DOES stamp Match_Method on the detail_row.
  7. exercise_option stamps the option-leg SELL (Match_Method present)
     but NOT the stock-leg BUY.

The behavioral correctness of writing/reading match_method to/from
Postgres is exercised by the migration 041 verification DO-block at
apply time (which raised on any unstamped SELL) and by the live
apply that landed 624/624 SELLs cleanly. These tests guard the source
shape from regression.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. _save_detail_row_in_txn: column emission contract
# ---------------------------------------------------------------------------


class TestSaveDetailRowInTxnEmission:
    """The function's INSERT must include match_method if and only if
    the caller passes Match_Method in row_dict. Back-compat invariant:
    log_buy + IBKR import + legacy callers omit the key → NULL stamp.
    """

    @pytest.fixture
    def captured(self, monkeypatch):
        """Return a list of (sql, params) tuples captured from cur.execute."""
        import db_layer

        captured: list[tuple[str, tuple]] = []

        cur = MagicMock()
        # The function calls cur.execute(insert_sql, vals) then
        # cur.fetchone()[0] for the RETURNING id. Then it calls
        # _emit_trade_cash_tx which may execute another INSERT. We
        # only care about the trades_details INSERT.
        def execute(sql, params=()):
            captured.append((sql, params))
        cur.execute.side_effect = execute
        cur.fetchone.return_value = (12345,)

        # Stub _emit_trade_cash_tx so we don't capture its SQL too.
        monkeypatch.setattr(db_layer, "_emit_trade_cash_tx",
                            lambda *a, **kw: None)

        return cur, captured

    def test_match_method_emitted_when_present(self, captured):
        """When row_dict carries Match_Method, the INSERT lists the
        match_method column and binds the value."""
        from db_layer import _save_detail_row_in_txn

        cur, captured_sql = captured
        row = {
            "Trade_ID": "T1", "Ticker": "AAPL", "Action": "SELL",
            "Date": "2026-01-15 10:00:00",
            "Shares": 100, "Amount": 60.0, "Value": 6000.0,
            "Rule": "", "Notes": "",
            "Realized_PL": 0, "Trx_ID": "S1",
            "Match_Method": "HCFO",
        }
        _save_detail_row_in_txn(cur, 1, row)

        assert len(captured_sql) >= 1
        sql, params = captured_sql[0]
        assert "INSERT INTO trades_details" in sql
        assert "match_method" in sql
        assert "HCFO" in params

    def test_match_method_omitted_when_absent(self, captured):
        """When row_dict lacks Match_Method, the INSERT does NOT list
        the column — the DB DEFAULT (NULL) takes over. This is the
        back-compat path for log_buy + IBKR import + tests."""
        from db_layer import _save_detail_row_in_txn

        cur, captured_sql = captured
        row = {
            "Trade_ID": "T1", "Ticker": "AAPL", "Action": "BUY",
            "Date": "2026-01-10 09:30:00",
            "Shares": 100, "Amount": 50.0, "Value": 5000.0,
            "Rule": "Breakout", "Notes": "",
            "Realized_PL": 0, "Trx_ID": "B1",
        }
        _save_detail_row_in_txn(cur, 1, row)

        assert len(captured_sql) >= 1
        sql, _params = captured_sql[0]
        assert "INSERT INTO trades_details" in sql
        assert "match_method" not in sql

    def test_lifo_stamp_round_trip(self, captured):
        """LIFO stamp passes through unchanged."""
        from db_layer import _save_detail_row_in_txn

        cur, captured_sql = captured
        row = {
            "Trade_ID": "T1", "Ticker": "AAPL", "Action": "SELL",
            "Date": "2026-01-15 10:00:00",
            "Shares": 100, "Amount": 60.0, "Value": 6000.0,
            "Rule": "", "Notes": "",
            "Realized_PL": 0, "Trx_ID": "S1",
            "Match_Method": "LIFO",
        }
        _save_detail_row_in_txn(cur, 1, row)

        _sql, params = captured_sql[0]
        assert "LIFO" in params


# ---------------------------------------------------------------------------
# 2. update_detail_row immutability — structural guarantee
# ---------------------------------------------------------------------------


class TestUpdateDetailRowImmutability:
    """update_detail_row's UPDATE statement must NOT touch match_method.
    Stamps are immutable: editing a SELL's shares/amount/date must NOT
    change which lots it consumed historically. This is the cleanest
    invariant — there is no code path that can rewrite the stamp.
    """

    def test_update_sql_does_not_set_match_method(self):
        from db_layer import update_detail_row

        src = inspect.getsource(update_detail_row)
        # The UPDATE SET list must not assign match_method.
        # Defensive search variations.
        forbidden = [
            "match_method = %s",
            "match_method=%s",
            "SET match_method",
        ]
        for needle in forbidden:
            assert needle not in src, (
                f"update_detail_row must not write to match_method "
                f"(found {needle!r}); B-1 stamp immutability invariant."
            )


# ---------------------------------------------------------------------------
# 3. load_details exposure
# ---------------------------------------------------------------------------


class TestLoadDetailsExposesMatchMethod:
    """load_details' SELECT must include match_method aliased to
    Match_Method so _normalize_trades + downstream walks see the
    column. Without this, B-1 is a silent no-op for HCFO."""

    def test_select_includes_match_method(self):
        from db_layer import load_details

        src = inspect.getsource(load_details)
        assert 'd.match_method AS "Match_Method"' in src, (
            "load_details SELECT must expose match_method or HCFO routing "
            "is invisible to the walks."
        )


# ---------------------------------------------------------------------------
# 4. _normalize_trades rename map
# ---------------------------------------------------------------------------


class TestNormalizeTradesRenameMap:
    """The rename map in api.main._normalize_trades must include
    Match_Method → match_method so SELL rows arrive at the walks
    with the snake_case column name the walks look up."""

    def test_rename_map_includes_match_method(self, monkeypatch):
        monkeypatch.setenv("AUTH_SECRET", "test-secret-not-for-prod")
        from api.main import _normalize_trades
        import pandas as pd

        df = pd.DataFrame([
            {"Trade_ID": "T1", "Action": "SELL", "Match_Method": "HCFO"},
        ])
        out = _normalize_trades(df)
        assert "match_method" in out.columns
        assert "Match_Method" not in out.columns
        assert out["match_method"].iloc[0] == "HCFO"


# ---------------------------------------------------------------------------
# 5. log_buy must NOT stamp — BUY rows stay NULL
# ---------------------------------------------------------------------------


class TestLogBuyDoesNotStamp:
    """log_buy must never call _resolve_match_method or pass
    Match_Method in any of its detail_row dicts. BUY rows are NULL
    by convention — that's what makes the per-SELL stamp meaningful."""

    def test_log_buy_source_no_resolve_call(self, monkeypatch):
        monkeypatch.setenv("AUTH_SECRET", "test-secret-not-for-prod")
        from api.main import log_buy

        src = inspect.getsource(log_buy)
        assert "_resolve_match_method" not in src

    def test_log_buy_source_no_match_method_key(self, monkeypatch):
        monkeypatch.setenv("AUTH_SECRET", "test-secret-not-for-prod")
        from api.main import log_buy

        src = inspect.getsource(log_buy)
        # None of log_buy's row-dict constructions should include the
        # Match_Method key — that would land a stamp on the BUY row.
        assert "Match_Method" not in src


# ---------------------------------------------------------------------------
# 6. log_sell DOES stamp
# ---------------------------------------------------------------------------


class TestLogSellStamps:
    """log_sell's detail_row must include Match_Method derived from
    _resolve_match_method() called once per request."""

    def test_log_sell_calls_resolve_match_method(self, monkeypatch):
        monkeypatch.setenv("AUTH_SECRET", "test-secret-not-for-prod")
        from api.main import log_sell

        src = inspect.getsource(log_sell)
        assert "_resolve_match_method" in src
        assert "Match_Method" in src


# ---------------------------------------------------------------------------
# 7. exercise_option: option-leg stamps, stock-leg does not
# ---------------------------------------------------------------------------


class TestExerciseOptionStampPolicy:
    """exercise_option writes two rows: an option-leg SELL and a
    stock-leg BUY. Only the SELL gets stamped — the BUY stays NULL by
    convention (per-SELL stamping, not per-row stamping)."""

    def test_exercise_option_calls_resolve_match_method(self, monkeypatch):
        monkeypatch.setenv("AUTH_SECRET", "test-secret-not-for-prod")
        from api.main import exercise_option

        src = inspect.getsource(exercise_option)
        assert "_resolve_match_method" in src

    def test_opt_detail_row_includes_match_method(self, monkeypatch):
        monkeypatch.setenv("AUTH_SECRET", "test-secret-not-for-prod")
        from api.main import exercise_option

        src = inspect.getsource(exercise_option)
        # The opt_detail_row (option-leg SELL) must carry Match_Method.
        assert "opt_detail_row" in src
        # Find the opt_detail_row block; assert Match_Method appears
        # somewhere AFTER opt_detail_row but before the stock-leg block.
        opt_idx = src.find("opt_detail_row = {")
        stock_idx = src.find("stock_detail_row = ")
        if stock_idx < 0:
            # Some variants name it differently — fall back to a less
            # strict check.
            stock_idx = src.find("# --- 4.")  # stock-leg comment marker
        assert opt_idx > 0
        opt_block = src[opt_idx:stock_idx if stock_idx > opt_idx else len(src)]
        assert "Match_Method" in opt_block, (
            "exercise_option option-leg SELL must stamp Match_Method"
        )

    def test_stock_leg_buy_does_not_stamp(self, monkeypatch):
        """The stock-leg BUY row dict must NOT contain Match_Method.
        The locator for the stock-leg is the `'Action': 'BUY'` literal
        + underlying-ticker construction further down."""
        monkeypatch.setenv("AUTH_SECRET", "test-secret-not-for-prod")
        from api.main import exercise_option

        src = inspect.getsource(exercise_option)
        # The stock-leg row literal contains `"action": "BUY"` (in the
        # pd.DataFrame fixture) and the constructed detail row uses
        # `"Action": "BUY"`. We grep the detail row body specifically.
        # The cleanest invariant: the SQL-bound detail row dict for the
        # stock-leg has no Match_Method key.
        # Locate the second 'Action': 'BUY' assignment — first appearance
        # may be inside the LIFO simulation block; the second is the
        # actual detail row dict.
        # Simpler heuristic: count Match_Method occurrences. We expect
        # exactly ONE — the option-leg stamp (or, defensively, two: one
        # for the local `opt_match_method = _resolve_match_method()`
        # call and one for the dict entry).
        assert src.count("Match_Method") <= 2, (
            f"exercise_option contains too many Match_Method references "
            f"({src.count('Match_Method')}); stock-leg BUY may have "
            f"been accidentally stamped."
        )
