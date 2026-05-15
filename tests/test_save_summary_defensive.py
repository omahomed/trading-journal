"""Defense-in-depth tests for the trades_summary writer boundary (Phase 2, Commit 5).

Phase 1 sweep audit identified _save_summary_with_closures_in_txn as ⚠️ Partial:
the function accepts whatever caller passes in summary_row and binds it
directly. Commit 1 fixed the known producers (log_buy scale-in, log_sell,
exercise_option scale-in stock side). But the writer itself was still
trusting — a regression in any caller, or a new caller, would re-introduce
string-NaN ('nan'/'none'/'null' literals) without the writer noticing.

This commit adds defense-in-depth: the writer sanitizes user-prose text
columns at function entry via the canonical clean_text_value (Commit 1).
The 5 columns covered: Rule, Buy_Notes, Sell_Rule, Sell_Notes, Notes.
Non-text columns (Status, Instrument_Type, NUMERIC fields) are left alone.

Sibling fix: save_summary_row's nested clean_value helper had the same
blind spot — it stripped pandas/numpy NaN but not 'nan'/'none'/'null'
strings. Extended with a parallel sentinel-strip arm (3 lines).

Tests are direct unit tests against a fake cursor — bypasses FastAPI to
isolate the writer's bind behavior.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import db_layer


# ---------------------------------------------------------------------------
# Fake cursor scaffolding (mirrors test_edit_mirror.py helper-unit pattern)
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Records every execute() call. fetchone() returns from a queue."""
    def __init__(self, fetchones):
        self._fetchones = list(fetchones)  # consumed FIFO
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, tuple(params) if params else ()))

    def executemany(self, sql, seq_of_params):
        self.executed.append((sql, tuple(seq_of_params)))

    def fetchone(self):
        if self._fetchones:
            return self._fetchones.pop(0)
        return None

    def close(self):
        pass


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
    def cursor(self, *a, **kw):
        return _CursorCM(self._cursor)
    def commit(self):
        self.commits += 1
    def rollback(self):
        pass
    def close(self):
        pass


class _CursorCM:
    def __init__(self, cur):
        self._cur = cur
    def __enter__(self):
        return self._cur
    def __exit__(self, *a):
        return False


class _ConnCM:
    def __init__(self, conn):
        self._conn = conn
    def __enter__(self):
        return self._conn
    def __exit__(self, *a):
        return False


def _well_formed_summary_row(**overrides):
    """A clean summary_row dict — every key set to a legitimate value.
    Tests override individual keys to exercise the sanitizer."""
    row = {
        "Trade_ID": "202605-001",
        "Ticker": "AAPL",
        "Status": "OPEN",
        "Open_Date": "2026-05-01",
        "Closed_Date": None,
        "Shares": 100.0,
        "Avg_Entry": 200.0,
        "Avg_Exit": 0.0,
        "Total_Cost": 20000.0,
        "Realized_PL": 0.0,
        "Unrealized_PL": 0.0,
        "Return_Pct": 0.0,
        "Sell_Rule": None,
        "Notes": None,
        "Stop_Loss": 195.0,
        "Rule": "br1.3 Cup w/o Handle",
        "Buy_Notes": "Initial entry on breakout",
        "Sell_Notes": None,
        "Risk_Budget": 500.0,
        "Instrument_Type": "STOCK",
        "Multiplier": 1.0,
    }
    row.update(overrides)
    return row


def _make_cursor_for_insert(returning_id=42):
    """Cursor where fetchone() returns:
       1st call: None (no existing row → INSERT path)
       2nd call: (returning_id,) (RETURNING id from INSERT)
    """
    return _FakeCursor([None, (returning_id,)])


def _make_cursor_for_update(existing_id=7, returning_id=7):
    """Cursor where fetchone() returns:
       1st call: (existing_id,) (existing row found → UPDATE path)
       2nd call: (returning_id,) (RETURNING id from UPDATE)
    """
    return _FakeCursor([(existing_id,), (returning_id,)])


def _filter_summary_writes(executed):
    """Return only the trades_summary UPDATE/INSERT statement(s)."""
    return [(sql, params) for sql, params in executed
            if "trades_summary" in sql and ("UPDATE" in sql or "INSERT" in sql)]


# Column-position constants for the UPSERT params tuple. UPDATE has 20
# binds + existing[0]; INSERT prepends portfolio_id + trade_id (offset +2).
_UPDATE_RULE_IDX       = 14
_UPDATE_BUY_NOTES_IDX  = 15
_UPDATE_SELL_RULE_IDX  = 11
_UPDATE_SELL_NOTES_IDX = 16
_UPDATE_NOTES_IDX      = 12

_INSERT_OFFSET         = 2  # portfolio_id, trade_id at positions 0,1
_INSERT_RULE_IDX       = _UPDATE_RULE_IDX + _INSERT_OFFSET
_INSERT_BUY_NOTES_IDX  = _UPDATE_BUY_NOTES_IDX + _INSERT_OFFSET
_INSERT_SELL_RULE_IDX  = _UPDATE_SELL_RULE_IDX + _INSERT_OFFSET
_INSERT_SELL_NOTES_IDX = _UPDATE_SELL_NOTES_IDX + _INSERT_OFFSET
_INSERT_NOTES_IDX      = _UPDATE_NOTES_IDX + _INSERT_OFFSET


# ---------------------------------------------------------------------------
# 9 tests against _save_summary_with_closures_in_txn
# ---------------------------------------------------------------------------


def test_writer_neutralizes_string_nan_in_rule():
    """Pass Rule='nan' (the literal string from str(np.nan)). Writer must
    bind None, not 'nan'."""
    cur = _make_cursor_for_insert()
    row = _well_formed_summary_row(Rule="nan")
    db_layer._save_summary_with_closures_in_txn(cur, 1, "202605-001", row, [])

    writes = _filter_summary_writes(cur.executed)
    assert len(writes) == 1
    sql, params = writes[0]
    assert "INSERT" in sql
    assert params[_INSERT_RULE_IDX] is None, \
        f"Rule should bind None, got {params[_INSERT_RULE_IDX]!r}"


def test_writer_neutralizes_np_nan_in_rule():
    """Pass Rule=np.nan directly (e.g., from a pandas Series). Writer must
    bind None."""
    cur = _make_cursor_for_insert()
    row = _well_formed_summary_row(Rule=np.nan)
    db_layer._save_summary_with_closures_in_txn(cur, 1, "202605-001", row, [])

    sql, params = _filter_summary_writes(cur.executed)[0]
    assert params[_INSERT_RULE_IDX] is None


def test_writer_neutralizes_string_none():
    """Pass Rule='None' (case-insensitive sentinel)."""
    cur = _make_cursor_for_insert()
    row = _well_formed_summary_row(Rule="None")
    db_layer._save_summary_with_closures_in_txn(cur, 1, "202605-001", row, [])

    sql, params = _filter_summary_writes(cur.executed)[0]
    assert params[_INSERT_RULE_IDX] is None


def test_writer_neutralizes_string_null():
    """Pass Rule='null' (case-insensitive sentinel)."""
    cur = _make_cursor_for_insert()
    row = _well_formed_summary_row(Rule="null")
    db_layer._save_summary_with_closures_in_txn(cur, 1, "202605-001", row, [])

    sql, params = _filter_summary_writes(cur.executed)[0]
    assert params[_INSERT_RULE_IDX] is None


def test_writer_preserves_legitimate_string():
    """Pass Rule='br3.2'. Writer must bind 'br3.2' unchanged."""
    cur = _make_cursor_for_insert()
    row = _well_formed_summary_row(Rule="br3.2")
    db_layer._save_summary_with_closures_in_txn(cur, 1, "202605-001", row, [])

    sql, params = _filter_summary_writes(cur.executed)[0]
    assert params[_INSERT_RULE_IDX] == "br3.2"


def test_writer_preserves_legitimate_with_internal_nan():
    """Pass Rule='Plan: nan-tolerant approach'. Substring 'nan' inside
    other prose must NOT be scrubbed — only standalone sentinel values.
    """
    cur = _make_cursor_for_insert()
    row = _well_formed_summary_row(Rule="Plan: nan-tolerant approach")
    db_layer._save_summary_with_closures_in_txn(cur, 1, "202605-001", row, [])

    sql, params = _filter_summary_writes(cur.executed)[0]
    assert params[_INSERT_RULE_IDX] == "Plan: nan-tolerant approach"


@pytest.mark.parametrize("col,bind_idx", [
    ("Rule",       _INSERT_RULE_IDX),
    ("Buy_Notes",  _INSERT_BUY_NOTES_IDX),
    ("Sell_Rule",  _INSERT_SELL_RULE_IDX),
    ("Sell_Notes", _INSERT_SELL_NOTES_IDX),
    ("Notes",      _INSERT_NOTES_IDX),
])
def test_writer_cleans_all_user_text_fields(col, bind_idx):
    """Each of the 5 user-prose text columns gets the same treatment:
    'nan' string → None at the bind site."""
    cur = _make_cursor_for_insert()
    row = _well_formed_summary_row(**{col: "nan"})
    db_layer._save_summary_with_closures_in_txn(cur, 1, "202605-001", row, [])

    sql, params = _filter_summary_writes(cur.executed)[0]
    assert params[bind_idx] is None, \
        f"{col} should bind None for 'nan' input, got {params[bind_idx]!r}"


def test_writer_does_not_touch_non_text_fields():
    """Pass Status='F', Instrument_Type='STOCK', Risk_Budget=2500.0,
    Shares=100.0 — all unchanged. Defends against an over-eager future
    refactor that mistakes enum-like columns for user-prose."""
    cur = _make_cursor_for_insert()
    row = _well_formed_summary_row(
        Status="F", Instrument_Type="STOCK",
        Risk_Budget=2500.0, Shares=100.0,
    )
    db_layer._save_summary_with_closures_in_txn(cur, 1, "202605-001", row, [])

    sql, params = _filter_summary_writes(cur.executed)[0]

    # INSERT param positions: 2=Ticker, 3=Status, 6=Shares, 19=Risk_Budget,
    # 20=Instrument_Type. (See _save_summary_with_closures_in_txn INSERT
    # statement for the canonical order.)
    assert params[3]  == "F"
    assert params[20] == "STOCK"
    assert params[19] == 2500.0
    assert params[6]  == 100.0


def test_writer_idempotent_with_well_formed_input():
    """Pass a fully-clean dict — no sentinels, no NaN, no garbage. Every
    user-text bind matches the input. Regression guard for the
    'doesn't change behavior for well-formed input' constraint."""
    cur = _make_cursor_for_insert()
    row = _well_formed_summary_row(
        Rule="br1.3 Cup w/o Handle",
        Buy_Notes="Initial entry on breakout",
        Sell_Rule="sr2.1 Stop hit",
        Sell_Notes="Trailing stop triggered",
        Notes="Added on dip",
    )
    db_layer._save_summary_with_closures_in_txn(cur, 1, "202605-001", row, [])

    sql, params = _filter_summary_writes(cur.executed)[0]
    assert params[_INSERT_RULE_IDX]       == "br1.3 Cup w/o Handle"
    assert params[_INSERT_BUY_NOTES_IDX]  == "Initial entry on breakout"
    assert params[_INSERT_SELL_RULE_IDX]  == "sr2.1 Stop hit"
    assert params[_INSERT_SELL_NOTES_IDX] == "Trailing stop triggered"
    assert params[_INSERT_NOTES_IDX]      == "Added on dip"


# ---------------------------------------------------------------------------
# 2 tests against save_summary_row (Q1 expansion — sentinel arm in clean_value)
# ---------------------------------------------------------------------------


def test_save_summary_row_neutralizes_string_nan(monkeypatch):
    """Q1 expansion: save_summary_row's nested clean_value now strips
    'nan'/'none'/'null' string sentinels in addition to NaN/numpy types.
    Pass Rule='nan' through the public save_summary_row, assert UPDATE
    binds None at the rule position."""
    cur = _FakeCursor([
        (1,),       # SELECT id FROM portfolios
        (7,),       # SELECT id FROM trades_summary (existing row)
        (7,),       # RETURNING id from UPDATE
    ])
    conn = _FakeConn(cur)
    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda: _ConnCM(conn))

    row = _well_formed_summary_row(Rule="nan")
    db_layer.save_summary_row("CanSlim", row)

    # Find the UPDATE on trades_summary
    updates = [(sql, params) for sql, params in cur.executed
               if "UPDATE trades_summary" in sql]
    assert len(updates) == 1
    sql, params = updates[0]
    # save_summary_row's UPDATE param order:
    #   ticker, status, open_date, closed_date, shares, avg_entry, avg_exit,
    #   total_cost, realized_pl, unrealized_pl, return_pct, sell_rule, notes,
    #   stop_loss, rule, buy_notes, sell_notes, risk_budget, [optional grade,
    #   instrument_type/multiplier, strategy], existing[0]
    # rule is at position 14 (0-indexed).
    assert params[14] is None, \
        f"Rule should bind None for 'nan' input, got {params[14]!r}"


def test_save_summary_row_preserves_legitimate_string(monkeypatch):
    """Q1 expansion: well-formed Rule passes through unchanged."""
    cur = _FakeCursor([
        (1,),       # portfolio lookup
        (7,),       # existing summary row
        (7,),       # RETURNING id
    ])
    conn = _FakeConn(cur)
    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda: _ConnCM(conn))

    row = _well_formed_summary_row(Rule="br1.3 Cup w/o Handle")
    db_layer.save_summary_row("CanSlim", row)

    updates = [(sql, params) for sql, params in cur.executed
               if "UPDATE trades_summary" in sql]
    sql, params = updates[0]
    assert params[14] == "br1.3 Cup w/o Handle"


# ---------------------------------------------------------------------------
# numpy scalar psycopg2 adapter (regression: schema "np" does not exist)
# ---------------------------------------------------------------------------
# Reproduces the user-reported failure: exercise-option passed a DataFrame-
# sourced Stop_Loss (np.float64) into the parameterized UPDATE inside
# _save_summary_with_closures_in_txn. psycopg2 has no built-in numpy
# adapter; under numpy 2.0+, the repr() fallback emitted "np.float64(123.45)"
# which Postgres parsed as schema "np" → InvalidSchemaName. db_layer
# registers an adapter at import that converts numpy scalars to native
# Python via .item() before psycopg2 sees them; these tests pin that down.


def test_numpy_float64_adapts_to_native_repr():
    """np.float64(123.45) must adapt to b'123.45', not b'np.float64(123.45)'."""
    from psycopg2.extensions import adapt
    quoted = adapt(np.float64(123.45)).getquoted()
    assert quoted == b"123.45", f"unexpected adapter output: {quoted!r}"


def test_numpy_int64_adapts_to_native_repr():
    from psycopg2.extensions import adapt
    quoted = adapt(np.int64(42)).getquoted()
    assert quoted == b"42", f"unexpected adapter output: {quoted!r}"


def test_numpy_bool_adapts_to_native_repr():
    from psycopg2.extensions import adapt
    quoted = adapt(np.bool_(True)).getquoted()
    assert quoted in (b"true", b"True"), f"unexpected adapter output: {quoted!r}"


def test_numpy_nan_adapts_to_null():
    """NaN must become NULL (matches the pd.isna() → None behaviour the
    per-row _clean helpers already use). Without this, np.float64('nan')
    would emit literal 'nan' which Postgres rejects on numeric columns."""
    from psycopg2.extensions import adapt
    quoted = adapt(np.float64("nan")).getquoted()
    assert quoted == b"NULL", f"NaN should map to NULL, got: {quoted!r}"


def test_dataframe_sourced_numpy_value_adapts_correctly():
    """End-to-end repro of the exercise-option failure shape: pull a value
    via DataFrame.iloc[0].get(...) (the exact path opt_row.get('stop_loss')
    takes) and verify the adapter handles it."""
    from psycopg2.extensions import adapt
    df = pd.DataFrame({"stop_loss": [195.50]})
    val = df.iloc[0].get("stop_loss")
    assert isinstance(val, np.floating), \
        f"DataFrame should yield numpy scalar, got {type(val).__name__}"
    quoted = adapt(val).getquoted()
    assert quoted == b"195.5", f"unexpected adapter output: {quoted!r}"
