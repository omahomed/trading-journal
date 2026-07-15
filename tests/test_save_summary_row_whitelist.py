"""Whitelist-refactor tests for save_summary_row (Phase 2, Commit 6).

Phase 1 sweep audit's largest architectural finding: save_summary_row
accepted partial dicts and silently bound omitted keys to defaults
(None or 0), wiping the existing row's values for those columns. Three
prior commits closed individual instances by adding preservation blocks
at the caller. Commit 6 is the systemic fix: the writer's UPDATE path
becomes partial-dict-safe — only columns whose keys are present in the
input dict get bound; omitted keys leave the existing DB value untouched.

Implementation:
- _TRADES_SUMMARY_UPDATE_COLUMNS module-level map documents the legitimate
  UPDATE surface (PascalCase dict-key → snake_case DB column)
- _build_summary_update_set_clauses helper builds the dynamic SET clause
  + params from the intersection of (dict keys, whitelist)
- INSERT path is unchanged (a fresh row needs values for every column)
- Schema-fallback retry preserved: on legacy-schema exception, drop the
  newer-migration columns from a working dict copy and retry

Tests are direct unit tests against save_summary_row using a fake cursor
and monkeypatched get_db_connection — same pattern as Commit 5's
test_save_summary_defensive.py.
"""
from __future__ import annotations

from typing import Any

import pytest

import db_layer


# ---------------------------------------------------------------------------
# Fake cursor scaffolding (same shape as test_save_summary_defensive.py)
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, fetchones):
        self._fetchones = list(fetchones)
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, tuple(params) if params else ()))

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


def _setup(monkeypatch, fetchones):
    """Stub get_db_connection to return a fake conn whose cursor fetchones
    queue is `fetchones`. Returns (conn, cur)."""
    cur = _FakeCursor(fetchones)
    conn = _FakeConn(cur)
    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda: _ConnCM(conn))
    return conn, cur


def _setup_for_update(monkeypatch, existing_id=7, returning_id=7):
    """Cursor returns: portfolio_id, existing summary id, RETURNING id."""
    return _setup(monkeypatch, [(1,), (existing_id,), (returning_id,)])


def _setup_for_insert(monkeypatch, returning_id=42):
    """Cursor returns: portfolio_id, no existing row, RETURNING id."""
    return _setup(monkeypatch, [(1,), None, (returning_id,)])


def _trades_summary_updates(executed):
    """Return only UPDATE statements against trades_summary."""
    return [(sql, params) for sql, params in executed
            if "UPDATE trades_summary" in sql]


def _trades_summary_inserts(executed):
    """Return only INSERT statements into trades_summary."""
    return [(sql, params) for sql, params in executed
            if "INSERT INTO trades_summary" in sql]


# ---------------------------------------------------------------------------
# 1. Partial-dict UPDATE — the headline behavior change
# ---------------------------------------------------------------------------


def test_partial_update_only_binds_provided_columns(monkeypatch):
    """Pass a dict with only Trade_ID, Ticker, Shares. The UPDATE must bind
    exactly 2 columns (ticker, shares) and not touch rule, buy_notes,
    risk_budget, etc. — the columns that existed before this refactor's
    silent-wipe surface."""
    conn, cur = _setup_for_update(monkeypatch)

    db_layer.save_summary_row("CanSlim", {
        "Trade_ID": "202605-001",
        "Ticker": "AAPL",
        "Shares": 100.0,
    })

    updates = _trades_summary_updates(cur.executed)
    assert len(updates) == 1
    sql, params = updates[0]

    # SET clause has only the two columns we passed
    assert "ticker = %s" in sql
    assert "shares = %s" in sql
    # And nothing else
    for excluded in ("rule", "buy_notes", "sell_rule", "sell_notes",
                     "risk_budget", "stop_loss", "notes", "avg_entry",
                     "realized_pl", "unrealized_pl"):
        assert f"{excluded} = %s" not in sql, \
            f"{excluded} should not be in SET clause for partial dict"

    # params: ticker, shares, existing[0]
    assert len(params) == 3
    assert params[0] == "AAPL"
    assert params[1] == 100.0
    assert params[2] == 7  # existing[0]


def test_explicit_none_sets_null(monkeypatch):
    """Pass {Trade_ID, Rule: None}. UPDATE must bind rule = NULL AND
    the auto-synced rules array = []. Migration 047: any Rule write
    also propagates to Rules to prevent divergence — Rule=None means
    both columns clear."""
    conn, cur = _setup_for_update(monkeypatch)

    db_layer.save_summary_row("CanSlim", {
        "Trade_ID": "202605-001",
        "Rule": None,
    })

    sql, params = _trades_summary_updates(cur.executed)[0]
    assert "rule = %s" in sql
    assert "rules = %s" in sql
    # rule NULL + rules '[]' + existing[0]
    assert len(params) == 3
    assert params[0] is None
    assert params[1] == "[]"
    assert params[2] == 7


def test_empty_dict_raises_value_error(monkeypatch):
    """Pass {Trade_ID} only — no columns to update. Must raise ValueError."""
    conn, cur = _setup_for_update(monkeypatch)

    with pytest.raises(ValueError, match="no columns to UPDATE"):
        db_layer.save_summary_row("CanSlim", {"Trade_ID": "202605-001"})


def test_unknown_column_ignored_with_warning(monkeypatch, capsys):
    """Pass an unknown column key. It must be ignored (not bound), and a
    warning print must appear in stdout for visibility."""
    conn, cur = _setup_for_update(monkeypatch)

    db_layer.save_summary_row("CanSlim", {
        "Trade_ID": "202605-001",
        "Ticker": "AAPL",
        "made_up_col": "X",
    })

    sql, params = _trades_summary_updates(cur.executed)[0]
    # Only ticker bound
    assert "ticker = %s" in sql
    assert "made_up_col" not in sql

    # Warning printed to stdout
    captured = capsys.readouterr()
    assert "ignored unknown columns" in captured.out
    assert "made_up_col" in captured.out


# ---------------------------------------------------------------------------
# 2. INSERT path — unchanged behavior
# ---------------------------------------------------------------------------


def test_insert_path_still_binds_all_default_columns(monkeypatch):
    """No existing row → INSERT path. Even on a partial dict, INSERT still
    binds defaults for omitted columns. The whitelist applies to UPDATE only.
    """
    conn, cur = _setup_for_insert(monkeypatch)

    db_layer.save_summary_row("CanSlim", {
        "Trade_ID": "202605-001",
        "Ticker": "AAPL",
        "Shares": 100.0,
    })

    inserts = _trades_summary_inserts(cur.executed)
    assert len(inserts) == 1
    sql, params = inserts[0]

    # INSERT statement still includes the full column list (legacy behavior).
    # Just spot-check a couple of always-bind columns are in the SQL.
    for col in ("ticker", "status", "shares", "avg_entry", "rule",
                "buy_notes", "risk_budget"):
        assert col in sql, f"INSERT should include {col}"


# ---------------------------------------------------------------------------
# 3. Full-dict caller — unchanged behavior (regression guard)
# ---------------------------------------------------------------------------


def _full_dict():
    """The 21-column dict that current callers pass."""
    return {
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
        "Notes": "Some general notes",
        "Stop_Loss": 195.0,
        "Rule": "br1.3 Cup w/o Handle",
        "Buy_Notes": "Initial entry",
        "Sell_Notes": None,
        "Risk_Budget": 500.0,
        "Instrument_Type": "STOCK",
        "Multiplier": 1.0,
    }


def test_full_dict_caller_unchanged_behavior(monkeypatch):
    """A caller passing the complete 21-column dict (e.g., set_trade_grade)
    must produce a SET clause that still includes every column. Whitelist
    refactor is semantically equivalent for full-dict callers.

    Migration 047: `Rule` auto-syncs to `Rules` (JSON-serialized array),
    so a full-dict caller supplying only `Rule` now binds one extra
    column (rules) — expected count is 21 base columns + existing[0]."""
    conn, cur = _setup_for_update(monkeypatch)

    db_layer.save_summary_row("CanSlim", _full_dict())

    sql, params = _trades_summary_updates(cur.executed)[0]
    expected_cols = (
        "ticker", "status", "open_date", "closed_date",
        "shares", "avg_entry", "avg_exit", "total_cost",
        "realized_pl", "unrealized_pl", "return_pct",
        "sell_rule", "notes", "stop_loss", "rule",
        "buy_notes", "sell_notes", "risk_budget",
        "instrument_type", "multiplier",
        "rules",   # auto-synced from Rule
    )
    for col in expected_cols:
        assert f"{col} = %s" in sql, f"full dict should bind {col}"

    # 21 columns (20 base + rules) + existing[0]
    assert len(params) == 22


# ---------------------------------------------------------------------------
# 4. Grade validation
# ---------------------------------------------------------------------------


def test_grade_validation_passes_through_valid(monkeypatch):
    """Grade=3 is valid — bind 3."""
    conn, cur = _setup_for_update(monkeypatch)
    db_layer.save_summary_row("CanSlim",
                              {"Trade_ID": "202605-001", "Grade": 3})

    sql, params = _trades_summary_updates(cur.executed)[0]
    assert "grade = %s" in sql
    assert params[0] == 3


def test_grade_validation_clamps_invalid_to_none(monkeypatch):
    """Grade=7 (out of 1-5 range) → bind None."""
    conn, cur = _setup_for_update(monkeypatch)
    db_layer.save_summary_row("CanSlim",
                              {"Trade_ID": "202605-001", "Grade": 7})

    sql, params = _trades_summary_updates(cur.executed)[0]
    assert "grade = %s" in sql
    assert params[0] is None


def test_grade_validation_handles_string(monkeypatch):
    """Grade='invalid' (non-int) → bind None."""
    conn, cur = _setup_for_update(monkeypatch)
    db_layer.save_summary_row("CanSlim",
                              {"Trade_ID": "202605-001", "Grade": "invalid"})

    sql, params = _trades_summary_updates(cur.executed)[0]
    assert "grade = %s" in sql
    assert params[0] is None


# ---------------------------------------------------------------------------
# 5. The headline regression — the c0435ee scenario
# ---------------------------------------------------------------------------


def test_recompute_partial_dict_safe_after_refactor(monkeypatch):
    """The bug c0435ee fixed by adding preservation blocks at callers: a
    recompute path passed only LIFO-derived fields to save_summary_row, and
    the writer wiped rule/buy_notes/sell_rule/sell_notes/risk_budget/
    stop_loss to defaults.

    With the whitelist refactor, the writer is partial-dict-safe at its
    own boundary. This test simulates the exact pre-c0435ee dict (no
    preservation block, only LIFO-derived fields) and asserts the
    omitted user-prose columns are NOT in the SET clause — DB values for
    them are preserved by the UPDATE not touching them.
    """
    conn, cur = _setup_for_update(monkeypatch)

    # Pre-c0435ee shape: only LIFO-derived fields.
    db_layer.save_summary_row("CanSlim", {
        "Trade_ID": "202605-001",
        "Status": "OPEN",
        "Shares": 100.0,
        "Avg_Entry": 200.0,
        "Total_Cost": 20000.0,
        "Realized_PL": 0.0,
        "Return_Pct": 0.0,
    })

    sql, params = _trades_summary_updates(cur.executed)[0]
    # The 6 columns the bug used to wipe MUST NOT appear in the SET clause.
    for excluded in ("rule", "buy_notes", "sell_rule", "sell_notes",
                     "risk_budget", "stop_loss"):
        assert f"{excluded} = %s" not in sql, \
            f"Whitelist must not bind {excluded} when key is omitted from dict"
