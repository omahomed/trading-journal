"""Regression tests for edit_transaction → trades_summary mirror (Phase 2, Commit 2).

The bug: c0435ee's recompute preservation block locks the summary's user-fields
(rule, buy_notes, sell_rule, sell_notes, risk_budget, stop_loss) to whatever
they were before the edit. So when the user edits a BUY's rule via
edit_transaction_endpoint, the detail's rule changes but the summary's rule
stays stale — and the card UI (trade-journal.tsx, perf-heatmap.tsx) reads
summary.rule directly, so the user's edit is invisible.

The fix: db_layer.mirror_detail_edit_to_summary re-derives the canonical
summary fields from the detail rows AFTER the detail UPDATE and BEFORE the
recompute. The recompute's preservation block then sees the just-mirrored
values and faithfully preserves them.

Mirror semantics:
  - earliest BUY (date ASC, id ASC) wins for summary.rule, buy_notes, stop_loss
  - latest SELL (date DESC, id DESC) on a CLOSED campaign wins for
    summary.sell_rule, sell_notes
  - OPEN campaign with partial sells: sell_rule/sell_notes left alone

Tests guard the mirror in two layers:
  1. Endpoint integration — assert edit_transaction_endpoint calls the mirror
     helper at the right point with the right args
  2. Helper unit — assert the mirror helper issues the right SQL UPDATEs given
     synthetic fetchone() return values

The c0435ee preservation tests in test_recompute_preserves_metadata.py
continue to pass — the mirror happens BEFORE recompute, so recompute's
preservation block reads the just-mirrored value.
"""
from __future__ import annotations

from typing import Any

import jwt
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import db_layer


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Endpoint integration scaffolding (mirrors test_recompute_preserves_metadata)
# ---------------------------------------------------------------------------


def _existing_summary_row(rule="initial rule", buy_notes="initial notes",
                          sell_rule=None, sell_notes=None,
                          stop_loss=195.0, status="OPEN"):
    return {
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "status": status,
        "open_date": pd.Timestamp("2026-05-01"),
        "closed_date": None,
        "shares": 100.0,
        "avg_entry": 200.0,
        "avg_exit": 0.0,
        "total_cost": 20000.0,
        "realized_pl": 0.0,
        "unrealized_pl": 0.0,
        "return_pct": 0.0,
        "rule": rule,
        "buy_notes": buy_notes,
        "sell_rule": sell_rule,
        "sell_notes": sell_notes,
        "risk_budget": 500.0,
        "stop_loss": stop_loss,
        "instrument_type": "STOCK",
        "multiplier": 1.0,
        "strategy": "CanSlim",
    }


def _buy_detail_row(detail_id=1, trx_id="B1", rule="initial rule",
                    notes="initial notes", stop_loss=195.0,
                    date="2026-05-01 09:30:00"):
    return {
        "detail_id": detail_id,
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "action": "BUY",
        "date": pd.Timestamp(date),
        "shares": 100.0,
        "amount": 200.0,
        "value": 20000.0,
        "rule": rule,
        "notes": notes,
        "realized_pl": 0,
        "stop_loss": stop_loss,
        "trx_id": trx_id,
        "instrument_type": "STOCK",
        "multiplier": 1.0,
    }


@pytest.fixture
def stubbed(monkeypatch):
    """Yield (state, client). Stubs db_layer reads, captures mirror calls and
    summary saves so tests can assert what edit_transaction_endpoint passes."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "summary_df": pd.DataFrame([_existing_summary_row()]),
        "details_df": pd.DataFrame([_buy_detail_row()]),
        "mirror_calls": [],
        "saved_with_closures": [],
        "saved_summaries": [],
        "updated_details": [],
        "audit_logs": [],
    }

    monkeypatch.setattr(db_layer, "load_summary",
                        lambda *a, **kw: state["summary_df"])
    monkeypatch.setattr(db_layer, "load_details",
                        lambda *a, **kw: state["details_df"])
    monkeypatch.setattr(main, "_normalize_trades", lambda df: df)

    def fake_mirror(portfolio, trade_id):
        state["mirror_calls"].append((portfolio, trade_id))
        # Default behavior: no-op on summary_df. Specific tests override this
        # to actually mutate state["summary_df"] (simulates the real mirror's
        # write through to summary so the recompute reads the post-mirror
        # state).
    monkeypatch.setattr(db_layer, "mirror_detail_edit_to_summary", fake_mirror)

    def fake_save_with_closures(portfolio, trade_id, summary_row, closures):
        state["saved_with_closures"].append({
            "portfolio": portfolio,
            "trade_id": trade_id,
            "summary_row": dict(summary_row),
            "closures": list(closures),
        })
        return 1
    monkeypatch.setattr(db_layer, "save_summary_with_closures",
                        fake_save_with_closures)

    def fake_save_summary_row(portfolio, row_dict):
        state["saved_summaries"].append({
            "portfolio": portfolio,
            "row": dict(row_dict),
        })
        return 1
    monkeypatch.setattr(db_layer, "save_summary_row", fake_save_summary_row)

    def fake_update_detail(portfolio, detail_id, row_dict):
        state["updated_details"].append({
            "portfolio": portfolio,
            "detail_id": detail_id,
            "row": dict(row_dict),
        })
    monkeypatch.setattr(db_layer, "update_detail_row", fake_update_detail)

    monkeypatch.setattr(db_layer, "save_detail_row",
                        lambda *a, **kw: 1)
    monkeypatch.setattr(db_layer, "delete_detail_row",
                        lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "delete_trade",
                        lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "delete_lot_closures_for_trade",
                        lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "generate_unique_trx_id",
                        lambda portfolio, trade_id, prefix: f"{prefix}1")

    def fake_log_audit(portfolio, action, trade_id, ticker, details, username='web'):
        state["audit_logs"].append({
            "portfolio": portfolio, "action": action, "trade_id": trade_id,
            "ticker": ticker, "details": details, "username": username,
        })
    monkeypatch.setattr(db_layer, "log_audit", fake_log_audit)

    monkeypatch.setattr(main, "validate_post_edit_matching",
                        lambda *a, **kw: None)

    if hasattr(main.limiter, "enabled"):
        original_enabled = main.limiter.enabled
        main.limiter.enabled = False
    else:
        original_enabled = None

    client = TestClient(main.app, headers=_auth_headers())
    try:
        yield state, client
    finally:
        if original_enabled is not None:
            main.limiter.enabled = original_enabled


# ---------------------------------------------------------------------------
# Endpoint integration tests (5)
# ---------------------------------------------------------------------------


def test_edit_b1_rule_mirrors_to_summary(stubbed):
    """Editing B1's rule triggers the mirror helper after detail update."""
    state, client = stubbed

    r = client.put("/api/trades/edit-transaction", json={
        "detail_id": 1,
        "portfolio": "CanSlim",
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "action": "BUY",
        "date": "2026-05-01 09:30",
        "shares": 100,
        "amount": 200.0,
        "rule": "br1.5 Pivot bounce",  # the edit
        "notes": "initial notes",
        "stop_loss": 195.0,
        "trx_id": "B1",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body

    # Detail update happened with the new rule
    assert state["updated_details"], "Expected update_detail_row call"
    assert state["updated_details"][-1]["row"]["Rule"] == "br1.5 Pivot bounce"

    # Mirror helper was called once with (portfolio, trade_id)
    assert state["mirror_calls"] == [("CanSlim", "202605-001")], \
        f"Expected one mirror call, got {state['mirror_calls']}"


def test_edit_b1_notes_mirrors_to_summary(stubbed):
    """Editing B1's notes triggers the mirror helper."""
    state, client = stubbed

    r = client.put("/api/trades/edit-transaction", json={
        "detail_id": 1,
        "portfolio": "CanSlim",
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "action": "BUY",
        "date": "2026-05-01 09:30",
        "shares": 100,
        "amount": 200.0,
        "rule": "initial rule",
        "notes": "Updated thesis: AI tailwind",  # the edit
        "stop_loss": 195.0,
        "trx_id": "B1",
    })
    assert r.status_code == 200, r.text
    assert state["updated_details"][-1]["row"]["Notes"] == "Updated thesis: AI tailwind"
    assert state["mirror_calls"] == [("CanSlim", "202605-001")]


def test_edit_b1_stop_loss_mirrors_to_summary(stubbed):
    """Editing B1's stop_loss triggers the mirror helper."""
    state, client = stubbed

    r = client.put("/api/trades/edit-transaction", json={
        "detail_id": 1,
        "portfolio": "CanSlim",
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "action": "BUY",
        "date": "2026-05-01 09:30",
        "shares": 100,
        "amount": 200.0,
        "rule": "initial rule",
        "notes": "initial notes",
        "stop_loss": 198.50,  # the edit
        "trx_id": "B1",
    })
    assert r.status_code == 200, r.text
    assert state["updated_details"][-1]["row"]["Stop_Loss"] == 198.50
    assert state["mirror_calls"] == [("CanSlim", "202605-001")]


def test_edit_then_recompute_preserves_mirrored_value(stubbed):
    """Regression guard for the c0435ee interaction.

    The mirror runs BEFORE the recompute. So when the recompute's
    preservation block reads summary.rule, it sees the just-mirrored value
    (the edit), not the stale pre-edit value. Therefore save_summary_with_closures
    captures the post-edit Rule, not the pre-edit one.
    """
    state, client = stubbed

    # Mirror stub mutates summary_df to simulate the real helper's write-through.
    # This way the recompute (which calls load_summary again) sees the post-mirror
    # state.
    def mirror_that_mutates_summary(portfolio, trade_id):
        state["mirror_calls"].append((portfolio, trade_id))
        # Simulate the helper writing summary.rule = "br1.5 Pivot bounce"
        # (the new earliest-BUY value after the edit).
        df = state["summary_df"].copy()
        mask = df["trade_id"] == trade_id
        df.loc[mask, "rule"] = "br1.5 Pivot bounce"
        df.loc[mask, "buy_notes"] = "Updated thesis"
        df.loc[mask, "stop_loss"] = 198.50
        state["summary_df"] = df
    db_layer.mirror_detail_edit_to_summary = mirror_that_mutates_summary  # type: ignore

    # Also simulate update_detail_row mutating details_df so recompute sees
    # the post-edit detail too (otherwise LIFO recomputes from stale details).
    original_update = db_layer.update_detail_row
    def update_that_mutates_details(portfolio, detail_id, row_dict):
        state["updated_details"].append({
            "portfolio": portfolio, "detail_id": detail_id, "row": dict(row_dict),
        })
        df = state["details_df"].copy()
        df.loc[df["detail_id"] == detail_id, "rule"] = row_dict["Rule"]
        df.loc[df["detail_id"] == detail_id, "notes"] = row_dict["Notes"]
        df.loc[df["detail_id"] == detail_id, "stop_loss"] = row_dict["Stop_Loss"]
        state["details_df"] = df
    db_layer.update_detail_row = update_that_mutates_details  # type: ignore

    r = client.put("/api/trades/edit-transaction", json={
        "detail_id": 1,
        "portfolio": "CanSlim",
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "action": "BUY",
        "date": "2026-05-01 09:30",
        "shares": 100,
        "amount": 200.0,
        "rule": "br1.5 Pivot bounce",
        "notes": "Updated thesis",
        "stop_loss": 198.50,
        "trx_id": "B1",
    })
    assert r.status_code == 200, r.text

    # Mirror was called
    assert state["mirror_calls"] == [("CanSlim", "202605-001")]

    # Recompute fired and captured save_summary_with_closures sees the
    # post-mirror values via the preservation block.
    assert state["saved_with_closures"], "Expected recompute to fire save_summary_with_closures"
    saved = state["saved_with_closures"][-1]["summary_row"]
    assert saved.get("Rule") == "br1.5 Pivot bounce", \
        f"Recompute should preserve mirrored Rule, got {saved.get('Rule')!r}"
    assert saved.get("Buy_Notes") == "Updated thesis", \
        f"Recompute should preserve mirrored Buy_Notes, got {saved.get('Buy_Notes')!r}"
    assert saved.get("Stop_Loss") == 198.50, \
        f"Recompute should preserve mirrored Stop_Loss, got {saved.get('Stop_Loss')!r}"

    # Restore the original update_detail_row so the fixture teardown works.
    db_layer.update_detail_row = original_update  # type: ignore


def test_mirror_failure_logs_audit_event(monkeypatch, stubbed):
    """If the mirror helper raises, the failure is logged to audit_trail
    with action='MIRROR_FAILED' so silent failures show up in the admin
    Audit Trail viewer."""
    state, client = stubbed

    def mirror_explodes(portfolio, trade_id):
        raise RuntimeError("simulated DB failure")
    monkeypatch.setattr(db_layer, "mirror_detail_edit_to_summary",
                        mirror_explodes)

    r = client.put("/api/trades/edit-transaction", json={
        "detail_id": 1,
        "portfolio": "CanSlim",
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "action": "BUY",
        "date": "2026-05-01 09:30",
        "shares": 100,
        "amount": 200.0,
        "rule": "br1.5 Pivot bounce",
        "notes": "initial notes",
        "stop_loss": 195.0,
        "trx_id": "B1",
    })
    # Request still succeeds — mirror failure is non-fatal.
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body

    # MIRROR_FAILED audit was logged.
    mirror_audits = [a for a in state["audit_logs"] if a["action"] == "MIRROR_FAILED"]
    assert len(mirror_audits) == 1, \
        f"Expected one MIRROR_FAILED audit, got {state['audit_logs']}"
    assert mirror_audits[0]["trade_id"] == "202605-001"
    assert "simulated DB failure" in mirror_audits[0]["details"]


# ---------------------------------------------------------------------------
# Helper unit tests (5) — bypass FastAPI, test mirror_detail_edit_to_summary
# directly against a stubbed cursor.
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Records every execute() call. fetchone() returns from a queue."""
    def __init__(self, fetchones):
        self._fetchones = list(fetchones)  # consumed FIFO
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


def _make_conn(fetchones):
    cur = _FakeCursor(fetchones)
    conn = _FakeConn(cur)
    return conn, cur


def _filter_updates(executed):
    """Return only UPDATE trades_summary statements from the cursor log."""
    return [(sql, params) for sql, params in executed
            if "UPDATE trades_summary" in sql]


def test_mirror_helper_writes_earliest_buy_fields(monkeypatch):
    """fetchone returns: portfolio_id row, then earliest BUY row, then a
    no-SELL (None). Helper must issue exactly one BUY-side UPDATE with the
    fetched values, no SELL UPDATE, and commit + cache-clear."""
    fetches = [
        (1,),                                 # portfolio_id
        ("br1.3 Cup w/o Handle",
         "Initial entry on breakout", 195.0), # earliest BUY (rule, notes, stop)
        None,                                 # latest SELL — none (open trade)
    ]
    conn, cur = _make_conn(fetches)

    cleared = {"summary": 0}
    class _LSStub:
        def clear(self): cleared["summary"] += 1
    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda: _ConnCM(conn))
    monkeypatch.setattr(db_layer, "load_summary", _LSStub())

    db_layer.mirror_detail_edit_to_summary("CanSlim", "202605-001")

    updates = _filter_updates(cur.executed)
    assert len(updates) == 1, f"Expected 1 UPDATE, got {len(updates)}"
    sql, params = updates[0]
    assert "rule = %s" in sql and "buy_notes = %s" in sql and "stop_loss = %s" in sql
    # params: (rule, notes, stop_loss, portfolio_id, trade_id)
    assert params == ("br1.3 Cup w/o Handle", "Initial entry on breakout",
                      195.0, 1, "202605-001")
    assert conn.commits == 1
    assert cleared["summary"] == 1


def test_mirror_helper_skips_sell_update_when_open(monkeypatch):
    """Latest-SELL SELECT joins on status='CLOSED'. For an OPEN campaign with
    a partial sell, the join filter excludes it, fetchone returns None, and
    no SELL UPDATE is issued."""
    fetches = [
        (1,),                                  # portfolio_id
        ("X", "n", 100.0),                     # earliest BUY
        None,                                  # latest SELL — None (status=OPEN filtered out)
    ]
    conn, cur = _make_conn(fetches)
    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda: _ConnCM(conn))
    monkeypatch.setattr(db_layer, "load_summary",
                        type("S", (), {"clear": lambda self: None})())

    db_layer.mirror_detail_edit_to_summary("CanSlim", "T1")

    updates = _filter_updates(cur.executed)
    assert len(updates) == 1
    sql, _ = updates[0]
    assert "sell_rule" not in sql, "Expected no SELL UPDATE for OPEN campaign"


def test_mirror_helper_writes_sell_fields_when_closed(monkeypatch):
    """Latest-SELL SELECT returns a row → SELL UPDATE issued."""
    fetches = [
        (1,),                                            # portfolio_id
        ("br1", "buy notes", 100.0),                     # earliest BUY
        ("sr1 trailing stop", "exited at trail"),        # latest SELL on CLOSED
    ]
    conn, cur = _make_conn(fetches)
    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda: _ConnCM(conn))
    monkeypatch.setattr(db_layer, "load_summary",
                        type("S", (), {"clear": lambda self: None})())

    db_layer.mirror_detail_edit_to_summary("CanSlim", "T1")

    updates = _filter_updates(cur.executed)
    assert len(updates) == 2, f"Expected 2 UPDATEs (BUY + SELL), got {len(updates)}"
    sell_sql, sell_params = updates[1]
    assert "sell_rule = %s" in sell_sql and "sell_notes = %s" in sell_sql
    assert sell_params == ("sr1 trailing stop", "exited at trail", 1, "T1")


def test_mirror_helper_no_op_when_no_buys(monkeypatch):
    """Mid-delete state: no BUY rows. Helper should issue zero UPDATEs."""
    fetches = [
        (1,),     # portfolio_id
        None,     # earliest BUY — none
        None,     # latest SELL — none
    ]
    conn, cur = _make_conn(fetches)
    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda: _ConnCM(conn))
    monkeypatch.setattr(db_layer, "load_summary",
                        type("S", (), {"clear": lambda self: None})())

    db_layer.mirror_detail_edit_to_summary("CanSlim", "T1")

    updates = _filter_updates(cur.executed)
    assert len(updates) == 0, f"Expected zero UPDATEs, got {updates}"
    assert conn.commits == 1, "commit should still fire (no-op transaction)"


def test_mirror_helper_sanitizes_nan_via_clean_text(monkeypatch):
    """If the detail row has np.nan or 'nan' string in rule/notes, the helper
    binds None to the UPDATE — defensive clean_text_value pass."""
    fetches = [
        (1,),                          # portfolio_id
        (np.nan, "nan", 195.0),        # earliest BUY — both pollution forms
        None,                          # no SELL
    ]
    conn, cur = _make_conn(fetches)
    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda: _ConnCM(conn))
    monkeypatch.setattr(db_layer, "load_summary",
                        type("S", (), {"clear": lambda self: None})())

    db_layer.mirror_detail_edit_to_summary("CanSlim", "T1")

    updates = _filter_updates(cur.executed)
    assert len(updates) == 1
    _, params = updates[0]
    # rule (np.nan), notes ('nan' string) — both → None
    assert params[0] is None, f"rule should sanitize to None, got {params[0]!r}"
    assert params[1] is None, f"notes should sanitize to None, got {params[1]!r}"
    assert params[2] == 195.0  # stop_loss preserved (numeric, not text)


# ---------------------------------------------------------------------------
# Scenario tests (3) — earliest-BUY-wins with multiple BUYs
# ---------------------------------------------------------------------------


def test_b2_edit_does_not_change_summary_when_b1_earlier(monkeypatch):
    """Setup: B1 (earlier) has rule='X', B2 (later) has rule='Y'. Even though
    we just edited B2, the helper's SELECT for earliest BUY returns B1's rule.
    UPDATE writes summary.rule = 'X' (unchanged from user perspective).
    """
    fetches = [
        (1,),
        ("X", "B1 notes", 195.0),    # earliest BUY (B1) — UNCHANGED by edit to B2
        None,
    ]
    conn, cur = _make_conn(fetches)
    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda: _ConnCM(conn))
    monkeypatch.setattr(db_layer, "load_summary",
                        type("S", (), {"clear": lambda self: None})())

    db_layer.mirror_detail_edit_to_summary("CanSlim", "T1")

    updates = _filter_updates(cur.executed)
    assert len(updates) == 1
    _, params = updates[0]
    assert params[0] == "X", f"summary.rule should pin to B1's 'X', got {params[0]!r}"


def test_b1_edit_promotes_new_rule_when_b2_exists(monkeypatch):
    """Setup: B1 edited to rule='Z' (was 'X'); B2 still rule='Y'. Earliest BUY
    is still B1 (it's earlier by date). SELECT returns B1's NEW rule 'Z'.
    UPDATE writes summary.rule = 'Z'.
    """
    fetches = [
        (1,),
        ("Z", "B1 notes updated", 195.0),  # earliest BUY (B1) post-edit
        None,
    ]
    conn, cur = _make_conn(fetches)
    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda: _ConnCM(conn))
    monkeypatch.setattr(db_layer, "load_summary",
                        type("S", (), {"clear": lambda self: None})())

    db_layer.mirror_detail_edit_to_summary("CanSlim", "T1")

    updates = _filter_updates(cur.executed)
    assert len(updates) == 1
    _, params = updates[0]
    assert params[0] == "Z", f"summary.rule should promote to 'Z', got {params[0]!r}"


def test_partial_sell_edit_does_not_mirror(monkeypatch):
    """OPEN campaign with B1 + S1 (partial sell). Editing S1's rule should
    NOT mirror — the latest-SELL SELECT joins on status='CLOSED' and the
    OPEN campaign returns None for the SELL fetch.
    """
    fetches = [
        (1,),
        ("X", "B1 notes", 195.0),  # earliest BUY
        None,                       # latest SELL — none (OPEN status filter)
    ]
    conn, cur = _make_conn(fetches)
    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda: _ConnCM(conn))
    monkeypatch.setattr(db_layer, "load_summary",
                        type("S", (), {"clear": lambda self: None})())

    db_layer.mirror_detail_edit_to_summary("CanSlim", "T1")

    updates = _filter_updates(cur.executed)
    # Only the BUY-side UPDATE; no SELL update issued
    assert len(updates) == 1
    sql, _ = updates[0]
    assert "sell_rule" not in sql
