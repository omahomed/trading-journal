"""Tests for /api/journal/batch-edit — multi-portfolio atomic save.

The endpoint coexists with /api/journal/edit (single-row) and writes
N rows in a single PG transaction. Tests verify:
  - Happy path (3 portfolios, fresh insert)
  - Conflict + force=false → 409, nothing written
  - Conflict + force=true → 200, rows updated (not duplicated)
  - Validation: missing end_nlv, missing total_holdings → 422
  - Unknown portfolio → 404
  - Transaction rollback on per-row failure
  - Snapshot semantics: auto-compute on insert, preserve on update
  - N=1 degenerate case
  - user_id propagation
  - beg_nlv chain is per-portfolio

A FakeCursor stand-in replays the DB the endpoint touches: portfolios
SELECT, trading_journal SELECT (existence + prior end_nlv), INSERT,
UPDATE. State lives in dicts so tests can assert post-conditions on
exact column values.
"""
from __future__ import annotations

from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"
_FOUNDER_UUID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


# Deterministic mock returns for the snapshot-auto-compute helpers so
# "fired" is distinguishable from "preserved" in assertions.
_AUTO_CYCLE = "POWERTREND"
_AUTO_DAY_NUM = 42
_AUTO_HEAT = 7.5
_AUTO_SPY_ATR = 1.234
_AUTO_NDX_ATR = 1.567


class _FakeCursor:
    """Replays the subset of psycopg2 cursor behavior the batch endpoint
    needs: SELECTs against portfolios + trading_journal, INSERT/UPDATE
    appending to in-memory journal_rows. The conn-level commit/rollback
    is owned by the parent fixture so per-test atomicity assertions work."""

    def __init__(self, state: dict[str, Any]):
        self.state = state
        self._last_returned: Any = None
        self._fetchall_buffer: list = []

    def execute(self, sql: str, params=()) -> None:
        sql_norm = " ".join(sql.split())
        # Portfolios SELECT: "SELECT id, user_id FROM portfolios WHERE name = %s"
        if "FROM portfolios WHERE name = %s" in sql_norm:
            name = params[0]
            pm = self.state["portfolios"].get(name)
            self._last_returned = (pm["id"], pm["user_id"]) if pm else None
            return

        # trading_journal existence + snapshot fields read
        if ("SELECT id, portfolio_heat, spy_atr, nasdaq_atr,"
                " market_cycle, mct_display_day_num FROM trading_journal" in sql_norm):
            pid, day = params
            for r in self.state["journal_rows"]:
                if r["portfolio_id"] == pid and r["day"] == day:
                    self._last_returned = (
                        r["id"],
                        r.get("portfolio_heat", 0.0),
                        r.get("spy_atr", 0.0),
                        r.get("nasdaq_atr", 0.0),
                        r.get("market_cycle", ""),
                        r.get("mct_display_day_num"),
                    )
                    return
            self._last_returned = None
            return

        # Pre-flight existence-only check: "SELECT 1 FROM trading_journal WHERE ..."
        if "SELECT 1 FROM trading_journal WHERE portfolio_id = %s AND day = %s" in sql_norm:
            pid, day = params
            for r in self.state["journal_rows"]:
                if r["portfolio_id"] == pid and r["day"] == day:
                    self._last_returned = (1,)
                    return
            self._last_returned = None
            return

        # prior-day end_nlv lookup
        if "SELECT end_nlv FROM trading_journal" in sql_norm and "day < %s" in sql_norm:
            pid, day = params
            candidates = [r for r in self.state["journal_rows"]
                          if r["portfolio_id"] == pid and r["day"] < day]
            if not candidates:
                self._last_returned = None
                return
            latest = max(candidates, key=lambda r: r["day"])
            self._last_returned = (latest["end_nlv"],)
            return

        # INSERT
        if sql_norm.startswith("INSERT INTO trading_journal"):
            # First param is user_id, second is portfolio_id, third is day
            row = {
                "id": self.state["_next_id"],
                "user_id": params[0],
                "portfolio_id": params[1],
                "day": params[2],
                "status": params[3],
                "market_window": params[4],
                "market_cycle": params[5],
                "mct_display_day_num": params[6],
                "above_21ema": params[7],
                "cash_change": params[8],
                "beg_nlv": params[9],
                "end_nlv": params[10],
                "daily_dollar_change": params[11],
                "daily_pct_change": params[12],
                "pct_invested": params[13],
                "spy": params[14],
                "nasdaq": params[15],
                "market_notes": params[16],
                "market_action": params[17],
                "portfolio_heat": params[18],
                "spy_atr": params[19],
                "nasdaq_atr": params[20],
                "score": params[21],
                "highlights": params[22],
                "lowlights": params[23],
                "mistakes": params[24],
                "top_lesson": params[25],
                "nlv_source": params[26],
                "holdings_source": params[27],
                "daily_thoughts": params[28],
            }
            self.state["_next_id"] += 1
            # Atomicity: the test owns rollback; the endpoint commits.
            # Buffer writes here so a failing test can assert nothing was
            # committed.
            self.state["pending_writes"].append(("insert", row))
            self._last_returned = (row["id"],)
            return

        # UPDATE
        if sql_norm.startswith("UPDATE trading_journal"):
            row_id = params[-1]
            self.state["pending_writes"].append(
                ("update", row_id, list(params[:-1]))
            )
            self._last_returned = (row_id,)
            return

        # Inject controlled failures for transaction-rollback test
        if self.state.get("fail_on_sql_substr"):
            if self.state["fail_on_sql_substr"] in sql_norm:
                raise RuntimeError(self.state.get("fail_message", "test forced failure"))

    def fetchone(self):
        return self._last_returned

    def fetchall(self):
        return self._fetchall_buffer

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeConn:
    def __init__(self, state: dict[str, Any]):
        self.state = state

    def cursor(self):
        return _FakeCursor(self.state)

    def commit(self):
        # Atomic flush: pending writes become committed journal_rows.
        for entry in self.state["pending_writes"]:
            kind = entry[0]
            if kind == "insert":
                self.state["journal_rows"].append(entry[1])
            elif kind == "update":
                row_id = entry[1]
                params = entry[2]
                for r in self.state["journal_rows"]:
                    if r["id"] == row_id:
                        # 26-param UPDATE; just copy the fields we care
                        # about for assertions (per the SQL column order).
                        r["status"], r["market_window"], r["market_cycle"] = params[0], params[1], params[2]
                        r["mct_display_day_num"], r["above_21ema"] = params[3], params[4]
                        r["cash_change"], r["beg_nlv"], r["end_nlv"] = params[5], params[6], params[7]
                        r["daily_dollar_change"], r["daily_pct_change"] = params[8], params[9]
                        r["pct_invested"], r["spy"], r["nasdaq"] = params[10], params[11], params[12]
                        r["market_notes"], r["market_action"] = params[13], params[14]
                        r["portfolio_heat"], r["spy_atr"], r["nasdaq_atr"] = params[15], params[16], params[17]
                        r["score"], r["highlights"], r["lowlights"] = params[18], params[19], params[20]
                        r["mistakes"], r["top_lesson"] = params[21], params[22]
                        r["nlv_source"], r["holdings_source"], r["daily_thoughts"] = params[23], params[24], params[25]
        self.state["pending_writes"] = []

    def rollback(self):
        self.state["pending_writes"] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # psycopg2's connection ctx manager rolls back on exception, commits
        # otherwise. We mirror that semantics so the endpoint's try/except
        # exercises the right path.
        if exc_type is not None:
            self.rollback()
        return False


@pytest.fixture
def batch_stubs(monkeypatch):
    """Yield (state, client). state.portfolios maps name → {id, user_id};
    state.journal_rows is the (in-memory) trading_journal table."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    import db_layer
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "portfolios": {
            "CanSlim": {"id": 1, "user_id": _FOUNDER_UUID},
            "457B Plan": {"id": 3, "user_id": _FOUNDER_UUID},
            "Long-Term Growth": {"id": 2, "user_id": _FOUNDER_UUID},
        },
        "journal_rows": [],
        "pending_writes": [],
        "_next_id": 100,
        "fail_on_sql_substr": None,
        "fail_message": None,
    }

    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda *a, **kw: _FakeConn(state))
    # Module-level alias in main.py is `db`, so patch through it too.
    monkeypatch.setattr(main.db, "get_db_connection",
                        lambda *a, **kw: _FakeConn(state))
    # load_journal.clear() is called at end; stub it to a no-op.
    monkeypatch.setattr(main.db.load_journal, "clear", lambda: None)

    monkeypatch.setattr(main, "_compute_mct_state_with_day_num",
                        lambda *a, **kw: (_AUTO_CYCLE, _AUTO_DAY_NUM))
    monkeypatch.setattr(main, "_compute_portfolio_heat",
                        lambda *a, **kw: _AUTO_HEAT)

    def fake_atr(ticker, *a, **kw):
        return _AUTO_NDX_ATR if "IXIC" in str(ticker) else _AUTO_SPY_ATR
    monkeypatch.setattr(main, "_compute_ticker_atr_pct", fake_atr)

    client = TestClient(main.app, headers=_auth_headers())
    yield state, client


def _shared(**overrides) -> dict:
    base = {
        "spy": 745.70, "nasdaq": 26343.97,
        "market_notes": "Test notes",
        "score": 5, "highlights": "{}",
        "mistakes": "",
        "nlv_source": "manual", "holdings_source": "manual",
    }
    base.update(overrides)
    return base


def _pf(name: str, **overrides) -> dict:
    base = {
        "portfolio": name,
        "end_nlv": 10000.0,
        "total_holdings": 9500.0,
        "cash_change": 0,
        "actions": "",
        "pct_invested": 95.0,
        "daily_dollar_change": 100.0,
        "daily_pct_change": 1.0,
    }
    base.update(overrides)
    return base


def _post(client, **overrides):
    body = {
        "day": "2026-05-22",
        "shared": _shared(),
        "portfolios": [_pf("CanSlim"), _pf("457B Plan"), _pf("Long-Term Growth")],
        "force_overwrite": False,
    }
    body.update(overrides)
    return client.post("/api/journal/batch-edit", json=body)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_happy_path_writes_three_rows(batch_stubs):
    """3 portfolios, no existing rows, all succeed. trading_journal ends
    up with 3 rows; shared fields identical across rows; per-portfolio
    fields match the request."""
    state, client = batch_stubs
    r = _post(client)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["rows_written"] == 3
    assert set(payload["portfolios"]) == {"CanSlim", "457B Plan", "Long-Term Growth"}

    assert len(state["journal_rows"]) == 3
    # Shared fields identical
    spy_values = {r["spy"] for r in state["journal_rows"]}
    assert spy_values == {745.70}
    notes = {r["market_notes"] for r in state["journal_rows"]}
    assert notes == {"Test notes"}


def test_conflict_force_false_returns_409_no_writes(batch_stubs):
    """Pre-create rows for some portfolios. force_overwrite=false →
    409 with conflict list; trading_journal unchanged."""
    state, client = batch_stubs
    state["journal_rows"].append({
        "id": 50, "portfolio_id": 1, "day": "2026-05-22",
        "end_nlv": 9000, "portfolio_heat": 3.3,
    })
    r = _post(client)
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["status"] == "exists"
    assert body["conflicting_portfolios"] == ["CanSlim"]
    # Verify no writes (the pre-existing row remains; no new rows added).
    assert len(state["journal_rows"]) == 1


def test_conflict_force_true_updates_existing(batch_stubs):
    """Pre-create row. force_overwrite=true → 200, row updated (not
    duplicated). Existing snapshot fields preserved (heat,
    market_cycle, atrs)."""
    state, client = batch_stubs
    state["journal_rows"].append({
        "id": 50, "portfolio_id": 1, "day": "2026-05-22",
        "end_nlv": 9000.0, "portfolio_heat": 3.3,
        "spy_atr": 0.9, "nasdaq_atr": 0.8,
        "market_cycle": "RALLY_ATTEMPT", "mct_display_day_num": 5,
    })
    r = _post(client, force_overwrite=True,
              portfolios=[_pf("CanSlim", end_nlv=10500.0)])
    assert r.status_code == 200, r.text
    assert r.json()["rows_written"] == 1
    assert len(state["journal_rows"]) == 1  # not duplicated
    updated = state["journal_rows"][0]
    assert updated["id"] == 50
    assert updated["end_nlv"] == 10500.0
    # Snapshot fields preserved, NOT recomputed to _AUTO_* values:
    assert updated["portfolio_heat"] == 3.3
    assert updated["market_cycle"] == "RALLY_ATTEMPT"
    assert updated["mct_display_day_num"] == 5
    assert updated["spy_atr"] == 0.9
    assert updated["nasdaq_atr"] == 0.8


def test_validation_missing_end_nlv_returns_422(batch_stubs):
    """One portfolio's end_nlv is None → 422; no writes."""
    state, client = batch_stubs
    r = _post(client, portfolios=[
        _pf("CanSlim"),
        _pf("457B Plan", end_nlv=None),
    ])
    assert r.status_code == 422, r.text
    errors = r.json()["errors"]
    assert any(e["portfolio"] == "457B Plan" and e["field"] == "end_nlv"
               for e in errors)
    assert len(state["journal_rows"]) == 0


def test_validation_missing_total_holdings_returns_422(batch_stubs):
    """One portfolio's total_holdings is None → 422; no writes."""
    state, client = batch_stubs
    r = _post(client, portfolios=[
        _pf("CanSlim", total_holdings=None),
    ])
    assert r.status_code == 422, r.text
    errors = r.json()["errors"]
    assert any(e["portfolio"] == "CanSlim" and e["field"] == "total_holdings"
               for e in errors)
    assert len(state["journal_rows"]) == 0


def test_unknown_portfolio_returns_404(batch_stubs):
    """Portfolio name not in the user's portfolios → 404."""
    state, client = batch_stubs
    r = _post(client, portfolios=[_pf("DoesNotExist")])
    assert r.status_code == 404, r.text
    assert "DoesNotExist" in r.json()["detail"]
    assert len(state["journal_rows"]) == 0


def test_transaction_rollback_on_per_row_failure(batch_stubs):
    """Inject a failure on the 2nd row's INSERT. No rows committed."""
    state, client = batch_stubs
    # Make the SECOND INSERT throw. Trigger on a unique substring in the
    # INSERT SQL; the cursor sees every execute, so we can fail on a
    # specific call by counting.
    insert_calls = {"count": 0}

    import api.main as main
    real_atr = main._compute_ticker_atr_pct
    # Inject failure by replacing _compute_portfolio_heat to throw on
    # the second invocation. Cleaner than instrumenting the cursor.
    call_count = {"n": 0}

    def fail_on_second(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated DB error on row 2")
        return _AUTO_HEAT
    main._compute_portfolio_heat = fail_on_second

    try:
        r = _post(client)
        assert r.status_code == 500, r.text
        assert "simulated DB error" in r.json()["detail"]
        # Critical: the FIRST row's INSERT was buffered into pending_writes,
        # but the exception aborts the with-block so _FakeConn.rollback()
        # fires and pending_writes is cleared. journal_rows stays empty.
        assert len(state["journal_rows"]) == 0
    finally:
        main._compute_portfolio_heat = lambda *a, **kw: _AUTO_HEAT


def test_snapshot_fired_on_fresh_insert(batch_stubs):
    """For brand-new rows, the auto-compute branches fire — the
    deterministic _AUTO_* values land in the row."""
    state, client = batch_stubs
    r = _post(client)
    assert r.status_code == 200
    for row in state["journal_rows"]:
        assert row["portfolio_heat"] == _AUTO_HEAT
        assert row["spy_atr"] == _AUTO_SPY_ATR
        assert row["nasdaq_atr"] == _AUTO_NDX_ATR
        assert row["market_cycle"] == _AUTO_CYCLE
        assert row["mct_display_day_num"] == _AUTO_DAY_NUM


def test_single_portfolio_degenerate_case(batch_stubs):
    """N=1 works identically to a single /api/journal/edit call.
    Writes exactly 1 row; snapshot fields auto-computed."""
    state, client = batch_stubs
    r = _post(client, portfolios=[_pf("CanSlim", end_nlv=12500.0)])
    assert r.status_code == 200
    assert r.json()["rows_written"] == 1
    assert len(state["journal_rows"]) == 1
    assert state["journal_rows"][0]["end_nlv"] == 12500.0
    assert state["journal_rows"][0]["portfolio_heat"] == _AUTO_HEAT


def test_user_id_propagation(batch_stubs):
    """Every inserted row carries the portfolio's user_id explicitly.
    Matches the founder UUID from the portfolios SELECT."""
    state, client = batch_stubs
    r = _post(client)
    assert r.status_code == 200
    for row in state["journal_rows"]:
        assert row["user_id"] == _FOUNDER_UUID


def test_beg_nlv_chain_is_per_portfolio(batch_stubs):
    """Each portfolio's beg_nlv resolves from its OWN prior end_nlv,
    not shared across portfolios. Pre-seed different prior-day rows
    per portfolio and verify each row's beg_nlv matches."""
    state, client = batch_stubs
    state["journal_rows"].extend([
        {"id": 10, "portfolio_id": 1, "day": "2026-05-21",
         "end_nlv": 99999.99, "portfolio_heat": 0},
        {"id": 11, "portfolio_id": 3, "day": "2026-05-21",
         "end_nlv": 12345.67, "portfolio_heat": 0},
        {"id": 12, "portfolio_id": 2, "day": "2026-05-21",
         "end_nlv": 46006.79, "portfolio_heat": 0},
    ])
    r = _post(client)
    assert r.status_code == 200

    new_rows = [r for r in state["journal_rows"] if r["day"] == "2026-05-22"]
    assert len(new_rows) == 3
    by_pid = {r["portfolio_id"]: r for r in new_rows}
    assert by_pid[1]["beg_nlv"] == 99999.99   # CanSlim
    assert by_pid[3]["beg_nlv"] == 12345.67   # 457B Plan
    assert by_pid[2]["beg_nlv"] == 46006.79   # Long-Term Growth
