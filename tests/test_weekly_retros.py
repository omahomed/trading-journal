"""Tests for the Phase 0 weekly retros server persistence (Migration 025).

Covers two surfaces:

  1. db_layer helpers (fake-cursor unit tests, no live DB) — mirroring the
     pattern in test_save_summary_defensive.py:
       - upsert INSERTs when no existing row
       - upsert UPDATEs the existing live row (last-write-wins)
       - upsert REVIVES a soft-deleted row (sets deleted_at = NULL) instead
         of inserting a duplicate — preserves any tags attached to the
         original id
       - _replace_weekly_retro_ticker_grades does DELETE-then-INSERT and
         skips fully-empty rows
       - soft_delete_weekly_retro UPDATEs deleted_at (does not DELETE)

  2. API endpoints (FastAPI TestClient with stubbed db_layer) — mirroring
     test_strategies_admin.py / test_journal_backfill_preservation.py:
       - GET returns 200 with the row
       - GET missing returns {"error": "not_found"} (frontend treats as
         fresh blank, no error UI)
       - PUT bad week_start surfaces a clear error
       - PUT non-Monday week_start (DB CHECK violation) returns a typed
         error message
       - PUT bad week_grade vocab returns a typed error
       - DELETE returns ok / not_found
       - List endpoint returns the array

Constraint behavior that requires a live Postgres (the Monday CHECK firing,
the partial-unique index allowing recycle, RLS isolation across user_ids)
is enforced by the migration at apply-time and exercised in manual smoke
tests post-deploy — same coverage gap as every other DB-side constraint in
this codebase.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

import db_layer


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "11111111-2222-3333-4444-555555555555"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Fake cursor scaffolding (mirrors test_save_summary_defensive.py)
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Records every execute() call. fetchone()/fetchall() return from queues."""
    def __init__(self, fetchones=None, fetchalls=None):
        self._fetchones = list(fetchones or [])
        self._fetchalls = list(fetchalls or [])
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, tuple(params) if params else ()))

    def executemany(self, sql, seq_of_params):
        self.executed.append((sql, tuple(seq_of_params)))

    def fetchone(self):
        if self._fetchones:
            return self._fetchones.pop(0)
        return None

    def fetchall(self):
        if self._fetchalls:
            return self._fetchalls.pop(0)
        return []

    def close(self):
        pass

    @property
    def rowcount(self):
        return 1


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


def _patch_conn(monkeypatch, cursor):
    """Helper: wire db_layer.get_db_connection to return a fake conn."""
    conn = _FakeConn(cursor)
    monkeypatch.setattr(db_layer, "get_db_connection", lambda: _ConnCM(conn))
    return conn


def _execute_values_capturing(captured):
    """Replacement for psycopg2.extras.execute_values that records the
    inserted rows in `captured` so tests can assert on them. Signature
    matches the real function: (cur, sql, argslist, template=None, ...)."""
    def fake_execute_values(cur, sql, argslist, *args, **kwargs):
        captured.append((sql, list(argslist)))
    return fake_execute_values


# ---------------------------------------------------------------------------
# db_layer.upsert_weekly_retro — INSERT path
# ---------------------------------------------------------------------------


def test_upsert_inserts_when_no_existing_row(monkeypatch):
    """No row exists for (portfolio_id, week_start) → INSERT path. The
    helper emits one SELECT for the portfolio, one SELECT for the existing
    row (returns None), one INSERT, and a final SELECT for the round-trip."""
    cur = _FakeCursor(fetchones=[
        {"id": 42},                                # portfolio lookup
        None,                                       # no existing retro
        {"id": 7},                                  # INSERT returning id
        {  # final re-fetch
            "id": 7, "portfolio": "CanSlim",
            "week_start": date(2026, 5, 4),
            "week_grade": "B+", "best_decision": "good", "worst_decision": "bad",
            "rule_change": False, "rule_change_text": "",
            "weekly_thoughts": "<p>fresh notes</p>",
            "created_at": datetime(2026, 5, 5),
            "updated_at": datetime(2026, 5, 5),
        },
    ], fetchalls=[[]])  # no ticker_grades children
    _patch_conn(monkeypatch, cur)

    result = db_layer.upsert_weekly_retro(
        "CanSlim", date(2026, 5, 4),
        week_grade="B+", best_decision="good", worst_decision="bad",
        weekly_thoughts="<p>fresh notes</p>",
    )

    assert result["id"] == 7
    assert result["week_grade"] == "B+"
    assert result["weekly_thoughts"] == "<p>fresh notes</p>"
    assert result["ticker_grades"] == {}
    # Verify an INSERT was emitted (not an UPDATE)
    statements = [sql for sql, _ in cur.executed]
    assert any("INSERT INTO weekly_retros" in s for s in statements)
    assert not any("UPDATE weekly_retros" in s for s in statements)
    # Phase 3: weekly_thoughts must be in the INSERT column list + values.
    insert_sql = next(s for s, _ in cur.executed if "INSERT INTO weekly_retros" in s)
    assert "weekly_thoughts" in insert_sql


# ---------------------------------------------------------------------------
# db_layer.upsert_weekly_retro — UPDATE path (live row exists)
# ---------------------------------------------------------------------------


def test_upsert_updates_when_live_row_exists(monkeypatch):
    """A live row exists → UPDATE path. Last-write-wins; no version check."""
    cur = _FakeCursor(fetchones=[
        {"id": 42},                                 # portfolio
        {"id": 7},                                  # existing live row
        {"id": 7},                                  # UPDATE returning
        {  # final re-fetch
            "id": 7, "portfolio": "CanSlim",
            "week_start": date(2026, 5, 4),
            "week_grade": "A", "best_decision": "x", "worst_decision": "y",
            "rule_change": True, "rule_change_text": "no FOMO",
            "weekly_thoughts": "<p>updated</p>",
            "created_at": datetime(2026, 5, 1),
            "updated_at": datetime(2026, 5, 6),
        },
    ], fetchalls=[[]])
    _patch_conn(monkeypatch, cur)

    result = db_layer.upsert_weekly_retro(
        "CanSlim", date(2026, 5, 4),
        week_grade="A", rule_change=True, rule_change_text="no FOMO",
        weekly_thoughts="<p>updated</p>",
    )

    assert result["id"] == 7
    assert result["week_grade"] == "A"
    assert result["rule_change"] is True
    assert result["weekly_thoughts"] == "<p>updated</p>"
    statements = [sql for sql, _ in cur.executed]
    assert any("UPDATE weekly_retros" in s for s in statements)
    assert not any("INSERT INTO weekly_retros" in s for s in statements)
    # Phase 3: weekly_thoughts must be in the UPDATE SET clause.
    update_sql = next(s for s, _ in cur.executed if "UPDATE weekly_retros" in s)
    assert "weekly_thoughts = %s" in update_sql


# ---------------------------------------------------------------------------
# db_layer.upsert_weekly_retro — REVIVAL path (soft-deleted row exists)
# ---------------------------------------------------------------------------


def test_upsert_revives_soft_deleted_row(monkeypatch):
    """A soft-deleted row exists for the same (portfolio_id, week_start) →
    the helper finds it via the live-OR-deleted SELECT and UPDATEs in
    place, setting deleted_at = NULL. NO new row is inserted, so any
    Phase 1 tags pointing at the original id survive."""
    cur = _FakeCursor(fetchones=[
        {"id": 42},                                 # portfolio
        {"id": 99},                                 # soft-deleted row found
        {"id": 99},                                 # UPDATE returning same id
        {  # re-fetch shows revived row
            "id": 99, "portfolio": "CanSlim",
            "week_start": date(2026, 5, 4),
            "week_grade": "B", "best_decision": "", "worst_decision": "",
            "rule_change": False, "rule_change_text": "",
            "weekly_thoughts": "<p>revived</p>",
            "created_at": datetime(2026, 4, 1),
            "updated_at": datetime(2026, 5, 13),
        },
    ], fetchalls=[[]])
    _patch_conn(monkeypatch, cur)

    result = db_layer.upsert_weekly_retro(
        "CanSlim", date(2026, 5, 4), week_grade="B",
        weekly_thoughts="<p>revived</p>",
    )

    # Same id — id stability is the whole point of revival.
    assert result["id"] == 99
    # Phase 3: the revived row reflects the new payload's weekly_thoughts.
    assert result["weekly_thoughts"] == "<p>revived</p>"
    statements = [sql for sql, _ in cur.executed]
    update_sql = next(s for s, _ in cur.executed if "UPDATE weekly_retros" in s)
    # The UPDATE must clear deleted_at so the partial-unique index sees a
    # live row again. Without this, tags lose their FK on the next prune.
    assert "deleted_at = NULL" in update_sql
    # Phase 3: the revival UPDATE must also write the new weekly_thoughts.
    assert "weekly_thoughts = %s" in update_sql
    assert not any("INSERT INTO weekly_retros" in s for s in statements)


# ---------------------------------------------------------------------------
# db_layer._replace_weekly_retro_ticker_grades — child replace semantics
# ---------------------------------------------------------------------------


def test_replace_ticker_grades_deletes_then_inserts(monkeypatch):
    """Mirrors the lot_closures recompute pattern: helper deletes every
    existing child row for the parent then re-inserts the current set."""
    cur = _FakeCursor()
    captured: list = []
    monkeypatch.setattr(db_layer, "execute_values",
                        _execute_values_capturing(captured))

    db_layer._replace_weekly_retro_ticker_grades(cur, 7, {
        "AAPL": {"grade": "A (Perfect)", "behavior": "Followed Plan", "notes": "ok"},
        "TSLA": {"grade": "C (Sloppy)", "behavior": "FOMO Entry", "notes": ""},
    })

    delete_calls = [s for s, _ in cur.executed if "DELETE FROM weekly_retro_ticker_grades" in s]
    assert len(delete_calls) == 1
    # One bulk INSERT via execute_values; two rows.
    assert len(captured) == 1
    _, rows = captured[0]
    assert len(rows) == 2
    tickers = sorted(r[1] for r in rows)
    assert tickers == ["AAPL", "TSLA"]


def test_replace_ticker_grades_skips_fully_empty_entries(monkeypatch):
    """A ticker with all three fields blank shouldn't write a row — that's
    the legacy 'opened the picker, didn't grade' shape from the UI."""
    cur = _FakeCursor()
    captured: list = []
    monkeypatch.setattr(db_layer, "execute_values",
                        _execute_values_capturing(captured))

    db_layer._replace_weekly_retro_ticker_grades(cur, 7, {
        "AAPL": {"grade": "", "behavior": "", "notes": ""},
        "TSLA": {"grade": "A (Perfect)", "behavior": "", "notes": ""},
    })

    # Only TSLA (has a grade) should land in the bulk insert.
    assert len(captured) == 1
    _, rows = captured[0]
    assert len(rows) == 1
    assert rows[0][1] == "TSLA"


def test_replace_ticker_grades_with_empty_map_only_deletes(monkeypatch):
    """An empty ticker_grades map → DELETE the existing children, no INSERT.
    This is the path "user cleared all per-ticker grades" hits."""
    cur = _FakeCursor()
    captured: list = []
    monkeypatch.setattr(db_layer, "execute_values",
                        _execute_values_capturing(captured))

    db_layer._replace_weekly_retro_ticker_grades(cur, 7, {})

    assert any("DELETE FROM weekly_retro_ticker_grades" in s for s, _ in cur.executed)
    assert captured == []


# ---------------------------------------------------------------------------
# db_layer.upsert_weekly_retro — validation
# ---------------------------------------------------------------------------


def test_upsert_rejects_unknown_portfolio(monkeypatch):
    """Unknown portfolio → ValueError before any write attempt."""
    cur = _FakeCursor(fetchones=[None])  # portfolio lookup misses
    _patch_conn(monkeypatch, cur)

    with pytest.raises(ValueError, match="Portfolio 'Bogus' not found"):
        db_layer.upsert_weekly_retro("Bogus", date(2026, 5, 4))


def test_upsert_rejects_invalid_week_grade(monkeypatch):
    """A week_grade outside the closed vocab → ValueError up-front, no DB
    call. Defense in depth vs. the DB CHECK constraint."""
    # No need to wire a cursor — validation fires before the DB hit.
    with pytest.raises(ValueError, match="Invalid week_grade"):
        db_layer.upsert_weekly_retro(
            "CanSlim", date(2026, 5, 4), week_grade="Z+",
        )


# ---------------------------------------------------------------------------
# db_layer.soft_delete_weekly_retro
# ---------------------------------------------------------------------------


def test_soft_delete_sets_deleted_at_not_hard_delete(monkeypatch):
    """soft_delete must UPDATE deleted_at, never DELETE."""
    cur = _FakeCursor(fetchones=[{"id": 7}])
    _patch_conn(monkeypatch, cur)

    ok = db_layer.soft_delete_weekly_retro(7)
    assert ok is True

    statements = [s for s, _ in cur.executed]
    assert any("UPDATE weekly_retros SET deleted_at = NOW()" in s for s in statements)
    assert not any("DELETE FROM weekly_retros" in s for s in statements)


def test_soft_delete_returns_false_when_no_row(monkeypatch):
    """Idempotent: deleting a non-existent or already-deleted row returns False."""
    cur = _FakeCursor(fetchones=[None])
    _patch_conn(monkeypatch, cur)

    ok = db_layer.soft_delete_weekly_retro(999)
    assert ok is False


# ---------------------------------------------------------------------------
# Endpoint tests — FastAPI TestClient + stubbed db_layer
# ---------------------------------------------------------------------------


@pytest.fixture
def weekly_stubs(monkeypatch):
    """Yield (state, client). state knobs control db_layer return values,
    state observations capture call args. Pattern mirrors test_strategies_admin."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "load_result": None,
        "list_result": [],
        "upsert_result": None,
        "upsert_raises": None,
        "delete_result": True,
        "load_calls": [],
        "list_calls": [],
        "upsert_calls": [],
        "delete_calls": [],
    }

    def fake_load(portfolio, week_start):
        state["load_calls"].append((portfolio, week_start))
        return state["load_result"]
    monkeypatch.setattr(db_layer, "load_weekly_retro", fake_load)

    def fake_list(portfolio, limit=200):
        state["list_calls"].append((portfolio, limit))
        return state["list_result"]
    monkeypatch.setattr(db_layer, "list_weekly_retros", fake_list)

    def fake_upsert(portfolio, week_start, **fields):
        state["upsert_calls"].append((portfolio, week_start, dict(fields)))
        if state["upsert_raises"] is not None:
            raise state["upsert_raises"]
        if state["upsert_result"] is not None:
            return state["upsert_result"]
        # Default: echo back a minimal row
        return {
            "id": 1, "portfolio": portfolio, "week_start": week_start.isoformat(),
            "week_grade": fields.get("week_grade"), "best_decision": fields.get("best_decision", ""),
            "worst_decision": fields.get("worst_decision", ""), "rule_change": fields.get("rule_change", False),
            "rule_change_text": fields.get("rule_change_text", ""),
            "weekly_thoughts": fields.get("weekly_thoughts", ""),
            "ticker_grades": fields.get("ticker_grades") or {},
            "created_at": "2026-05-13T00:00:00", "updated_at": "2026-05-13T00:00:00",
        }
    monkeypatch.setattr(db_layer, "upsert_weekly_retro", fake_upsert)

    def fake_delete(retro_id):
        state["delete_calls"].append(retro_id)
        return state["delete_result"]
    monkeypatch.setattr(db_layer, "soft_delete_weekly_retro", fake_delete)

    if hasattr(main.limiter, "enabled"):
        original = main.limiter.enabled
        main.limiter.enabled = False
    else:
        original = None

    client = TestClient(main.app, headers=_auth_headers())
    try:
        yield state, client
    finally:
        if original is not None:
            main.limiter.enabled = original


def test_get_returns_row(weekly_stubs):
    """Happy path: row exists → server returns the dict directly."""
    state, client = weekly_stubs
    state["load_result"] = {
        "id": 7, "portfolio": "CanSlim", "week_start": "2026-05-04",
        "week_grade": "B+", "best_decision": "x", "worst_decision": "y",
        "rule_change": False, "rule_change_text": "",
        "weekly_thoughts": "<p>my reflections</p>",
        "ticker_grades": {"AAPL": {"grade": "A (Perfect)", "behavior": "Followed Plan", "notes": ""}},
        "created_at": "2026-05-05T00:00:00", "updated_at": "2026-05-05T00:00:00",
    }

    r = client.get("/api/weekly-retros?portfolio=CanSlim&week_start=2026-05-04")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 7
    assert body["week_grade"] == "B+"
    assert body["weekly_thoughts"] == "<p>my reflections</p>"
    assert body["ticker_grades"]["AAPL"]["grade"] == "A (Perfect)"


def test_get_missing_returns_not_found_body(weekly_stubs):
    """Missing row → HTTP 200 with {"error": "not_found"}. The frontend
    treats this as a fresh blank retro — NOT an error UI state."""
    state, client = weekly_stubs
    state["load_result"] = None

    r = client.get("/api/weekly-retros?portfolio=CanSlim&week_start=2026-05-04")
    assert r.status_code == 200
    assert r.json() == {"error": "not_found"}


def test_get_bad_week_start_format(weekly_stubs):
    """Unparseable week_start → typed error message, no DB call."""
    state, client = weekly_stubs
    r = client.get("/api/weekly-retros?portfolio=CanSlim&week_start=not-a-date")
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert "bad week_start" in body["error"]
    assert state["load_calls"] == []


def test_list_returns_envelope_shape(weekly_stubs, monkeypatch):
    """Phase 6 — list endpoint now returns {weeks, ytd_stats} (no longer
    a bare array). The Review History tab that consumed the array shape
    is gone, so this is a coordinated cutover. Detailed shape tests for
    the envelope live in test_pinned_entities.py; this is a smoke test
    that the route still wires through db.list_weekly_retros_rail."""
    state, client = weekly_stubs
    monkeypatch.setattr(
        db_layer, "list_weekly_retros_rail",
        lambda _: {
            "weeks": [
                {"id": 2, "key": "2026-05-11", "week_start": "2026-05-11",
                 "week_end": "2026-05-15", "year": 2026, "month": 5,
                 "title": "May 11 – May 15", "has_content": True,
                 "pinned": False, "sparkline_value": 4.62, "week_grade": "A"},
            ],
            "ytd_stats": {"total_weeks": 1, "weeks_graded": 1,
                          "avg_grade": "A", "weeks_pinned": 0},
        },
    )
    r = client.get("/api/weekly-retros/list?portfolio=CanSlim")
    assert r.status_code == 200
    body = r.json()
    assert "weeks" in body and "ytd_stats" in body
    assert body["weeks"][0]["week_start"] == "2026-05-11"
    assert body["ytd_stats"]["avg_grade"] == "A"


def test_put_happy_path(weekly_stubs):
    """PUT forwards the body to db.upsert_weekly_retro and returns the row."""
    state, client = weekly_stubs

    r = client.put("/api/weekly-retros", json={
        "portfolio": "CanSlim",
        "week_start": "2026-05-04",
        "week_grade": "B+",
        "best_decision": "good",
        "worst_decision": "bad",
        "rule_change": True,
        "rule_change_text": "rule",
        "weekly_thoughts": "<p>this week was great</p>",
        "ticker_grades": {
            "AAPL": {"grade": "A (Perfect)", "behavior": "Followed Plan", "notes": ""},
        },
    })
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 1
    assert body["week_grade"] == "B+"
    # Helper was called with parsed date and keyword fields.
    assert len(state["upsert_calls"]) == 1
    portfolio, week_start, fields = state["upsert_calls"][0]
    assert portfolio == "CanSlim"
    assert week_start == date(2026, 5, 4)
    assert fields["week_grade"] == "B+"
    # Phase 3: weekly_thoughts forwarded to the helper.
    assert fields["weekly_thoughts"] == "<p>this week was great</p>"
    assert fields["ticker_grades"]["AAPL"]["behavior"] == "Followed Plan"


def test_put_omits_weekly_thoughts_defaults_to_empty(weekly_stubs):
    """A PUT body without weekly_thoughts → helper is called with ''.
    Mirrors the column's NOT NULL DEFAULT '' semantics from migration 027."""
    state, client = weekly_stubs

    r = client.put("/api/weekly-retros", json={
        "portfolio": "CanSlim",
        "week_start": "2026-05-04",
        "week_grade": "B+",
    })
    assert r.status_code == 200
    _, _, fields = state["upsert_calls"][0]
    assert fields["weekly_thoughts"] == ""


def test_put_bad_week_start(weekly_stubs):
    """Bad date string → typed error, no DB call."""
    state, client = weekly_stubs

    r = client.put("/api/weekly-retros", json={
        "portfolio": "CanSlim", "week_start": "garbage",
    })
    assert r.status_code == 200
    body = r.json()
    assert "error" in body and "bad week_start" in body["error"]
    assert state["upsert_calls"] == []


def test_put_db_check_violation_monday(weekly_stubs):
    """If the DB Monday CHECK fires (frontend should never send a non-Monday
    but we belt-and-braces it), the helper raises and we surface a clear msg."""
    import psycopg2
    state, client = weekly_stubs
    state["upsert_raises"] = psycopg2.errors.CheckViolation(
        'new row for relation "weekly_retros" violates check constraint '
        '"weekly_retros_week_start_monday"'
    )

    r = client.put("/api/weekly-retros", json={
        "portfolio": "CanSlim", "week_start": "2026-05-04",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["error"] == "week_start must be a Monday"


def test_put_db_check_violation_grade(weekly_stubs):
    """Same shape for the week_grade vocab CHECK firing at the DB level."""
    import psycopg2
    state, client = weekly_stubs
    state["upsert_raises"] = psycopg2.errors.CheckViolation(
        'violates check constraint "weekly_retros_week_grade_vocab"'
    )

    r = client.put("/api/weekly-retros", json={
        "portfolio": "CanSlim", "week_start": "2026-05-04",
    })
    assert r.status_code == 200
    assert r.json()["error"] == "invalid week_grade"


def test_put_ticker_grades_must_be_object(weekly_stubs):
    """A list or string for ticker_grades is a client bug — reject early
    rather than blow up in the helper with a confusing TypeError."""
    state, client = weekly_stubs
    r = client.put("/api/weekly-retros", json={
        "portfolio": "CanSlim", "week_start": "2026-05-04",
        "ticker_grades": ["AAPL", "TSLA"],
    })
    assert r.status_code == 200
    assert "must be an object" in r.json()["error"]
    assert state["upsert_calls"] == []


def test_delete_happy_path(weekly_stubs):
    """DELETE returns ok with the id."""
    state, client = weekly_stubs
    state["delete_result"] = True

    r = client.delete("/api/weekly-retros/7")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "id": 7}
    assert state["delete_calls"] == [7]


def test_delete_not_found(weekly_stubs):
    """Idempotent delete of an unknown id → not_found error body."""
    state, client = weekly_stubs
    state["delete_result"] = False

    r = client.delete("/api/weekly-retros/999")
    assert r.status_code == 200
    assert r.json() == {"error": "not_found"}


# ---------------------------------------------------------------------------
# Phase 4.6 — Close the Week (axis grading + reviewed lock)
# ---------------------------------------------------------------------------


def test_derive_overall_grade_averages_three_axes():
    """A/B/A averages to (4.0 + 3.0 + 4.0)/3 = 3.67 → A- per the
    _NUMERIC_BUCKETS table. Mirrors the design's VAL_GRADE in
    sections-bottom.jsx, which the frontend ports verbatim."""
    assert db_layer._derive_overall_grade("A", "B", "A") == "A-"


def test_derive_overall_grade_returns_none_when_any_axis_missing():
    """Frontend leaves week_grade as-is when not all 3 axes are set;
    backend mirrors by returning None. _derive_overall_grade is only
    invoked when callers know all 3 are non-null, but the helper is
    defensive on its own."""
    assert db_layer._derive_overall_grade("A", "B", None) is None
    assert db_layer._derive_overall_grade(None, "B", "A") is None
    assert db_layer._derive_overall_grade("", "B", "A") is None


def test_derive_overall_grade_collapses_d_plus_minus_to_d():
    """Pre-existing inconsistency in db_layer: _NUMERIC_BUCKETS has
    D+/D/D- granularity but the column vocab CHECK only allows plain D.
    The helper collapses to D so the value it returns is always
    vocab-clean. F/F/F → F (0.0 mean → F bucket). D/D/D → D."""
    assert db_layer._derive_overall_grade("D", "D", "D") == "D"
    assert db_layer._derive_overall_grade("F", "F", "F") == "F"


def test_upsert_overwrites_week_grade_when_override_false_and_axes_set(monkeypatch):
    """Server authority: when overall_override is False AND every axis is
    non-null, server derives week_grade and overwrites the client value.
    Prevents the client poisoning the canonical overall."""
    cur = _FakeCursor(fetchones=[
        {"id": 42},                                 # portfolio
        None,                                       # no existing retro
        {"id": 7},                                  # INSERT returning
        {  # final re-fetch
            "id": 7, "portfolio": "CanSlim",
            "week_start": date(2026, 5, 4),
            # The server-derived overall (A/A/A → A) wins over the
            # client-sent "C" below.
            "week_grade": "A",
            "execution_grade": "A", "process_grade": "A", "pnl_grade": "A",
            "overall_override": False, "reviewed_at": None,
            "best_decision": "", "worst_decision": "",
            "rule_change": False, "rule_change_text": "",
            "weekly_thoughts": "",
            "created_at": datetime(2026, 5, 5),
            "updated_at": datetime(2026, 5, 5),
        },
    ], fetchalls=[[]])
    _patch_conn(monkeypatch, cur)

    result = db_layer.upsert_weekly_retro(
        "CanSlim", date(2026, 5, 4),
        # Client lies — claims week_grade="C" but axes are all A.
        week_grade="C",
        execution_grade="A", process_grade="A", pnl_grade="A",
        overall_override=False,
    )

    # The persisted (and returned) week_grade is the derived A, not C.
    assert result["week_grade"] == "A"
    # The INSERT must have used the derived value.
    insert_sql, insert_params = next(
        (s, p) for s, p in cur.executed if "INSERT INTO weekly_retros" in s
    )
    # week_grade is the 3rd positional column (after portfolio_id, week_start).
    assert insert_params[2] == "A"


def test_upsert_trusts_client_week_grade_when_override_true(monkeypatch):
    """When overall_override is True, server uses the client-supplied
    week_grade verbatim. The user's intent is to deviate from the
    average; the server respects it."""
    cur = _FakeCursor(fetchones=[
        {"id": 42}, None, {"id": 7},
        {
            "id": 7, "portfolio": "CanSlim",
            "week_start": date(2026, 5, 4),
            "week_grade": "B",
            "execution_grade": "A", "process_grade": "A", "pnl_grade": "A",
            "overall_override": True, "reviewed_at": None,
            "best_decision": "", "worst_decision": "",
            "rule_change": False, "rule_change_text": "",
            "weekly_thoughts": "",
            "created_at": datetime(2026, 5, 5),
            "updated_at": datetime(2026, 5, 5),
        },
    ], fetchalls=[[]])
    _patch_conn(monkeypatch, cur)

    result = db_layer.upsert_weekly_retro(
        "CanSlim", date(2026, 5, 4),
        week_grade="B",
        execution_grade="A", process_grade="A", pnl_grade="A",
        overall_override=True,
    )

    assert result["week_grade"] == "B"
    insert_sql, insert_params = next(
        (s, p) for s, p in cur.executed if "INSERT INTO weekly_retros" in s
    )
    assert insert_params[2] == "B"  # client value preserved


def test_upsert_skips_derivation_when_any_axis_is_null(monkeypatch):
    """If any axis is null, server can't derive overall — preserves the
    client-supplied week_grade as-is. This covers the in-progress state
    (user has graded 2 of 3 axes)."""
    cur = _FakeCursor(fetchones=[
        {"id": 42}, None, {"id": 7},
        {
            "id": 7, "portfolio": "CanSlim",
            "week_start": date(2026, 5, 4),
            "week_grade": "C",
            "execution_grade": "A", "process_grade": "A", "pnl_grade": None,
            "overall_override": False, "reviewed_at": None,
            "best_decision": "", "worst_decision": "",
            "rule_change": False, "rule_change_text": "",
            "weekly_thoughts": "",
            "created_at": datetime(2026, 5, 5),
            "updated_at": datetime(2026, 5, 5),
        },
    ], fetchalls=[[]])
    _patch_conn(monkeypatch, cur)

    result = db_layer.upsert_weekly_retro(
        "CanSlim", date(2026, 5, 4),
        week_grade="C",
        execution_grade="A", process_grade="A", pnl_grade=None,
        overall_override=False,
    )

    # Axes incomplete → server doesn't derive → client value preserved.
    assert result["week_grade"] == "C"


def test_upsert_rejects_axis_grade_with_invalid_vocab(monkeypatch):
    """Up-front validation: every axis grade is validated against the
    closed vocab before any DB call. Defense in depth vs the column
    CHECK constraint."""
    with pytest.raises(ValueError, match="Invalid execution_grade"):
        db_layer.upsert_weekly_retro(
            "CanSlim", date(2026, 5, 4), execution_grade="Z+",
        )
    with pytest.raises(ValueError, match="Invalid process_grade"):
        db_layer.upsert_weekly_retro(
            "CanSlim", date(2026, 5, 4), process_grade="Z",
        )
    with pytest.raises(ValueError, match="Invalid pnl_grade"):
        db_layer.upsert_weekly_retro(
            "CanSlim", date(2026, 5, 4), pnl_grade="Z-",
        )


def test_upsert_rejects_axis_change_when_existing_row_is_reviewed(monkeypatch):
    """Lock validation: when the persisted row has reviewed_at set AND
    the incoming payload does NOT clear reviewed_at, any change to a
    graded field raises WeeklyRetroLockedError (→ 409 at the API
    layer). The frontend's disabled selectors are UX; this is the
    correctness backstop."""
    cur = _FakeCursor(fetchones=[
        {"id": 42},                                 # portfolio
        {  # existing row IS reviewed, with axes A/A/A
            "id": 7,
            "reviewed_at": datetime(2026, 5, 13, 12, 0, 0),
            "execution_grade": "A", "process_grade": "A", "pnl_grade": "A",
            "week_grade": "A", "overall_override": False,
        },
    ])
    _patch_conn(monkeypatch, cur)

    with pytest.raises(db_layer.WeeklyRetroLockedError, match="grade locked"):
        db_layer.upsert_weekly_retro(
            "CanSlim", date(2026, 5, 4),
            week_grade="B",
            execution_grade="B", process_grade="B", pnl_grade="B",
            overall_override=False,
            # Keeps reviewed_at set → lock applies.
            reviewed_at=datetime(2026, 5, 13, 12, 0, 0).isoformat(),
        )


def test_upsert_allows_axis_change_when_payload_clears_reviewed_at(monkeypatch):
    """Sibling of the lock test: when the same payload clears
    reviewed_at (un-review + edit in one call), grade changes are
    allowed. Lock check skips entirely."""
    cur = _FakeCursor(fetchones=[
        {"id": 42},                                 # portfolio
        {  # existing IS reviewed with axes A/A/A
            "id": 7,
            "reviewed_at": datetime(2026, 5, 13, 12, 0, 0),
            "execution_grade": "A", "process_grade": "A", "pnl_grade": "A",
            "week_grade": "A", "overall_override": False,
        },
        {"id": 7},                                  # UPDATE returning
        {  # final re-fetch (now un-reviewed, axes B/B/B)
            "id": 7, "portfolio": "CanSlim",
            "week_start": date(2026, 5, 4),
            "week_grade": "B",
            "execution_grade": "B", "process_grade": "B", "pnl_grade": "B",
            "overall_override": False, "reviewed_at": None,
            "best_decision": "", "worst_decision": "",
            "rule_change": False, "rule_change_text": "",
            "weekly_thoughts": "",
            "created_at": datetime(2026, 5, 5),
            "updated_at": datetime(2026, 5, 14),
        },
    ], fetchalls=[[]])
    _patch_conn(monkeypatch, cur)

    # Same payload as the lock test, EXCEPT reviewed_at=None.
    result = db_layer.upsert_weekly_retro(
        "CanSlim", date(2026, 5, 4),
        week_grade="B",
        execution_grade="B", process_grade="B", pnl_grade="B",
        overall_override=False,
        reviewed_at=None,
    )
    assert result["execution_grade"] == "B"
    assert result["reviewed_at"] is None


def test_upsert_allows_non_grade_changes_when_reviewed(monkeypatch):
    """Lock applies to graded fields only. Reflections, rule change,
    and ticker grades remain editable on a reviewed retro. The lock
    check finds no graded diff → no error → UPDATE proceeds."""
    cur = _FakeCursor(fetchones=[
        {"id": 42},                                 # portfolio
        {  # existing IS reviewed with axes A/A/A and reflections empty
            "id": 7,
            "reviewed_at": datetime(2026, 5, 13, 12, 0, 0),
            "execution_grade": "A", "process_grade": "A", "pnl_grade": "A",
            "week_grade": "A", "overall_override": False,
        },
        {"id": 7},                                  # UPDATE returning
        {  # final re-fetch
            "id": 7, "portfolio": "CanSlim",
            "week_start": date(2026, 5, 4),
            "week_grade": "A",
            "execution_grade": "A", "process_grade": "A", "pnl_grade": "A",
            "overall_override": False,
            "reviewed_at": datetime(2026, 5, 13, 12, 0, 0),
            "best_decision": "new reflection", "worst_decision": "",
            "rule_change": False, "rule_change_text": "",
            "weekly_thoughts": "",
            "created_at": datetime(2026, 5, 5),
            "updated_at": datetime(2026, 5, 14),
        },
    ], fetchalls=[[]])
    _patch_conn(monkeypatch, cur)

    result = db_layer.upsert_weekly_retro(
        "CanSlim", date(2026, 5, 4),
        # Unchanged grades.
        week_grade="A",
        execution_grade="A", process_grade="A", pnl_grade="A",
        overall_override=False,
        reviewed_at=datetime(2026, 5, 13, 12, 0, 0).isoformat(),
        # Only reflections changed.
        best_decision="new reflection",
    )

    assert result["best_decision"] == "new reflection"
    assert result["reviewed_at"] is not None


def test_put_axis_grades_flow_through_to_upsert(weekly_stubs):
    """API contract: PUT body forwards the 5 new fields to
    upsert_weekly_retro. Smoke that the body parser doesn't drop them."""
    state, client = weekly_stubs
    r = client.put("/api/weekly-retros", json={
        "portfolio": "CanSlim", "week_start": "2026-05-04",
        "execution_grade": "A-", "process_grade": "B+", "pnl_grade": "A",
        "overall_override": True,
        "week_grade": "A-",
        "reviewed_at": "2026-05-14T10:00:00+00:00",
    })
    assert r.status_code == 200
    assert len(state["upsert_calls"]) == 1
    portfolio, ws, fields = state["upsert_calls"][0]
    assert fields["execution_grade"] == "A-"
    assert fields["process_grade"] == "B+"
    assert fields["pnl_grade"] == "A"
    assert fields["overall_override"] is True
    assert fields["reviewed_at"] == "2026-05-14T10:00:00+00:00"


def test_put_returns_409_when_upsert_raises_locked_error(weekly_stubs):
    """The WeeklyRetroLockedError raised by db_layer when a reviewed row
    is mutated propagates as HTTP 409 with the detail message. Mirrors
    the existing /api/pins/toggle 409 idiom for state conflicts."""
    state, client = weekly_stubs
    state["upsert_raises"] = db_layer.WeeklyRetroLockedError(
        "grade locked; un-review to edit"
    )
    r = client.put("/api/weekly-retros", json={
        "portfolio": "CanSlim", "week_start": "2026-05-04",
        "execution_grade": "B",
    })
    assert r.status_code == 409
    assert r.json()["detail"] == "grade locked; un-review to edit"


def test_legacy_row_serializes_with_null_axes_and_reviewed(monkeypatch):
    """Pre-Phase-4.6 rows have axis columns NULL, overall_override False
    (column default), reviewed_at NULL. _serialize_weekly_retro must
    pass these through cleanly so the frontend's hydration sees a clean
    'overall set, axes never entered' state."""
    parent = {
        "id": 7, "portfolio": "CanSlim",
        "week_start": date(2026, 5, 4),
        "week_grade": "B+",
        "execution_grade": None, "process_grade": None, "pnl_grade": None,
        "overall_override": False, "reviewed_at": None,
        "best_decision": "ok", "worst_decision": "",
        "rule_change": False, "rule_change_text": "",
        "weekly_thoughts": "",
        "created_at": datetime(2026, 5, 5),
        "updated_at": datetime(2026, 5, 5),
    }
    out = db_layer._serialize_weekly_retro(parent, {})

    assert out["week_grade"] == "B+"
    assert out["execution_grade"] is None
    assert out["process_grade"] is None
    assert out["pnl_grade"] is None
    assert out["overall_override"] is False
    assert out["reviewed_at"] is None
