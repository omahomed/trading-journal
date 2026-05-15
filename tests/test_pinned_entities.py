"""Tests for the Phase 6 pinned_entities surface (Migration 029).

Two surfaces, mirroring test_tag_system.py:

  1. db_layer helpers — fake-cursor unit tests of the SQL toggle_pin
     emits per branch (insert / revive / soft-delete), plus the
     avg_grade_from_letters letter→numeric→letter bucketing.

  2. API endpoints — FastAPI TestClient with stubbed db_layer. Covers
     the POST /api/pins/toggle contract and the request-body validation.

Constraint behavior that needs a live Postgres (partial-unique blocking
duplicate live pins, RLS isolation) is enforced by the migration and
exercised in post-deploy smoke tests — same coverage gap convention as
every other DB-side constraint in this codebase.
"""
from __future__ import annotations

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
# Fake cursor scaffolding (matches the test_weekly_retros / test_tag_system
# convention — same shape, same fetchone/fetchall queue API)
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, fetchones=None, fetchalls=None):
        self._fetchones = list(fetchones or [])
        self._fetchalls = list(fetchalls or [])
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, tuple(params) if params else ()))

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
    conn = _FakeConn(cursor)
    monkeypatch.setattr(db_layer, "get_db_connection", lambda: _ConnCM(conn))
    return conn


# ===========================================================================
# db_layer.toggle_pin
# ===========================================================================


def test_toggle_pin_inserts_when_no_existing_row(monkeypatch):
    """No existing row → INSERT a new pin, return True (now pinned)."""
    cur = _FakeCursor(fetchones=[None])  # SELECT returns nothing
    _patch_conn(monkeypatch, cur)

    pinned = db_layer.toggle_pin("weekly_retro", 42)
    assert pinned is True
    assert any("INSERT INTO pinned_entities" in s for s, _ in cur.executed)


def test_toggle_pin_soft_deletes_live_row(monkeypatch):
    """Live row exists → UPDATE deleted_at = NOW(). Return False (unpinned)."""
    cur = _FakeCursor(fetchones=[{"id": 7, "deleted_at": None}])
    _patch_conn(monkeypatch, cur)

    pinned = db_layer.toggle_pin("weekly_retro", 42)
    assert pinned is False
    update_sqls = [s for s, _ in cur.executed if "UPDATE pinned_entities" in s]
    assert any("deleted_at = NOW()" in s for s in update_sqls)
    assert all("INSERT" not in s for s, _ in cur.executed)


def test_toggle_pin_revives_soft_deleted_row(monkeypatch):
    """Soft-deleted row exists → UPDATE deleted_at = NULL (revive). Return
    True. Reuses the same row id, NOT inserts a new one. Mirrors the
    create_tag_assignment idempotent-restore idiom."""
    cur = _FakeCursor(fetchones=[{"id": 7, "deleted_at": "2026-05-12T12:00:00"}])
    _patch_conn(monkeypatch, cur)

    pinned = db_layer.toggle_pin("weekly_retro", 42)
    assert pinned is True
    update_sqls = [s for s, _ in cur.executed if "UPDATE pinned_entities" in s]
    assert any("deleted_at = NULL" in s for s in update_sqls)
    assert all("INSERT" not in s for s, _ in cur.executed)


def test_toggle_pin_double_call_returns_to_original(monkeypatch):
    """Toggle twice in sequence ends at the original state. Combines the
    SELECT-then-INSERT branch with the SELECT-then-soft-delete branch.

    Each toggle gets its own (mocked) connection because db_layer doesn't
    reuse cursors across `with get_db_connection()` blocks.
    """
    # First toggle: no existing row → INSERT, return True
    cur1 = _FakeCursor(fetchones=[None])
    conn1 = _FakeConn(cur1)
    # Second toggle: row exists live → UPDATE soft-delete, return False
    cur2 = _FakeCursor(fetchones=[{"id": 7, "deleted_at": None}])
    conn2 = _FakeConn(cur2)
    calls = iter([_ConnCM(conn1), _ConnCM(conn2)])
    monkeypatch.setattr(db_layer, "get_db_connection", lambda: next(calls))

    first = db_layer.toggle_pin("weekly_retro", 42)
    second = db_layer.toggle_pin("weekly_retro", 42)
    assert first is True
    assert second is False


def test_toggle_pin_rejects_unknown_entity_type(monkeypatch):
    """Defense in depth vs. the CHECK constraint. Raises ValueError before
    touching the DB so callers get a typed error, not an IntegrityError."""
    with pytest.raises(ValueError, match="Unknown entity_type"):
        db_layer.toggle_pin("bogus", 42)


def test_toggle_pin_accepts_both_entity_types(monkeypatch):
    """Both weekly_retro and daily_journal are valid per the CHECK + the
    Python tuple. Phase 6 only mounts weekly_retro; Phase 7 will use
    daily_journal without a schema change."""
    cur = _FakeCursor(fetchones=[None, None])
    _patch_conn(monkeypatch, cur)
    assert db_layer.toggle_pin("weekly_retro", 1) is True
    # Same connection's cursor pops the second fetchone — both succeed.
    cur._fetchones.append(None)
    _patch_conn(monkeypatch, cur)
    assert db_layer.toggle_pin("daily_journal", 1) is True


# ===========================================================================
# db_layer.list_pinned_entity_ids
# ===========================================================================


def test_list_pinned_entity_ids_returns_set_of_live_ids(monkeypatch):
    """Returns the set of live entity_ids for the type. Tuple-fetched
    cursor (no RealDictCursor) — the SQL selects a single column."""
    cur = _FakeCursor(fetchalls=[[(7,), (12,), (99,)]])
    _patch_conn(monkeypatch, cur)

    out = db_layer.list_pinned_entity_ids("weekly_retro")
    assert out == {7, 12, 99}


def test_list_pinned_entity_ids_empty_returns_empty_set(monkeypatch):
    cur = _FakeCursor(fetchalls=[[]])
    _patch_conn(monkeypatch, cur)
    assert db_layer.list_pinned_entity_ids("weekly_retro") == set()


def test_list_pinned_entity_ids_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown entity_type"):
        db_layer.list_pinned_entity_ids("bogus")


# ===========================================================================
# db_layer.avg_grade_from_letters — 4.3 GPA bucketing
# ===========================================================================


def test_avg_grade_empty_list_returns_none():
    assert db_layer.avg_grade_from_letters([]) is None
    assert db_layer.avg_grade_from_letters(None) is None


def test_avg_grade_ignores_falsy_and_unknown_values():
    """None, empty string, and unrecognized letters drop out of the
    arithmetic. All-falsy input → None."""
    assert db_layer.avg_grade_from_letters([None, "", "bogus"]) is None


def test_avg_grade_single_letter_round_trips():
    """A single A+ averages to 4.3 → bucket back to A+. Letter→numeric→
    letter is the identity for a one-element list."""
    assert db_layer.avg_grade_from_letters(["A+"]) == "A+"
    assert db_layer.avg_grade_from_letters(["B"]) == "B"
    assert db_layer.avg_grade_from_letters(["F"]) == "F"


def test_avg_grade_arithmetic_mean_buckets_correctly():
    """A (4.0) + B (3.0) = 3.5 → A- bucket. Pins the midpoint behavior:
    3.5 is the lower bound of A- so it rounds up, not B+."""
    assert db_layer.avg_grade_from_letters(["A", "B"]) == "A-"


def test_avg_grade_three_letters_picks_nearest_bucket():
    """A + B + C-  → (4.0 + 3.0 + 1.7) / 3 = 2.9 → B."""
    assert db_layer.avg_grade_from_letters(["A", "B", "C-"]) == "B"


def test_avg_grade_drops_unknown_letters_from_denominator():
    """A + 'bogus' → mean over [4.0] only, returns A. The denominator is
    the count of *recognized* letters, not the input length."""
    assert db_layer.avg_grade_from_letters(["A", "bogus"]) == "A"


# ===========================================================================
# db_layer.weekly_return_series_for_portfolio
# ===========================================================================


def test_weekly_return_series_empty_journal_returns_empty(monkeypatch):
    """No journal data → empty dict. The list endpoint guards against
    this by returning an empty envelope."""
    import pandas as pd
    monkeypatch.setattr(db_layer, "load_journal", lambda _: pd.DataFrame())
    assert db_layer.weekly_return_series_for_portfolio("CanSlim") == {}


def test_weekly_return_series_buckets_by_iso_monday(monkeypatch):
    """Days within the same ISO week (Mon-Sun) bucket together. Their
    chained 1+r product gives the weekly_return_pct. Mon=0..Sun=6 per
    Python's date.weekday()."""
    import pandas as pd
    # Week 1: Mon 2026-05-11 +10%, Tue 2026-05-12 +10% → chained = 21%
    # Week 2: Mon 2026-05-18 +5%, no other rows → 5%
    df = pd.DataFrame([
        {"day": "2026-05-11", "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
        {"day": "2026-05-12", "beg_nlv": 110.0, "end_nlv": 121.0, "cash_change": 0.0},
        {"day": "2026-05-18", "beg_nlv": 121.0, "end_nlv": 127.05, "cash_change": 0.0},
    ])
    monkeypatch.setattr(db_layer, "load_journal", lambda _: df)
    out = db_layer.weekly_return_series_for_portfolio("CanSlim")
    assert "2026-05-11" in out
    assert "2026-05-18" in out
    assert abs(out["2026-05-11"]["weekly_return_pct"] - 21.0) < 0.01
    assert abs(out["2026-05-18"]["weekly_return_pct"] - 5.0) < 0.01
    # Friday-of-week pinned correctly
    assert out["2026-05-11"]["week_end"] == "2026-05-15"


def test_weekly_return_series_matches_phase5_weekly_metrics(monkeypatch):
    """Cross-validation with Phase 5's weekly_metrics. Same journal, same
    week → both compute the same weekly_return_pct. Pins the contract
    that the rail's sparkline tooltip never drifts from the top-tile
    Weekly Return % shown after the user navigates."""
    import pandas as pd
    import nlv_service
    df = pd.DataFrame([
        {"day": "2026-05-11", "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
        {"day": "2026-05-12", "beg_nlv": 110.0, "end_nlv": 121.0, "cash_change": 0.0},
    ])
    monkeypatch.setattr(db_layer, "load_journal", lambda _: df)
    monkeypatch.setattr(nlv_service.db, "load_journal", lambda _: df)
    monkeypatch.setattr(nlv_service.db, "load_summary", lambda _: pd.DataFrame())
    series = db_layer.weekly_return_series_for_portfolio("CanSlim")
    phase5 = nlv_service.weekly_metrics("CanSlim", "2026-05-11")
    # Both should report ~21% for the week of 2026-05-11.
    assert abs(series["2026-05-11"]["weekly_return_pct"] -
               phase5["weekly_return_pct"]) < 0.01


# ===========================================================================
# db_layer.list_weekly_retros_rail — envelope shape
# ===========================================================================


def test_list_weekly_retros_rail_empty_account_returns_envelope_with_zeros(monkeypatch):
    """No journal data + no saved retros → envelope with empty weeks and
    zero ytd_stats. avg_grade defaults to None when weeks_graded == 0."""
    import pandas as pd
    monkeypatch.setattr(db_layer, "list_weekly_retros", lambda _, **k: [])
    monkeypatch.setattr(db_layer, "load_journal", lambda _: pd.DataFrame())
    monkeypatch.setattr(db_layer, "list_pinned_entity_ids", lambda _: set())
    out = db_layer.list_weekly_retros_rail("CanSlim")
    assert out["weeks"] == []
    assert out["ytd_stats"] == {
        "total_weeks": 0, "weeks_graded": 0,
        "avg_grade": None, "weeks_pinned": 0,
    }


def test_list_weekly_retros_rail_emits_synthetic_rows_for_missing_weeks(monkeypatch):
    """Journal has rows for two past weeks; user has saved only one retro.
    The other week shows up as a synthetic row with id=None,
    has_content=False. Sparkline values populate on the synthetic row too —
    the week has NLV data, just no graded retro.

    Uses past dates (2024 — comfortably before any plausible test
    wall-clock today) so the grid range always covers both Mondays.
    """
    import pandas as pd

    # Two past weeks of journal data: 2024-01-08 (Mon) and 2024-01-15 (Mon).
    df = pd.DataFrame([
        {"day": "2024-01-08", "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
        {"day": "2024-01-15", "beg_nlv": 110.0, "end_nlv": 121.0, "cash_change": 0.0},
    ])
    saved = [{"id": 7, "portfolio": "CanSlim", "week_start": "2024-01-08",
              "week_grade": "B+", "best_decision": "x", "worst_decision": "",
              "rule_change": False, "rule_change_text": "",
              "weekly_thoughts": "", "ticker_grades": {},
              "created_at": "2024-01-09T00:00:00",
              "updated_at": "2024-01-09T00:00:00"}]
    monkeypatch.setattr(db_layer, "list_weekly_retros", lambda _, **k: saved)
    monkeypatch.setattr(db_layer, "load_journal", lambda _: df)
    monkeypatch.setattr(db_layer, "list_pinned_entity_ids", lambda _: set())

    out = db_layer.list_weekly_retros_rail("CanSlim")
    keys = {w["key"] for w in out["weeks"]}
    assert "2024-01-08" in keys
    assert "2024-01-15" in keys
    saved_row = next(w for w in out["weeks"] if w["key"] == "2024-01-08")
    synth_row = next(w for w in out["weeks"] if w["key"] == "2024-01-15")
    assert saved_row["id"] == 7
    assert saved_row["has_content"] is True
    assert saved_row["week_grade"] == "B+"
    assert synth_row["id"] is None
    assert synth_row["has_content"] is False
    assert synth_row["week_grade"] is None
    # Synthetic row should still expose sparkline_value from journal data.
    assert synth_row["sparkline_value"] is not None
    assert abs(synth_row["sparkline_value"] - 10.0) < 0.01


def test_list_weekly_retros_rail_weeks_sorted_newest_first(monkeypatch):
    """Weeks emit in descending week_start order so the rail's top item
    is always the most recent."""
    import pandas as pd
    df = pd.DataFrame([
        {"day": "2024-01-08", "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
        {"day": "2024-01-15", "beg_nlv": 110.0, "end_nlv": 121.0, "cash_change": 0.0},
    ])
    monkeypatch.setattr(db_layer, "list_weekly_retros", lambda _, **k: [])
    monkeypatch.setattr(db_layer, "load_journal", lambda _: df)
    monkeypatch.setattr(db_layer, "list_pinned_entity_ids", lambda _: set())
    out = db_layer.list_weekly_retros_rail("CanSlim")
    week_starts = [w["week_start"] for w in out["weeks"]]
    assert week_starts == sorted(week_starts, reverse=True)
    # At least 2 weeks present (the two journal Mondays).
    assert len(week_starts) >= 2


def test_list_weekly_retros_rail_marks_pinned_rows(monkeypatch):
    """Rows whose id is in the pinned set surface as pinned=True; others
    as pinned=False. Synthetic rows (id=None) are never pinned regardless."""
    import pandas as pd
    df = pd.DataFrame([
        {"day": "2026-05-11", "beg_nlv": 100.0, "end_nlv": 110.0, "cash_change": 0.0},
    ])
    saved = [{"id": 7, "portfolio": "CanSlim", "week_start": "2026-05-11",
              "week_grade": "A", "best_decision": "win", "worst_decision": "",
              "rule_change": False, "rule_change_text": "",
              "weekly_thoughts": "", "ticker_grades": {},
              "created_at": "2026-05-12T00:00:00",
              "updated_at": "2026-05-12T00:00:00"}]
    monkeypatch.setattr(db_layer, "list_weekly_retros", lambda _, **k: saved)
    monkeypatch.setattr(db_layer, "load_journal", lambda _: df)
    monkeypatch.setattr(db_layer, "list_pinned_entity_ids", lambda _: {7})
    out = db_layer.list_weekly_retros_rail("CanSlim")
    saved_row = next(w for w in out["weeks"] if w["key"] == "2026-05-11")
    assert saved_row["pinned"] is True


# ===========================================================================
# API endpoint — POST /api/pins/toggle
# ===========================================================================
# Use the same TestClient pattern as test_weekly_retros.py: import api.main
# once at module load, monkey-patch its AUTH_SECRET + db helpers per test,
# disable the slowapi limiter so test bursts don't trip the rate cap.
# Do NOT importlib.reload — that re-runs Sentry init + slowapi setup and
# hangs the test under FastAPI's lifespan startup.


@pytest.fixture
def pins_client(monkeypatch):
    """Returns (state, client) — state knobs control toggle_pin results."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, list | bool] = {"toggle_calls": [], "next_pinned": True}

    def fake_toggle(et, eid):
        state["toggle_calls"].append((et, eid))
        return state["next_pinned"]
    monkeypatch.setattr(db_layer, "toggle_pin", fake_toggle)

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


def test_pins_toggle_rejects_unknown_entity_type(pins_client):
    """Defense in depth — body validation runs before db_layer is touched."""
    state, client = pins_client
    resp = client.post(
        "/api/pins/toggle",
        json={"entity_type": "bogus", "entity_id": 1},
    )
    assert resp.status_code == 200
    assert resp.json() == {"error": "invalid_entity_type"}
    assert state["toggle_calls"] == []


def test_pins_toggle_rejects_non_integer_entity_id(pins_client):
    state, client = pins_client
    resp = client.post(
        "/api/pins/toggle",
        json={"entity_type": "weekly_retro", "entity_id": "not-a-number"},
    )
    assert resp.status_code == 200
    assert "error" in resp.json()
    assert state["toggle_calls"] == []


def test_pins_toggle_happy_path_returns_pinned_state(pins_client):
    """Valid request → db_layer.toggle_pin called → returns {pinned: bool}."""
    state, client = pins_client
    state["next_pinned"] = True
    resp = client.post(
        "/api/pins/toggle",
        json={"entity_type": "weekly_retro", "entity_id": 42},
    )
    assert resp.status_code == 200
    assert resp.json() == {"pinned": True}
    assert state["toggle_calls"] == [("weekly_retro", 42)]


def test_pins_toggle_idempotent_double_call(pins_client):
    """Two calls → server reports each new state. Idempotency is enforced
    in db_layer.toggle_pin; this test pins that the endpoint surfaces the
    result verbatim, no double-toggle bouncing."""
    state, client = pins_client
    body = {"entity_type": "weekly_retro", "entity_id": 42}
    state["next_pinned"] = True
    r1 = client.post("/api/pins/toggle", json=body)
    state["next_pinned"] = False
    r2 = client.post("/api/pins/toggle", json=body)
    assert r1.json() == {"pinned": True}
    assert r2.json() == {"pinned": False}
    assert state["toggle_calls"] == [("weekly_retro", 42), ("weekly_retro", 42)]


# ===========================================================================
# API endpoint — GET /api/weekly-retros/list now returns envelope shape
# ===========================================================================


def test_weekly_retros_list_returns_envelope_shape(monkeypatch):
    """Endpoint contract: response carries `weeks` array + `ytd_stats`
    object. This is the breaking shape change for Phase 6 — the old
    bare-array response is gone with the Review History tab."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    monkeypatch.setattr(
        db_layer, "list_weekly_retros_rail",
        lambda _: {
            "weeks": [{"id": 1, "key": "2026-05-11", "week_start": "2026-05-11",
                       "week_end": "2026-05-15", "year": 2026, "month": 5,
                       "title": "May 11 – May 15", "has_content": True,
                       "pinned": False, "sparkline_value": 4.62,
                       "week_grade": "B+"}],
            "ytd_stats": {"total_weeks": 1, "weeks_graded": 1,
                          "avg_grade": "B+", "weeks_pinned": 0},
        },
    )
    client = TestClient(main.app, headers=_auth_headers())
    resp = client.get("/api/weekly-retros/list?portfolio=CanSlim")
    assert resp.status_code == 200
    body = resp.json()
    assert "weeks" in body
    assert "ytd_stats" in body
    assert body["ytd_stats"]["avg_grade"] == "B+"
    assert len(body["weeks"]) == 1
    assert body["weeks"][0]["key"] == "2026-05-11"
