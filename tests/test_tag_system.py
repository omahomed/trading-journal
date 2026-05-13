"""Tests for the Phase 1 tag system (Migration 026).

Two surfaces, mirroring the test_weekly_retros.py pattern:

  1. db_layer helpers — fake-cursor unit tests against the SQL the helpers
     emit. Covers the upsert/idempotent-restore branching, the validation
     rejections (color palette, name length, unknown entity_type), and the
     soft-delete-not-hard-delete contract.

  2. API endpoints — FastAPI TestClient with stubbed db_layer. Covers HTTP
     surface: validation, error body translation (UniqueViolation →
     tag_name_exists, count cap → tag_limit_reached), and the entity_type
     CHECK enforcement.

Constraint behavior that requires a live Postgres (case-insensitive
collision via LOWER(name) partial-unique, RLS isolation, recycle after
soft-delete via partial-unique index, polymorphic CHECK rejection) is
enforced by the migration at apply-time and exercised in manual smoke
tests post-deploy — same coverage gap as every other DB-side constraint
in this codebase.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import jwt
import psycopg2
import pytest
from fastapi.testclient import TestClient

import db_layer


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "11111111-2222-3333-4444-555555555555"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Fake cursor scaffolding (mirrors test_weekly_retros.py)
# ---------------------------------------------------------------------------


class _FakeCursor:
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
    conn = _FakeConn(cursor)
    monkeypatch.setattr(db_layer, "get_db_connection", lambda: _ConnCM(conn))
    return conn


# ---------------------------------------------------------------------------
# db_layer.create_tag — validation
# ---------------------------------------------------------------------------


def test_create_tag_rejects_empty_name(monkeypatch):
    """Empty name → ValueError before any DB call."""
    with pytest.raises(ValueError, match="cannot be empty"):
        db_layer.create_tag("CanSlim", "", "rose")


def test_create_tag_rejects_long_name(monkeypatch):
    """Name >60 chars → ValueError before any DB call. Matches strategies'
    cap (db_layer.py:3927) for parity."""
    with pytest.raises(ValueError, match="1-60 characters"):
        db_layer.create_tag("CanSlim", "x" * 61, "rose")


def test_create_tag_rejects_unknown_color(monkeypatch):
    """Color outside the closed palette → ValueError. Defense in depth vs.
    the API-layer check; ensures the server never stores a value the
    frontend's TAG_PALETTE doesn't recognize."""
    with pytest.raises(ValueError, match="invalid_color"):
        db_layer.create_tag("CanSlim", "drawdown", "purple")


def test_create_tag_accepts_all_palette_colors(monkeypatch):
    """All five palette keys are accepted; happy path emits the INSERT."""
    for color in ("rose", "amber", "emerald", "sky", "violet"):
        cur = _FakeCursor(fetchones=[
            {"id": 1},                                 # portfolio lookup
            {"id": 7, "portfolio": "CanSlim", "name": "drawdown",
             "color": color, "created_at": datetime(2026, 5, 13),
             "updated_at": datetime(2026, 5, 13)},
        ])
        _patch_conn(monkeypatch, cur)
        result = db_layer.create_tag("CanSlim", "drawdown", color)
        assert result["color"] == color


def test_create_tag_unknown_portfolio(monkeypatch):
    """Unknown portfolio → ValueError, no INSERT emitted."""
    cur = _FakeCursor(fetchones=[None])  # portfolio lookup misses
    _patch_conn(monkeypatch, cur)
    with pytest.raises(ValueError, match="not found"):
        db_layer.create_tag("Bogus", "drawdown", "rose")
    assert not any("INSERT" in s for s, _ in cur.executed)


# ---------------------------------------------------------------------------
# db_layer.update_tag
# ---------------------------------------------------------------------------


def test_update_tag_whitelists_name_and_color(monkeypatch):
    """update_tag accepts only name and color. Builds a SET clause that
    only includes the provided fields plus updated_at = NOW()."""
    cur = _FakeCursor(fetchones=[
        {"id": 7},                                     # UPDATE returning
        {"id": 7, "portfolio": "CanSlim", "name": "renamed",
         "color": "amber", "created_at": datetime(2026, 5, 1),
         "updated_at": datetime(2026, 5, 13)},
    ])
    _patch_conn(monkeypatch, cur)

    result = db_layer.update_tag(7, name="renamed", color="amber")
    assert result is not None
    assert result["name"] == "renamed"
    assert result["color"] == "amber"
    update_sql = next(s for s, _ in cur.executed if "UPDATE tags" in s)
    assert "name = %s" in update_sql
    assert "color = %s" in update_sql
    assert "updated_at = NOW()" in update_sql


def test_update_tag_rejects_invalid_color(monkeypatch):
    """Color validation runs BEFORE the DB call."""
    with pytest.raises(ValueError, match="invalid_color"):
        db_layer.update_tag(7, color="purple")


def test_update_tag_returns_none_for_missing(monkeypatch):
    """If the UPDATE doesn't hit (id missing or already deleted), returns
    None instead of raising."""
    cur = _FakeCursor(fetchones=[None])
    _patch_conn(monkeypatch, cur)
    result = db_layer.update_tag(999, name="new")
    assert result is None


# ---------------------------------------------------------------------------
# db_layer.soft_delete_tag
# ---------------------------------------------------------------------------


def test_soft_delete_tag_uses_update_not_delete(monkeypatch):
    """soft_delete must UPDATE deleted_at, never DELETE FROM."""
    cur = _FakeCursor(fetchones=[{"id": 7}])
    _patch_conn(monkeypatch, cur)
    assert db_layer.soft_delete_tag(7) is True
    statements = [s for s, _ in cur.executed]
    assert any("UPDATE tags SET deleted_at = NOW()" in s for s in statements)
    assert not any("DELETE FROM tags" in s for s in statements)


def test_soft_delete_tag_returns_false_when_no_row(monkeypatch):
    """Idempotent: deleting an unknown id returns False."""
    cur = _FakeCursor(fetchones=[None])
    _patch_conn(monkeypatch, cur)
    assert db_layer.soft_delete_tag(999) is False


# ---------------------------------------------------------------------------
# db_layer.create_tag_assignment — three branching paths
# ---------------------------------------------------------------------------


def test_create_assignment_inserts_when_no_existing(monkeypatch):
    """Fresh attach: no row exists → INSERT path."""
    cur = _FakeCursor(fetchones=[
        {"portfolio_id": 1},                                 # tag lookup
        None,                                                 # no existing
        {"id": 99},                                           # INSERT returning
        {  # final re-fetch
            "id": 99, "tag_id": 7, "tag_name": "drawdown", "tag_color": "rose",
            "entity_type": "weekly_retro", "entity_id": 42,
            "created_at": datetime(2026, 5, 13),
        },
    ])
    _patch_conn(monkeypatch, cur)

    result = db_layer.create_tag_assignment(7, "weekly_retro", 42)
    assert result["id"] == 99
    statements = [s for s, _ in cur.executed]
    assert any("INSERT INTO tag_assignments" in s for s in statements)
    assert not any("UPDATE tag_assignments" in s for s in statements)


def test_create_assignment_revives_soft_deleted(monkeypatch):
    """Idempotent restore: soft-deleted row exists → UPDATE deleted_at = NULL.
    NO new row inserted, ID stable."""
    cur = _FakeCursor(fetchones=[
        {"portfolio_id": 1},                                 # tag lookup
        {"id": 50, "deleted_at": datetime(2026, 4, 1)},       # soft-deleted
        {"id": 50},                                           # UPDATE returning
        {  # re-fetch
            "id": 50, "tag_id": 7, "tag_name": "drawdown", "tag_color": "rose",
            "entity_type": "weekly_retro", "entity_id": 42,
            "created_at": datetime(2026, 4, 1),
        },
    ])
    _patch_conn(monkeypatch, cur)

    result = db_layer.create_tag_assignment(7, "weekly_retro", 42)
    # ID stability is the whole point.
    assert result["id"] == 50
    statements = [s for s, _ in cur.executed]
    update_sql = next(s for s, _ in cur.executed if "UPDATE tag_assignments" in s)
    assert "deleted_at = NULL" in update_sql
    assert not any("INSERT INTO tag_assignments" in s for s in statements)


def test_create_assignment_noop_when_live_exists(monkeypatch):
    """Re-attaching an already-live tag is a no-op — return existing row,
    no UPDATE, no INSERT."""
    cur = _FakeCursor(fetchones=[
        {"portfolio_id": 1},                                 # tag lookup
        {"id": 99, "deleted_at": None},                       # live row exists
        {  # re-fetch
            "id": 99, "tag_id": 7, "tag_name": "drawdown", "tag_color": "rose",
            "entity_type": "weekly_retro", "entity_id": 42,
            "created_at": datetime(2026, 5, 13),
        },
    ])
    _patch_conn(monkeypatch, cur)

    result = db_layer.create_tag_assignment(7, "weekly_retro", 42)
    assert result["id"] == 99
    statements = [s for s, _ in cur.executed]
    assert not any("INSERT INTO tag_assignments" in s for s in statements)
    assert not any("UPDATE tag_assignments" in s for s in statements)


def test_create_assignment_rejects_unknown_entity_type(monkeypatch):
    """Unknown entity_type → ValueError BEFORE any DB call."""
    with pytest.raises(ValueError, match="Unknown entity_type"):
        db_layer.create_tag_assignment(7, "bogus_entity", 42)


def test_create_assignment_rejects_missing_or_deleted_tag(monkeypatch):
    """Soft-deleted tag → ValueError. Attaching a deleted tag would create
    an invisible assignment (load filters tags.deleted_at IS NULL)."""
    cur = _FakeCursor(fetchones=[None])  # tag lookup misses
    _patch_conn(monkeypatch, cur)
    with pytest.raises(ValueError, match="not found"):
        db_layer.create_tag_assignment(999, "weekly_retro", 42)


# ---------------------------------------------------------------------------
# db_layer.soft_delete_tag_assignment
# ---------------------------------------------------------------------------


def test_soft_delete_assignment_uses_update_not_delete(monkeypatch):
    cur = _FakeCursor(fetchones=[{"id": 99}])
    _patch_conn(monkeypatch, cur)
    assert db_layer.soft_delete_tag_assignment(99) is True
    statements = [s for s, _ in cur.executed]
    assert any("UPDATE tag_assignments SET deleted_at = NOW()" in s
               for s in statements)
    assert not any("DELETE FROM tag_assignments" in s for s in statements)


def test_soft_delete_assignment_returns_false_when_no_row(monkeypatch):
    cur = _FakeCursor(fetchones=[None])
    _patch_conn(monkeypatch, cur)
    assert db_layer.soft_delete_tag_assignment(999) is False


# ---------------------------------------------------------------------------
# db_layer.load_tag_assignments — JOIN filter
# ---------------------------------------------------------------------------


def test_load_assignments_joins_tags_with_deleted_at_filter(monkeypatch):
    """The JOIN must filter tags.deleted_at IS NULL so soft-deleted tags'
    assignments don't appear in the result. This is how a soft-deleted tag
    visually disappears without us having to mass-update its children."""
    cur = _FakeCursor()
    _patch_conn(monkeypatch, cur)
    db_layer.load_tag_assignments("weekly_retro", 42)
    select_sql = cur.executed[0][0]
    assert "JOIN tags t ON t.id = a.tag_id AND t.deleted_at IS NULL" in select_sql


def test_load_assignments_rejects_unknown_entity_type():
    """Validation guard at helper entry."""
    with pytest.raises(ValueError, match="Unknown entity_type"):
        db_layer.load_tag_assignments("bogus_entity", 42)


# ---------------------------------------------------------------------------
# Endpoint tests — FastAPI TestClient + stubbed db_layer
# ---------------------------------------------------------------------------


@pytest.fixture
def tag_stubs(monkeypatch):
    """Yield (state, client). Same shape as test_weekly_retros.py."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "load_tags_result": [],
        "create_tag_result": None,
        "create_tag_raises": None,
        "update_tag_result": None,
        "update_tag_raises": None,
        "delete_tag_result": True,

        "load_assignments_result": [],
        "count_assignments_result": 0,
        "create_assignment_result": None,
        "create_assignment_raises": None,
        "delete_assignment_result": True,

        "create_tag_calls": [],
        "create_assignment_calls": [],
        "load_assignments_calls": [],
        "count_calls": [],
    }

    def fake_load_tags(portfolio):
        return state["load_tags_result"]
    monkeypatch.setattr(db_layer, "load_tags", fake_load_tags)

    def fake_create_tag(portfolio, name, color):
        state["create_tag_calls"].append((portfolio, name, color))
        if state["create_tag_raises"] is not None:
            raise state["create_tag_raises"]
        if state["create_tag_result"] is not None:
            return state["create_tag_result"]
        return {
            "id": 1, "portfolio": portfolio, "name": name, "color": color,
            "created_at": "2026-05-13T00:00:00", "updated_at": "2026-05-13T00:00:00",
        }
    monkeypatch.setattr(db_layer, "create_tag", fake_create_tag)

    def fake_update_tag(tag_id, **fields):
        if state["update_tag_raises"] is not None:
            raise state["update_tag_raises"]
        return state["update_tag_result"]
    monkeypatch.setattr(db_layer, "update_tag", fake_update_tag)

    def fake_soft_delete_tag(tag_id):
        return state["delete_tag_result"]
    monkeypatch.setattr(db_layer, "soft_delete_tag", fake_soft_delete_tag)

    def fake_load_assignments(entity_type, entity_id):
        state["load_assignments_calls"].append((entity_type, entity_id))
        return state["load_assignments_result"]
    monkeypatch.setattr(db_layer, "load_tag_assignments", fake_load_assignments)

    def fake_count(entity_type, entity_id):
        state["count_calls"].append((entity_type, entity_id))
        return state["count_assignments_result"]
    monkeypatch.setattr(db_layer, "count_live_tag_assignments", fake_count)

    def fake_create_assignment(tag_id, entity_type, entity_id):
        state["create_assignment_calls"].append((tag_id, entity_type, entity_id))
        if state["create_assignment_raises"] is not None:
            raise state["create_assignment_raises"]
        if state["create_assignment_result"] is not None:
            return state["create_assignment_result"]
        return {
            "id": 99, "tag_id": tag_id, "tag_name": "drawdown",
            "tag_color": "rose", "entity_type": entity_type,
            "entity_id": entity_id, "created_at": "2026-05-13T00:00:00",
        }
    monkeypatch.setattr(db_layer, "create_tag_assignment", fake_create_assignment)

    def fake_soft_delete_assignment(assignment_id):
        return state["delete_assignment_result"]
    monkeypatch.setattr(db_layer, "soft_delete_tag_assignment",
                        fake_soft_delete_assignment)

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


# ---- GET /api/tags ----------------------------------------------------------


def test_get_tags_returns_array(tag_stubs):
    state, client = tag_stubs
    state["load_tags_result"] = [
        {"id": 1, "portfolio": "CanSlim", "name": "drawdown", "color": "rose",
         "created_at": "2026-05-12T00:00:00", "updated_at": "2026-05-12T00:00:00"},
    ]
    r = client.get("/api/tags?portfolio=CanSlim")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert body[0]["name"] == "drawdown"


# ---- POST /api/tags ---------------------------------------------------------


def test_post_tag_happy_path(tag_stubs):
    state, client = tag_stubs
    r = client.post("/api/tags", json={
        "portfolio": "CanSlim", "name": "drawdown", "color": "rose",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "drawdown"
    assert body["color"] == "rose"
    assert state["create_tag_calls"] == [("CanSlim", "drawdown", "rose")]


def test_post_tag_rejects_invalid_color(tag_stubs):
    """Closed palette enforced API-side. No DB call."""
    state, client = tag_stubs
    r = client.post("/api/tags", json={
        "portfolio": "CanSlim", "name": "drawdown", "color": "purple",
    })
    assert r.status_code == 200
    assert r.json() == {"error": "invalid_color"}
    assert state["create_tag_calls"] == []


def test_post_tag_rejects_empty_name(tag_stubs):
    state, client = tag_stubs
    r = client.post("/api/tags", json={
        "portfolio": "CanSlim", "name": "   ", "color": "rose",
    })
    assert r.status_code == 200
    assert "name" in r.json()["error"].lower()


def test_post_tag_translates_unique_violation(tag_stubs):
    """Case-insensitive collision (Drawdown vs drawdown) → typed
    {"error": "tag_name_exists"} body for clean frontend handling."""
    state, client = tag_stubs
    state["create_tag_raises"] = psycopg2.errors.UniqueViolation(
        "duplicate key violates uq_tags_portfolio_name_live"
    )
    r = client.post("/api/tags", json={
        "portfolio": "CanSlim", "name": "drawdown", "color": "rose",
    })
    assert r.status_code == 200
    assert r.json() == {"error": "tag_name_exists"}


# ---- PATCH /api/tags/{id} ---------------------------------------------------


def test_patch_tag_happy_path(tag_stubs):
    state, client = tag_stubs
    state["update_tag_result"] = {
        "id": 1, "portfolio": "CanSlim", "name": "renamed", "color": "amber",
        "created_at": "2026-05-12T00:00:00", "updated_at": "2026-05-13T00:00:00",
    }
    r = client.patch("/api/tags/1", json={"name": "renamed", "color": "amber"})
    assert r.status_code == 200
    assert r.json()["name"] == "renamed"


def test_patch_tag_rejects_invalid_color(tag_stubs):
    state, client = tag_stubs
    r = client.patch("/api/tags/1", json={"color": "purple"})
    assert r.status_code == 200
    assert r.json() == {"error": "invalid_color"}


def test_patch_tag_returns_not_found(tag_stubs):
    state, client = tag_stubs
    state["update_tag_result"] = None
    r = client.patch("/api/tags/999", json={"name": "x"})
    assert r.status_code == 200
    assert r.json() == {"error": "not_found"}


# ---- DELETE /api/tags/{id} --------------------------------------------------


def test_delete_tag_happy_path(tag_stubs):
    state, client = tag_stubs
    r = client.delete("/api/tags/1")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "id": 1}


def test_delete_tag_not_found(tag_stubs):
    state, client = tag_stubs
    state["delete_tag_result"] = False
    r = client.delete("/api/tags/999")
    assert r.status_code == 200
    assert r.json() == {"error": "not_found"}


# ---- GET /api/tags/assignments ---------------------------------------------


def test_get_assignments_returns_array(tag_stubs):
    state, client = tag_stubs
    state["load_assignments_result"] = [
        {"id": 99, "tag_id": 1, "tag_name": "drawdown", "tag_color": "rose",
         "entity_type": "weekly_retro", "entity_id": 42,
         "created_at": "2026-05-13T00:00:00"},
    ]
    r = client.get("/api/tags/assignments?entity_type=weekly_retro&entity_id=42")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["tag_name"] == "drawdown"


def test_get_assignments_rejects_unknown_entity_type(tag_stubs):
    state, client = tag_stubs
    r = client.get("/api/tags/assignments?entity_type=bogus&entity_id=42")
    assert r.status_code == 200
    assert r.json() == {"error": "invalid_entity_type"}


# ---- POST /api/tags/assignments --------------------------------------------


def test_post_assignment_happy_path(tag_stubs):
    state, client = tag_stubs
    r = client.post("/api/tags/assignments", json={
        "tag_id": 1, "entity_type": "weekly_retro", "entity_id": 42,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["tag_id"] == 1
    assert body["entity_id"] == 42


def test_post_assignment_rejects_invalid_entity_type(tag_stubs):
    state, client = tag_stubs
    r = client.post("/api/tags/assignments", json={
        "tag_id": 1, "entity_type": "bogus", "entity_id": 42,
    })
    assert r.status_code == 200
    assert r.json() == {"error": "invalid_entity_type"}
    assert state["create_assignment_calls"] == []


def test_post_assignment_enforces_max_10_cap_for_new_tag(tag_stubs):
    """11th unique tag on the same entity → tag_limit_reached. The cap
    counts current live assignments; if >=10 AND the new tag isn't already
    among them, reject."""
    state, client = tag_stubs
    state["count_assignments_result"] = 10
    # No existing assignment for tag_id=99 → it'd be the 11th.
    state["load_assignments_result"] = [
        {"id": i, "tag_id": i, "tag_name": f"t{i}", "tag_color": "sky",
         "entity_type": "weekly_retro", "entity_id": 42,
         "created_at": "2026-05-13T00:00:00"}
        for i in range(1, 11)
    ]
    r = client.post("/api/tags/assignments", json={
        "tag_id": 99, "entity_type": "weekly_retro", "entity_id": 42,
    })
    assert r.status_code == 200
    assert r.json() == {"error": "tag_limit_reached"}
    assert state["create_assignment_calls"] == []


def test_post_assignment_at_cap_allows_reattach_of_already_assigned_tag(tag_stubs):
    """The cap doesn't block re-attaching a tag that's ALREADY in the count.
    This keeps idempotent re-attach (the helper's no-op path) usable even
    at the cap."""
    state, client = tag_stubs
    state["count_assignments_result"] = 10
    state["load_assignments_result"] = [
        {"id": 99, "tag_id": 7, "tag_name": "t", "tag_color": "sky",
         "entity_type": "weekly_retro", "entity_id": 42,
         "created_at": "2026-05-13T00:00:00"}
    ] + [
        {"id": i, "tag_id": i, "tag_name": f"t{i}", "tag_color": "sky",
         "entity_type": "weekly_retro", "entity_id": 42,
         "created_at": "2026-05-13T00:00:00"}
        for i in range(2, 11)
    ]
    r = client.post("/api/tags/assignments", json={
        "tag_id": 7, "entity_type": "weekly_retro", "entity_id": 42,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["tag_id"] == 7
    # Helper was called (not blocked by the cap).
    assert state["create_assignment_calls"] == [(7, "weekly_retro", 42)]


def test_post_assignment_validates_tag_id_type(tag_stubs):
    state, client = tag_stubs
    r = client.post("/api/tags/assignments", json={
        "tag_id": "not-an-int", "entity_type": "weekly_retro", "entity_id": 42,
    })
    assert r.status_code == 200
    assert "tag_id" in r.json()["error"].lower()


# ---- DELETE /api/tags/assignments/{id} -------------------------------------


def test_delete_assignment_happy_path(tag_stubs):
    state, client = tag_stubs
    r = client.delete("/api/tags/assignments/99")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "id": 99}


def test_delete_assignment_not_found(tag_stubs):
    state, client = tag_stubs
    state["delete_assignment_result"] = False
    r = client.delete("/api/tags/assignments/999")
    assert r.status_code == 200
    assert r.json() == {"error": "not_found"}
