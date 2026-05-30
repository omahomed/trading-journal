"""Tests for the pinned_routes surface (Migration 042).

Mirrors test_pinned_entities.py's two-surface structure:

  1. db_layer helpers — fake-cursor unit tests of the SQL toggle_pin_route
     emits per branch (insert / revive / soft-delete), plus the
     list_pinned_routes ordering contract and the route_path validator.

  2. API endpoints — FastAPI TestClient with stubbed db_layer. Covers
     GET /api/pinned-routes + POST /api/pinned-routes/toggle contracts
     and their request-body validation.

Constraint behavior that needs a live Postgres (partial-unique blocking
duplicate live pins, RLS isolation, CHECK rejecting bad route_paths) is
exercised by the migration + post-deploy smoke — same coverage-gap
convention as test_pinned_entities.py.
"""
from __future__ import annotations

import datetime as dt

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
# Fake cursor scaffolding (matches test_pinned_entities.py verbatim — same
# shape, same fetchone/fetchall queue API)
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
# db_layer.toggle_pin_route
# ===========================================================================


def test_toggle_pin_route_inserts_when_no_existing_row(monkeypatch):
    """No existing row → INSERT a new pin, return True (now pinned)."""
    cur = _FakeCursor(fetchones=[None])
    _patch_conn(monkeypatch, cur)

    pinned = db_layer.toggle_pin_route("/log-buy")
    assert pinned is True
    assert any("INSERT INTO pinned_routes" in s for s, _ in cur.executed)


def test_toggle_pin_route_soft_deletes_live_row(monkeypatch):
    """Live row exists → UPDATE deleted_at = NOW(). Return False (unpinned)."""
    cur = _FakeCursor(fetchones=[{"id": 7, "deleted_at": None}])
    _patch_conn(monkeypatch, cur)

    pinned = db_layer.toggle_pin_route("/log-buy")
    assert pinned is False
    update_sqls = [s for s, _ in cur.executed if "UPDATE pinned_routes" in s]
    assert any("deleted_at = NOW()" in s for s in update_sqls)
    assert all("INSERT" not in s for s, _ in cur.executed)


def test_toggle_pin_route_revives_soft_deleted_row(monkeypatch):
    """Soft-deleted row exists → UPDATE deleted_at = NULL (revive). Return
    True. Reuses the same row id so pinned_at sort stability is preserved
    across unpin/re-pin cycles — the FIFO ordering contract."""
    cur = _FakeCursor(fetchones=[{"id": 7, "deleted_at": "2026-05-12T12:00:00"}])
    _patch_conn(monkeypatch, cur)

    pinned = db_layer.toggle_pin_route("/log-buy")
    assert pinned is True
    update_sqls = [s for s, _ in cur.executed if "UPDATE pinned_routes" in s]
    assert any("deleted_at = NULL" in s for s in update_sqls)
    assert all("INSERT" not in s for s, _ in cur.executed)


def test_toggle_pin_route_double_call_returns_to_original(monkeypatch):
    """Toggle twice in sequence ends at the original state. Combines the
    SELECT-then-INSERT branch with the SELECT-then-soft-delete branch.
    Each toggle gets its own (mocked) connection because db_layer doesn't
    reuse cursors across `with get_db_connection()` blocks."""
    cur1 = _FakeCursor(fetchones=[None])
    conn1 = _FakeConn(cur1)
    cur2 = _FakeCursor(fetchones=[{"id": 7, "deleted_at": None}])
    conn2 = _FakeConn(cur2)
    calls = iter([_ConnCM(conn1), _ConnCM(conn2)])
    monkeypatch.setattr(db_layer, "get_db_connection", lambda: next(calls))

    first = db_layer.toggle_pin_route("/log-buy")
    second = db_layer.toggle_pin_route("/log-buy")
    assert first is True
    assert second is False


def test_toggle_pin_route_rejects_missing_leading_slash():
    """Defense in depth vs. the CHECK constraint. Raises ValueError before
    touching the DB so callers get a typed error, not an IntegrityError."""
    with pytest.raises(ValueError, match="Invalid route_path"):
        db_layer.toggle_pin_route("log-buy")


def test_toggle_pin_route_rejects_uppercase():
    """The CHECK constraint is conservative — uppercase letters are
    rejected. Matches the nav.ts inventory shape (kebab-case lowercase)."""
    with pytest.raises(ValueError, match="Invalid route_path"):
        db_layer.toggle_pin_route("/Log-Buy")


def test_toggle_pin_route_rejects_underscores():
    """Underscores not allowed — kebab-case enforced."""
    with pytest.raises(ValueError, match="Invalid route_path"):
        db_layer.toggle_pin_route("/log_buy")


def test_toggle_pin_route_rejects_non_string():
    """Defense vs. callers passing int / None / dict."""
    with pytest.raises(ValueError, match="Invalid route_path"):
        db_layer.toggle_pin_route(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid route_path"):
        db_layer.toggle_pin_route(42)  # type: ignore[arg-type]


def test_toggle_pin_route_rejects_empty_string():
    with pytest.raises(ValueError, match="Invalid route_path"):
        db_layer.toggle_pin_route("")


def test_toggle_pin_route_accepts_canonical_nav_paths(monkeypatch):
    """Sample of canonical nav.ts paths — all must pass validation. If
    this fails after a nav.ts addition, either the new path violates the
    kebab-case convention (fix the path) or the CHECK constraint needs
    relaxing (follow-up migration)."""
    cur = _FakeCursor(fetchones=[None])
    _patch_conn(monkeypatch, cur)
    # Each path needs its own connection because of the with-block pattern.
    for path in (
        "/log-buy", "/log-sell", "/active-campaign", "/m-factor",
        "/trade-journal", "/performance-heatmap", "/period-review",
        "/ai-coach", "/settings", "/admin",
    ):
        cur = _FakeCursor(fetchones=[None])
        _patch_conn(monkeypatch, cur)
        # Should not raise.
        db_layer.toggle_pin_route(path)


def test_toggle_pin_route_accepts_nested_paths(monkeypatch):
    """Multi-segment paths like /foo/bar should also pass — the regex
    explicitly allows (/segment)* after the first segment."""
    cur = _FakeCursor(fetchones=[None])
    _patch_conn(monkeypatch, cur)
    db_layer.toggle_pin_route("/foo/bar/baz")  # should not raise


# ===========================================================================
# db_layer.list_pinned_routes
# ===========================================================================


def test_list_pinned_routes_returns_list_of_dicts_in_fifo_order(monkeypatch):
    """Returns list of {route_path, pinned_at} dicts. The DB's ORDER BY
    pinned_at ASC, id ASC is the FIFO contract — first-pinned shows first."""
    t1 = dt.datetime(2026, 5, 10, 9, 0, tzinfo=dt.timezone.utc)
    t2 = dt.datetime(2026, 5, 11, 9, 0, tzinfo=dt.timezone.utc)
    t3 = dt.datetime(2026, 5, 12, 9, 0, tzinfo=dt.timezone.utc)
    # Fixture rows arrive from the cursor in the DB-order (pinned_at ASC).
    cur = _FakeCursor(fetchalls=[[
        {"route_path": "/log-buy",          "pinned_at": t1},
        {"route_path": "/active-campaign",  "pinned_at": t2},
        {"route_path": "/trade-journal",    "pinned_at": t3},
    ]])
    _patch_conn(monkeypatch, cur)

    out = db_layer.list_pinned_routes()
    assert out == [
        {"route_path": "/log-buy",          "pinned_at": t1},
        {"route_path": "/active-campaign",  "pinned_at": t2},
        {"route_path": "/trade-journal",    "pinned_at": t3},
    ]


def test_list_pinned_routes_empty_returns_empty_list(monkeypatch):
    cur = _FakeCursor(fetchalls=[[]])
    _patch_conn(monkeypatch, cur)
    assert db_layer.list_pinned_routes() == []


def test_list_pinned_routes_sql_filters_deleted_and_orders_by_pinned_at(monkeypatch):
    """Structural assertion on the emitted SQL: must filter deleted_at
    IS NULL (otherwise unpinned routes leak into the list) and ORDER BY
    pinned_at ASC (otherwise FIFO is violated)."""
    cur = _FakeCursor(fetchalls=[[]])
    _patch_conn(monkeypatch, cur)
    db_layer.list_pinned_routes()
    select_sqls = [s for s, _ in cur.executed if "SELECT" in s]
    assert any("FROM pinned_routes" in s for s in select_sqls)
    assert any("deleted_at IS NULL" in s for s in select_sqls)
    assert any("ORDER BY pinned_at ASC" in s for s in select_sqls)


# ===========================================================================
# API endpoint — GET /api/pinned-routes
# ===========================================================================
# Use the same TestClient pattern as test_pinned_entities.py: import api.main
# once at module load, monkey-patch its AUTH_SECRET + db helpers per test,
# disable the slowapi limiter so test bursts don't trip the rate cap.


@pytest.fixture
def routes_client(monkeypatch):
    """Returns (state, client) — state knobs control list/toggle results."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict = {
        "list_calls": 0,
        "next_list": [],
        "toggle_calls": [],
        "next_pinned": True,
        "raise_on_toggle": None,
    }

    def fake_list():
        state["list_calls"] += 1
        return state["next_list"]
    monkeypatch.setattr(db_layer, "list_pinned_routes", fake_list)

    def fake_toggle(route_path):
        state["toggle_calls"].append(route_path)
        if state["raise_on_toggle"] is not None:
            raise state["raise_on_toggle"]
        return state["next_pinned"]
    monkeypatch.setattr(db_layer, "toggle_pin_route", fake_toggle)

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


def test_get_pinned_routes_returns_empty_envelope_when_no_pins(routes_client):
    state, client = routes_client
    state["next_list"] = []
    resp = client.get("/api/pinned-routes")
    assert resp.status_code == 200
    assert resp.json() == {"routes": []}
    assert state["list_calls"] == 1


def test_get_pinned_routes_returns_routes_in_fifo_order(routes_client):
    """Endpoint surfaces the db_layer result verbatim — same FIFO order
    the helper emits. pinned_at is serialized as ISO-8601."""
    state, client = routes_client
    t1 = dt.datetime(2026, 5, 10, 9, 0, tzinfo=dt.timezone.utc)
    t2 = dt.datetime(2026, 5, 11, 9, 0, tzinfo=dt.timezone.utc)
    state["next_list"] = [
        {"route_path": "/log-buy",         "pinned_at": t1},
        {"route_path": "/active-campaign", "pinned_at": t2},
    ]
    resp = client.get("/api/pinned-routes")
    assert resp.status_code == 200
    body = resp.json()
    assert "routes" in body
    assert len(body["routes"]) == 2
    assert body["routes"][0]["route_path"] == "/log-buy"
    assert body["routes"][0]["pinned_at"] == t1.isoformat()
    assert body["routes"][1]["route_path"] == "/active-campaign"
    assert body["routes"][1]["pinned_at"] == t2.isoformat()


# ===========================================================================
# API endpoint — POST /api/pinned-routes/toggle
# ===========================================================================


def test_pinned_routes_toggle_happy_path_returns_pinned_state(routes_client):
    """Valid request → db_layer.toggle_pin_route called → {pinned: bool}."""
    state, client = routes_client
    state["next_pinned"] = True
    resp = client.post(
        "/api/pinned-routes/toggle",
        json={"route_path": "/log-buy"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"pinned": True}
    assert state["toggle_calls"] == ["/log-buy"]


def test_pinned_routes_toggle_idempotent_double_call(routes_client):
    """Two calls → server reports each new state. Idempotency is enforced
    in db_layer.toggle_pin_route; this pins that the endpoint surfaces
    the result verbatim, no double-toggle bouncing."""
    state, client = routes_client
    body = {"route_path": "/log-buy"}
    state["next_pinned"] = True
    r1 = client.post("/api/pinned-routes/toggle", json=body)
    state["next_pinned"] = False
    r2 = client.post("/api/pinned-routes/toggle", json=body)
    assert r1.json() == {"pinned": True}
    assert r2.json() == {"pinned": False}
    assert state["toggle_calls"] == ["/log-buy", "/log-buy"]


def test_pinned_routes_toggle_rejects_missing_route_path(routes_client):
    state, client = routes_client
    resp = client.post("/api/pinned-routes/toggle", json={})
    assert resp.status_code == 200
    assert "error" in resp.json()
    assert state["toggle_calls"] == []


def test_pinned_routes_toggle_rejects_non_string_route_path(routes_client):
    state, client = routes_client
    resp = client.post("/api/pinned-routes/toggle", json={"route_path": 42})
    assert resp.status_code == 200
    assert "error" in resp.json()
    assert state["toggle_calls"] == []


def test_pinned_routes_toggle_rejects_empty_string(routes_client):
    state, client = routes_client
    resp = client.post("/api/pinned-routes/toggle", json={"route_path": ""})
    assert resp.status_code == 200
    assert "error" in resp.json()
    assert state["toggle_calls"] == []


def test_pinned_routes_toggle_surfaces_value_error_as_400_like_body(routes_client):
    """db_layer raises ValueError on bad route_path (the regex check fires
    BEFORE the DB CHECK constraint would). Endpoint catches and returns
    {error: ...} so the frontend can display the message."""
    state, client = routes_client
    state["raise_on_toggle"] = ValueError("Invalid route_path: '/Log-Buy'")
    resp = client.post(
        "/api/pinned-routes/toggle",
        json={"route_path": "/Log-Buy"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body
    assert "Invalid route_path" in body["error"]
