"""Tests for the Phase 4 Weekly Snapshot persistence (Migration 028).

Covers two surfaces:

  1. db_layer helpers (fake-cursor unit tests, no live DB) — mirroring
     test_weekly_retros.py:
       - save_weekly_retro_snapshot INSERTs and serializes with view_url
       - save returns None when the parent retro isn't visible to the
         current tenant (RLS miss surfaces here as "row not found")
       - list returns serialized rows in (sort_order, created_at) order
       - list returns None when the retro doesn't exist / isn't visible
       - list returns [] when the retro exists but has no live snapshots
       - soft_delete UPDATEs deleted_at and returns True on hit, False
         on miss

  2. API endpoints (FastAPI TestClient with stubbed db_layer + r2):
       - POST 415 on disallowed MIME
       - POST 413 on oversize body
       - POST 404 when retro not visible (db returns None)
       - POST happy path returns the row
       - GET 200 returns the list
       - GET 404 when retro not visible
       - DELETE returns deleted=True on hit
       - DELETE 404 on miss / cross-tenant
       - view_url composition is correct

The live-DB constraint behavior (RLS, FK cascade on retro hard delete)
is enforced by the migration at apply-time and exercised post-deploy.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import io
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
# Fake cursor scaffolding (matches test_weekly_retros.py)
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, fetchones=None, fetchalls=None):
        self._fetchones = list(fetchones or [])
        self._fetchalls = list(fetchalls or [])
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, tuple(params) if params else ()))

    def fetchone(self):
        return self._fetchones.pop(0) if self._fetchones else None

    def fetchall(self):
        return self._fetchalls.pop(0) if self._fetchalls else []

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
    def __init__(self, cur): self._cur = cur
    def __enter__(self): return self._cur
    def __exit__(self, *a): return False


class _ConnCM:
    def __init__(self, conn): self._conn = conn
    def __enter__(self): return self._conn
    def __exit__(self, *a): return False


def _patch_conn(monkeypatch, cursor):
    conn = _FakeConn(cursor)
    monkeypatch.setattr(db_layer, "get_db_connection", lambda: _ConnCM(conn))
    return conn


# ===========================================================================
# db_layer.save_weekly_retro_snapshot
# ===========================================================================


def test_save_inserts_and_serializes_with_view_url(monkeypatch):
    """Parent retro is visible → INSERT runs, row is serialized with
    view_url composed from R2_PUBLIC_URL + storage_ref."""
    monkeypatch.setenv("R2_PUBLIC_URL", "https://cdn.example.com")
    cur = _FakeCursor(fetchones=[
        {"1": 1},                                    # retro ownership check
        {                                            # INSERT RETURNING
            "id": 99, "weekly_retro_id": 7,
            "storage_ref": "weekly_retros/7/abc.png",
            "file_name": "chart.png", "mime_type": "image/png",
            "file_size_bytes": 1234, "width": 800, "height": 600,
            "sort_order": 0, "caption": "",
            "created_at": datetime(2026, 5, 13, 12, 0, 0),
        },
    ])
    _patch_conn(monkeypatch, cur)

    row = db_layer.save_weekly_retro_snapshot(
        "CanSlim", 7, "weekly_retros/7/abc.png",
        file_name="chart.png", mime_type="image/png",
        file_size_bytes=1234, width=800, height=600,
    )
    assert row is not None
    assert row["id"] == 99
    assert row["weekly_retro_id"] == 7
    assert row["view_url"] == "https://cdn.example.com/weekly_retros/7/abc.png"
    assert row["sort_order"] == 0
    assert row["caption"] == ""
    # INSERT was emitted
    assert any("INSERT INTO weekly_retro_snapshots" in s for s, _ in cur.executed)


def test_save_returns_none_when_retro_not_visible(monkeypatch):
    """RLS-style miss: ownership check returns no row → save returns None.
    Caller maps to 404."""
    cur = _FakeCursor(fetchones=[None])  # ownership check empty
    _patch_conn(monkeypatch, cur)

    row = db_layer.save_weekly_retro_snapshot(
        "CanSlim", 999, "weekly_retros/999/x.png",
    )
    assert row is None
    # No INSERT happened
    assert not any("INSERT INTO weekly_retro_snapshots" in s for s, _ in cur.executed)


def test_save_view_url_falls_back_when_r2_public_url_unset(monkeypatch):
    """Local dev without R2_PUBLIC_URL configured — view_url falls back
    to the bare storage_ref instead of rendering a broken absolute URL."""
    monkeypatch.delenv("R2_PUBLIC_URL", raising=False)
    cur = _FakeCursor(fetchones=[
        {"1": 1},
        {
            "id": 1, "weekly_retro_id": 1,
            "storage_ref": "weekly_retros/1/x.png",
            "file_name": None, "mime_type": None,
            "file_size_bytes": None, "width": None, "height": None,
            "sort_order": 0, "caption": "",
            "created_at": datetime(2026, 5, 13),
        },
    ])
    _patch_conn(monkeypatch, cur)
    row = db_layer.save_weekly_retro_snapshot("CanSlim", 1, "weekly_retros/1/x.png")
    assert row["view_url"] == "weekly_retros/1/x.png"


# ===========================================================================
# db_layer.list_weekly_retro_snapshots
# ===========================================================================


def test_list_returns_rows_with_view_url(monkeypatch):
    monkeypatch.setenv("R2_PUBLIC_URL", "https://cdn.example.com")
    cur = _FakeCursor(
        fetchones=[{"1": 1}],
        fetchalls=[[
            {
                "id": 1, "weekly_retro_id": 5,
                "storage_ref": "weekly_retros/5/a.png",
                "file_name": "a.png", "mime_type": "image/png",
                "file_size_bytes": 100, "width": None, "height": None,
                "sort_order": 0, "caption": "",
                "created_at": datetime(2026, 5, 13, 10),
            },
            {
                "id": 2, "weekly_retro_id": 5,
                "storage_ref": "weekly_retros/5/b.png",
                "file_name": "b.png", "mime_type": "image/png",
                "file_size_bytes": 200, "width": None, "height": None,
                "sort_order": 0, "caption": "",
                "created_at": datetime(2026, 5, 13, 11),
            },
        ]],
    )
    _patch_conn(monkeypatch, cur)
    rows = db_layer.list_weekly_retro_snapshots("CanSlim", 5)
    assert rows is not None and len(rows) == 2
    assert rows[0]["view_url"] == "https://cdn.example.com/weekly_retros/5/a.png"
    assert rows[1]["view_url"] == "https://cdn.example.com/weekly_retros/5/b.png"


def test_list_returns_none_when_retro_not_visible(monkeypatch):
    cur = _FakeCursor(fetchones=[None])
    _patch_conn(monkeypatch, cur)
    assert db_layer.list_weekly_retro_snapshots("CanSlim", 999) is None


def test_list_returns_empty_when_no_snapshots(monkeypatch):
    cur = _FakeCursor(fetchones=[{"1": 1}], fetchalls=[[]])
    _patch_conn(monkeypatch, cur)
    rows = db_layer.list_weekly_retro_snapshots("CanSlim", 5)
    assert rows == []


# ===========================================================================
# db_layer.soft_delete_weekly_retro_snapshot
# ===========================================================================


def test_soft_delete_returns_true_on_hit(monkeypatch):
    cur = _FakeCursor(fetchones=[{"id": 42}])
    _patch_conn(monkeypatch, cur)
    assert db_layer.soft_delete_weekly_retro_snapshot(42) is True
    sql, _ = cur.executed[0]
    assert "UPDATE weekly_retro_snapshots" in sql
    assert "SET deleted_at = NOW()" in sql


def test_soft_delete_returns_false_on_miss(monkeypatch):
    cur = _FakeCursor(fetchones=[None])
    _patch_conn(monkeypatch, cur)
    assert db_layer.soft_delete_weekly_retro_snapshot(42) is False


# ===========================================================================
# API endpoints — TestClient with stubbed db_layer + r2
# ===========================================================================
#
# Auth: middleware uses JWT_SECRET; default to the test secret across tests
# via a fixture that patches os.environ before importing api.main.


@pytest.fixture
def client(monkeypatch):
    # AUTH_SECRET is read at import time → must patch the module attribute
    # AND the env var so any subsequent reads see the test secret.
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    from api import main as api_main
    monkeypatch.setattr(api_main, "AUTH_SECRET", _TEST_SECRET)
    # Force R2 "available" so the upload endpoint takes the happy path.
    monkeypatch.setattr(api_main, "_is_r2_available", lambda: True)
    return TestClient(api_main.app), api_main


def test_upload_rejects_disallowed_mime(client, monkeypatch):
    tc, api_main = client
    # No db call expected — endpoint short-circuits on MIME.
    r = tc.post(
        "/api/weekly-retros/5/snapshots",
        files={"file": ("note.txt", b"hello", "text/plain")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 415
    body = r.json()
    assert body["detail"]["error"] == "unsupported_media_type"


def test_upload_rejects_oversize(client, monkeypatch):
    tc, api_main = client
    # 15MB + 1 byte
    big = b"x" * (15 * 1024 * 1024 + 1)
    r = tc.post(
        "/api/weekly-retros/5/snapshots",
        files={"file": ("big.png", big, "image/png")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 413
    body = r.json()
    assert body["detail"]["error"] == "file_too_large"
    assert body["detail"]["limit_bytes"] == 15 * 1024 * 1024


def test_upload_returns_404_when_retro_not_owned(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(api_main.r2, "upload_blob",
                        lambda f, key, content_type=None: key)
    monkeypatch.setattr(api_main.db, "save_weekly_retro_snapshot",
                        lambda *a, **kw: None)
    r = tc.post(
        "/api/weekly-retros/999/snapshots",
        files={"file": ("a.png", b"\x89PNG fake", "image/png")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "retro_not_found"


def test_upload_happy_path_returns_row(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(api_main.r2, "upload_blob",
                        lambda f, key, content_type=None: key)

    captured = {}
    def _save(portfolio, retro_id, storage_ref, **kw):
        captured["portfolio"] = portfolio
        captured["retro_id"] = retro_id
        captured["storage_ref"] = storage_ref
        captured.update(kw)
        return {
            "id": 1, "weekly_retro_id": retro_id, "storage_ref": storage_ref,
            "view_url": f"https://cdn.example.com/{storage_ref}",
            "file_name": kw.get("file_name"), "mime_type": kw.get("mime_type"),
            "file_size_bytes": kw.get("file_size_bytes"),
            "width": kw.get("width"), "height": kw.get("height"),
            "sort_order": 0, "caption": "",
            "created_at": "2026-05-13T12:00:00",
        }
    monkeypatch.setattr(api_main.db, "save_weekly_retro_snapshot", _save)

    r = tc.post(
        "/api/weekly-retros/7/snapshots",
        files={"file": ("chart.png", b"\x89PNG fakebytes", "image/png")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == 1
    assert body["weekly_retro_id"] == 7
    assert body["storage_ref"].startswith("weekly_retros/7/")
    assert body["storage_ref"].endswith(".png")
    assert body["mime_type"] == "image/png"
    assert captured["mime_type"] == "image/png"
    assert captured["file_name"] == "chart.png"


def test_list_returns_rows(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(
        api_main.db, "list_weekly_retro_snapshots",
        lambda portfolio, retro_id: [
            {"id": 1, "weekly_retro_id": retro_id, "view_url": "u1",
             "storage_ref": "k1", "file_name": None, "mime_type": None,
             "file_size_bytes": None, "width": None, "height": None,
             "sort_order": 0, "caption": "", "created_at": "2026-05-13"},
        ],
    )
    r = tc.get("/api/weekly-retros/5/snapshots?portfolio=CanSlim",
               headers=_auth_headers())
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["view_url"] == "u1"


def test_list_returns_404_when_retro_missing(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(api_main.db, "list_weekly_retro_snapshots",
                        lambda *a, **kw: None)
    r = tc.get("/api/weekly-retros/999/snapshots?portfolio=CanSlim",
               headers=_auth_headers())
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "retro_not_found"


def test_delete_returns_deleted_true(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(api_main.db, "soft_delete_weekly_retro_snapshot",
                        lambda sid: True)
    r = tc.delete("/api/weekly-retros/snapshots/42", headers=_auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert body == {"deleted": True, "id": 42}


def test_delete_returns_404_on_miss(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(api_main.db, "soft_delete_weekly_retro_snapshot",
                        lambda sid: False)
    r = tc.delete("/api/weekly-retros/snapshots/9999", headers=_auth_headers())
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "snapshot_not_found"
