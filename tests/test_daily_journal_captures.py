"""Tests for the Phase 7 Daily Journal Captures persistence (Migration 031).

Covers two surfaces, mirroring test_weekly_retro_snapshots.py:

  1. db_layer helpers (fake-cursor unit tests, no live DB):
       - save_daily_journal_capture INSERTs and serializes with view_url
       - save returns None when the parent journal isn't visible to the
         current tenant (RLS miss surfaces here as "row not found")
       - list returns serialized rows in (sort_order, created_at) order
       - list returns None when the journal doesn't exist / isn't visible
       - list returns [] when the journal exists but has no live captures
       - soft_delete UPDATEs deleted_at and returns True on hit, False
         on miss
       - the defensive `tj.deleted_at IS NULL` predicate is asserted on
         the ownership SQL (Phase 7 audit concern #2 — full unique
         constraint on (portfolio_id, day) means soft-deleted rows
         linger; captures must not attach to them)

  2. API endpoints (FastAPI TestClient with stubbed db_layer + r2):
       - POST 415 / 413 / 404 / happy path
       - GET 200 / 404
       - DELETE deleted=True / 404 on miss
       - POST /api/snapshots/upload returns 410 for snapshot_type="note"
         (the Phase 7 retirement of the legacy EOD upload routing)
"""
from __future__ import annotations

from datetime import datetime

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
# Fake cursor scaffolding (matches test_weekly_retro_snapshots.py)
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
# db_layer.save_daily_journal_capture
# ===========================================================================


def test_save_inserts_and_serializes_with_view_url(monkeypatch):
    """Parent journal is visible → INSERT runs, row is serialized with
    view_url composed from R2_PUBLIC_URL + storage_ref."""
    monkeypatch.setenv("R2_PUBLIC_URL", "https://cdn.example.com")
    cur = _FakeCursor(fetchones=[
        {"1": 1},                                    # journal ownership check
        {                                            # INSERT RETURNING
            "id": 99, "daily_journal_id": 7,
            "storage_ref": "daily_journal/7/abc.png",
            "file_name": "chart.png", "mime_type": "image/png",
            "file_size_bytes": 1234, "width": 800, "height": 600,
            "sort_order": 0, "caption": "",
            "created_at": datetime(2026, 5, 15, 12, 0, 0),
        },
    ])
    _patch_conn(monkeypatch, cur)

    row = db_layer.save_daily_journal_capture(
        "CanSlim", 7, "daily_journal/7/abc.png",
        file_name="chart.png", mime_type="image/png",
        file_size_bytes=1234, width=800, height=600,
    )
    assert row is not None
    assert row["id"] == 99
    assert row["daily_journal_id"] == 7
    assert row["view_url"] == "https://cdn.example.com/daily_journal/7/abc.png"
    assert row["sort_order"] == 0
    assert row["caption"] == ""
    # INSERT was emitted
    assert any("INSERT INTO daily_journal_captures" in s for s, _ in cur.executed)


def test_save_returns_none_when_journal_not_visible(monkeypatch):
    """RLS-style miss: ownership check returns no row → save returns None.
    Caller maps to 404."""
    cur = _FakeCursor(fetchones=[None])  # ownership check empty
    _patch_conn(monkeypatch, cur)

    row = db_layer.save_daily_journal_capture(
        "CanSlim", 999, "daily_journal/999/x.png",
    )
    assert row is None
    # No INSERT happened
    assert not any("INSERT INTO daily_journal_captures" in s for s, _ in cur.executed)


def test_ownership_check_filters_soft_deleted_journals(monkeypatch):
    """Phase 7 defensive check (audit Section 1, Concern 2): the SQL
    that verifies journal ownership must include `j.deleted_at IS NULL`
    so that captures can't be attached to soft-deleted journal rows
    that still occupy the (portfolio_id, day) unique slot."""
    cur = _FakeCursor(fetchones=[None])  # missing because of deleted_at
    _patch_conn(monkeypatch, cur)
    db_layer.save_daily_journal_capture("CanSlim", 42, "daily_journal/42/x.png")
    # First execute call is the ownership check
    sql, _ = cur.executed[0]
    assert "trading_journal" in sql
    assert "deleted_at IS NULL" in sql


def test_save_view_url_falls_back_when_r2_public_url_unset(monkeypatch):
    """Local dev without R2_PUBLIC_URL configured — view_url falls back
    to the bare storage_ref instead of rendering a broken absolute URL."""
    monkeypatch.delenv("R2_PUBLIC_URL", raising=False)
    cur = _FakeCursor(fetchones=[
        {"1": 1},
        {
            "id": 1, "daily_journal_id": 1,
            "storage_ref": "daily_journal/1/x.png",
            "file_name": None, "mime_type": None,
            "file_size_bytes": None, "width": None, "height": None,
            "sort_order": 0, "caption": "",
            "created_at": datetime(2026, 5, 15),
        },
    ])
    _patch_conn(monkeypatch, cur)
    row = db_layer.save_daily_journal_capture("CanSlim", 1, "daily_journal/1/x.png")
    assert row["view_url"] == "daily_journal/1/x.png"


# ===========================================================================
# db_layer.list_daily_journal_captures
# ===========================================================================


def test_list_returns_rows_with_view_url(monkeypatch):
    monkeypatch.setenv("R2_PUBLIC_URL", "https://cdn.example.com")
    cur = _FakeCursor(
        fetchones=[{"1": 1}],
        fetchalls=[[
            {
                "id": 1, "daily_journal_id": 5,
                "storage_ref": "daily_journal/5/a.png",
                "file_name": "a.png", "mime_type": "image/png",
                "file_size_bytes": 100, "width": None, "height": None,
                "sort_order": 0, "caption": "",
                "created_at": datetime(2026, 5, 15, 10),
            },
            {
                "id": 2, "daily_journal_id": 5,
                "storage_ref": "daily_journal/5/b.png",
                "file_name": "b.png", "mime_type": "image/png",
                "file_size_bytes": 200, "width": None, "height": None,
                "sort_order": 0, "caption": "",
                "created_at": datetime(2026, 5, 15, 11),
            },
        ]],
    )
    _patch_conn(monkeypatch, cur)
    rows = db_layer.list_daily_journal_captures("CanSlim", 5)
    assert rows is not None and len(rows) == 2
    assert rows[0]["view_url"] == "https://cdn.example.com/daily_journal/5/a.png"
    assert rows[1]["view_url"] == "https://cdn.example.com/daily_journal/5/b.png"


def test_list_returns_none_when_journal_not_visible(monkeypatch):
    cur = _FakeCursor(fetchones=[None])
    _patch_conn(monkeypatch, cur)
    assert db_layer.list_daily_journal_captures("CanSlim", 999) is None


def test_list_returns_empty_when_no_captures(monkeypatch):
    cur = _FakeCursor(fetchones=[{"1": 1}], fetchalls=[[]])
    _patch_conn(monkeypatch, cur)
    rows = db_layer.list_daily_journal_captures("CanSlim", 5)
    assert rows == []


# ===========================================================================
# db_layer.soft_delete_daily_journal_capture
# ===========================================================================


def test_soft_delete_returns_true_on_hit(monkeypatch):
    cur = _FakeCursor(fetchones=[{"id": 42}])
    _patch_conn(monkeypatch, cur)
    assert db_layer.soft_delete_daily_journal_capture(42) is True
    sql, _ = cur.executed[0]
    assert "UPDATE daily_journal_captures" in sql
    assert "SET deleted_at = NOW()" in sql


def test_soft_delete_returns_false_on_miss(monkeypatch):
    cur = _FakeCursor(fetchones=[None])
    _patch_conn(monkeypatch, cur)
    assert db_layer.soft_delete_daily_journal_capture(42) is False


# ===========================================================================
# Daily score → letter helper
# ===========================================================================


def test_daily_score_to_letter():
    assert db_layer._daily_score_to_letter(5) == "A+"
    assert db_layer._daily_score_to_letter(4) == "A"
    assert db_layer._daily_score_to_letter(3) == "B"
    assert db_layer._daily_score_to_letter(2) == "C"
    assert db_layer._daily_score_to_letter(1) == "D"
    assert db_layer._daily_score_to_letter(0) is None
    assert db_layer._daily_score_to_letter(None) is None
    assert db_layer._daily_score_to_letter("not-a-number") is None


# ===========================================================================
# API endpoints — TestClient with stubbed db_layer + r2
# ===========================================================================


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    from api import main as api_main
    monkeypatch.setattr(api_main, "AUTH_SECRET", _TEST_SECRET)
    monkeypatch.setattr(api_main, "_is_r2_available", lambda: True)
    return TestClient(api_main.app), api_main


def test_upload_rejects_disallowed_mime(client):
    tc, _ = client
    r = tc.post(
        "/api/daily-journals/5/captures",
        files={"file": ("note.txt", b"hello", "text/plain")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 415
    body = r.json()
    assert body["detail"]["error"] == "unsupported_media_type"


def test_upload_rejects_oversize(client):
    tc, _ = client
    big = b"x" * (15 * 1024 * 1024 + 1)
    r = tc.post(
        "/api/daily-journals/5/captures",
        files={"file": ("big.png", big, "image/png")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 413
    body = r.json()
    assert body["detail"]["error"] == "file_too_large"


def test_upload_returns_404_when_journal_not_owned(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(api_main.r2, "upload_blob",
                        lambda f, key, content_type=None: key)
    monkeypatch.setattr(api_main.db, "save_daily_journal_capture",
                        lambda *a, **kw: None)
    r = tc.post(
        "/api/daily-journals/999/captures",
        files={"file": ("a.png", b"\x89PNG fake", "image/png")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "journal_not_found"


def test_upload_happy_path_returns_row(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(api_main.r2, "upload_blob",
                        lambda f, key, content_type=None: key)

    captured: dict = {}
    def _save(portfolio, journal_id, storage_ref, **kw):
        captured["portfolio"] = portfolio
        captured["journal_id"] = journal_id
        captured["storage_ref"] = storage_ref
        captured.update(kw)
        return {
            "id": 1, "daily_journal_id": journal_id, "storage_ref": storage_ref,
            "view_url": f"https://cdn.example.com/{storage_ref}",
            "file_name": kw.get("file_name"), "mime_type": kw.get("mime_type"),
            "file_size_bytes": kw.get("file_size_bytes"),
            "width": kw.get("width"), "height": kw.get("height"),
            "sort_order": 0, "caption": "",
            "created_at": "2026-05-15T12:00:00",
        }
    monkeypatch.setattr(api_main.db, "save_daily_journal_capture", _save)

    r = tc.post(
        "/api/daily-journals/7/captures",
        files={"file": ("chart.png", b"\x89PNG fakebytes", "image/png")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == 1
    assert body["daily_journal_id"] == 7
    assert body["storage_ref"].startswith("daily_journal/7/")
    assert body["storage_ref"].endswith(".png")
    assert body["mime_type"] == "image/png"
    assert captured["mime_type"] == "image/png"
    assert captured["file_name"] == "chart.png"


def test_list_returns_rows(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(
        api_main.db, "list_daily_journal_captures",
        lambda portfolio, journal_id: [
            {"id": 1, "daily_journal_id": journal_id, "view_url": "u1",
             "storage_ref": "k1", "file_name": None, "mime_type": None,
             "file_size_bytes": None, "width": None, "height": None,
             "sort_order": 0, "caption": "", "created_at": "2026-05-15"},
        ],
    )
    r = tc.get("/api/daily-journals/5/captures?portfolio=CanSlim",
               headers=_auth_headers())
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["view_url"] == "u1"


def test_list_returns_404_when_journal_missing(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(api_main.db, "list_daily_journal_captures",
                        lambda *a, **kw: None)
    r = tc.get("/api/daily-journals/999/captures?portfolio=CanSlim",
               headers=_auth_headers())
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "journal_not_found"


def test_delete_returns_deleted_true(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(api_main.db, "soft_delete_daily_journal_capture",
                        lambda cid: True)
    r = tc.delete("/api/daily-journals/captures/42", headers=_auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert body == {"deleted": True, "id": 42}


def test_delete_returns_404_on_miss(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(api_main.db, "soft_delete_daily_journal_capture",
                        lambda cid: False)
    r = tc.delete("/api/daily-journals/captures/9999", headers=_auth_headers())
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "capture_not_found"


# ===========================================================================
# Phase 7 retirement: POST /api/snapshots/upload no longer accepts notes
# ===========================================================================


def test_eod_upload_returns_410_for_note_snapshot_type(client):
    """The pre-Phase-7 path that wrote user-uploaded notes to
    trade_images.image_type='eod_note' is retired. Calls now return
    410 with a pointer to the new daily captures endpoint. Verifies the
    one frontend code path that still calls this surface gets a clear
    signal that it should switch to Daily Captures."""
    tc, _ = client
    r = tc.post(
        "/api/snapshots/upload",
        files={"file": ("a.png", b"\x89PNG fake", "image/png")},
        data={"portfolio": "CanSlim", "day": "2026-05-15", "snapshot_type": "note"},
        headers=_auth_headers(),
    )
    assert r.status_code == 410
    body = r.json()
    assert body["detail"]["error"] == "endpoint_retired"
    assert "daily-journals" in body["detail"]["message"]


# ===========================================================================
# Phase 7 regression fix — daily NotesRail tag batching
# ===========================================================================
# The user-reported regression on the daily rail's filter bar prompted a
# pin-down test: prove that list_daily_journals_rail attaches tags per
# item via _daily_journal_tags_batch. The investigation found the
# helper is already wired post-Phase-7 (db_layer.py:5873, :5979); these
# tests pin the contract so any future regression that breaks tag
# propagation fails loudly here.


def test_daily_journal_tags_batch_returns_attached_tags(monkeypatch):
    """The batch helper returns {journal_id: [{name, color}]} for every
    tag_assignments row whose entity_type='daily_journal' matches the
    requested ids. Tags are filtered to live rows (deleted_at IS NULL)."""
    cur = _FakeCursor(fetchalls=[[
        {"entity_id": 7, "name": "earnings", "color": "amber"},
        {"entity_id": 7, "name": "winner", "color": "emerald"},
        {"entity_id": 9, "name": "loss", "color": "rose"},
    ]])
    _patch_conn(monkeypatch, cur)
    out = db_layer._daily_journal_tags_batch([7, 8, 9])
    assert out[7] == [
        {"name": "earnings", "color": "amber"},
        {"name": "winner", "color": "emerald"},
    ]
    assert out[8] == []         # journaled day with no tags
    assert out[9] == [{"name": "loss", "color": "rose"}]
    # SQL filters by entity_type + deleted_at + tags.deleted_at.
    sql, params = cur.executed[0]
    assert "FROM tag_assignments a" in sql
    assert "entity_type = 'daily_journal'" in sql
    assert "deleted_at IS NULL" in sql
    assert params[0] == [7, 8, 9]


def test_daily_journal_tags_batch_returns_empty_dict_for_empty_input(monkeypatch):
    """No journal ids → no SQL fired, empty dict returned. Defends
    against the empty-ANY([]) pathological query."""
    cur = _FakeCursor()
    _patch_conn(monkeypatch, cur)
    out = db_layer._daily_journal_tags_batch([])
    assert out == {}
    assert cur.executed == []


def test_daily_score_to_letter_handles_negative_and_out_of_range_scores():
    """Defense against unexpected score values — 0 and out-of-range
    collapse to None (draft row in the rail's tri-state dot)."""
    assert db_layer._daily_score_to_letter(-1) is None
    assert db_layer._daily_score_to_letter(0) is None
    assert db_layer._daily_score_to_letter(99) is None       # out of range
