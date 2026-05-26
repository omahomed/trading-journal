"""Tests for the Phase 4.1 inline thoughts-image upload endpoint.

POST /api/weekly-retros/{retro_id}/thoughts-images

Surface tested:
  - 415 on disallowed MIME (text/plain)
  - 413 on body > 15MB
  - 404 when retro is not visible to the caller (ownership miss)
  - 200 happy path returns {"view_url": "..."} with the R2 prefix
  - Object key pattern includes the "thoughts/" segment so future
    cleanup can differentiate inline images from gallery snapshots
  - r2.upload_blob is called with the correct content_type
  - 415 fires BEFORE the ownership check (cheaper rejection)
"""
from __future__ import annotations

import jwt
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "11111111-2222-3333-4444-555555555555"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(monkeypatch):
    """TestClient fixture matching test_weekly_retro_snapshots.py:
    patches AUTH_SECRET both at the env var AND at the import-time
    captured module attribute. Stubs _is_r2_available so the upload
    path takes the happy branch."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    from api import main as api_main
    monkeypatch.setattr(api_main, "AUTH_SECRET", _TEST_SECRET)
    monkeypatch.setattr(api_main, "_is_r2_available", lambda: True)
    monkeypatch.setenv("R2_PUBLIC_URL", "https://cdn.example.com")
    return TestClient(api_main.app), api_main


def test_upload_rejects_disallowed_mime(client):
    tc, _ = client
    r = tc.post(
        "/api/weekly-retros/5/thoughts-images",
        files={"file": ("note.txt", b"hello world", "text/plain")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 415
    assert r.json()["detail"]["error"] == "unsupported_media_type"


def test_upload_rejects_oversize(client):
    tc, _ = client
    big = b"x" * (15 * 1024 * 1024 + 1)  # 15MB + 1 byte
    r = tc.post(
        "/api/weekly-retros/5/thoughts-images",
        files={"file": ("big.png", big, "image/png")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 413
    assert r.json()["detail"]["error"] == "file_too_large"
    assert r.json()["detail"]["limit_bytes"] == 15 * 1024 * 1024


def test_upload_returns_404_when_retro_not_owned(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(api_main.db, "verify_retro_ownership",
                        lambda portfolio, retro_id: False)
    r = tc.post(
        "/api/weekly-retros/999/thoughts-images",
        files={"file": ("a.png", b"\x89PNG fake", "image/png")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "retro_not_found"


def test_upload_happy_path_returns_view_url(client, monkeypatch):
    tc, api_main = client
    monkeypatch.setattr(api_main.db, "verify_retro_ownership",
                        lambda portfolio, retro_id: True)
    captured = {}
    def _upload_blob(file_obj, key, content_type=None):
        captured["key"] = key
        captured["content_type"] = content_type
        return key
    monkeypatch.setattr(api_main.r2, "upload_blob", _upload_blob)

    r = tc.post(
        "/api/weekly-retros/7/thoughts-images",
        files={"file": ("chart.png", b"\x89PNG fakebytes", "image/png")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "view_url" in body
    assert body["view_url"].startswith("https://cdn.example.com/")
    # Object key uses the thoughts/ subprefix so future cleanup can
    # differentiate inline images from gallery snapshots.
    assert "/thoughts/" in captured["key"]
    assert captured["key"].startswith("weekly_retros/7/thoughts/")
    assert captured["key"].endswith(".png")
    assert captured["content_type"] == "image/png"


def test_upload_uses_thoughts_subprefix_for_jpeg(client, monkeypatch):
    """Sanity: different MIME → correct extension, prefix unchanged."""
    tc, api_main = client
    monkeypatch.setattr(api_main.db, "verify_retro_ownership",
                        lambda portfolio, retro_id: True)
    captured = {}
    monkeypatch.setattr(api_main.r2, "upload_blob",
                        lambda f, key, content_type=None: captured.setdefault("key", key) or key)
    r = tc.post(
        "/api/weekly-retros/3/thoughts-images",
        files={"file": ("photo.jpg", b"\xff\xd8\xff jpeg-fake", "image/jpeg")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 200
    assert captured["key"].startswith("weekly_retros/3/thoughts/")
    assert captured["key"].endswith(".jpg")


def test_415_fires_before_ownership_check(client, monkeypatch):
    """MIME rejection happens before the DB lookup — we should NEVER
    call verify_retro_ownership for a clearly bad MIME. Cheap rejection
    is a small but meaningful denial-of-service mitigation."""
    tc, api_main = client
    ownership_calls = []
    monkeypatch.setattr(
        api_main.db, "verify_retro_ownership",
        lambda portfolio, retro_id: ownership_calls.append((portfolio, retro_id)) or True,
    )
    r = tc.post(
        "/api/weekly-retros/5/thoughts-images",
        files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")},
        data={"portfolio": "CanSlim"},
        headers=_auth_headers(),
    )
    assert r.status_code == 415
    assert ownership_calls == []
