"""Tests for /api/images/upload — MarketSurge Vision OCR background-task path.

Background: prior to 2026-06-03 the upload handler ran
vision_extract.extract_fundamentals inline before returning. The Claude
Vision API call is slow (5-15s) and has been observed to hang, which
left the Log Buy "Saving…" state stuck until the user refreshed (which
also killed the in-flight upload, so charts went missing).

The fix moves Vision OCR to a FastAPI BackgroundTask: the upload response
returns once R2 + DB metadata are persisted; fundamentals land in
trade_fundamentals when Vision finishes (or fail silently if Vision
errors — the chart itself is durable).

Coverage:
  1. MarketSurge upload response carries fundamentals=null even when the
     Vision extractor would have returned a payload — proves the Vision
     call is NOT awaited inline.
  2. Vision still runs (as a background task) after the response is
     sent — proves the work isn't lost.
  3. Non-MarketSurge uploads (entry, position_change) don't schedule the
     Vision task at all — only MarketSurge does.
  4. R2 upload failure short-circuits with a clean error (no Vision
     attempt).
"""
from __future__ import annotations

import io
from typing import Any
from unittest.mock import MagicMock

import jwt
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def stubbed(monkeypatch):
    """Patch R2, DB, and Vision so we can exercise upload_image without
    real network / DB / Anthropic calls."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)
    monkeypatch.setattr(main, "_is_r2_available", lambda: True)

    state: dict[str, Any] = {
        "r2_calls": [],
        "save_image_calls": [],
        "save_fundamentals_calls": [],
        "vision_calls": [],
        "vision_returns": {"PE_ratio": 25, "EPS_rating": 95},
        "r2_object_key": "uploads/test-key.png",
        "image_id": 42,
    }

    def fake_r2_upload(file_like, portfolio, trade_id, ticker, save_type):
        state["r2_calls"].append((portfolio, trade_id, ticker, save_type))
        return state["r2_object_key"]

    monkeypatch.setattr(main.r2, "upload_image", fake_r2_upload)
    monkeypatch.setattr(
        main.db, "save_trade_image",
        lambda *args, **kw: state["save_image_calls"].append(args) or state["image_id"],
    )
    monkeypatch.setattr(
        main.db, "save_trade_fundamentals",
        lambda *args, **kw: state["save_fundamentals_calls"].append(args),
    )

    # Install a fake vision_extract module so the background task can
    # import it cleanly without pulling in anthropic.
    import sys
    import types
    fake_vision = types.ModuleType("vision_extract")

    def fake_extract(image_bytes, file_name="image.png"):
        state["vision_calls"].append((len(image_bytes), file_name))
        return state["vision_returns"]
    fake_vision.extract_fundamentals = fake_extract  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vision_extract", fake_vision)

    if hasattr(main.limiter, "enabled"):
        original_enabled = main.limiter.enabled
        main.limiter.enabled = False
    else:
        original_enabled = None

    client = TestClient(main.app, headers=_auth_headers())
    try:
        yield state, client, main
    finally:
        if original_enabled is not None:
            main.limiter.enabled = original_enabled


def _do_upload(client: TestClient, image_type: str = "marketsurge"):
    files = {"file": ("test.png", io.BytesIO(b"fake-image-bytes"), "image/png")}
    data = {
        "portfolio": "CanSlim",
        "trade_id": "202606-001",
        "ticker": "AAPL",
        "image_type": image_type,
    }
    return client.post("/api/images/upload", files=files, data=data)


def test_marketsurge_response_returns_null_fundamentals_inline(stubbed):
    """The upload response carries fundamentals=null — Vision did not
    block the response. The Vision spy may still run as a background
    task; that's separately covered. Here we lock the response shape."""
    state, client, _ = stubbed
    r = _do_upload(client, image_type="marketsurge")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["image_id"] == state["image_id"]
    assert body["object_key"] == state["r2_object_key"]
    # Key invariant — fundamentals are NOT returned inline anymore. The
    # frontend has its own /api/fundamentals/{id} polling path; the
    # synchronous return value is intentionally null.
    assert body["fundamentals"] is None


def test_marketsurge_vision_runs_after_response_via_background_task(stubbed):
    """TestClient drains FastAPI background tasks before returning, so
    by the time .post() resolves, Vision should have been called once
    and fundamentals should have been persisted."""
    state, client, _ = stubbed
    _do_upload(client, image_type="marketsurge")
    assert len(state["vision_calls"]) == 1
    assert state["vision_calls"][0][1] == "test.png"
    assert len(state["save_fundamentals_calls"]) == 1


def test_non_marketsurge_upload_skips_vision_entirely(stubbed):
    """Entry / position_change uploads don't schedule Vision."""
    state, client, _ = stubbed
    for image_type in ("entry", "position_change"):
        state["vision_calls"].clear()
        state["save_fundamentals_calls"].clear()
        r = _do_upload(client, image_type=image_type)
        assert r.status_code == 200, r.text
        assert state["vision_calls"] == []
        assert state["save_fundamentals_calls"] == []


def test_r2_failure_short_circuits_no_vision_attempt(stubbed, monkeypatch):
    """If R2 upload fails, the handler returns an error WITHOUT scheduling
    Vision (no image_id to attach fundamentals to anyway)."""
    state, client, main = stubbed
    monkeypatch.setattr(main.r2, "upload_image", lambda *a, **kw: None)
    r = _do_upload(client, image_type="marketsurge")
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert "R2" in body["error"]
    assert state["vision_calls"] == []
