"""Tests for the SR8 Cascade Monitor endpoint.

The endpoint wraps mors.monitor.analyze() per position in
mors/positions.json. These tests mock mors.monitor.analyze (the heavy
function — it replays cascade history) so we can assert on the
endpoint's enrichment + summary aggregation without spinning up the
MORS engine or hitting yfinance.

Coverage:
  1. Happy path — analyze() succeeds for all positions; response shape
     is the documented {summary, positions, meta} envelope with
     correctly-aggregated summary chip values.
  2. Per-position fetch failure — one position's analyze() raises;
     other positions still render normally; the failed position
     buckets into a row with fetch_failed=true.
  3. is_action derivation — terminated rows are actionable; signal_today
     + delta_dollars over the 500 tolerance is actionable; below
     tolerance is hold.
  4. early_warn computed when held position is within 2 points of its
     tier target.
  5. /api/sr8/refresh wires refresh=True through to analyze().
  6. refresh rejects nlv <= 0.
  7. Empty positions.json returns an empty payload (not a crash).
"""
from __future__ import annotations

from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def _analyze_result(
    ticker: str,
    *,
    shares_held: float = 100.0,
    avg_price: float = 100.0,
    current_price: float = 120.0,
    current_pct_nlv: float = 15.0,
    cascade_core: int = 15,
    tier_pct_nlv: float = 15.0,
    target_dollars: float = 60000.0,
    delta_dollars: float = 0.0,
    delta_shares: int = 0,
    last_signal: str = "GREEN",
    signal_today: bool = False,
    terminated: bool = False,
    phase: int = 2,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "shares_held": shares_held,
        "avg_price": avg_price,
        "current_price": current_price,
        "current_dollars": shares_held * current_price,
        "current_pct_nlv": current_pct_nlv,
        "cascade_core": cascade_core,
        "tier_pct_nlv": tier_pct_nlv,
        "target_dollars": target_dollars,
        "delta_dollars": delta_dollars,
        "delta_shares": delta_shares,
        "unreal_dollars": (current_price - avg_price) * shares_held,
        "unreal_pct": (current_price - avg_price) / avg_price * 100,
        "last_signal": last_signal,
        "last_signal_date": "2026-04-13",
        "last_bar_date": "2026-04-18",
        "signal_today": signal_today,
        "terminated": terminated,
        "phase": phase,
    }


@pytest.fixture
def stubbed(monkeypatch):
    """Patch the endpoint's positions loader + the MORS analyze import
    so tests can drive both sides without disk / yfinance access."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "positions": [],
        "analyze_by_ticker": {},  # ticker → (callable | result_dict | exception)
        "analyze_calls": [],       # capture (pos_ticker, nlv, refresh)
    }

    monkeypatch.setattr(main, "_sr8_load_positions", lambda: list(state["positions"]))

    # Replace the module-level import indirection inside _sr8_run_monitor.
    # That helper does `from mors.monitor import analyze as mors_analyze`
    # at call time, so we install a fake mors.monitor module.
    import sys
    import types

    fake_mod = types.ModuleType("mors.monitor")

    def fake_analyze(pos: dict[str, Any], nlv: float, refresh: bool = False):
        state["analyze_calls"].append((pos.get("ticker"), nlv, refresh))
        entry = state["analyze_by_ticker"].get(pos.get("ticker"))
        if entry is None:
            # default: succeed with stock GREEN/hold result
            return _analyze_result(pos.get("ticker", ""))
        if isinstance(entry, Exception):
            raise entry
        if callable(entry):
            return entry(pos, nlv, refresh)
        return entry  # dict

    fake_mod.analyze = fake_analyze  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mors.monitor", fake_mod)

    # Anchor mtime helper so meta.fetched_at is stable across CI.
    monkeypatch.setattr(main, "_sr8_fetched_at_iso", lambda: "2026-04-13T16:00:00")

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


def test_happy_path_response_envelope(stubbed):
    """All 3 positions analyze cleanly. Response shape = the documented
    envelope with summary + positions + meta. cascade_breakdown reflects
    the per-position cascade_core."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "AAA", "b1_date": "2026-04-01", "b1_price": 100, "shares_held": 100, "avg_price": 100},
        {"ticker": "BBB", "b1_date": "2026-04-02", "b1_price": 200, "shares_held": 50, "avg_price": 200},
        {"ticker": "CCC", "b1_date": "2026-04-03", "b1_price": 300, "shares_held": 30, "avg_price": 300},
    ]
    state["analyze_by_ticker"] = {
        "AAA": _analyze_result("AAA", cascade_core=20, current_pct_nlv=25.0),
        "BBB": _analyze_result("BBB", cascade_core=15, current_pct_nlv=12.0),
        "CCC": _analyze_result("CCC", cascade_core=15, current_pct_nlv=8.0),
    }

    r = client.get("/api/sr8/monitor?nlv=500000")
    assert r.status_code == 200, r.text
    body = r.json()

    # Envelope keys
    assert set(body.keys()) >= {"summary", "positions", "meta"}
    assert body["meta"]["nlv"] == 500000
    assert body["meta"]["fetched_at"] == "2026-04-13T16:00:00"

    # Positions
    assert len(body["positions"]) == 3
    tickers = [p["ticker"] for p in body["positions"]]
    assert tickers == ["AAA", "BBB", "CCC"]
    for p in body["positions"]:
        assert p["fetch_failed"] is False
        assert "b1_date" in p and "b1_price" in p  # pass-through fields

    # Summary
    assert body["summary"]["total_positions"] == 3
    assert body["summary"]["cascade_breakdown"]["cascade_20"] == 1
    assert body["summary"]["cascade_breakdown"]["cascade_15"] == 2


def test_per_position_fetch_failure_isolated(stubbed):
    """One position's analyze() raises; the other two render normally.
    The failed row's fetch_failed=true and current_price is null."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "AAA", "b1_date": "2026-04-01", "b1_price": 100, "shares_held": 100, "avg_price": 100},
        {"ticker": "FAIL", "b1_date": "2026-04-01", "b1_price": 50, "shares_held": 200, "avg_price": 50},
        {"ticker": "CCC", "b1_date": "2026-04-03", "b1_price": 300, "shares_held": 30, "avg_price": 300},
    ]
    state["analyze_by_ticker"] = {
        "AAA": _analyze_result("AAA"),
        "FAIL": RuntimeError("yfinance fetch failed"),
        "CCC": _analyze_result("CCC"),
    }

    r = client.get("/api/sr8/monitor?nlv=500000")
    assert r.status_code == 200, r.text
    body = r.json()
    rows_by_ticker = {p["ticker"]: p for p in body["positions"]}

    assert rows_by_ticker["AAA"]["fetch_failed"] is False
    assert rows_by_ticker["CCC"]["fetch_failed"] is False
    assert rows_by_ticker["FAIL"]["fetch_failed"] is True
    assert rows_by_ticker["FAIL"]["current_price"] is None
    assert "yfinance fetch failed" in rows_by_ticker["FAIL"]["fetch_error"]
    # Failed rows DO carry pass-through b1 fields so the UI can still
    # render the ticker context.
    assert rows_by_ticker["FAIL"]["shares_held"] == 200


def test_is_action_terminated(stubbed):
    """Terminated (weekly GD) is always actionable."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "TERM", "b1_date": "2026-04-01", "b1_price": 100, "shares_held": 100, "avg_price": 100},
    ]
    state["analyze_by_ticker"] = {
        "TERM": _analyze_result("TERM", terminated=True, last_signal="TERMINATED"),
    }

    r = client.get("/api/sr8/monitor?nlv=500000")
    body = r.json()
    assert body["positions"][0]["is_action"] is True


def test_is_action_signal_today_above_tolerance(stubbed):
    """signal_today=True AND delta_dollars > 500 → actionable."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "AAA", "b1_date": "2026-04-01", "b1_price": 100, "shares_held": 100, "avg_price": 100},
        {"ticker": "BBB", "b1_date": "2026-04-02", "b1_price": 200, "shares_held": 50, "avg_price": 200},
    ]
    state["analyze_by_ticker"] = {
        # signal_today + above tolerance → action
        "AAA": _analyze_result("AAA", signal_today=True, delta_dollars=12345.0, current_pct_nlv=22.0),
        # signal_today but BELOW tolerance ($300 < $500 epsilon) → hold
        "BBB": _analyze_result("BBB", signal_today=True, delta_dollars=300.0),
    }

    r = client.get("/api/sr8/monitor?nlv=500000")
    body = r.json()
    rows = {p["ticker"]: p for p in body["positions"]}
    assert rows["AAA"]["is_action"] is True
    assert rows["BBB"]["is_action"] is False


def test_early_warn_within_two_points_of_target(stubbed):
    """Held position is NEAR when 0 <= (target - current_pct_nlv) <= 2.
    Tier target 11.25, current 10.0 → diff 1.25 → NEAR."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "NEAR", "b1_date": "2026-04-01", "b1_price": 100, "shares_held": 100, "avg_price": 100},
        {"ticker": "FAR", "b1_date": "2026-04-02", "b1_price": 200, "shares_held": 50, "avg_price": 200},
    ]
    state["analyze_by_ticker"] = {
        "NEAR": _analyze_result(
            "NEAR", tier_pct_nlv=11.25, current_pct_nlv=10.0, signal_today=False
        ),
        "FAR": _analyze_result(
            "FAR", tier_pct_nlv=15.0, current_pct_nlv=5.0, signal_today=False
        ),
    }

    r = client.get("/api/sr8/monitor?nlv=500000")
    rows = {p["ticker"]: p for p in r.json()["positions"]}
    assert rows["NEAR"]["early_warn"] is True
    assert rows["FAR"]["early_warn"] is False


def test_refresh_endpoint_passes_refresh_true(stubbed):
    """POST /api/sr8/refresh invokes analyze() with refresh=True so the
    engine pulls fresh weekly closes from yfinance."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "AAA", "b1_date": "2026-04-01", "b1_price": 100, "shares_held": 100, "avg_price": 100},
    ]

    r = client.post("/api/sr8/refresh", json={"nlv": 500000})
    assert r.status_code == 200, r.text
    # The fake analyze recorded its calls — confirm refresh=True propagated.
    assert state["analyze_calls"] == [("AAA", 500000.0, True)]


def test_refresh_rejects_non_positive_nlv(stubbed):
    """nlv must be > 0 — body payload validation."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "AAA", "b1_date": "2026-04-01", "b1_price": 100, "shares_held": 100, "avg_price": 100},
    ]

    r = client.post("/api/sr8/refresh", json={"nlv": 0})
    assert r.status_code == 200  # endpoint pattern returns error in body, not HTTP code
    assert "error" in r.json()


def test_empty_positions_returns_clean_envelope(stubbed):
    """No positions in positions.json → endpoint returns the envelope
    with zero-state values, no crash."""
    state, client = stubbed
    state["positions"] = []

    r = client.get("/api/sr8/monitor?nlv=500000")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["total_positions"] == 0
    assert body["summary"]["flagged_count"] == 0
    assert body["positions"] == []
