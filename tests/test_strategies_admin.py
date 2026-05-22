"""Tests for the Phase 2 strategies admin + retroactive-tagging endpoints.

Covers:
  - POST   /api/strategies            (founder-gated)
  - PUT    /api/strategies/{name}     (founder-gated)
  - PATCH  /api/trades/{id}/strategy  (any authed user)
  - POST   /api/trades/bulk-strategy  (any authed user)

Mirrors tests/test_log_buy_strategy.py: stubbed db_layer helpers,
FastAPI TestClient, rate-limiter disabled per-session.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"
_FOUNDER_ID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"  # matches main.FOUNDER_USER_ID default
_OTHER_USER_ID = "11111111-2222-3333-4444-555555555555"


def _token_for(user_id: str) -> str:
    return jwt.encode({"sub": user_id}, _TEST_SECRET, algorithm="HS256")


def _founder_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_token_for(_FOUNDER_ID)}"}


def _non_founder_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_token_for(_OTHER_USER_ID)}"}


@pytest.fixture
def stubbed(monkeypatch):
    """Yield (state, client) — same shape as test_log_buy_strategy.

    state knobs:
      strategies          — what db.load_strategies returns
      retag_result        — bool returned by db.update_trade_strategy
      bulk_result         — (updated_count, missing_ids) tuple returned by
                            db.bulk_update_trade_strategy
      create_raises       — exception class to raise from db.create_strategy
      create_returns      — dict returned on successful create
      update_returns      — dict returned by db.update_strategy

    state observations:
      retag_calls         — list of (portfolio, trade_id, strategy)
      bulk_calls          — list of (portfolio, trade_ids, strategy)
      create_calls        — list of dicts passed to db.create_strategy
      update_calls        — list of (name, fields_dict)
      audit_logs          — list of (action, trade_id, ticker, details)
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    import db_layer

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)
    monkeypatch.setattr(main, "FOUNDER_USER_ID", _FOUNDER_ID)

    state: dict[str, Any] = {
        # Post-Migration-038 shape. allowed_portfolio_names = None means
        # the strategy is universal; an explicit list narrows visibility.
        "strategies": [
            {"name": "CanSlim", "description": None, "color": "#6366f1", "is_active": True,
             "created_at": datetime(2026, 1, 1), "allowed_portfolio_names": ["CanSlim"]},
            {"name": "StockTalk", "description": None, "color": "#d97706", "is_active": True,
             "created_at": datetime(2026, 1, 2),
             "allowed_portfolio_names": ["CanSlim", "Long-Term Growth"]},
            {"name": "21eStrategy", "description": None, "color": "#0d9488", "is_active": True,
             "created_at": datetime(2026, 1, 3), "allowed_portfolio_names": None},
            {"name": "LongTerm", "description": None, "color": "#0284c7", "is_active": True,
             "created_at": datetime(2026, 1, 4),
             "allowed_portfolio_names": ["457B Plan", "Long-Term Growth"]},
            {"name": "50sStrategy", "description": None, "color": "#be185d", "is_active": True,
             "created_at": datetime(2026, 1, 5),
             "allowed_portfolio_names": ["457B Plan", "Long-Term Growth"]},
        ],
        "retag_result": True,
        "bulk_result": (0, []),
        "create_raises": None,
        "create_returns": None,
        "update_returns": None,
        "retag_calls": [],
        "bulk_calls": [],
        "create_calls": [],
        "update_calls": [],
        "audit_logs": [],
    }

    def fake_load_strategies(active_only=True, portfolio_name=None):
        rows = [s for s in state["strategies"] if not active_only or s["is_active"]]
        if portfolio_name is None:
            return rows
        # Mirror the SQL filter: NULL allowed_portfolio_names = universal;
        # explicit list narrows.
        return [
            s for s in rows
            if s.get("allowed_portfolio_names") is None
               or portfolio_name in (s.get("allowed_portfolio_names") or [])
        ]
    monkeypatch.setattr(db_layer, "load_strategies", fake_load_strategies)

    def fake_update_trade_strategy(portfolio, trade_id, strategy):
        state["retag_calls"].append((portfolio, trade_id, strategy))
        return state["retag_result"]
    monkeypatch.setattr(db_layer, "update_trade_strategy", fake_update_trade_strategy)

    def fake_bulk_update(portfolio, trade_ids, strategy):
        state["bulk_calls"].append((portfolio, list(trade_ids), strategy))
        return state["bulk_result"]
    monkeypatch.setattr(db_layer, "bulk_update_trade_strategy", fake_bulk_update)

    def fake_create_strategy(name, color, description=None, is_active=True):
        state["create_calls"].append({
            "name": name, "color": color, "description": description, "is_active": is_active,
        })
        if state["create_raises"] is not None:
            raise state["create_raises"]
        if state["create_returns"] is not None:
            return state["create_returns"]
        return {
            "name": name, "description": description, "color": color,
            "is_active": is_active, "created_at": datetime(2026, 5, 8),
        }
    monkeypatch.setattr(db_layer, "create_strategy", fake_create_strategy)

    def fake_update_strategy(name, **fields):
        state["update_calls"].append((name, dict(fields)))
        if state["update_returns"] is not None:
            return state["update_returns"]
        # Find the existing row + apply the patch in-memory.
        existing = next((s for s in state["strategies"] if s["name"] == name), None)
        if existing is None:
            return None
        merged = {**existing, **fields}
        return merged
    monkeypatch.setattr(db_layer, "update_strategy", fake_update_strategy)

    def fake_log_audit(*args, **kwargs):
        state["audit_logs"].append((args, kwargs))
    monkeypatch.setattr(db_layer, "log_audit", fake_log_audit)

    original_enabled = getattr(main.limiter, "enabled", True)
    if hasattr(main.limiter, "enabled"):
        main.limiter.enabled = False

    client = TestClient(main.app)
    try:
        yield state, client
    finally:
        if hasattr(main.limiter, "enabled"):
            main.limiter.enabled = original_enabled


# ---------------------------------------------------------------------------
# GET /api/strategies — per-portfolio scoping (Migration 038)
# ---------------------------------------------------------------------------
# The visibility matrix locked by the feature spec:
#
#   Strategy        | allowed_portfolio_names              | Visible in
#   CanSlim         | ['CanSlim']                          | CanSlim
#   StockTalk       | ['CanSlim', 'Long-Term Growth']      | CanSlim, LTG
#   21eStrategy     | NULL                                 | All 3
#   LongTerm        | ['457B Plan', 'Long-Term Growth']    | 457B, LTG
#   50sStrategy     | ['457B Plan', 'Long-Term Growth']    | 457B, LTG


def test_list_strategies_no_filter_returns_all(stubbed):
    """No ?portfolio= param → return every active strategy. Admin path."""
    _, client = stubbed
    r = client.get("/api/strategies", headers=_non_founder_headers())
    assert r.status_code == 200
    names = {s["name"] for s in r.json()}
    assert names == {"CanSlim", "StockTalk", "21eStrategy", "LongTerm", "50sStrategy"}


def test_list_strategies_canslim_portfolio(stubbed):
    """CanSlim portfolio sees CanSlim, StockTalk, 21eStrategy (NULL allowed)."""
    _, client = stubbed
    r = client.get("/api/strategies?portfolio=CanSlim", headers=_non_founder_headers())
    assert r.status_code == 200
    names = {s["name"] for s in r.json()}
    assert names == {"CanSlim", "StockTalk", "21eStrategy"}


def test_list_strategies_457b_portfolio(stubbed):
    """457B Plan sees LongTerm, 50sStrategy, 21eStrategy (NULL allowed).
    NOT CanSlim (scoped to CanSlim) or StockTalk (scoped to CanSlim+LTG)."""
    _, client = stubbed
    r = client.get("/api/strategies?portfolio=457B%20Plan", headers=_non_founder_headers())
    assert r.status_code == 200
    names = {s["name"] for s in r.json()}
    assert names == {"LongTerm", "50sStrategy", "21eStrategy"}


def test_list_strategies_long_term_growth_portfolio(stubbed):
    """Long-Term Growth sees StockTalk + LongTerm + 50sStrategy + 21eStrategy."""
    _, client = stubbed
    r = client.get("/api/strategies?portfolio=Long-Term%20Growth", headers=_non_founder_headers())
    assert r.status_code == 200
    names = {s["name"] for s in r.json()}
    assert names == {"StockTalk", "21eStrategy", "LongTerm", "50sStrategy"}


def test_list_strategies_unknown_portfolio_returns_only_universal(stubbed):
    """Defensive: an unknown portfolio name returns only NULL-allowed
    (universal) strategies. Don't surface 'no strategies available'
    just because the user typo'd a portfolio."""
    _, client = stubbed
    r = client.get("/api/strategies?portfolio=Nonexistent", headers=_non_founder_headers())
    assert r.status_code == 200
    names = {s["name"] for s in r.json()}
    assert names == {"21eStrategy"}


def test_serialize_strategy_exposes_allowed_portfolio_names(stubbed):
    """Response shape includes allowed_portfolio_names so the frontend can
    decide which strategies to suppress / surface in edit fallbacks."""
    _, client = stubbed
    r = client.get("/api/strategies?portfolio=CanSlim", headers=_non_founder_headers())
    body = r.json()
    canslim = next(s for s in body if s["name"] == "CanSlim")
    twentyone = next(s for s in body if s["name"] == "21eStrategy")
    assert canslim["allowed_portfolio_names"] == ["CanSlim"]
    assert twentyone["allowed_portfolio_names"] is None


# ---------------------------------------------------------------------------
# Founder-gating
# ---------------------------------------------------------------------------


def test_post_strategies_rejects_non_founder(stubbed):
    """A logged-in non-founder gets {error: forbidden_not_admin}."""
    state, client = stubbed

    r = client.post(
        "/api/strategies",
        json={"name": "Momentum", "color": "#22c55e"},
        headers=_non_founder_headers(),
    )
    assert r.status_code == 200
    assert r.json() == {"error": "forbidden_not_admin"}
    # No DB write attempted.
    assert state["create_calls"] == []


def test_put_strategies_rejects_non_founder(stubbed):
    """Same gate on the update endpoint."""
    state, client = stubbed

    r = client.put(
        "/api/strategies/CanSlim",
        json={"description": "evil"},
        headers=_non_founder_headers(),
    )
    assert r.status_code == 200
    assert r.json() == {"error": "forbidden_not_admin"}
    assert state["update_calls"] == []


# ---------------------------------------------------------------------------
# Strategy CRUD
# ---------------------------------------------------------------------------


def test_post_strategies_happy_path(stubbed):
    """Founder POST returns the persisted row."""
    _, client = stubbed

    r = client.post(
        "/api/strategies",
        json={"name": "Momentum", "color": "#22c55e", "description": "swing"},
        headers=_founder_headers(),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Momentum"
    assert body["color"] == "#22c55e"
    assert body["is_active"] is True


def test_post_strategies_rejects_invalid_hex(stubbed):
    """ValueError from db helper surfaces as {error}."""
    state, client = stubbed
    state["create_raises"] = ValueError("Color must be a six-digit hex string like '#6366f1'")

    r = client.post(
        "/api/strategies",
        json={"name": "Momentum", "color": "not-a-color"},
        headers=_founder_headers(),
    )
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert "hex" in body["error"].lower()


def test_put_strategies_partial_update(stubbed):
    """PUT only forwards recognised fields to the helper. A typo'd key
    in the body should NOT reach db.update_strategy."""
    state, client = stubbed

    r = client.put(
        "/api/strategies/CanSlim",
        json={"description": "Updated", "bogus_field": "ignored"},
        headers=_founder_headers(),
    )
    assert r.status_code == 200
    assert "error" not in r.json()
    # Helper called with only the recognised key.
    assert state["update_calls"] == [("CanSlim", {"description": "Updated"})]


# ---------------------------------------------------------------------------
# Per-trade retagging
# ---------------------------------------------------------------------------


def test_patch_trade_strategy_happy_path(stubbed):
    """PATCH updates and audit-logs."""
    state, client = stubbed

    r = client.patch(
        "/api/trades/202604-001/strategy",
        json={"strategy": "StockTalk"},
        headers=_non_founder_headers(),
    )
    assert r.status_code == 200
    body = r.json()
    assert body == {"ok": True, "trade_id": "202604-001", "strategy": "StockTalk"}
    assert state["retag_calls"] == [("CanSlim", "202604-001", "StockTalk")]
    # Audit logged
    assert any("STRATEGY_TAG" in str(a) for a in state["audit_logs"])


def test_patch_trade_strategy_rejects_unknown_strategy(stubbed):
    """Unknown strategy → error, no DB call."""
    state, client = stubbed

    r = client.patch(
        "/api/trades/202604-001/strategy",
        json={"strategy": "Bogus"},
        headers=_non_founder_headers(),
    )
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert "Bogus" in body["error"]
    assert state["retag_calls"] == []


# ---------------------------------------------------------------------------
# Bulk retagging
# ---------------------------------------------------------------------------


def test_bulk_strategy_partial_missing_returns_failed(stubbed):
    """Missing trade_ids surface in `failed: [...]` while valid ones
    still commit. Per Phase 2 design (commit-valid + report-missing for
    row-level misses; reject-batch only for invalid strategies)."""
    state, client = stubbed
    state["bulk_result"] = (2, ["202604-099"])  # 2 updated, 1 missing

    r = client.post(
        "/api/trades/bulk-strategy",
        json={"trade_ids": ["202604-001", "202604-002", "202604-099"], "strategy": "StockTalk"},
        headers=_non_founder_headers(),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["updated"] == 2
    assert body["failed"] == ["202604-099"]
    assert body["strategy"] == "StockTalk"
    assert state["bulk_calls"] == [("CanSlim", ["202604-001", "202604-002", "202604-099"], "StockTalk")]


def test_bulk_strategy_unknown_strategy_rejects_entire_batch(stubbed):
    """Invalid strategy → entire batch rejected up-front (no DB call).
    Distinct from the missing-trade_id path."""
    state, client = stubbed

    r = client.post(
        "/api/trades/bulk-strategy",
        json={"trade_ids": ["202604-001", "202604-002"], "strategy": "Bogus"},
        headers=_non_founder_headers(),
    )
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    assert "Bogus" in body["error"]
    # No DB write for any row.
    assert state["bulk_calls"] == []
