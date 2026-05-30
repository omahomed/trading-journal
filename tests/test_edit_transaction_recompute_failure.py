"""Regression test for edit_transaction's post-edit recompute failure path.

Pre-fix: a `_recompute_summary_matching` failure was silently swallowed by a
`try/except Exception: pass` at api/main.py:4706-4710. The detail-row UPDATE
had already committed in its own transaction at that point, so a recompute
failure left the row saved but the summary card stale, with no UI signal.

Post-fix: the swallow is replaced with the same 500-raising pattern used in
log_sell at api/main.py:4172-4185 (Phase 2 B-2). The operator now sees the
divergence immediately and can recover via Trade Manager → Edit any
transaction → Save.

This mirrors the test_recompute_failure_returns_500 class in
tests/test_log_sell_consolidation.py.
"""
from __future__ import annotations

from typing import Any

import jwt
import pandas as pd
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def _existing_summary_row() -> dict[str, Any]:
    return {
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "status": "OPEN",
        "open_date": pd.Timestamp("2026-05-01"),
        "closed_date": None,
        "shares": 100.0,
        "avg_entry": 200.0,
        "avg_exit": 0.0,
        "total_cost": 20000.0,
        "realized_pl": 0.0,
        "return_pct": 0.0,
        "rule": "br1.3 Cup w/o Handle",
        "buy_notes": "Initial entry on breakout",
        "sell_rule": None,
        "sell_notes": None,
        "risk_budget": 500.0,
        "stop_loss": 195.0,
        "instrument_type": "STOCK",
        "multiplier": 1.0,
    }


def _buy_detail_row() -> dict[str, Any]:
    return {
        "detail_id": 1,
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "action": "BUY",
        "date": pd.Timestamp("2026-05-01 09:30:00"),
        "shares": 100.0,
        "amount": 200.0,
        "value": 20000.0,
        "rule": "br1.3 Cup w/o Handle",
        "notes": "Initial entry on breakout",
        "realized_pl": 0,
        "stop_loss": 195.0,
        "trx_id": "B1",
        "instrument_type": "STOCK",
        "multiplier": 1.0,
    }


@pytest.fixture
def stubbed(monkeypatch):
    """Yield (state, client). Stubs db_layer reads/writes and wraps
    _recompute_summary_matching so the test can force a raise."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    import db_layer

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "summary_df": pd.DataFrame([_existing_summary_row()]),
        "details_df": pd.DataFrame([_buy_detail_row()]),
        "recompute_should_raise": None,
        "updated_details": [],
        "audit_logs": [],
        "recompute_called": False,
    }

    monkeypatch.setattr(db_layer, "load_summary",
                        lambda *a, **kw: state["summary_df"])
    monkeypatch.setattr(db_layer, "load_details",
                        lambda *a, **kw: state["details_df"])
    monkeypatch.setattr(main, "_normalize_trades", lambda df: df)

    def fake_update_detail(portfolio, detail_id, row_dict):
        state["updated_details"].append({
            "portfolio": portfolio,
            "detail_id": detail_id,
            "row": dict(row_dict),
        })
    monkeypatch.setattr(db_layer, "update_detail_row", fake_update_detail)

    monkeypatch.setattr(db_layer, "mirror_detail_edit_to_summary",
                        lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "save_summary_row",
                        lambda *a, **kw: 1)
    monkeypatch.setattr(db_layer, "save_summary_with_closures",
                        lambda *a, **kw: 1)
    monkeypatch.setattr(db_layer, "save_detail_row",
                        lambda *a, **kw: 1)
    monkeypatch.setattr(db_layer, "delete_trade",
                        lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "delete_lot_closures_for_trade",
                        lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "generate_unique_trx_id",
                        lambda portfolio, trade_id, prefix: f"{prefix}1")

    def fake_log_audit(portfolio, action, trade_id, ticker, details, username='web'):
        state["audit_logs"].append({
            "portfolio": portfolio, "action": action, "trade_id": trade_id,
            "ticker": ticker, "details": details, "username": username,
        })
    monkeypatch.setattr(db_layer, "log_audit", fake_log_audit)

    # validate_post_edit_matching: pass through (no LIFO-safety rejection).
    monkeypatch.setattr(main, "validate_post_edit_matching",
                        lambda *a, **kw: None)

    # Wrap _recompute_summary_matching so the test can both observe the call
    # and force a raise.
    def wrapped_recompute(portfolio, trade_id, ticker,
                         fallback_open_date="", overrides=None):
        state["recompute_called"] = True
        if state["recompute_should_raise"] is not None:
            raise state["recompute_should_raise"]
        return None
    monkeypatch.setattr(main, "_recompute_summary_matching", wrapped_recompute)

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


def _edit_body(**overrides) -> dict[str, Any]:
    body = {
        "detail_id": 1,
        "portfolio": "CanSlim",
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "action": "BUY",
        "date": "2026-05-01 09:30",
        "shares": 100,
        "amount": 200.0,
        "rule": "br1.5 Pivot bounce",
        "notes": "edited notes",
        "stop_loss": 195.0,
        "trx_id": "B1",
    }
    body.update(overrides)
    return body


class TestRecomputeFailureRaises500:
    """The pre-fix swallow at edit_transaction's recompute call returned 200
    silently. Post-fix a recompute failure raises HTTPException(500) so the
    operator sees the stale-summary divergence immediately."""

    def test_recompute_failure_returns_500(self, stubbed):
        state, client = stubbed
        state["recompute_should_raise"] = RuntimeError("simulated recompute bug")

        r = client.put("/api/trades/edit-transaction", json=_edit_body())
        assert r.status_code == 500, r.text

    def test_500_detail_mentions_detail_id_and_error(self, stubbed):
        """The error detail string must carry detail_id, the exception class
        name, and the underlying message so the operator can identify the
        orphaned detail row from the single error response."""
        state, client = stubbed
        state["recompute_should_raise"] = RuntimeError("simulated recompute bug")

        r = client.put("/api/trades/edit-transaction", json=_edit_body())
        assert r.status_code == 500
        detail = r.json()["detail"]
        assert "detail_id=1" in detail
        assert "RuntimeError" in detail
        assert "simulated recompute bug" in detail

    def test_detail_row_was_updated_before_500(self, stubbed):
        """Sanity: the detail-row UPDATE commits BEFORE the recompute call,
        so the 500 message's claim that the edit was saved is accurate."""
        state, client = stubbed
        state["recompute_should_raise"] = RuntimeError("boom")

        client.put("/api/trades/edit-transaction", json=_edit_body())
        assert len(state["updated_details"]) == 1
        saved = state["updated_details"][0]["row"]
        assert saved["Rule"] == "br1.5 Pivot bounce"

    def test_print_line_emitted_on_failure(self, stubbed, capsys):
        """The print(...) breadcrumb must fire before the raise so the failure
        is visible in server logs even if the 500 response is lost."""
        state, client = stubbed
        state["recompute_should_raise"] = RuntimeError("boom")

        client.put("/api/trades/edit-transaction", json=_edit_body())
        captured = capsys.readouterr()
        assert "[edit_transaction] post-edit recompute failed" in captured.out
        assert "202605-001" in captured.out

    def test_recompute_success_returns_200(self, stubbed):
        """Control: when the recompute does not raise, the endpoint returns
        200 as before. Guards against the fix accidentally inverting the
        happy path."""
        state, client = stubbed

        r = client.put("/api/trades/edit-transaction", json=_edit_body())
        assert r.status_code == 200, r.text
        assert state["recompute_called"] is True
        assert r.json().get("status") == "ok"
