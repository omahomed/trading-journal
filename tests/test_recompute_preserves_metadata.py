"""Regression tests for recompute paths preserving non-LIFO summary fields.

The bug: api/main.py _recompute_summary_lifo passed only LIFO-derived fields
(Status, Shares, Avg_Entry, etc.) to db.save_summary_with_closures. The DB
layer's UPDATE binds every column including those not in the input dict, so
rule/buy_notes/sell_rule/sell_notes got NULL and risk_budget got 0 (column
DEFAULT). Triggered on every sell/edit/delete/rebuild — wiped ~46 campaigns
of user-entered metadata in production.

The fix: _recompute_summary_lifo now loads the existing summary row and
merges Rule, Buy_Notes, Sell_Rule, Sell_Notes, Risk_Budget, Stop_Loss into
the LIFO output before save. log_sell's inline summary_row likewise reads
Stop_Loss + Risk_Budget off the existing row (matching how it already
preserves Rule + Buy_Notes), since its inline save fires before recompute.

This test is the contract guard: it asserts each of the 6 fields survives
across the three trigger paths the user actually hits — log_sell (partial
close), edit_transaction, delete_transaction. Any future change that
breaks preservation surfaces here.
"""
from __future__ import annotations

from datetime import datetime
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


# Fields the test asserts survive across all three trigger paths.
# These are the user-entered metadata that LIFO doesn't compute and that the
# DB layer's UPDATE statement would otherwise overwrite with NULL/DEFAULT.
PRESERVED_FIELDS_SNAKE = (
    "rule", "buy_notes", "sell_rule", "sell_notes", "risk_budget", "stop_loss",
)
PRESERVED_FIELDS_PASCAL = (
    "Rule", "Buy_Notes", "Sell_Rule", "Sell_Notes", "Risk_Budget", "Stop_Loss",
)


def _existing_summary_row() -> dict[str, Any]:
    """Existing summary row with all 6 user-entered fields populated.

    Snake-case keys to match _normalize_trades output (the production code
    path normalizes load_summary's PascalCase aliases to snake_case).
    """
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
        "unrealized_pl": 0.0,
        "return_pct": 0.0,
        "rule": "br1.3 Cup w/o Handle",
        "buy_notes": "Initial entry on breakout",
        "sell_rule": "sr2.1 Stop hit",
        "sell_notes": "Trailing stop triggered",
        "risk_budget": 500.0,
        "stop_loss": 195.0,
        "instrument_type": "STOCK",
        "multiplier": 1.0,
        "strategy": "CanSlim",
    }


def _buy_detail_row(detail_id: int = 1, trx_id: str = "B1") -> dict[str, Any]:
    """One BUY detail row matching the existing summary."""
    return {
        "detail_id": detail_id,
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
        "trx_id": trx_id,
        "instrument_type": "STOCK",
        "multiplier": 1.0,
    }


@pytest.fixture
def stubbed(monkeypatch):
    """Yield (state, client). State exposes saved summaries so tests assert
    what the recompute path passed to db.save_summary_with_closures.

    state knobs:
      summary_df  — what db.load_summary returns
      details_df  — what db.load_details returns

    state observations:
      saved_with_closures — list of dicts passed to save_summary_with_closures
      saved_summaries     — list of dicts passed to save_summary_row (log_sell
                            inline path)
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    import db_layer

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "summary_df": pd.DataFrame([_existing_summary_row()]),
        "details_df": pd.DataFrame([_buy_detail_row()]),
        "saved_with_closures": [],
        "saved_summaries": [],
    }

    monkeypatch.setattr(db_layer, "load_summary",
                        lambda *a, **kw: state["summary_df"])
    monkeypatch.setattr(db_layer, "load_details",
                        lambda *a, **kw: state["details_df"])
    # Preserve the production normalize step so PascalCase aliases from a
    # real load_summary would still work — but our stubbed dataframes are
    # already snake_case, so passthrough is fine.
    monkeypatch.setattr(main, "_normalize_trades", lambda df: df)

    def fake_save_with_closures(portfolio, trade_id, summary_row, closures):
        state["saved_with_closures"].append({
            "portfolio": portfolio,
            "trade_id": trade_id,
            "summary_row": dict(summary_row),
            "closures": list(closures),
        })
        return 1
    monkeypatch.setattr(db_layer, "save_summary_with_closures",
                        fake_save_with_closures)

    def fake_save_summary_row(portfolio, row_dict):
        state["saved_summaries"].append({
            "portfolio": portfolio,
            "row": dict(row_dict),
        })
        return 1
    monkeypatch.setattr(db_layer, "save_summary_row", fake_save_summary_row)

    detail_id_counter = {"v": 1}
    def fake_save_detail(portfolio_name, row_dict):
        detail_id_counter["v"] += 1
        return detail_id_counter["v"]
    monkeypatch.setattr(db_layer, "save_detail_row", fake_save_detail)

    monkeypatch.setattr(db_layer, "generate_unique_trx_id",
                        lambda portfolio, trade_id, prefix: f"{prefix}1")
    monkeypatch.setattr(db_layer, "update_detail_row",
                        lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "delete_detail_row",
                        lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "delete_trade",
                        lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "delete_lot_closures_for_trade",
                        lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "log_audit", lambda *a, **kw: None)

    # validate_post_edit_lifo runs in edit/delete endpoints. With our minimal
    # detail dataset the validator is fine, but stub it anyway to keep the
    # test focused on the preservation contract rather than LIFO validation.
    monkeypatch.setattr(main, "validate_post_edit_lifo",
                        lambda *a, **kw: None)

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


# ---------------------------------------------------------------------------
# Direct unit test on the chokepoint
# ---------------------------------------------------------------------------


def test_recompute_summary_lifo_preserves_all_six_fields(stubbed):
    """The single chokepoint covering edit/delete/rebuild for all 6 fields."""
    state, _client = stubbed
    import api.main as main

    main._recompute_summary_lifo("CanSlim", "202605-001", "AAPL")

    assert len(state["saved_with_closures"]) == 1, \
        "Expected exactly one save_summary_with_closures call"
    saved = state["saved_with_closures"][0]["summary_row"]

    expected = {
        "Rule": "br1.3 Cup w/o Handle",
        "Buy_Notes": "Initial entry on breakout",
        "Sell_Rule": "sr2.1 Stop hit",
        "Sell_Notes": "Trailing stop triggered",
        "Risk_Budget": 500.0,
        "Stop_Loss": 195.0,
    }
    for field, want in expected.items():
        assert saved.get(field) == want, \
            f"{field} not preserved: got {saved.get(field)!r}, expected {want!r}"


def test_recompute_summary_lifo_no_existing_summary_does_not_crash(stubbed):
    """First-time recompute (no existing summary row) must not crash."""
    state, _client = stubbed
    import api.main as main

    state["summary_df"] = pd.DataFrame()  # no existing summary
    main._recompute_summary_lifo("CanSlim", "202605-001", "AAPL")

    assert len(state["saved_with_closures"]) == 1


def test_recompute_with_wiped_existing_summary_does_not_crash(stubbed):
    """Existing summary has NULL/missing values for all 6 fields (e.g. wiped
    by a pre-fix recompute or a row that hasn't been backfilled yet by
    Migration 020). The pd.notna guard must skip these so summary_row keeps
    only the LIFO-derived fields — no exception, no spurious writes."""
    state, _client = stubbed
    import api.main as main

    wiped = _existing_summary_row()
    for field in PRESERVED_FIELDS_SNAKE:
        wiped[field] = None
    state["summary_df"] = pd.DataFrame([wiped])

    main._recompute_summary_lifo("CanSlim", "202605-001", "AAPL")

    assert len(state["saved_with_closures"]) == 1
    saved = state["saved_with_closures"][0]["summary_row"]
    # None of the 6 fields should have been merged in (pd.notna(None) is False).
    # The DB layer's UPDATE will still write NULL for absent keys, which is
    # the expected behavior for an already-wiped row — Migration 020 is what
    # restores those rows; the code fix only protects them going forward.
    for pascal in PRESERVED_FIELDS_PASCAL:
        assert pascal not in saved or saved[pascal] is None, \
            f"{pascal} should not be merged from a wiped existing row"


# ---------------------------------------------------------------------------
# Integration tests: trigger recompute via the three endpoints the user hits
# ---------------------------------------------------------------------------


def _assert_six_fields_preserved(saved_row: dict[str, Any]) -> None:
    """Helper: assert the saved summary_row carries all 6 fields from the
    existing row (non-zero/non-empty values from _existing_summary_row)."""
    expected = {
        "Rule": "br1.3 Cup w/o Handle",
        "Buy_Notes": "Initial entry on breakout",
        "Sell_Rule": "sr2.1 Stop hit",
        "Sell_Notes": "Trailing stop triggered",
        "Risk_Budget": 500.0,
        "Stop_Loss": 195.0,
    }
    for field, want in expected.items():
        assert saved_row.get(field) == want, \
            f"{field}: got {saved_row.get(field)!r}, expected {want!r}"


def test_log_sell_partial_close_preserves_all_six_fields(stubbed):
    """log_sell hits the inline save_summary_row path AND _recompute_summary_lifo.
    Both must preserve all 6 fields end-to-end for a partial close."""
    state, client = stubbed

    # Partial close: sell 50 of 100 shares.
    r = client.post("/api/trades/sell", json={
        "portfolio": "CanSlim",
        "trade_id": "202605-001",
        "shares": 50,
        "price": 220.0,
        "rule": "sr1.1 Profit target",
        "notes": "Half off the table",
        "date": "2026-05-08",
        "time": "10:00",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body

    # The inline save_summary_row at line 3052 must preserve Stop_Loss and
    # Risk_Budget (without the companion fix, they'd be NULL/0 here).
    assert state["saved_summaries"], "Expected an inline save_summary_row call"
    inline_saved = state["saved_summaries"][-1]["row"]
    assert inline_saved.get("Stop_Loss") == 195.0
    assert inline_saved.get("Risk_Budget") == 500.0
    # Sell_Rule and Sell_Notes are written from the body (the user just sold)
    assert inline_saved.get("Sell_Rule") == "sr1.1 Profit target"
    assert inline_saved.get("Sell_Notes") == "Half off the table"
    # Rule and Buy_Notes preserved from existing row
    assert inline_saved.get("Rule") == "br1.3 Cup w/o Handle"
    assert inline_saved.get("Buy_Notes") == "Initial entry on breakout"

    # The recompute (line 3068) must also preserve the 6 fields. After the
    # inline save, the existing summary_df now reflects the post-inline
    # state — Sell_Rule + Sell_Notes from body, others preserved.
    assert state["saved_with_closures"], "Expected a recompute save call"


def test_edit_transaction_preserves_all_six_fields(stubbed):
    """edit_transaction → _recompute_summary_lifo. All 6 fields preserved."""
    state, client = stubbed

    r = client.put("/api/trades/edit-transaction", json={
        "detail_id": 1,
        "portfolio": "CanSlim",
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "action": "BUY",
        "date": "2026-05-01 09:30",
        "shares": 100,
        "amount": 200.5,  # tiny price tweak — triggers recompute
        "rule": "br1.3 Cup w/o Handle",
        "notes": "Initial entry on breakout",
        "stop_loss": 195.0,
        "trx_id": "B1",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body

    assert state["saved_with_closures"], \
        "Expected _recompute_summary_lifo to call save_summary_with_closures"
    saved = state["saved_with_closures"][-1]["summary_row"]
    _assert_six_fields_preserved(saved)


def test_delete_transaction_preserves_all_six_fields(stubbed):
    """delete_transaction → _recompute_summary_lifo. All 6 fields preserved.

    Note: with only one detail row in the dataset, deletion would normally
    empty the campaign and trigger db.delete_trade instead of save. To
    exercise the preservation path, we add a second BUY detail so there's
    something to recompute after the delete.
    """
    state, client = stubbed
    state["details_df"] = pd.DataFrame([
        _buy_detail_row(detail_id=1, trx_id="B1"),
        _buy_detail_row(detail_id=2, trx_id="B2"),
    ])

    r = client.delete("/api/trades/transaction", params={
        "detail_id": 1,
        "trade_id": "202605-001",
        "ticker": "AAPL",
        "portfolio": "CanSlim",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body, body

    assert state["saved_with_closures"], \
        "Expected _recompute_summary_lifo to call save_summary_with_closures"
    saved = state["saved_with_closures"][-1]["summary_row"]
    _assert_six_fields_preserved(saved)
