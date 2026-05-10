"""Contract guard for recompute paths.

Two classes of summary fields survive a recompute:

  USER-AUTHORED  (preserved across recomputes — content the user typed):
    rule, buy_notes, sell_rule, sell_notes, stop_loss, notes

  DERIVED  (recomputed on every state change — computed values that should
            always reflect current state):
    risk_budget  ← compute_trade_risk over LIFO inventory (Group 7-1)
    (plus the standard LIFO outputs: status, shares, avg_entry, total_cost,
    realized_pl, return_pct, avg_exit, closed_date)

The preservation contract guards the *user-authored* fields against a
historical bug: api/main.py _recompute_summary_lifo once passed only
LIFO-derived fields to db.save_summary_with_closures, and the DB layer's
UPDATE binds every column — including those absent from the input dict —
so rule/buy_notes/sell_rule/sell_notes/notes/stop_loss got wiped to
NULL on every sell/edit/delete/rebuild. ~46 production campaigns lost
their user-entered metadata to this. The fix loads the existing summary
row and merges those fields into the LIFO output before save; this test
asserts they continue to survive.

risk_budget was historically in that preservation set too, because the
buggy additive scale-in logic was the only path that ever wrote it —
making it look user-authored. Group 7-1 reframed it as a derived field
recomputed via compute_trade_risk(txns, multiplier) on every state
change, so preserving it would now be the bug. The assertions for that
field flipped from "preserved" to "recomputed correctly" — not a
weakening of the contract, an alignment with corrected semantics.
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


# User-authored fields that survive a recompute by being merged from the
# existing summary row into the LIFO output before save.
PRESERVED_FIELDS_SNAKE = (
    "rule", "buy_notes", "sell_rule", "sell_notes", "stop_loss",
)
PRESERVED_FIELDS_PASCAL = (
    "Rule", "Buy_Notes", "Sell_Rule", "Sell_Notes", "Stop_Loss",
)

# The single BUY detail row used by the default fixture: 100 shares @ $200,
# stop $195. compute_trade_risk should return 100 × (200 − 195) × 1 = 500.00.
EXPECTED_RECOMPUTED_RISK = 500.0


def _existing_summary_row() -> dict[str, Any]:
    """Existing summary row with all preserved + derived fields populated.

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
        "notes": "General notes on this campaign",
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
# Direct unit tests on the chokepoint
# ---------------------------------------------------------------------------


def test_recompute_preserves_user_authored_and_recomputes_risk(stubbed):
    """The single chokepoint: user-authored fields preserved across recompute,
    risk_budget recomputed via compute_trade_risk."""
    state, _client = stubbed
    import api.main as main

    main._recompute_summary_lifo("CanSlim", "202605-001", "AAPL")

    assert len(state["saved_with_closures"]) == 1, \
        "Expected exactly one save_summary_with_closures call"
    saved = state["saved_with_closures"][0]["summary_row"]

    # User-authored: preserved from existing summary row.
    expected_preserved = {
        "Rule": "br1.3 Cup w/o Handle",
        "Buy_Notes": "Initial entry on breakout",
        "Sell_Rule": "sr2.1 Stop hit",
        "Sell_Notes": "Trailing stop triggered",
        "Stop_Loss": 195.0,
    }
    for field, want in expected_preserved.items():
        assert saved.get(field) == want, \
            f"{field} not preserved: got {saved.get(field)!r}, expected {want!r}"

    # Derived: recomputed from LIFO inventory of the BUY detail
    # (100 shs × (200 − 195) × 1 = $500.00). Independent of whatever
    # value the existing summary row carried for risk_budget.
    assert saved.get("Risk_Budget") == EXPECTED_RECOMPUTED_RISK, \
        f"Risk_Budget should be recomputed: got {saved.get('Risk_Budget')!r}, " \
        f"expected {EXPECTED_RECOMPUTED_RISK!r}"


def test_recompute_recomputes_risk_independent_of_existing_value(stubbed):
    """Risk_Budget is derived, not preserved: bumping the existing summary's
    risk_budget to a wild value does not change the recomputed result."""
    state, _client = stubbed
    import api.main as main

    bumped = _existing_summary_row()
    bumped["risk_budget"] = 99999.0  # nonsense — should be ignored
    state["summary_df"] = pd.DataFrame([bumped])

    main._recompute_summary_lifo("CanSlim", "202605-001", "AAPL")

    saved = state["saved_with_closures"][0]["summary_row"]
    assert saved.get("Risk_Budget") == EXPECTED_RECOMPUTED_RISK


def test_recompute_preserves_notes(stubbed):
    """Phase 2 Commit 6 expansion: the c0435ee preservation block also
    preserves summary.notes (the legacy general-notes column). Without this,
    every recompute would silently wipe notes — the same shape of bug
    c0435ee fixed for the user-prose fields. Still applies post-Group 7-1.
    """
    state, _client = stubbed
    import api.main as main

    main._recompute_summary_lifo("CanSlim", "202605-001", "AAPL")

    assert len(state["saved_with_closures"]) == 1
    saved = state["saved_with_closures"][0]["summary_row"]
    assert saved.get("Notes") == "General notes on this campaign", \
        f"Notes not preserved: got {saved.get('Notes')!r}"


def test_recompute_no_existing_summary_does_not_crash(stubbed):
    """First-time recompute (no existing summary row) must not crash and
    must still write a computed risk_budget from the details."""
    state, _client = stubbed
    import api.main as main

    state["summary_df"] = pd.DataFrame()  # no existing summary
    main._recompute_summary_lifo("CanSlim", "202605-001", "AAPL")

    assert len(state["saved_with_closures"]) == 1
    saved = state["saved_with_closures"][0]["summary_row"]
    # Risk_Budget still computed from details even with no existing summary.
    assert saved.get("Risk_Budget") == EXPECTED_RECOMPUTED_RISK


def test_recompute_with_wiped_existing_summary_recomputes_risk(stubbed):
    """Existing summary has NULL for every user-authored field AND risk_budget
    (e.g. wiped by a pre-fix recompute, or a row that hasn't been backfilled
    yet by Migration 020). The pd.notna guard correctly skips the user-
    authored fields, but Risk_Budget should STILL be populated by the
    derived recompute — this is the property that means a corrupted
    risk_budget self-heals on the next state change."""
    state, _client = stubbed
    import api.main as main

    wiped = _existing_summary_row()
    for field in PRESERVED_FIELDS_SNAKE:
        wiped[field] = None
    wiped["risk_budget"] = None
    state["summary_df"] = pd.DataFrame([wiped])

    main._recompute_summary_lifo("CanSlim", "202605-001", "AAPL")

    assert len(state["saved_with_closures"]) == 1
    saved = state["saved_with_closures"][0]["summary_row"]
    # None of the user-authored fields should have been merged in
    # (pd.notna(None) is False). DB UPDATE will write NULL for absent keys,
    # which is the expected behavior for an already-wiped row.
    for pascal in PRESERVED_FIELDS_PASCAL:
        assert pascal not in saved or saved[pascal] is None, \
            f"{pascal} should not be merged from a wiped existing row"
    # Risk_Budget IS in the saved row, recomputed fresh from inventory.
    assert saved.get("Risk_Budget") == EXPECTED_RECOMPUTED_RISK, \
        "Risk_Budget should self-heal via recompute even when existing was NULL"


# ---------------------------------------------------------------------------
# Integration tests: trigger recompute via the endpoints the user hits
# ---------------------------------------------------------------------------


def _assert_user_authored_preserved_and_risk_recomputed(
    saved_row: dict[str, Any], expected_risk: float = EXPECTED_RECOMPUTED_RISK
) -> None:
    """Helper: assert the saved summary_row carries the 5 user-authored
    fields from the existing row AND a freshly-recomputed Risk_Budget.

    `expected_risk` defaults to the single-BUY fixture value but tests with
    multi-BUY setups (e.g. delete) pass the value derived from their
    post-stub inventory.
    """
    expected_preserved = {
        "Rule": "br1.3 Cup w/o Handle",
        "Buy_Notes": "Initial entry on breakout",
        "Sell_Rule": "sr2.1 Stop hit",
        "Sell_Notes": "Trailing stop triggered",
        "Stop_Loss": 195.0,
    }
    for field, want in expected_preserved.items():
        assert saved_row.get(field) == want, \
            f"{field}: got {saved_row.get(field)!r}, expected {want!r}"
    assert saved_row.get("Risk_Budget") == expected_risk, \
        f"Risk_Budget should be recomputed: got {saved_row.get('Risk_Budget')!r}, " \
        f"expected {expected_risk!r}"


def test_log_sell_partial_close_preserves_authored_and_recomputes_risk(stubbed):
    """log_sell hits the inline save_summary_row path AND _recompute_summary_lifo.
    User-authored fields are preserved at both writes; Risk_Budget on the
    final (recompute) write is the derived value over the post-SELL
    inventory."""
    state, client = stubbed

    # Partial close: sell 50 of 100 shares. After this, inventory has the
    # remaining 50 shares at the same lot's $200 entry / $195 stop, so the
    # recomputed risk is 50 × 5 = $250.
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

    # Inline save preserves Stop_Loss and (still) writes the existing
    # risk_budget into the row — it's the recompute that flips it to
    # the derived value, not the inline write. The inline behavior is
    # dead-but-harmless: the recompute overwrites it.
    assert state["saved_summaries"], "Expected an inline save_summary_row call"
    inline_saved = state["saved_summaries"][-1]["row"]
    assert inline_saved.get("Stop_Loss") == 195.0
    assert inline_saved.get("Sell_Rule") == "sr1.1 Profit target"
    assert inline_saved.get("Sell_Notes") == "Half off the table"
    assert inline_saved.get("Rule") == "br1.3 Cup w/o Handle"
    assert inline_saved.get("Buy_Notes") == "Initial entry on breakout"

    # The recompute write is the canonical end-state. After the SELL, the
    # detail set is unchanged from the stub (still just the original BUY)
    # because save_detail_row is stubbed and load_details returns the same
    # frame. So Risk_Budget = 100 × (200 − 195) × 1 = $500 here — the
    # stub doesn't actually mutate the inventory, but the *path* is
    # exercised and the value is the derived one, not the preserved one.
    assert state["saved_with_closures"], "Expected a recompute save call"
    recompute_saved = state["saved_with_closures"][-1]["summary_row"]
    _assert_user_authored_preserved_and_risk_recomputed(recompute_saved)


def test_edit_transaction_preserves_authored_and_recomputes_risk(stubbed):
    """edit_transaction → _recompute_summary_lifo. User-authored preserved,
    Risk_Budget recomputed."""
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
    _assert_user_authored_preserved_and_risk_recomputed(saved)


def test_delete_transaction_preserves_authored_and_recomputes_risk(stubbed):
    """delete_transaction → _recompute_summary_lifo. User-authored preserved,
    Risk_Budget recomputed.

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
    # The stub's delete_detail_row is a no-op, so load_details still
    # returns BOTH BUY rows when the recompute reads them. compute_trade_risk
    # walks 2 × (100 shs × $5 risk-per-share) = $1000. The behaviour-level
    # contract — "delete triggers a recompute and writes the derived value
    # to summary" — is what this test exercises, not the precise inventory
    # transition (which the unit tests in test_calc.py already cover).
    _assert_user_authored_preserved_and_risk_recomputed(saved, expected_risk=1000.0)


# ---------------------------------------------------------------------------
# Group 7-1 specific: the additive-scale-in bug scenario
# ---------------------------------------------------------------------------


def test_scale_in_with_lot1_at_BE_does_not_inflate_risk(stubbed):
    """The exact additive-bug scenario the audit called out.

    Pre-Group 7-1: log_buy scale-in stored existing_risk_budget + new_lot_risk
    regardless of whether the existing lot's stop had moved up. So lot 1 at
    BE (stop = entry, risk = 0) + lot 2 with a normal stop would persist as
    "lot 1's frozen historical risk + lot 2" — inflating the headline risk
    figure.

    After Group 7-1: the recompute walks current inventory with current
    stops, so lot 1 at BE contributes 0 and only lot 2 counts.

    Setup: existing summary's risk_budget is 99999 (nonsense / inflated by
    the old bug). Details show lot 1 (100 shs @ $200, stop $200 = BE) plus
    lot 2 (50 shs @ $210, stop $205, risk $250). After recompute, the
    written Risk_Budget should equal $250 — NOT 99999, NOT 0 + 750, NOT
    the existing value carried forward.
    """
    state, _client = stubbed
    import api.main as main

    # Existing summary carries an inflated (pre-fix) risk_budget value.
    bumped = _existing_summary_row()
    bumped["risk_budget"] = 99999.0
    state["summary_df"] = pd.DataFrame([bumped])

    # Two BUY lots: lot 1 moved to BE (stop=entry), lot 2 with live stop.
    state["details_df"] = pd.DataFrame([
        {**_buy_detail_row(detail_id=1, trx_id="B1"),
         "shares": 100.0, "amount": 200.0, "stop_loss": 200.0,
         "date": pd.Timestamp("2026-05-01 09:30:00")},
        {**_buy_detail_row(detail_id=2, trx_id="B2"),
         "shares": 50.0, "amount": 210.0, "stop_loss": 205.0,
         "date": pd.Timestamp("2026-05-03 10:00:00")},
    ])

    main._recompute_summary_lifo("CanSlim", "202605-001", "AAPL")

    saved = state["saved_with_closures"][-1]["summary_row"]
    # Lot 1: 100 × max(0, 200 − 200) = 0   (BE)
    # Lot 2: 50  × max(0, 210 − 205) = 250
    # Total: 250 — NOT 99999 (preserved old value), NOT 750 (only lot 2's
    # naive risk), NOT 0 (would miss the live-stop lot).
    assert saved.get("Risk_Budget") == 250.0, \
        f"Risk_Budget should equal lot 2's contribution alone: " \
        f"got {saved.get('Risk_Budget')!r}, expected 250.0"


def test_fully_closed_campaign_risk_is_zero(stubbed):
    """Once the position is fully sold out, Trade Risk $ is 0 — no forward
    exposure. (Old code path would have kept whatever was preserved.)"""
    state, _client = stubbed
    import api.main as main

    state["details_df"] = pd.DataFrame([
        {**_buy_detail_row(detail_id=1, trx_id="B1"),
         "action": "BUY",  "shares": 100.0, "amount": 200.0, "stop_loss": 195.0,
         "date": pd.Timestamp("2026-05-01 09:30:00")},
        {**_buy_detail_row(detail_id=2, trx_id="S1"),
         "action": "SELL", "shares": 100.0, "amount": 220.0, "stop_loss": 0.0,
         "date": pd.Timestamp("2026-05-08 10:00:00")},
    ])

    main._recompute_summary_lifo("CanSlim", "202605-001", "AAPL")

    saved = state["saved_with_closures"][-1]["summary_row"]
    assert saved.get("Risk_Budget") == 0.0, \
        f"Fully-closed campaign should have Risk_Budget = 0, got {saved.get('Risk_Budget')!r}"
