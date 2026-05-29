"""Tests for Phase 2 B-2's log_sell consolidation.

Pre-B-2: log_sell wrote summary inline via save_summary_row, then ran
a second pass through _recompute_summary_lifo for lot_closures. B-2
deleted the inline path; the recompute is now the sole summary writer.

Coverage:
  1. Recompute failure raises HTTPException(500) with detail_id +
     trx_id context (replaces the prior silent-print swallow).
  2. Response payload (realized_pl, remaining_shares, is_closed) is
     derived from the recompute summary, not the deleted inline walk.
  3. The overrides dict passed to _recompute_summary_lifo drives
     Sell_Rule + Sell_Notes (and Grade) over the preserve-existing
     loop — the body's values win.
  4. Structural invariant: save_summary_row is NEVER called by
     log_sell (only save_summary_with_closures from the recompute).
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
    """Existing campaign with one open BUY; user-entered metadata populated."""
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
        "sell_rule": "PRIOR SELL RULE",
        "sell_notes": "PRIOR SELL NOTES",
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
    """Yield (state, client). State captures every db_layer write so tests
    can assert on the post-B-2 single-writer contract.

    state knobs:
      summary_df          — what load_summary returns
      details_df          — what load_details returns (start: one BUY)
      recompute_should_raise — if not None, _recompute_summary_lifo raises
                                this exception instead of running

    state observations:
      saved_summaries     — captured save_summary_row calls (must be 0
                            post-B-2; if non-empty, B-2's inline-delete
                            regressed)
      saved_with_closures — captured save_summary_with_closures calls
                            (recompute path — must fire on every SELL)
      saved_details       — every detail row passed to save_detail_row
                            (the SELL we just logged is here)
      recompute_overrides — last `overrides` dict passed to
                            _recompute_summary_lifo (so the test can
                            inspect what body→recompute plumbing did)
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    import db_layer

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "summary_df": pd.DataFrame([_existing_summary_row()]),
        "details_df": pd.DataFrame([_buy_detail_row()]),
        "recompute_should_raise": None,
        "saved_summaries": [],
        "saved_with_closures": [],
        "saved_details": [],
        "recompute_overrides": None,
    }

    monkeypatch.setattr(db_layer, "load_summary",
                        lambda *a, **kw: state["summary_df"])
    monkeypatch.setattr(db_layer, "load_details",
                        lambda *a, **kw: state["details_df"])
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
        state["saved_summaries"].append({"portfolio": portfolio, "row": dict(row_dict)})
        return 1
    monkeypatch.setattr(db_layer, "save_summary_row", fake_save_summary_row)

    detail_id_counter = {"v": 1}
    def fake_save_detail(portfolio_name, row_dict):
        detail_id_counter["v"] += 1
        state["saved_details"].append({"portfolio": portfolio_name, "row": dict(row_dict)})
        # Append the inserted row to details_df in the post-normalize
        # snake_case shape so the subsequent load_details (called by the
        # recompute) sees the new SELL. Without this the recompute would
        # walk a stale BUY-only DataFrame and never produce CLOSED status.
        new_row = {
            "detail_id": detail_id_counter["v"],
            "trade_id": row_dict.get("Trade_ID"),
            "ticker": row_dict.get("Ticker"),
            "action": row_dict.get("Action"),
            "date": pd.to_datetime(row_dict.get("Date")),
            "shares": float(row_dict.get("Shares") or 0),
            "amount": float(row_dict.get("Amount") or 0),
            "value": float(row_dict.get("Value") or 0),
            "rule": row_dict.get("Rule"),
            "notes": row_dict.get("Notes"),
            "realized_pl": float(row_dict.get("Realized_PL") or 0),
            "stop_loss": row_dict.get("Stop_Loss"),
            "trx_id": row_dict.get("Trx_ID"),
            "instrument_type": row_dict.get("Instrument_Type", "STOCK"),
            "multiplier": float(row_dict.get("Multiplier") or 1.0),
            "match_method": row_dict.get("Match_Method"),
        }
        state["details_df"] = pd.concat(
            [state["details_df"], pd.DataFrame([new_row])],
            ignore_index=True,
        )
        return detail_id_counter["v"]
    monkeypatch.setattr(db_layer, "save_detail_row", fake_save_detail)

    monkeypatch.setattr(db_layer, "generate_unique_trx_id",
                        lambda portfolio, trade_id, prefix: f"{prefix}1")
    monkeypatch.setattr(db_layer, "log_audit", lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "delete_trade", lambda *a, **kw: None)
    monkeypatch.setattr(db_layer, "delete_lot_closures_for_trade",
                        lambda *a, **kw: None)

    # Wrap _recompute_summary_lifo so we can both capture its `overrides`
    # input and optionally force it to raise (for the 500-path test).
    real_recompute = main._recompute_summary_lifo
    def wrapped_recompute(portfolio, trade_id, ticker,
                         fallback_open_date="", overrides=None):
        state["recompute_overrides"] = dict(overrides) if overrides else None
        if state["recompute_should_raise"] is not None:
            raise state["recompute_should_raise"]
        return real_recompute(
            portfolio, trade_id, ticker, fallback_open_date,
            overrides=overrides,
        )
    monkeypatch.setattr(main, "_recompute_summary_lifo", wrapped_recompute)

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


def _sell_body(**overrides) -> dict[str, Any]:
    body = {
        "portfolio": "CanSlim",
        "trade_id": "202605-001",
        "shares": 50,
        "price": 220.0,
        "rule": "sr1.1 Profit target",
        "notes": "Half off the table",
        "date": "2026-05-08",
        "time": "10:00",
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# 1. Recompute failure → 500 with detail-row context
# ---------------------------------------------------------------------------


class TestRecomputeFailureRaises500:
    """The pre-B-2 swallow at log_sell's recompute call printed a warning
    and returned 200. Post-B-2 (inline summary write deleted) a recompute
    failure means the summary card is stale. Raising 500 ensures the
    operator sees the failure immediately and can recover via Trade
    Manager → Edit any transaction → Save.
    """

    def test_recompute_failure_returns_500(self, stubbed):
        state, client = stubbed
        state["recompute_should_raise"] = RuntimeError("simulated recompute bug")

        r = client.post("/api/trades/sell", json=_sell_body())
        assert r.status_code == 500, r.text

    def test_500_detail_mentions_detail_id_and_trx_id(self, stubbed):
        """The error detail string must carry both detail_id and trx_id
        so the operator can identify the orphaned detail row from the
        single error message."""
        state, client = stubbed
        state["recompute_should_raise"] = RuntimeError("simulated recompute bug")

        r = client.post("/api/trades/sell", json=_sell_body())
        assert r.status_code == 500
        detail = r.json()["detail"]
        assert "detail_id=" in detail
        # trx_id=S1 because generate_unique_trx_id stub returns "{prefix}1".
        assert "trx_id=S1" in detail
        assert "RuntimeError" in detail
        assert "simulated recompute bug" in detail

    def test_sell_detail_row_was_saved_before_500(self, stubbed):
        """Sanity: the detail row commits BEFORE the recompute call, so the
        500 message's claim that the SELL was saved is accurate."""
        state, client = stubbed
        state["recompute_should_raise"] = RuntimeError("boom")

        client.post("/api/trades/sell", json=_sell_body())
        assert len(state["saved_details"]) == 1
        saved_sell = state["saved_details"][0]["row"]
        assert saved_sell["Action"] == "SELL"
        assert saved_sell["Shares"] == 50.0


# ---------------------------------------------------------------------------
# 2. Response payload derived from recompute summary
# ---------------------------------------------------------------------------


class TestResponsePayloadFromRecompute:
    """Post-B-2 the response carries realized_pl + remaining_shares +
    is_closed from the recompute summary (not the deleted inline walk)."""

    def test_response_realized_pl_matches_recompute_summary(self, stubbed):
        state, client = stubbed
        # OPEN position pre-SELL: 100 sh @ $200. SELL 50 @ $220 → realized
        # PL = 50 × (220 − 200) = 1000 on a partial close.
        r = client.post("/api/trades/sell", json=_sell_body())
        assert r.status_code == 200, r.text
        body = r.json()
        # The recompute's summary_row.Realized_PL drives the response field.
        assert state["saved_with_closures"], "Expected a recompute call"
        recompute_summary = state["saved_with_closures"][-1]["summary_row"]
        assert body["realized_pl"] == recompute_summary["Realized_PL"]

    def test_response_remaining_shares_open_position(self, stubbed):
        state, client = stubbed
        # Partial close → status OPEN → remaining_shares > 0.
        r = client.post("/api/trades/sell", json=_sell_body(shares=50))
        assert r.status_code == 200, r.text
        body = r.json()
        recompute_summary = state["saved_with_closures"][-1]["summary_row"]
        assert recompute_summary["Status"] == "OPEN"
        # Response uses summary["Shares"] when OPEN.
        assert body["remaining_shares"] == recompute_summary["Shares"]
        assert not body["is_closed"]

    def test_response_remaining_shares_closed_position(self, stubbed):
        state, client = stubbed
        # Full close — SELL the entire 100-share position.
        r = client.post("/api/trades/sell", json=_sell_body(shares=100))
        assert r.status_code == 200, r.text
        body = r.json()
        recompute_summary = state["saved_with_closures"][-1]["summary_row"]
        assert recompute_summary["Status"] == "CLOSED"
        # CLOSED → remaining_shares forced to 0.0 (summary["Shares"] for
        # a closed trade is the lifetime bought total, not residual).
        assert body["remaining_shares"] == 0.0
        assert body["is_closed"] is True

    def test_response_drops_summary_id_field(self, stubbed):
        """summary_id was returned by the deleted inline path. Post-B-2
        the field is absent — verified non-consumer via prior grep."""
        state, client = stubbed
        r = client.post("/api/trades/sell", json=_sell_body())
        assert r.status_code == 200
        assert "summary_id" not in r.json()


# ---------------------------------------------------------------------------
# 3. Overrides plumbing — body values win over preserve-existing
# ---------------------------------------------------------------------------


class TestOverridesPlumbing:
    """The recompute's preserve-existing-fields loop would carry over the
    PRIOR SELL's sell_rule / sell_notes. The overrides dict log_sell
    builds from the request body must override those values."""

    def test_overrides_carries_sell_rule_from_body(self, stubbed):
        state, client = stubbed
        client.post("/api/trades/sell", json=_sell_body(
            rule="sr1.1 Profit target",
        ))
        assert state["recompute_overrides"] is not None
        assert state["recompute_overrides"]["Sell_Rule"] == "sr1.1 Profit target"

    def test_overrides_carries_sell_notes_from_body(self, stubbed):
        state, client = stubbed
        client.post("/api/trades/sell", json=_sell_body(
            notes="Half off the table",
        ))
        assert state["recompute_overrides"]["Sell_Notes"] == "Half off the table"

    def test_saved_summary_sell_rule_is_body_not_prior(self, stubbed):
        """End-to-end: the saved summary's Sell_Rule must be the body's
        value, NOT the prior summary row's 'PRIOR SELL RULE'."""
        state, client = stubbed
        r = client.post("/api/trades/sell", json=_sell_body(
            rule="sr1.1 Profit target",
            notes="fresh-from-body",
        ))
        assert r.status_code == 200
        saved = state["saved_with_closures"][-1]["summary_row"]
        assert saved["Sell_Rule"] == "sr1.1 Profit target"
        assert saved["Sell_Notes"] == "fresh-from-body"
        # PRIOR values must NOT have survived.
        assert saved["Sell_Rule"] != "PRIOR SELL RULE"
        assert saved["Sell_Notes"] != "PRIOR SELL NOTES"

    def test_grade_override_when_valid(self, stubbed):
        state, client = stubbed
        client.post("/api/trades/sell", json=_sell_body(grade=4))
        assert state["recompute_overrides"]["Grade"] == 4

    def test_grade_override_skipped_when_out_of_range(self, stubbed):
        """Out-of-range grade (1-5 valid) is silently skipped (matches
        the pre-B-2 inline path's behavior). overrides won't contain
        the Grade key."""
        state, client = stubbed
        client.post("/api/trades/sell", json=_sell_body(grade=99))
        assert "Grade" not in (state["recompute_overrides"] or {})

    def test_grade_override_skipped_when_empty_string(self, stubbed):
        """Empty / non-numeric Grade skipped without crashing."""
        state, client = stubbed
        client.post("/api/trades/sell", json=_sell_body(grade=""))
        assert "Grade" not in (state["recompute_overrides"] or {})


# ---------------------------------------------------------------------------
# 4. Structural: no inline save_summary_row call
# ---------------------------------------------------------------------------


class TestNoInlineSummaryWrite:
    """Pre-B-2 log_sell called db.save_summary_row directly to land the
    inline summary. Post-B-2 ONLY save_summary_with_closures (from the
    recompute) should fire. A regression where the inline call is
    accidentally restored would show up as a non-empty saved_summaries
    list here."""

    def test_save_summary_row_not_called_by_log_sell(self, stubbed):
        state, client = stubbed
        r = client.post("/api/trades/sell", json=_sell_body())
        assert r.status_code == 200, r.text
        assert state["saved_summaries"] == [], (
            f"log_sell must not call save_summary_row directly post-B-2; "
            f"got {len(state['saved_summaries'])} call(s): "
            f"{state['saved_summaries']}"
        )
        # The recompute's save_summary_with_closures IS expected.
        assert len(state["saved_with_closures"]) == 1
