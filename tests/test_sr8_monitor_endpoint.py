"""Tests for the SR8 Cascade Monitor endpoint.

The endpoint pulls SR8 positions from the trades DB (open positions in the
named portfolio where b1_max_return_pct >= 50) and wraps mors.monitor.analyze()
per position. These tests mock both:
  - _sr8_load_db_positions (so we don't need a real DB)
  - mors.monitor.analyze (so we don't replay cascade history or hit yfinance)

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
  7. Empty position list returns an empty payload (not a crash).
  8. Portfolio query param threads to _sr8_load_db_positions and lands in
     meta.portfolio.
  9. _sr8_load_db_positions classifies SR8 via max(live B1 return,
     stored b1_max_return_pct) >= 50 (mirrors the frontend
     positions.ts + sell-rule.ts classifier), excludes options, and
     synthesizes b1_date from the first BUY in trades_details. A stale
     stored peak below 50 is rescued by a live B1 return >= 50; a
     live-price fetch failure falls back to the stored peak alone.
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
    current_tier: str = "GREEN",
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
        "current_tier": current_tier,
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
        "load_calls": [],          # capture portfolio names the loader saw
    }

    def fake_load(portfolio: str):
        state["load_calls"].append(portfolio)
        return list(state["positions"])

    monkeypatch.setattr(main, "_sr8_load_db_positions", fake_load)

    # Replace the module-level import indirection inside _sr8_run_monitor.
    # That helper does `from mors.monitor import analyze as mors_analyze`
    # at call time, so we install a fake mors.monitor module.
    import sys
    import types

    fake_mod = types.ModuleType("mors.monitor")

    def fake_analyze(pos: dict[str, Any], nlv: float, refresh: bool = False,
                     activation_nlv=None):
        # Ignore activation_nlv in the fake — the endpoint passes it through
        # but the analyze() output shape it drives is asserted elsewhere
        # (tests/test_sr8_monitor_anchor.py). Here we care only about the
        # endpoint plumbing.
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
    envelope with summary + positions + meta. tier_breakdown reflects
    the per-position current_tier distribution (replaces the obsolete
    20-cas / 15-cas split — only one floor schedule exists now)."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "AAA", "b1_date": "2026-04-01", "b1_price": 100, "shares_held": 100, "avg_price": 100},
        {"ticker": "BBB", "b1_date": "2026-04-02", "b1_price": 200, "shares_held": 50, "avg_price": 200},
        {"ticker": "CCC", "b1_date": "2026-04-03", "b1_price": 300, "shares_held": 30, "avg_price": 300},
    ]
    state["analyze_by_ticker"] = {
        "AAA": _analyze_result("AAA", current_tier="GREEN",     current_pct_nlv=25.0),
        "BBB": _analyze_result("BBB", current_tier="QUICK",     current_pct_nlv=12.0),
        "CCC": _analyze_result("CCC", current_tier="QUICKSAND", current_pct_nlv=8.0),
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
        assert "current_tier" in p  # live cascade tier surfaced
        assert "cascade_core" not in p  # legacy field removed

    # Summary tier_breakdown
    assert body["summary"]["total_positions"] == 3
    assert "cascade_breakdown" not in body["summary"]  # legacy field removed
    assert body["summary"]["tier_breakdown"] == {
        "green": 1, "quick": 1, "quicksand": 1, "gd": 0,
    }


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


def test_is_action_non_green_above_tolerance(stubbed):
    """A non-Green tier (QUICK here) with delta_dollars > $500 floor
    tolerance is actionable; the same tier below tolerance is hold.
    signal_today is intentionally NOT set — the new gate is tier+floor,
    not "did the signal fire today" (see test_is_action_stale_signal_
    no_longer_gates for the load-bearing case)."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "AAA", "b1_date": "2026-04-01", "b1_price": 100, "shares_held": 100, "avg_price": 100},
        {"ticker": "BBB", "b1_date": "2026-04-02", "b1_price": 200, "shares_held": 50, "avg_price": 200},
    ]
    state["analyze_by_ticker"] = {
        # QUICK + above tolerance → action
        "AAA": _analyze_result("AAA", current_tier="QUICK", tier_pct_nlv=10.0,
                               delta_dollars=12345.0, current_pct_nlv=22.0),
        # QUICK but BELOW tolerance ($300 < $500 epsilon) → hold
        "BBB": _analyze_result("BBB", current_tier="QUICK", tier_pct_nlv=10.0,
                               delta_dollars=300.0, current_pct_nlv=10.05),
    }

    r = client.get("/api/sr8/monitor?nlv=500000")
    body = r.json()
    rows = {p["ticker"]: p for p in body["positions"]}
    assert rows["AAA"]["is_action"] is True
    assert rows["BBB"]["is_action"] is False


def test_is_action_stale_signal_no_longer_gates(stubbed):
    """The load-bearing case for this build: a QUICK position whose
    weekly bar already passed (signal_today=False) but is STILL above
    its floor (delta_dollars > tol) must remain actionable. Pre-change,
    the signal_today gate dropped it into Hold the moment the firing
    week elapsed — even though the user never trimmed."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "STALE", "b1_date": "2026-03-01", "b1_price": 60, "shares_held": 320, "avg_price": 75},
    ]
    state["analyze_by_ticker"] = {
        # signal fired 2 weeks ago — last_signal_date != last_bar_date.
        "STALE": _analyze_result(
            "STALE",
            current_tier="QUICK", tier_pct_nlv=10.0,
            current_pct_nlv=14.3,                # still above the 10% floor
            delta_dollars=21_500.0,              # well above $500 tol
            delta_shares=215,
            signal_today=False,                  # ← would have killed action under old gate
            last_signal="QUICK",
        ),
    }

    r = client.get("/api/sr8/monitor?nlv=500000")
    body = r.json()
    row = body["positions"][0]
    assert row["is_action"] is True
    # Sanity: the delta_shares the UI needs to render "TRIM N sh"
    # is still on the row.
    assert row["delta_shares"] == 215


def test_is_action_green_tier_never_actionable(stubbed):
    """Green never sells — even if the engine somehow surfaces a
    non-zero delta (it shouldn't, but defense-in-depth). Both the
    plain GREEN label and the GREEN(sub-entry) variant fold into the
    same hold path."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "GRN",     "b1_date": "2026-04-01", "b1_price": 100, "shares_held": 100, "avg_price": 100},
        {"ticker": "GRN_SUB", "b1_date": "2026-04-02", "b1_price": 200, "shares_held": 50,  "avg_price": 200},
    ]
    state["analyze_by_ticker"] = {
        "GRN":     _analyze_result("GRN",     current_tier="GREEN",            delta_dollars=99_999.0),
        "GRN_SUB": _analyze_result("GRN_SUB", current_tier="GREEN(sub-entry)", delta_dollars=99_999.0),
    }

    r = client.get("/api/sr8/monitor?nlv=500000")
    rows = {p["ticker"]: p for p in r.json()["positions"]}
    assert rows["GRN"]["is_action"] is False
    assert rows["GRN_SUB"]["is_action"] is False


def test_early_warn_within_two_points_of_target(stubbed):
    """Held position is NEAR when 0 <= (target - current_pct_nlv) <= 2.
    Spec floors are 15/10/5/0 now — a QUICK-tier position with 10%
    floor and 8.5% current % NLV is 1.5 points below the floor → NEAR."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "NEAR", "b1_date": "2026-04-01", "b1_price": 100, "shares_held": 100, "avg_price": 100},
        {"ticker": "FAR", "b1_date": "2026-04-02", "b1_price": 200, "shares_held": 50, "avg_price": 200},
    ]
    state["analyze_by_ticker"] = {
        "NEAR": _analyze_result(
            "NEAR", current_tier="QUICK", tier_pct_nlv=10.0, current_pct_nlv=8.5, signal_today=False
        ),
        "FAR": _analyze_result(
            "FAR", current_tier="GREEN", tier_pct_nlv=15.0, current_pct_nlv=5.0, signal_today=False
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
    """No SR8 positions in the portfolio → endpoint returns the envelope
    with zero-state values, no crash."""
    state, client = stubbed
    state["positions"] = []

    r = client.get("/api/sr8/monitor?nlv=500000")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["total_positions"] == 0
    assert body["summary"]["flagged_count"] == 0
    assert body["positions"] == []


def test_portfolio_param_threads_to_loader_and_meta(stubbed):
    """?portfolio=X is forwarded to the DB loader and echoed in meta."""
    state, client = stubbed
    state["positions"] = [
        {"ticker": "AAA", "b1_date": "2026-04-01", "b1_price": 100, "shares_held": 100, "avg_price": 100},
    ]

    r = client.get("/api/sr8/monitor?nlv=500000&portfolio=TQQQ%20Strategy")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meta"]["portfolio"] == "TQQQ Strategy"
    assert state["load_calls"] == ["TQQQ Strategy"]

    # Default falls back to CanSlim when omitted.
    state["load_calls"].clear()
    r2 = client.get("/api/sr8/monitor?nlv=500000")
    assert r2.json()["meta"]["portfolio"] == "CanSlim"
    assert state["load_calls"] == ["CanSlim"]


def test_load_db_positions_filters_and_synthesizes(monkeypatch):
    """_sr8_load_db_positions: classifies SR8 via max(live B1 return,
    stored b1_max_return_pct) >= 50, drops options, drops rows missing
    b1_price, and synthesizes b1_date from the first BUY in
    trades_details. This is the campaign-classifier parity test —
    positions.ts:154-170 + sell-rule.ts:8-13."""
    import pandas as pd
    import api.main as main

    summary_rows = [
        # SR8 stock — included via stored peak (live not needed)
        {"Trade_ID": "T1", "Ticker": "MU",  "Status": "OPEN", "Shares": 135,
         "Avg_Entry": 80.0, "B1_Entry_Price": 60.0, "B1_Max_Return_Pct": 119.14,
         "Instrument_Type": "STOCK"},
        # Stored peak below 50 but LIVE B1 return >= 50 — included
        # (this is the audit case: DELL/ARM/PENG/FPS with stale stored
        # peaks. Live price of 105.6 vs B1 60.0 = +76% → SR8.)
        {"Trade_ID": "T2", "Ticker": "DELL", "Status": "OPEN", "Shares": 318,
         "Avg_Entry": 70.0, "B1_Entry_Price": 60.0, "B1_Max_Return_Pct": 47.81,
         "Instrument_Type": "STOCK"},
        # Stored peak below 50 AND live B1 also below — excluded
        # (live 100 vs B1 95 = +5.3%, stored 12%, effective_max = 12%).
        {"Trade_ID": "T3", "Ticker": "SNOW", "Status": "OPEN", "Shares": 30,
         "Avg_Entry": 100.0, "B1_Entry_Price": 95.0, "B1_Max_Return_Pct": 12.0,
         "Instrument_Type": "STOCK"},
        # Closed — excluded by status filter regardless of peak
        {"Trade_ID": "T4", "Ticker": "PLTR", "Status": "CLOSED", "Shares": 0,
         "Avg_Entry": 0, "B1_Entry_Price": 20.0, "B1_Max_Return_Pct": 200.0,
         "Instrument_Type": "STOCK"},
        # SR8-tier but option — excluded (engine is stock-only)
        {"Trade_ID": "T5", "Ticker": "NVDA250620C500", "Status": "OPEN", "Shares": 10,
         "Avg_Entry": 5.0, "B1_Entry_Price": 4.0, "B1_Max_Return_Pct": 80.0,
         "Instrument_Type": "OPTION"},
        # Stored peak NaN but live B1 return >= 50 — included via live
        # alone (a brand-new SR8 promotion whose peak hasn't been
        # heal-written yet).
        {"Trade_ID": "T6", "Ticker": "NEW", "Status": "OPEN", "Shares": 100,
         "Avg_Entry": 50.0, "B1_Entry_Price": 50.0, "B1_Max_Return_Pct": None,
         "Instrument_Type": "STOCK"},
    ]
    details_rows = [
        # MU: first BUY 2026-02-08, then add-on
        {"Trade_ID": "T1", "Ticker": "MU", "Action": "BUY", "Date": "2026-02-08", "Amount": 60.0},
        {"Trade_ID": "T1", "Ticker": "MU", "Action": "BUY", "Date": "2026-02-15", "Amount": 65.0},
        # DELL: single BUY
        {"Trade_ID": "T2", "Ticker": "DELL", "Action": "BUY", "Date": "2026-02-06", "Amount": 60.0},
        # SNOW
        {"Trade_ID": "T3", "Ticker": "SNOW", "Action": "BUY", "Date": "2026-04-01", "Amount": 95.0},
        # NEW
        {"Trade_ID": "T6", "Ticker": "NEW", "Action": "BUY", "Date": "2026-05-01", "Amount": 50.0},
    ]

    def fake_load_summary(portfolio: str) -> pd.DataFrame:
        assert portfolio == "CanSlim"
        return pd.DataFrame(summary_rows)

    def fake_load_details(portfolio: str) -> pd.DataFrame:
        assert portfolio == "CanSlim"
        return pd.DataFrame(details_rows)

    monkeypatch.setattr(main.db, "load_summary", fake_load_summary)
    monkeypatch.setattr(main.db, "load_details", fake_load_details)

    # Live-price overlay — mirrors the prod path the campaign uses.
    live_prices = {
        "MU":   200.0,   # live B1 = +233%, stored 119% → effective 233 ✓
        "DELL": 105.6,   # live B1 = +76%,  stored 47.81 → effective 76 ✓ (stale-peak rescue)
        "SNOW": 100.0,   # live B1 = +5.3%, stored 12   → effective 12  ✗
        "NEW":  80.0,    # live B1 = +60%,  stored None → effective 60 ✓ (stored-NaN rescue)
    }
    captured: dict[str, Any] = {}

    def fake_live_overlay(ticker_list, portfolio):
        captured["tickers"] = list(ticker_list)
        captured["portfolio"] = portfolio
        return {t: live_prices[t] for t in ticker_list if t in live_prices}

    monkeypatch.setattr(main, "_fetch_live_prices_with_manual_overlay", fake_live_overlay)

    positions = main._sr8_load_db_positions("CanSlim")
    by_ticker = {p["ticker"]: p for p in positions}

    # The DELL + NEW rescues are the load-bearing assertions for this fix.
    assert set(by_ticker.keys()) == {"MU", "DELL", "NEW"}, positions
    assert by_ticker["MU"]["b1_date"] == "2026-02-08"  # first BUY, not the add-on
    assert by_ticker["MU"]["b1_price"] == 60.0
    assert by_ticker["MU"]["shares_held"] == 135
    assert by_ticker["MU"]["avg_price"] == 80.0
    assert by_ticker["DELL"]["b1_date"] == "2026-02-06"
    assert by_ticker["DELL"]["b1_price"] == 60.0
    assert by_ticker["NEW"]["b1_date"] == "2026-05-01"
    assert by_ticker["NEW"]["b1_price"] == 50.0

    # Selection plumbing: the overlay was called with the same portfolio
    # the caller passed (so manual_price overlays land), and the upper-
    # cased candidate ticker set.
    assert captured["portfolio"] == "CanSlim"
    assert set(captured["tickers"]) == {"MU", "DELL", "SNOW", "NEW"}


def test_load_db_positions_live_fetch_failure_falls_back_to_stored(monkeypatch):
    """When the live-price overlay raises, selection must still resolve
    via the stored peak alone — no worse than the pre-rescue behavior.
    A stored peak >= 50 still includes the row; rows that relied on a
    live rescue drop out cleanly."""
    import pandas as pd
    import api.main as main

    summary_rows = [
        # Stored peak >= 50 — included via stored
        {"Trade_ID": "T1", "Ticker": "MU",  "Status": "OPEN", "Shares": 135,
         "Avg_Entry": 80.0, "B1_Entry_Price": 60.0, "B1_Max_Return_Pct": 119.14,
         "Instrument_Type": "STOCK"},
        # Stored peak below 50 — would need live to rescue; live fails → drops
        {"Trade_ID": "T2", "Ticker": "DELL", "Status": "OPEN", "Shares": 318,
         "Avg_Entry": 70.0, "B1_Entry_Price": 60.0, "B1_Max_Return_Pct": 47.81,
         "Instrument_Type": "STOCK"},
    ]
    details_rows = [
        {"Trade_ID": "T1", "Ticker": "MU", "Action": "BUY", "Date": "2026-02-08", "Amount": 60.0},
        {"Trade_ID": "T2", "Ticker": "DELL", "Action": "BUY", "Date": "2026-02-06", "Amount": 60.0},
    ]

    monkeypatch.setattr(main.db, "load_summary",
                        lambda p: pd.DataFrame(summary_rows))
    monkeypatch.setattr(main.db, "load_details",
                        lambda p: pd.DataFrame(details_rows))

    def explode(ticker_list, portfolio):
        raise RuntimeError("yfinance 429")

    monkeypatch.setattr(main, "_fetch_live_prices_with_manual_overlay", explode)

    positions = main._sr8_load_db_positions("CanSlim")
    by_ticker = {p["ticker"]: p for p in positions}
    assert set(by_ticker.keys()) == {"MU"}, positions


def test_load_db_positions_returns_empty_on_db_error(monkeypatch):
    """DB errors surface as an empty list — the endpoint shows an empty
    page, not a 500."""
    import api.main as main

    def boom(portfolio: str):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(main.db, "load_summary", boom)
    assert main._sr8_load_db_positions("CanSlim") == []
