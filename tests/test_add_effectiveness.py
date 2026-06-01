"""Tests for GET /api/analytics/add-effectiveness.

Covers the spec checklist from the build directive:
  1. Pyramid-up classification (add.price > pre-add blended cost)
  2. Average-down classification (synthetic add below blended cost)
  3. Realized P&L attribution from lot_closures (sum where buy_trx_id =
     add.trx_id; multiplier already applied by save_summary_with_closures)
  4. Win-rate math (closed adds with realized_pl > 0 / adds with ≥1 closure)
  5. Date-window filter (add inside window included; add outside excluded)
  6. Strategy filter (filters in-scope campaigns by trades_summary.strategy)
  7. Multiplier path (an options add — open shares × multiplier in
     unrealized P&L; realized comes pre-scaled from lot_closures)
  8. Discipline guardrail (average_down_count + list)

The fixture stubs db_layer.load_summary / load_details / load_lot_closures
and patches _fetch_live_prices_with_manual_overlay so tests can control
the price snapshot without hitting yfinance. The endpoint's LIFO walk is
NOT stubbed — it runs against the synthetic detail rows the tests build,
so the pre-add blended-cost replay is end-to-end.
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


# ---------------------------------------------------------------------------
# Builders — keep them small + composable. Each test sets up its own data.
# ---------------------------------------------------------------------------


def _summary_row(
    trade_id: str,
    ticker: str,
    *,
    multiplier: float = 1.0,
    strategy: str = "CanSlim",
    status: str = "OPEN",
    instrument_type: str = "STOCK",
) -> dict[str, Any]:
    return {
        "trade_id": trade_id,
        "ticker": ticker,
        "status": status,
        "open_date": pd.Timestamp("2026-01-02"),
        "closed_date": None,
        "shares": 0.0,
        "avg_entry": 0.0,
        "avg_exit": 0.0,
        "total_cost": 0.0,
        "realized_pl": 0.0,
        "unrealized_pl": 0.0,
        "return_pct": 0.0,
        "rule": "",
        "buy_notes": "",
        "sell_rule": "",
        "sell_notes": "",
        "risk_budget": 100.0,
        "stop_loss": 0.0,
        "instrument_type": instrument_type,
        "multiplier": multiplier,
        "strategy": strategy,
        "manual_price": None,
    }


def _detail(
    trade_id: str,
    ticker: str,
    trx_id: str,
    action: str,
    date: str,
    shares: float,
    amount: float,
    *,
    rule: str = "",
    multiplier: float = 1.0,
    instrument_type: str = "STOCK",
    match_method: str | None = None,
) -> dict[str, Any]:
    return {
        "detail_id": 0,  # tests don't care about per-row PK
        "trade_id": trade_id,
        "ticker": ticker,
        "action": action,
        "date": pd.Timestamp(date),
        "shares": shares,
        "amount": amount,
        "value": shares * amount * multiplier,
        "rule": rule,
        "notes": "",
        "realized_pl": 0,
        "stop_loss": 0.0,
        "trx_id": trx_id,
        "instrument_type": instrument_type,
        "multiplier": multiplier,
        "match_method": match_method,
    }


def _closure(
    trade_id: str,
    buy_trx_id: str,
    sell_trx_id: str,
    shares: float,
    buy_price: float,
    sell_price: float,
    *,
    multiplier: float = 1.0,
    closed_at: str = "2026-04-01",
) -> dict[str, Any]:
    """realized_pl already multiplier-scaled, matching save_summary_with_closures."""
    return {
        "trade_id": trade_id,
        "buy_trx_id": buy_trx_id,
        "sell_trx_id": sell_trx_id,
        "shares": shares,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "multiplier": multiplier,
        "realized_pl": (sell_price - buy_price) * shares * multiplier,
        "closed_at": pd.Timestamp(closed_at),
    }


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed(monkeypatch):
    """Yield (state, client). State holds the dataframes db_layer is patched
    to return + the price dict the helper returns. Tests mutate state BEFORE
    making the request.
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    import db_layer

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "summary_rows": [],
        "details_rows": [],
        "lot_closures_rows": [],
        "prices": {},  # readable_ticker -> price
        "price_calls": [],  # list of (ticker_list, portfolio)
    }

    def fake_load_summary(portfolio, status=None):
        df = pd.DataFrame(state["summary_rows"])
        if df.empty:
            return df
        if status:
            df = df[df["status"].astype(str).str.upper() == status.upper()]
        return df.copy()

    def fake_load_details(portfolio, trade_id=None):
        df = pd.DataFrame(state["details_rows"])
        if df.empty:
            return df
        if trade_id is not None:
            df = df[df["trade_id"] == trade_id]
        return df.copy()

    def fake_load_lot_closures(portfolio, trade_id=None, trade_ids=None):
        df = pd.DataFrame(state["lot_closures_rows"])
        if df.empty:
            return df
        if trade_id is not None:
            df = df[df["trade_id"] == trade_id]
        elif trade_ids is not None:
            df = df[df["trade_id"].isin(list(trade_ids))]
        return df.copy()

    monkeypatch.setattr(db_layer, "load_summary", fake_load_summary)
    monkeypatch.setattr(db_layer, "load_details", fake_load_details)
    monkeypatch.setattr(db_layer, "load_lot_closures", fake_load_lot_closures)

    # _normalize_trades expects DataFrame; our builder rows are already
    # lowercase, so it's a no-op. Keep the real one in place to test the
    # actual code path.

    def fake_fetch_prices(ticker_list, portfolio=""):
        state["price_calls"].append((tuple(ticker_list), portfolio))
        return {t: state["prices"][t] for t in ticker_list if t in state["prices"]}

    monkeypatch.setattr(
        main, "_fetch_live_prices_with_manual_overlay", fake_fetch_prices,
    )

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


def _GET(client, **params):
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v != "")
    return client.get(f"/api/analytics/add-effectiveness?{qs}")


# ---------------------------------------------------------------------------
# 1. Pyramid-up classification
# ---------------------------------------------------------------------------


def test_pyramid_up_classification(stubbed):
    """B1 at $100, then A1 at $120 (above blended cost $100) → pyramid-up.
    No average-down rows; discipline counters are 0."""
    state, client = stubbed
    state["summary_rows"] = [_summary_row("202601-001", "AAPL")]
    state["details_rows"] = [
        _detail("202601-001", "AAPL", "B1", "BUY", "2026-01-05", 100, 100, rule="br1.3 Cup w/o Handle"),
        _detail("202601-001", "AAPL", "A1", "BUY", "2026-02-10", 50, 120, rule="br3.2 Reclaim 50s"),
    ]
    state["prices"] = {"AAPL": 130.0}

    r = _GET(client, portfolio="CanSlim", start="2026-01-01", end="2026-12-31")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["discipline"]["average_down_count"] == 0
    assert body["discipline"]["average_downs"] == []
    # Only one add, classified pyramid-up. Rule = the add's own rule.
    assert len(body["rules"]) == 1
    row = body["rules"][0]
    assert row["rule"] == "br3.2 Reclaim 50s"
    assert row["add_count"] == 1
    # Extension at add: (120 - 100) / 100 * 100 = 20.0 (%)
    assert row["avg_extension_at_add"] == pytest.approx(20.0, abs=1e-6)
    # Unrealized = (130 - 120) × 50 × 1 = 500
    assert row["unrealized_pl"] == pytest.approx(500.0, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Average-down classification
# ---------------------------------------------------------------------------


def test_average_down_classification(stubbed):
    """B1 at $200, then A1 at $180 (below blended cost $200) → average-down.
    Discipline counter increments and the row appears in average_downs."""
    state, client = stubbed
    state["summary_rows"] = [_summary_row("202602-007", "FOO")]
    state["details_rows"] = [
        _detail("202602-007", "FOO", "B1", "BUY", "2026-02-01", 100, 200, rule="br1.0"),
        _detail("202602-007", "FOO", "A1", "BUY", "2026-02-15", 50, 180, rule="br9.9 Bad add"),
    ]
    state["prices"] = {"FOO": 175.0}

    r = _GET(client, portfolio="CanSlim", start="2026-01-01", end="2026-12-31")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["discipline"]["average_down_count"] == 1
    [bad] = body["discipline"]["average_downs"]
    assert bad["trade_id"] == "202602-007"
    assert bad["trx_id"] == "A1"
    assert bad["ticker"] == "FOO"
    assert bad["rule"] == "br9.9 Bad add"
    # Extension: (180 - 200) / 200 * 100 = -10.0
    assert body["rules"][0]["avg_extension_at_add"] == pytest.approx(-10.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Realized attribution from lot_closures
# ---------------------------------------------------------------------------


def test_realized_attribution_from_lot_closures(stubbed):
    """SELL closes A1's 30 shares at +$15/share → realized_pl = 450.
    The add itself has 20 shares remaining → unrealized at current price."""
    state, client = stubbed
    state["summary_rows"] = [_summary_row("202603-001", "MSFT")]
    state["details_rows"] = [
        _detail("202603-001", "MSFT", "B1", "BUY", "2026-03-01", 100, 300, rule="br1.0"),
        _detail("202603-001", "MSFT", "A1", "BUY", "2026-03-10", 50, 320, rule="br3.2 Reclaim 50s"),
        # LIFO SELL closes A1 first (30 of the 50 add shares).
        _detail(
            "202603-001", "MSFT", "S1", "SELL", "2026-03-20", 30, 335,
            rule="sr1", match_method="LIFO",
        ),
    ]
    state["lot_closures_rows"] = [
        _closure("202603-001", "A1", "S1", 30, 320, 335, closed_at="2026-03-20"),
    ]
    state["prices"] = {"MSFT": 340.0}

    r = _GET(client, portfolio="CanSlim", start="2026-01-01", end="2026-12-31")
    body = r.json()
    [row] = body["rules"]
    # Realized = (335 - 320) × 30 × 1 = 450
    assert row["realized_pl"] == pytest.approx(450.0, abs=0.01)
    # Closed_count = 1 (the A1 add has a closure row)
    assert row["closed_count"] == 1
    # Unrealized for the 20 shares still open: (340 - 320) × 20 × 1 = 400
    assert row["unrealized_pl"] == pytest.approx(400.0, abs=0.01)
    # Headline totals reflect the same.
    totals = body["totals"]
    assert totals["total_realized_pl"] == pytest.approx(450.0, abs=0.01)
    assert totals["total_unrealized_pl"] == pytest.approx(400.0, abs=0.01)


# ---------------------------------------------------------------------------
# 4. Win-rate math
# ---------------------------------------------------------------------------


def test_win_rate_math(stubbed):
    """Three adds under the same rule: A1 closes +, A2 closes -, A3 closes +.
    win_rate = 2/3 = 0.6667. closed_count = 3. avg_realized_per_add =
    (total_realized) / 3."""
    state, client = stubbed
    state["summary_rows"] = [
        _summary_row("T1", "X"), _summary_row("T2", "Y"), _summary_row("T3", "Z"),
    ]
    rows: list[dict[str, Any]] = []
    closures: list[dict[str, Any]] = []
    for tid, tk, add_px, sell_px in [
        ("T1", "X", 100, 110),  # win  +10/share × 10 shares = +100
        ("T2", "Y", 200, 190),  # loss -10/share × 10 shares = -100
        ("T3", "Z", 300, 310),  # win  +10/share × 10 shares = +100
    ]:
        rows += [
            _detail(tid, tk, "B1", "BUY", "2026-01-05", 50, add_px - 5, rule="br1.0"),
            _detail(tid, tk, "A1", "BUY", "2026-02-05", 10, add_px, rule="br3.2 Reclaim 50s"),
            _detail(
                tid, tk, "S1", "SELL", "2026-03-05", 10, sell_px,
                rule="sr1", match_method="LIFO",
            ),
        ]
        closures.append(_closure(tid, "A1", "S1", 10, add_px, sell_px))
    state["details_rows"] = rows
    state["lot_closures_rows"] = closures
    state["prices"] = {"X": 0, "Y": 0, "Z": 0}  # no current prices → 0 unrealized

    r = _GET(client, portfolio="CanSlim", start="2026-01-01", end="2026-12-31")
    body = r.json()
    [row] = body["rules"]
    assert row["add_count"] == 3
    assert row["closed_count"] == 3
    assert row["win_rate"] == pytest.approx(2 / 3, abs=1e-4)
    # Total realized: +100 - 100 + 100 = +100; avg = 100 / 3 ≈ 33.33
    assert row["realized_pl"] == pytest.approx(100.0, abs=0.01)
    assert row["avg_realized_per_add"] == pytest.approx(100.0 / 3, abs=0.01)
    assert body["totals"]["overall_win_rate"] == pytest.approx(2 / 3, abs=1e-4)


# ---------------------------------------------------------------------------
# 5. Date-window filter
# ---------------------------------------------------------------------------


def test_date_window_filter_excludes_adds_outside(stubbed):
    """A1 (in window) is counted; A2 (outside window) is excluded."""
    state, client = stubbed
    state["summary_rows"] = [_summary_row("202604-001", "TSLA")]
    state["details_rows"] = [
        _detail("202604-001", "TSLA", "B1", "BUY", "2026-01-05", 100, 200, rule="br1.0"),
        _detail("202604-001", "TSLA", "A1", "BUY", "2026-02-10", 50, 210, rule="br3.2 Reclaim 50s"),
        # Add OUTSIDE the requested window:
        _detail("202604-001", "TSLA", "A2", "BUY", "2026-06-10", 50, 220, rule="br3.2 Reclaim 50s"),
    ]
    state["prices"] = {"TSLA": 230.0}

    r = _GET(client, portfolio="CanSlim", start="2026-01-01", end="2026-03-31")
    body = r.json()
    # Only A1 falls inside [2026-01-01, 2026-03-31].
    assert body["totals"]["total_adds"] == 1
    assert body["rules"][0]["add_count"] == 1


def test_date_window_inclusive_end(stubbed):
    """The end date is INCLUSIVE — an add timestamped 2026-03-31 09:30 still
    falls inside end=2026-03-31."""
    state, client = stubbed
    state["summary_rows"] = [_summary_row("X1", "X")]
    state["details_rows"] = [
        _detail("X1", "X", "B1", "BUY", "2026-01-05", 100, 100, rule="r1"),
        _detail("X1", "X", "A1", "BUY", "2026-03-31 09:30:00", 10, 120, rule="r2"),
    ]
    state["prices"] = {"X": 130.0}

    r = _GET(client, portfolio="CanSlim", start="2026-01-01", end="2026-03-31")
    assert r.json()["totals"]["total_adds"] == 1


# ---------------------------------------------------------------------------
# 6. Strategy filter
# ---------------------------------------------------------------------------


def test_strategy_filter_restricts_in_scope_campaigns(stubbed):
    """Two campaigns, one CanSlim and one StockTalk. Filtering on
    strategy=StockTalk excludes the CanSlim campaign's adds entirely."""
    state, client = stubbed
    state["summary_rows"] = [
        _summary_row("CANS-1", "AAA", strategy="CanSlim"),
        _summary_row("STOC-1", "BBB", strategy="StockTalk"),
    ]
    state["details_rows"] = [
        _detail("CANS-1", "AAA", "B1", "BUY", "2026-01-05", 100, 100, rule="r1"),
        _detail("CANS-1", "AAA", "A1", "BUY", "2026-02-05", 50, 110, rule="rA"),
        _detail("STOC-1", "BBB", "B1", "BUY", "2026-01-06", 100, 200, rule="r1"),
        _detail("STOC-1", "BBB", "A1", "BUY", "2026-02-06", 50, 210, rule="rB"),
    ]
    state["prices"] = {"AAA": 120.0, "BBB": 220.0}

    r = _GET(client, portfolio="CanSlim", start="2026-01-01", end="2026-12-31",
             strategy="StockTalk")
    body = r.json()
    assert body["totals"]["total_adds"] == 1
    [row] = body["rules"]
    assert row["rule"] == "rB"


def test_strategy_filter_all_includes_everything(stubbed):
    """strategy=all (case-insensitive) means no filter — both campaigns counted."""
    state, client = stubbed
    state["summary_rows"] = [
        _summary_row("T1", "AAA", strategy="CanSlim"),
        _summary_row("T2", "BBB", strategy="StockTalk"),
    ]
    state["details_rows"] = [
        _detail("T1", "AAA", "B1", "BUY", "2026-01-05", 100, 100, rule="r1"),
        _detail("T1", "AAA", "A1", "BUY", "2026-02-05", 50, 110, rule="rX"),
        _detail("T2", "BBB", "B1", "BUY", "2026-01-06", 100, 200, rule="r1"),
        _detail("T2", "BBB", "A1", "BUY", "2026-02-06", 50, 210, rule="rX"),
    ]
    state["prices"] = {"AAA": 120, "BBB": 220}

    r = _GET(client, portfolio="CanSlim", start="2026-01-01", end="2026-12-31",
             strategy="all")
    body = r.json()
    assert body["totals"]["total_adds"] == 2


# ---------------------------------------------------------------------------
# 7. Multiplier path (options add)
# ---------------------------------------------------------------------------


def test_options_add_uses_multiplier_for_unrealized(stubbed):
    """Options campaign with multiplier=100. Add at $5 premium, current at
    $6 → unrealized per contract = $1 × 100 (multiplier) × open_contracts.
    Realized comes pre-scaled from lot_closures (no extra math in endpoint).
    """
    state, client = stubbed
    state["summary_rows"] = [
        _summary_row("OPT-1", "FOO 261016 $50C", multiplier=100, instrument_type="OPTION"),
    ]
    state["details_rows"] = [
        _detail(
            "OPT-1", "FOO 261016 $50C", "B1", "BUY", "2026-01-05",
            10, 4, rule="opt-r1", multiplier=100, instrument_type="OPTION",
        ),
        _detail(
            "OPT-1", "FOO 261016 $50C", "A1", "BUY", "2026-02-05",
            10, 5, rule="opt-add", multiplier=100, instrument_type="OPTION",
        ),
        # LIFO SELL of 5 contracts → consumes from A1 first, leaving A1 open=5.
        _detail(
            "OPT-1", "FOO 261016 $50C", "S1", "SELL", "2026-03-01",
            5, 7, rule="sr1", multiplier=100, instrument_type="OPTION",
            match_method="LIFO",
        ),
    ]
    # Closures for the partial close of A1 — pre-scaled (5 contracts × $2 × 100).
    state["lot_closures_rows"] = [
        _closure(
            "OPT-1", "A1", "S1", 5, 5, 7,
            multiplier=100, closed_at="2026-03-01",
        ),
    ]
    # Open contracts on A1: 10 - 5 = 5.
    state["prices"] = {"FOO 261016 $50C": 6.0}

    r = _GET(client, portfolio="CanSlim", start="2026-01-01", end="2026-12-31")
    body = r.json()
    [row] = body["rules"]
    # Realized = (7 - 5) × 5 × 100 = 1000 — comes through unchanged from closures.
    assert row["realized_pl"] == pytest.approx(1000.0, abs=0.01)
    # Unrealized = (6 - 5) × 5 (open contracts) × 100 (multiplier) = 500
    assert row["unrealized_pl"] == pytest.approx(500.0, abs=0.01)


# ---------------------------------------------------------------------------
# 8. Single batched price call for all in-scope tickers
# ---------------------------------------------------------------------------


def test_single_batched_price_fetch_across_window(stubbed):
    """Even with multiple adds across multiple tickers, the endpoint should
    call the price helper EXACTLY ONCE, with all distinct tickers."""
    state, client = stubbed
    state["summary_rows"] = [
        _summary_row("T1", "AAA"), _summary_row("T2", "BBB"),
        _summary_row("T3", "CCC"),
    ]
    state["details_rows"] = []
    for tid, tk in [("T1", "AAA"), ("T2", "BBB"), ("T3", "CCC")]:
        state["details_rows"] += [
            _detail(tid, tk, "B1", "BUY", "2026-01-05", 100, 100, rule="r1"),
            _detail(tid, tk, "A1", "BUY", "2026-02-05", 50, 110, rule="rA"),
            _detail(tid, tk, "A2", "BUY", "2026-02-15", 50, 115, rule="rA"),
        ]
    state["prices"] = {"AAA": 120, "BBB": 120, "CCC": 120}

    r = _GET(client, portfolio="CanSlim", start="2026-01-01", end="2026-12-31")
    assert r.status_code == 200, r.text
    # Exactly one call.
    assert len(state["price_calls"]) == 1
    ticker_list, portfolio_arg = state["price_calls"][0]
    assert set(ticker_list) == {"AAA", "BBB", "CCC"}
    assert portfolio_arg == "CanSlim"


# ---------------------------------------------------------------------------
# Empty-state response
# ---------------------------------------------------------------------------


def test_empty_response_when_no_adds_in_window(stubbed):
    """No adds → fully-zero response, same shape, no crash."""
    state, client = stubbed
    state["summary_rows"] = [_summary_row("T1", "AAA")]
    state["details_rows"] = [
        _detail("T1", "AAA", "B1", "BUY", "2026-01-05", 100, 100, rule="r1"),
    ]
    r = _GET(client, portfolio="CanSlim", start="2026-01-01", end="2026-12-31")
    assert r.status_code == 200
    body = r.json()
    assert body["rules"] == []
    assert body["totals"]["total_adds"] == 0
    assert body["discipline"]["average_down_count"] == 0
    assert body["window"]["portfolio"] == "CanSlim"
