"""Tests for the manual_price override path in nlv_service.compute_nlv.

The override exists because yfinance can't reliably resolve OCC option
symbols. When trades_summary.manual_price is set, NLV / market_value /
unrealized_pl must use the override instead of the live yfinance result
*and* must take precedence over the cost-basis fallback when yfinance fails.

These tests stub db.get_cash_balance / db.load_summary / the price provider
so they run without a database. The endpoint-level integration test is
guarded by DATABASE_URL like the TWR ones.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import pytest


requires_db = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping endpoint tests",
)


def _summary_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a fake load_summary df with the columns compute_nlv reads."""
    return pd.DataFrame(rows, columns=[
        "Trade_ID", "Ticker", "Shares", "Avg_Entry", "Manual_Price",
    ])


class FakePriceProvider:
    def __init__(self, prices: dict[str, float]):
        self._prices = prices

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        return {s: self._prices[s] for s in symbols if s in self._prices}


@pytest.fixture
def patched_compute_nlv(monkeypatch):
    """Yield a compute_nlv configured with controlled cash, summary, and prices."""
    import nlv_service

    state: dict[str, Any] = {"cash": 0.0, "summary": _summary_df([]), "prices": {}}

    monkeypatch.setattr(nlv_service.db, "get_cash_balance", lambda pid: state["cash"])
    monkeypatch.setattr(nlv_service.db, "load_summary",
                        lambda name, status=None: state["summary"])
    monkeypatch.setattr(nlv_service, "get_price_provider",
                        lambda: FakePriceProvider(state["prices"]))

    def configure(*, cash=0.0, rows=None, prices=None):
        state["cash"] = cash
        state["summary"] = _summary_df(rows or [])
        state["prices"] = prices or {}

    return configure, lambda: nlv_service.compute_nlv(1, "TestPort")


# ---------------------------------------------------------------------------
# Equity positions — override behavior
# ---------------------------------------------------------------------------


def test_no_override_uses_live_price(patched_compute_nlv) -> None:
    configure, run = patched_compute_nlv
    configure(
        cash=10_000.0,
        rows=[{"Trade_ID": "T1", "Ticker": "AAPL", "Shares": 100,
               "Avg_Entry": 150.0, "Manual_Price": None}],
        prices={"AAPL": 200.0},
    )
    out = run()
    pos = out["positions"][0]
    assert pos["current_price"] == 200.0
    assert pos["market_value"] == 20_000.0
    assert pos.get("price_source") != "manual"
    assert out["nlv"] == 30_000.0


def test_override_takes_precedence_over_live_price(patched_compute_nlv) -> None:
    """When manual_price is set, it wins even if yfinance returns a value."""
    configure, run = patched_compute_nlv
    configure(
        cash=10_000.0,
        rows=[{"Trade_ID": "T1", "Ticker": "AAPL", "Shares": 100,
               "Avg_Entry": 150.0, "Manual_Price": 175.0}],
        prices={"AAPL": 200.0},  # yfinance has this, but override wins
    )
    out = run()
    pos = out["positions"][0]
    assert pos["current_price"] == 175.0
    assert pos["market_value"] == 17_500.0
    assert pos["price_source"] == "manual"
    # NLV uses the override, not yfinance.
    assert out["nlv"] == 27_500.0


def test_override_used_when_live_price_missing(patched_compute_nlv) -> None:
    """The whole point: yfinance fails for OCC option symbols; override fills the gap."""
    configure, run = patched_compute_nlv
    configure(
        cash=10_000.0,
        rows=[{"Trade_ID": "T1", "Ticker": "AAPL", "Shares": 100,
               "Avg_Entry": 150.0, "Manual_Price": 180.0}],
        prices={},  # yfinance returned nothing
    )
    out = run()
    pos = out["positions"][0]
    assert pos["current_price"] == 180.0
    assert pos["market_value"] == 18_000.0
    assert pos["price_source"] == "manual"
    # No price_unavailable flag because the override resolved the price.
    assert "price_unavailable" not in pos


def test_nan_manual_price_does_not_leak_into_response(patched_compute_nlv) -> None:
    """Regression: load_summary's Decimal-to-numeric conversion turns DB
    NULLs in manual_price into pandas NaN. The override filter previously
    only checked `is None`, so NaN survived `float()` and `<= 0` and ended
    up in the response dict, which Starlette's JSONResponse rejects via
    `allow_nan=False` — surfacing as a 500 from /api/prices/batch."""
    configure, run = patched_compute_nlv
    configure(
        cash=10_000.0,
        rows=[{"Trade_ID": "T1", "Ticker": "AAPL", "Shares": 100,
               "Avg_Entry": 150.0, "Manual_Price": np.nan}],
        prices={"AAPL": 200.0},
    )
    out = run()
    pos = out["positions"][0]
    # NaN must be filtered — fall through to live price.
    assert pos["current_price"] == 200.0
    assert pos.get("price_source") != "manual"
    # NLV must be a finite, JSON-serializable number.
    assert isinstance(out["nlv"], (int, float))
    assert out["nlv"] == 30_000.0


def test_zero_or_negative_override_ignored(patched_compute_nlv) -> None:
    """A 0 or negative override is treated as 'no override' — falls back to
    cost basis when yfinance is also missing."""
    configure, run = patched_compute_nlv
    configure(
        cash=10_000.0,
        rows=[{"Trade_ID": "T1", "Ticker": "AAPL", "Shares": 100,
               "Avg_Entry": 150.0, "Manual_Price": 0.0}],
        prices={},
    )
    out = run()
    pos = out["positions"][0]
    assert pos["current_price"] is None
    assert pos["market_value"] == 15_000.0  # cost basis = shares × avg_entry
    assert pos.get("price_unavailable") is True


# ---------------------------------------------------------------------------
# Option positions — override applies the 100x contract multiplier
# ---------------------------------------------------------------------------


def test_option_override_applies_contract_multiplier(patched_compute_nlv) -> None:
    """is_option_ticker matches the 'TICKER YYMMDD $STRIKEC/P' format. The
    override is per-contract premium; the 100x multiplier still applies."""
    configure, run = patched_compute_nlv
    configure(
        cash=5_000.0,
        rows=[{"Trade_ID": "OPT1", "Ticker": "AAPL 260321 $200C",
               "Shares": 2, "Avg_Entry": 3.50, "Manual_Price": 5.20}],
        prices={},  # yfinance can't resolve OCC chains — that's the whole problem
    )
    out = run()
    pos = out["positions"][0]
    assert pos["current_price"] == 5.20
    # 2 contracts × $5.20 × 100 = $1,040
    assert pos["market_value"] == 1_040.0
    assert pos["price_source"] == "manual"
    assert out["nlv"] == 6_040.0


# ---------------------------------------------------------------------------
# Migration tolerance — DataFrame without Manual_Price column
# ---------------------------------------------------------------------------


def test_pre_migration_summary_df_does_not_crash(monkeypatch) -> None:
    """If load_summary returns columns without Manual_Price (DB before
    migration 012 ran), compute_nlv must still work — overrides are simply
    treated as absent."""
    import nlv_service

    pre_migration = pd.DataFrame([
        {"Trade_ID": "T1", "Ticker": "AAPL", "Shares": 100, "Avg_Entry": 150.0},
    ], columns=["Trade_ID", "Ticker", "Shares", "Avg_Entry"])

    monkeypatch.setattr(nlv_service.db, "get_cash_balance", lambda pid: 10_000.0)
    monkeypatch.setattr(nlv_service.db, "load_summary",
                        lambda name, status=None: pre_migration)
    monkeypatch.setattr(nlv_service, "get_price_provider",
                        lambda: FakePriceProvider({"AAPL": 200.0}))

    out = nlv_service.compute_nlv(1, "TestPort")
    assert out["positions"][0]["current_price"] == 200.0
    assert out["nlv"] == 30_000.0


# ---------------------------------------------------------------------------
# Endpoint integration — requires DATABASE_URL with the migration applied
# ---------------------------------------------------------------------------


@requires_db
def test_set_manual_price_round_trip() -> None:
    """Set + clear an override on a real CanSlim open trade. Skips when
    CanSlim has no open option positions (edge case)."""
    import db_layer as db

    portfolios = db.list_portfolios()
    canslim = next((p for p in portfolios if p.get("name") == "CanSlim"), None)
    if canslim is None:
        pytest.skip("CanSlim portfolio not present in this database")

    summary = db.load_summary("CanSlim", status="OPEN")
    if summary is None or summary.empty:
        pytest.skip("No open positions to test against")
    if "Manual_Price" not in summary.columns:
        pytest.skip("manual_price column missing — migration 012 not applied")

    target = summary.iloc[0]
    trade_id = str(target["Trade_ID"])
    original = target.get("Manual_Price")

    # Set
    set_res = db.set_manual_price("CanSlim", trade_id, 99.99)
    assert set_res is not None
    assert float(set_res["manual_price"]) == 99.99

    # Restore (use original or clear)
    restore_to = float(original) if original is not None else None
    db.set_manual_price("CanSlim", trade_id, restore_to)
