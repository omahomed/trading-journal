"""Tests for /api/prices/lookup — verifying the endpoint returns
ema_21 + sma_50 alongside price + ATR, with null-on-sparse-history
semantics.

The single-ticker endpoint is what Position Sizer + Log Buy +
New Entry hit on every ticker input. Adding MA levels to its payload
is the "display 21 EMA / 50 SMA like ATR" change; these tests pin
the wire format so the frontend can rely on the null contract.
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


def _make_history_df(rows: int) -> pd.DataFrame:
    """Deterministic OHLCV with a rising Close so the EMA / SMA math
    lands on distinguishable numbers we can assert on."""
    return pd.DataFrame({
        "Open":   [100.0 + i * 0.1 for i in range(rows)],
        "High":   [101.0 + i * 0.1 for i in range(rows)],
        "Low":    [99.0 + i * 0.1 for i in range(rows)],
        "Close":  [100.5 + i * 0.1 for i in range(rows)],
        "Volume": [1_000_000] * rows,
    })


@pytest.fixture
def lookup_stubs(monkeypatch):
    """Stub yfinance + disable rate limiter for the single-ticker
    /api/prices/lookup endpoint. Test controls the returned bar count
    via `rows[ticker]`."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)
    main._atr_cache.clear()

    rows: dict[str, int] = {}

    class _FakeTicker:
        def __init__(self, t: str): self.t = t
        def history(self, period: str = "90d", **_):
            n = rows.get(self.t, 62)
            if n <= 0: return pd.DataFrame()
            return _make_history_df(n)

    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", _FakeTicker)

    original_enabled = main.limiter.enabled
    main.limiter.enabled = False
    client = TestClient(main.app, headers=_auth_headers())
    state: dict[str, Any] = {"rows": rows, "main": main}
    yield state, client
    main.limiter.enabled = original_enabled


def _get(client, ticker: str):
    return client.get(f"/api/prices/lookup?ticker={ticker}")


def test_ema_21_and_sma_50_present_on_healthy_ticker(lookup_stubs):
    """62 bars → ema_21 and sma_50 both present, numeric, rounded to 2dp."""
    state, client = lookup_stubs
    state["rows"]["AAPL"] = 62
    r = _get(client, "AAPL")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert isinstance(body["price"], (int, float))
    assert isinstance(body["ema_21"], (int, float))
    assert isinstance(body["sma_50"], (int, float))
    # Rounded to 2dp — no long floating-point tails.
    assert body["ema_21"] == round(body["ema_21"], 2)
    assert body["sma_50"] == round(body["sma_50"], 2)


def test_ema_matches_pandas_ewm_span_21(lookup_stubs):
    """The endpoint's EMA should equal a direct pandas computation
    (ewm(span=21, adjust=False).mean()) on the same close series."""
    state, client = lookup_stubs
    state["rows"]["MSFT"] = 62
    r = _get(client, "MSFT")
    body = r.json()

    close = _make_history_df(62)["Close"]
    expected = round(float(close.ewm(span=21, adjust=False).mean().iloc[-1]), 2)
    assert body["ema_21"] == expected


def test_sma50_matches_pandas_rolling(lookup_stubs):
    """SMA equals the tail-50 mean of the same close series."""
    state, client = lookup_stubs
    state["rows"]["GOOG"] = 62
    r = _get(client, "GOOG")
    body = r.json()

    close = _make_history_df(62)["Close"]
    expected = round(float(close.tail(50).mean()), 2)
    assert body["sma_50"] == expected


def test_sma_50_null_when_history_below_50_bars(lookup_stubs):
    """Ticker with 30 bars — has enough for EMA (21) but not SMA (50).
    Endpoint returns sma_50 = null and ema_21 numeric."""
    state, client = lookup_stubs
    state["rows"]["YOUNG"] = 30
    r = _get(client, "YOUNG")
    body = r.json()
    assert body["sma_50"] is None
    assert isinstance(body["ema_21"], (int, float))


def test_both_null_when_history_below_21_bars(lookup_stubs):
    """Ticker with 10 bars — not enough for either MA. Both null."""
    state, client = lookup_stubs
    state["rows"]["FRESH"] = 10
    r = _get(client, "FRESH")
    body = r.json()
    assert body["ema_21"] is None
    assert body["sma_50"] is None


def test_empty_history_503_still_raises(lookup_stubs):
    """0 bars from yfinance → 503; MAs never enter the picture."""
    state, client = lookup_stubs
    state["rows"]["DEAD"] = 0
    r = _get(client, "DEAD")
    assert r.status_code == 503


def test_cache_short_circuit_preserves_ma_fields(lookup_stubs):
    """The in-memory _atr_cache stores the whole payload. A cached
    second call must return the same ema_21 / sma_50 as the first —
    no round-trip to yfinance."""
    state, client = lookup_stubs
    state["rows"]["META"] = 62
    r1 = _get(client, "META").json()

    # Zero out yfinance so any bypass of the cache would error.
    state["rows"]["META"] = 0
    r2 = _get(client, "META").json()

    assert r1 == r2
    assert r2["ema_21"] is not None
    assert r2["sma_50"] is not None
