"""Tests for /api/prices/lookup-batch — batched live price + ATR lookup.

The single-ticker /api/prices/lookup endpoint has a 30/minute slowapi
limit that's been silently starving Portfolio Heat's fan-out loop on
the user's 3 oldest CanSlim positions (SNDK, DELL, COHR). The batch
endpoint trades N HTTP calls for 1, dropping the per-page rate-limit
slot usage from N to 1.

Coverage:
  - Happy path (3 ok tickers)
  - Mixed status (ok + empty + sparse + error)
  - Validation: missing / over-50 tickers → 400
  - Whitespace + case normalization
  - Dedupe
  - Cache short-circuit (cached ticker doesn't trigger yfinance)
  - Rate limit decorator (11th call within a minute → 429)
"""
from __future__ import annotations

import time
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


def _make_history_df(rows: int = 40) -> pd.DataFrame:
    """Deterministic OHLCV with non-degenerate ranges so ATR computes
    to a real value (and so the sma_low > 0 guard fires correctly)."""
    return pd.DataFrame({
        "Open":  [100.0 + i * 0.1 for i in range(rows)],
        "High":  [101.0 + i * 0.1 for i in range(rows)],
        "Low":   [99.0 + i * 0.1 for i in range(rows)],
        "Close": [100.5 + i * 0.1 for i in range(rows)],
        "Volume": [1_000_000] * rows,
    })


@pytest.fixture
def batch_stubs(monkeypatch):
    """Stub the yfinance call inside /api/prices/lookup-batch and reset
    the in-process ATR cache + limiter between tests."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    # Reset the cache so per-test stubs aren't bypassed by carryover.
    main._atr_cache.clear()

    # Per-ticker mock router. Test sets `behavior[ticker] = "ok" | "empty"
    # | "sparse" | "error"` and the fake yfinance returns accordingly.
    behavior: dict[str, str] = {}

    class _FakeTicker:
        def __init__(self, t: str): self.t = t
        def history(self, period: str = "40d", **_):
            b = behavior.get(self.t, "ok")
            if b == "empty":  return pd.DataFrame()
            if b == "sparse": return _make_history_df(rows=10)
            if b == "error":  raise RuntimeError(f"yfinance simulated error for {self.t}")
            return _make_history_df(rows=40)

    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", _FakeTicker)

    # Disable the limiter so most tests don't hit 429; the dedicated
    # rate-limit test re-enables it explicitly.
    original_enabled = main.limiter.enabled
    main.limiter.enabled = False

    client = TestClient(main.app, headers=_auth_headers())
    state: dict[str, Any] = {"behavior": behavior, "main": main}
    yield state, client

    main.limiter.enabled = original_enabled


def _get(client, tickers: str):
    return client.get(f"/api/prices/lookup-batch?tickers={tickers}")


# ─────────────────────────────────────────────────────────────────────────────


def test_happy_path_three_ok(batch_stubs):
    """All three tickers fall through the full ATR path and return ok."""
    state, client = batch_stubs
    state["behavior"].update({"AAPL": "ok", "NVDA": "ok", "SNDK": "ok"})
    r = _get(client, "AAPL,NVDA,SNDK")
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert [x["ticker"] for x in results] == ["AAPL", "NVDA", "SNDK"]
    for row in results:
        assert row["status"] == "ok"
        assert isinstance(row["price"], (int, float)) and row["price"] > 0
        assert isinstance(row["atr_pct"], (int, float)) and row["atr_pct"] > 0


def test_mixed_status(batch_stubs):
    """Every status branch fires once. Each result carries the right tag
    + the appropriate price/atr_pct shape (null vs number)."""
    state, client = batch_stubs
    state["behavior"].update({
        "GOOD": "ok", "EMPTY": "empty", "SHORT": "sparse", "BAD": "error",
    })
    r = _get(client, "GOOD,EMPTY,SHORT,BAD")
    assert r.status_code == 200, r.text
    by_t = {x["ticker"]: x for x in r.json()["results"]}

    assert by_t["GOOD"]["status"] == "ok"
    assert by_t["GOOD"]["price"] is not None
    assert by_t["GOOD"]["atr_pct"] is not None

    assert by_t["EMPTY"]["status"] == "empty"
    assert by_t["EMPTY"]["price"] is None
    assert by_t["EMPTY"]["atr_pct"] is None

    assert by_t["SHORT"]["status"] == "sparse"
    assert by_t["SHORT"]["price"] is not None         # last close surfaced
    assert by_t["SHORT"]["atr_pct"] == 0.0             # but ATR not meaningful

    assert by_t["BAD"]["status"] == "error"
    assert by_t["BAD"]["price"] is None
    assert by_t["BAD"]["atr_pct"] is None


def test_empty_tickers_param_returns_400(batch_stubs):
    _, client = batch_stubs
    r = client.get("/api/prices/lookup-batch?tickers=")
    assert r.status_code == 400
    assert "No tickers" in r.json()["detail"]


def test_only_whitespace_tickers_returns_400(batch_stubs):
    """`?tickers=,,,  ,` normalizes to zero valid tickers → 400."""
    _, client = batch_stubs
    r = _get(client, ",,,%20,")
    assert r.status_code == 400
    assert "No valid tickers" in r.json()["detail"]


def test_too_many_tickers_returns_400(batch_stubs):
    """51 distinct tickers exceeds the 50-per-call ceiling."""
    _, client = batch_stubs
    tickers = ",".join(f"T{i}" for i in range(51))
    r = _get(client, tickers)
    assert r.status_code == 400
    assert "Too many tickers" in r.json()["detail"]


def test_normalization_whitespace_and_case(batch_stubs):
    """' aapl , NVDA ' → both uppercased and trimmed."""
    state, client = batch_stubs
    state["behavior"].update({"AAPL": "ok", "NVDA": "ok"})
    r = _get(client, "%20aapl%20,%20NVDA%20")
    assert r.status_code == 200, r.text
    tickers = [x["ticker"] for x in r.json()["results"]]
    assert tickers == ["AAPL", "NVDA"]


def test_dedupe(batch_stubs):
    """`AAPL,AAPL,AAPL` should produce one result, not three."""
    state, client = batch_stubs
    state["behavior"].update({"AAPL": "ok"})
    r = _get(client, "AAPL,AAPL,AAPL")
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["ticker"] == "AAPL"


def test_cache_short_circuits_yfinance(batch_stubs):
    """A cached ticker doesn't trigger yfinance. Pre-seed the cache and
    flip the per-ticker mock to 'error' — if the cache works, the result
    is still ok; if it doesn't, the error fires."""
    state, client = batch_stubs
    main = state["main"]
    main._atr_cache["CACHED"] = (time.time(), {
        "ticker": "CACHED", "price": 250.00, "atr": 2.5, "atr_pct": 3.45,
    })
    state["behavior"]["CACHED"] = "error"  # would fail if cache were bypassed
    r = _get(client, "CACHED")
    assert r.status_code == 200, r.text
    row = r.json()["results"][0]
    assert row["status"] == "ok"
    assert row["price"] == 250.00
    assert row["atr_pct"] == 3.45


def test_rate_limit_decorator(batch_stubs):
    """11th call within a minute returns 429. Re-enables the limiter
    (the fixture disabled it for everyone else)."""
    state, client = batch_stubs
    main = state["main"]
    state["behavior"]["X"] = "ok"
    # Reset any prior counter state by toggling enabled off-then-on.
    main.limiter.enabled = True
    main.limiter.reset()
    try:
        last_status = None
        for i in range(11):
            last_status = _get(client, "X").status_code
        assert last_status == 429
    finally:
        main.limiter.enabled = False
