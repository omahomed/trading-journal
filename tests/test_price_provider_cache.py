"""Tests for YFinanceProvider's per-ticker TTL cache.

The cache exists because the dashboard fires 3 endpoints that each
independently resolve current prices for the user's open positions
(/nlv, /returns, /prices/batch). Without caching, a 24-position
portfolio paid 72 sequential yfinance round-trips per page render —
observed at 12-20s in prod. With a 30s TTL, only the first endpoint
in any given page load actually hits yfinance.
"""
from __future__ import annotations

import time

import pytest

from price_providers import YFinanceProvider


@pytest.fixture(autouse=True)
def _reset_cache():
    YFinanceProvider.clear_cache()
    yield
    YFinanceProvider.clear_cache()


def test_cached_ticker_skips_yfinance_fetch(monkeypatch) -> None:
    """A ticker present in the cache must be returned without invoking
    yfinance — proves the cache short-circuits the slow path."""
    YFinanceProvider._PRICE_CACHE["AAPL"] = (time.time(), 200.0)

    fetch_calls: list[str] = []

    class _ExplodingTicker:
        def __init__(self, symbol):
            fetch_calls.append(symbol)
            raise AssertionError(f"yfinance fetched {symbol} despite cache hit")

    import sys
    fake_yf = type(sys)("yfinance")  # ModuleType
    fake_yf.Ticker = _ExplodingTicker  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)

    out = YFinanceProvider().get_current_prices(["AAPL"])
    assert out == {"AAPL": 200.0}
    assert fetch_calls == [], "Cache miss occurred — fetched when it shouldn't have"


def test_expired_entry_refetches(monkeypatch) -> None:
    """An entry older than TTL must trigger a fresh fetch."""
    expired = time.time() - (YFinanceProvider._CACHE_TTL_SECONDS + 1)
    YFinanceProvider._PRICE_CACHE["AAPL"] = (expired, 100.0)

    fetched: list[str] = []

    class _StubHist:
        empty = False
        def __getitem__(self, key):
            class _Col:
                def __init__(self, val): self.val = val
                @property
                def iloc(self):
                    class _ILoc:
                        def __init__(self, val): self.val = val
                        def __getitem__(self, _): return self.val
                    return _ILoc(self.val)
            return _Col(250.0)

    class _StubTicker:
        def __init__(self, symbol):
            fetched.append(symbol)
            self.symbol = symbol
        def history(self, period, auto_adjust):
            return _StubHist()

    import sys
    fake_yf = type(sys)("yfinance")
    fake_yf.Ticker = _StubTicker  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)

    out = YFinanceProvider().get_current_prices(["AAPL"])
    assert fetched == ["AAPL"], "Expired entry should have been refetched"
    assert out["AAPL"] == 250.0


def test_concurrent_fetches_for_same_ticker_coalesce(monkeypatch) -> None:
    """Regression: when two threads both find an empty cache and both
    request the same ticker, only ONE yfinance fetch must happen — the
    second thread waits for the first to fill the cache then reads it.

    Without coalescing, the dashboard's three parallel endpoints
    (/nlv, /returns, /prices/batch) all fired their own yfinance round-trips
    simultaneously. The second-arriving request paid the full cost again
    instead of riding on the first's result.
    """
    import threading

    fetch_count = 0
    fetch_started = threading.Event()
    release_fetch = threading.Event()
    fetch_lock = threading.Lock()

    class _SlowHist:
        empty = False
        def __getitem__(self, key):
            class _Col:
                @property
                def iloc(self):
                    class _ILoc:
                        def __getitem__(self, _): return 250.0
                    return _ILoc()
            return _Col()

    class _SlowTicker:
        def __init__(self, symbol):
            nonlocal fetch_count
            with fetch_lock:
                fetch_count += 1
            self.symbol = symbol
        def history(self, period, auto_adjust):
            # Signal that the fetch has started, then block until released.
            # This gives the second thread time to also enter get_current_prices
            # and find the cache still empty — replicates the stampede setup.
            fetch_started.set()
            release_fetch.wait(timeout=2.0)
            return _SlowHist()

    import sys
    fake_yf = type(sys)("yfinance")
    fake_yf.Ticker = _SlowTicker  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)

    results: dict[int, dict[str, float]] = {}

    def worker(idx: int) -> None:
        results[idx] = YFinanceProvider().get_current_prices(["AAPL"])

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))

    t1.start()
    # Wait for thread 1 to enter the fetch (the slow yfinance call), then
    # start thread 2 — guarantees thread 2 races to the cache and finds it
    # empty, exactly the prod stampede scenario.
    fetch_started.wait(timeout=2.0)
    t2.start()
    # Let thread 1's fetch complete; thread 2 must read from the cache.
    release_fetch.set()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert results[1] == {"AAPL": 250.0}
    assert results[2] == {"AAPL": 250.0}
    assert fetch_count == 1, (
        f"Expected exactly one yfinance fetch under stampede; got {fetch_count}"
    )


def test_partial_cache_hit_only_fetches_misses(monkeypatch) -> None:
    """When some tickers are cached and others aren't, only the misses
    should hit yfinance — saves the redundant work."""
    YFinanceProvider._PRICE_CACHE["AAPL"] = (time.time(), 200.0)

    fetched: list[str] = []

    class _StubHist:
        empty = False
        def __getitem__(self, key):
            class _Col:
                @property
                def iloc(self):
                    class _ILoc:
                        def __getitem__(self, _): return 50.0
                    return _ILoc()
            return _Col()

    class _StubTicker:
        def __init__(self, symbol):
            fetched.append(symbol)
        def history(self, period, auto_adjust):
            return _StubHist()

    import sys
    fake_yf = type(sys)("yfinance")
    fake_yf.Ticker = _StubTicker  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)

    out = YFinanceProvider().get_current_prices(["AAPL", "MSFT"])
    assert fetched == ["MSFT"], "AAPL was cached; only MSFT should fetch"
    assert out["AAPL"] == 200.0  # from cache
    assert out["MSFT"] == 50.0   # from fetch
