"""Pluggable live-price providers for the NLV service.

Today there's a single impl (yfinance), but the interface is designed so we
can drop in Polygon, Finnhub, IBKR, or a broker-specific feed later without
touching the callers. A PriceProvider is responsible for:
    - Returning the last-known price for a list of tickers
    - Tolerating partial failures: if two of five tickers fail, the other
      three still come back. NLV math degrades gracefully (falls back to
      cost basis for the unknown ones) rather than erroring out entirely.

Providers should NOT raise on missing tickers — they should return a dict
that simply omits the failed ones. The caller handles gaps.

A short-TTL per-ticker cache lives in front of the yfinance impl so that
the 3 endpoints a single dashboard load triggers (/nlv, /returns,
/prices/batch — all of which independently resolve current prices) share
a single round-trip's worth of work. Without it, a 24-position portfolio
pays 72 sequential yfinance fetches per page render.
"""
from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod


class PriceProvider(ABC):
    """Interface for anything that can resolve current prices for a basket
    of tickers. Implementations must be thread-safe and side-effect-free
    from the caller's perspective."""

    @abstractmethod
    def get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Return a mapping ticker → last-known price (float).

        Missing tickers are omitted from the result (no exception). Callers
        should tolerate a shorter-than-input dict.
        """
        raise NotImplementedError


class YFinanceProvider(PriceProvider):
    """Free, delayed, flaky but good-enough for beta.

    Uses the batch yf.download(period='1d') path to match /api/prices/batch —
    both endpoints return the regular-session close. Using fast_info.last_price
    instead would sometimes include after-hours ticks, causing the Dashboard
    NLV and the Active Campaign Current column to disagree after 4pm ET.

    Silently drops tickers that error — the NLV service handles missing
    entries by falling back to cost basis.

    A short per-ticker memo lives at module scope: yfinance fetches one
    symbol at a time (yf.download with a list is inconsistent across
    versions), so a 24-position basket is 24 round-trips. The dashboard
    fans those across 3 endpoints. Without caching, that's 72 round-trips
    per page render — observed at 12-20s on prod. With a 30s TTL, the
    second and third endpoint calls hit memory.
    """

    # (timestamp_seconds, price). Module-level so the cache is shared
    # across requests in the same process. Thread-safe via the lock below.
    _PRICE_CACHE: dict[str, tuple[float, float]] = {}
    _CACHE_TTL_SECONDS = 30.0
    _CACHE_LOCK = threading.Lock()

    # Per-ticker fetch lock. Coalesces concurrent yfinance requests for the
    # same symbol so the dashboard's parallel /nlv + /returns + /prices/batch
    # don't all fire 24 yfinance round-trips simultaneously: the first
    # request fills the cache while the others wait, then read from memory.
    _FETCH_LOCKS: dict[str, threading.Lock] = {}
    _FETCH_LOCKS_GUARD = threading.Lock()

    @classmethod
    def _get_fetch_lock(cls, symbol: str) -> threading.Lock:
        with cls._FETCH_LOCKS_GUARD:
            lock = cls._FETCH_LOCKS.get(symbol)
            if lock is None:
                lock = threading.Lock()
                cls._FETCH_LOCKS[symbol] = lock
            return lock

    def get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        if not tickers:
            return {}
        try:
            import yfinance as yf
            import pandas as pd
        except ImportError:
            return {}

        # Drop empty/duplicate symbols before hitting yfinance
        clean = sorted({t.strip().upper() for t in tickers if t and t.strip()})
        if not clean:
            return {}

        now = time.time()
        result: dict[str, float] = {}
        to_fetch: list[str] = []

        # Cache check — copy hits into result, mark misses for fetch.
        with self._CACHE_LOCK:
            for sym in clean:
                cached = self._PRICE_CACHE.get(sym)
                if cached and (now - cached[0]) < self._CACHE_TTL_SECONDS:
                    result[sym] = cached[1]
                else:
                    to_fetch.append(sym)

        def _fetch_one(symbol: str) -> None:
            """Fetch a single symbol via yf.Ticker.history. Wrapped in a
            per-ticker lock so two parallel requests for the same symbol
            don't both hit yfinance — the second one waits on the first
            and reads its result from the cache."""
            lock = self._get_fetch_lock(symbol)
            with lock:
                # Re-check cache after acquiring the lock: while we waited
                # another thread may have already fetched and populated it.
                with self._CACHE_LOCK:
                    cached = self._PRICE_CACHE.get(symbol)
                    if cached and (time.time() - cached[0]) < self._CACHE_TTL_SECONDS:
                        result[symbol] = cached[1]
                        return
                try:
                    tk = yf.Ticker(symbol)
                    hist = tk.history(period="1d", auto_adjust=False)
                    if hist is None or hist.empty:
                        return
                    val = hist["Close"].iloc[-1]
                    if pd.isna(val):
                        return
                    price = float(val)
                    if price > 0:
                        result[symbol] = price
                        with self._CACHE_LOCK:
                            self._PRICE_CACHE[symbol] = (time.time(), price)
                except Exception:
                    # Drop on any error — caller falls back to cost basis
                    return

        for sym in to_fetch:
            _fetch_one(sym)
        return result

    @classmethod
    def clear_cache(cls) -> None:
        """Test hook — drop everything in the price cache and locks."""
        with cls._CACHE_LOCK:
            cls._PRICE_CACHE.clear()
        with cls._FETCH_LOCKS_GUARD:
            cls._FETCH_LOCKS.clear()


# Module-level default. Swapping providers later is a one-line change here,
# or via env var PRICE_PROVIDER=polygon|finnhub|etc once those impls exist.
def _build_default() -> PriceProvider:
    name = (os.environ.get("PRICE_PROVIDER") or "yfinance").lower()
    if name == "yfinance":
        return YFinanceProvider()
    # Future: elif name == "polygon": return PolygonProvider()
    # Unknown provider name falls back to yfinance rather than crashing boot
    return YFinanceProvider()


_default_provider: PriceProvider | None = None


def get_price_provider() -> PriceProvider:
    """Return the shared process-wide provider, constructed lazily."""
    global _default_provider
    if _default_provider is None:
        _default_provider = _build_default()
    return _default_provider
