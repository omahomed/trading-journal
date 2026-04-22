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
"""
from __future__ import annotations

import os
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

    Uses yf.Ticker(symbol).fast_info to avoid the full `info` dict which
    sometimes times out on thin tickers. Falls back to a 1-day history if
    fast_info is empty. Silently drops tickers that error — the NLV service
    handles missing entries by falling back to cost basis.
    """

    def get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        if not tickers:
            return {}
        try:
            import yfinance as yf
        except ImportError:
            return {}

        result: dict[str, float] = {}
        for t in tickers:
            symbol = t.strip().upper()
            if not symbol:
                continue
            try:
                tk = yf.Ticker(symbol)
                price: float | None = None
                # fast_info is cached and cheap
                fi = getattr(tk, "fast_info", None)
                if fi is not None:
                    for attr in ("last_price", "lastPrice", "regular_market_price"):
                        val = getattr(fi, attr, None) if not isinstance(fi, dict) else fi.get(attr)
                        if val is not None:
                            price = float(val)
                            break
                if price is None:
                    hist = tk.history(period="1d")
                    if not hist.empty:
                        price = float(hist["Close"].iloc[-1])
                if price is not None and price > 0:
                    result[symbol] = price
            except Exception:
                # Drop the ticker on any error — caller falls back gracefully
                continue
        return result


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
