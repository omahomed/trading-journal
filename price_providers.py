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

    Uses the batch yf.download(period='1d') path to match /api/prices/batch —
    both endpoints return the regular-session close. Using fast_info.last_price
    instead would sometimes include after-hours ticks, causing the Dashboard
    NLV and the Active Campaign Current column to disagree after 4pm ET.

    Silently drops tickers that error — the NLV service handles missing
    entries by falling back to cost basis.
    """

    def get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        if not tickers:
            return {}
        try:
            import yfinance as yf
            import pandas as pd
        except ImportError:
            return {}

        # Drop empty/duplicate symbols before hitting yfinance
        clean = list({t.strip().upper() for t in tickers if t and t.strip()})
        if not clean:
            return {}

        result: dict[str, float] = {}
        try:
            data = yf.download(clean, period="1d", progress=False)
            # yf.download returns a MultiIndex column frame for multi-ticker,
            # a flat frame for single-ticker. Normalize by pulling 'Close'.
            close = data["Close"] if "Close" in data.columns else data
            if close.empty:
                return {}
            last = close.iloc[-1]
            if len(clean) == 1:
                # Single ticker: last is a scalar Series/value
                val = float(last) if not pd.isna(last) else None
                if val is not None and val > 0:
                    result[clean[0]] = val
            else:
                # Multi-ticker: last is a Series indexed by symbol
                for sym in clean:
                    if sym in last.index and not pd.isna(last[sym]):
                        val = float(last[sym])
                        if val > 0:
                            result[sym] = val
        except Exception:
            # A download failure for the whole basket → empty result; caller
            # falls back to cost basis per position. No partial retry today.
            pass
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
