"""Helpers for normalizing ticker symbols across the app.

Today this is just the readable option format ↔ OCC format conversion used
when pricing options via yfinance. Extracted into its own module so every
call site (price lookup, batch prices, NLV service) shares the same regex.
"""
from __future__ import annotations

import re

# Readable option format used throughout the app:
#     UNDERLYING YYMMDD $STRIKE<C|P>
# e.g. "LUMN 270115 $7C", "AAPL 260620 $195.5P"
_OPTION_RE = re.compile(r"^(\S+)\s+(\d{6})\s+\$([0-9.]+)(C|P)$")


def is_option_ticker(ticker: str | None) -> bool:
    """True when the ticker matches the readable option format."""
    if not ticker:
        return False
    return "$" in ticker and bool(re.search(r"\d{6}", ticker))


def to_occ_symbol(readable_ticker: str | None) -> str | None:
    """Convert a readable option ticker to OCC format (accepted by yfinance).

    Returns None on anything that doesn't match the readable format — caller
    should fall back to treating the position as un-priceable.
    """
    if not readable_ticker:
        return None
    try:
        m = _OPTION_RE.match(readable_ticker.strip())
        if not m:
            return None
        underlying = m.group(1)
        expiry = m.group(2)
        strike = float(m.group(3))
        put_call = m.group(4)
        strike_int = int(strike * 1000)
        return f"{underlying}{expiry}{put_call}{strike_int:08d}"
    except Exception:
        return None
