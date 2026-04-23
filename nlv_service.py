"""Derived Net Liquidation Value (NLV) for a portfolio.

    NLV = cash_balance + Σ(open_position.shares × live_price)

Cash balance is sourced from the cash_transactions ledger (migrations 009).
Open positions come from the trades_summary table. Live prices come from
the configured PriceProvider (yfinance today; swappable later).

Prices that can't be resolved don't crash the calculation — we fall back to
the position's cost basis so the NLV remains monotonically meaningful, and
we flag the position as price_unavailable so the UI can show a warning.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import db_layer as db
from price_providers import get_price_provider
from tickers import is_option_ticker, to_occ_symbol

# Options positions at 100x multiplier (1 contract = 100 shares of underlying).
# Mirrors the convention in active-campaign.tsx so NLV matches the React UI.
_OPTION_CONTRACT_MULTIPLIER = 100


def compute_nlv(portfolio_id: int, portfolio_name: str) -> dict[str, Any]:
    """Snapshot current NLV for a portfolio.

    Returns a dict shaped for JSON serialization:
        {
          "cash": float,
          "market_value": float,
          "nlv": float,
          "positions": [ { ticker, shares, avg_entry, current_price,
                           market_value, unrealized_pl,
                           price_unavailable? }, ... ],
          "as_of": ISO timestamp
        }
    """
    cash = db.get_cash_balance(portfolio_id)

    summary_df = db.load_summary(portfolio_name, status="OPEN")
    positions: list[dict[str, Any]] = []

    if summary_df is None or summary_df.empty:
        return {
            "cash": round(cash, 2),
            "market_value": 0.0,
            "nlv": round(cash, 2),
            "positions": [],
            "as_of": datetime.now().isoformat(),
        }

    # Build a lookup from the app's readable ticker → the symbol yfinance
    # actually accepts. Options need an OCC conversion; equities pass through.
    # Keeping the dict keyed by the original readable ticker lets us resolve
    # prices back to positions without another transformation on the read side.
    original_tickers = summary_df["ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    ticker_to_yf: dict[str, str | None] = {}
    yf_symbols: list[str] = []
    for t in original_tickers:
        if is_option_ticker(t):
            occ = to_occ_symbol(t)
            ticker_to_yf[t] = occ  # may be None if malformed
            if occ:
                yf_symbols.append(occ)
        else:
            ticker_to_yf[t] = t
            yf_symbols.append(t)
    prices = get_price_provider().get_current_prices(yf_symbols) if yf_symbols else {}

    market_value = 0.0
    for _, row in summary_df.iterrows():
        ticker = str(row.get("ticker", "") or "").upper()
        shares = float(row.get("shares", 0) or 0)
        avg_entry = float(row.get("avg_entry", 0) or 0)
        if shares <= 0:
            continue

        is_option = is_option_ticker(ticker)
        multiplier = _OPTION_CONTRACT_MULTIPLIER if is_option else 1
        yf_sym = ticker_to_yf.get(ticker)
        live = prices.get(yf_sym) if yf_sym else None

        if live is not None:
            mv = shares * live * multiplier
            position = {
                "ticker": ticker,
                "shares": shares,
                "avg_entry": round(avg_entry, 4),
                "current_price": round(live, 4),
                "market_value": round(mv, 2),
                "unrealized_pl": round(mv - shares * avg_entry * multiplier, 2),
            }
        else:
            # Price unknown — fall back to cost basis so NLV stays sensible.
            # For equities: cost = shares × avg_entry. For options: avg_entry
            # is already the per-contract premium × 100, so we use shares ×
            # avg_entry directly — do NOT apply the multiplier again.
            cost = shares * avg_entry
            mv = cost
            position = {
                "ticker": ticker,
                "shares": shares,
                "avg_entry": round(avg_entry, 4),
                "current_price": None,
                "market_value": round(cost, 2),
                "unrealized_pl": 0.0,
                "price_unavailable": True,
            }
        market_value += mv
        positions.append(position)

    return {
        "cash": round(cash, 2),
        "market_value": round(market_value, 2),
        "nlv": round(cash + market_value, 2),
        "positions": positions,
        "as_of": datetime.now().isoformat(),
    }
