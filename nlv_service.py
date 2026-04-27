"""Derived Net Liquidation Value (NLV) + returns for a portfolio.

    NLV = cash_balance + Σ(open_position.shares × live_price)

Cash balance is sourced from the cash_transactions ledger (migrations 009).
Open positions come from the trades_summary table. Live prices come from
the configured PriceProvider (yfinance today; swappable later).

Prices that can't be resolved don't crash the calculation — we fall back to
the position's cost basis so the NLV remains monotonically meaningful, and
we flag the position as price_unavailable so the UI can show a warning.

Also exposes:
  - compute_returns()     — money-snapshot LTD/YTD: (NLV − net_contributions)/net_contributions
  - compute_twr_returns() — time-weighted LTD/YTD chained from daily journal returns
"""
from __future__ import annotations

import math
from datetime import datetime, date
from typing import Any

import pandas as pd

import db_layer as db
from price_providers import get_price_provider
from tickers import is_option_ticker, to_occ_symbol
from trade_calc import normalize_journal_columns

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

    # load_summary returns Title_Case columns (Ticker, Shares, Avg_Entry...)
    # because it's the legacy CSV-era convention. Read from those directly —
    # we don't need _normalize_trades here since we only touch three fields.
    ticker_col = "Ticker" if "Ticker" in summary_df.columns else "ticker"
    shares_col = "Shares" if "Shares" in summary_df.columns else "shares"
    entry_col  = "Avg_Entry" if "Avg_Entry" in summary_df.columns else "avg_entry"

    # Build a lookup from the app's readable ticker → the symbol yfinance
    # actually accepts. Options need an OCC conversion; equities pass through.
    # Keeping the dict keyed by the original readable ticker lets us resolve
    # prices back to positions without another transformation on the read side.
    original_tickers = (
        summary_df[ticker_col].dropna().astype(str).str.strip().str.upper().unique().tolist()
    )
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

    # Manual price overrides — keyed by upper-cased ticker. When set, takes
    # precedence over the live yfinance result. Primarily a workaround for
    # OCC option symbols yfinance can't resolve; equities can use it too.
    # Tolerant of pre-migration-012 dataframes where the column doesn't exist.
    manual_col = "Manual_Price" if "Manual_Price" in summary_df.columns else (
        "manual_price" if "manual_price" in summary_df.columns else None
    )
    manual_overrides: dict[str, float] = {}
    if manual_col is not None:
        for _, row in summary_df.iterrows():
            mp = row.get(manual_col)
            # load_summary's Decimal-to-numeric conversion turns DB NULLs
            # into NaN, which slips past `mp is None` and survives both
            # float() and `<= 0`. Use pd.isna to filter both shapes.
            if pd.isna(mp):
                continue
            try:
                mp_f = float(mp)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(mp_f) or mp_f <= 0:
                continue
            tkr = str(row.get(ticker_col, "") or "").upper()
            if tkr:
                manual_overrides[tkr] = mp_f

    market_value = 0.0
    for _, row in summary_df.iterrows():
        ticker = str(row.get(ticker_col, "") or "").upper()
        shares = float(row.get(shares_col, 0) or 0)
        avg_entry = float(row.get(entry_col, 0) or 0)
        if shares <= 0:
            continue

        is_option = is_option_ticker(ticker)
        multiplier = _OPTION_CONTRACT_MULTIPLIER if is_option else 1
        yf_sym = ticker_to_yf.get(ticker)
        live = prices.get(yf_sym) if yf_sym else None
        override = manual_overrides.get(ticker)

        # Resolution order: manual override → yfinance live → cost-basis fallback.
        resolved: float | None = override if override is not None else live

        if resolved is not None:
            mv = shares * resolved * multiplier
            position = {
                "ticker": ticker,
                "shares": shares,
                "avg_entry": round(avg_entry, 4),
                "current_price": round(resolved, 4),
                "market_value": round(mv, 2),
                "unrealized_pl": round(mv - shares * avg_entry * multiplier, 2),
            }
            if override is not None:
                position["price_source"] = "manual"
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


def _as_year(value) -> int | None:
    """Best-effort year extraction from a DATE / TIMESTAMP / ISO string."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.year
    try:
        return int(str(value)[:4])
    except (ValueError, TypeError):
        return None


def compute_returns(portfolio_id: int, portfolio_name: str,
                    portfolio_row: dict[str, Any]) -> dict[str, Any]:
    """LTD + YTD returns for a portfolio. Builds on compute_nlv().

    LTD:
        net_contributions = Σ cash_tx with source IN (deposit, withdraw, reconcile)
        ltd_pl     = NLV − net_contributions
        ltd_pct    = (ltd_pl / net_contributions) × 100      [guarded for /0]

    YTD:
        A portfolio with a reset_date (or created_at) in the current year
        has its whole lifespan within YTD, so YTD == LTD. Any portfolio
        that started in a prior year needs a start-of-year NLV snapshot to
        compute YTD meaningfully — that snapshot only exists once the EOD
        cron (Phase 4) is running. Until then: ytd_available = false.
    """
    nlv_snap = compute_nlv(portfolio_id, portfolio_name)
    nlv = float(nlv_snap["nlv"])

    net_contributions = db.get_net_contributions(portfolio_id)
    ltd_pl = nlv - net_contributions
    ltd_pct = (ltd_pl / net_contributions * 100) if net_contributions > 0 else 0.0

    this_year = datetime.now().year
    effective_year = _as_year(portfolio_row.get("reset_date")) \
                     or _as_year(portfolio_row.get("created_at")) \
                     or this_year

    ytd_available = effective_year >= this_year
    ytd_pl: float | None = ltd_pl if ytd_available else None
    ytd_pct: float | None = ltd_pct if ytd_available else None

    return {
        "nlv": round(nlv, 2),
        "net_contributions": round(net_contributions, 2),
        "ltd_pl": round(ltd_pl, 2),
        "ltd_pct": round(ltd_pct, 4),
        "ytd_pl": round(ytd_pl, 2) if ytd_pl is not None else None,
        "ytd_pct": round(ytd_pct, 4) if ytd_pct is not None else None,
        "ytd_available": ytd_available,
        "as_of": datetime.now().isoformat(),
    }


def _compute_twr_from_journal_df(df: pd.DataFrame) -> dict[str, Any]:
    """Pure: given a normalized journal DataFrame with day/beg_nlv/end_nlv/cash_change,
    return TWR LTD + YTD percentages.

    Daily TWR uses the flow-at-start-of-day convention (Modified Dietz daily):

        adjusted_beg = beg_nlv + cash_change
        daily_return = (end_nlv − adjusted_beg) / adjusted_beg

    LTD = (∏(1 + daily_return) − 1) × 100 over the entire history.
    YTD = (∏(1 + daily_return) − 1) × 100 over rows with day >= Jan 1 of
          the current year. Available only when at least one row falls in
          the current year.

    Rows where adjusted_beg <= 0 (typo'd or pre-funding entries) contribute a
    daily_return of 0 — they pass through the cumprod without distorting it.
    """
    empty_result = {
        "twr_ltd_pct": 0.0,
        "twr_ytd_pct": None,
        "twr_ytd_available": False,
        "as_of": datetime.now().isoformat(),
    }
    if df is None or df.empty:
        return empty_result

    work = df.copy()
    work["day"] = pd.to_datetime(work["day"], errors="coerce")
    work = work.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    if work.empty:
        return empty_result

    for col in ("beg_nlv", "end_nlv", "cash_change"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
        else:
            work[col] = 0.0

    work["adjusted_beg"] = work["beg_nlv"] + work["cash_change"]
    work["daily_return"] = 0.0
    mask = work["adjusted_beg"] > 0
    work.loc[mask, "daily_return"] = (
        (work.loc[mask, "end_nlv"] - work.loc[mask, "adjusted_beg"])
        / work.loc[mask, "adjusted_beg"]
    )

    ltd_curve = (1.0 + work["daily_return"]).cumprod()
    twr_ltd_pct = float((ltd_curve.iloc[-1] - 1.0) * 100.0)

    this_year = datetime.now().year
    jan1 = pd.Timestamp(year=this_year, month=1, day=1)
    ytd = work[work["day"] >= jan1]
    if ytd.empty:
        twr_ytd_pct: float | None = None
        twr_ytd_available = False
    else:
        ytd_curve = (1.0 + ytd["daily_return"]).cumprod()
        twr_ytd_pct = float((ytd_curve.iloc[-1] - 1.0) * 100.0)
        twr_ytd_available = True

    return {
        "twr_ltd_pct": round(twr_ltd_pct, 4),
        "twr_ytd_pct": round(twr_ytd_pct, 4) if twr_ytd_pct is not None else None,
        "twr_ytd_available": twr_ytd_available,
        "as_of": datetime.now().isoformat(),
    }


def compute_twr_returns(portfolio_name: str) -> dict[str, Any]:
    """Time-weighted LTD + YTD for a portfolio, chained from journal daily returns.

    This is the answer to 'what compound return did the strategy produce,
    independent of when I deposited?'. Unlike compute_returns()'s snapshot
    ratio, it correctly accounts for cash-flow timing.

    Returns the same shape as _compute_twr_from_journal_df. Empty journal
    yields zeros / unavailable YTD rather than an error so the UI can render
    a stable tile state.
    """
    df = db.load_journal(portfolio_name)
    if df is None or df.empty:
        return _compute_twr_from_journal_df(pd.DataFrame())
    return _compute_twr_from_journal_df(normalize_journal_columns(df))
