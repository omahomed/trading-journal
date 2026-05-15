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


def dashboard_metrics(portfolio_id: int, portfolio_name: str) -> dict[str, Any]:
    """Aggregated read view for the dashboard. Single source of truth: every
    journal-derived field comes from the latest saved trading_journal row.

    Architecturally this is the user-facing flip from "live computed NLV
    drives the dashboard" to "the broker-pulled, EOD-saved journal value
    drives the dashboard". Live yfinance prices no longer feed any field
    on this response — exposure, drawdown, position sizing, and risk math
    all read from journal.end_nlv.

    Two classes of field in the response:

    1. Journal-derived (always present when `journal_available: true`):
       nlv, total_holdings, exposure_pct, cash, drawdown_*, ltd_pct, ytd_pct,
       nlv_delta_*, as_of_date.

    2. State flag: journal_available.

    Total Holdings is computed on read as `pct_invested × end_nlv / 100`.
    There's no `total_holdings` column; pct_invested is NUMERIC(10,4),
    which round-trips real prod values to within $0.01 (verified for
    end_nlv=486630.39 / pct_invested=188.5413 → total_holdings=917498.79).
    """
    now_iso = datetime.now().isoformat()
    empty: dict[str, Any] = {
        "journal_available": False,
        "as_of_date": None,
        "nlv": None,
        "nlv_delta_dollar": None,
        "nlv_delta_pct": None,
        "total_holdings": None,
        "exposure_pct": None,
        "cash": None,
        "drawdown_current_pct": None,
        "drawdown_peak_nlv": None,
        "drawdown_peak_date": None,
        "ltd_pct": None,
        "ltd_pl_dollar": None,
        "ytd_pct": None,
        "ytd_pl_dollar": None,
        "ytd_available": False,
        "as_of": now_iso,
    }

    journal_df = db.load_journal(portfolio_name)
    if journal_df is None or journal_df.empty:
        return empty

    work = normalize_journal_columns(journal_df).copy()
    work["day"] = pd.to_datetime(work["day"], errors="coerce")
    work = work.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    if work.empty:
        return empty

    for col in ("end_nlv", "pct_invested", "daily_dollar_change", "daily_pct_change"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
        else:
            work[col] = 0.0

    latest = work.iloc[-1]
    journal_nlv = float(latest["end_nlv"])
    pct_invested = float(latest["pct_invested"])
    total_holdings = round(journal_nlv * pct_invested / 100.0, 2)
    cash = round(journal_nlv - total_holdings, 2)

    # Deltas: only meaningful from the second journal entry onward. The
    # frontend renders no subtext for the first entry rather than showing
    # "+$0.00 (+0.00%)" which would imply no movement when really we just
    # have nothing to compare against.
    if len(work) >= 2:
        nlv_delta_dollar: float | None = round(float(latest["daily_dollar_change"]), 2)
        nlv_delta_pct: float | None = round(float(latest["daily_pct_change"]), 4)
    else:
        nlv_delta_dollar = None
        nlv_delta_pct = None

    # Drawdown: peak across the entire journal window, current = latest NLV.
    # Uses argmax-style "first occurrence of the running max" so if the
    # peak is hit twice we report the earlier date (more conservative —
    # makes "days since peak" longer).
    end_nlvs = work["end_nlv"]
    peak_nlv = float(end_nlvs.max())
    peak_idx = int(end_nlvs.idxmax())
    peak_row = work.iloc[peak_idx]
    peak_date = peak_row["day"].date().isoformat() if pd.notna(peak_row["day"]) else None
    drawdown_current_pct = round(((journal_nlv - peak_nlv) / peak_nlv) * 100.0, 4) if peak_nlv > 0 else 0.0

    # TWR LTD/YTD straight from the same journal frame (no second DB load).
    twr = _compute_twr_from_journal_df(work)

    # Dollar P&L versions of LTD/YTD — distinct from the TWR percent.
    # LTD uses the cash_transactions ledger so it agrees with what the
    # snapshot-style /returns endpoint reports. YTD reads the journal's
    # year boundary directly (no equivalent ledger view yet).
    ltd_pl_dollar = _compute_ltd_pl_dollar(portfolio_id, journal_nlv)
    ytd_pl_dollar = _compute_ytd_pl_dollar(work, journal_nlv) if twr["twr_ytd_available"] else None

    as_of_date = latest["day"].date().isoformat() if pd.notna(latest["day"]) else None

    return {
        "journal_available": True,
        "as_of_date": as_of_date,
        "nlv": round(journal_nlv, 2),
        "nlv_delta_dollar": nlv_delta_dollar,
        "nlv_delta_pct": nlv_delta_pct,
        "total_holdings": total_holdings,
        "exposure_pct": round(pct_invested, 4),
        "cash": cash,
        "drawdown_current_pct": drawdown_current_pct,
        "drawdown_peak_nlv": round(peak_nlv, 2),
        "drawdown_peak_date": peak_date,
        "ltd_pct": twr["twr_ltd_pct"],
        "ltd_pl_dollar": ltd_pl_dollar,
        "ytd_pct": twr["twr_ytd_pct"],
        "ytd_pl_dollar": ytd_pl_dollar,
        "ytd_available": twr["twr_ytd_available"],
        "as_of": now_iso,
    }


def _compute_ltd_pl_dollar(portfolio_id: int, journal_nlv: float) -> float | None:
    """Dollar version of LTD return: journal NLV minus net contributions.

    Distinct from `ltd_pct` (TWR — accounts for cash-flow timing). The
    dollar answer is the simpler "how much did I make vs. what I put in"
    snapshot. Uses cash_transactions ledger so the result agrees with
    /returns. Returns None if the ledger lookup fails — the dashboard
    renders a fallback sub-label in that case.
    """
    try:
        net_contrib = db.get_net_contributions(portfolio_id)
        return round(journal_nlv - float(net_contrib), 2)
    except Exception:
        return None


def _compute_ytd_pl_dollar(journal_df: pd.DataFrame, journal_nlv: float) -> float | None:
    """Dollar version of YTD return: current NLV − YTD baseline − YTD cash flows.

    Baseline rule:
      1. Last journal entry of prior year → its end_nlv (preferred —
         that's the broker-confirmed value at year-end).
      2. Else first entry of current year → its beg_nlv (the routine
         records yesterday's close as today's beg, so this is the
         year-start baseline for portfolios that opened mid-year).
      3. Else None — can't anchor YTD without a baseline.

    Cash flows: sum of cash_change across current-year journal rows.
    Cash_change is the user-entered Cash +/- on Daily Routine, capturing
    deposits / withdrawals between days.
    """
    this_year = datetime.now().year
    year_mask = journal_df["day"].dt.year == this_year
    year_rows = journal_df[year_mask]
    if year_rows.empty:
        return None

    prior_rows = journal_df[journal_df["day"].dt.year < this_year]
    if not prior_rows.empty:
        baseline = float(prior_rows.iloc[-1]["end_nlv"])
    else:
        first_year_row = year_rows.iloc[0]
        baseline = float(first_year_row.get("beg_nlv") or first_year_row.get("end_nlv") or 0.0)

    if baseline <= 0:
        return None

    cash_flows = float(year_rows.get("cash_change", pd.Series(dtype=float)).sum())
    return round(journal_nlv - baseline - cash_flows, 2)


def _twr_chained_pct(daily_returns: pd.Series) -> float:
    """Chain daily returns: (∏(1 + r) − 1) × 100. Empty → 0.0."""
    if daily_returns is None or daily_returns.empty:
        return 0.0
    curve = (1.0 + daily_returns).cumprod()
    return float((curve.iloc[-1] - 1.0) * 100.0)


def _prepare_journal_for_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort-by-day + per-day TWR return column. Mirrors the same daily-Dietz
    formula used by /api/journal/history and _compute_twr_from_journal_df:

        adjusted_beg = beg_nlv + cash_change
        daily_return = (end_nlv − adjusted_beg) / adjusted_beg  if adjusted_beg > 0 else 0

    Returns a *copy* with day coerced to datetime, sorted ascending, with
    beg_nlv / end_nlv / cash_change / daily_return all present and numeric.
    Empty input returns an empty DataFrame with the same columns.
    """
    cols = ["day", "beg_nlv", "end_nlv", "cash_change", "daily_return"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    work = df.copy()
    work["day"] = pd.to_datetime(work["day"], errors="coerce")
    work = work.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=cols)

    for col in ("beg_nlv", "end_nlv", "cash_change"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
        else:
            work[col] = 0.0

    adjusted_beg = work["beg_nlv"] + work["cash_change"]
    work["daily_return"] = 0.0
    mask = adjusted_beg > 0
    work.loc[mask, "daily_return"] = (
        (work.loc[mask, "end_nlv"] - adjusted_beg[mask]) / adjusted_beg[mask]
    )
    return work


def _parse_week_start(week_start: str) -> date | None:
    """Best-effort YYYY-MM-DD parse → date. Returns None on garbage input."""
    try:
        return datetime.strptime(str(week_start).strip()[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError, AttributeError):
        return None


def _compute_win_rate_ytd(summary_df: pd.DataFrame, week_end: date) -> dict[str, Any]:
    """YTD win-rate from trades_summary.

    Win Rate = wins / (wins + losses + flat). The flat-in-denominator
    convention matches the existing `computeWinRate` in
    frontend/src/lib/analytics-stats.ts and Phase 5's locked spec.

    Source: trades_summary.realized_pl (campaign-level — one row per
    campaign, scale-outs roll up). Status filter: 'CLOSED'. Date filter:
    Jan 1 of week_end's year ≤ closed_date ≤ week_end.

    Empty result → all zeros with rate=0 (caller renders "—" subtitle).
    """
    empty = {"rate": 0.0, "wins": 0, "losses": 0, "flat": 0, "total": 0}
    if summary_df is None or summary_df.empty:
        return empty

    status_col = "Status" if "Status" in summary_df.columns else (
        "status" if "status" in summary_df.columns else None
    )
    closed_col = "Closed_Date" if "Closed_Date" in summary_df.columns else (
        "closed_date" if "closed_date" in summary_df.columns else None
    )
    pl_col = "Realized_PL" if "Realized_PL" in summary_df.columns else (
        "realized_pl" if "realized_pl" in summary_df.columns else None
    )
    if not (status_col and closed_col and pl_col):
        return empty

    work = summary_df.copy()
    work[status_col] = work[status_col].astype(str).str.upper().str.strip()
    work = work[work[status_col] == "CLOSED"]
    if work.empty:
        return empty

    work[closed_col] = pd.to_datetime(work[closed_col], errors="coerce")
    work = work.dropna(subset=[closed_col])
    jan1 = pd.Timestamp(year=week_end.year, month=1, day=1)
    end_ts = pd.Timestamp(week_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    work = work[(work[closed_col] >= jan1) & (work[closed_col] <= end_ts)]
    if work.empty:
        return empty

    pl = pd.to_numeric(work[pl_col], errors="coerce").fillna(0.0)
    wins = int((pl > 0).sum())
    losses = int((pl < 0).sum())
    flat = int((pl == 0).sum())
    total = wins + losses + flat
    rate = (wins / total) if total > 0 else 0.0
    return {"rate": round(rate, 4), "wins": wins, "losses": losses, "flat": flat, "total": total}


def weekly_metrics(portfolio_name: str, week_start: str) -> dict[str, Any]:
    """Performance metrics for the Weekly Retro top-tile row.

    Reuses the same daily-Dietz TWR math as Period Review and
    `_compute_twr_from_journal_df` — no new formula in this function. The
    weekly aggregation mirrors Period Review's `aggregatePeriods`:

        weekStartNLV = first in-week row's beg_nlv
        weekEndNLV   = last  in-week row's end_nlv
        weekCashFlow = Σ in-week cash_change
        weekly_pnl   = weekEndNLV − (weekStartNLV + weekCashFlow)
        weekly_return_pct = (∏(1 + daily_return) − 1) × 100  over in-week rows

    LTD / YTD are chained from the same daily_return series, anchored to
    inception (first journal row) and Jan 1 of week_end's year. All metrics
    are computed *as of* week_end so historical weeks are stable.

    Week window:
        Mon (week_start, user-supplied) … Sunday (week_start + 6 days).
        week_end echoed in the response = Friday (week_start + 4).

    Edge cases:
      - No journal rows in the requested week → weekly_pnl = 0,
        weekly_return_pct = 0. LTD/YTD still compute from history.
      - Week_start unparseable → returns an error dict.
      - Account inception inside the requested week → LTD == YTD ==
        weekly_return_pct (cumprod is over the same in-week rows for all
        three) — naturally falls out of the math, no special-case needed.
      - No prior-year rows when computing YTD → YTD == LTD (all rows are
        current-year, so the cumprods agree).
      - Win Rate with 0 trades YTD → rate=0, total=0.
    """
    week_start_date = _parse_week_start(week_start)
    if week_start_date is None:
        return {"error": "Invalid week_start (expected YYYY-MM-DD)"}

    week_end_date = week_start_date + pd.Timedelta(days=4).to_pytimedelta()
    week_window_end = week_start_date + pd.Timedelta(days=6).to_pytimedelta()

    journal_df = db.load_journal(portfolio_name)
    work = _prepare_journal_for_returns(
        normalize_journal_columns(journal_df) if journal_df is not None and not journal_df.empty
        else pd.DataFrame()
    )

    # As-of cutoff: only rows with day <= end of the week window. Keeps
    # historical weeks stable even after later journal rows are added.
    cutoff = pd.Timestamp(week_window_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    work = work[work["day"] <= cutoff]

    # Weekly slice: rows within [Monday, Sunday].
    week_start_ts = pd.Timestamp(week_start_date)
    in_week_mask = (work["day"] >= week_start_ts) & (work["day"] <= cutoff)
    week_rows = work[in_week_mask]

    if week_rows.empty:
        weekly_pnl = 0.0
        weekly_return_pct = 0.0
    else:
        beg_nlv = float(week_rows.iloc[0]["beg_nlv"])
        end_nlv = float(week_rows.iloc[-1]["end_nlv"])
        cash_flow = float(week_rows["cash_change"].sum())
        weekly_pnl = round(end_nlv - (beg_nlv + cash_flow), 2)
        weekly_return_pct = round(_twr_chained_pct(week_rows["daily_return"]), 4)

    # LTD: all rows up through week_end. YTD: rows in week_end.year through
    # week_end. Both use the same chained daily_return so they agree by
    # construction with Period Review's "LTD Return %" column.
    if work.empty:
        ltd_pct = 0.0
        ytd_pct = 0.0
    else:
        ltd_pct = round(_twr_chained_pct(work["daily_return"]), 4)
        jan1 = pd.Timestamp(year=week_end_date.year, month=1, day=1)
        ytd_rows = work[work["day"] >= jan1]
        if ytd_rows.empty:
            # No current-year rows up to week_end → fall back to LTD per
            # Phase 5 spec ("ytd_pct uses inception as fallback").
            ytd_pct = ltd_pct
        else:
            ytd_pct = round(_twr_chained_pct(ytd_rows["daily_return"]), 4)

    summary_df = db.load_summary(portfolio_name)
    win_rate = _compute_win_rate_ytd(summary_df, week_end_date)

    return {
        "weekly_pnl": weekly_pnl,
        "weekly_return_pct": weekly_return_pct,
        "ytd_pct": ytd_pct,
        "ltd_pct": ltd_pct,
        "win_rate": win_rate,
        "week_start": week_start_date.isoformat(),
        "week_end": week_end_date.isoformat(),
        "as_of": datetime.now().isoformat(),
    }
