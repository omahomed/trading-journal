#!/usr/bin/env python3
"""End-of-day journal save via IBKR Client Portal real-time NAV.

Workflow:
    1. Fetch real-time account summary from the local Client Portal Gateway
       (must be running at https://localhost:5050 with an active session).
    2. Pull SPY + ^IXIC closing prices for the target date via yfinance.
    3. Build a journal entry dict and hand it to api.main.journal_edit() —
       reuses the same auto-fill (market_cycle, mct_display_day_num,
       portfolio_heat, spy_atr, nasdaq_atr) the Daily Routine UI uses.
    4. Idempotent: if a journal row already exists for the date, it's
       updated in place (same upsert path as the UI). Existing report-card
       scores, notes, etc. are preserved per journal_edit's merge logic.

This bypasses the need to wait for IBKR Flex Query's T+1 cycle. Designed
to be run as a cron at 4:05 PM CT after market close, or manually from a
terminal whenever the user wants to lock in the day's numbers.

Requires DATABASE_URL in env or .streamlit/secrets.toml so api.main.db
can connect to prod Postgres.

Usage:
    python scripts/eod_journal_save.py
    python scripts/eod_journal_save.py --date 2026-04-29
    python scripts/eod_journal_save.py --portfolio CanSlim --account U21853041
    python scripts/eod_journal_save.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from ibkr_client_portal import (  # noqa: E402
    ClientPortalError,
    fetch_account_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eod_save")

# Default mapping. Today only CanSlim is hooked up; expand the dict as
# additional portfolios get wired to IBKR accounts. Keep keys aligned with
# the journal's `portfolio` column values exactly (the journal_edit endpoint
# resolves them via portfolios.name).
DEFAULT_PORTFOLIO_MAP = {
    "CanSlim": "U21853041",
}


def _today_ct() -> str:
    """Return today's date in Central Time as YYYY-MM-DD.

    Project convention: all journal day-stamps are CT regardless of where
    the script runs. Mac mini is local but launchd jobs may run with
    different TZ depending on user account settings.
    """
    return datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")


def _fetch_close(ticker: str, target_date: str) -> float:
    """Closing price for `ticker` on `target_date` via yfinance.

    Pulls a small ±7 day window so weekend/holiday lookups still land on the
    most recent trading session. Returns 0.0 if no data — caller decides
    whether that's fatal or not.
    """
    import yfinance as yf
    try:
        target = pd.to_datetime(target_date).date()
        start = (pd.Timestamp(target) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(target) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if df.empty:
            return 0.0
        df = df[df.index.date <= target]
        if df.empty:
            return 0.0
        return round(float(df["Close"].iloc[-1]), 2)
    except Exception as e:
        log.warning("yfinance close fetch for %s on %s failed: %s",
                    ticker, target_date, e)
        return 0.0


def _fetch_prev_end_nlv(portfolio: str, target_date: str) -> float:
    """Return the end_nlv of the most recent journal entry strictly before
    target_date. Mirrors what /api/journal/latest?before=… returns for the
    Daily Routine UI. Returns 0.0 if there's no prior entry (brand-new
    portfolio, or first-ever save).
    """
    from db_layer import load_journal
    df = load_journal(portfolio)
    if df is None or df.empty:
        return 0.0
    # load_journal returns columns with Pascal-case aliases ("Day", "End NLV").
    days = pd.to_datetime(df["Day"], errors="coerce")
    cutoff = pd.Timestamp(target_date)
    prior = df[days < cutoff]
    if prior.empty:
        return 0.0
    prior = prior.copy()
    prior["_day"] = pd.to_datetime(prior["Day"], errors="coerce")
    prior = prior.sort_values("_day", ascending=False)
    try:
        return float(prior.iloc[0]["End NLV"] or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_entry(
    portfolio: str, target_date: str, summary: dict,
    spy_close: float, ndx_close: float, prev_end_nlv: float,
) -> dict:
    """Translate Client Portal summary + index closes into a journal_edit
    payload. journal_edit handles the merge with existing values + auto-
    computes market_cycle, mct_display_day_num, ATRs, and portfolio_heat,
    so we only fill the IBKR-derived fields here.

    cash_change is left at 0 — this script doesn't know about deposits or
    withdrawals (those are user-initiated and tracked separately via the
    cash_transactions ledger). User can override via the Daily Routine UI
    if needed.

    Daily change math mirrors the Daily Routine form's computation
    (frontend daily-routine.tsx, ~line 210):
        daily_dollar_change = end_nlv - beg_nlv - cash_change
        daily_pct_change    = daily_dollar_change / (beg_nlv + cash_change)
    where beg_nlv is the prior trading day's end_nlv. Doing it here rather
    than in journal_edit because journal_edit's merge layer was designed
    to receive these from the form, not derive them itself.
    """
    nlv = float(summary["nlv"])
    holdings = float(summary["position_value"])
    pct_invested = round((holdings / nlv) * 100.0, 4) if nlv > 0 else 0.0

    cash_change = 0.0
    beg_nlv = float(prev_end_nlv or 0.0)
    daily_dollar_change = round(nlv - beg_nlv - cash_change, 2) if beg_nlv > 0 else 0.0
    adjusted_beg = beg_nlv + cash_change
    daily_pct_change = (
        round((daily_dollar_change / adjusted_beg) * 100.0, 4)
        if adjusted_beg > 0 else 0.0
    )

    return {
        "portfolio": portfolio,
        "day": target_date,
        "end_nlv": nlv,
        "beg_nlv": beg_nlv,
        # Holdings come through pct_invested; the schema doesn't store
        # "holdings_dollar" directly but reconstructs it from pct_invested
        # × end_nlv at read time.
        "pct_invested": pct_invested,
        "cash_change": cash_change,
        "daily_dollar_change": daily_dollar_change,
        "daily_pct_change": daily_pct_change,
        "spy": spy_close,
        "nasdaq": ndx_close,
        # Provenance flags so the dashboard can show "auto-pulled from IBKR"
        # vs. user-typed. Mirrors what the Daily Routine UI sends when the
        # IBKR Flex auto-fill succeeds.
        "nlv_source": "ibkr_auto",
        "holdings_source": "ibkr_auto",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None,
                    help="Target date YYYY-MM-DD (default: today in Central Time)")
    ap.add_argument("--portfolio", default="CanSlim",
                    help="Portfolio name in trading_journal (default: CanSlim)")
    ap.add_argument("--account-id", default=None,
                    help="IBKR account ID. Defaults from the built-in portfolio map.")
    ap.add_argument("--gateway-url", default="https://localhost:5050",
                    help="Client Portal gateway URL (default: localhost:5050)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the payload that would be saved, but don't write to DB.")
    args = ap.parse_args()

    target_date = args.date or _today_ct()
    portfolio = args.portfolio
    account_id = args.account_id or DEFAULT_PORTFOLIO_MAP.get(portfolio)
    if not account_id:
        log.error("No IBKR account ID for portfolio %r. Pass --account-id "
                  "or add the mapping to DEFAULT_PORTFOLIO_MAP.", portfolio)
        return 2

    log.info("Fetching IBKR summary for %s (account %s)…", portfolio, account_id)
    try:
        summary = fetch_account_summary(account_id, gateway_url=args.gateway_url)
    except ClientPortalError as e:
        log.error("Client Portal fetch failed: %s", e)
        log.error("Is the gateway running? Did you log in via the browser?")
        return 1

    log.info("  NLV=%s  cash=%s  positions=%s",
             summary["nlv"], summary["cash"], summary["position_value"])

    log.info("Pulling SPY + ^IXIC closes for %s…", target_date)
    spy_close = _fetch_close("SPY", target_date)
    ndx_close = _fetch_close("^IXIC", target_date)
    log.info("  SPY=%s  ^IXIC=%s", spy_close, ndx_close)

    # Tenancy: this script runs outside the FastAPI auth middleware, so the
    # current_user_id ContextVar is unset by default. db_layer.get_db_connection
    # only sets `app.user_id` on the session when the ContextVar is populated,
    # and trading_journal's user_id column has no DB-level default — INSERTs
    # would violate the NOT NULL constraint. We set the founder UUID
    # explicitly here (matches the value all pre-auth backfilled rows carry,
    # per project_founder_uuid memory) so every DB op in this run lands in
    # the right tenant. Set BEFORE the load_journal lookup, not just before
    # the save, so RLS doesn't return empty for the prior-day baseline.
    import db_layer
    FOUNDER_UUID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"
    ctx_token = db_layer.current_user_id.set(FOUNDER_UUID)
    try:
        log.info("Looking up prior trading day's end_nlv (for daily %% baseline)…")
        prev_end_nlv = _fetch_prev_end_nlv(portfolio, target_date)
        log.info("  prev end_nlv=%s", prev_end_nlv)

        entry = _build_entry(
            portfolio, target_date, summary, spy_close, ndx_close, prev_end_nlv,
        )

        if args.dry_run:
            log.warning("Dry-run — no DB write. Payload:")
            for k, v in entry.items():
                log.info("  %s = %r", k, v)
            return 0

        log.info("Saving journal entry via api.main.journal_edit…")
        from api.main import journal_edit
        result = journal_edit(entry)
    finally:
        db_layer.current_user_id.reset(ctx_token)

    if result.get("status") == "ok":
        log.info("Saved (journal id=%s)", result.get("id"))
        return 0
    log.error("Save failed: %s", result)
    return 1


if __name__ == "__main__":
    sys.exit(main())
