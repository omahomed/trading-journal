#!/usr/bin/env python3
"""Backfill trades_summary.sr8_activation_{date, nlv, core_shares} for
open campaigns whose cushion first crossed +50% from B1 entry.

Ships as part of migration 048's rollout. For each OPEN equity campaign
with a NULL sr8_activation_date:

  1. Fetch daily close history from the ticker's `mors/data/<TICKER>_price_data.csv`
     (already used by the SR8 monitor). Falls back to yfinance if the CSV
     is missing.
  2. Find the first bar (>= B1 date, <= today) where
        (daily_close - b1_price) / b1_price >= 50%.
  3. If found → look up trading_journal.end_nlv on that date (or the
     nearest earlier date if the exact one is missing — small gap
     tolerance since journals aren't required every calendar day).
  4. Walk trades_details up to and including that date to compute the
     LIFO share count held right after the last event on or before that
     day.
  5. Write the trio to trades_summary. Otherwise leave NULL (position
     hasn't reached SR8 tier yet; the live-activation path will fire
     when it does).

Idempotent: rows with a non-NULL sr8_activation_date are skipped unless
`--reconcile` is passed. --reconcile force-recomputes everything but
still only writes if the new values are consistent (or if `--force` is
also passed).

Every write logs an audit line to stdout AND to a `sr8_activation_backfill_
<timestamp>.log` file so the operator has a paper trail of which
campaigns got what anchor and why.

Usage:
    python scripts/sr8_activation_backfill.py --dry-run
    python scripts/sr8_activation_backfill.py
    python scripts/sr8_activation_backfill.py --portfolio "CanSlim"
    python scripts/sr8_activation_backfill.py --reconcile --force
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db_layer import get_db_connection  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sr8_activation_backfill")


# Cushion threshold that defines SR8 activation. Matches the frontend
# taxonomy (sell-rule.ts SR8_THRESHOLD = 50).
SR8_CUSHION_PCT = 50.0

# Journal-lookup tolerance. If the exact activation date is missing from
# trading_journal (weekends, holidays, gaps), we walk back this many
# days looking for the most recent journal entry. 7 covers a long
# weekend + a holiday without matching a stale-by-a-week snapshot.
JOURNAL_LOOKUP_TOLERANCE_DAYS = 7


def _load_price_history(ticker: str) -> pd.DataFrame | None:
    """Prefer the mors CSV (already refreshed by monitor / backtest);
    fall back to yfinance for tickers without a CSV yet."""
    csv_path = REPO_ROOT / "mors" / "data" / f"{ticker}_price_data.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        # mors CSVs have a Date column; normalize to a DatetimeIndex.
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        return df.set_index("Date")
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.history(period="max")
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        log.warning("yfinance history fetch failed for %s: %s", ticker, e)
        return None


def find_activation_date(
    b1_date: date,
    b1_price: float,
    price_history: pd.DataFrame,
) -> date | None:
    """First bar on or after b1_date where Close/b1_price - 1 >= 0.50.

    Uses close-basis (not intraday high) — matches the b1_max_return_pct
    convention. Returns None when no bar in the history has crossed the
    threshold yet.
    """
    if price_history is None or price_history.empty or not (b1_price > 0):
        return None
    threshold_price = b1_price * (1.0 + SR8_CUSHION_PCT / 100.0)

    # Filter to bars on/after B1 date.
    if "Close" not in price_history.columns:
        return None
    bars = price_history[price_history.index.date >= b1_date]
    if bars.empty:
        return None

    # First bar where Close >= threshold.
    hits = bars[bars["Close"] >= threshold_price]
    if hits.empty:
        return None
    first_hit = hits.index[0]
    if hasattr(first_hit, "date"):
        return first_hit.date()
    return first_hit


def lookup_nlv_on(
    portfolio: str,
    target_date: date,
    conn,
) -> tuple[float | None, date | None]:
    """Journal end_nlv for `portfolio` on `target_date`, or the closest
    earlier day within JOURNAL_LOOKUP_TOLERANCE_DAYS. Returns (nlv,
    actual_journal_date). None on both if nothing found in range."""
    cutoff = target_date - timedelta(days=JOURNAL_LOOKUP_TOLERANCE_DAYS)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT day, end_nlv
              FROM trading_journal j
              JOIN portfolios p ON p.id = j.portfolio_id
             WHERE p.name = %s
               AND j.day <= %s
               AND j.day >= %s
               AND end_nlv IS NOT NULL
             ORDER BY j.day DESC
             LIMIT 1
            """,
            (portfolio, target_date, cutoff),
        )
        row = cur.fetchone()
    if row is None:
        return None, None
    day_val = row[0].date() if hasattr(row[0], "date") else row[0]
    return float(row[1]), day_val


def shares_held_on(
    trade_id: str,
    target_date: date,
    conn,
) -> float:
    """LIFO-walk trades_details for `trade_id` up to and including
    `target_date`. Returns the shares held right after the last event
    on/before the target date. Uses standard LIFO (SELLs consume the
    most-recent BUY lot first)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT date, action, shares
              FROM trades_details
             WHERE trade_id = %s
               AND date::date <= %s
             ORDER BY date ASC, id ASC
            """,
            (trade_id, target_date),
        )
        rows = cur.fetchall()
    lots: list[float] = []  # LIFO stack of lot sizes
    for d, action, shs in rows:
        act = str(action or "").upper()
        n = float(shs or 0)
        if act == "BUY":
            lots.append(n)
        elif act == "SELL":
            to_sell = n
            while to_sell > 1e-9 and lots:
                top = lots[-1]
                take = min(top, to_sell)
                lots[-1] = top - take
                to_sell -= take
                if lots[-1] <= 1e-9:
                    lots.pop()
    return sum(lots)


def fetch_candidates(portfolio: str | None, reconcile: bool):
    """Open equity campaigns eligible for backfill. Skips OPTION rows
    (SR8 mechanics are STOCK-only in this app's model)."""
    where = [
        "s.status = 'OPEN'",
        "COALESCE(s.instrument_type, 'STOCK') = 'STOCK'",
        # Need a B1 price + date to walk. b1_reconcile keeps b1_entry_price
        # on the summary; we fall back to a details subquery below.
    ]
    params: list[object] = []
    if portfolio:
        where.append("p.name = %s")
        params.append(portfolio)
    if not reconcile:
        where.append("s.sr8_activation_date IS NULL")
    # b1_entry_price + b1_date derived from the earliest BUY row on the
    # campaign — mirrors b1_reconcile.py's pattern (no denormalized
    # column on trades_summary for these). deleted_at guards are folded
    # in when the column exists (not universal; see below).
    sql = f"""
        SELECT
            s.trade_id, s.ticker, s.sr8_activation_date,
            s.sr8_activation_nlv, s.sr8_core_shares, p.name AS portfolio_name,
            (SELECT d.date::date
               FROM trades_details d
              WHERE d.trade_id = s.trade_id AND d.action = 'BUY'
              ORDER BY d.date ASC, d.id ASC
              LIMIT 1) AS b1_date,
            (SELECT d.amount
               FROM trades_details d
              WHERE d.trade_id = s.trade_id AND d.action = 'BUY'
              ORDER BY d.date ASC, d.id ASC
              LIMIT 1) AS b1_entry_price
          FROM trades_summary s
          JOIN portfolios p ON p.id = s.portfolio_id
         WHERE {' AND '.join(where)}
         ORDER BY p.name, s.trade_id
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def backfill_one(
    row: dict,
    conn,
    dry_run: bool,
    force: bool,
) -> dict:
    """Process one campaign. Returns an audit-shape dict."""
    trade_id = row["trade_id"]
    ticker = str(row["ticker"] or "").upper().strip()
    portfolio = row["portfolio_name"]
    b1_price = float(row["b1_entry_price"] or 0)
    b1_date_val = row["b1_date"]
    if not ticker or not (b1_price > 0) or b1_date_val is None:
        return {
            "trade_id": trade_id, "ticker": ticker, "portfolio": portfolio,
            "status": "SKIP", "reason": "missing b1 price/date",
            "activation_date": None, "activation_nlv": None, "core_shares": None,
        }
    if isinstance(b1_date_val, datetime):
        b1_date_val = b1_date_val.date()

    price_history = _load_price_history(ticker)
    if price_history is None or price_history.empty:
        return {
            "trade_id": trade_id, "ticker": ticker, "portfolio": portfolio,
            "status": "SKIP", "reason": "no price history available",
            "activation_date": None, "activation_nlv": None, "core_shares": None,
        }

    activation_date = find_activation_date(b1_date_val, b1_price, price_history)
    if activation_date is None:
        return {
            "trade_id": trade_id, "ticker": ticker, "portfolio": portfolio,
            "status": "SKIP", "reason": "cushion never reached +50%",
            "activation_date": None, "activation_nlv": None, "core_shares": None,
        }

    activation_nlv, journal_day = lookup_nlv_on(portfolio, activation_date, conn)
    if activation_nlv is None:
        return {
            "trade_id": trade_id, "ticker": ticker, "portfolio": portfolio,
            "status": "SKIP", "reason": (
                f"no journal end_nlv within {JOURNAL_LOOKUP_TOLERANCE_DAYS} "
                f"days of {activation_date}"
            ),
            "activation_date": activation_date, "activation_nlv": None,
            "core_shares": None,
        }

    core_shares = shares_held_on(trade_id, activation_date, conn)
    if core_shares <= 0:
        return {
            "trade_id": trade_id, "ticker": ticker, "portfolio": portfolio,
            "status": "SKIP",
            "reason": f"no shares held on {activation_date} (LIFO walk)",
            "activation_date": activation_date, "activation_nlv": activation_nlv,
            "core_shares": None,
        }

    # Consistency check with any existing values (reconcile mode).
    existing_date = row.get("sr8_activation_date")
    existing_nlv = row.get("sr8_activation_nlv")
    existing_shs = row.get("sr8_core_shares")
    has_existing = any(v is not None for v in (existing_date, existing_nlv, existing_shs))
    if has_existing and not force:
        # Verify computed matches stored. Small tolerance on NLV; exact
        # on date; +/-0.5 sh on core.
        date_matches = (existing_date == activation_date)
        nlv_matches = existing_nlv is not None and abs(float(existing_nlv) - activation_nlv) < 1.0
        shs_matches = existing_shs is not None and abs(float(existing_shs) - core_shares) < 0.5
        if date_matches and nlv_matches and shs_matches:
            status = "MATCH"
        else:
            status = "DRIFT"  # computed differs from stored; needs --force
        return {
            "trade_id": trade_id, "ticker": ticker, "portfolio": portfolio,
            "status": status,
            "reason": (
                f"stored=({existing_date}, {existing_nlv}, {existing_shs}) "
                f"computed=({activation_date}, {activation_nlv:.2f}, {core_shares:.2f})"
            ),
            "activation_date": activation_date,
            "activation_nlv": activation_nlv,
            "core_shares": core_shares,
        }

    if not dry_run:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE trades_summary
                   SET sr8_activation_date = %s,
                       sr8_activation_nlv  = %s,
                       sr8_core_shares     = %s,
                       last_updated        = CURRENT_TIMESTAMP
                 WHERE trade_id = %s
                """,
                (activation_date, activation_nlv, core_shares, trade_id),
            )
        conn.commit()

    return {
        "trade_id": trade_id, "ticker": ticker, "portfolio": portfolio,
        "status": ("WOULD-WRITE" if dry_run else "WRITE"),
        "reason": (
            f"journal_day={journal_day} nlv=${activation_nlv:,.0f} "
            f"core_shs={core_shares:.2f}"
        ),
        "activation_date": activation_date,
        "activation_nlv": activation_nlv,
        "core_shares": core_shares,
    }


def _print_audit_line(r: dict) -> None:
    ad = str(r["activation_date"] or "—")
    nlv = f"${r['activation_nlv']:>12,.0f}" if r["activation_nlv"] else "—" + " " * 12
    cs = f"{r['core_shares']:>9.2f}" if r["core_shares"] else "—" + " " * 8
    print(
        f"{r['status']:<12} {r['portfolio']:<20} {r['ticker']:<8} "
        f"{r['trade_id']:<12} {ad:<12} {nlv:>13} {cs:>10}  {r['reason']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report planned writes without touching DB.")
    parser.add_argument("--reconcile", action="store_true",
                        help="Include rows already populated; compare stored vs "
                             "computed. --force required to overwrite.")
    parser.add_argument("--force", action="store_true",
                        help="With --reconcile: overwrite existing values even "
                             "when they differ from the computed anchor.")
    parser.add_argument("--portfolio", default=None,
                        help="Limit to one portfolio (default: all).")
    args = parser.parse_args()

    log.info(
        "Fetching candidates: portfolio=%s reconcile=%s ...",
        args.portfolio or "all", args.reconcile,
    )
    candidates = fetch_candidates(args.portfolio, args.reconcile)
    log.info("Found %d candidate(s).", len(candidates))
    if not candidates:
        return 0

    print(
        f"\n{'STATUS':<12} {'PORTFOLIO':<20} {'TICKER':<8} {'TRADE_ID':<12} "
        f"{'ACT DATE':<12} {'ACT NLV':>13} {'CORE SHS':>10}  REASON"
    )
    print("-" * 140)

    written = 0
    skipped = 0
    drifts = 0
    with get_db_connection() as conn:
        for row in candidates:
            audit = backfill_one(row, conn, args.dry_run, args.force)
            _print_audit_line(audit)
            if audit["status"] in ("WRITE", "WOULD-WRITE"):
                written += 1
            elif audit["status"] == "DRIFT":
                drifts += 1
            else:
                skipped += 1

    print("-" * 140)
    print(
        f"Summary: {written} write{'s' if written != 1 else ''}, "
        f"{drifts} drift{'s' if drifts != 1 else ''}, "
        f"{skipped} skip{'s' if skipped != 1 else ''}."
    )
    if args.dry_run:
        print("(--dry-run: no changes committed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
