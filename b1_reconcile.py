"""Reconcile trades_summary.b1_max_return_pct — the persistent Sell Rule tier.

The Sell Rule tier is `max(current_b1_return, stored b1_max_return_pct)`
(migration 036). The stored peak is promoted live by the frontend, but only
while the app is open. A leader that peaks above 50% while the app is closed
keeps a stale/NULL stored peak and mis-tiers as SR11 on the pullback even
though its close-basis B1 return topped 50%. SR11 (BE stop-out) and SR8
(RS-defended core) prescribe OPPOSITE sell actions, so that mis-tier can
drive a wrong exit.

This module recomputes the close-basis peak B1 return for open equity
positions from yfinance daily closes and raises any stored peak that has
fallen behind. Writes go through db.update_b1_max_return_pct, whose SQL guard
only ever RAISES the stored peak — promote to SR8, never demote a real core
out of it. Idempotent and safe to re-run.

Consumers:
  - scripts/backfill_b1_max_returns.py — CLI wrapper (manual / ad-hoc runs).
  - api/main.py — daily in-process background task on Railway, so the tier
    self-heals without any manual step. Uses the DATABASE_URL already present
    in the API's environment.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

from db_layer import get_db_connection, update_b1_max_return_pct, snapshot_sr8_activation_if_null

log = logging.getLogger("b1_reconcile")

# Tier ladder mirror — kept in sync with frontend/src/lib/sell-rule.ts. The
# backend is NOT the source of truth for classification; this is diagnostic
# labeling only (the frontend classifies from the stored peak this job writes).
SR8_THRESHOLD = 50.0
SR1_THRESHOLD = 10.0


def _latest_portfolio_nlv(portfolio_name: str) -> float | None:
    """Latest trading_journal.end_nlv for `portfolio_name`. Used by the
    SR8 activation snapshot to record today's NLV as the anchor when
    b1_max first crosses +50%. Falls back to None on missing data —
    the caller then skips the snapshot rather than write a bad value."""
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT j.end_nlv
                  FROM trading_journal j
                  JOIN portfolios p ON p.id = j.portfolio_id
                 WHERE p.name = %s AND j.end_nlv IS NOT NULL
                 ORDER BY j.day DESC
                 LIMIT 1
                """,
                (portfolio_name,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return float(row[0])
    except Exception as exc:
        log.warning("Latest NLV lookup failed for %s: %s", portfolio_name, exc)
    return None


def fetch_candidates(portfolio: str | None = None, reconcile: bool = False) -> list[dict]:
    """Find OPEN equity campaigns to backfill / reconcile.

    reconcile=False (default): only rows with b1_max_return_pct IS NULL.
    reconcile=True: every open equity row, so a stored peak that has fallen
    behind the true close-basis peak can be raised. The monotonic guard in
    update_b1_max_return_pct makes already-current rows no-ops.

    Returns rows shaped:
        { trade_id, ticker, portfolio_name, stored_max_pct,
          b1_entry_date, b1_entry_price }
    Skips campaigns missing a B1 BUY row (data corruption / pre-app imports).
    """
    sql = """
        SELECT
            s.trade_id,
            s.ticker,
            p.name AS portfolio_name,
            s.b1_max_return_pct AS stored_max_pct,
            (SELECT d.date
             FROM trades_details d
             WHERE d.trade_id = s.trade_id
               AND d.portfolio_id = s.portfolio_id
               AND d.action = 'BUY'
               AND d.deleted_at IS NULL
             ORDER BY d.date ASC, d.id ASC
             LIMIT 1) AS b1_entry_date,
            (SELECT d.amount
             FROM trades_details d
             WHERE d.trade_id = s.trade_id
               AND d.portfolio_id = s.portfolio_id
               AND d.action = 'BUY'
               AND d.deleted_at IS NULL
             ORDER BY d.date ASC, d.id ASC
             LIMIT 1) AS b1_entry_price
        FROM trades_summary s
        JOIN portfolios p ON s.portfolio_id = p.id
        WHERE s.status = 'OPEN'
          AND s.deleted_at IS NULL
          AND COALESCE(s.instrument_type, 'STOCK') = 'STOCK'
    """
    if not reconcile:
        sql += " AND s.b1_max_return_pct IS NULL"
    params: list = []
    if portfolio:
        sql += " AND p.name = %s"
        params.append(portfolio)
    sql += " ORDER BY p.name, s.trade_id"

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return rows


def compute_max_b1_return(ticker: str, entry_date: date, entry_price: float) -> float | None:
    """Pull daily closes for [entry_date, today] and return the peak B1
    return %. Close-basis by design. Returns None when yfinance has no data.
    """
    end = date.today() + timedelta(days=1)  # yfinance end is exclusive
    raw = yf.download(
        ticker,
        start=entry_date.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if raw is None or raw.empty:
        return None
    closes = raw["Close"]
    # yfinance can return a DataFrame (multi-ticker call) or a Series.
    # Normalize to a 1-D series.
    if isinstance(closes, pd.DataFrame):
        if closes.shape[1] != 1:
            return None
        closes = closes.iloc[:, 0]
    peak_close = float(closes.max())
    if not pd.notna(peak_close) or entry_price <= 0:
        return None
    return ((peak_close - entry_price) / entry_price) * 100.0


def classify_tier(pct: float) -> str:
    """Mirror frontend/src/lib/sell-rule.ts. Diagnostic label only."""
    if pct < SR1_THRESHOLD:
        return "SR1"
    if pct < SR8_THRESHOLD:
        return "SR11"
    return "SR8"


def reconcile_open_positions(
    portfolio: str | None = None,
    reconcile: bool = True,
    dry_run: bool = False,
    sleep: float = 0.5,
) -> dict:
    """Recompute the close-basis peak for candidate open positions and raise
    any stored peak that has fallen behind.

    Returns a summary:
        {
          "total": int,
          "counters": {raised, unchanged, skipped_no_b1, skipped_no_data, errors},
          "rows": [ {ticker, entry_date, entry_price, stored, pct, tier,
                     action, trade_id, portfolio}, ... ],
        }
    action ∈ {"RAISED", "keep", "raise" (dry-run), "skip", "err"}.

    Never raises for a per-ticker failure — those become "err"/"skip" rows so
    a single bad symbol can't abort the batch (important for the always-on
    in-process caller).
    """
    candidates = fetch_candidates(portfolio, reconcile=reconcile)
    counters = {
        "raised": 0,
        "unchanged": 0,
        "skipped_no_b1": 0,
        "skipped_no_data": 0,
        "errors": 0,
    }
    rows: list[dict] = []

    for row in candidates:
        ticker = (row.get("ticker") or "").strip().upper()
        trade_id = row.get("trade_id")
        portfolio_name = row.get("portfolio_name")
        entry_raw = row.get("b1_entry_date")
        entry_price_raw = row.get("b1_entry_price")
        stored_raw = row.get("stored_max_pct")
        stored = float(stored_raw) if stored_raw is not None else None

        if not ticker or entry_raw is None or entry_price_raw is None:
            counters["skipped_no_b1"] += 1
            log.warning("Skipping %s/%s: missing B1 entry data", portfolio_name, trade_id)
            continue

        # b1_entry_date is a timestamp; normalize to date.
        if isinstance(entry_raw, datetime):
            entry_date = entry_raw.date()
        elif isinstance(entry_raw, date):
            entry_date = entry_raw
        else:
            entry_date = pd.to_datetime(entry_raw).date()
        entry_price = float(entry_price_raw)

        try:
            if sleep:
                time.sleep(sleep)
            pct = compute_max_b1_return(ticker, entry_date, entry_price)
        except Exception as exc:
            counters["errors"] += 1
            log.error("yfinance error for %s (%s): %s", ticker, trade_id, exc)
            continue

        if pct is None:
            counters["skipped_no_data"] += 1
            log.warning("No yfinance data for %s (%s) from %s", ticker, trade_id, entry_date)
            rows.append({
                "ticker": ticker, "entry_date": entry_date, "entry_price": entry_price,
                "stored": stored, "pct": None, "tier": "—", "action": "skip",
                "trade_id": trade_id, "portfolio": portfolio_name,
            })
            continue

        tier = classify_tier(pct)
        # Anticipate the monotonic guard: it writes only when stored is NULL
        # or strictly less than pct.
        would_raise = stored is None or stored < pct

        if dry_run:
            action = "raise" if would_raise else "keep"
        elif not would_raise:
            action = "keep"
            counters["unchanged"] += 1
        else:
            try:
                result = update_b1_max_return_pct(portfolio_name, trade_id, pct)
                if result is None:
                    counters["errors"] += 1
                    action = "err"
                    log.error("Trade %s/%s not found on write", portfolio_name, trade_id)
                elif result.get("was_updated"):
                    counters["raised"] += 1
                    action = "RAISED"
                    log.info("Raised %s (%s) b1_max_return_pct → %.2f%% (%s)",
                             ticker, trade_id, pct, tier)
                    # Auto-snapshot the SR8 activation anchor on first
                    # +50% crossing. Guarded by pct >= SR8_THRESHOLD and
                    # the helper's own NULL-only condition, so it's
                    # idempotent and doesn't clobber prior anchors.
                    if pct >= SR8_THRESHOLD:
                        try:
                            nlv = _latest_portfolio_nlv(portfolio_name)
                            if nlv is not None:
                                snap = snapshot_sr8_activation_if_null(
                                    portfolio_name, trade_id,
                                    activation_date=date.today(),
                                    activation_nlv=nlv,
                                )
                                if snap and snap.get("was_written"):
                                    log.info(
                                        "SR8 anchor snapshot %s (%s): "
                                        "date=%s nlv=$%.0f core_shs=%.2f",
                                        ticker, trade_id,
                                        snap["activation_date"],
                                        snap["activation_nlv"],
                                        snap["core_shares"],
                                    )
                        except Exception as anchor_exc:
                            # Snapshot failure MUST NOT abort the reconcile.
                            log.warning(
                                "SR8 anchor snapshot skipped for %s/%s: %s",
                                portfolio_name, trade_id, anchor_exc,
                            )
                else:
                    counters["unchanged"] += 1
                    action = "keep"
            except Exception as exc:
                counters["errors"] += 1
                action = "err"
                log.error("DB write failed for %s/%s: %s", portfolio_name, trade_id, exc)

        rows.append({
            "ticker": ticker, "entry_date": entry_date, "entry_price": entry_price,
            "stored": stored, "pct": pct, "tier": tier, "action": action,
            "trade_id": trade_id, "portfolio": portfolio_name,
        })

    return {"total": len(candidates), "counters": counters, "rows": rows}
