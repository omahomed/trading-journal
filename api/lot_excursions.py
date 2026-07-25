"""Per-lot MAE/MFE excursions — one row per BUY (B1, A1, A2, …).

Existing MAE/MFE tracking (migration 046 / api/mae_mfe_reconcile.py)
lives on trades_summary and is always anchored to B1's fill price. That
answers "how much drawdown did the campaign experience from initial
entry?" but not "how much drawdown did each add-on experience from ITS
own fill?" — which is the data needed to calibrate a broker-stop ATR
multiple for A-series lots the same way we did for B1 (the 0.75× ATR21
finding from the entry-level study).

This module computes per-lot MAE/MFE for every BUY row in a campaign,
each measured from that lot's own fill_price + fill_date → the
campaign's close_date (or today for open campaigns).

Design:
  * Pure computation. No DB writes. Two consumers:
      - GET /api/trades/{trade_id}/lot-excursions (sidecar display)
      - scripts/export_lot_excursions.py (CSV research export)
  * Reuses the battle-tested math from api/mae_mfe_reconcile:
      compute_excursions_from_frame — handles bar-0/exit-day rules
      compute_atr21_from_frame     — reference ATR21 impl
    Everything downstream (0.75× cutoffs, day-count nuances) then reads
    the same way regardless of grain.
  * One yfinance fetch per campaign, not per lot. Fetches the widest
    window (B1 date − 45 calendar days for pre-B1 ATR context, through
    close_date + 1) once, then slices per-lot via pandas.
  * Options skipped upstream (same COALESCE(instrument_type, 'STOCK')
    filter as migration 046). yfinance doesn't serve OCC symbols and
    the concept doesn't map cleanly.

Anchor / window semantics per lot:
  entry_date  = lot's fill_date
  entry_price = lot's fill_price
  window end  = campaign's closed_date (CLOSED) or today (OPEN)
  same-day sells → seed bar-0 MAE/MFE candidates for THIS lot's day
  exit-day sells (close_date) → applied when campaign is CLOSED (all
                                lots share the same terminal event)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from api.mae_mfe_reconcile import (
    _ATR21_LOOKBACK_DAYS,
    _ATR_PERIOD,
    _download_history,
    compute_atr21_from_frame,
    compute_excursions_from_frame,
)
from db_layer import get_db_connection


log = logging.getLogger("lot_excursions")


# ─────────────────────────────────────────────────────────────────────
# Candidate selection
# ─────────────────────────────────────────────────────────────────────


def fetch_campaign_lots(
    portfolio: str | None = None,
    trade_id: str | None = None,
    include_closed: bool = True,
    since: date | None = None,
) -> list[dict]:
    """Return per-lot BUY rows across matching equity campaigns.

    Each output row:
        { trade_id, portfolio_name, portfolio_id, ticker, status,
          closed_date, trx_id, fill_date, fill_price, shares,
          same_day_low_exit_price, same_day_high_exit_price,
          close_day_low_exit_price, close_day_high_exit_price,
          realized_pl }

    same_day_*_exit_price is scoped to THIS LOT's fill_date (not B1's) —
    seeds the bar-0 candidates for the per-lot walk. close_day_* is the
    campaign-level exit-day activity, shared across lots.

    Options filtered out via COALESCE(instrument_type, 'STOCK') = 'STOCK'
    (same rule as fetch_candidates in api/mae_mfe_reconcile — yfinance
    can't serve OCC option symbols).

    Ordering: portfolio → trade_id → lot fill_date/id. Callers relying
    on B1-before-A1 grouping get it for free.
    """
    sql = """
        SELECT
            s.trade_id,
            p.name AS portfolio_name,
            s.portfolio_id,
            s.ticker,
            s.status,
            s.closed_date,
            s.realized_pl,
            b.trx_id,
            b.date AS fill_date,
            b.amount AS fill_price,
            b.shares,
            -- Same-day SELL prices scoped to THIS lot's date. NULL when
            -- no sell landed on the same day (the common case for a
            -- clean add-on). Same MIN/MAX pattern as fetch_candidates
            -- for the campaign-level same-day rule.
            (SELECT MIN(NULLIF(d.amount, 0)) FROM trades_details d
              WHERE d.trade_id = s.trade_id
                AND d.portfolio_id = s.portfolio_id
                AND d.action = 'SELL'
                AND d.deleted_at IS NULL
                AND d.date::date = b.date::date
            ) AS same_day_low_exit_price,
            (SELECT MAX(NULLIF(d.amount, 0)) FROM trades_details d
              WHERE d.trade_id = s.trade_id
                AND d.portfolio_id = s.portfolio_id
                AND d.action = 'SELL'
                AND d.deleted_at IS NULL
                AND d.date::date = b.date::date
            ) AS same_day_high_exit_price,
            -- Campaign-level exit-day sells. All lots share these:
            -- when the campaign closes, every open lot exits together.
            (SELECT MIN(NULLIF(d.amount, 0)) FROM trades_details d
              WHERE d.trade_id = s.trade_id
                AND d.portfolio_id = s.portfolio_id
                AND d.action = 'SELL'
                AND d.deleted_at IS NULL
                AND s.status = 'CLOSED'
                AND s.closed_date IS NOT NULL
                AND d.date::date = s.closed_date::date
            ) AS close_day_low_exit_price,
            (SELECT MAX(NULLIF(d.amount, 0)) FROM trades_details d
              WHERE d.trade_id = s.trade_id
                AND d.portfolio_id = s.portfolio_id
                AND d.action = 'SELL'
                AND d.deleted_at IS NULL
                AND s.status = 'CLOSED'
                AND s.closed_date IS NOT NULL
                AND d.date::date = s.closed_date::date
            ) AS close_day_high_exit_price
        FROM trades_summary s
        JOIN portfolios p ON s.portfolio_id = p.id
        JOIN trades_details b
          ON b.trade_id = s.trade_id
         AND b.portfolio_id = s.portfolio_id
         AND b.action = 'BUY'
         AND b.deleted_at IS NULL
        WHERE s.deleted_at IS NULL
          AND COALESCE(s.instrument_type, 'STOCK') = 'STOCK'
    """
    params: list[Any] = []
    if not include_closed:
        sql += " AND s.status = 'OPEN'"
    if since:
        # Same OPEN-or-recently-closed gate as fetch_candidates.
        sql += " AND (s.status = 'OPEN' OR s.closed_date >= %s)"
        params.append(since)
    if portfolio:
        sql += " AND p.name = %s"
        params.append(portfolio)
    if trade_id:
        sql += " AND s.trade_id = %s"
        params.append(trade_id)
    sql += " ORDER BY p.name, s.trade_id, b.date ASC, b.id ASC"

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────
# Per-campaign compute (one yfinance fetch, sliced per lot)
# ─────────────────────────────────────────────────────────────────────


def _as_date(value: Any) -> date | None:
    """Coerce DB timestamps to a pure date. Returns None for missing values.

    datetime (and pd.Timestamp, a datetime subclass) are ALSO instances
    of date — the isinstance(..., date) branch alone would return them
    verbatim, and downstream .isoformat() would leak the time
    component. Strip via .date() first when the object carries time.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def compute_lot_excursions_for_campaign(lots: list[dict]) -> list[dict]:
    """Compute per-lot MAE/MFE for one campaign's BUY rows.

    Input: `lots` — all BUY rows for a single campaign (same trade_id,
    same ticker), ordered by fill_date. Typically the output of
    `fetch_campaign_lots()` filtered/grouped by trade_id.

    Fetches OHLC ONCE for the widest window (B1_date − 45d → close_date + 1)
    and slices per lot. Falls back to per-lot fetch only if the shared
    fetch returns an empty frame (network hiccup on the widest range).

    Output row per lot:
        { trade_id, portfolio_name, portfolio_id, ticker, trx_id,
          fill_date, fill_price, shares, status, closed_date,
          window_end_date, days_held,
          mae_pct, mfe_pct, days_to_mae, days_to_mfe,
          atr21_at_fill_pct, mae_atr_multiple, mfe_atr_multiple,
          min_low, min_low_date, max_high, max_high_date,
          realized_pl, error }

    error is None on success, or a string like "no_bars" / "bad_lot"
    when a lot can't be computed. Callers can filter or surface it.
    """
    if not lots:
        return []

    ticker = lots[0]["ticker"]
    trade_id = lots[0]["trade_id"]
    status = (lots[0].get("status") or "").upper()
    closed_date = _as_date(lots[0].get("closed_date"))

    # Window end: closed_date (CLOSED) or today (OPEN). +1 for
    # yfinance's exclusive end.
    window_end_date = closed_date if closed_date is not None else date.today()

    # Widest fetch: from B1's date − ATR lookback (need pre-B1 bars for
    # ATR21 anchor) through the window end. Single network call.
    b1_date = _as_date(lots[0]["fill_date"])
    if b1_date is None:
        return [_error_row(lot, "bad_lot", window_end_date) for lot in lots]

    fetch_start = b1_date - timedelta(days=_ATR21_LOOKBACK_DAYS)
    df_all = _download_history(ticker, fetch_start, window_end_date + timedelta(days=1))
    if df_all.empty:
        # Every lot marked no_bars — caller decides skip vs error.
        return [_error_row(lot, "no_bars", window_end_date) for lot in lots]

    out: list[dict] = []
    for lot in lots:
        row = _compute_one_lot(
            lot,
            df_all,
            closed_date=closed_date,
            window_end_date=window_end_date,
            status=status,
        )
        row["ticker"] = ticker
        row["trade_id"] = trade_id
        out.append(row)
    return out


def _compute_one_lot(
    lot: dict,
    df_all: pd.DataFrame,
    closed_date: date | None,
    window_end_date: date,
    status: str,
) -> dict:
    """Slice the shared campaign frame to THIS lot's window and compute."""
    fill_date = _as_date(lot["fill_date"])
    fill_price = float(lot["fill_price"] or 0)
    shares = float(lot.get("shares") or 0)

    base = _base_row(lot, window_end_date)

    if fill_date is None or fill_price <= 0:
        base["error"] = "bad_lot"
        return base

    # ATR21 at fill: 21+ bars ENDING the bar BEFORE fill_date. Frozen
    # per lot at compute time — this endpoint doesn't persist so no
    # drift concern; every call gets today's atr21_at_fill snapshot.
    pre_lot_df = df_all[df_all.index.date < fill_date]
    atr21 = compute_atr21_from_frame(pre_lot_df.tail(_ATR_PERIOD)) if len(pre_lot_df) >= _ATR_PERIOD else None

    # Window slice: from fill_date through window_end_date, inclusive.
    lot_df = df_all[(df_all.index.date >= fill_date) & (df_all.index.date <= window_end_date)]
    if lot_df.empty:
        base["error"] = "no_bars_in_window"
        base["atr21_at_fill_pct"] = atr21
        return base

    # Exit-day args: campaign-level (all lots share terminal event) —
    # but only apply when the campaign is CLOSED. compute_excursions_from_frame
    # already gates on n>1, so B1 same-day-close campaigns handle
    # naturally via same_day_* seeds only.
    exit_low = float(lot["close_day_low_exit_price"]) if lot.get("close_day_low_exit_price") and status == "CLOSED" else None
    exit_high = float(lot["close_day_high_exit_price"]) if lot.get("close_day_high_exit_price") and status == "CLOSED" else None

    same_day_low = float(lot["same_day_low_exit_price"]) if lot.get("same_day_low_exit_price") else None
    same_day_high = float(lot["same_day_high_exit_price"]) if lot.get("same_day_high_exit_price") else None

    result = compute_excursions_from_frame(
        lot_df, fill_price,
        same_day_low_exit_price=same_day_low,
        same_day_high_exit_price=same_day_high,
        exit_low_price=exit_low,
        exit_high_price=exit_high,
    )
    if result is None:
        base["error"] = "compute_failed"
        base["atr21_at_fill_pct"] = atr21
        return base

    # Absolute min-low / max-high anchors for CSV export. Callers can
    # cross-check the pct math and see the actual price levels touched.
    lows = lot_df["Low"].astype(float)
    highs = lot_df["High"].astype(float)
    min_low = float(lows.min())
    max_high = float(highs.max())
    min_low_date = lows.idxmin().date().isoformat() if not lows.empty else None
    max_high_date = highs.idxmax().date().isoformat() if not highs.empty else None

    mae_atr = round(abs(result["mae_pct"]) / atr21, 3) if atr21 and atr21 > 0 else None
    mfe_atr = round(result["mfe_pct"] / atr21, 3) if atr21 and atr21 > 0 else None

    base.update({
        "mae_pct":            result["mae_pct"],
        "mfe_pct":            result["mfe_pct"],
        "days_to_mae":        result["days_to_mae"],
        "days_to_mfe":        result["days_to_mfe"],
        "atr21_at_fill_pct":  atr21,
        "mae_atr_multiple":   mae_atr,
        "mfe_atr_multiple":   mfe_atr,
        "min_low":            round(min_low, 4),
        "min_low_date":       min_low_date,
        "max_high":           round(max_high, 4),
        "max_high_date":      max_high_date,
        "shares":             shares,
        "days_held":          (window_end_date - fill_date).days,
    })
    return base


def _base_row(lot: dict, window_end_date: date) -> dict:
    """Skeleton output row — populated fields set to None so consumers
    can serialize without KeyError even on error rows."""
    fill_date = _as_date(lot["fill_date"])
    return {
        "trade_id":           lot.get("trade_id"),
        "portfolio_name":     lot.get("portfolio_name"),
        "portfolio_id":       lot.get("portfolio_id"),
        "ticker":             lot.get("ticker"),
        "status":             lot.get("status"),
        "closed_date":        _iso(lot.get("closed_date")),
        "trx_id":             lot.get("trx_id"),
        "fill_date":          fill_date.isoformat() if fill_date else None,
        "fill_price":         float(lot["fill_price"]) if lot.get("fill_price") else None,
        "shares":             float(lot["shares"]) if lot.get("shares") else None,
        "window_end_date":    window_end_date.isoformat(),
        "days_held":          None,
        "mae_pct":            None,
        "mfe_pct":            None,
        "days_to_mae":        None,
        "days_to_mfe":        None,
        "atr21_at_fill_pct":  None,
        "mae_atr_multiple":   None,
        "mfe_atr_multiple":   None,
        "min_low":            None,
        "min_low_date":       None,
        "max_high":           None,
        "max_high_date":      None,
        "realized_pl":        float(lot["realized_pl"]) if lot.get("realized_pl") else None,
        "error":              None,
    }


def _error_row(lot: dict, error: str, window_end_date: date) -> dict:
    r = _base_row(lot, window_end_date)
    r["error"] = error
    return r


def _iso(value: Any) -> str | None:
    d = _as_date(value)
    return d.isoformat() if d else None


# ─────────────────────────────────────────────────────────────────────
# Group + drive
# ─────────────────────────────────────────────────────────────────────


def compute_all_lot_excursions(
    portfolio: str | None = None,
    trade_id: str | None = None,
    include_closed: bool = True,
    since: date | None = None,
    sleep: float = 0.3,
) -> list[dict]:
    """Fetch matching campaigns' lots + compute per-lot excursions.

    Groups the flat lot list by trade_id so each campaign's shared bar
    fetch happens once. Sleeps between campaigns to be nice to yfinance
    on the CSV-export path (60 tickers × 3 lots would otherwise be a
    rapid burst).
    """
    lots = fetch_campaign_lots(
        portfolio=portfolio,
        trade_id=trade_id,
        include_closed=include_closed,
        since=since,
    )
    if not lots:
        return []

    # Group by (portfolio_id, trade_id) tuple — trade_id ALONE is not
    # unique across portfolios (a "202603-001" exists in every portfolio
    # that opened its first March 2026 campaign). Grouping by trade_id
    # alone would glue different-portfolio campaigns together and
    # smear each campaign's closed_date / exit prices across lots that
    # don't belong to it → nonsensical windows (fill_date after
    # window_end_date) and spurious "no_bars_in_window" errors.
    grouped: dict[tuple, list[dict]] = {}
    for lot in lots:
        key = (lot.get("portfolio_id"), lot["trade_id"])
        grouped.setdefault(key, []).append(lot)

    import time as _time
    out: list[dict] = []
    for i, (key, campaign_lots) in enumerate(grouped.items()):
        if i > 0 and sleep > 0:
            _time.sleep(sleep)
        try:
            rows = compute_lot_excursions_for_campaign(campaign_lots)
        except Exception as exc:
            log.exception("compute failed for %s: %s", key, exc)
            rows = [_error_row(lot, "compute_exception", date.today()) for lot in campaign_lots]
        out.extend(rows)
    return out
