"""Reconcile trades_summary MAE / MFE / retracement columns.

Per-trade excursion metrics, measured against the B1 (initial buy) fill
price on daily bars:

    mae_pct         = (min(daily low since entry) - entry) / entry     [≤ 0]
    mfe_pct         = (max(daily high since entry) - entry) / entry    [≥ 0]
    atr21_entry_pct = ATR21 (%) on the ~30 bars ending the day BEFORE
                      entry — FROZEN SNAPSHOT, only ever computed once
                      per trade so ×ATR multiples don't drift.
    days_to_mae     = trading days from entry to the low print (0-based;
    days_to_mfe       same-day entry = 0)
    max_retrace_pct = largest peak-to-trough decline (daily high → low)
                      between entry and now — mid-run give-back,
                      distinct from MAE.

Options are skipped: consistent with b1_reconcile's `COALESCE(
instrument_type, 'STOCK') = 'STOCK'` filter — yfinance doesn't serve
OCC option symbols and the concept doesn't map cleanly.

Consumers:
  * scripts/backfill_mae_mfe.py — CLI wrapper for one-off runs.
  * api/main.py — daily in-process asyncio loop mirroring the
    b1_reconcile pattern.

Idempotent: safe to re-run. atr21_entry_pct is written once (per row)
and never touched again; every other field is recomputed from a fresh
yfinance pull. Per-ticker failures degrade to "err"/"skip" rows so a
single bad symbol can't abort the batch.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

from db_layer import get_db_connection


log = logging.getLogger("mae_mfe_reconcile")

# ATR21 pre-entry lookback. Need at least 22 bars (21 TRs + 1 prior
# close for the shift). 45 calendar days ≈ 30 trading days after
# weekends/holidays — comfortable slack.
_ATR21_LOOKBACK_DAYS = 45
_ATR_PERIOD = 21


# ─────────────────────────────────────────────────────────────────────
# Candidate selection
# ─────────────────────────────────────────────────────────────────────


def fetch_candidates(
    portfolio: str | None = None,
    include_closed: bool = False,
    since: date | None = None,
) -> list[dict]:
    """Find equity campaigns whose MAE/MFE state needs updating.

    Default: OPEN equity positions only (daily loop + backfill scope).
    include_closed=True flips it to every equity campaign, for Phase 2
    when we're ready to backfill historical closed trades. Filtered
    out today by default so the daily loop stays fast and the ~500-row
    closed-trade backfill happens only when explicitly requested.

    Returns rows shaped:
        { trade_id, ticker, portfolio_name, portfolio_id,
          status, b1_entry_date, b1_entry_price,
          atr21_entry_pct, mae_mfe_last_updated }

    Rows missing a B1 BUY row (data corruption / pre-app imports) are
    filtered out by the JOIN's own IS NOT NULL — the reconciler just
    won't see them.
    """
    sql = """
        SELECT
            s.trade_id,
            s.ticker,
            p.name AS portfolio_name,
            s.portfolio_id,
            s.status,
            s.closed_date,
            b1.b1_entry_date,
            b1.b1_entry_price,
            s.atr21_entry_pct,
            s.mae_mfe_last_updated,
            -- Same-day SELL activity on the entry_date. Under the "skip
            -- bar 0 OHLC unless there was actual same-day exit
            -- activity" rule, these become the bar 0 MAE / MFE
            -- candidates (fill price = what the trader actually
            -- experienced, not what the bar's range shows). NULL when
            -- no sell landed on entry_date, in which case bar 0 is
            -- skipped entirely for both MAE and MFE.
            sd.same_day_low_exit_price,
            sd.same_day_high_exit_price,
            -- Exit-day SELL activity on closed_date (CLOSED trades
            -- only). Under the SYMMETRIC exit-day skip rule, these
            -- become the last-bar MAE / MFE candidates and the last
            -- bar's OHLC is ignored — the trader was out the moment
            -- the final SELL filled, so any lower low / higher high
            -- that printed AFTER the exit is not part of their
            -- position's excursion. NULL for OPEN trades or when the
            -- close_date has no sell rows (data glitch).
            cd.close_day_low_exit_price,
            cd.close_day_high_exit_price
        FROM trades_summary s
        JOIN portfolios p ON s.portfolio_id = p.id
        JOIN LATERAL (
            SELECT d.date AS b1_entry_date, d.amount AS b1_entry_price
              FROM trades_details d
             WHERE d.trade_id = s.trade_id
               AND d.portfolio_id = s.portfolio_id
               AND d.action = 'BUY'
               AND d.deleted_at IS NULL
             ORDER BY d.date ASC, d.id ASC
             LIMIT 1
        ) b1 ON TRUE
        LEFT JOIN LATERAL (
            -- Both extremes in a single scan of same-day sells. Sells
            -- with amount = 0 or NULL don't count (data glitches /
            -- non-executed placeholder rows).
            --
            -- Cast BOTH sides to ::date because trades_details.date is
            -- a TIMESTAMP with real times (e.g. 08:36:00 from importer
            -- flows), while b1_entry_date is also a timestamp from the
            -- same source — same-day sells with different intraday
            -- times would silently miss without the cast. Pre-fix,
            -- every row returned NULL and same_day-* was never used.
            SELECT
                MIN(NULLIF(d.amount, 0)) AS same_day_low_exit_price,
                MAX(NULLIF(d.amount, 0)) AS same_day_high_exit_price
              FROM trades_details d
             WHERE d.trade_id = s.trade_id
               AND d.portfolio_id = s.portfolio_id
               AND d.action = 'SELL'
               AND d.deleted_at IS NULL
               AND d.date::date = b1.b1_entry_date::date
        ) sd ON TRUE
        LEFT JOIN LATERAL (
            -- Symmetric to sd, for sells on the CLOSE date. Only
            -- returns rows for CLOSED trades — the join predicate
            -- gates on s.status. Same ::date cast rationale as sd.
            SELECT
                MIN(NULLIF(d.amount, 0)) AS close_day_low_exit_price,
                MAX(NULLIF(d.amount, 0)) AS close_day_high_exit_price
              FROM trades_details d
             WHERE d.trade_id = s.trade_id
               AND d.portfolio_id = s.portfolio_id
               AND d.action = 'SELL'
               AND d.deleted_at IS NULL
               AND s.status = 'CLOSED'
               AND s.closed_date IS NOT NULL
               AND d.date::date = s.closed_date::date
        ) cd ON TRUE
        WHERE s.deleted_at IS NULL
          AND COALESCE(s.instrument_type, 'STOCK') = 'STOCK'
    """
    params: list = []
    if not include_closed:
        sql += " AND s.status = 'OPEN'"
    # `since` is inclusive-lower on the trade's LIFE OVERLAP with the
    # window. It's here to sidestep the pre-2026 legacy import garbage
    # (corporate actions, wrong entry prices, delisted symbols) that
    # can't be reliably backfilled from yfinance. The filter keeps a
    # trade in-scope if it was open on or after `since` — i.e. still
    # open now, or closed at/after `since`. This is what the operator
    # wants when they ask for a per-year backfill.
    if since:
        sql += " AND (s.status = 'OPEN' OR s.closed_date >= %s)"
        params.append(since)
    if portfolio:
        sql += " AND p.name = %s"
        params.append(portfolio)
    sql += " ORDER BY p.name, s.trade_id"

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────
# Yfinance helpers (thin wrappers for testability + retry surface)
# ─────────────────────────────────────────────────────────────────────


def _download_history(
    ticker: str, start: date, end: date
) -> pd.DataFrame:
    """Pull OHLC bars from yfinance for [start, end]. End is exclusive
    per yfinance convention, so callers pass tomorrow to include today.

    Returns an empty DataFrame on any failure or empty response — the
    caller decides how to log/skip. Never raises.
    """
    try:
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        log.error("yfinance download failed for %s: %s", ticker, exc)
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    # yfinance sometimes returns MultiIndex columns for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


# ─────────────────────────────────────────────────────────────────────
# Pure math (no DB, no yfinance — deterministic; unit-tested)
# ─────────────────────────────────────────────────────────────────────


def compute_atr21_from_frame(df: pd.DataFrame) -> float | None:
    """ATR21% = SMA(TR, 21) / SMA(Low, 21) × 100 on the last 21 bars.

    Matches api/main.py:_compute_ticker_atr_pct exactly (that function
    is the reference implementation for every ATR% consumer in the
    app). Requires at least 21 bars ending at the anchor date; returns
    None otherwise.
    """
    if df.empty or len(df) < _ATR_PERIOD:
        return None
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    sma_tr = float(tr.tail(_ATR_PERIOD).mean())
    sma_low = float(low.tail(_ATR_PERIOD).mean())
    if sma_low <= 0:
        return None
    return round((sma_tr / sma_low) * 100, 4)


def compute_excursions_from_frame(
    df: pd.DataFrame,
    entry_price: float,
    same_day_low_exit_price: float | None = None,
    same_day_high_exit_price: float | None = None,
    exit_low_price: float | None = None,
    exit_high_price: float | None = None,
) -> dict | None:
    """Walk the OHLC frame in date order and derive:
        mae_pct, mfe_pct, days_to_mae, days_to_mfe, max_retrace_pct

    Entry-day (bar 0) treatment:
      * The entry-day bar's own OHLC extremes do NOT count toward MAE
        or MFE — without intraday granularity, we can't attribute
        bar 0's low/high to a moment AFTER the trader's entry. On a
        reversal-candle entry (buy near close after an intraday
        washout), bar 0's low happened BEFORE entry and doesn't
        reflect anything the trader actually experienced.
      * EXCEPTION: if the trader had SELL activity on entry_date, the
        fill price is a known point they were at. Same-day sell prices
        below entry seed the bar 0 MAE candidate; same-day sell prices
        above entry seed the bar 0 MFE candidate.
      * When there's no same-day sell activity: bar 0 contributes 0 to
        MAE and MFE.

    Exit-day (bar N-1) treatment — SYMMETRIC to entry day:
      * When exit_low_price / exit_high_price are provided (the trade
        is CLOSED and its close-date sells are known), the last bar's
        OHLC extremes are IGNORED. The trader was OUT the moment the
        last SELL filled — any lower low or higher high that printed
        AFTER their exit price is not part of the position's excursion.
      * The exit price(s) themselves ARE known-touched levels: they
        can improve mae_low (if lower than any prior bar's low) or
        mfe_high (if higher than any prior bar's high).
      * When exit_*_price are None (OPEN trade, or CLOSED trade with
        no sell rows on closed_date — rare data glitch), the last bar
        walks normally with its full OHLC. Preserves back-compat.
      * n == 1 (same-day round-trip: entry_date == closed_date): bar 0
        IS bar N-1, so this treatment reduces to the entry-day rule.
        The exit_*_price args are ignored (same_day_*_exit_price
        already covers it). Caller doesn't need to special-case n == 1.

    Retracement: peak-to-trough on bars 1..N-1 low against running
    peak. On exit day (when substituted), exit_low_price acts as the
    "low" for retrace measurement — a sell that undercut the running
    peak IS a give-back the position experienced before closing.

    Returns None when the frame is empty or entry_price is not
    positive. Callers guard the "any input missing" case upstream, so
    the divisions here are safe.
    """
    if df.empty or entry_price <= 0:
        return None

    lows = df["Low"].astype(float).to_numpy()
    highs = df["High"].astype(float).to_numpy()
    n = len(lows)

    # Bar 0 seeds: entry_price (contributes 0) unless a same-day sell
    # gives us a lower/higher known-touched price. Both "sell price"
    # inputs are optional — most trades have none.
    mae_low = float(entry_price)
    mfe_high = float(entry_price)
    days_to_mae = 0
    days_to_mfe = 0
    if same_day_low_exit_price is not None and same_day_low_exit_price > 0:
        p = float(same_day_low_exit_price)
        if p < mae_low:
            mae_low = p
    if same_day_high_exit_price is not None and same_day_high_exit_price > 0:
        p = float(same_day_high_exit_price)
        if p > mfe_high:
            mfe_high = p

    # Running peak seeded from bar 0's HIGH (unchanged). Retrace is
    # only measured on bars 1..N-1 anyway, so bar 0's high is the
    # correct starting anchor for the peak — a lower low on bar 1
    # against bar 0's high IS a mid-run give-back.
    running_peak = float(highs[0])
    max_retrace_pct = 0.0

    # Exit-day substitution: when the trade is CLOSED and we know its
    # close-date sell prices, treat bar N-1 as a "known touch only"
    # bar (symmetric to bar 0). Only apply when n > 1; otherwise bar 0
    # already handles the same-day round-trip via same_day_*_exit_price.
    has_exit_day_rule = (
        n > 1
        and (
            (exit_low_price is not None and exit_low_price > 0)
            or (exit_high_price is not None and exit_high_price > 0)
        )
    )
    last_walk_i = n - 2 if has_exit_day_rule else n - 1

    # Middle bars — full OHLC in play. On OPEN trades this is bars
    # 1..N-1; on CLOSED trades with exit-day substitution it's 1..N-2.
    for i in range(1, last_walk_i + 1):
        low_i = float(lows[i])
        high_i = float(highs[i])
        if low_i < mae_low:
            mae_low = low_i
            days_to_mae = i
        if high_i > mfe_high:
            mfe_high = high_i
            days_to_mfe = i
        # Peak-to-trough retrace: current-bar low against the peak of
        # PRIOR bars (before this bar's high enters the running peak).
        if running_peak > 0:
            retrace = (low_i - running_peak) / running_peak * 100.0
            if retrace < max_retrace_pct:
                max_retrace_pct = retrace
        if high_i > running_peak:
            running_peak = high_i

    # Exit-day treatment — the exit price(s) are known-touched. Lower
    # of the day's sells → potential MAE candidate; higher → MFE.
    if has_exit_day_rule:
        exit_day_idx = n - 1
        if exit_low_price is not None and exit_low_price > 0:
            p_low = float(exit_low_price)
            if p_low < mae_low:
                mae_low = p_low
                days_to_mae = exit_day_idx
            # A sell that undercut the running peak IS a give-back.
            if running_peak > 0:
                retrace = (p_low - running_peak) / running_peak * 100.0
                if retrace < max_retrace_pct:
                    max_retrace_pct = retrace
        if exit_high_price is not None and exit_high_price > 0:
            p_high = float(exit_high_price)
            if p_high > mfe_high:
                mfe_high = p_high
                days_to_mfe = exit_day_idx

    return {
        "mae_pct":         round((mae_low  - entry_price) / entry_price * 100.0, 4),
        "mfe_pct":         round((mfe_high - entry_price) / entry_price * 100.0, 4),
        "days_to_mae":     int(days_to_mae),
        "days_to_mfe":     int(days_to_mfe),
        "max_retrace_pct": round(float(max_retrace_pct), 4),
    }


# ─────────────────────────────────────────────────────────────────────
# Yfinance-backed compute wrappers (thin — just fetch + delegate)
# ─────────────────────────────────────────────────────────────────────


def compute_atr21_entry(ticker: str, entry_date: date) -> float | None:
    """ATR21% on the ~30 daily bars ending the day BEFORE entry.
    Frozen snapshot: caller guards `IF atr21_entry_pct IS NULL` so this
    only runs once per trade.
    """
    start = entry_date - timedelta(days=_ATR21_LOOKBACK_DAYS)
    df = _download_history(ticker, start, entry_date)  # end exclusive → excludes entry_date
    return compute_atr21_from_frame(df)


def compute_excursions(
    ticker: str,
    entry_date: date,
    entry_price: float,
    same_day_low_exit_price: float | None = None,
    same_day_high_exit_price: float | None = None,
    closed_date: date | None = None,
    exit_low_price: float | None = None,
    exit_high_price: float | None = None,
) -> dict | None:
    """Fetch daily OHLC and derive MAE/MFE/retrace.

    Window: [entry_date, closed_date] when the trade is CLOSED (both
    inclusive), else [entry_date, today]. Closing the window at
    closed_date is required for the exit-day skip rule — without it,
    bar N-1 would be TODAY's bar (post-exit), not the exit day.

    Returns None when yfinance has no data for the window (e.g. very
    recent same-day entries before end-of-day tape settles). Callers
    treat None as a skip — the next reconcile run picks it up when
    data lands.

    same_day_*_exit_price args seed the entry-day (bar 0) MAE / MFE
    candidates for same-day partial sells.

    exit_*_price args + closed_date together trigger the exit-day skip
    rule: the last bar's OHLC extremes are ignored, exit prices act as
    known-touched levels. See compute_excursions_from_frame's docstring
    for the full rule.
    """
    end_anchor = (closed_date + timedelta(days=1)) if closed_date is not None \
        else (date.today() + timedelta(days=1))  # yfinance end is exclusive
    df = _download_history(ticker, entry_date, end_anchor)
    return compute_excursions_from_frame(
        df, entry_price,
        same_day_low_exit_price=same_day_low_exit_price,
        same_day_high_exit_price=same_day_high_exit_price,
        exit_low_price=exit_low_price if closed_date is not None else None,
        exit_high_price=exit_high_price if closed_date is not None else None,
    )


# ─────────────────────────────────────────────────────────────────────
# Write path
# ─────────────────────────────────────────────────────────────────────


def _update_row(
    portfolio_id: int,
    trade_id: str,
    mae: dict,
    atr21_entry_pct: float | None,
    write_atr21: bool,
) -> None:
    """Persist the excursion fields + last_updated stamp for a row.

    write_atr21=True means the caller computed a fresh atr21_entry_pct
    (first observation for this trade) and we should write it. Otherwise
    the stored value is preserved — the spec calls for a frozen snapshot.
    """
    now = date.today()
    sql = """
        UPDATE trades_summary
           SET mae_pct              = %s,
               mfe_pct              = %s,
               days_to_mae          = %s,
               days_to_mfe          = %s,
               max_retrace_pct      = %s,
               mae_mfe_last_updated = %s
    """
    params: list = [
        mae["mae_pct"], mae["mfe_pct"],
        mae["days_to_mae"], mae["days_to_mfe"],
        mae["max_retrace_pct"], now,
    ]
    if write_atr21:
        sql += ", atr21_entry_pct = %s"
        params.append(atr21_entry_pct)
    sql += """
         WHERE portfolio_id = %s
           AND trade_id     = %s
           AND deleted_at IS NULL
    """
    params.extend([portfolio_id, trade_id])
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        conn.commit()


# ─────────────────────────────────────────────────────────────────────
# Top-level driver
# ─────────────────────────────────────────────────────────────────────


def reconcile_open_positions(
    portfolio: str | None = None,
    include_closed: bool = False,
    dry_run: bool = False,
    sleep: float = 0.3,
    since: date | None = None,
) -> dict:
    """Sweep candidate positions, recompute MAE/MFE, persist.

    Returns:
        {
          "total": int,
          "counters": {updated, skipped_no_data, skipped_bad_row,
                       atr_snapshotted, errors},
          "rows": [ per-candidate action rows ],
        }

    action ∈ {"UPDATED", "update" (dry-run), "skip", "err"}.
    Never raises for a per-ticker failure — those become "err"/"skip"
    rows so a single bad symbol can't abort the batch (matches the
    b1_reconcile always-on caller invariant).
    """
    candidates = fetch_candidates(portfolio, include_closed=include_closed, since=since)
    counters = {
        "updated": 0,
        "skipped_no_data": 0,
        "skipped_bad_row": 0,
        "atr_snapshotted": 0,
        "errors": 0,
    }
    rows: list[dict] = []

    for row in candidates:
        ticker = (row.get("ticker") or "").strip().upper()
        trade_id = row.get("trade_id")
        portfolio_id = row.get("portfolio_id")
        portfolio_name = row.get("portfolio_name")
        entry_raw = row.get("b1_entry_date")
        entry_price_raw = row.get("b1_entry_price")
        stored_atr = row.get("atr21_entry_pct")

        if not ticker or entry_raw is None or entry_price_raw is None:
            counters["skipped_bad_row"] += 1
            log.warning("Skipping %s/%s: missing B1 entry data",
                        portfolio_name, trade_id)
            rows.append({
                "ticker": ticker, "trade_id": trade_id,
                "portfolio": portfolio_name, "action": "skip",
                "reason": "missing b1 entry",
            })
            continue

        entry_date = (
            entry_raw.date() if isinstance(entry_raw, datetime)
            else entry_raw if isinstance(entry_raw, date)
            else pd.to_datetime(entry_raw).date()
        )
        try:
            entry_price = float(entry_price_raw)
        except (TypeError, ValueError):
            counters["skipped_bad_row"] += 1
            rows.append({
                "ticker": ticker, "trade_id": trade_id,
                "portfolio": portfolio_name, "action": "skip",
                "reason": "bad entry price",
            })
            continue

        # Same-day sell prices (if any) participate in bar 0 MAE / MFE
        # per the entry-day rule — see compute_excursions_from_frame's
        # docstring. Cast safely: psycopg2 sometimes returns Decimals
        # for NUMERIC columns which don't compose with the pure-float
        # math downstream.
        same_day_low = row.get("same_day_low_exit_price")
        same_day_high = row.get("same_day_high_exit_price")
        try:
            same_day_low_f = float(same_day_low) if same_day_low is not None else None
            same_day_high_f = float(same_day_high) if same_day_high is not None else None
        except (TypeError, ValueError):
            same_day_low_f = None
            same_day_high_f = None

        # Exit-day (close_date) sell prices — trigger the symmetric
        # exit-day skip rule for CLOSED trades. NULL for OPEN trades or
        # when close_date has no sell rows (defensive: rare data glitch).
        closed_date_raw = row.get("closed_date")
        exit_low = row.get("close_day_low_exit_price")
        exit_high = row.get("close_day_high_exit_price")
        closed_date_d: date | None = None
        if closed_date_raw is not None:
            closed_date_d = (
                closed_date_raw.date() if isinstance(closed_date_raw, datetime)
                else closed_date_raw if isinstance(closed_date_raw, date)
                else pd.to_datetime(closed_date_raw).date()
            )
        try:
            exit_low_f = float(exit_low) if exit_low is not None else None
            exit_high_f = float(exit_high) if exit_high is not None else None
        except (TypeError, ValueError):
            exit_low_f = None
            exit_high_f = None

        try:
            if sleep:
                time.sleep(sleep)
            excursions = compute_excursions(
                ticker, entry_date, entry_price,
                same_day_low_exit_price=same_day_low_f,
                same_day_high_exit_price=same_day_high_f,
                closed_date=closed_date_d,
                exit_low_price=exit_low_f,
                exit_high_price=exit_high_f,
            )
        except Exception as exc:
            counters["errors"] += 1
            log.error("MAE/MFE compute failed for %s (%s): %s",
                      ticker, trade_id, exc)
            rows.append({
                "ticker": ticker, "trade_id": trade_id,
                "portfolio": portfolio_name, "action": "err",
                "reason": str(exc),
            })
            continue

        if excursions is None:
            counters["skipped_no_data"] += 1
            log.warning("No yfinance data for %s (%s) from %s",
                        ticker, trade_id, entry_date)
            rows.append({
                "ticker": ticker, "trade_id": trade_id,
                "portfolio": portfolio_name, "action": "skip",
                "reason": "no yfinance data",
            })
            continue

        # ATR21 entry snapshot — only computed the first time we see
        # this trade. Once written, spec says "frozen, never recomputed."
        atr21 = float(stored_atr) if stored_atr is not None else None
        write_atr21 = False
        if atr21 is None:
            try:
                if sleep:
                    time.sleep(sleep)
                atr21 = compute_atr21_entry(ticker, entry_date)
            except Exception as exc:
                log.warning("ATR21 snapshot failed for %s (%s): %s — "
                            "leaving null; retry next run",
                            ticker, trade_id, exc)
                atr21 = None
            if atr21 is not None:
                write_atr21 = True
                counters["atr_snapshotted"] += 1

        if dry_run:
            action = "update"
        else:
            try:
                _update_row(portfolio_id, trade_id, excursions,
                            atr21, write_atr21)
                counters["updated"] += 1
                action = "UPDATED"
            except Exception as exc:
                counters["errors"] += 1
                action = "err"
                log.error("DB write failed for %s/%s: %s",
                          portfolio_name, trade_id, exc)

        rows.append({
            "ticker": ticker,
            "trade_id": trade_id,
            "portfolio": portfolio_name,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "atr21_entry_pct": atr21,
            "atr21_snapshotted_this_run": write_atr21,
            **excursions,
            "action": action,
        })

    return {"total": len(candidates), "counters": counters, "rows": rows}
