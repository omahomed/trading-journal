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
            b1.b1_entry_date,
            b1.b1_entry_price,
            s.atr21_entry_pct,
            s.mae_mfe_last_updated
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
        WHERE s.deleted_at IS NULL
          AND COALESCE(s.instrument_type, 'STOCK') = 'STOCK'
    """
    params: list = []
    if not include_closed:
        sql += " AND s.status = 'OPEN'"
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
    df: pd.DataFrame, entry_price: float
) -> dict | None:
    """Walk the OHLC frame in date order and derive:
        mae_pct, mfe_pct, days_to_mae, days_to_mfe, max_retrace_pct

    * MAE / MFE reference the B1 entry price (spec §"Definitions") —
      running min-low and max-high across all bars in the frame.
    * days_to_mae / days_to_mfe are 0-based against the first bar in
      the frame — same-day entry means both = 0.
    * max_retrace_pct is the largest peak-to-trough drawdown (running
      max of high, minus low on subsequent bars) between entry and the
      last frame bar. Distinct from MAE: MAE is off entry, retrace is
      off the running peak. Signed ≤ 0.

    Returns None when the frame is empty. Callers guard entry_price>0
    upstream so the divisions here are safe.
    """
    if df.empty or entry_price <= 0:
        return None

    lows = df["Low"].astype(float).to_numpy()
    highs = df["High"].astype(float).to_numpy()
    n = len(lows)

    mae_low = float(lows[0])
    mfe_high = float(highs[0])
    days_to_mae = 0
    days_to_mfe = 0
    # Running peak seeded from bar 0's high. Retrace is measured on
    # bars 1..N-1 only — bar 0's own intrabar range would show up as a
    # "retrace" against bar 0's high, but that's the entry bar so it's
    # not a give-back, it's the origination of the position.
    running_peak = float(highs[0])
    max_retrace_pct = 0.0

    for i in range(n):
        low_i = float(lows[i])
        high_i = float(highs[i])
        # Update MAE / MFE against the entry price.
        if low_i < mae_low:
            mae_low = low_i
            days_to_mae = i
        if high_i > mfe_high:
            mfe_high = high_i
            days_to_mfe = i
        # Peak-to-trough retrace: current-bar low against the peak of
        # PRIOR bars (before this bar's high enters the running peak).
        # Order matters — if we updated running_peak first, bar N's
        # low would count against bar N's own high, which is intrabar
        # range, not a mid-run give-back.
        if i > 0 and running_peak > 0:
            retrace = (low_i - running_peak) / running_peak * 100.0
            if retrace < max_retrace_pct:
                max_retrace_pct = retrace
        if high_i > running_peak:
            running_peak = high_i

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
    ticker: str, entry_date: date, entry_price: float,
) -> dict | None:
    """Fetch [entry_date, today] daily OHLC and derive MAE/MFE/retrace.

    Returns None when yfinance has no data for the window (e.g. very
    recent same-day entries before end-of-day tape settles). Callers
    treat None as a skip — the next reconcile run picks it up when
    data lands.
    """
    end = date.today() + timedelta(days=1)  # yfinance end is exclusive
    df = _download_history(ticker, entry_date, end)
    return compute_excursions_from_frame(df, entry_price)


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
    candidates = fetch_candidates(portfolio, include_closed=include_closed)
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

        try:
            if sleep:
                time.sleep(sleep)
            excursions = compute_excursions(ticker, entry_date, entry_price)
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
