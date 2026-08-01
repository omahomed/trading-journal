"""Daily updater for market_data.

Fetches a recent window of bars from yfinance, recomputes indicators, and
upserts into market_data. Idempotent — safe to call multiple times per day.

Phase 2's MCTEngine invokes update_if_needed() at the start of each request
that requires current data. Phase 1 ships the module unwired.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from psycopg2.extras import execute_values

from db_layer import get_db_connection
from api.market_data_repo import get_latest_date


SYMBOL = "^IXIC"
# SPY is the companion index for the dual-index FTD gate (see mct_engine.py
# FTD_DUAL_INDEX_START). Ingested alongside ^IXIC so the engine's SPY
# side-channel (precomputed via _precompute_spy_ftd_confirmations) always
# has same-day data by the time the FTD gate is checked. Kept in the same
# market_data table under symbol="SPY" — no schema change needed.
COMPANION_SYMBOL = "SPY"

RECENT_WINDOW = 210  # enough lookback to keep sma_200 accurate at the right edge

log = logging.getLogger(__name__)


def update_latest_bar(symbol: str = SYMBOL) -> dict:
    """Fetch a recent window from yfinance, recompute indicators, upsert.

    After a successful upsert, runs the V11 MCT engine over the most recent
    window and persists any new signals to market_signals (idempotent via
    ON CONFLICT). On the first run this replays the full history; subsequent
    runs only emit signals for newly added bars.

    Returns: {"symbol", "trade_date", "rows_upserted", "action", "mct_signals"}.
        action is one of: "upsert" | "no-data".

    Companion index: when the primary symbol is ^IXIC, SPY is also ingested
    (no-op if already fresh) so the dual-index FTD gate (mct_engine.py
    FTD_DUAL_INDEX_START) has same-day SPY data available before the engine
    replay runs. SPY failures are logged but don't halt IXIC ingestion —
    the engine's "refuse to fire if SPY missing" guard covers the fallback.
    """
    if symbol == SYMBOL:
        try:
            _ingest_symbol_only(COMPANION_SYMBOL)
        except Exception as e:
            log.warning("[%s] companion ingest failed: %s (IXIC-only replay will "
                        "refuse to fire post-cutover FTDs until SPY lands)",
                        COMPANION_SYMBOL, e)

    df = _fetch_window(symbol, days=int(RECENT_WINDOW * 1.6))  # slack for weekends/holidays
    if df.empty:
        log.warning("yfinance returned no rows for %s", symbol)
        return {
            "symbol": symbol,
            "trade_date": None,
            "rows_upserted": 0,
            "action": "no-data",
            "mct_signals": None,
        }
    df = _compute_indicators(df).tail(RECENT_WINDOW)
    with get_db_connection() as conn:
        n = _upsert_rows(symbol, df, conn)

    mct_summary = _run_engine_and_write_signals(symbol)

    return {
        "symbol": symbol,
        "trade_date": df["trade_date"].iloc[-1],
        "rows_upserted": n,
        "action": "upsert",
        "mct_signals": mct_summary,
    }


def _ingest_symbol_only(symbol: str) -> dict:
    """Fetch + upsert bars for `symbol` without triggering the engine replay.

    Used to keep companion indexes (currently: SPY) fresh alongside ^IXIC
    without running the MCT engine on them (the engine only reads SPY as a
    side-input via mct_endpoint_adapter._precompute_spy_ftd_confirmations —
    it doesn't need per-SPY signals in market_signals). Same post-close
    freshness gate as update_if_needed — no-ops if today's bar already
    landed. Called by update_latest_bar for the companion symbol.
    """
    latest = get_latest_date(symbol)
    target = _last_business_day()
    if latest is not None and latest >= target:
        return {"symbol": symbol, "trade_date": latest,
                "rows_upserted": 0, "action": "no-op"}
    df = _fetch_window(symbol, days=int(RECENT_WINDOW * 1.6))
    if df.empty:
        log.warning("yfinance returned no rows for %s", symbol)
        return {"symbol": symbol, "trade_date": None,
                "rows_upserted": 0, "action": "no-data"}
    df = _compute_indicators(df).tail(RECENT_WINDOW)
    with get_db_connection() as conn:
        n = _upsert_rows(symbol, df, conn)
    return {"symbol": symbol,
            "trade_date": df["trade_date"].iloc[-1],
            "rows_upserted": n, "action": "upsert"}


def update_if_needed(symbol: str = SYMBOL) -> dict:
    """Skip the network round-trip if market_data already has today's bar.

    "Today" = the most recent US weekday. Market holidays are not modeled
    here — yfinance returns no new bar on holidays so the next call simply
    no-ops once the day after rolls over.

    Returns the same shape as update_latest_bar, plus action="no-op" when
    no fetch was performed.
    """
    latest = get_latest_date(symbol)
    target = _last_business_day()
    if latest is not None and latest >= target:
        return {
            "symbol": symbol,
            "trade_date": latest,
            "rows_upserted": 0,
            "action": "no-op",
        }
    return update_latest_bar(symbol)


def _fetch_window(symbol: str, days: int) -> pd.DataFrame:
    end = datetime.utcnow().date() + timedelta(days=1)  # exclusive end → inclusive of today
    start = end - timedelta(days=days)
    raw = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
    if raw.empty:
        return pd.DataFrame()
    df = raw.reset_index()
    df["trade_date"] = pd.to_datetime(df["Date"]).dt.date
    df = df[["trade_date", "Open", "High", "Low", "Close", "Volume"]].rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return _patch_lagged_close_from_info(symbol, df)


def _patch_lagged_close_from_info(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Fill a settled NaN close from yf.Ticker.info when yf.history lags.

    yfinance has two endpoints with different update cadences. The
    .history() endpoint that _fetch_window uses is sometimes hours-to-
    a-day behind on the latest settled close — observed on 2026-06-16
    when .history returned the 06-15 row with OHLCV populated but
    Close = NaN, while .info.regularMarketPrice already had the correct
    settled price (26683.94). The lag is invisible to the user but
    silently strands the M Factor page on stale data because the
    NaN-OHLC guard in _upsert_rows correctly refuses to write the
    partial row.

    Fallback rule: when the LATEST NaN-close bar is for a date STRICTLY
    in the past (the session has definitely closed), patch its close
    from info.regularMarketPrice. We do NOT patch in-progress bars
    (date == today) — those are genuinely intraday partials and must
    stay dropped, otherwise we'd write a live quote as if it were a
    settled close. Older historical NaN closes are real data corruption
    and also stay dropped.

    Idempotent: if .history already had a settled close, .info isn't
    consulted; if .info fails or returns a non-finite value, the row
    stays NaN and gets dropped downstream.
    """
    if df.empty or "close" not in df.columns:
        return df
    nan_mask = df["close"].isna()
    if not nan_mask.any():
        return df

    today = datetime.utcnow().date()
    candidates = df.index[nan_mask & (df["trade_date"] < today)]
    if len(candidates) == 0:
        return df  # only today's session NaN — let the guard drop it
    # Patch the most recent NaN-close that's settled; older NaNs are
    # corruption and shouldn't be papered over with a single .info value.
    idx = candidates.max()

    try:
        info = yf.Ticker(symbol).info
    except Exception as e:
        log.warning("[%s] .info fetch failed for NaN-close fallback: %s", symbol, e)
        return df
    fallback = info.get("regularMarketPrice")
    if fallback is None or not pd.notna(fallback) or float(fallback) <= 0:
        log.warning(
            "[%s] no usable .info.regularMarketPrice for %s (got %r); "
            "leaving close NaN for guard to drop",
            symbol, df.loc[idx, "trade_date"], fallback,
        )
        return df

    df.loc[idx, "close"] = float(fallback)
    log.info(
        "[%s] patched %s close from .info.regularMarketPrice = %.4f "
        "(yf.history lagged the settled close)",
        symbol, df.loc[idx, "trade_date"], float(fallback),
    )
    return df


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("trade_date").reset_index(drop=True).copy()
    df["ema_8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()
    return df


def _upsert_rows(symbol: str, df: pd.DataFrame, conn) -> int:
    if df.empty:
        return 0

    # Drop partial / in-progress bars. yfinance returns a row for today's
    # session before close (open/high/low/volume populated, close = NaN)
    # while the market is still trading. Writing NaN to market_data.close
    # poisons every downstream consumer: the engine ratchet reads it, the
    # rally-prefix response embeds it in `price`, FastAPI's JSON encoder
    # throws on NaN, and every M Factor request 500s — until the row is
    # manually deleted. Skip these rows here; the next call after close
    # will pick up the settled bar. Indicator columns (ema/sma) are
    # allowed to be NaN — they tail-pad legitimately in early windows.
    ohlc_cols = ["open", "high", "low", "close"]
    nan_mask = df[ohlc_cols].isna().any(axis=1)
    if nan_mask.any():
        log.warning(
            "[%s] dropping %d partial bar(s) with NaN in OHLC: %s",
            symbol, int(nan_mask.sum()),
            df.loc[nan_mask, "trade_date"].tolist(),
        )
        df = df[~nan_mask]
        if df.empty:
            return 0

    rows = [
        (
            symbol,
            r.trade_date,
            float(r.open),
            float(r.high),
            float(r.low),
            float(r.close),
            int(r.volume) if pd.notna(r.volume) else None,
            None if pd.isna(r.ema_8) else float(r.ema_8),
            None if pd.isna(r.ema_21) else float(r.ema_21),
            None if pd.isna(r.sma_50) else float(r.sma_50),
            None if pd.isna(r.sma_200) else float(r.sma_200),
        )
        for r in df.itertuples(index=False)
    ]
    sql = """
        INSERT INTO market_data (
            symbol, trade_date, open, high, low, close, volume,
            ema_8, ema_21, sma_50, sma_200
        ) VALUES %s
        ON CONFLICT (symbol, trade_date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            ema_8 = EXCLUDED.ema_8,
            ema_21 = EXCLUDED.ema_21,
            sma_50 = EXCLUDED.sma_50,
            sma_200 = EXCLUDED.sma_200,
            updated_at = NOW()
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    return len(rows)


def _last_business_day(now_utc: Optional[datetime] = None) -> date:
    """Return the most recent trading day whose bar is SETTLED (post-close).

    The M Factor engine only operates on settled daily bars — an intraday
    fetch produces an early-morning snapshot whose close / high / low is
    wildly wrong vs. the real end-of-day (e.g., 2026-07-27 morning bar
    ingested at 8:45 AM CT showed a stale +0.73% close vs. the real
    -0.18% end-of-day 6+ hours later). All engine reads (STEP_0 pink-day
    gate, VIOLATION_21EMA, drawdown vs ref_high) are close-based, so a
    stale intraday bar produces wrong state.

    Time-of-day gate: NASDAQ closes 4 PM ET = 20:00 UTC (EDT) / 21:00 UTC
    (EST). We require ≥ 22:00 UTC to consider today "settled" — that's
    ~1 hour after close in EDT / immediately after close in EST, giving
    yfinance time to publish the daily bar without racing the intraday
    snapshot. Before that threshold on a weekday, today is treated as
    still open and the target rolls back to the previous weekday, so
    update_if_needed no-ops instead of fetching intraday data.

    Weekend fall-through is unchanged: Sat/Sun always roll back to
    Friday regardless of time.
    """
    now = now_utc or datetime.utcnow()
    d = now.date()
    # Today weekday but still intraday → roll back one day (which the
    # weekend loop below then normalizes to Friday for Mon-morning calls).
    if d.weekday() < 5 and now.hour < 22:
        d -= timedelta(days=1)
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= timedelta(days=1)
    return d


def _run_engine_and_write_signals(symbol: str = SYMBOL) -> dict:
    """Run the MCT engine over the full market_data history and persist signals.

    Idempotent — the unique constraint on (trade_date, signal_type) means
    re-runs over previously processed bars are no-ops in the database.
    First-run cost: replay the full history once. Subsequent runs only emit
    new signals for bars added since the last upsert.
    """
    from api.mct_engine import MCTEngine, EngineConfig
    from api.market_data_repo import get_history, get_latest_date
    from api.mct_signals_writer import write_signals

    latest = get_latest_date(symbol)
    if latest is None:
        return {"events_emitted": 0, "rows_inserted": 0, "reason": "no market_data"}

    history = get_history(symbol, date(2010, 1, 1), latest)
    if history.empty:
        return {"events_emitted": 0, "rows_inserted": 0, "reason": "empty history"}

    # Use the same config as the production endpoint adapter — single source
    # of truth for engine behavior across endpoints + daily updater. SPY-side
    # FTD confirmations are precomputed once from cutover forward and passed
    # into every replay, keeping engine behavior identical whether triggered
    # from an endpoint request or the nightly update cycle.
    from api.mct_endpoint_adapter import _default_config, _precompute_spy_ftd_confirmations
    spy_confirmations = (_precompute_spy_ftd_confirmations(latest)
                          if symbol == "^IXIC" else None)
    config = _default_config(float(history["high"].iloc[0]),
                             spy_confirmations=spy_confirmations)
    engine = MCTEngine(config)
    result = engine.run(history)
    inserted = write_signals(result.signals)
    return {
        "events_emitted": len(result.signals),
        "rows_inserted": inserted,
        "first_date": history["trade_date"].iloc[0],
        "last_date": history["trade_date"].iloc[-1],
    }
