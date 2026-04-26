#!/usr/bin/env python3
"""Backfill market_data with historical OHLC + computed indicators.

Default range: 2010-01-01 → yesterday. Data source: yfinance with
auto_adjust=False to keep raw OHLC for indicator computation.

Idempotent — safe to re-run. Uses INSERT ... ON CONFLICT (symbol, trade_date)
DO UPDATE so existing rows are refreshed with the latest indicator values.

Usage:
    python scripts/backfill_market_data.py
    python scripts/backfill_market_data.py --start 2010-01-01 --symbol ^IXIC
    python scripts/backfill_market_data.py --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from psycopg2.extras import execute_values

# Make repo root importable when this script is invoked directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db_layer import get_db_connection  # noqa: E402


DEFAULT_SYMBOL = "^IXIC"
DEFAULT_START = "2010-01-01"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backfill_market_data")


def fetch_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Pull daily OHLCV from yfinance for [start, end] inclusive."""
    log.info("Fetching %s from %s to %s", symbol, start, end)
    raw = yf.Ticker(symbol).history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
        auto_adjust=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol} {start}–{end}")
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
    log.info("Fetched %d rows", len(df))
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ema_8, ema_21, sma_50, sma_200 using the project convention.

    EMAs use ewm(span=N, adjust=False); SMAs use rolling(window=N).mean().
    """
    df = df.sort_values("trade_date").reset_index(drop=True).copy()
    df["ema_8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()
    return df


def upsert(symbol: str, df: pd.DataFrame) -> int:
    """Bulk upsert into market_data. Returns row count attempted."""
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
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
    return len(rows)


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Inclusive end date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Ticker symbol (default: ^IXIC)")
    args = parser.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end) if args.end else date.today() - timedelta(days=1)

    if end < start:
        log.error("end (%s) is before start (%s)", end, start)
        return 2

    df = fetch_history(args.symbol, start, end)
    df = compute_indicators(df)

    nan_sma200 = int(df["sma_200"].isna().sum())
    log.info(
        "Indicator stats: %d rows, %d sma_200 NULLs (expected ~199 for first window)",
        len(df),
        nan_sma200,
    )

    n = upsert(args.symbol, df)
    log.info("Upserted %d rows into market_data for %s", n, args.symbol)
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
