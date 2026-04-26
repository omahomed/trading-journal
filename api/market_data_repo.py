"""Read-only data access for the market_data table.

Phase 2's MCTEngine consumes these helpers to pull OHLC + indicator history
without going back to yfinance on every request. All queries hit the canonical
persisted store populated by scripts/backfill_market_data.py and
api/market_data_updater.py.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from db_layer import get_db_connection


# Columns selected for DataFrame results (no symbol — caller already knows it).
_DF_COLUMNS = (
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ema_8",
    "ema_21",
    "sma_50",
    "sma_200",
)
_DF_SELECT = ", ".join(_DF_COLUMNS)

_FLOAT_COLS = ("open", "high", "low", "close", "ema_8", "ema_21", "sma_50", "sma_200")


def get_bar(symbol: str, trade_date: date) -> Optional[dict]:
    """Return the bar for (symbol, trade_date) or None if not present."""
    sql = f"""
        SELECT symbol, {_DF_SELECT}
          FROM market_data
         WHERE symbol = %s AND trade_date = %s
         LIMIT 1
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol, trade_date))
        row = cur.fetchone()
    if not row:
        return None
    keys = ("symbol",) + _DF_COLUMNS
    return dict(zip(keys, row))


def get_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Return all bars for `symbol` in [start, end] inclusive, ascending by trade_date."""
    sql = f"""
        SELECT {_DF_SELECT}
          FROM market_data
         WHERE symbol = %s
           AND trade_date BETWEEN %s AND %s
         ORDER BY trade_date ASC
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol, start, end))
        rows = cur.fetchall()
    return _rows_to_df(rows)


def get_recent(symbol: str, n: int) -> pd.DataFrame:
    """Return the most recent N bars for `symbol`, ascending by trade_date."""
    sql = f"""
        SELECT {_DF_SELECT} FROM (
            SELECT {_DF_SELECT}
              FROM market_data
             WHERE symbol = %s
             ORDER BY trade_date DESC
             LIMIT %s
        ) t
        ORDER BY trade_date ASC
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol, n))
        rows = cur.fetchall()
    return _rows_to_df(rows)


def get_latest_date(symbol: str) -> Optional[date]:
    """Return the most recent trade_date in market_data for `symbol`, or None."""
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(trade_date) FROM market_data WHERE symbol = %s",
            (symbol,),
        )
        row = cur.fetchone()
    return row[0] if row and row[0] is not None else None


def _rows_to_df(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(_DF_COLUMNS))
    df = pd.DataFrame(rows, columns=list(_DF_COLUMNS))
    # NUMERIC comes back as Decimal; cast to float for downstream math.
    for col in _FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    return df
