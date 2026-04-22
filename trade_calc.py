"""Pure trade calculation helpers.

Extracted from api/main.py so the core accounting logic (LIFO, risk budgeting,
journal column normalization) can be unit-tested without loading the FastAPI
app or touching the database. Handlers in api/main.py import these and wrap
them with DB I/O + validation.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


_JOURNAL_COLUMN_RENAME = {
    "Day": "day", "Status": "status", "Market Window": "market_window",
    "Market Cycle": "market_cycle",
    "> 21e": "above_21ema", "Cash -/+": "cash_change",
    "Beg NLV": "beg_nlv", "End NLV": "end_nlv",
    "Daily $ Change": "daily_dollar_change", "Daily % Change": "daily_pct_change",
    "% Invested": "pct_invested", "SPY": "spy", "Nasdaq": "nasdaq",
    "Market_Notes": "market_notes", "Market_Action": "market_action",
    "Portfolio_Heat": "portfolio_heat", "SPY_ATR": "spy_atr", "Nasdaq_ATR": "nasdaq_atr",
    "Score": "score", "Highlights": "highlights", "Lowlights": "lowlights",
    "Mistakes": "mistakes", "Top_Lesson": "top_lesson",
}


def normalize_journal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename journal columns to snake_case. Unknown columns pass through untouched."""
    return df.rename(columns={k: v for k, v in _JOURNAL_COLUMN_RENAME.items() if k in df.columns})


def calc_risk_budget(shares: float, entry: float, stop_loss: float) -> float:
    """Initial $ at risk: shares * (entry - stop_loss), floored at 0.

    Returns 0 when the stop is missing, non-positive, or at/above entry — the
    "no stop" signal downstream risk views rely on to distinguish unsized
    entries from truly zero-risk ones.
    """
    if not (stop_loss and stop_loss > 0 and entry > stop_loss and shares > 0):
        return 0.0
    return round(shares * (entry - stop_loss), 2)


def compute_lifo_summary(
    txns: pd.DataFrame,
    trade_id: str,
    ticker: str,
    fallback_open_date: str = "",
) -> dict[str, Any] | None:
    """Fold a campaign's BUY/SELL transactions into a summary row via LIFO.

    Expects a DataFrame with columns: date, action (BUY/SELL), shares, amount
    (price per share). Column names must already be normalized (snake_case).
    Returns None when no valid transactions exist — caller should interpret
    that as "delete this summary entirely."
    """
    if txns.empty:
        return None

    txns = txns.copy()
    txns["date"] = pd.to_datetime(txns["date"], errors="coerce")
    txns = txns.dropna(subset=["date"]).sort_values("date")
    if txns.empty:
        return None

    inventory: list[dict[str, float]] = []
    total_realized = 0.0
    for _, tx in txns.iterrows():
        action = str(tx.get("action", "")).upper()
        tx_shares = float(tx.get("shares", 0) or 0)
        tx_price = float(tx.get("amount", 0) or 0)
        if action == "BUY":
            inventory.append({"price": tx_price, "shares": tx_shares})
        elif action == "SELL":
            to_sell = tx_shares
            while to_sell > 0 and inventory:
                last = inventory[-1]
                take = min(to_sell, last["shares"])
                total_realized += (tx_price - last["price"]) * take
                last["shares"] -= take
                to_sell -= take
                if last["shares"] < 0.0001:
                    inventory.pop()

    remaining_shares = sum(lot["shares"] for lot in inventory)
    remaining_cost = sum(lot["shares"] * lot["price"] for lot in inventory)
    avg_entry = remaining_cost / remaining_shares if remaining_shares > 0 else 0.0

    sells = txns[txns["action"].str.upper() == "SELL"]
    total_sell_val = float((sells["shares"].astype(float) * sells["amount"].astype(float)).sum())
    total_sell_shs = float(sells["shares"].astype(float).sum())
    avg_exit = total_sell_val / total_sell_shs if total_sell_shs > 0 else 0.0

    buys = txns[txns["action"].str.upper() == "BUY"]
    total_cost_all = float((buys["shares"].astype(float) * buys["amount"].astype(float)).sum())
    total_buy_shs = float(buys["shares"].astype(float).sum())
    is_closed = remaining_shares < 0.01 and total_sell_shs > 0
    return_pct = (total_realized / total_cost_all * 100) if is_closed and total_cost_all > 0 else 0.0

    first_date = txns["date"].min()
    open_date = first_date.strftime("%Y-%m-%d") if pd.notna(first_date) else (fallback_open_date or "")
    last_date = txns["date"].max()
    closed_date = last_date.strftime("%Y-%m-%d") if is_closed and pd.notna(last_date) else None

    return {
        "Trade_ID": trade_id, "Ticker": ticker,
        "Status": "CLOSED" if is_closed else "OPEN",
        "Open_Date": open_date,
        "Closed_Date": closed_date,
        "Shares": float(remaining_shares if not is_closed else total_buy_shs),
        "Avg_Entry": float(round(avg_entry, 4)),
        "Avg_Exit": float(round(avg_exit, 4)) if avg_exit > 0 else 0.0,
        "Total_Cost": float(round(remaining_cost if not is_closed else total_cost_all, 2)),
        "Realized_PL": float(round(total_realized, 2)),
        "Return_Pct": float(round(return_pct, 4)),
    }
