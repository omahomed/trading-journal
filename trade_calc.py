"""Pure trade calculation helpers.

Extracted from api/main.py so the core accounting logic (LIFO, risk budgeting,
journal column normalization) can be unit-tested without loading the FastAPI
app or touching the database. Handlers in api/main.py import these and wrap
them with DB I/O + validation.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd


_OPTION_TICKER_RE = re.compile(r"^\S+\s+\d{6}\s+\$[0-9.]+(C|P)$")


def is_option_ticker(ticker: str | None) -> bool:
    """True if ticker matches the readable option format `SYMBOL YYMMDD $STRIKE C|P`.

    Matches the same shape api/main.py uses to route price lookups through
    the OCC encoder. Used as a *fallback* for legacy rows that pre-date the
    instrument_type column; new writes should set the column explicitly.
    """
    return bool(ticker and _OPTION_TICKER_RE.match(ticker.strip()))


def multiplier_for_ticker(ticker: str | None) -> float:
    """Standard contract multiplier inferred from ticker shape. 100 for equity
    options, 1 for stocks. Mini options / futures options would override this
    by passing an explicit multiplier on the trade row.
    """
    return 100.0 if is_option_ticker(ticker) else 1.0


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


def calc_risk_budget(
    shares: float, entry: float, stop_loss: float, multiplier: float = 1.0
) -> float:
    """Initial $ at risk: shares * (entry - stop_loss) * multiplier, floored at 0.

    Returns 0 when the stop is missing, non-positive, or at/above entry — the
    "no stop" signal downstream risk views rely on to distinguish unsized
    entries from truly zero-risk ones. Multiplier defaults to 1 (stocks); pass
    100 for equity options so dollar risk reflects notional, not premium-per-
    share.

    NOTE: Group 7-1 reframes risk_budget as a *derived* field recomputed on
    every state change via compute_trade_risk. This single-lot helper remains
    for log_buy's initial inline write (where it equals compute_trade_risk
    for the single-BUY case), but the canonical source of trades_summary.
    risk_budget is now compute_trade_risk called from _recompute_summary_lifo.
    """
    if not (stop_loss and stop_loss > 0 and entry > stop_loss and shares > 0):
        return 0.0
    return round(shares * (entry - stop_loss) * multiplier, 2)


def compute_trade_risk(
    txns: pd.DataFrame, multiplier: float = 1.0
) -> float:
    """Holistic Trade Risk $ for a campaign — sum of open-lot exposure.

    Instrument-aware. The formula depends on `multiplier`:

      STOCKS (multiplier == 1):
        Σ over open BUY-lot remainders of
          lot_shares × max(0, lot_entry − lot_stop) × multiplier
        Lots with no stop contribute 0 (legacy stock "unsized" convention).

      OPTIONS (multiplier > 1):
        Σ over open BUY-lot remainders of
          lot_shares × lot_entry × multiplier
        i.e. premium paid (cost). Stops on options do NOT drive Trade Risk $
        math — Group 7-3 policy: max loss on a long option is the premium,
        regardless of whether a stop was set. Decorative "50%" stops from
        the prior practice are ignored here.

    Walks the same LIFO inventory algorithm as compute_lifo_summary, then
    applies the per-instrument rule above to whatever remains. Returns 0
    when no inventory remains (fully closed campaign) or when the txns
    DataFrame is empty.

    Independent of any prior stored value — every call reads only the
    current detail rows and produces the answer fresh. This is the property
    that fixes the Group 7 additive scale-in bug and the frozen-after-sell
    bug: callers no longer need to know the formula or carry forward state.

    Expects a DataFrame with columns: date, action (BUY/SELL), shares,
    amount (price per share), stop_loss (per-row stop). Column names must
    already be normalized (snake_case).

    Multiplier scales the result (1 for stocks, 100 for equity options).
    """
    if txns.empty:
        return 0.0

    txns = txns.copy()
    txns["date"] = pd.to_datetime(txns["date"], errors="coerce")
    txns = txns.dropna(subset=["date"]).sort_values("date")
    if txns.empty:
        return 0.0

    # Phase 2 B-1: per-SELL matching. Trade Risk $ depends on what
    # remains in inventory after each sell, so the HCFO branch shifts
    # the residue toward lower-cost lots. arrival_seq + date on each
    # lot let the LIFO branch sort explicitly post-HCFO mutation
    # instead of relying on the inventory[-1] adjacency invariant.
    inventory: list[dict[str, Any]] = []
    arrival_seq = 0
    for _, tx in txns.iterrows():
        action = str(tx.get("action", "")).upper()
        tx_shares = float(tx.get("shares", 0) or 0)
        tx_price = float(tx.get("amount", 0) or 0)
        tx_stop = float(tx.get("stop_loss", 0) or 0)
        tx_date = tx.get("date")
        if action == "BUY":
            inventory.append({
                "price": tx_price, "shares": tx_shares, "stop": tx_stop,
                "date": tx_date, "arrival_seq": arrival_seq,
            })
            arrival_seq += 1
        elif action == "SELL":
            # NaN-safe: pandas fills missing match_method with NaN in mixed
            # fixtures, and bool(NaN) is True, so `or` doesn't catch it.
            raw_method = tx.get("match_method")
            sell_method = (
                "LIFO" if pd.isna(raw_method) or not raw_method
                else str(raw_method).upper()
            )
            if sell_method == "HCFO":
                inventory.sort(key=lambda lot: (-lot["price"], lot["date"]))
            else:  # LIFO
                inventory.sort(key=lambda lot: -lot["arrival_seq"])
            to_sell = tx_shares
            while to_sell > 0 and inventory:
                lot = inventory[0]
                take = min(to_sell, lot["shares"])
                lot["shares"] -= take
                to_sell -= take
                if lot["shares"] < 0.0001:
                    inventory.pop(0)

    total_risk = 0.0
    for lot in inventory:
        if lot["shares"] <= 0:
            continue
        if multiplier > 1:
            # Option-like instrument: Trade Risk $ = premium paid (cost).
            # Stops on options do not drive Trade Risk $ math (Group 7-3
            # policy: max loss on a long option is the premium).
            total_risk += lot["shares"] * lot["price"] * multiplier
        elif lot["stop"] > 0 and lot["price"] > lot["stop"]:
            # Stock with a real stop: distance-to-stop formula, floored at 0.
            total_risk += lot["shares"] * (lot["price"] - lot["stop"]) * multiplier
        # else: stock with no stop set → contributes 0 (legacy "unsized"
        # convention; preserved for backward compatibility with calc_risk_budget).

    return round(total_risk, 2)


def compute_lifo_summary(
    txns: pd.DataFrame,
    trade_id: str,
    ticker: str,
    fallback_open_date: str = "",
    multiplier: float = 1.0,
    with_closures: bool = False,
) -> dict[str, Any] | None | tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Fold a campaign's BUY/SELL transactions into a summary row via LIFO.

    Expects a DataFrame with columns: date, action (BUY/SELL), shares, amount
    (price per share). Column names must already be normalized (snake_case).
    Returns None when no valid transactions exist — caller should interpret
    that as "delete this summary entirely."

    Multiplier scales every dollar amount returned (total_cost, realized_pl).
    Avg_entry / avg_exit / shares stay in per-contract units. Return_pct is
    invariant under multiplier (it cancels in the ratio).

    When `with_closures=True` (opt-in for the lot_closures persistence path),
    returns a `(summary, closures)` tuple instead of just `summary`. Each
    closure is one BUY × SELL pairing produced inside the LIFO walk, with
    keys: sell_trx_id, buy_trx_id, shares, buy_price, sell_price, multiplier,
    realized_pl, closed_at. Summary may still be None for empty/all-null-date
    inputs; closures is `[]` in that case. The flag is opt-in so existing
    callers (and tests) keep the simpler return shape unchanged.
    """
    if txns.empty:
        return (None, []) if with_closures else None

    txns = txns.copy()
    txns["date"] = pd.to_datetime(txns["date"], errors="coerce")
    txns = txns.dropna(subset=["date"]).sort_values("date")
    if txns.empty:
        return (None, []) if with_closures else None

    # Phase 2 B-1: per-SELL matching. Each SELL reads its own match_method
    # stamp. NULL → LIFO by historical convention (migration 041 backfilled
    # every existing SELL to 'LIFO'). arrival_seq + date on each lot let
    # the LIFO branch sort explicitly post-HCFO mutation, so the inventory
    # need not stay in arrival order across the walk.
    inventory: list[dict[str, Any]] = []
    closures: list[dict[str, Any]] = []
    total_realized = 0.0
    arrival_seq = 0
    for _, tx in txns.iterrows():
        action = str(tx.get("action", "")).upper()
        tx_shares = float(tx.get("shares", 0) or 0)
        tx_price = float(tx.get("amount", 0) or 0)
        tx_trx_id = str(tx.get("trx_id", "") or "")
        if action == "BUY":
            inventory.append({
                "price": tx_price, "shares": tx_shares, "trx_id": tx_trx_id,
                "date": tx.get("date"), "arrival_seq": arrival_seq,
            })
            arrival_seq += 1
        elif action == "SELL":
            # NaN-safe per-SELL switch (see compute_trade_risk for rationale).
            raw_method = tx.get("match_method")
            sell_method = (
                "LIFO" if pd.isna(raw_method) or not raw_method
                else str(raw_method).upper()
            )
            if sell_method == "HCFO":
                inventory.sort(key=lambda lot: (-lot["price"], lot["date"]))
            else:  # LIFO
                inventory.sort(key=lambda lot: -lot["arrival_seq"])
            to_sell = tx_shares
            sell_date = tx.get("date")
            while to_sell > 0 and inventory:
                lot = inventory[0]
                take = min(to_sell, lot["shares"])
                pair_pl = (tx_price - lot["price"]) * take
                total_realized += pair_pl
                closures.append({
                    "sell_trx_id": tx_trx_id,
                    "buy_trx_id": str(lot.get("trx_id", "") or ""),
                    "shares": float(take),
                    "buy_price": float(lot["price"]),
                    "sell_price": float(tx_price),
                    "multiplier": float(multiplier),
                    "realized_pl": float(pair_pl * multiplier),
                    "closed_at": sell_date,
                })
                lot["shares"] -= take
                to_sell -= take
                if lot["shares"] < 0.0001:
                    inventory.pop(0)

    remaining_shares = sum(lot["shares"] for lot in inventory)
    remaining_cost = sum(lot["shares"] * lot["price"] for lot in inventory)

    sells = txns[txns["action"].str.upper() == "SELL"]
    total_sell_val = float((sells["shares"].astype(float) * sells["amount"].astype(float)).sum())
    total_sell_shs = float(sells["shares"].astype(float).sum())
    avg_exit = total_sell_val / total_sell_shs if total_sell_shs > 0 else 0.0

    buys = txns[txns["action"].str.upper() == "BUY"]
    total_cost_all = float((buys["shares"].astype(float) * buys["amount"].astype(float)).sum())
    total_buy_shs = float(buys["shares"].astype(float).sum())
    # Open trade: average over remaining inventory (LIFO post-sell). Closed
    # trade: fall back to the volume-weighted buy average so the campaign
    # face card keeps its entry price after the position is gone.
    if remaining_shares > 0:
        avg_entry = remaining_cost / remaining_shares
    elif total_buy_shs > 0:
        avg_entry = total_cost_all / total_buy_shs
    else:
        avg_entry = 0.0
    is_closed = remaining_shares < 0.01 and total_sell_shs > 0
    # Return % is multiplier-invariant — it's a ratio of two notionals — so we
    # apply multiplier only to absolute dollars (Total_Cost, Realized_PL).
    return_pct = (total_realized / total_cost_all * 100) if is_closed and total_cost_all > 0 else 0.0

    first_date = txns["date"].min()
    open_date = first_date.strftime("%Y-%m-%d") if pd.notna(first_date) else (fallback_open_date or "")
    last_date = txns["date"].max()
    closed_date = last_date.strftime("%Y-%m-%d") if is_closed and pd.notna(last_date) else None

    cost_to_report = remaining_cost if not is_closed else total_cost_all
    summary = {
        "Trade_ID": trade_id, "Ticker": ticker,
        "Status": "CLOSED" if is_closed else "OPEN",
        "Open_Date": open_date,
        "Closed_Date": closed_date,
        "Shares": float(remaining_shares if not is_closed else total_buy_shs),
        "Avg_Entry": float(round(avg_entry, 4)),
        "Avg_Exit": float(round(avg_exit, 4)) if avg_exit > 0 else 0.0,
        "Total_Cost": float(round(cost_to_report * multiplier, 2)),
        "Realized_PL": float(round(total_realized * multiplier, 2)),
        "Return_Pct": float(round(return_pct, 4)),
    }
    return (summary, closures) if with_closures else summary


def validate_post_edit_lifo(
    txns_for_trade: pd.DataFrame,
    detail_id: int,
    proposed_action: str,
    proposed_shares: float,
    proposed_amount: float,
    proposed_date: str,
) -> str | None:
    """Detect LIFO-breaking edits BEFORE the underlying detail UPDATE/DELETE
    commits. Returns an error message when the proposed edit would leave SELL
    shares unmatched (gross under-allocation OR chronological mismatch where a
    SELL precedes its only supporting BUY); returns None if the edit is safe.

    Why this exists: the recompute path silently drops unmatched SELLs
    (compute_lifo_summary's `while to_sell > 0 and inventory:` loop just exits
    when inventory empties), which produces undetectable wrong realized_pl on
    the campaign card. Six production trades carried bad data because of
    edits that pre-dated this guard.

    `proposed_action` is "BUY" / "SELL" for an edit, or "DELETE" to simulate
    soft-deleting the row. If the caller omits action (empty string), we fall
    back to the existing row's action — Trade Manager and Trade Journal both
    send it today, but this protects against partial-edit callers that would
    otherwise simulate a blank-action row and silently drop the buy's
    inventory contribution. `txns_for_trade` is the current detail rows for
    the campaign (already filtered to one trade_id). For "DELETE", the other
    proposed_* args are ignored.
    """
    if txns_for_trade.empty or "detail_id" not in txns_for_trade.columns:
        return None

    txns = txns_for_trade.copy()
    if proposed_action == "DELETE":
        txns = txns[txns["detail_id"] != detail_id]
    else:
        mask = txns["detail_id"] == detail_id
        if mask.any():
            existing_action = ""
            if "action" in txns.columns:
                existing_action = str(txns.loc[mask, "action"].iloc[0] or "")
            effective_action = proposed_action or existing_action
            txns.loc[mask, "action"] = effective_action
            txns.loc[mask, "shares"] = float(proposed_shares)
            txns.loc[mask, "amount"] = float(proposed_amount)
            if proposed_date:
                txns.loc[mask, "date"] = proposed_date

    if txns.empty:
        return None  # Whole campaign goes away — caller cleans up summary.

    result = compute_lifo_summary(
        txns, trade_id="", ticker="", multiplier=1.0, with_closures=True,
    )
    if result is None:
        return None
    _summary, closures = result

    sells = txns[txns["action"].astype(str).str.upper() == "SELL"]
    if sells.empty:
        return None
    total_sell_shs = float(pd.to_numeric(sells["shares"], errors="coerce").fillna(0).sum())
    matched_sell_shs = sum(float(c.get("shares", 0) or 0) for c in closures)
    if matched_sell_shs + 0.0001 < total_sell_shs:
        unmatched = total_sell_shs - matched_sell_shs
        return (
            f"This edit would leave {unmatched:g} sell shares unmatched by buys. "
            f"Adjust or remove the sells first, or undo the buy deletion."
        )
    return None
