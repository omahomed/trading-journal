#!/usr/bin/env python3
"""Transform Robinhood MCP responses into the parser's expected CSV shape.

The Robinhood MCP (mcp__claude_ai_Robinhood__*) can only be called from
an interactive Claude Code session — it's OAuth-scoped to the human
account and doesn't reach the production backend. The workaround for
"sync my Robinhood to the app": in-session, Claude calls the MCP for
equity + option orders, pipes the JSON through this script to get a CSV
in the exact format the human-downloaded Robinhood transaction-history
export uses, then POSTs that CSV to the same
/api/imports/robinhood/preview + /commit endpoints the manual-download
flow already goes through. Zero new backend, zero new frontend.

Format target (mirrors what read_csv in import_robinhood_csv.py expects):
  "Activity Date","Process Date","Settle Date","Instrument",
  "Description","Trans Code","Quantity","Price","Amount"

Trans Code mapping:
  Equity  side=buy               → "Buy"
  Equity  side=sell              → "Sell"
  Option  side=buy,  effect=open  → "BTO"  (Buy to Open)
  Option  side=sell, effect=close → "STC"  (Sell to Close)
  Option  side=sell, effect=open  → "STO"  (Sell to Open — short; parser
                                            skips + warns; kept in CSV so
                                            the audit trail is complete)
  Option  side=buy,  effect=close → "BTC"  (Buy to Close — closing short;
                                            same handling as STO)

Only orders with state == "filled" are emitted. Cancelled / rejected /
partially-filled orders are skipped — nothing was executed on them.

Usage:
  # Pipe MCP JSON into stdin:
  cat orders.json | scripts/mcp_robinhood_to_csv.py > trades.csv

  # Or two files (equity + options) via flags:
  scripts/mcp_robinhood_to_csv.py \\
    --equity  equity_orders.json \\
    --options option_orders.json \\
    > trades.csv

  # Optional --exclude filters (see docstring on parse_exclusions).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, date
from io import StringIO
from typing import Any


CSV_COLUMNS = [
    "Activity Date", "Process Date", "Settle Date",
    "Instrument", "Description", "Trans Code",
    "Quantity", "Price", "Amount",
]


def _dollar(v: float) -> str:
    """Format a positive money value as '$1,234.56'."""
    return f"${v:,.2f}"


def _paren_dollar(v: float) -> str:
    """Format a negative money value as '($1,234.56)'.

    Robinhood's CSV convention: parens indicate debits (money out).
    The parser's parse_currency handles both styles.
    """
    return f"(${abs(v):,.2f})"


def _fmt_amount(signed_dollars: float) -> str:
    """Sign-aware amount formatter."""
    if signed_dollars < 0:
        return _paren_dollar(signed_dollars)
    return _dollar(signed_dollars)


def _mdy_from_iso(iso: str) -> str:
    """MCP returns ISO timestamps like '2026-07-08T16:52:58.888000Z';
    Robinhood's CSV writes activity dates as 'M/D/YYYY' (no zero pad).
    The parser's parse_date handles both, but we emit the CSV in the
    original shape for maximum fidelity."""
    if not iso:
        return ""
    # Take the date portion only.
    d = iso.split("T", 1)[0]
    try:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        return f"{dt.month}/{dt.day}/{dt.year}"
    except ValueError:
        return d


def _plus_settle(iso: str, offset_days: int = 1) -> str:
    """Approximate settle date as trade date + N business days.

    Robinhood equity T+1 settles as of 2024; options T+1 also. We don't
    bother with real business-day math — the parser ignores Process/Settle
    Date entirely; only Activity Date drives its walk. This exists purely
    to keep the CSV shape faithful to the human-download format for
    debuggability.
    """
    if not iso:
        return ""
    d = iso.split("T", 1)[0]
    try:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        # Simple +offset; skip weekends (not strictly correct but fine for
        # display-only columns).
        step = offset_days
        while step > 0:
            dt = date.fromordinal(dt.toordinal() + 1)
            if dt.weekday() < 5:
                step -= 1
        return f"{dt.month}/{dt.day}/{dt.year}"
    except ValueError:
        return d


def _fmt_option_desc(underlying: str, expiration: str, opt_type: str, strike: float) -> str:
    """Render the option Description Robinhood uses.

    Example: 'GDYN 9/18/2026 Call $5.00'

    - expiration: 'YYYY-MM-DD' from MCP → 'M/D/YYYY'
    - opt_type: 'call' or 'put' → 'Call' or 'Put'
    - strike:   float → '$5.00' (two-decimal, no thousand separators
                needed at typical option strikes)
    """
    try:
        exp = datetime.strptime(expiration, "%Y-%m-%d").date()
        exp_str = f"{exp.month}/{exp.day}/{exp.year}"
    except (ValueError, TypeError):
        exp_str = expiration or ""
    otype = "Call" if str(opt_type or "").lower() == "call" else "Put"
    return f"{underlying} {exp_str} {otype} ${strike:.2f}"


def _option_trans_code(side: str, position_effect: str) -> str:
    """MCP → CSV Trans Code for an option leg."""
    side = str(side or "").lower()
    eff = str(position_effect or "").lower()
    if side == "buy" and eff == "open":
        return "BTO"
    if side == "sell" and eff == "close":
        return "STC"
    if side == "sell" and eff == "open":
        return "STO"
    if side == "buy" and eff == "close":
        return "BTC"
    return ""


def _equity_trans_code(side: str) -> str:
    return "Buy" if str(side or "").lower() == "buy" else "Sell"


def equity_order_to_row(order: dict) -> dict | None:
    """Convert a filled MCP equity order into one CSV row.

    Uses average_price (fill VWAP) and cumulative_quantity so a partial
    fill or a multi-execution order emits ONE aggregated row, not one
    row per execution. The parser then reconstructs campaigns from
    one-row-per-order same as the human download.
    """
    if order.get("state") != "filled":
        return None
    qty = float(order.get("cumulative_quantity") or 0)
    if qty <= 0:
        return None
    price = float(order.get("average_price") or order.get("price") or 0)
    fees = float(order.get("fees") or 0)
    side = order.get("side") or ""
    symbol = order.get("symbol") or ""
    trade_iso = order.get("last_transaction_at") or order.get("created_at") or ""
    # Amount sign: buys are debits (parens), sells are credits (positive).
    signed = -(qty * price + fees) if side == "buy" else (qty * price - fees)
    return {
        "Activity Date": _mdy_from_iso(trade_iso),
        "Process Date":  _mdy_from_iso(trade_iso),
        "Settle Date":   _plus_settle(trade_iso, 1),
        "Instrument":    symbol,
        "Description":   symbol,  # human CSV has company name + CUSIP;
                                  # symbol alone is fine — parser only
                                  # reads Description for option contracts.
        "Trans Code":    _equity_trans_code(side),
        "Quantity":      f"{qty:g}",
        "Price":         _dollar(price),
        "Amount":        _fmt_amount(signed),
    }


def option_order_to_rows(order: dict) -> list[dict]:
    """Convert a filled MCP option order into CSV rows — one per leg.

    Multi-leg orders (spreads) emit one row per leg. Each leg's Trans
    Code encodes the (side, position_effect) pair; the parser groups
    them into option-contract campaigns downstream.
    """
    if order.get("state") != "filled":
        return []
    underlying = order.get("chain_symbol") or ""
    price_per_contract = float(order.get("price") or 0)
    trade_iso = order.get("last_transaction_at") or order.get("created_at") or ""
    trade_date_str = _mdy_from_iso(trade_iso)
    settle_date_str = _plus_settle(trade_iso, 1)
    out: list[dict] = []
    for leg in order.get("legs") or []:
        qty = float(leg.get("ratio_quantity") or 1) * float(order.get("processed_quantity") or 0)
        if qty <= 0:
            continue
        strike = float(leg.get("strike_price") or 0)
        expiration = leg.get("expiration_date") or ""
        opt_type = leg.get("option_type") or ""
        side = leg.get("side") or ""
        effect = leg.get("position_effect") or ""
        trans = _option_trans_code(side, effect)
        if not trans:
            continue
        desc = _fmt_option_desc(underlying, expiration, opt_type, strike)
        # Amount: each contract represents 100 shares, so
        # contract_count * price_per_share * 100.
        multiplier = float(order.get("trade_value_multiplier") or 100)
        gross = qty * price_per_contract * multiplier
        # BTO / BTC (money out) → parens; STC / STO (money in) → positive.
        money_out = trans in ("BTO", "BTC")
        signed = -gross if money_out else gross
        out.append({
            "Activity Date": trade_date_str,
            "Process Date":  trade_date_str,
            "Settle Date":   settle_date_str,
            "Instrument":    underlying,
            "Description":   desc,
            "Trans Code":    trans,
            "Quantity":      f"{qty:g}",
            "Price":         f"${price_per_contract:.2f}",
            "Amount":        _fmt_amount(signed),
        })
    return out


def parse_exclusions(spec: str) -> list[tuple[str, str, str, float]]:
    """Parse --exclude arg: comma-separated (date,ticker,side,quantity) tuples,
    semicolon-separated between tuples.

    Example:
      --exclude "2026-06-26,MRVL,sell,30;2026-06-25,AAPL,buy,10"

    Each tuple: date is YYYY-MM-DD (matches order created_at prefix),
    ticker is uppercase, side is 'buy'/'sell', quantity is a float
    (compared with |a - b| < 1e-4 tolerance to handle Robinhood's
    fractional quantities).
    """
    out = []
    if not spec:
        return out
    for tup in spec.split(";"):
        parts = [p.strip() for p in tup.split(",")]
        if len(parts) != 4:
            continue
        d, tkr, side, qty_s = parts
        try:
            qty = float(qty_s)
        except ValueError:
            continue
        out.append((d, tkr.upper(), side.lower(), qty))
    return out


def _match_exclusion(order: dict, exclusions: list[tuple[str, str, str, float]]) -> bool:
    """Return True if this equity order matches any exclusion."""
    if not exclusions:
        return False
    order_date = (order.get("last_transaction_at") or order.get("created_at") or "").split("T", 1)[0]
    order_tkr = str(order.get("symbol") or "").upper()
    order_side = str(order.get("side") or "").lower()
    order_qty = float(order.get("cumulative_quantity") or 0)
    for d, tkr, side, qty in exclusions:
        if (d == order_date
            and tkr == order_tkr
            and side == order_side
            and abs(order_qty - qty) < 1e-4):
            return True
    return False


def transform(
    equity_orders: list[dict] | None,
    option_orders: list[dict] | None,
    exclusions: list[tuple[str, str, str, float]] | None = None,
) -> str:
    """Full transform: MCP order lists → Robinhood-CSV text.

    Rows are sorted by Activity Date descending (matches the human
    download's newest-first ordering, which the parser normalizes via
    _csv_row_sort_key regardless).
    """
    exclusions = exclusions or []
    rows: list[dict] = []
    for o in equity_orders or []:
        if _match_exclusion(o, exclusions):
            continue
        r = equity_order_to_row(o)
        if r:
            rows.append(r)
    for o in option_orders or []:
        # Exclusion match on options is intentionally not supported here —
        # options don't have a simple (ticker, side, qty) tuple key
        # (they'd need strike + expiration). Filter option dupes manually.
        rows.extend(option_order_to_rows(o))
    # Sort by Activity Date desc (mimics Robinhood export ordering).
    def _sort_key(r: dict) -> tuple:
        try:
            parts = r["Activity Date"].split("/")
            return (-int(parts[2]), -int(parts[0]), -int(parts[1]))
        except (KeyError, ValueError, IndexError):
            return (0, 0, 0)
    rows.sort(key=_sort_key)

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def _load_json_maybe(path: str | None) -> Any:
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def _extract_orders(payload: Any) -> list[dict]:
    """Normalize MCP JSON shape → list of order dicts.

    Accepts either the raw MCP response ({data: {orders: [...]}}), a
    bare list, or a dict with an 'orders' key.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], dict):
            return payload["data"].get("orders") or []
        if "orders" in payload:
            return payload["orders"] or []
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--equity", help="JSON file with equity orders (MCP get_equity_orders shape)")
    ap.add_argument("--options", help="JSON file with option orders (MCP get_option_orders shape)")
    ap.add_argument("--exclude", default="",
                    help='Semicolon-separated equity exclusions: "date,ticker,side,qty" tuples')
    ap.add_argument("--out", default="-", help="Output CSV path; '-' for stdout")
    args = ap.parse_args()

    equity_payload = _load_json_maybe(args.equity)
    option_payload = _load_json_maybe(args.options)

    # If neither file was given, expect combined JSON on stdin.
    if equity_payload is None and option_payload is None:
        stdin_data = json.load(sys.stdin)
        if isinstance(stdin_data, dict):
            equity_payload = stdin_data.get("equity")
            option_payload = stdin_data.get("options")

    equity_orders = _extract_orders(equity_payload)
    option_orders = _extract_orders(option_payload)

    csv_text = transform(
        equity_orders, option_orders,
        exclusions=parse_exclusions(args.exclude),
    )

    if args.out == "-":
        sys.stdout.write(csv_text)
    else:
        with open(args.out, "w") as f:
            f.write(csv_text)
        print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
