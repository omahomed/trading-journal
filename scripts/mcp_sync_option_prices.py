#!/usr/bin/env python3
"""Update trades_summary.manual_price for open options from broker MCP quotes.

Broker MCPs (Robinhood at claude.ai Robinhood, IBKR at claude.ai
Interactive Brokers IBKR) are session-scoped — they only respond in an
interactive Claude Code session, not from a cron or the FastAPI backend.
Workflow: in-session, Claude calls the MCP's option-positions +
option-quotes tools, normalizes the result to the shape below, then
pipes it here to do the matching + DB writes.

Normalized input shape (JSON list on stdin OR --input file):
  [
    {
      "underlying":   "AVGO",
      "expiration":   "2026-09-18",     # YYYY-MM-DD
      "strike":       390.00,
      "option_type":  "call" | "put",
      "mark_price":   43.70,             # per-contract dollars
    },
    ...
  ]

Matching: joins to trades_summary WHERE status='OPEN' AND
instrument_type='OPTION' on the app's canonical ticker format
"SYMBOL YYMMDD $STRIKE[.XX]C|P" (see scripts/import_robinhood_csv.py:
encode_option_ticker).

Behavior:
  * Dry-run by default — prints the plan, exits without touching the DB.
  * --commit writes; wraps every UPDATE in a single transaction so the
    whole sweep either lands or rolls back.
  * Positions in the input with no matching app row are reported as
    "unmatched" — safe (no write); operator can investigate.
  * App rows with no matching input position are reported as "stale"
    (not touched here). Their manual_price stays whatever it was.

Usage:
  # Dry-run against Long-Term Growth (piped input):
  cat positions.json | python scripts/mcp_sync_option_prices.py \\
    --portfolio "Long-Term Growth"

  # Commit against CanSlim from a file:
  python scripts/mcp_sync_option_prices.py \\
    --portfolio "CanSlim" \\
    --input canslim_positions.json \\
    --commit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db_layer import get_db_connection  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("option_price_sync")


def encode_option_ticker(underlying: str, expiration: str, opt_type: str, strike: float) -> str:
    """Match scripts/import_robinhood_csv.py:encode_option_ticker exactly.

    Format: 'UNDERLYING YYMMDD $STRIKEC' (or 'P').
      • Strike rendered without a decimal when a whole number ($390C),
        otherwise with one decimal ($22.5C). No trailing zeros.
      • Expiration compacted to YYMMDD.
      • Option type folded to a single uppercase letter.
    """
    exp = datetime.strptime(expiration, "%Y-%m-%d").date()
    exp_str = f"{exp.year % 100:02d}{exp.month:02d}{exp.day:02d}"
    pc = "C" if str(opt_type).lower().startswith("c") else "P"
    if float(strike).is_integer():
        strike_str = f"{int(strike)}"
    else:
        # Robinhood's Description writes fractional strikes with two
        # decimals; the import parser normalizes trailing zeros away.
        # Mirror that: "22.5" not "22.50" or "22.500".
        strike_str = f"{strike:.10f}".rstrip("0").rstrip(".")
    return f"{underlying} {exp_str} ${strike_str}{pc}"


def load_input(args: argparse.Namespace) -> list[dict]:
    """Read normalized positions from --input file or stdin."""
    if args.input:
        with open(args.input) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)
    if not isinstance(data, list):
        raise ValueError("input must be a JSON list of positions")
    return data


def validate_position(p: dict) -> str | None:
    """Return an error string if the position is malformed, else None."""
    required = ("underlying", "expiration", "strike", "option_type", "mark_price")
    for k in required:
        if k not in p:
            return f"missing {k}"
    try:
        datetime.strptime(p["expiration"], "%Y-%m-%d")
    except ValueError:
        return f"bad expiration format: {p['expiration']}"
    try:
        strike = float(p["strike"])
        price = float(p["mark_price"])
    except (TypeError, ValueError):
        return "strike and mark_price must be numeric"
    if strike <= 0 or price < 0:
        return f"strike must be > 0 and mark_price >= 0"
    if str(p["option_type"]).lower()[0] not in ("c", "p"):
        return f"option_type must be call or put; got {p['option_type']!r}"
    return None


def resolve_portfolio_id(cur, portfolio_name: str) -> int:
    cur.execute("SELECT id FROM portfolios WHERE name = %s", (portfolio_name,))
    row = cur.fetchone()
    if not row:
        raise ValueError(f"portfolio not found: {portfolio_name!r}")
    return row[0]


def load_open_option_rows(cur, portfolio_id: int) -> dict[str, dict]:
    """Return {ticker: {trade_id, avg_entry, manual_price, shares}} for
    every open option campaign in the portfolio.

    Ticker is the natural join key — canonical "SYM YYMMDD $STRIKEC/P"
    already stored in the app, matches encode_option_ticker's output.
    """
    cur.execute(
        """
        SELECT ts.trade_id, ts.ticker, ts.avg_entry, ts.manual_price, ts.shares
          FROM trades_summary ts
         WHERE ts.portfolio_id = %s
           AND ts.deleted_at IS NULL
           AND ts.status = 'OPEN'
           AND ts.instrument_type = 'OPTION'
        """,
        (portfolio_id,),
    )
    return {r[1]: {"trade_id": r[0], "avg_entry": r[2],
                   "manual_price": r[3], "shares": r[4]}
            for r in cur.fetchall()}


def build_updates(positions: list[dict], app_rows: dict[str, dict]) -> tuple[list[dict], list[dict], list[str]]:
    """Match input positions to app rows and stage updates.

    Returns (planned_updates, unmatched_positions, stale_tickers)
      * planned_updates: [{trade_id, ticker, new_price, old_price, delta}]
      * unmatched_positions: input positions with no ticker match
      * stale_tickers: app rows that didn't appear in input
    """
    planned = []
    unmatched = []
    matched_tickers = set()
    for p in positions:
        err = validate_position(p)
        if err:
            unmatched.append({**p, "error": err})
            continue
        ticker = encode_option_ticker(
            p["underlying"], p["expiration"], p["option_type"], float(p["strike"])
        )
        row = app_rows.get(ticker)
        if not row:
            unmatched.append({**p, "computed_ticker": ticker, "error": "no matching app campaign"})
            continue
        matched_tickers.add(ticker)
        old_price = float(row["manual_price"]) if row["manual_price"] is not None else None
        new_price = float(p["mark_price"])
        planned.append({
            "trade_id": row["trade_id"],
            "ticker": ticker,
            "new_price": new_price,
            "old_price": old_price,
            "delta": (new_price - old_price) if old_price is not None else None,
        })
    stale = sorted(set(app_rows.keys()) - matched_tickers)
    return planned, unmatched, stale


def print_report(portfolio: str, planned: list[dict], unmatched: list[dict],
                 stale: list[str], committed: bool) -> None:
    print("=" * 64)
    print(f"OPTION PRICE SYNC — {portfolio}")
    print(f"Mode: {'COMMITTED' if committed else 'DRY-RUN'}")
    print("=" * 64)
    print(f"Matched updates: {len(planned)}")
    for p in planned:
        old = "—" if p["old_price"] is None else f"${p['old_price']:.4f}"
        delta = "" if p["delta"] is None else f"  Δ={p['delta']:+.4f}"
        print(f"  {p['trade_id']:11s} {p['ticker']:22s}  {old:>10s} → ${p['new_price']:.4f}{delta}")
    if unmatched:
        print(f"\nUnmatched input positions: {len(unmatched)}")
        for u in unmatched:
            print(f"  {u.get('computed_ticker', '?'):22s} — {u.get('error', '?')}")
    if stale:
        print(f"\nApp options with no input position (untouched): {len(stale)}")
        for s in stale:
            print(f"  {s}")


def run(args: argparse.Namespace) -> int:
    positions = load_input(args)
    with get_db_connection() as conn, conn.cursor() as cur:
        portfolio_id = resolve_portfolio_id(cur, args.portfolio)
        app_rows = load_open_option_rows(cur, portfolio_id)
        planned, unmatched, stale = build_updates(positions, app_rows)

        if args.commit and planned:
            for p in planned:
                cur.execute(
                    """
                    UPDATE trades_summary
                       SET manual_price = %s,
                           manual_price_set_at = NOW(),
                           last_updated = NOW()
                     WHERE portfolio_id = %s
                       AND trade_id = %s
                       AND deleted_at IS NULL
                    """,
                    (p["new_price"], portfolio_id, p["trade_id"]),
                )
            conn.commit()
        elif not args.commit:
            conn.rollback()  # be explicit — no-op unless earlier UPDATEs ran

    print_report(args.portfolio, planned, unmatched, stale, args.commit)
    return 0 if not unmatched else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--portfolio", required=True,
                   help="Target portfolio name (e.g. 'Long-Term Growth', 'CanSlim')")
    p.add_argument("--input", default=None,
                   help="Path to normalized JSON input; stdin if omitted")
    p.add_argument("--commit", action="store_true",
                   help="Write the manual_price updates. Without this, dry-run only.")
    return p


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
