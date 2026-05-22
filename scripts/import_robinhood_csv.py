#!/usr/bin/env python3
"""One-shot Robinhood CSV importer.

Parses a Robinhood activity-history export and writes trades_summary +
trades_details + cash_transactions into a target portfolio. Defaults to
dry-run; pass --commit to actually write.

Design highlights:
  - Pure-function parsers for currency / date / option Description /
    quantity, exhaustively unit-tested.
  - Classifier routes each row by Trans Code into one of: equity_trade,
    option_trade, option_expire, cash_deposit, income, fee, empty,
    unknown.
  - Campaign reconstruction walks rows chronologically per
    ticker / option contract, grouping into trades_summary campaigns
    using the "extend until 0, then new campaign on next buy" rule.
  - DB writer uses raw psycopg2 + db_layer.get_db_connection() so the
    script doesn't drag in api.main (FastAPI / Sentry init).
  - Summary fields (status / shares / avg_entry / realized_pl) computed
    via trade_calc.compute_lifo_summary — same LIFO walk the app uses.

Cash source mapping (within existing CHECK constraint
{deposit,withdraw,buy,sell,reconcile}):
  - Deposits (ACH, RTP, DCF, ITRF) → source='deposit', note='<code>: ...'
    ITRF (internal account transfer) is treated identically to ACH;
    only the note prefix differs ('ITRF: Transfer from Brokerage to
    Joint' etc.) for traceability.
  - Income (CDIV, MDIV, INT)       → SKIPPED (counted in report, not
    imported). Robinhood interest + dividend rows are accounting
    artifacts that don't belong in the trading-journal cash ledger;
    keeping them out also avoids the LTD-return-denominator pollution
    that 'deposit' as a workaround would cause.
  - Short options (STO, BTC)       → SKIPPED, loud-warned. The journal
    schema models long-only equity + long-option campaigns; short
    options need manual handling after import.
  - Fees (GOLD, GDBP, MINT)        → SKIPPED (counted in report).
  - Buy / Sell                     → cash entry emitted automatically
    by the detail-row insert path (source='buy'/'sell').

Usage:
    python scripts/import_robinhood_csv.py \\
      --csv path/to/robinhood.csv \\
      --portfolio "Long-Term Growth" \\
      --since 2026-01-01

    # Dry-run with the 2024-25 sample CSV (bypasses date cutoff):
    python scripts/import_robinhood_csv.py \\
      --csv sample.csv --portfolio "Long-Term Growth" \\
      --allow-pre-cutoff

    # Real import (one-shot):
    python scripts/import_robinhood_csv.py \\
      --csv real.csv --portfolio "Long-Term Growth" \\
      --reset-cash-ledger --commit
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db_layer import get_db_connection  # noqa: E402
from trade_calc import compute_lifo_summary  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("robinhood_import")


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function parsers
# ─────────────────────────────────────────────────────────────────────────────

_CURRENCY_RE = re.compile(r"[\$,]")


def parse_currency(s: str) -> float:
    """'$682.32' → 682.32, '($2,729.30)' → -2729.30, '' → 0.0.

    Robinhood uses parentheses for debits / negative amounts. Strip the
    dollar sign and commas, then handle the parens explicitly.
    """
    if s is None:
        return 0.0
    s = str(s).strip()
    if not s:
        return 0.0
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]
    s = _CURRENCY_RE.sub("", s).strip()
    if not s:
        return 0.0
    try:
        v = float(s)
    except ValueError:
        return 0.0
    return -v if negative else v


def parse_date(s: str) -> date | None:
    """'2/10/2025' → date(2025, 2, 10), '5/21/26' → date(2026, 5, 21).

    Accepts M/D/YYYY (Robinhood's older 4-digit-year format), M/D/YY
    (the format on real 2026-era exports), and ISO YYYY-MM-DD. Two-
    digit years follow Python's default %y mapping: 00–68 → 2000s,
    69–99 → 1900s.
    """
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


_OPTION_DESC_RE = re.compile(
    r"^(?P<underlying>\S+)\s+"
    r"(?P<exp>\d{1,2}/\d{1,2}/\d{2,4})\s+"
    r"(?P<type>Call|Put)\s+"
    r"\$(?P<strike>[0-9]+(?:\.[0-9]+)?)\s*$",
    re.IGNORECASE,
)


def parse_option_description(desc: str) -> dict | None:
    """'PLTR 1/10/2025 Call $67.00' → {underlying, expiration, option_type, strike}.

    Returns None if the description isn't an option (e.g., equity row).
    """
    if not desc:
        return None
    m = _OPTION_DESC_RE.match(desc.strip())
    if not m:
        return None
    exp = parse_date(m.group("exp"))
    if exp is None:
        return None
    return {
        "underlying": m.group("underlying").upper(),
        "expiration": exp,
        "option_type": m.group("type").capitalize(),
        "strike": float(m.group("strike")),
    }


def parse_quantity(s: str) -> float:
    """'3' → 3.0, '3S' → 3.0 (strip trailing letters seen on OEXP rows),
    '0.5' → 0.5 (fractional shares Robinhood does), '' → 0.0."""
    if s is None:
        return 0.0
    s = str(s).strip()
    if not s:
        return 0.0
    # Strip any trailing non-digit-non-period characters (e.g., '3S' → '3').
    s = re.sub(r"[^\d.\-]+$", "", s)
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def encode_option_ticker(underlying: str, exp: date, opt_type: str, strike: float) -> str:
    """Encode option components in the app's readable storage format:
    `{UNDERLYING} {YYMMDD} ${STRIKE}{C|P}` e.g. 'PLTR 250110 $67C'.

    Strike is rendered without trailing zeros — '$67' not '$67.0' when
    integer, '$67.5' when fractional. The _to_occ_symbol regex at
    api/main.py:1468 accepts both forms via `\\$([0-9.]+)`.
    """
    yymmdd = exp.strftime("%y%m%d")
    cp = "C" if str(opt_type).lower().startswith("c") else "P"
    if strike == int(strike):
        strike_s = str(int(strike))
    else:
        strike_s = f"{strike:.2f}".rstrip("0").rstrip(".")
    return f"{underlying.upper()} {yymmdd} ${strike_s}{cp}"


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

CASH_DEPOSIT_CODES = {"ACH", "RTP", "DCF", "ITRF"}
INCOME_CODES = {"CDIV", "MDIV", "INT"}
FEE_CODES = {"GOLD", "GDBP", "MINT"}
EQUITY_TRADE_CODES = {"Buy", "Sell"}
OPTION_TRADE_CODES = {"BTO", "STC"}
OPTION_EXPIRE_CODES = {"OEXP"}
SHORT_OPTION_CODES = {"STO", "BTC"}


def classify(row: dict) -> str:
    """Route a CSV row to a category. 'unknown' for anything we haven't
    seen — surfaced in the verification report so we can extend later.

    Categories that get inserted: equity_trade, option_trade,
    option_expire, cash_deposit.
    Categories that are counted but skipped: income, fee,
    short_option_trade (the last is loud-skipped in warnings so the
    user knows to handle short positions manually).
    """
    code = (row.get("Trans Code") or "").strip()
    if not code and not any(row.values()):
        return "empty"
    if not code:
        return "empty"
    if code in EQUITY_TRADE_CODES:
        return "equity_trade"
    if code in OPTION_TRADE_CODES:
        return "option_trade"
    if code in OPTION_EXPIRE_CODES:
        return "option_expire"
    if code in SHORT_OPTION_CODES:
        return "short_option_trade"
    if code in CASH_DEPOSIT_CODES:
        return "cash_deposit"
    if code in INCOME_CODES:
        return "income"
    if code in FEE_CODES:
        return "fee"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# CSV reader
# ─────────────────────────────────────────────────────────────────────────────


def read_csv(path: Path) -> list[dict]:
    """Read a Robinhood activity CSV. Python's csv.DictReader handles
    quoted-newline fields (CUSIPs sometimes wrap a description line)."""
    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Strip surrounding whitespace on all keys/values for resilience.
            rows.append({(k or "").strip(): (v or "").strip() for k, v in r.items()})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Campaign reconstruction
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TxRecord:
    """One BUY or SELL transaction inside a campaign."""
    action: str        # 'BUY' or 'SELL'
    date: date
    shares: float
    price: float        # per share
    value: float        # signed total (positive for SELL credit, positive for BUY debit-magnitude)
    raw_row: dict       # original CSV row, for debugging


@dataclass
class Campaign:
    """A reconstructed trade campaign: one ticker (or option contract),
    contiguous buys + sells until shares reach zero."""
    ticker: str
    instrument_type: str   # 'STOCK' or 'OPTION'
    multiplier: float      # 1 for stock, 100 for option
    open_date: date
    txns: list[TxRecord] = field(default_factory=list)
    # Option-only metadata (None for stock):
    option_meta: dict | None = None

    @property
    def shares_remaining(self) -> float:
        s = 0.0
        for t in self.txns:
            s += t.shares if t.action == "BUY" else -t.shares
        return s

    @property
    def status(self) -> str:
        return "CLOSED" if abs(self.shares_remaining) < 1e-6 else "OPEN"

    def add_buy(self, tx: TxRecord) -> None:
        self.txns.append(tx)

    def add_sell(self, tx: TxRecord) -> None:
        self.txns.append(tx)


def _csv_row_sort_key(row: dict, original_index: int) -> tuple:
    """Sort by Activity Date ASC, then original CSV row order as a proxy
    for intraday sequencing (Robinhood doesn't provide a timestamp).
    Robinhood's default CSV is reverse-chronological, so we reverse the
    row index to approximate "earlier first" within a day."""
    d = parse_date(row.get("Activity Date") or row.get("Settle Date") or "")
    if d is None:
        d = date.min
    return (d, -original_index)


def reconstruct_equity_campaigns(
    rows: list[dict],
    warnings_out: list[str],
) -> list[Campaign]:
    """Group equity Buy/Sell rows into campaigns.

    Rule: each Buy extends the current open campaign for that ticker.
    Each Sell consumes shares from the current open. When shares reach
    0, close the campaign; next Buy starts a new one.

    Sells without an open campaign are logged as warnings and skipped
    — the source CSV doesn't cover the full position history.
    """
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda ix_r: _csv_row_sort_key(ix_r[1], ix_r[0]))

    campaigns: list[Campaign] = []
    open_by_ticker: dict[str, Campaign] = {}

    for _orig_idx, row in indexed:
        if classify(row) != "equity_trade":
            continue
        ticker = (row.get("Instrument") or "").strip().upper()
        if not ticker:
            warnings_out.append(f"Equity row with empty Instrument: {row}")
            continue
        d = parse_date(row.get("Activity Date") or "")
        if d is None:
            warnings_out.append(f"Equity row with unparseable date: {row}")
            continue
        shares = parse_quantity(row.get("Quantity") or "")
        price = parse_currency(row.get("Price") or "")
        value = abs(parse_currency(row.get("Amount") or ""))
        code = (row.get("Trans Code") or "").strip()

        if code == "Buy":
            camp = open_by_ticker.get(ticker)
            if camp is None:
                camp = Campaign(ticker=ticker, instrument_type="STOCK",
                                multiplier=1.0, open_date=d)
                open_by_ticker[ticker] = camp
            camp.add_buy(TxRecord(action="BUY", date=d, shares=shares,
                                  price=price, value=value, raw_row=row))
        else:  # 'Sell'
            camp = open_by_ticker.get(ticker)
            if camp is None:
                warnings_out.append(
                    f"SELL without prior BUY for {ticker} on {d.isoformat()} — skipped"
                )
                continue
            camp.add_sell(TxRecord(action="SELL", date=d, shares=shares,
                                   price=price, value=value, raw_row=row))
            if camp.shares_remaining <= 1e-6:
                campaigns.append(camp)
                del open_by_ticker[ticker]

    campaigns.extend(open_by_ticker.values())
    return campaigns


def reconstruct_option_campaigns(
    rows: list[dict],
    warnings_out: list[str],
) -> list[Campaign]:
    """Group option BTO/STC/OEXP rows into campaigns.

    Key: (underlying, expiration, strike, option_type). Each option
    contract is a distinct instrument. OEXP rows attach as $0 SELLs
    that close the position to a full premium loss (assuming the
    contract expired worthless — which OEXP implies).
    """
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda ix_r: _csv_row_sort_key(ix_r[1], ix_r[0]))

    campaigns: list[Campaign] = []
    open_by_key: dict[tuple, Campaign] = {}

    for _orig_idx, row in indexed:
        category = classify(row)
        if category not in ("option_trade", "option_expire"):
            continue

        opt = parse_option_description(row.get("Description") or "")
        if opt is None:
            warnings_out.append(
                f"Option row with unparseable Description: {row.get('Description')!r}"
            )
            continue

        instrument_ticker = (row.get("Instrument") or "").strip().upper() or opt["underlying"]
        opt_ticker = encode_option_ticker(
            instrument_ticker, opt["expiration"], opt["option_type"], opt["strike"]
        )
        key = (instrument_ticker, opt["expiration"], opt["strike"],
               opt["option_type"].lower())

        d = parse_date(row.get("Activity Date") or "")
        if d is None:
            warnings_out.append(f"Option row with unparseable date: {row}")
            continue

        shares = parse_quantity(row.get("Quantity") or "")
        price = parse_currency(row.get("Price") or "")
        value = abs(parse_currency(row.get("Amount") or ""))
        code = (row.get("Trans Code") or "").strip()

        if code == "BTO":
            camp = open_by_key.get(key)
            if camp is None:
                camp = Campaign(ticker=opt_ticker, instrument_type="OPTION",
                                multiplier=100.0, open_date=d, option_meta=opt)
                open_by_key[key] = camp
            camp.add_buy(TxRecord(action="BUY", date=d, shares=shares,
                                  price=price, value=value, raw_row=row))
        elif code == "STC":
            camp = open_by_key.get(key)
            if camp is None:
                warnings_out.append(
                    f"STC without prior BTO for {opt_ticker} on {d.isoformat()} — skipped"
                )
                continue
            camp.add_sell(TxRecord(action="SELL", date=d, shares=shares,
                                   price=price, value=value, raw_row=row))
            if camp.shares_remaining <= 1e-6:
                campaigns.append(camp)
                del open_by_key[key]
        elif code == "OEXP":
            camp = open_by_key.get(key)
            if camp is None:
                warnings_out.append(
                    f"OEXP without open BTO for {opt_ticker} on {d.isoformat()} — skipped"
                )
                continue
            # Expiration → SELL at $0, full premium loss.
            camp.add_sell(TxRecord(action="SELL", date=d, shares=shares,
                                   price=0.0, value=0.0, raw_row=row))
            if camp.shares_remaining <= 1e-6:
                campaigns.append(camp)
                del open_by_key[key]

    campaigns.extend(open_by_key.values())
    return campaigns


# ─────────────────────────────────────────────────────────────────────────────
# Cash transaction extraction
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CashRecord:
    date: date
    amount: float            # signed
    source: str              # 'deposit' (within schema CHECK)
    note: str                # 'ACH: ...', 'ITRF: ...', etc.
    raw_row: dict


def extract_cash_rows(rows: list[dict], warnings_out: list[str]) -> list[CashRecord]:
    """Pull every cash-deposit row into a CashRecord list.

    Buy/Sell cash rows are emitted later automatically by the detail-row
    insert path (see _emit_trade_cash_tx in db_layer). Income rows
    (CDIV/MDIV/INT), fee rows, and short-option rows are intentionally
    NOT included — they're counted in the verification report instead.
    """
    out: list[CashRecord] = []
    for row in rows:
        if classify(row) != "cash_deposit":
            continue
        d = parse_date(row.get("Activity Date") or "")
        if d is None:
            warnings_out.append(f"Cash row with unparseable date: {row}")
            continue
        amount = parse_currency(row.get("Amount") or "")
        code = (row.get("Trans Code") or "").strip()
        description = row.get("Description") or "External transfer"
        note = f"{code}: {description}"

        out.append(CashRecord(date=d, amount=amount, source="deposit",
                              note=note, raw_row=row))
    return out


def collect_short_option_warnings(rows: list[dict]) -> list[str]:
    """Group STO/BTC rows by contract (underlying, exp, strike, type)
    and produce one warning per contract showing aggregated counts.
    The user uses this list to know what to manually replay after the
    import."""
    by_contract: dict[tuple, dict[str, float]] = defaultdict(
        lambda: {"STO": 0.0, "BTC": 0.0, "desc": ""}
    )
    total_rows = 0
    for row in rows:
        if classify(row) != "short_option_trade":
            continue
        total_rows += 1
        opt = parse_option_description(row.get("Description") or "")
        if opt is None:
            # Description didn't parse — surface raw row.
            by_contract[("UNPARSED", row.get("Description") or "?")]["STO"] += 0
            by_contract[("UNPARSED", row.get("Description") or "?")]["desc"] = (
                row.get("Description") or "?"
            )
            continue
        key = (opt["underlying"], opt["expiration"],
               opt["strike"], opt["option_type"])
        qty = parse_quantity(row.get("Quantity") or "")
        code = (row.get("Trans Code") or "").strip()
        if code in ("STO", "BTC"):
            by_contract[key][code] += qty
            by_contract[key]["desc"] = row.get("Description") or ""

    if not by_contract:
        return []

    out = [
        f"STO/BTC short option rows skipped ({total_rows}) — handle "
        f"manually after import. Affected positions:"
    ]
    for key, counts in by_contract.items():
        desc = counts["desc"] or " ".join(str(p) for p in key)
        out.append(
            f"  - {desc} (STO {counts['STO']:g} / BTC {counts['BTC']:g})"
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DB writer
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_portfolio(cur, portfolio_name: str) -> tuple[int, str]:
    """Return (portfolio_id, user_id) for the named portfolio.

    user_id is sourced from portfolios.user_id — the portfolio itself
    owns its user binding. This script connects via raw psycopg2
    without a request scope, so `app.user_id` isn't set on the
    session, which means the DEFAULT on tenant tables' user_id column
    evaluates to NULL and the NOT NULL constraint fires. Every INSERT
    in this script passes user_id explicitly.
    """
    cur.execute(
        "SELECT id, user_id FROM portfolios WHERE name = %s",
        (portfolio_name,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Portfolio '{portfolio_name}' not found")
    return row[0], str(row[1])


def _next_trade_id_seq(cur, portfolio_id: int, ym: str) -> int:
    """Return the next sequence number for trade_ids in this YYYYMM
    + portfolio. Counts active AND soft-deleted rows so a deleted id
    is never recycled."""
    cur.execute(
        "SELECT trade_id FROM trades_summary "
        "WHERE portfolio_id = %s AND trade_id LIKE %s",
        (portfolio_id, f"{ym}-%"),
    )
    seqs = []
    for (tid,) in cur.fetchall():
        try:
            seqs.append(int(str(tid).split("-")[-1]))
        except (ValueError, IndexError):
            pass
    return (max(seqs) + 1) if seqs else 1


def _build_lifo_summary(campaign: Campaign, trade_id: str) -> dict:
    """Run the canonical LIFO walk on a campaign's transactions to get
    status / shares / avg_entry / avg_exit / realized_pl / etc.

    trade_calc.compute_lifo_summary expects a DataFrame with columns:
    date, action, shares, amount. Multiplier scales dollar totals.
    """
    rows = [
        {"date": tx.date.isoformat(), "action": tx.action,
         "shares": tx.shares, "amount": tx.price}
        for tx in campaign.txns
    ]
    df = pd.DataFrame(rows)
    summary = compute_lifo_summary(
        df, trade_id, campaign.ticker,
        fallback_open_date=campaign.open_date.isoformat(),
        multiplier=campaign.multiplier,
    )
    return summary or {}


def write_campaigns(
    cur,
    portfolio_id: int,
    user_id: str,
    portfolio_name: str,
    campaigns: list[Campaign],
    strategy: str,
) -> tuple[int, int]:
    """Insert summary + detail rows for each campaign. Returns
    (summary_count, detail_count). Cash transactions for BUY/SELL are
    emitted automatically here too (no DB trigger fires for this
    script's connection).

    All INSERTs carry user_id explicitly — see _resolve_portfolio for
    why the column's DEFAULT can't be relied on.
    """
    summary_count = 0
    detail_count = 0

    # Per-month seq counters cached so we don't re-query for every
    # campaign in the same month.
    seq_cache: dict[str, int] = {}

    for camp in campaigns:
        ym = camp.open_date.strftime("%Y%m")
        if ym not in seq_cache:
            seq_cache[ym] = _next_trade_id_seq(cur, portfolio_id, ym)
        seq = seq_cache[ym]
        trade_id = f"{ym}-{seq:03d}"
        seq_cache[ym] = seq + 1

        lifo = _build_lifo_summary(camp, trade_id)

        # Insert summary row.
        summary_cols = [
            "user_id", "portfolio_id", "trade_id", "ticker", "status",
            "open_date", "shares", "avg_entry", "avg_exit", "total_cost",
            "realized_pl", "unrealized_pl", "return_pct", "rule",
            "strategy", "instrument_type", "multiplier",
        ]
        closed_date_col = lifo.get("Closed_Date")
        if closed_date_col:
            summary_cols.append("closed_date")

        summary_vals = [
            user_id, portfolio_id, trade_id, camp.ticker,
            lifo.get("Status", camp.status),
            camp.open_date,
            lifo.get("Shares", 0),
            lifo.get("Avg_Entry", 0),
            lifo.get("Avg_Exit", 0),
            lifo.get("Total_Cost", 0),
            lifo.get("Realized_PL", 0),
            lifo.get("Unrealized_PL", 0),
            lifo.get("Return_Pct", 0),
            "",                  # rule — user will tag later
            strategy,
            camp.instrument_type,
            camp.multiplier,
        ]
        if closed_date_col:
            summary_vals.append(closed_date_col)

        placeholders = ", ".join(["%s"] * len(summary_vals))
        cur.execute(
            f"INSERT INTO trades_summary ({', '.join(summary_cols)}) "
            f"VALUES ({placeholders})",
            tuple(summary_vals),
        )
        summary_count += 1

        # Insert detail rows. trx_id per camp: B1, A1, A2, ..., S1, S2, ...
        b_count = 0
        s_count = 0
        for tx in camp.txns:
            if tx.action == "BUY":
                if b_count == 0:
                    trx_id = "B1"
                else:
                    trx_id = f"A{b_count}"
                b_count += 1
            else:
                s_count += 1
                trx_id = f"S{s_count}"

            value = tx.value if tx.value > 0 else tx.shares * tx.price * camp.multiplier

            detail_cols = [
                "user_id", "portfolio_id", "trade_id", "ticker", "action",
                "date", "shares", "amount", "value", "trx_id",
                "instrument_type", "multiplier",
            ]
            detail_vals = [
                user_id, portfolio_id, trade_id, camp.ticker, tx.action,
                tx.date, tx.shares, tx.price, value, trx_id,
                camp.instrument_type, camp.multiplier,
            ]
            placeholders = ", ".join(["%s"] * len(detail_vals))
            cur.execute(
                f"INSERT INTO trades_details ({', '.join(detail_cols)}) "
                f"VALUES ({placeholders}) RETURNING id",
                tuple(detail_vals),
            )
            detail_id = cur.fetchone()[0]
            detail_count += 1

            # Mirror _emit_trade_cash_tx: BUY → -value, SELL → +value.
            # Skip OEXP zero-value sells.
            if value > 0:
                cash_amount = -value if tx.action == "BUY" else value
                cur.execute(
                    "INSERT INTO cash_transactions "
                    "(user_id, portfolio_id, date, amount, source, "
                    " trade_detail_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (user_id, portfolio_id, tx.date, cash_amount,
                     tx.action.lower(), detail_id),
                )

    return summary_count, detail_count


def write_cash_rows(
    cur, portfolio_id: int, user_id: str, cash_rows: list[CashRecord]
) -> int:
    """Insert deposit cash transactions."""
    for c in cash_rows:
        cur.execute(
            "INSERT INTO cash_transactions "
            "(user_id, portfolio_id, date, amount, source, note) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (user_id, portfolio_id, c.date, c.amount, c.source, c.note),
        )
    return len(cash_rows)


def reset_cash_ledger(cur, portfolio_id: int, user_id: str) -> int:
    """Wipe ALL cash_transactions rows for this (portfolio, user).
    Used once for the real Long-Term Growth import to remove migration
    039's reseed. user_id in the WHERE clause is belt-and-suspenders —
    portfolio_id alone is sufficient since portfolios are user-scoped,
    but the explicit binding makes cross-user contamination impossible
    even if portfolio_id were somehow wrong."""
    cur.execute(
        "DELETE FROM cash_transactions "
        "WHERE portfolio_id = %s AND user_id = %s",
        (portfolio_id, user_id),
    )
    return cur.rowcount


def check_existing_trades(cur, portfolio_id: int, since: date) -> int:
    """Count trades_summary rows already in the target portfolio with
    open_date >= since. Non-zero = previous import likely happened.
    Used to warn the user before they commit a duplicate import."""
    cur.execute(
        "SELECT COUNT(*) FROM trades_summary "
        "WHERE portfolio_id = %s AND open_date >= %s AND deleted_at IS NULL",
        (portfolio_id, since),
    )
    return cur.fetchone()[0]


# ─────────────────────────────────────────────────────────────────────────────
# Driver + verification report
# ─────────────────────────────────────────────────────────────────────────────


def filter_by_date(rows: list[dict], since: date) -> tuple[list[dict], int]:
    """Return (kept_rows, dropped_count). Rows missing a date are kept
    (defensive — better to surface them in 'unknown' than silently drop)."""
    kept = []
    dropped = 0
    for r in rows:
        d = parse_date(r.get("Activity Date") or r.get("Settle Date") or "")
        if d is None:
            kept.append(r)
            continue
        if d >= since:
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


def classify_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[classify(r)] += 1
    return dict(counts)


def print_report(
    csv_path: Path,
    portfolio: str,
    since: date,
    commit: bool,
    counts: dict[str, int],
    pre_cutoff: int,
    equity_campaigns: list[Campaign],
    option_campaigns: list[Campaign],
    cash_rows: list[CashRecord],
    warnings: list[str],
    written: dict[str, int] | None,
    reset_count: int | None,
    existing_trades: int,
) -> None:
    print("=" * 64)
    print("ROBINHOOD IMPORT — PORTFOLIO:", portfolio)
    print("=" * 64)
    print(f"Source CSV:   {csv_path}")
    print(f"Date filter:  >= {since.isoformat()}")
    print(f"Mode:         {'COMMITTED' if commit else 'DRY-RUN'}")
    print()

    print("Classified rows:")
    label_map = [
        ("equity_trade",       "equity_trade"),
        ("option_trade",       "option_trade"),
        ("option_expire",      "option_expire"),
        ("cash_deposit",       "cash_deposit"),
        ("income",             "income (skipped)"),
        ("fee",                "fee (skipped)"),
        ("short_option_trade", "short_option (skipped, manual handling)"),
        ("empty",              "empty (skipped)"),
        ("unknown",            "unknown (skipped)"),
    ]
    for key, label in label_map:
        print(f"  {label:<42s} {counts.get(key, 0)}")
    print()
    print(f"Filtered out (pre-cutoff): {pre_cutoff}")
    print()

    eq_closed = sum(1 for c in equity_campaigns if c.status == "CLOSED")
    eq_open = sum(1 for c in equity_campaigns if c.status == "OPEN")
    opt_closed = sum(1 for c in option_campaigns if c.status == "CLOSED")
    opt_open = sum(1 for c in option_campaigns if c.status == "OPEN")
    print("Campaigns reconstructed:")
    print(f"  Equity, closed:   {eq_closed}")
    print(f"  Equity, open:     {eq_open}")
    print(f"  Option, closed:   {opt_closed}")
    print(f"  Option, open:     {opt_open}")
    print()
    print(f"Cash transactions: {len(cash_rows)}")
    print()

    if existing_trades > 0:
        print(f"⚠️  WARNING: target portfolio already has {existing_trades} "
              f"trades_summary rows since {since.isoformat()}.")
        print("    A previous import may have already happened. Verify before --commit.")
        print()

    if reset_count is not None:
        print(f"Cleared {reset_count} existing cash_transactions before import.")
        print()

    if warnings:
        print("Warnings:")
        for w in warnings[:20]:
            print(f"  - {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")
        print()

    label = "Inserted" if commit else "Would insert (dry-run)"
    if written:
        print(f"{label}:")
        print(f"  trades_summary rows:    {written.get('summary', 0)}")
        print(f"  trades_details rows:    {written.get('details', 0)}")
        print(f"  cash_transactions:      {written.get('cash', 0)}")
    print("=" * 64)


def run_import(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        return 2

    since = datetime.strptime(args.since, "%Y-%m-%d").date()
    if args.allow_pre_cutoff:
        since = date(1900, 1, 1)
        log.info("--allow-pre-cutoff: date filter relaxed to %s", since)

    raw_rows = read_csv(csv_path)
    kept_rows, pre_cutoff = filter_by_date(raw_rows, since)

    counts = classify_counts(kept_rows)
    warnings: list[str] = []

    equity_campaigns = reconstruct_equity_campaigns(kept_rows, warnings)
    option_campaigns = reconstruct_option_campaigns(kept_rows, warnings)
    cash_rows = extract_cash_rows(kept_rows, warnings)
    warnings.extend(collect_short_option_warnings(kept_rows))

    written: dict[str, int] | None = None
    reset_count: int | None = None
    existing_trades = 0

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            portfolio_id, user_id = _resolve_portfolio(cur, args.portfolio)
            cur.execute("SET LOCAL app.user_id = %s", (user_id,))
            existing_trades = check_existing_trades(cur, portfolio_id, since)

            if args.reset_cash_ledger:
                reset_count = reset_cash_ledger(cur, portfolio_id, user_id)

            s_count, d_count = write_campaigns(
                cur, portfolio_id, user_id, args.portfolio,
                equity_campaigns + option_campaigns,
                args.strategy,
            )
            c_count = write_cash_rows(cur, portfolio_id, user_id, cash_rows)
            written = {"summary": s_count, "details": d_count, "cash": c_count}

            if args.commit:
                conn.commit()
                log.info("Committed: %d summary, %d details, %d cash rows",
                         s_count, d_count, c_count)
            else:
                conn.rollback()
                log.info("Dry-run: rolled back all writes")

    print_report(
        csv_path=csv_path, portfolio=args.portfolio, since=since,
        commit=args.commit, counts=counts, pre_cutoff=pre_cutoff,
        equity_campaigns=equity_campaigns,
        option_campaigns=option_campaigns,
        cash_rows=cash_rows, warnings=warnings, written=written,
        reset_count=reset_count, existing_trades=existing_trades,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Import a Robinhood CSV into a portfolio. Dry-run by default."
    )
    p.add_argument("--csv", required=True,
                   help="Path to the Robinhood activity-history CSV")
    p.add_argument("--portfolio", required=True,
                   help="Target portfolio name (e.g. 'Long-Term Growth')")
    p.add_argument("--since", default="2026-01-01",
                   help="Filter rows with Activity Date >= this (YYYY-MM-DD)")
    p.add_argument("--strategy", default="LongTerm",
                   help="Strategy to tag new trades with (default: LongTerm)")
    p.add_argument("--reset-cash-ledger", action="store_true",
                   help="DELETE existing cash_transactions for this portfolio before import")
    p.add_argument("--allow-pre-cutoff", action="store_true",
                   help="Testing flag: relax --since to 1900-01-01")
    p.add_argument("--commit", action="store_true",
                   help="Actually write changes. Without this, the script rolls back.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    return run_import(args)


if __name__ == "__main__":
    sys.exit(main())
