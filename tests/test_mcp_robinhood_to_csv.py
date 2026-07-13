"""Unit tests for scripts/mcp_robinhood_to_csv.py.

Locks the MCP-JSON → Robinhood-CSV shape so the transformer stays
compatible with import_robinhood_csv.py's parser. Fixture JSON matches
the actual response shape observed via
mcp__claude_ai_Robinhood__get_equity_orders / get_option_orders in a
live Claude Code session (July 2026 sample).
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcp_robinhood_to_csv as tr  # noqa: E402
import import_robinhood_csv as rh   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — trimmed to the fields the transformer actually reads.
# ─────────────────────────────────────────────────────────────────────────────

EQUITY_FILLED_BUY = {
    "state": "filled",
    "symbol": "PENG",
    "side": "buy",
    "cumulative_quantity": "50.000000",
    "average_price": "79.285000",
    "fees": "0.000000",
    "created_at": "2026-07-10T18:02:51.433501Z",
    "last_transaction_at": "2026-07-10T18:02:51.62Z",
}

EQUITY_FILLED_SELL = {
    "state": "filled",
    "symbol": "HOOD",
    "side": "sell",
    "cumulative_quantity": "25.000000",
    "average_price": "111.351400",
    "fees": "0.060000",
    "created_at": "2026-07-08T14:35:28.901758Z",
    "last_transaction_at": "2026-07-08T14:35:29.011Z",
}

EQUITY_CANCELLED = {
    "state": "cancelled",
    "symbol": "DRAM",
    "side": "buy",
    "cumulative_quantity": "0.000000",
    "average_price": None,
    "created_at": "2026-07-08T05:09:56.008567Z",
    "last_transaction_at": "2026-07-08T12:27:01.087Z",
}

OPTION_BTO_LONG_CALL = {
    "state": "filled",
    "chain_symbol": "AVGO",
    "type": "limit",
    "direction": "debit",
    "processed_quantity": "1.00000",
    "price": "42.00000000",
    "trade_value_multiplier": "100.0000",
    "opening_strategy": "long_call",
    "closing_strategy": None,
    "created_at": "2026-07-08T16:52:58.399162Z",
    "last_transaction_at": None,
    "legs": [
        {
            "side": "buy",
            "position_effect": "open",
            "ratio_quantity": 1,
            "expiration_date": "2026-09-18",
            "strike_price": "390.0000",
            "option_type": "call",
        }
    ],
}

OPTION_STC_LONG_CALL = {
    "state": "filled",
    "chain_symbol": "TWLO",
    "direction": "credit",
    "processed_quantity": "1.00000",
    "price": "9.40000000",
    "trade_value_multiplier": "100.0000",
    "opening_strategy": None,
    "closing_strategy": "long_call",
    "created_at": "2026-07-06T16:38:44.598501Z",
    "last_transaction_at": None,
    "legs": [
        {
            "side": "sell",
            "position_effect": "close",
            "ratio_quantity": 1,
            "expiration_date": "2026-07-17",
            "strike_price": "210.0000",
            "option_type": "call",
        }
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Row-level tests
# ─────────────────────────────────────────────────────────────────────────────


def test_equity_buy_row_uses_average_price_and_signs_debit_amount():
    row = tr.equity_order_to_row(EQUITY_FILLED_BUY)
    assert row is not None
    assert row["Instrument"] == "PENG"
    assert row["Trans Code"] == "Buy"
    assert row["Quantity"] == "50"
    assert row["Price"] == "$79.28"
    # Buy → amount in parens (debit): 50 * 79.285 = 3964.25, no fees.
    assert row["Amount"] == "($3,964.25)"
    assert row["Activity Date"] == "7/10/2026"


def test_equity_sell_row_has_positive_amount_including_fees():
    row = tr.equity_order_to_row(EQUITY_FILLED_SELL)
    assert row is not None
    # 25 * 111.3514 - 0.06 = 2783.725
    assert row["Trans Code"] == "Sell"
    assert row["Amount"].startswith("$2,783.")
    assert not row["Amount"].startswith("(")


def test_cancelled_equity_order_is_dropped():
    assert tr.equity_order_to_row(EQUITY_CANCELLED) is None


def test_option_bto_row_normalizes_ticker_and_flags_debit():
    rows = tr.option_order_to_rows(OPTION_BTO_LONG_CALL)
    assert len(rows) == 1
    row = rows[0]
    assert row["Instrument"] == "AVGO"
    assert row["Description"] == "AVGO 9/18/2026 Call $390.00"
    assert row["Trans Code"] == "BTO"
    assert row["Quantity"] == "1"
    assert row["Price"] == "$42.00"
    # 1 contract * 42 * 100 = 4200, debit → parens.
    assert row["Amount"] == "($4,200.00)"


def test_option_stc_row_is_credit_and_uses_correct_trans_code():
    rows = tr.option_order_to_rows(OPTION_STC_LONG_CALL)
    assert len(rows) == 1
    row = rows[0]
    assert row["Trans Code"] == "STC"
    assert row["Description"] == "TWLO 7/17/2026 Call $210.00"
    # 1 * 9.40 * 100 = 940 credit.
    assert row["Amount"] == "$940.00"


def test_option_trans_code_mapping():
    assert tr._option_trans_code("buy",  "open")  == "BTO"
    assert tr._option_trans_code("sell", "close") == "STC"
    assert tr._option_trans_code("sell", "open")  == "STO"
    assert tr._option_trans_code("buy",  "close") == "BTC"
    assert tr._option_trans_code("weird", "open") == ""


# ─────────────────────────────────────────────────────────────────────────────
# Exclusion filter
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_exclusions_reads_the_documented_format():
    ex = tr.parse_exclusions("2026-06-26,MRVL,sell,30;2026-06-25,AAPL,buy,10")
    assert ex == [
        ("2026-06-26", "MRVL", "sell", 30.0),
        ("2026-06-25", "AAPL", "buy",  10.0),
    ]


def test_exclusion_drops_matching_equity_order():
    order = {
        "state": "filled",
        "symbol": "MRVL",
        "side": "sell",
        "cumulative_quantity": "30.000000",
        "average_price": "266.79",
        "created_at": "2026-06-26T14:00:00Z",
        "last_transaction_at": "2026-06-26T14:00:00Z",
    }
    exclusions = tr.parse_exclusions("2026-06-26,MRVL,sell,30")
    csv_text = tr.transform([order], [], exclusions=exclusions)
    # Only the header row survives.
    assert csv_text.strip().count("\n") == 0


def test_exclusion_ignores_non_matching_order():
    order = {
        "state": "filled",
        "symbol": "MRVL",
        "side": "buy",   # exclusion targets sell — should NOT match
        "cumulative_quantity": "30.000000",
        "average_price": "266.79",
        "created_at": "2026-06-26T14:00:00Z",
        "last_transaction_at": "2026-06-26T14:00:00Z",
    }
    csv_text = tr.transform([order], [], exclusions=tr.parse_exclusions("2026-06-26,MRVL,sell,30"))
    # Header + one row.
    assert csv_text.strip().count("\n") == 1


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end contract with the human-download parser
# ─────────────────────────────────────────────────────────────────────────────


def test_transformed_csv_round_trips_through_the_parser():
    """The whole point of this transformer is that its output feeds
    scripts/import_robinhood_csv.py unmodified. This test guards that
    contract: the CSV goes into read_csv_from_text + classify_counts and
    the trade rows come out on the equity/option side, not "unknown"."""
    csv_text = tr.transform(
        [EQUITY_FILLED_BUY, EQUITY_FILLED_SELL, EQUITY_CANCELLED],
        [OPTION_BTO_LONG_CALL, OPTION_STC_LONG_CALL],
    )
    rows = rh.read_csv_from_text(csv_text)
    counts = rh.classify_counts(rows)
    assert counts.get("equity_trade") == 2   # cancelled dropped by transformer
    assert counts.get("option_trade") == 2
    assert counts.get("unknown", 0) == 0


def test_full_csv_has_the_expected_columns_in_order():
    """Header must match the parser's expectation letter for letter."""
    csv_text = tr.transform([EQUITY_FILLED_BUY], [])
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader)
    assert header == tr.CSV_COLUMNS
