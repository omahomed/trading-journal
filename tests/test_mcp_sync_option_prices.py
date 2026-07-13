"""Unit tests for scripts/mcp_sync_option_prices.py.

The DB write path is intentionally NOT tested here — it just fires an
UPDATE per planned row inside an existing transaction. What CAN break
across future changes is the ticker encoding (must stay identical to
import_robinhood_csv.encode_option_ticker) and the match / unmatch /
stale bookkeeping. Those are all pure functions, so this file lives
DB-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcp_sync_option_prices as sync   # noqa: E402
import import_robinhood_csv as rh        # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Ticker encoding — must stay byte-identical to the importer's encode
# (they're two halves of the same round-trip: import writes tickers with
# encode_option_ticker, this sync matches against them with the copy here).
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("underlying,exp,opt,strike,expected", [
    ("AVGO", "2026-09-18", "call", 390.00, "AVGO 260918 $390C"),
    ("TWLO", "2026-09-18", "call", 210.00, "TWLO 260918 $210C"),
    ("FIVN", "2026-10-16", "call",  22.50, "FIVN 261016 $22.5C"),
    ("RBLX", "2026-08-21", "put",   50.00, "RBLX 260821 $50P"),
    # Fractional strike with trailing zeros must be trimmed to $22.5, not $22.50.
    ("XYZ",  "2027-01-15", "call",   0.5,  "XYZ 270115 $0.5C"),
    # Whole-dollar strikes never render a decimal.
    ("SPY",  "2026-12-18", "put",  500.00, "SPY 261218 $500P"),
])
def test_encode_option_ticker_matches_expected(underlying, exp, opt, strike, expected):
    assert sync.encode_option_ticker(underlying, exp, opt, strike) == expected


def test_encode_matches_the_import_parsers_encoding():
    """Round-trip contract: the sync's encode must match the importer's.
    If either implementation drifts, this test fires."""
    from datetime import date
    cases = [
        ("AVGO", date(2026, 9, 18),  "call", 390.00),
        ("FIVN", date(2026, 10, 16), "call",  22.5),
        ("RBLX", date(2026, 8, 21),  "put",   50.00),
    ]
    for underlying, exp, opt, strike in cases:
        assert (
            sync.encode_option_ticker(underlying, exp.isoformat(), opt, strike)
            == rh.encode_option_ticker(underlying, exp, opt, strike)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Position validation
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_position_accepts_a_well_formed_row():
    assert sync.validate_position({
        "underlying": "AVGO", "expiration": "2026-09-18",
        "strike": 390.0, "option_type": "call", "mark_price": 43.7,
    }) is None


def test_validate_position_flags_missing_keys():
    err = sync.validate_position({"underlying": "AVGO"})
    assert err is not None and "missing" in err


def test_validate_position_flags_bad_expiration():
    err = sync.validate_position({
        "underlying": "AVGO", "expiration": "not-a-date",
        "strike": 100.0, "option_type": "call", "mark_price": 1.0,
    })
    assert err is not None and "expiration" in err


def test_validate_position_flags_non_call_or_put():
    err = sync.validate_position({
        "underlying": "AVGO", "expiration": "2026-09-18",
        "strike": 100.0, "option_type": "straddle", "mark_price": 1.0,
    })
    assert err is not None and "option_type" in err


def test_validate_position_rejects_negative_mark_price():
    err = sync.validate_position({
        "underlying": "AVGO", "expiration": "2026-09-18",
        "strike": 100.0, "option_type": "call", "mark_price": -1.0,
    })
    assert err is not None


# ─────────────────────────────────────────────────────────────────────────────
# Matching + bookkeeping
# ─────────────────────────────────────────────────────────────────────────────


def _app_row(trade_id: str, avg_entry: float, manual_price: float | None = None,
             shares: float = 1.0) -> dict:
    return {"trade_id": trade_id, "avg_entry": avg_entry,
            "manual_price": manual_price, "shares": shares}


def test_build_updates_matches_by_ticker():
    positions = [
        {"underlying": "AVGO", "expiration": "2026-09-18",
         "strike": 390.0, "option_type": "call", "mark_price": 43.7},
    ]
    app_rows = {"AVGO 260918 $390C": _app_row("202607-011", 42.0, manual_price=40.0)}
    planned, unmatched, stale = sync.build_updates(positions, app_rows)
    assert len(planned) == 1
    assert planned[0]["trade_id"] == "202607-011"
    assert planned[0]["new_price"] == 43.7
    assert planned[0]["old_price"] == 40.0
    assert planned[0]["delta"] == pytest.approx(3.7)
    assert unmatched == []
    assert stale == []


def test_build_updates_reports_unmatched_when_no_app_row():
    positions = [{
        "underlying": "AVGO", "expiration": "2026-09-18",
        "strike": 999.0, "option_type": "call", "mark_price": 1.0,
    }]
    planned, unmatched, stale = sync.build_updates(positions, app_rows={})
    assert planned == []
    assert len(unmatched) == 1
    assert unmatched[0]["computed_ticker"] == "AVGO 260918 $999C"
    assert "no matching" in unmatched[0]["error"]


def test_build_updates_reports_stale_when_app_row_has_no_input():
    app_rows = {
        "AVGO 260918 $390C": _app_row("202607-011", 42.0, manual_price=40.0),
        "TWLO 260918 $210C": _app_row("202607-010", 26.9, manual_price=25.0),
    }
    positions = [{
        "underlying": "AVGO", "expiration": "2026-09-18",
        "strike": 390.0, "option_type": "call", "mark_price": 43.7,
    }]
    planned, unmatched, stale = sync.build_updates(positions, app_rows)
    assert len(planned) == 1
    assert stale == ["TWLO 260918 $210C"]


def test_build_updates_still_plans_when_app_row_has_null_manual_price():
    """Newly-imported options have manual_price=NULL. The delta is
    unknown, but the plan must still fire."""
    app_rows = {"AVGO 260918 $390C": _app_row("202607-011", 42.0, manual_price=None)}
    positions = [{
        "underlying": "AVGO", "expiration": "2026-09-18",
        "strike": 390.0, "option_type": "call", "mark_price": 43.7,
    }]
    planned, _, _ = sync.build_updates(positions, app_rows)
    assert len(planned) == 1
    assert planned[0]["old_price"] is None
    assert planned[0]["delta"] is None


def test_build_updates_skips_malformed_input_but_matches_others():
    app_rows = {"AVGO 260918 $390C": _app_row("202607-011", 42.0, manual_price=40.0)}
    positions = [
        {"underlying": "AVGO"},  # missing several fields
        {"underlying": "AVGO", "expiration": "2026-09-18",
         "strike": 390.0, "option_type": "call", "mark_price": 43.7},
    ]
    planned, unmatched, _ = sync.build_updates(positions, app_rows)
    assert len(planned) == 1
    assert len(unmatched) == 1
    assert "missing" in unmatched[0]["error"]
