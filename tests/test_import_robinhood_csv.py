"""Tests for scripts/import_robinhood_csv.py.

Pure-function parser + classifier + campaign-reconstruction tests
run without a DB. The DB-writer path is unit-tested only at the
helper level (classification/reconstruction); end-to-end commit
verification is intentionally deferred to a manual dry-run against
the real CSV — the script's --commit flag carries a big "no
production runs in CI" caveat and a TestClient fixture for one-shot
import scripts would be more complexity than value.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import import_robinhood_csv as rh  # noqa: E402

FIXTURE_CSV = SCRIPTS / "fixtures" / "robinhood_sample.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Parser tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParseCurrency:
    @pytest.mark.parametrize("s,expected", [
        ("$682.32", 682.32),
        ("$1,050.46", 1050.46),
        ("($2,729.30)", -2729.30),
        ("($5.00)", -5.0),
        ("$0.00", 0.0),
        ("", 0.0),
        (None, 0.0),
        ("100", 100.0),  # no $ sign
        # Real-CSV whitespace quirks (Amendment 4):
        ("$57.88 ", 57.88),              # trailing space
        (" ($2,729.30)", -2729.30),      # leading space
        ("$6,500.00\t", 6500.00),        # trailing tab
        ("  $1,050.46  ", 1050.46),      # both sides
    ])
    def test_parses(self, s, expected):
        assert rh.parse_currency(s) == pytest.approx(expected)

    def test_garbage_returns_zero(self):
        assert rh.parse_currency("not a number") == 0.0


class TestParseDate:
    @pytest.mark.parametrize("s,expected", [
        ("2/10/2025", date(2025, 2, 10)),
        ("12/31/2026", date(2026, 12, 31)),
        ("1/1/2026", date(2026, 1, 1)),
        ("2026-05-21", date(2026, 5, 21)),     # ISO fallback
        # M/D/YY (real 2026-era Robinhood export — Amendment 1):
        ("5/21/26", date(2026, 5, 21)),
        ("5/21/2026", date(2026, 5, 21)),       # 4-digit kept
        ("1/12/26", date(2026, 1, 12)),
        ("5/21/68", date(2068, 5, 21)),          # %y 00-68 boundary
        ("5/21/69", date(1969, 5, 21)),          # %y 69-99 boundary
    ])
    def test_parses(self, s, expected):
        assert rh.parse_date(s) == expected

    def test_empty_returns_none(self):
        assert rh.parse_date("") is None
        assert rh.parse_date(None) is None

    def test_garbage_returns_none(self):
        assert rh.parse_date("not a date") is None


class TestParseOptionDescription:
    def test_standard_call(self):
        result = rh.parse_option_description("PLTR 1/10/2025 Call $67.00")
        assert result == {
            "underlying": "PLTR",
            "expiration": date(2025, 1, 10),
            "option_type": "Call",
            "strike": 67.00,
        }

    def test_put(self):
        result = rh.parse_option_description("AAPL 3/15/2026 Put $180.50")
        assert result["option_type"] == "Put"
        assert result["strike"] == 180.50

    def test_case_insensitive(self):
        result = rh.parse_option_description("PLTR 1/10/2025 CALL $67.00")
        assert result["option_type"] == "Call"

    def test_equity_description_returns_none(self):
        assert rh.parse_option_description("Palantir Technologies") is None
        assert rh.parse_option_description("Apple Inc.") is None

    def test_malformed_returns_none(self):
        assert rh.parse_option_description("PLTR Call $67") is None  # no date
        assert rh.parse_option_description("") is None
        assert rh.parse_option_description(None) is None


class TestParseQuantity:
    @pytest.mark.parametrize("s,expected", [
        ("3", 3.0),
        ("3S", 3.0),       # OEXP trailing letter
        ("100", 100.0),
        ("0.5", 0.5),      # fractional shares
        ("1.25", 1.25),
        ("0.002007", 0.002007),  # real-CSV NVDA dividend-reinvest
        ("", 0.0),
        (None, 0.0),
        # Whitespace handling (Amendment 4):
        (" 25 ", 25.0),
        ("\t100", 100.0),
    ])
    def test_parses(self, s, expected):
        assert rh.parse_quantity(s) == pytest.approx(expected)


class TestEncodeOptionTicker:
    def test_integer_strike(self):
        t = rh.encode_option_ticker("PLTR", date(2025, 1, 10), "Call", 67.0)
        assert t == "PLTR 250110 $67C"

    def test_fractional_strike(self):
        t = rh.encode_option_ticker("AAPL", date(2026, 3, 15), "Put", 180.5)
        assert t == "AAPL 260315 $180.5P"

    def test_uppercase_ticker(self):
        t = rh.encode_option_ticker("pltr", date(2025, 1, 10), "call", 67.0)
        assert t == "PLTR 250110 $67C"


# ─────────────────────────────────────────────────────────────────────────────
# Classifier tests
# ─────────────────────────────────────────────────────────────────────────────


class TestClassify:
    @pytest.mark.parametrize("code,expected", [
        ("Buy", "equity_trade"),
        ("Sell", "equity_trade"),
        ("BTO", "option_trade"),
        ("STC", "option_trade"),
        ("OEXP", "option_expire"),
        ("ACH", "cash_deposit"),
        ("RTP", "cash_deposit"),
        ("DCF", "cash_deposit"),
        ("ITRF", "cash_deposit"),     # internal transfer (Amendment 2)
        ("STO", "short_option_trade"),  # Amendment 2 — skipped, loud
        ("BTC", "short_option_trade"),  # Amendment 2 — skipped, loud
        ("CDIV", "income"),
        ("MDIV", "income"),
        ("INT", "income"),
        ("GOLD", "fee"),
        ("GDBP", "fee"),
        ("MINT", "fee"),
    ])
    def test_each_code(self, code, expected):
        assert rh.classify({"Trans Code": code}) == expected

    def test_unknown_code(self):
        assert rh.classify({"Trans Code": "FOO"}) == "unknown"

    def test_empty_row(self):
        assert rh.classify({}) == "empty"
        assert rh.classify({"Trans Code": "", "Activity Date": ""}) == "empty"


# ─────────────────────────────────────────────────────────────────────────────
# Campaign reconstruction tests
# ─────────────────────────────────────────────────────────────────────────────


def _eq_row(date_str, code, ticker, qty, price, amount):
    return {
        "Activity Date": date_str, "Trans Code": code, "Instrument": ticker,
        "Quantity": str(qty), "Price": price, "Amount": amount,
        "Description": ticker,
    }


class TestEquityCampaigns:
    def test_single_buy_no_sell_is_open(self):
        rows = [_eq_row("3/15/2026", "Buy", "AAPL", 10, "$100.00", "($1000.00)")]
        warnings = []
        camps = rh.reconstruct_equity_campaigns(rows, warnings)
        assert len(camps) == 1
        assert camps[0].ticker == "AAPL"
        assert camps[0].status == "OPEN"
        assert camps[0].shares_remaining == 10
        assert not warnings

    def test_single_buy_matching_sell_is_closed(self):
        rows = [
            _eq_row("3/15/2026", "Buy",  "AAPL", 10, "$100.00", "($1000.00)"),
            _eq_row("3/20/2026", "Sell", "AAPL", 10, "$110.00", "$1100.00"),
        ]
        warnings = []
        camps = rh.reconstruct_equity_campaigns(rows, warnings)
        assert len(camps) == 1
        assert camps[0].status == "CLOSED"

    def test_multiple_buys_partial_sell_stays_open(self):
        rows = [
            _eq_row("3/15/2026", "Buy",  "AAPL", 10, "$100.00", "($1000.00)"),
            _eq_row("3/16/2026", "Buy",  "AAPL", 5,  "$105.00", "($525.00)"),
            _eq_row("3/20/2026", "Sell", "AAPL", 8,  "$110.00", "$880.00"),
        ]
        warnings = []
        camps = rh.reconstruct_equity_campaigns(rows, warnings)
        assert len(camps) == 1
        assert camps[0].status == "OPEN"
        assert camps[0].shares_remaining == 7

    def test_full_close_then_new_buy_makes_two_campaigns(self):
        rows = [
            _eq_row("3/15/2026", "Buy",  "AAPL", 10, "$100.00", "($1000.00)"),
            _eq_row("3/20/2026", "Sell", "AAPL", 10, "$110.00", "$1100.00"),
            _eq_row("3/25/2026", "Buy",  "AAPL", 5,  "$105.00", "($525.00)"),
        ]
        warnings = []
        camps = rh.reconstruct_equity_campaigns(rows, warnings)
        assert len(camps) == 2
        assert camps[0].status == "CLOSED"
        assert camps[1].status == "OPEN"
        assert camps[1].shares_remaining == 5

    def test_sell_without_prior_buy_warns_and_skips(self):
        rows = [_eq_row("3/20/2026", "Sell", "GOOG", 3, "$150.00", "$450.00")]
        warnings = []
        camps = rh.reconstruct_equity_campaigns(rows, warnings)
        assert len(camps) == 0
        assert any("SELL without prior BUY" in w for w in warnings)


class TestOptionCampaigns:
    def _opt_row(self, date_str, code, qty, price, amount,
                 underlying="PLTR", exp="4/19/2026", strike=30, opt_type="Call"):
        return {
            "Activity Date": date_str, "Trans Code": code,
            "Instrument": underlying,
            "Description": f"{underlying} {exp} {opt_type} ${strike:.2f}",
            "Quantity": str(qty), "Price": price, "Amount": amount,
        }

    def test_bto_open_only(self):
        rows = [self._opt_row("3/15/2026", "BTO", 3, "$2.50", "($750.00)")]
        warnings = []
        camps = rh.reconstruct_option_campaigns(rows, warnings)
        assert len(camps) == 1
        assert camps[0].status == "OPEN"
        assert camps[0].instrument_type == "OPTION"
        assert camps[0].multiplier == 100.0

    def test_oexp_closes_position_as_zero_value(self):
        rows = [
            self._opt_row("3/15/2026", "BTO",  3, "$2.50", "($750.00)"),
            self._opt_row("4/19/2026", "OEXP", 3, "", ""),
        ]
        warnings = []
        camps = rh.reconstruct_option_campaigns(rows, warnings)
        assert len(camps) == 1
        assert camps[0].status == "CLOSED"
        # The OEXP tx should have value 0 (full premium loss).
        oexp = camps[0].txns[-1]
        assert oexp.action == "SELL"
        assert oexp.price == 0.0
        assert oexp.value == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Cash extraction
# ─────────────────────────────────────────────────────────────────────────────


class TestCashExtraction:
    def _row(self, code, amount, instrument="", desc=""):
        return {
            "Activity Date": "3/15/2026", "Trans Code": code,
            "Instrument": instrument, "Description": desc,
            "Amount": amount, "Quantity": "", "Price": "",
        }

    def test_ach_deposit(self):
        rows = [self._row("ACH", "$5000.00", desc="Brokerage deposit")]
        warnings = []
        cash = rh.extract_cash_rows(rows, warnings)
        assert len(cash) == 1
        assert cash[0].amount == 5000.0
        assert cash[0].source == "deposit"
        assert "ACH" in cash[0].note

    def test_itrf_deposit(self):
        """ITRF (internal transfer) treated as cash_deposit; note
        carries 'ITRF:' prefix for traceability (Amendment 2)."""
        rows = [self._row("ITRF", "$6,500.00",
                          desc="Transfer from Brokerage to Joint")]
        cash = rh.extract_cash_rows(rows, [])
        assert len(cash) == 1
        assert cash[0].source == "deposit"
        assert cash[0].note.startswith("ITRF:")
        assert "Brokerage to Joint" in cash[0].note
        assert cash[0].amount == 6500.0

    def test_income_rows_no_longer_extracted(self):
        """Amendment 3: CDIV/MDIV/INT classify as 'income' but are
        intentionally skipped — not inserted into cash_transactions."""
        rows = [
            self._row("CDIV", "$12.50", instrument="AAPL"),
            self._row("MDIV", "$5.00", instrument="VOO"),
            self._row("INT", "$3.42"),
        ]
        cash = rh.extract_cash_rows(rows, [])
        assert cash == []

    def test_fees_excluded(self):
        rows = [
            self._row("GOLD", "($5.00)"),
            self._row("GDBP", "($0.50)"),
            self._row("MINT", "($1.00)"),
        ]
        cash = rh.extract_cash_rows(rows, [])
        assert cash == []

    def test_short_options_excluded(self):
        """Amendment 2: STO/BTC rows are not cash deposits — they're
        skipped with a loud warning instead."""
        rows = [
            self._row("STO", "$100.00",
                      desc="USAR 3/20/2026 Call $30.00"),
            self._row("BTC", "($50.00)",
                      desc="USAR 3/20/2026 Call $30.00"),
        ]
        cash = rh.extract_cash_rows(rows, [])
        assert cash == []


class TestShortOptionWarnings:
    def _sho_row(self, code, qty, desc, date_str="2/25/2026"):
        return {
            "Activity Date": date_str, "Trans Code": code,
            "Instrument": desc.split()[0], "Description": desc,
            "Quantity": str(qty), "Price": "$1.00", "Amount": "$100.00",
        }

    def test_empty_when_no_short_options(self):
        rows = [{"Trans Code": "Buy", "Activity Date": "3/1/2026"}]
        assert rh.collect_short_option_warnings(rows) == []

    def test_groups_by_contract(self):
        rows = [
            self._sho_row("STO", 5, "USAR 3/20/2026 Call $30.00"),
            self._sho_row("BTC", 5, "USAR 3/20/2026 Call $30.00"),
        ]
        warnings = rh.collect_short_option_warnings(rows)
        assert len(warnings) == 2  # 1 header + 1 contract line
        assert "STO/BTC short option rows skipped (2)" in warnings[0]
        assert "handle manually" in warnings[0]
        # The contract line should show aggregated counts.
        assert "USAR" in warnings[1]
        assert "STO 5" in warnings[1]
        assert "BTC 5" in warnings[1]

    def test_multiple_contracts(self):
        rows = [
            self._sho_row("STO", 1, "OKLO 3/20/2026 Call $110.00"),
            self._sho_row("STO", 2, "OKLO 3/20/2026 Call $110.00"),
            self._sho_row("BTC", 3, "OKLO 3/20/2026 Call $110.00"),
            self._sho_row("STO", 2, "LEU 3/20/2026 Call $340.00"),
            self._sho_row("BTC", 2, "LEU 3/20/2026 Call $340.00"),
        ]
        warnings = rh.collect_short_option_warnings(rows)
        # 1 header + 2 distinct contracts.
        assert len(warnings) == 3
        oklo_line = next(w for w in warnings if "OKLO" in w)
        assert "STO 3" in oklo_line
        assert "BTC 3" in oklo_line
        leu_line = next(w for w in warnings if "LEU" in w)
        assert "STO 2" in leu_line
        assert "BTC 2" in leu_line


# ─────────────────────────────────────────────────────────────────────────────
# Integration: end-to-end against the fixture CSV (no DB writes)
# ─────────────────────────────────────────────────────────────────────────────


class TestFixtureIntegration:
    """Verify the script's reading + classification + reconstruction
    paths against the synthetic sample CSV. No DB connection involved
    — the writer path is exercised separately via dry-run when the
    user has the real CSV."""

    def test_csv_reads_cleanly(self):
        rows = rh.read_csv(FIXTURE_CSV)
        assert len(rows) >= 15

    def test_classification_counts_match_fixture(self):
        rows = rh.read_csv(FIXTURE_CSV)
        # Date filter doesn't matter here — every fixture row is 2026.
        counts = rh.classify_counts(rows)
        # Synthetic fixture composition (see scripts/fixtures/robinhood_sample.csv):
        #   5 equity trades, 3 option trades (BTO+STC+OEXP one of each),
        #   2 ACH/RTP, 1 CDIV, 1 INT, 2 GOLD/GDBP, 1 empty, 1 unknown (WHAT),
        #   1 GOOG SELL (counts as equity_trade in classifier — reconstruction
        #   skips it later via the orphan-SELL warning).
        assert counts.get("equity_trade") == 6   # 5 PLTR/AAPL + 1 orphan GOOG
        assert counts.get("option_trade") == 2
        assert counts.get("option_expire") == 1
        assert counts.get("cash_deposit") == 2
        assert counts.get("income") == 2
        assert counts.get("fee") == 2
        assert counts.get("empty") == 1
        assert counts.get("unknown") == 1

    def test_equity_campaigns_reconstructed(self):
        rows = rh.read_csv(FIXTURE_CSV)
        warnings: list[str] = []
        camps = rh.reconstruct_equity_campaigns(rows, warnings)
        # PLTR: Buy 10, Buy 5, Sell 15 (closes) → new Buy 8 (open) = 2 campaigns.
        # AAPL: Buy 5 (open) = 1 campaign.
        # Total: 3.
        tickers = sorted(c.ticker for c in camps)
        assert tickers == ["AAPL", "PLTR", "PLTR"]
        statuses = sorted(c.status for c in camps)
        assert statuses == ["CLOSED", "OPEN", "OPEN"]
        # Orphan GOOG sell should be in warnings, not in campaigns.
        assert any("GOOG" in w and "SELL without prior BUY" in w for w in warnings)

    def test_option_campaigns_reconstructed(self):
        rows = rh.read_csv(FIXTURE_CSV)
        camps = rh.reconstruct_option_campaigns(rows, [])
        # PLTR 4/19 $30 Call: BTO 3, STC 2, OEXP 1S → closes (one campaign).
        assert len(camps) == 1
        assert camps[0].ticker == "PLTR 260419 $30C"
        assert camps[0].status == "CLOSED"

    def test_cash_rows_extracted(self):
        rows = rh.read_csv(FIXTURE_CSV)
        cash = rh.extract_cash_rows(rows, [])
        # After Amendment 3: income rows skipped, only deposits remain.
        # Synthetic fixture has 2 deposits (ACH, RTP). The 2 income rows
        # (CDIV, INT) classify as 'income' but extract_cash_rows ignores
        # them.
        assert len(cash) == 2
        assert all(c.source == "deposit" for c in cash)
        assert all(
            c.note.startswith("ACH:") or c.note.startswith("RTP:")
            for c in cash
        )


# ─────────────────────────────────────────────────────────────────────────────
# Parametrized classification audit against both fixtures (Amendment 5)
# ─────────────────────────────────────────────────────────────────────────────


REAL_FIXTURE_CSV = SCRIPTS / "fixtures" / "long_term_growth_real.csv"


@pytest.mark.parametrize("csv_path,expected_counts", [
    (FIXTURE_CSV, {
        "equity_trade": 6,        # 5 PLTR/AAPL + 1 orphan GOOG sell
        "option_trade": 2,
        "option_expire": 1,
        "cash_deposit": 2,        # ACH + RTP
        "income": 2,              # CDIV + INT (skipped, not inserted)
        "fee": 2,                 # GOLD + GDBP (skipped)
        "short_option_trade": 0,
        "empty": 1,
        "unknown": 1,             # the 'WHAT' row
    }),
    (REAL_FIXTURE_CSV, {
        "equity_trade": 97,         # 66 Buy + 31 Sell
        "option_trade": 49,          # 29 BTO + 20 STC
        "option_expire": 0,
        "cash_deposit": 13,          # 10 ACH + 3 ITRF
        "income": 4,                 # 1 CDIV + 3 INT (skipped)
        "fee": 4,                    # 4 MINT (skipped)
        "short_option_trade": 10,    # 5 STO + 5 BTC (skipped, warn)
        "empty": 2,
        "unknown": 0,
    }),
])
def test_classification_audit_against_fixture(csv_path, expected_counts):
    """Catches drift between the importer's view of CSV shape and the
    actual data. If Robinhood adds a new Trans Code we don't recognize
    it'd land in 'unknown' here and fail the assertion."""
    rows = rh.read_csv(csv_path)
    counts = rh.classify_counts(rows)
    for cat, exp in expected_counts.items():
        assert counts.get(cat, 0) == exp, (
            f"{csv_path.name}: '{cat}' expected={exp} actual={counts.get(cat, 0)} "
            f"(full counts={dict(counts)})"
        )


class TestRealCsvReconstruction:
    """Light reconstruction sanity checks against the real CSV. We don't
    pin exact campaign counts here (they depend on chronological
    walking + orphan-sell warnings); we check shape invariants
    instead."""

    def test_short_option_warnings_present(self):
        rows = rh.read_csv(REAL_FIXTURE_CSV)
        warnings = rh.collect_short_option_warnings(rows)
        assert warnings, "Expected STO/BTC warnings for real CSV"
        assert "skipped (10)" in warnings[0]
        # Expect each of the 4 known short contracts to appear.
        joined = "\n".join(warnings)
        assert "USAR" in joined
        assert "OKLO" in joined
        assert "LEU" in joined
        assert "ZETA" in joined

    def test_no_unknown_codes(self):
        """If we get 'unknown' on the real CSV, we have an unmapped
        Trans Code that needs explicit handling."""
        rows = rh.read_csv(REAL_FIXTURE_CSV)
        counts = rh.classify_counts(rows)
        assert counts.get("unknown", 0) == 0, (
            f"Unknown codes found in real CSV: "
            f"{[r.get('Trans Code') for r in rows if rh.classify(r) == 'unknown']}"
        )

    def test_cash_deposits_match_real_csv(self):
        """13 deposit rows: 10 ACH + 3 ITRF. All source='deposit'.
        ITRF notes start with 'ITRF:'."""
        rows = rh.read_csv(REAL_FIXTURE_CSV)
        cash = rh.extract_cash_rows(rows, [])
        assert len(cash) == 13
        itrf = [c for c in cash if c.note.startswith("ITRF:")]
        ach = [c for c in cash if c.note.startswith("ACH:")]
        assert len(itrf) == 3
        assert len(ach) == 10
