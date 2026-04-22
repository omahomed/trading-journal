"""Unit tests for db_layer's pure input validators.

These functions don't touch the database and are trivial to stub out in tests,
which makes them the right first targets — they prove the pytest plumbing and
CI pipeline work end-to-end. As we extract more pure logic from db_layer.py
(LIFO, risk math, journal backfill) we'll add targeted tests alongside those
refactors.
"""
from __future__ import annotations

import pytest

from db_layer import validate_trade_id, validate_shares, validate_price


class TestValidateTradeId:
    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_trade_id("")

    def test_rejects_none(self) -> None:
        with pytest.raises(ValueError):
            validate_trade_id(None)

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty after trimming"):
            validate_trade_id("   ")

    def test_trims_surrounding_whitespace(self) -> None:
        assert validate_trade_id("  202602-001  ") == "202602-001"

    def test_rejects_overly_long(self) -> None:
        with pytest.raises(ValueError, match="too long"):
            validate_trade_id("X" * 51)

    def test_accepts_max_length(self) -> None:
        assert validate_trade_id("X" * 50) == "X" * 50


class TestValidateShares:
    # Note: validate_shares wraps every error path in a generic "Invalid shares
    # value" message because the internal try/except catches its own raises.
    # Distinguishing the underlying reason would require refactoring the
    # validator; these tests pin down today's behaviour so we don't regress.

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="Invalid shares"):
            validate_shares(0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="Invalid shares"):
            validate_shares(-1)

    def test_rejects_unreasonably_large(self) -> None:
        with pytest.raises(ValueError, match="Invalid shares"):
            validate_shares(1_000_001)

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(ValueError, match="Invalid shares"):
            validate_shares("not-a-number")

    def test_accepts_fractional(self) -> None:
        assert validate_shares("12.5") == 12.5


class TestValidatePrice:
    # Same pattern as validate_shares — the wrapper hides the specific cause,
    # so tests assert on the generic "Invalid <field>" message.

    def test_allows_zero(self) -> None:
        # Zero is a valid price (e.g. assignment of shares at $0 basis).
        assert validate_price(0) == 0.0

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="Invalid Price"):
            validate_price(-0.01)

    def test_rejects_unreasonably_large(self) -> None:
        with pytest.raises(ValueError, match="Invalid Price"):
            validate_price(1_000_001)

    def test_uses_custom_field_name_in_errors(self) -> None:
        with pytest.raises(ValueError, match="Invalid Stop Loss"):
            validate_price(-5, field_name="Stop Loss")

    def test_accepts_string(self) -> None:
        assert validate_price("123.45") == 123.45
