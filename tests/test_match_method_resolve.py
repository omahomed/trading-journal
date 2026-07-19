"""Tests for api.main._resolve_match_method — the per-call env-read
helper that resolves MATCH_METHOD at SELL-write time.

Behavior:
  Unset / empty / "LIFO" / "lifo" / etc.  → 'LIFO' (default)
  Any other non-empty value               → ValueError

HCFO was removed as a stamp option (2026-07-18). New sells only stamp
LIFO; a lingering MATCH_METHOD=HCFO env raises loudly. The calc engine
still walks HCFO on recompute for historical SELL rows — this helper
is only about *new* stamps.

Per-call resolution (rather than module-load) is what makes these
tests cleanly isolated via monkeypatch.setenv / delenv.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def resolve(monkeypatch):
    """Import the helper after AUTH_SECRET is set so import-time
    side effects don't crash the test module."""
    monkeypatch.setenv("AUTH_SECRET", "test-secret-not-for-prod")
    from api.main import _resolve_match_method
    return _resolve_match_method


class TestDefault:
    def test_default_is_lifo_when_env_unset(self, resolve, monkeypatch):
        monkeypatch.delenv("MATCH_METHOD", raising=False)
        assert resolve() == "LIFO"

    def test_empty_string_treated_as_unset(self, resolve, monkeypatch):
        """Empty MATCH_METHOD env var resolves to LIFO, not a ValueError.
        Deploy configs that template the variable but leave it blank
        should fall back gracefully."""
        monkeypatch.setenv("MATCH_METHOD", "")
        assert resolve() == "LIFO"

    def test_whitespace_only_treated_as_unset(self, resolve, monkeypatch):
        monkeypatch.setenv("MATCH_METHOD", "   ")
        assert resolve() == "LIFO"


class TestValidValues:
    def test_lifo_when_env_set_lifo(self, resolve, monkeypatch):
        monkeypatch.setenv("MATCH_METHOD", "LIFO")
        assert resolve() == "LIFO"

    def test_lowercase_lifo(self, resolve, monkeypatch):
        monkeypatch.setenv("MATCH_METHOD", "lifo")
        assert resolve() == "LIFO"

    def test_surrounding_whitespace_stripped(self, resolve, monkeypatch):
        monkeypatch.setenv("MATCH_METHOD", "  LIFO  ")
        assert resolve() == "LIFO"


class TestInvalidValues:
    def test_hcfo_now_raises(self, resolve, monkeypatch):
        """HCFO was removed as a stamp option. A lingering deploy config
        with MATCH_METHOD=HCFO must fail loudly instead of silently
        stamping new sells with a discontinued method."""
        monkeypatch.setenv("MATCH_METHOD", "HCFO")
        with pytest.raises(ValueError, match="HCFO was removed"):
            resolve()

    def test_invalid_value_raises_valueerror(self, resolve, monkeypatch):
        monkeypatch.setenv("MATCH_METHOD", "FIFO")
        with pytest.raises(ValueError, match="MATCH_METHOD"):
            resolve()

    def test_average_cost_raises(self, resolve, monkeypatch):
        monkeypatch.setenv("MATCH_METHOD", "AVERAGE")
        with pytest.raises(ValueError):
            resolve()

    def test_garbage_value_raises(self, resolve, monkeypatch):
        monkeypatch.setenv("MATCH_METHOD", "asdfqwer")
        with pytest.raises(ValueError):
            resolve()

    def test_error_message_names_valid_options(self, resolve, monkeypatch):
        """Operators reading the deploy log should see the allowed
        vocabulary in the error, not a bare 'invalid value'."""
        monkeypatch.setenv("MATCH_METHOD", "X")
        with pytest.raises(ValueError) as exc:
            resolve()
        assert "LIFO" in str(exc.value)


class TestPerCallResolution:
    """Helper reads os.environ on each call — not at module load. This is
    what makes monkeypatch.setenv work reliably across consecutive tests
    in the same process."""

    def test_changes_between_calls_are_observed(self, resolve, monkeypatch):
        monkeypatch.setenv("MATCH_METHOD", "LIFO")
        assert resolve() == "LIFO"
        monkeypatch.delenv("MATCH_METHOD")
        assert resolve() == "LIFO"
