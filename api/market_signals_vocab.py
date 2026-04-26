"""V11 market signal vocabulary.

Authoritative set of signal_type values written to the `market_signals` table.
Phase 2's MCT engine validates against this set before insert. Stored as a
Python frozenset rather than a SQL CHECK constraint so the vocabulary can
evolve during Phase 2 without DROP/ADD CONSTRAINT churn.
"""

from __future__ import annotations


ALLOWED_SIGNAL_TYPES: frozenset[str] = frozenset({
    "CORRECTION_DECLARED",
    "CORRECTION_NULLIFIED",
    "STEP_0_RALLY_DAY",
    "STEP_1_FTD",
    "STEP_2_CLOSE_ABOVE_21EMA",
    "STEP_3_LOW_ABOVE_21EMA",
    "STEP_4_LOW_ABOVE_21EMA_3BARS",
    "STEP_5_LOW_ABOVE_50SMA_3BARS",
    "STEP_6_MA_STACK_SLOW",
    "STEP_7_MA_STACK_FULL",
    "STEP_8_POWERTREND_ON",
    "POWERTREND_OFF",
    "RALLY_INVALIDATED",
    "POST_FTD_SOFT_FAIL",
    "VIOLATION_21EMA",
    "CONFIRMED_BREAK_21EMA",
    "VIOLATION_50SMA",
    "CHARACTER_BREAK",
    "TRENDING_REGIME_BREAK",
    "RECOVERY",
    "V10_SOFT_RESET",
    "V10_FULL_INVALIDATION",
    "CAP_AT_100_ACTIVATED",
    "CAP_AT_100_RELEASED",
})


def is_valid_signal_type(signal_type: str) -> bool:
    return signal_type in ALLOWED_SIGNAL_TYPES
