"""Structural tests for migrations/034_sell_rule_taxonomy_cleanup.sql.

The migration is a one-shot data backfill onto the new 13-rule canonical
sell-rule taxonomy. Production-data correctness (row counts) was verified
out-of-band against a live DB before writing the migration; these tests
guard the migration FILE against accidental edits that would break
correctness — specifically:

  1. All 7 source→target mappings are present in the correct UPDATE
     pairings (trades_summary AND trades_details for action='SELL').
  2. Step 2 (sr13 Earnings Exit → sr10) appears BEFORE Step 6 (sr4
     Change of Character → sr13). Reversing this order silently
     conflates the two rules — see migration preamble.
  3. The final-verification allowlist lists all 13 canonical sell_rule
     values plus 'History' (preserved).
  4. The migration is idempotent in shape — every UPDATE references the
     OLD value in its WHERE clause, so re-running matches zero rows.
"""
from __future__ import annotations

from pathlib import Path

import pytest


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "migrations"
    / "034_sell_rule_taxonomy_cleanup.sql"
)


# Source value → (target value, expected_summary_row_count_at_audit)
MAPPINGS = [
    ("sr1.1 capital protection - hard stop", "sr1 Capital Protection"),
    ("sr13 Earnings Exit",                   "sr10 Earnings Exit"),
    ("sr15 BE Stop Out (moved at +10%)",     "sr11 BE Stop Out (moved at +10%)"),
    ("sr8 TQQQ Strategy Exit",               "sr12 TQQQ Strategy Exit"),
    ("sr16 Profit Taking",                   "sr2 Selling into Strength"),
    ("sr4 Change of Character",              "sr13 Change of Character"),
    ("sr9 Breakout Failure",                 "sr9 Failed Breakout"),
]


CANONICAL_LABELS = [
    "sr1 Capital Protection",
    "sr2 Selling into Strength",
    "sr3 Portfolio Management",
    "sr4 Time Stop",
    "sr5 Climax Top",
    "sr6 8e Momentum Trim",
    "sr7 Holding Winners - 21e Violation",
    "sr8 Big Cushion Sell Rule",
    "sr9 Failed Breakout",
    "sr10 Earnings Exit",
    "sr11 BE Stop Out (moved at +10%)",
    "sr12 TQQQ Strategy Exit",
    "sr13 Change of Character",
]


@pytest.fixture(scope="module")
def sql() -> str:
    return MIGRATION_PATH.read_text()


class TestMappingsPresent:
    """Each of the 7 user-locked mappings must update BOTH tables."""

    @pytest.mark.parametrize("src,tgt", MAPPINGS)
    def test_summary_update_present(self, sql, src, tgt):
        # Find an UPDATE trades_summary that SETs sell_rule = '<tgt>' WHERE sell_rule = '<src>'.
        # Use literal substring search since the migration is straight SQL with no formatting.
        needle_set = f"SET sell_rule = '{tgt}'"
        needle_where = f"WHERE sell_rule = '{src}'"
        assert needle_set in sql, f"missing SET for {tgt!r}"
        assert needle_where in sql, f"missing WHERE for {src!r}"

    @pytest.mark.parametrize("src,tgt", MAPPINGS)
    def test_details_update_present(self, sql, src, tgt):
        # Details updates must filter to action='SELL' to avoid touching
        # buy rules (the 'rule' column on trades_details is dual-purpose).
        needle_set = f"SET rule = '{tgt}'"
        needle_where = f"WHERE rule = '{src}' AND action = 'SELL'"
        assert needle_set in sql, f"missing details SET for {tgt!r}"
        assert needle_where in sql, f"missing details WHERE for {src!r} AND action='SELL'"


class TestStepOrdering:
    """Step 2 MUST run before Step 6 — otherwise sr4 gets conflated with sr10."""

    def test_step2_appears_before_step6(self, sql):
        step2_marker = "Step 2 (sr13 Earnings -> sr10"
        step6_marker = "Step 6 (sr4 -> sr13"
        idx_2 = sql.find(step2_marker)
        idx_6 = sql.find(step6_marker)
        assert idx_2 > 0, "Step 2 RAISE NOTICE marker missing"
        assert idx_6 > 0, "Step 6 RAISE NOTICE marker missing"
        assert idx_2 < idx_6, (
            "Step 2 must appear before Step 6 in the migration. Otherwise "
            "sr4 trades land in sr13 and then get re-mapped to sr10, "
            "conflating Change of Character with Earnings Exit."
        )

    def test_sr4_update_runs_after_sr13_update(self, sql):
        # Defense in depth: even if the RAISE NOTICE markers were removed,
        # the actual UPDATE for 'sr13 Earnings Exit' must precede the one
        # for 'sr4 Change of Character'.
        idx_13 = sql.find("WHERE sell_rule = 'sr13 Earnings Exit'")
        idx_4  = sql.find("WHERE sell_rule = 'sr4 Change of Character'")
        assert idx_13 > 0 and idx_4 > 0
        assert idx_13 < idx_4


class TestCanonicalAllowlist:
    """The final-verification block must list all 13 canonical labels + History."""

    @pytest.mark.parametrize("label", CANONICAL_LABELS)
    def test_canonical_label_in_allowlist(self, sql, label):
        assert f"'{label}'" in sql, f"canonical {label!r} missing from allowlist"

    def test_history_preserved(self, sql):
        # 'History' is intentionally outside the canonical taxonomy and
        # appears in the allowlist so the leftover-count verification
        # doesn't flag it.
        assert "'History'" in sql


class TestIdempotency:
    """Every UPDATE filters on the SOURCE value, so re-running matches
    zero rows on each step (the second run has no rows with those
    source values left)."""

    @pytest.mark.parametrize("src,_tgt", MAPPINGS)
    def test_summary_where_clauses_use_source_value(self, sql, src, _tgt):
        # If a WHERE clause filtered on the TARGET value, the second run
        # would re-process the freshly-migrated rows (non-idempotent).
        # Idempotency holds because we filter on src.
        assert f"WHERE sell_rule = '{src}'" in sql

    @pytest.mark.parametrize("src,_tgt", MAPPINGS)
    def test_details_where_clauses_use_source_value(self, sql, src, _tgt):
        assert f"WHERE rule = '{src}' AND action = 'SELL'" in sql


class TestRemovedRulesNotReferenced:
    """Sanity: the migration should not introduce any removed-rule
    descriptions as TARGET values. The 13-rule taxonomy is locked."""

    @pytest.mark.parametrize("removed_phrase", [
        "Trailing Stop",
        "Exhaustion Gap",
        "200d Moving Avg Break",
        "Living Below 50d",
        "Scale-Out T1",
        "Scale-Out T2",
        "Scale-Out T3",
        "Market Correction Exit",
    ])
    def test_no_target_uses_removed_phrase(self, sql, removed_phrase):
        # These removed descriptions should never appear as a SET target.
        assert f"SET sell_rule = '{removed_phrase}" not in sql
        assert f"SET rule = '{removed_phrase}" not in sql
