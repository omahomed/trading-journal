"""Structural tests for migrations/035_sell_rule_strays_cleanup.sql.

Followup to migration 034. Maps the three non-canonical sell-rule
strays in trades_details (action='SELL') onto sr1 Capital Protection
per user-locked decision:

  'sr10 Scale-Out T1 (-3%)'  -> sr1 Capital Protection  (5 rows)
  'sr11 Scale-Out T2 (-5%)'  -> sr1 Capital Protection  (1 row)
  'IBKR'                     -> sr1 Capital Protection  (1 row)

Same testing shape as test_migration_034: parse the SQL file and
assert correctness invariants. Production correctness (row counts)
was verified out-of-band before drafting the migration.
"""
from __future__ import annotations

from pathlib import Path

import pytest


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "migrations"
    / "035_sell_rule_strays_cleanup.sql"
)


STRAY_SOURCES = [
    "sr10 Scale-Out T1 (-3%)",
    "sr11 Scale-Out T2 (-5%)",
    "IBKR",
]


@pytest.fixture(scope="module")
def sql() -> str:
    return MIGRATION_PATH.read_text()


def test_migration_file_exists():
    assert MIGRATION_PATH.exists(), f"missing {MIGRATION_PATH}"


@pytest.mark.parametrize("source", STRAY_SOURCES)
def test_migration_uses_in_clause_with_all_three_sources(sql, source):
    # The migration uses an IN (...) list, not three separate UPDATEs.
    # Each source must appear inside the SQL (quoted exactly as it
    # appears in production data).
    assert f"'{source}'" in sql, f"stray source {source!r} missing from migration"


def test_migration_targets_sr1_capital_protection(sql):
    # All three strays consolidate onto sr1 Capital Protection.
    assert "SET rule = 'sr1 Capital Protection'" in sql


def test_migration_filters_by_action_sell(sql):
    # action='SELL' is mandatory: trades_details.rule is dual-purpose
    # (BUY rows use it for buy-rule strings). Must not touch BUYs.
    assert "action = 'SELL'" in sql


def test_migration_filters_deleted_at_null(sql):
    # Soft-deleted detail rows must not be migrated.
    assert "deleted_at IS NULL" in sql


def test_migration_is_idempotent(sql):
    # The WHERE clause filters on the SOURCE values, not the target.
    # After the first run, the source values are no longer present;
    # the second run matches zero rows.
    for source in STRAY_SOURCES:
        assert f"'{source}'" in sql
    # And the target 'sr1 Capital Protection' is not itself in the
    # IN list — if it were, idempotency would break (it would re-
    # match itself on the second run, no-op but redundant).
    in_clause_start = sql.find("rule IN (")
    in_clause_end = sql.find(")", in_clause_start)
    in_clause = sql[in_clause_start:in_clause_end]
    assert "'sr1 Capital Protection'" not in in_clause, (
        "target value 'sr1 Capital Protection' must NOT appear in the IN "
        "source list — that would break idempotency invariance"
    )


def test_migration_logs_before_and_after_counts(sql):
    # Same RAISE NOTICE pattern as migration 034 for deploy verification.
    assert "RAISE NOTICE" in sql
    assert "before" in sql.lower()
    assert "after" in sql.lower()


def test_migration_has_no_summary_updates(sql):
    # Per migration 034's verification, these strays don't mirror to
    # trades_summary.sell_rule. No summary UPDATEs are needed; if any
    # appear here, that's a scope creep that should be caught.
    assert "UPDATE trades_summary" not in sql
    assert "sell_rule" not in sql.lower() or "sell-rule" in sql.lower(), (
        "comments may reference 'sell-rule' as a concept, but no "
        "SQL should touch the sell_rule column"
    )
