"""Structural tests for migrations/041_add_match_method_column.sql.

The migration is Phase 2 B-0: it adds the `match_method` column to
trades_details and backfills every existing SELL row with 'LIFO'.
Nothing in api/ or scripts/ reads the column yet — that's B-1+.

Production-data correctness (row counts pre/post) is verified by the
DO $$ verification block inside the migration itself (it RAISE
EXCEPTIONs if any SELL is left unstamped). These tests guard the
migration FILE against accidental edits that would break the
invariants behind that verification — same pattern used by
test_migration_034_sell_rule_taxonomy and 035_sell_rule_strays.
"""
from __future__ import annotations

from pathlib import Path

import pytest


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "migrations"
    / "041_add_match_method_column.sql"
)


@pytest.fixture(scope="module")
def sql() -> str:
    return MIGRATION_PATH.read_text()


class TestColumnAddition:
    """Step 1 must ADD COLUMN IF NOT EXISTS so the migration is
    re-runnable end-to-end."""

    def test_adds_column_if_not_exists(self, sql):
        assert "ADD COLUMN IF NOT EXISTS match_method TEXT" in sql

    def test_targets_trades_details(self, sql):
        # ALTER TABLE trades_details ... must appear; defensive against
        # accidental table-name typos.
        assert "ALTER TABLE trades_details" in sql

    def test_column_is_text(self, sql):
        # TEXT (not VARCHAR(n)) — matches the migration narrative and
        # avoids an arbitrary length cap on a fixed-vocab column.
        assert "match_method TEXT" in sql


class TestCheckConstraint:
    """Step 2 must add a named CHECK constraint guarding the vocab."""

    def test_constraint_named(self, sql):
        # Naming the constraint lets re-runs probe pg_constraint and
        # skip cleanly. Anonymous constraint would re-fire on re-run
        # and need a different idempotency guard.
        assert "trades_details_match_method_check" in sql

    def test_constraint_guarded_by_pg_constraint_probe(self, sql):
        # The IF NOT EXISTS probe is what makes Step 2 re-runnable.
        # Removing it would make the second run fail with
        # 'constraint already exists'.
        assert "FROM pg_constraint" in sql
        assert "trades_details_match_method_check" in sql
        # The probe must wrap the ADD CONSTRAINT — verify ordering.
        probe_idx = sql.find("FROM pg_constraint")
        add_idx = sql.find("ADD CONSTRAINT trades_details_match_method_check")
        assert probe_idx > 0 and add_idx > 0
        assert probe_idx < add_idx

    def test_allows_lifo(self, sql):
        # CHECK must allow 'LIFO'.
        assert "'LIFO'" in sql

    def test_allows_hcfo(self, sql):
        # CHECK must allow 'HCFO' so B-1 can stamp HCFO sells without
        # a second schema migration.
        assert "'HCFO'" in sql

    def test_allows_null(self, sql):
        # CHECK must allow NULL so BUY rows (and any future non-
        # matching action types) stay valid without backfill.
        assert "match_method IS NULL" in sql

    def test_no_other_values_allowed(self, sql):
        # Defensive: the CHECK clause should not accidentally allow
        # 'FIFO' or other matching-method names.
        # Find the CHECK expression body. Spans one line in the file.
        check_line = next(
            line for line in sql.splitlines()
            if "CHECK (match_method IN" in line
        )
        assert "'FIFO'" not in check_line
        assert "'AVERAGE'" not in check_line


class TestBackfill:
    """Step 3 backfills existing SELL rows with 'LIFO'. The IN list
    must be pinned, not LIKE/substring, so the action vocab is
    explicit and any future inventory-reducing action requires
    an intentional migration update."""

    def test_backfill_uses_explicit_in_list(self, sql):
        # IN ('SELL') — pinned vocab. LIKE 'S%' or similar would
        # silently accept future action types we haven't audited.
        assert "action IN ('SELL')" in sql

    def test_backfill_stamps_lifo(self, sql):
        assert "SET match_method = 'LIFO'" in sql

    def test_backfill_filters_on_null(self, sql):
        # WHERE match_method IS NULL makes Step 3 a no-op on re-run.
        # Without this guard, re-running would touch every SELL row
        # again (harmless but noisy + breaks idempotency contract).
        assert "WHERE match_method IS NULL" in sql

    def test_backfill_does_not_touch_buys(self, sql):
        # Belt-and-suspenders: the migration must not anywhere stamp
        # BUY rows. Look for any SET clause that would touch them.
        forbidden = [
            "action = 'BUY'",
            "action IN ('BUY')",
            "action IN ('BUY', 'SELL')",
            "action IN ('SELL', 'BUY')",
        ]
        for needle in forbidden:
            assert needle not in sql, f"unexpected reference to BUY: {needle!r}"


class TestVerificationBlock:
    """Step 4 must fail loudly if any SELL is left unstamped after
    backfill. This is the migration's primary correctness gate."""

    def test_raises_on_incomplete_backfill(self, sql):
        # RAISE EXCEPTION on unstamped_count > 0 — without this the
        # migration would silently commit a half-backfilled state.
        assert "RAISE EXCEPTION" in sql
        assert "unstamped_count" in sql

    def test_counts_unstamped_sells(self, sql):
        # The verification COUNT(*) must filter on action='SELL' AND
        # match_method IS NULL to identify what was missed.
        assert "action IN ('SELL')" in sql
        assert "match_method IS NULL" in sql

    def test_emits_success_notice(self, sql):
        # RAISE NOTICE on the happy path — surfaced in migrations/run.py
        # output so the operator sees the stamped count.
        assert "RAISE NOTICE" in sql
        assert "Migration 041 complete" in sql


class TestIdempotency:
    """All three operations must be re-runnable cleanly."""

    def test_add_column_if_not_exists(self, sql):
        assert "ADD COLUMN IF NOT EXISTS match_method" in sql

    def test_constraint_guarded_by_probe(self, sql):
        # The DO $$ block wrapping the ADD CONSTRAINT must use IF NOT
        # EXISTS (SELECT 1 FROM pg_constraint ...).
        assert "IF NOT EXISTS (" in sql
        assert "FROM pg_constraint" in sql

    def test_backfill_uses_is_null_guard(self, sql):
        # Already covered by TestBackfill but kept here for the
        # idempotency contract — re-running must update 0 rows.
        assert "WHERE match_method IS NULL" in sql


class TestSafetyInvariants:
    """The migration must not delete or rewrite existing data."""

    def test_no_delete_statements(self, sql):
        # DELETE is forbidden in B-0. Only ADD + UPDATE allowed.
        # We look for the keyword as a statement-start; substring
        # match is safe because the migration text doesn't reference
        # 'DELETE' in any comment context.
        assert "DELETE FROM" not in sql.upper()

    def test_no_drop_statements(self, sql):
        # No DROP COLUMN / DROP CONSTRAINT / DROP TABLE — pure
        # additive change.
        assert "DROP " not in sql.upper()

    def test_no_truncate(self, sql):
        assert "TRUNCATE" not in sql.upper()


class TestNarrativeAnchors:
    """The migration preamble must accurately describe Phase 2's
    B-0 → B-1 → B-2 → B-3 progression so future maintainers can
    locate the wider context."""

    def test_mentions_phase_2_b_zero(self, sql):
        assert "Phase 2 B-0" in sql

    def test_mentions_lifo_and_hcfo(self, sql):
        # Both stamps must be documented in the preamble.
        assert "LIFO" in sql
        assert "HCFO" in sql

    def test_mentions_no_behavioral_change(self, sql):
        # The preamble must explicitly note nothing reads the column
        # yet — protects against a future maintainer wiring a read in
        # B-0 thinking it's safe.
        body_lower = sql.lower()
        assert "unused" in body_lower or "nothing in api" in body_lower
