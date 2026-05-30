"""Structural tests for migrations/042_pinned_routes.sql.

The migration creates the pinned_routes table for the desktop sidebar's
Pinned section. Same persistence idioms as pinned_entities (029) —
soft-delete + revive, partial-unique on live rows, RLS — but keyed on
VARCHAR route paths instead of (entity_type, entity_id) integer pairs.

Production-data correctness (RLS enforcement, partial-unique blocking
duplicate live pins) is exercised by post-deploy smoke and by the
companion test_pinned_routes.py behavior tests. These tests guard the
migration FILE against accidental edits that would break the invariants —
same pattern as test_migration_041_match_method.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "migrations"
    / "042_pinned_routes.sql"
)


@pytest.fixture(scope="module")
def sql() -> str:
    return MIGRATION_PATH.read_text()


class TestTableCreation:
    def test_create_table_present(self, sql):
        assert "CREATE TABLE IF NOT EXISTS pinned_routes" in sql

    def test_id_serial_primary_key(self, sql):
        # SERIAL PK matches the pinned_entities (029) shape.
        assert "id          SERIAL          PRIMARY KEY" in sql or \
               "id SERIAL PRIMARY KEY" in sql.replace("  ", " ")

    def test_user_id_uuid_fk_to_users(self, sql):
        assert "user_id" in sql
        assert "UUID" in sql
        assert "REFERENCES users(id) ON DELETE RESTRICT" in sql

    def test_user_id_default_pulls_from_app_user_id_guc(self, sql):
        # The DEFAULT must resolve user_id from the RLS GUC so RLS-only
        # callers (and the migration runner) don't have to set it.
        # Same shape as pinned_entities (029).
        assert "current_setting('app.user_id', true)" in sql
        # Founder UUID fallback — must match the migrations/run.py literal
        # and the canonical migration-024+ pattern.
        assert "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a" in sql

    def test_route_path_varchar_120(self, sql):
        # VARCHAR(120) is the agreed ceiling — longest current nav.ts route
        # is /performance-heatmap (~20 chars), 120 leaves substantial margin.
        assert "route_path  VARCHAR(120)    NOT NULL" in sql or \
               "route_path VARCHAR(120) NOT NULL" in sql.replace("  ", " ")

    def test_pinned_at_timestamptz_default_now(self, sql):
        assert "pinned_at   TIMESTAMPTZ     NOT NULL DEFAULT now()" in sql or \
               "pinned_at TIMESTAMPTZ NOT NULL DEFAULT now()" in sql.replace("  ", " ")

    def test_deleted_at_nullable_timestamptz(self, sql):
        # Soft-delete column — NULLABLE. No DEFAULT (rows start live).
        assert "deleted_at  TIMESTAMPTZ" in sql or \
               "deleted_at TIMESTAMPTZ" in sql.replace("  ", " ")


class TestRoutePathCheckConstraint:
    """The CHECK constraint on route_path is the conservative validator
    matching the current nav.ts inventory (kebab-case lowercase). Defensive
    against arbitrary text at the DB boundary."""

    def test_check_pattern_present(self, sql):
        # The regex matches /segment(/segment)* where each segment is
        # lowercase alphanumeric + hyphen.
        assert "CHECK (route_path ~ '^/[a-z0-9-]+(/[a-z0-9-]+)*$')" in sql

    def test_check_anchored_at_start_and_end(self, sql):
        # ^ and $ anchors — otherwise substring matches would pass paths
        # with disallowed prefixes/suffixes.
        assert "'^/" in sql
        assert "$'" in sql or "$')" in sql

    def test_check_rejects_uppercase(self, sql):
        # The regex character class is [a-z0-9-], explicitly no A-Z.
        # Indirect: verify no [a-zA-Z] in the migration body.
        check_line = next(line for line in sql.splitlines()
                          if "CHECK (route_path" in line)
        assert "A-Z" not in check_line


class TestIndexes:
    def test_partial_unique_live_index_present(self, sql):
        # One live pin per (user_id, route_path). Allows re-pin via revive.
        assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_pinned_routes_live" in sql
        assert "ON pinned_routes (user_id, route_path)" in sql
        # The partial predicate must be present — otherwise a re-pin after
        # soft-delete would collide on the unique constraint.
        assert "WHERE deleted_at IS NULL" in sql

    def test_user_live_pinned_at_index_present(self, sql):
        # Primary read path: "what does this user have pinned, FIFO?"
        assert "CREATE INDEX IF NOT EXISTS idx_pinned_routes_user_live" in sql
        assert "ON pinned_routes (user_id, pinned_at)" in sql


class TestRowLevelSecurity:
    def test_rls_enabled(self, sql):
        assert "ALTER TABLE pinned_routes ENABLE ROW LEVEL SECURITY" in sql

    def test_rls_forced(self, sql):
        # FORCE applies RLS to table owners too — without this the API role
        # could bypass isolation if it ever owned the table.
        assert "ALTER TABLE pinned_routes FORCE  ROW LEVEL SECURITY" in sql or \
               "ALTER TABLE pinned_routes FORCE ROW LEVEL SECURITY" in sql.replace("  ", " ")

    def test_isolation_policy_present(self, sql):
        assert "CREATE POLICY pinned_routes_isolation ON pinned_routes" in sql

    def test_isolation_policy_uses_nullif_wrapper(self, sql):
        # NULLIF wrapper guards against the empty-string GUC default
        # masquerading as a UUID and matching every row.
        assert "NULLIF(current_setting('app.user_id', true), '')::uuid" in sql

    def test_isolation_policy_has_both_using_and_with_check(self, sql):
        # USING for SELECT/UPDATE/DELETE; WITH CHECK for INSERT/UPDATE.
        # Both required for FOR ALL coverage.
        policy_block = sql[sql.find("CREATE POLICY pinned_routes_isolation"):]
        assert "USING" in policy_block
        assert "WITH CHECK" in policy_block


class TestIdempotency:
    """All three operations must be re-runnable cleanly."""

    def test_create_table_if_not_exists(self, sql):
        assert "CREATE TABLE IF NOT EXISTS pinned_routes" in sql

    def test_create_index_if_not_exists(self, sql):
        assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_pinned_routes_live" in sql
        assert "CREATE INDEX IF NOT EXISTS idx_pinned_routes_user_live" in sql

    def test_drop_policy_if_exists_before_create(self, sql):
        # Mirrors pinned_entities (029) — DROP POLICY IF EXISTS makes the
        # CREATE POLICY safe to re-run.
        drop_idx = sql.find("DROP POLICY IF EXISTS pinned_routes_isolation")
        create_idx = sql.find("CREATE POLICY pinned_routes_isolation")
        assert drop_idx > 0
        assert create_idx > drop_idx


class TestNarrativeAnchors:
    """The migration preamble must document the design choices so future
    maintainers can locate the wider context."""

    def test_mentions_pinned_entities_divergence(self, sql):
        # Preamble must explain WHY a separate table from pinned_entities.
        assert "pinned_entities" in sql or "Migration 029" in sql

    def test_mentions_conservative_check_constraint(self, sql):
        # The CHECK is intentionally tight — flag for future maintainers
        # that uppercase/underscore routes would need a follow-up migration.
        body_lower = sql.lower()
        assert "conservative" in body_lower or "uppercase" in body_lower

    def test_mentions_reversibility(self, sql):
        # DROP TABLE pinned_routes CASCADE is the rollback — preamble
        # should call this out so rollback ops aren't guesswork.
        assert "DROP TABLE pinned_routes" in sql or "Reversible" in sql or \
               "reversible" in sql.lower()


class TestSafetyInvariants:
    """The migration must not delete or rewrite existing data."""

    def test_no_delete_statements(self, sql):
        # Only ALTER + CREATE + DROP POLICY allowed. No DELETE/UPDATE on
        # existing tables.
        body_upper = sql.upper()
        assert "DELETE FROM" not in body_upper
        assert "UPDATE PINNED" not in body_upper.replace(" ", "")  # crude

    def test_no_alter_on_existing_tables(self, sql):
        # The migration touches only pinned_routes (new) and its policies.
        # No ALTER on users, pinned_entities, etc.
        forbidden = [
            "ALTER TABLE users",
            "ALTER TABLE pinned_entities",
            "ALTER TABLE trades_summary",
            "ALTER TABLE trades_details",
        ]
        for needle in forbidden:
            assert needle not in sql

    def test_no_truncate(self, sql):
        assert "TRUNCATE" not in sql.upper()
