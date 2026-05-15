-- ============================================================================
-- Migration 029: pinned_entities — Phase 6 (NotesRail favorites)
-- ============================================================================
-- Polymorphic pin table mirroring the Phase 1 tag_assignments idiom
-- (migration 026). Phase 6 mounts on weekly_retro only; Phase 7 will reuse
-- the same table with entity_type='daily_journal'. The schema is
-- future-proofed via the CHECK constraint on entity_type; adding a third
-- type later is one ALTER TABLE DROP CONSTRAINT + ADD CONSTRAINT.
--
-- Soft-delete + idempotent revival is the persistence contract:
--   pin   → row exists with deleted_at IS NULL
--   unpin → UPDATE deleted_at = NOW() (same row, soft-deleted)
--   re-pin same entity → UPDATE deleted_at = NULL (revives the row,
--                        same id; preserves the original pinned_at for
--                        sort-by-recency stability across toggles).
-- toggle_pin() in db_layer implements the SELECT-then-branch logic.
--
-- The partial-unique index on (user_id, entity_type, entity_id) WHERE
-- deleted_at IS NULL guarantees at most one live pin per entity per user.
--
-- No audit trigger — pins are personal UI preferences, not trade data.
-- Same precedent as strategies (019), cash_transactions (009), and
-- tag_assignments (026). audit_trail in this codebase is trade-centric.
--
-- RLS enabled + FORCE'd per the canonical migration-024+ pattern for
-- user-scoped tables. The NULLIF wrapper on app.user_id prevents the
-- empty-string GUC default from masquerading as a legitimate UUID.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

CREATE TABLE IF NOT EXISTS pinned_entities (
    id           SERIAL          PRIMARY KEY,
    user_id      UUID            NOT NULL REFERENCES users(id) ON DELETE RESTRICT
                                 DEFAULT (
                                     COALESCE(
                                         NULLIF(current_setting('app.user_id', true), '')::uuid,
                                         'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
                                     )
                                 ),
    entity_type  VARCHAR(32)     NOT NULL
                                 CHECK (entity_type IN ('weekly_retro','daily_journal')),
    entity_id    INTEGER         NOT NULL,
    pinned_at    TIMESTAMPTZ     NOT NULL DEFAULT now(),
    deleted_at   TIMESTAMPTZ
);

-- Live pair uniqueness — re-pinning the same entity is a no-op via the
-- idempotent revival in toggle_pin. Mirrors uq_tag_assignments_triple_live.
CREATE UNIQUE INDEX IF NOT EXISTS idx_pinned_entities_live
    ON pinned_entities (user_id, entity_type, entity_id)
    WHERE deleted_at IS NULL;

-- "Which entities of this type are pinned?" — primary read path that
-- powers the rail's Pinned section.
CREATE INDEX IF NOT EXISTS idx_pinned_entities_type_live
    ON pinned_entities (entity_type) WHERE deleted_at IS NULL;

ALTER TABLE pinned_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE pinned_entities FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS pinned_entities_isolation ON pinned_entities;
CREATE POLICY pinned_entities_isolation ON pinned_entities FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
-- Expect: empty table.
--   SELECT count(*) FROM pinned_entities;
--
-- Expect: RLS enabled with pinned_entities_isolation policy.
--   SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class
--     WHERE relname = 'pinned_entities';
--
-- Expect: CHECK rejects unknown entity_type.
--   INSERT INTO pinned_entities (entity_type, entity_id) VALUES ('bogus', 1);
--   → ERROR: violates check constraint
--
-- Expect: partial-unique allows re-pinning after soft-delete (same triple).
--   INSERT INTO pinned_entities (entity_type, entity_id) VALUES ('weekly_retro', 1);
--   UPDATE pinned_entities SET deleted_at = NOW() WHERE entity_type = 'weekly_retro' AND entity_id = 1;
--   INSERT INTO pinned_entities (entity_type, entity_id) VALUES ('weekly_retro', 1);  -- succeeds
