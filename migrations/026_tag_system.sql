-- ============================================================================
-- Migration 026: Polymorphic user-created tag system — Phase 1
-- ============================================================================
-- Adds a portfolio-scoped `tags` table and a polymorphic `tag_assignments`
-- join table. Phase 1 mounts on Weekly Retro only (entity_type =
-- 'weekly_retro'). Daily journal (Phase 7) and trades_summary (Phase 8) reuse
-- the same tables/endpoints with different entity_type values — schema is
-- already future-proofed via the CHECK constraint.
--
-- The Phase 0 weekly_retros migration (025) introduced the partial-unique +
-- soft-delete pattern for the first time in this codebase. This migration
-- reuses that idiom for `tags` so users can recycle a tag name (delete
-- "drawdown", later create "drawdown" again with a different color).
--
-- Idempotent restore: tag_assignments uses the same partial-unique pattern,
-- and the create_tag_assignment helper in db_layer detects a soft-deleted row
-- for the same (tag_id, entity_type, entity_id) triple and REVIVES it instead
-- of inserting a duplicate. This keeps id stability across detach → reattach
-- cycles, so any future per-assignment metadata (e.g. "first attached at"
-- audit) is preserved.
--
-- Cascade semantics:
--   tag_assignments.tag_id → tags.id ON DELETE CASCADE
--     Hard delete of a tag (rare; soft-delete is the normal path) wipes its
--     assignments. Soft-delete leaves the assignments live but invisible —
--     load_tag_assignments LEFT JOINs tags filtered by tags.deleted_at IS
--     NULL, so a soft-deleted tag's pills disappear from the UI without us
--     having to mass-update child rows.
--
-- No audit trigger — matches strategies (019), cash_transactions (009), and
-- lot_closures (017). Audit_trail is trade-centric in this codebase.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. tags — per-portfolio palette
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tags (
    id            SERIAL          PRIMARY KEY,
    user_id       UUID            NOT NULL REFERENCES users(id) ON DELETE RESTRICT
                                  DEFAULT NULLIF(current_setting('app.user_id', true), '')::uuid,
    portfolio_id  INTEGER         NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    name          VARCHAR(60)     NOT NULL,
    color         VARCHAR(20)     NOT NULL,
    created_at    TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ     NOT NULL DEFAULT now(),
    deleted_at    TIMESTAMPTZ
);

-- Case-insensitive uniqueness on live rows only. "Drawdown" and "drawdown"
-- collide; display case is preserved on first create. The partial predicate
-- lets a soft-deleted name be recycled by a subsequent create — same idiom
-- used in migration 025 for weekly_retros (portfolio_id, week_start).
CREATE UNIQUE INDEX IF NOT EXISTS uq_tags_portfolio_name_live
    ON tags (portfolio_id, LOWER(name)) WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_tags_user_portfolio_live
    ON tags (user_id, portfolio_id) WHERE deleted_at IS NULL;

ALTER TABLE tags ENABLE ROW LEVEL SECURITY;
ALTER TABLE tags FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON tags;
CREATE POLICY tenant_isolation ON tags FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ----------------------------------------------------------------------------
-- 2. tag_assignments — polymorphic many-to-many
-- ----------------------------------------------------------------------------
-- entity_id is INTEGER because every Phase 1 target uses SERIAL PKs:
--   weekly_retro     → weekly_retros.id     (migration 025, Phase 0)
--   daily_journal    → trading_journal.id   (existing; Phase 7 mount)
--   trades_summary   → trades_summary.id    (existing; Phase 8 mount)
-- No TEXT-with-cast complexity needed.
--
-- Polymorphic FK validation is app-layer only (Postgres has no built-in
-- polymorphic FKs). RLS prevents cross-tenant injection; app validates
-- entity_type against the closed set; CHECK constraint pins the vocabulary.
--
-- Decision (Phase 1): entity_type is restricted to the three currently-known
-- values. Per-ticker grade attachment ('weekly_retro_ticker_grade') is
-- DELIBERATELY NOT pre-baked — adding it later requires only an ALTER TABLE
-- DROP CONSTRAINT + ADD CONSTRAINT. Keep the schema honest to current intent.
CREATE TABLE IF NOT EXISTS tag_assignments (
    id            SERIAL          PRIMARY KEY,
    user_id       UUID            NOT NULL REFERENCES users(id) ON DELETE RESTRICT
                                  DEFAULT NULLIF(current_setting('app.user_id', true), '')::uuid,
    portfolio_id  INTEGER         NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    tag_id        INTEGER         NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    entity_type   VARCHAR(20)     NOT NULL
                                  CHECK (entity_type IN ('weekly_retro','daily_journal','trades_summary')),
    entity_id     INTEGER         NOT NULL,
    created_at    TIMESTAMPTZ     NOT NULL DEFAULT now(),
    deleted_at    TIMESTAMPTZ
);

-- Live triple uniqueness — re-attaching the same tag to the same entity is a
-- no-op via the idempotent restore in create_tag_assignment.
CREATE UNIQUE INDEX IF NOT EXISTS uq_tag_assignments_triple_live
    ON tag_assignments (entity_type, entity_id, tag_id) WHERE deleted_at IS NULL;

-- "What tags on this entry?" — primary read path.
CREATE INDEX IF NOT EXISTS idx_tag_assignments_entity_live
    ON tag_assignments (entity_type, entity_id) WHERE deleted_at IS NULL;

-- "What entries have this tag?" — Phase 3 reverse-lookup path.
CREATE INDEX IF NOT EXISTS idx_tag_assignments_tag_live
    ON tag_assignments (tag_id) WHERE deleted_at IS NULL;

-- Cross-portfolio reverse-lookup + RLS plan optimization (avoids JOIN to
-- tags for tenant scoping).
CREATE INDEX IF NOT EXISTS idx_tag_assignments_portfolio_live
    ON tag_assignments (portfolio_id) WHERE deleted_at IS NULL;

ALTER TABLE tag_assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE tag_assignments FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON tag_assignments;
CREATE POLICY tenant_isolation ON tag_assignments FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
-- Expect: empty tables.
--   SELECT count(*) FROM tags;
--   SELECT count(*) FROM tag_assignments;
--
-- Expect: RLS enabled with tenant_isolation policy on both.
--   SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class
--     WHERE relname IN ('tags','tag_assignments');
--
-- Expect: case-insensitive unique collides on insert of same name diff case.
--   INSERT INTO tags (portfolio_id, name, color) VALUES (1, 'Drawdown', 'rose');
--   INSERT INTO tags (portfolio_id, name, color) VALUES (1, 'drawdown', 'sky');
--   → ERROR: duplicate key value violates unique constraint "uq_tags_portfolio_name_live"
--
-- Expect: recycle works after soft-delete.
--   UPDATE tags SET deleted_at = NOW() WHERE LOWER(name) = 'drawdown' AND portfolio_id = 1;
--   INSERT INTO tags (portfolio_id, name, color) VALUES (1, 'drawdown', 'sky');  -- succeeds
--
-- Expect: CHECK rejects unknown entity_type.
--   INSERT INTO tag_assignments (portfolio_id, tag_id, entity_type, entity_id) VALUES (1, 1, 'bogus', 1);
--   → ERROR: violates check constraint
