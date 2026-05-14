-- ============================================================================
-- Migration 028: weekly_retro_snapshots — Phase 4 Weekly Snapshot
-- ============================================================================
-- New table for image attachments on weekly retros. Users upload charts /
-- screenshots / dashboards during the week and review them in the retro.
-- Phase 4 ships the upload + display + delete cycle ("Phase A" of the
-- snapshot feature); captions, reorder, and programmatic chart rendering
-- are explicit follow-up work (the columns are pre-provisioned so those
-- phases need only schema-additive migrations, not schema-changing ones).
--
-- Storage model: bytes live in Cloudflare R2 (reuse the existing
-- infrastructure that already serves trade chart images + daily-report
-- EOD snapshots). The DB holds metadata only — storage_ref is the R2
-- object key (e.g., "weekly_retros/123/abc-def-uuid.png"). Browser fetches
-- bytes directly from R2 via R2_PUBLIC_URL; backend is not in the serving
-- hot path.
--
-- Why a new table instead of piggybacking trade_images (which the daily
-- card uses via synthetic trade_id = "EOD-{day}")? Three reasons:
--   - Need a proper FK to weekly_retros so retro deletion cascades to its
--     snapshots; trade_images has no semantic FK.
--   - Future caption/sort_order/dimensions columns line up with Phase B/C
--     features; trade_images lacks them all.
--   - Soft-delete via deleted_at (snapshots are user-attached annotations
--     where accidental deletion warrants the ability to recover); the
--     trade_images table hard-deletes.
--
-- The pre-Phase-4 audit (Section 4) covered the rationale in full.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

CREATE TABLE IF NOT EXISTS weekly_retro_snapshots (
    id              SERIAL          PRIMARY KEY,
    -- Founder-fallback DEFAULT mirrors the audit-trigger-safe pattern from
    -- migration 024. If app.user_id is unset (e.g., during a migration
    -- backfill), the row gets the founder's UUID instead of NULL-violating.
    user_id         UUID            NOT NULL DEFAULT (
        COALESCE(
            NULLIF(current_setting('app.user_id', true), '')::uuid,
            'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
        )
    ),
    -- ON DELETE CASCADE is safe here because the parent retro is itself
    -- user-scoped — the cascade only ever fires within a single tenant,
    -- never across.
    weekly_retro_id INTEGER         NOT NULL REFERENCES weekly_retros(id) ON DELETE CASCADE,
    -- R2 object key. Composed as "weekly_retros/{retro_id}/{uuid4}.{ext}".
    storage_ref     TEXT            NOT NULL,
    file_name       VARCHAR(255),
    mime_type       VARCHAR(64),
    file_size_bytes INTEGER,
    width           INTEGER,
    height          INTEGER,
    -- Pre-provisioned for the Phase 4-followup reorder feature. Default 0
    -- means v1 inserts always get the same position; rendering orders by
    -- (sort_order, created_at) which collapses to created_at for v1.
    sort_order      INTEGER         NOT NULL DEFAULT 0,
    -- Pre-provisioned for the Phase 4-followup caption feature. Default ''
    -- so v1 endpoint doesn't have to set it.
    caption         TEXT            NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    deleted_at      TIMESTAMPTZ
);

-- List-by-retro is the dominant query (loading the snapshot grid for an
-- expanded retro). Composite index on (retro, sort_order, created_at)
-- serves the ORDER BY directly. Partial WHERE deleted_at IS NULL because
-- soft-deleted rows are never listed.
CREATE INDEX IF NOT EXISTS idx_wretro_snapshots_retro
    ON weekly_retro_snapshots (weekly_retro_id, sort_order, created_at)
    WHERE deleted_at IS NULL;

-- Per-user scan for future "all my snapshots" features (e.g., cross-week
-- search). Cheap to maintain at our scale.
CREATE INDEX IF NOT EXISTS idx_wretro_snapshots_user
    ON weekly_retro_snapshots (user_id);

ALTER TABLE weekly_retro_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_retro_snapshots FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON weekly_retro_snapshots;
CREATE POLICY tenant_isolation ON weekly_retro_snapshots FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
-- Expect: table exists, no rows yet (frontend lands in same PR).
--   SELECT count(*) FROM weekly_retro_snapshots;
--
-- Expect: RLS enabled + forced.
--   SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class
--     WHERE relname = 'weekly_retro_snapshots';
--
-- Expect: cascade fires on retro delete.
--   DELETE FROM weekly_retros WHERE id = <some_id>;
--   SELECT count(*) FROM weekly_retro_snapshots WHERE weekly_retro_id = <some_id>;
--   → 0
