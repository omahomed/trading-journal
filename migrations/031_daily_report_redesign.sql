-- ============================================================================
-- Migration 031: daily_report_redesign — Phase 7 schema
-- ============================================================================
-- Two schema changes that together unblock the Daily Report architectural
-- treatment that mirrors Weekly Retro:
--
--   1. ALTER TABLE trading_journal ADD COLUMN daily_thoughts TEXT — backs
--      the new rich-text "Daily Thoughts" editor (shared <ThoughtsEditor>
--      with Weekly Retro). The existing `lowlights` column stays put and
--      becomes the "Daily Recap" markdown editor (rename only, no data
--      migration needed).
--
--   2. CREATE TABLE daily_journal_captures — user-uploaded image gallery
--      for daily journal entries. Mirrors weekly_retro_snapshots from
--      migration 028 byte-for-byte except for the FK target (parent is
--      trading_journal, not weekly_retros).
--
-- Why a separate table instead of repurposing trade_images (which the
-- daily card uses today via synthetic trade_id = "EOD-{day}")? Same three
-- reasons that motivated 028:
--   - Proper FK to trading_journal so journal-row CASCADE works.
--   - Future caption/sort_order/dimensions columns line up with planned
--     follow-up features; trade_images lacks them all.
--   - Soft-delete via deleted_at — daily captures are user attachments
--     where accidental deletion warrants recoverability. trade_images
--     hard-deletes (no deleted_at column).
--
-- Backfill of existing user uploads (image_type='eod_note' rows in
-- trade_images) is in migration 032, separated so a backfill failure
-- doesn't roll back the schema add.
--
-- Audit triggers: trading_journal does NOT have an audit-trail INSERT
-- trigger (the one on `trades_summary` per migration 022 is the only such
-- trigger). So the migration-024 safe-pattern (founder UUID fallback) is
-- needed only on the captures table's user_id DEFAULT — applied below.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1) trading_journal.daily_thoughts — rich-text editor body.
-- ----------------------------------------------------------------------------
-- NOT NULL DEFAULT '' so existing rows backfill silently. Matches the
-- weekly_retros.weekly_thoughts column (migration 027) shape.
ALTER TABLE trading_journal
    ADD COLUMN IF NOT EXISTS daily_thoughts TEXT NOT NULL DEFAULT '';


-- ----------------------------------------------------------------------------
-- 2) daily_journal_captures — image gallery for daily journal entries.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_journal_captures (
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
    -- ON DELETE CASCADE is safe here because the parent journal is itself
    -- user-scoped — the cascade only ever fires within a single tenant.
    daily_journal_id INTEGER        NOT NULL REFERENCES trading_journal(id) ON DELETE CASCADE,
    -- R2 object key. Composed as "daily_journal/{journal_id}/{uuid4}.{ext}".
    storage_ref     TEXT            NOT NULL,
    file_name       VARCHAR(255),
    mime_type       VARCHAR(64),
    file_size_bytes INTEGER,
    width           INTEGER,
    height          INTEGER,
    -- Pre-provisioned for the follow-up reorder feature. Default 0 means
    -- v1 inserts always get the same position; rendering orders by
    -- (sort_order, created_at) which collapses to created_at for v1.
    sort_order      INTEGER         NOT NULL DEFAULT 0,
    -- Pre-provisioned for the follow-up caption feature. Default '' so v1
    -- endpoint doesn't have to set it.
    caption         TEXT            NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    deleted_at      TIMESTAMPTZ
);

-- List-by-journal is the dominant query (loading the capture grid for the
-- visible day). Composite index on (journal, sort_order, created_at)
-- serves the ORDER BY directly. Partial WHERE deleted_at IS NULL because
-- soft-deleted rows are never listed.
CREATE INDEX IF NOT EXISTS idx_djcaptures_journal
    ON daily_journal_captures (daily_journal_id, sort_order, created_at)
    WHERE deleted_at IS NULL;

-- Per-user scan for future cross-day search. Cheap to maintain.
CREATE INDEX IF NOT EXISTS idx_djcaptures_user
    ON daily_journal_captures (user_id);

ALTER TABLE daily_journal_captures ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_journal_captures FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON daily_journal_captures;
CREATE POLICY tenant_isolation ON daily_journal_captures FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
--   SELECT column_name FROM information_schema.columns
--    WHERE table_name = 'trading_journal' AND column_name = 'daily_thoughts';
--   → 1 row.
--
--   SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class
--    WHERE relname = 'daily_journal_captures';
--   → 1 row with both flags true.
--
--   SELECT count(*) FROM daily_journal_captures;
--   → 0 (backfill runs in migration 032).
