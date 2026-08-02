-- ============================================================================
-- Migration 057: Watch List column on weekly_retros
-- ============================================================================
-- Adds a second TEXT column, mirroring migration 027 (weekly_thoughts), to
-- hold the user's weekend watch-list prose (IBD 50 / Growth 250 / Big Cap 20
-- notes + inline pasted charts). Stored as HTML; the frontend WatchList
-- component reuses the shared ThoughtsEditor (contentEditable + DOMPurify
-- sanitizer) so the exact same inline-tag whitelist and image-paste path
-- apply — no separate security surface.
--
-- Images pasted here upload to the SAME R2 prefix as Weekly Thoughts
-- (weekly_retros/{retro_id}/thoughts/…) since they're both attachments of
-- the same weekly_retros row. Future R2 cleanup by scanning the concatenation
-- of weekly_thoughts + watch_list HTML remains straightforward.
--
-- NOT NULL DEFAULT '' — same treatment as weekly_thoughts. Every existing
-- row backfills to the empty string, matching the "show placeholder" logic
-- in the frontend for empty editors.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

ALTER TABLE weekly_retros
    ADD COLUMN IF NOT EXISTS watch_list TEXT NOT NULL DEFAULT '';


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
--   SELECT column_name, data_type, is_nullable, column_default
--     FROM information_schema.columns
--    WHERE table_name = 'weekly_retros' AND column_name = 'watch_list';
--
--   SELECT COUNT(*) FROM weekly_retros WHERE watch_list IS NULL;
--   → 0
--   SELECT COUNT(*) FROM weekly_retros WHERE watch_list = '';
--   → COUNT(*) of the table prior to first user write
