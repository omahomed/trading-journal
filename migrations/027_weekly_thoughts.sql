-- ============================================================================
-- Migration 027: Weekly Thoughts column on weekly_retros — Phase 3
-- ============================================================================
-- Adds a single TEXT column to hold the user's weekly reflection prose.
-- Stored as HTML (the WeeklyThoughts component is contentEditable + a
-- formatting toolbar wired to document.execCommand). DOMPurify on the
-- frontend sanitizes pasted content to a small inline-tag whitelist
-- (b/i/u/s/strike/em/strong/a/br/p/div/span) so the column never receives
-- script tags, event handlers, or arbitrary classes/styles.
--
-- NOT NULL with DEFAULT '' so the canonical "no thoughts yet" value is the
-- empty string. Every existing weekly_retros row gets '' on column add,
-- and any future INSERT that omits the field also defaults to ''. Frontend
-- treats '' as "show placeholder" via a positioned div with pointer-events:
-- none over the empty editor.
--
-- No CHECK constraint — HTML content is intentionally free-form. The
-- frontend sanitizer is the security boundary; the column itself is just
-- text storage.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

ALTER TABLE weekly_retros
    ADD COLUMN IF NOT EXISTS weekly_thoughts TEXT NOT NULL DEFAULT '';


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
-- Expect: column exists with the expected type and default.
--   SELECT column_name, data_type, is_nullable, column_default
--     FROM information_schema.columns
--    WHERE table_name = 'weekly_retros' AND column_name = 'weekly_thoughts';
--
-- Expect: every existing row has '' (the DEFAULT applies on add).
--   SELECT COUNT(*) FROM weekly_retros WHERE weekly_thoughts IS NULL;
--   → 0
--   SELECT COUNT(*) FROM weekly_retros WHERE weekly_thoughts = '';
--   → COUNT(*) of the table prior to first user write
