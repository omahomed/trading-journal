-- ============================================================================
-- Migration 052: trading_journal.game_plan — pre-commit intent for tomorrow
-- ============================================================================
-- New free-text column captured via the Daily Journal shell's "Game Plan"
-- section. Writing model: the plan lives on day X's journal row and
-- represents intent for the NEXT trading day. Editable window is enforced
-- at the API layer (POST /api/journal/game-plan), not in a DB constraint —
-- lockdown = start of the next weekday after X (Fri plans stay editable
-- through Sunday 23:59 CT). Historical rows / retroactive edits still work
-- via /api/journal/edit (which does not enforce the window; used for
-- migrations, imports, and Manage Logs backfill).
--
-- Nullable, no default. Empty string vs NULL both render as "unwritten"
-- on the page. `save_journal_entry` in db_layer.py carries the standard
-- try/except progressive fallback so pre-migration DBs keep working
-- during the rollout window.
-- ============================================================================

ALTER TABLE trading_journal
    ADD COLUMN IF NOT EXISTS game_plan TEXT;

-- ============================================================================
-- Verification (manual, after COMMIT)
-- ============================================================================
--   SELECT column_name, data_type, is_nullable
--     FROM information_schema.columns
--    WHERE table_name = 'trading_journal' AND column_name = 'game_plan';
--   -- expect: game_plan | text | YES
-- ============================================================================
