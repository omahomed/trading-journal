-- ============================================================================
-- Migration 013: trading_journal.nlv_source — provenance tag for End NLV
-- ============================================================================
-- Daily Routine's End NLV field can now be auto-filled from IBKR Flex Query.
-- This column records *where* a row's NLV came from, so future investigations
-- ("why is this number off?") can tell auto-pulled rows from manually-entered
-- ones at a glance. The dashboard does not surface this field — it's
-- diagnostic-only.
--
-- Allowed values:
--   'manual'        — user typed it (default; pre-IBKR rows + fallback rows)
--   'ibkr_auto'     — auto-filled from IBKR and saved unchanged
--   'ibkr_override' — auto-filled from IBKR but user typed over it before save
--
-- Default = 'manual' so existing rows backfill cleanly. db_layer.save_journal_entry
-- carries a try/except fallback that omits the column if it doesn't exist yet,
-- so the app keeps working before this migration runs.
-- ============================================================================

ALTER TABLE trading_journal
    ADD COLUMN IF NOT EXISTS nlv_source VARCHAR(20) NOT NULL DEFAULT 'manual';

-- Drop any pre-existing version of the constraint so re-runs are idempotent
-- (Postgres < 17 doesn't support ADD CONSTRAINT IF NOT EXISTS for CHECKs).
ALTER TABLE trading_journal
    DROP CONSTRAINT IF EXISTS trading_journal_nlv_source_check;

ALTER TABLE trading_journal
    ADD CONSTRAINT trading_journal_nlv_source_check
    CHECK (nlv_source IN ('manual', 'ibkr_auto', 'ibkr_override'));
