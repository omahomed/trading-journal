-- ============================================================================
-- Migration 014: trading_journal.holdings_source — provenance tag for Total Holdings
-- ============================================================================
-- Sibling of nlv_source (migration 013). Daily Routine's Total Holdings field
-- can now be auto-filled from IBKR Flex Query's position_value (= stock +
-- options + bonds + other non-cash position categories). This column records
-- where a row's Total Holdings came from. Dashboard does not surface it —
-- diagnostic-only, same contract as nlv_source.
--
-- Allowed values:
--   'manual'        — user typed it (default; pre-IBKR rows + IBKR-failure rows)
--   'ibkr_auto'     — auto-filled from IBKR and saved unchanged
--   'ibkr_override' — auto-filled from IBKR but user typed over it before save
--
-- Independent of nlv_source — the user can accept the IBKR NLV but override
-- Holdings (or vice versa), so each field carries its own source tag. Default
-- 'manual' so existing rows backfill cleanly. save_journal_entry carries a
-- try/except fallback that omits the column if it doesn't exist yet.
-- ============================================================================

ALTER TABLE trading_journal
    ADD COLUMN IF NOT EXISTS holdings_source VARCHAR(20) NOT NULL DEFAULT 'manual';

-- Idempotent: drop-then-re-add lets re-runs pass since Postgres < 17 doesn't
-- support ADD CONSTRAINT IF NOT EXISTS for CHECKs.
ALTER TABLE trading_journal
    DROP CONSTRAINT IF EXISTS trading_journal_holdings_source_check;

ALTER TABLE trading_journal
    ADD CONSTRAINT trading_journal_holdings_source_check
    CHECK (holdings_source IN ('manual', 'ibkr_auto', 'ibkr_override'));
