-- ============================================================================
-- Migration 015: trading_journal.mct_display_day_num
-- ============================================================================
-- Snapshot the MCT V11 "D{N}" badge value into the journal row at the moment
-- the user runs Daily Routine. Pairs with the existing market_cycle column
-- (state name) so the Daily Journal page can render the badge directly from
-- the row instead of replaying the engine on every page visit.
--
-- Anchoring rules (computed by api/main.py:_compute_mct_state_with_day_num):
--   POWERTREND          → bars since STEP_8_POWERTREND_ON (pt_on_idx)
--   UPTREND / RALLY MODE → bars since cycle STEP_0 (cycle_start_idx)
--   CORRECTION          → NULL (no day count rendered)
--
-- Nullable: legacy rows that pre-date this migration stay NULL until backfill.
-- The Daily Journal badge falls back to "STATE" with no D-suffix when NULL.
-- ============================================================================

ALTER TABLE trading_journal
    ADD COLUMN IF NOT EXISTS mct_display_day_num INTEGER;
