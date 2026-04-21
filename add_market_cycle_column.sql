-- ============================================
-- MIGRATION: Add market_cycle column to trading_journal
-- Run this once. Column captures the NASDAQ cycle state
-- (POWERTREND / UPTREND / RALLY MODE / CORRECTION) at the
-- time the journal entry is saved.
-- ============================================

ALTER TABLE trading_journal
    ADD COLUMN IF NOT EXISTS market_cycle VARCHAR(50);

-- Verify the column
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'trading_journal' AND column_name = 'market_cycle';
