-- ============================================
-- MIGRATION: Add grade column to trades_summary
-- 1-5 star rating per trade campaign. NULL = unrated.
-- ============================================

ALTER TABLE trades_summary
    ADD COLUMN IF NOT EXISTS grade SMALLINT
    CHECK (grade IS NULL OR (grade BETWEEN 1 AND 5));

-- Verify
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'trades_summary' AND column_name = 'grade';
