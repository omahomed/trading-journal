-- ============================================================================
-- Migration 016: instrument_type + multiplier — first-class options support
-- ============================================================================
-- The LIFO engine and Trade Journal display layer have always treated rows as
-- 1× per-share equity. Equity options need a 100× multiplier on cost basis,
-- realized P&L, unrealized P&L, and notional value — every dollar amount we
-- show for an option trade is currently off by 100×.
--
-- Shape:
--   - instrument_type: 'STOCK' | 'OPTION' (room to grow: FUTURE, etc.)
--   - multiplier:      contract size; 1 for stocks, 100 for equity options
--
-- Why store both columns instead of re-detecting from ticker on every read:
--   1. Locks contract terms at trade time (mini options, futures options would
--      diverge from the 100× default).
--   2. Read paths (NLV, exposure, ACS, risk_manager) stay simple — one column
--      lookup, no regex parsing scattered across the codebase.
--   3. Backfill below uses the same ticker pattern that _is_option_ticker()
--      already recognises, so existing rows heal without code changes.
-- ============================================================================

ALTER TABLE trades_summary
    ADD COLUMN IF NOT EXISTS instrument_type VARCHAR(10) NOT NULL DEFAULT 'STOCK',
    ADD COLUMN IF NOT EXISTS multiplier      NUMERIC(8, 2) NOT NULL DEFAULT 1;

ALTER TABLE trades_details
    ADD COLUMN IF NOT EXISTS instrument_type VARCHAR(10) NOT NULL DEFAULT 'STOCK',
    ADD COLUMN IF NOT EXISTS multiplier      NUMERIC(8, 2) NOT NULL DEFAULT 1;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.constraint_column_usage
        WHERE constraint_name = 'trades_summary_instrument_type_check'
    ) THEN
        ALTER TABLE trades_summary
            ADD CONSTRAINT trades_summary_instrument_type_check
            CHECK (instrument_type IN ('STOCK', 'OPTION'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.constraint_column_usage
        WHERE constraint_name = 'trades_details_instrument_type_check'
    ) THEN
        ALTER TABLE trades_details
            ADD CONSTRAINT trades_details_instrument_type_check
            CHECK (instrument_type IN ('STOCK', 'OPTION'));
    END IF;
END $$;

-- Backfill: any existing row whose ticker matches the readable option format
-- (`UNDERLYING YYMMDD $STRIKE C|P`) gets flagged as an option with the
-- standard 100× equity-option multiplier. Same regex shape that
-- api/main.py:_is_option_ticker() already uses for live-price routing.
UPDATE trades_summary
   SET instrument_type = 'OPTION',
       multiplier      = 100
 WHERE instrument_type = 'STOCK'
   AND ticker ~ '^\S+ \d{6} \$[0-9.]+(C|P)$';

UPDATE trades_details
   SET instrument_type = 'OPTION',
       multiplier      = 100
 WHERE instrument_type = 'STOCK'
   AND ticker ~ '^\S+ \d{6} \$[0-9.]+(C|P)$';

CREATE INDEX IF NOT EXISTS idx_summary_instrument_type
    ON trades_summary (instrument_type);
