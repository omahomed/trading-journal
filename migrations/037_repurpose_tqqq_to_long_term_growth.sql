-- migrations/037_repurpose_tqqq_to_long_term_growth.sql
--
-- Repurpose the TQQQ Strategy portfolio into "Long-Term Growth":
--   1. Wipe the historical TQQQ trades (~6 campaigns). FK cascades
--      handle trades_details, lot_closures, daily_captures, journal
--      entries, etc. — the existing ON DELETE CASCADE wiring on
--      portfolio_id covers all child tables.
--   2. Rename the portfolio row in place. The portfolio's id is
--      preserved through the rename, so:
--        - The user's stored active-portfolio id in localStorage
--          continues to resolve.
--        - No need to touch trades_summary.portfolio_id values.
--   3. Reset starting_capital + reset_date to seed the new portfolio's
--      lifetime baseline.
--
-- Migration 038 (next) backfills the strategies.allowed_portfolio_names
-- column referencing the new 'Long-Term Growth' name, so 037 must apply
-- first. The numbered migration runner enforces this ordering.
--
-- Idempotent: re-running after the rename is a no-op. The lookup
-- `WHERE name = 'TQQQ Strategy'` returns no row, the NOTICE fires, and
-- the function returns without modifying anything.

DO $$
DECLARE
    tqqq_id    INTEGER;
    trade_count INTEGER;
BEGIN
    SELECT id INTO tqqq_id
      FROM portfolios
     WHERE name = 'TQQQ Strategy';

    IF tqqq_id IS NULL THEN
        RAISE NOTICE 'No TQQQ Strategy portfolio found — migration 037 is a no-op';
        RETURN;
    END IF;

    SELECT COUNT(*) INTO trade_count
      FROM trades_summary
     WHERE portfolio_id = tqqq_id;

    -- Wipe trades (cascade handles all child tables).
    DELETE FROM trades_summary WHERE portfolio_id = tqqq_id;

    -- Rename + reset metadata.
    UPDATE portfolios
       SET name             = 'Long-Term Growth',
           starting_capital = 4500.00,
           reset_date       = '2026-01-01'
     WHERE id = tqqq_id;

    RAISE NOTICE 'Repurposed TQQQ Strategy → Long-Term Growth (% trades wiped)',
                 trade_count;
END $$;
