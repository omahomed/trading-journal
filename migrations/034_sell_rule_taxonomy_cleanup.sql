-- migrations/034_sell_rule_taxonomy_cleanup.sql
--
-- Backfill trades_summary.sell_rule and trades_details.rule (SELL rows
-- only) onto the new 13-rule canonical sell-rule taxonomy. The seven
-- source→target mappings are user-locked decisions made during taxonomy
-- review.
--
-- Why we also touch trades_details: db_layer.mirror_detail_edit_to_summary
-- re-derives trades_summary.sell_rule from the final SELL detail's
-- rule on every edit_transaction call. If we updated only the summary,
-- any subsequent user edit on those campaigns would revert the summary
-- to the stale rule string. The pre-migration audit found:
--   - trades_summary.sell_rule: 7 source values, 44 rows total
--   - trades_details.rule (action='SELL'):  same 7 source values,
--     73 rows total (campaigns can have multiple SELL transactions;
--     only the latest mirrors to summary)
--
-- The 233 'History' rows are intentionally outside the canonical
-- taxonomy and are preserved as-is.
--
-- CRITICAL ORDERING: Step 2 (sr13 Earnings Exit → sr10) MUST run before
-- Step 6 (sr4 Change of Character → sr13). Otherwise the sr4 rows land
-- in sr13 and then get converted to sr10, conflating Change of
-- Character with Earnings Exit.
--
-- Idempotent: re-running matches zero rows on every step because the
-- WHERE clauses reference the source values which no longer exist
-- after the first run.

DO $$
DECLARE
    s_count integer;
    d_count integer;
    leftover integer;
BEGIN
    -- ---------------------------------------------------------------
    -- Step 1: sr1.1 capital protection - hard stop → sr1 Capital Protection
    -- ---------------------------------------------------------------
    SELECT count(*) INTO s_count FROM trades_summary
     WHERE sell_rule = 'sr1.1 capital protection - hard stop' AND deleted_at IS NULL;
    SELECT count(*) INTO d_count FROM trades_details
     WHERE rule = 'sr1.1 capital protection - hard stop' AND action = 'SELL' AND deleted_at IS NULL;
    RAISE NOTICE 'Step 1 (sr1.1 -> sr1 Capital Protection): summary=% details=%', s_count, d_count;

    UPDATE trades_summary
       SET sell_rule = 'sr1 Capital Protection'
     WHERE sell_rule = 'sr1.1 capital protection - hard stop';

    UPDATE trades_details
       SET rule = 'sr1 Capital Protection'
     WHERE rule = 'sr1.1 capital protection - hard stop' AND action = 'SELL';

    -- ---------------------------------------------------------------
    -- Step 2: sr13 Earnings Exit → sr10 Earnings Exit
    --   MUST run BEFORE Step 6, otherwise sr4 ends up as sr10.
    -- ---------------------------------------------------------------
    SELECT count(*) INTO s_count FROM trades_summary
     WHERE sell_rule = 'sr13 Earnings Exit' AND deleted_at IS NULL;
    SELECT count(*) INTO d_count FROM trades_details
     WHERE rule = 'sr13 Earnings Exit' AND action = 'SELL' AND deleted_at IS NULL;
    RAISE NOTICE 'Step 2 (sr13 Earnings -> sr10 Earnings Exit): summary=% details=%', s_count, d_count;

    UPDATE trades_summary
       SET sell_rule = 'sr10 Earnings Exit'
     WHERE sell_rule = 'sr13 Earnings Exit';

    UPDATE trades_details
       SET rule = 'sr10 Earnings Exit'
     WHERE rule = 'sr13 Earnings Exit' AND action = 'SELL';

    -- ---------------------------------------------------------------
    -- Step 3: sr15 BE Stop Out → sr11 BE Stop Out
    -- ---------------------------------------------------------------
    SELECT count(*) INTO s_count FROM trades_summary
     WHERE sell_rule = 'sr15 BE Stop Out (moved at +10%)' AND deleted_at IS NULL;
    SELECT count(*) INTO d_count FROM trades_details
     WHERE rule = 'sr15 BE Stop Out (moved at +10%)' AND action = 'SELL' AND deleted_at IS NULL;
    RAISE NOTICE 'Step 3 (sr15 -> sr11 BE Stop Out): summary=% details=%', s_count, d_count;

    UPDATE trades_summary
       SET sell_rule = 'sr11 BE Stop Out (moved at +10%)'
     WHERE sell_rule = 'sr15 BE Stop Out (moved at +10%)';

    UPDATE trades_details
       SET rule = 'sr11 BE Stop Out (moved at +10%)'
     WHERE rule = 'sr15 BE Stop Out (moved at +10%)' AND action = 'SELL';

    -- ---------------------------------------------------------------
    -- Step 4: sr8 TQQQ Strategy Exit → sr12 TQQQ Strategy Exit
    -- ---------------------------------------------------------------
    SELECT count(*) INTO s_count FROM trades_summary
     WHERE sell_rule = 'sr8 TQQQ Strategy Exit' AND deleted_at IS NULL;
    SELECT count(*) INTO d_count FROM trades_details
     WHERE rule = 'sr8 TQQQ Strategy Exit' AND action = 'SELL' AND deleted_at IS NULL;
    RAISE NOTICE 'Step 4 (sr8 -> sr12 TQQQ Strategy Exit): summary=% details=%', s_count, d_count;

    UPDATE trades_summary
       SET sell_rule = 'sr12 TQQQ Strategy Exit'
     WHERE sell_rule = 'sr8 TQQQ Strategy Exit';

    UPDATE trades_details
       SET rule = 'sr12 TQQQ Strategy Exit'
     WHERE rule = 'sr8 TQQQ Strategy Exit' AND action = 'SELL';

    -- ---------------------------------------------------------------
    -- Step 5: sr16 Profit Taking → sr2 Selling into Strength
    -- ---------------------------------------------------------------
    SELECT count(*) INTO s_count FROM trades_summary
     WHERE sell_rule = 'sr16 Profit Taking' AND deleted_at IS NULL;
    SELECT count(*) INTO d_count FROM trades_details
     WHERE rule = 'sr16 Profit Taking' AND action = 'SELL' AND deleted_at IS NULL;
    RAISE NOTICE 'Step 5 (sr16 -> sr2 Selling into Strength): summary=% details=%', s_count, d_count;

    UPDATE trades_summary
       SET sell_rule = 'sr2 Selling into Strength'
     WHERE sell_rule = 'sr16 Profit Taking';

    UPDATE trades_details
       SET rule = 'sr2 Selling into Strength'
     WHERE rule = 'sr16 Profit Taking' AND action = 'SELL';

    -- ---------------------------------------------------------------
    -- Step 6: sr4 Change of Character → sr13 Change of Character
    --   MUST run AFTER Step 2 (above).
    -- ---------------------------------------------------------------
    SELECT count(*) INTO s_count FROM trades_summary
     WHERE sell_rule = 'sr4 Change of Character' AND deleted_at IS NULL;
    SELECT count(*) INTO d_count FROM trades_details
     WHERE rule = 'sr4 Change of Character' AND action = 'SELL' AND deleted_at IS NULL;
    RAISE NOTICE 'Step 6 (sr4 -> sr13 Change of Character): summary=% details=%', s_count, d_count;

    UPDATE trades_summary
       SET sell_rule = 'sr13 Change of Character'
     WHERE sell_rule = 'sr4 Change of Character';

    UPDATE trades_details
       SET rule = 'sr13 Change of Character'
     WHERE rule = 'sr4 Change of Character' AND action = 'SELL';

    -- ---------------------------------------------------------------
    -- Step 7: sr9 Breakout Failure → sr9 Failed Breakout (rename)
    -- ---------------------------------------------------------------
    SELECT count(*) INTO s_count FROM trades_summary
     WHERE sell_rule = 'sr9 Breakout Failure' AND deleted_at IS NULL;
    SELECT count(*) INTO d_count FROM trades_details
     WHERE rule = 'sr9 Breakout Failure' AND action = 'SELL' AND deleted_at IS NULL;
    RAISE NOTICE 'Step 7 (sr9 Breakout Failure -> sr9 Failed Breakout): summary=% details=%', s_count, d_count;

    UPDATE trades_summary
       SET sell_rule = 'sr9 Failed Breakout'
     WHERE sell_rule = 'sr9 Breakout Failure';

    UPDATE trades_details
       SET rule = 'sr9 Failed Breakout'
     WHERE rule = 'sr9 Breakout Failure' AND action = 'SELL';

    -- ---------------------------------------------------------------
    -- Final verification: trades_summary should have zero non-canonical
    -- sell_rule values (excluding 'History', which is intentionally
    -- preserved per user direction).
    -- ---------------------------------------------------------------
    SELECT count(*) INTO leftover
      FROM trades_summary
     WHERE sell_rule IS NOT NULL AND sell_rule <> ''
       AND deleted_at IS NULL
       AND sell_rule NOT IN (
           'sr1 Capital Protection',
           'sr2 Selling into Strength',
           'sr3 Portfolio Management',
           'sr4 Time Stop',
           'sr5 Climax Top',
           'sr6 8e Momentum Trim',
           'sr7 Holding Winners - 21e Violation',
           'sr8 Big Cushion Sell Rule',
           'sr9 Failed Breakout',
           'sr10 Earnings Exit',
           'sr11 BE Stop Out (moved at +10%)',
           'sr12 TQQQ Strategy Exit',
           'sr13 Change of Character',
           'History'
       );
    RAISE NOTICE 'trades_summary non-canonical remaining: %', leftover;

    -- Detail-level leftovers (action='SELL' only). May be non-zero if
    -- rules outside the user's locked mapping exist on detail rows
    -- (e.g. 'sr10 Scale-Out T1 (-3%)', 'IBKR'). Reported for visibility;
    -- not an error.
    SELECT count(*) INTO leftover
      FROM trades_details
     WHERE rule IS NOT NULL AND rule <> ''
       AND action = 'SELL'
       AND deleted_at IS NULL
       AND rule NOT IN (
           'sr1 Capital Protection',
           'sr2 Selling into Strength',
           'sr3 Portfolio Management',
           'sr4 Time Stop',
           'sr5 Climax Top',
           'sr6 8e Momentum Trim',
           'sr7 Holding Winners - 21e Violation',
           'sr8 Big Cushion Sell Rule',
           'sr9 Failed Breakout',
           'sr10 Earnings Exit',
           'sr11 BE Stop Out (moved at +10%)',
           'sr12 TQQQ Strategy Exit',
           'sr13 Change of Character',
           'History'
       );
    RAISE NOTICE 'trades_details (SELL) non-canonical remaining: % (outside locked mapping, left as-is)', leftover;
END $$;
