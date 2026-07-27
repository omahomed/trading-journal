-- ============================================================================
-- Migration 054: Reset Long-Term Growth to a fresh 2026-07-27 baseline.
-- ============================================================================
-- Second full reset (see migration 040 for the March 31 precedent). User's
-- capital-deployed number in the app has drifted from reality
-- ($52,450 actual vs. whatever the accumulated cash_transactions +
-- journal chain say). Simpler to nuke history and re-anchor than reconcile
-- 80 journal rows + 205 cash txns.
--
-- Divergence from migration 040 (which cutoff-filtered on open_date <
-- '2026-03-31'):
--
--   * Delete only CLOSED trades (status='CLOSED'). Open campaigns are
--     preserved with their historical cost basis + open_date. User's
--     framing: "positions carry over, the reset is about the dashboard
--     baseline." ACS keeps showing the 12 currently-open tickers (AMKR,
--     BE, CRWD, DDOG, DRAM, LLYX, MITK, NBIS, SOXL, SPYM, STX, VIAV)
--     with their real avg_entry.
--   * Wipe ALL cash_transactions (not date-filtered). The old chain is
--     inconsistent; we're restarting from a single $52,450 deposit.
--     Orphan-mirror sweep from migration 040 not needed — no post-
--     cutoff trades preserved with pre-cutoff cash mirrors.
--   * Wipe ALL trading_journal. Fresh baseline; today's journal will
--     read beg_nlv=0 + cash_change=52,450 → daily P&L = end_nlv - 52,450.
--   * Wipe daily_journal_captures + audit_trail scoped to LTG. Retained
--     tags (weekly_retros, retro_ticker_grades) are historical
--     performance metadata; NOT wiped (would lose research value).
--
-- Cascade map (unchanged from migration 040):
--   - trades_summary → trades_details       ON DELETE CASCADE
--   - trading_journal → daily_journal_captures ON DELETE CASCADE
--   - cash_transactions.trade_detail_id     ON DELETE SET NULL
--   - trade_images / trade_fundamentals / trade_lessons / lot_closures
--     / audit_trail carry trade_id as string with no FK; sweep explicitly.
--
-- Idempotency: re-run finds status='CLOSED' set empty (already deleted
-- last time), UPDATE portfolios lands on the same values, INSERT initial
-- deposit uses the same DO $$ IF EXISTS pattern. Safe to apply twice.
-- ============================================================================

DO $$
DECLARE
    ltg_id              INTEGER;
    cutoff              DATE     := '2026-07-27';
    new_capital         NUMERIC  := 52450;
    dead_count          INTEGER;
    ti_count            INTEGER;
    tf_count            INTEGER;
    tl_count            INTEGER;
    lc_count            INTEGER;
    at_count            INTEGER;
    cash_delete_count   INTEGER;
    trade_delete_count  INTEGER;
    tj_count            INTEGER;
    reseed_action       TEXT;
BEGIN
    SELECT id INTO ltg_id
      FROM portfolios
     WHERE name = 'Long-Term Growth';

    IF ltg_id IS NULL THEN
        RAISE NOTICE 'No Long-Term Growth portfolio found — migration 054 is a no-op';
        RETURN;
    END IF;

    -- Step 1: snapshot dead trade IDs (CLOSED only — OPEN preserved).
    CREATE TEMP TABLE dead_trades ON COMMIT DROP AS
      SELECT trade_id
        FROM trades_summary
       WHERE portfolio_id = ltg_id
         AND deleted_at IS NULL
         AND status = 'CLOSED';
    SELECT COUNT(*) INTO dead_count FROM dead_trades;

    -- Step 2: FK-less child tables scoped to dead trades.
    DELETE FROM trade_images
     WHERE portfolio_id = ltg_id
       AND trade_id IN (SELECT trade_id FROM dead_trades);
    GET DIAGNOSTICS ti_count = ROW_COUNT;

    DELETE FROM trade_fundamentals
     WHERE portfolio_id = ltg_id
       AND trade_id IN (SELECT trade_id FROM dead_trades);
    GET DIAGNOSTICS tf_count = ROW_COUNT;

    DELETE FROM trade_lessons
     WHERE portfolio_id = ltg_id
       AND trade_id IN (SELECT trade_id FROM dead_trades);
    GET DIAGNOSTICS tl_count = ROW_COUNT;

    DELETE FROM lot_closures
     WHERE portfolio_id = ltg_id
       AND trade_id IN (SELECT trade_id FROM dead_trades);
    GET DIAGNOSTICS lc_count = ROW_COUNT;

    -- audit_trail: sweep pre-cutoff rows. Preserving open trades means
    -- their audit history is still meaningful; only wipe rows for
    -- trade_ids we're actually deleting.
    DELETE FROM audit_trail
     WHERE portfolio_id = ltg_id
       AND trade_id IN (SELECT trade_id FROM dead_trades);
    GET DIAGNOSTICS at_count = ROW_COUNT;

    -- Step 3: full cash_transactions wipe. Reset means single-deposit
    -- restart; existing chain (mix of deposits, buy/sell mirrors,
    -- withdrawals) is discarded.
    DELETE FROM cash_transactions WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS cash_delete_count = ROW_COUNT;

    -- Step 4: delete CLOSED trades. Cascades to trades_details via
    -- (portfolio_id, trade_id) FK. Open trades preserved.
    DELETE FROM trades_summary
     WHERE portfolio_id = ltg_id
       AND status = 'CLOSED';
    GET DIAGNOSTICS trade_delete_count = ROW_COUNT;

    -- Step 5: full trading_journal wipe. Fresh baseline; cascades to
    -- daily_journal_captures via migration 031's FK.
    DELETE FROM trading_journal WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS tj_count = ROW_COUNT;

    -- Step 6: portfolio metadata. starting_capital + reset_date drive
    -- the Settings display and Risk Manager's drawdown-from-peak baseline.
    UPDATE portfolios
       SET starting_capital = new_capital,
           reset_date       = cutoff
     WHERE id = ltg_id;

    -- Step 7: re-seed the canonical 'Initial capital' deposit. Matches
    -- db_layer._sync_initial_deposit invariant (one row per portfolio
    -- where source='deposit' AND note='Initial capital').
    IF EXISTS (
        SELECT 1 FROM cash_transactions
         WHERE portfolio_id = ltg_id
           AND source = 'deposit'
           AND note   = 'Initial capital'
    ) THEN
        UPDATE cash_transactions
           SET amount = new_capital,
               date   = cutoff
         WHERE portfolio_id = ltg_id
           AND source = 'deposit'
           AND note   = 'Initial capital';
        reseed_action := 'updated';
    ELSE
        INSERT INTO cash_transactions
              (portfolio_id, date, amount, source, note)
        VALUES
              (ltg_id, cutoff, new_capital, 'deposit', 'Initial capital');
        reseed_action := 'inserted';
    END IF;

    RAISE NOTICE 'Migration 054 complete';
    RAISE NOTICE '  Dead trades (CLOSED) identified: %', dead_count;
    RAISE NOTICE '  FK-less child rows deleted: trade_images=%, trade_fundamentals=%, trade_lessons=%, lot_closures=%, audit_trail=%',
                 ti_count, tf_count, tl_count, lc_count, at_count;
    RAISE NOTICE '  cash_transactions wiped: %', cash_delete_count;
    RAISE NOTICE '  trades_summary CLOSED deleted (cascading to trades_details): %', trade_delete_count;
    RAISE NOTICE '  trading_journal wiped: %', tj_count;
    RAISE NOTICE '  Portfolio updated: starting_capital=%, reset_date=%', new_capital, cutoff;
    RAISE NOTICE '  Initial deposit row %', reseed_action;
END $$;

-- ============================================================================
-- Verification (manual, after COMMIT)
-- ============================================================================
--   -- expect: only OPEN campaigns remain
--   SELECT status, COUNT(*) FROM trades_summary
--    WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = 'Long-Term Growth')
--      AND deleted_at IS NULL
--    GROUP BY status;
--
--   -- expect: single row, $52,450 initial deposit on 2026-07-27
--   SELECT date, amount, source, note FROM cash_transactions
--    WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = 'Long-Term Growth');
--
--   -- expect: 0 rows (fresh baseline)
--   SELECT COUNT(*) FROM trading_journal
--    WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = 'Long-Term Growth');
--
--   -- expect: starting_capital = 52450, reset_date = 2026-07-27
--   SELECT starting_capital, reset_date FROM portfolios
--    WHERE name = 'Long-Term Growth';
-- ============================================================================
