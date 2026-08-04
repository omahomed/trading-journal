-- ============================================================================
-- Migration 058: Reset 457B Plan trade log to a fresh 2026-08-04 baseline.
-- ============================================================================
-- User has been trading in the 457B Plan account (employer-sponsored) but
-- hasn't kept the app's trade log in sync — the OPEN campaigns and CLOSED
-- history no longer reflect reality. Fresh start for the trade log only;
-- journal_log (NLV entries) and cash_transactions stay untouched.
--
-- Divergence from migration 054 (Long-Term Growth reset):
--   * Wipes BOTH open and closed campaigns (user chose "include 8 closed as
--     well"). Migration 054's OPEN-preserved model doesn't fit here — the
--     entire trade history is stale, not just the closed subset.
--   * PRESERVES trading_journal (300 rows). User: "the journal log and NLV
--     entries are all good." The Journal Log page + Risk Manager keep
--     working against the same NLV series they had yesterday.
--   * PRESERVES cash_transactions (32 rows). Position Sizer / cash-balance
--     reads keep working. FK cash_transactions.trade_detail_id ON DELETE
--     SET NULL means cash rows tied to deleted details silently become
--     orphans (their `trade_detail_id` = NULL) rather than blocking the
--     cascade — the math still adds up, the "linked to" reference just
--     disappears from the audit view.
--   * DOES NOT touch portfolios.starting_capital or portfolios.reset_date.
--     No NLV baseline change; drawdown-from-peak in Risk Manager keeps its
--     existing anchor.
--
-- Cascade map:
--   - trades_summary → trades_details       ON DELETE CASCADE
--   - cash_transactions.trade_detail_id     ON DELETE SET NULL (implicit)
--   - trade_images / trade_fundamentals / lot_closures / audit_trail
--     carry trade_id as string with no FK; sweep explicitly.
--
-- Idempotency: re-run finds trades_summary set empty for 457B, all sweeps
-- 0-row, RAISE NOTICE reports 0s. Safe to apply twice.
--
-- Downstream-consumer audit (per CLAUDE.md's standing rule):
--   * Portfolio Heat / Log Buy scale-in picker / Position Sizer /
--     Trade Journal: all four call api.tradesOpen(portfolio) with a
--     .catch(() => []) fallback → empty-state renders cleanly, no fake
--     $100k basis. journalLatest returns the actual NLV (which stays
--     intact via the preserved trading_journal), so pos-size %'s stay
--     honest even against zero positions.
--   * ACS: shows empty "Positions" table when tradesOpen is empty. No
--     regression.
-- ============================================================================

DO $$
DECLARE
    p457_id             INTEGER;
    ti_count            INTEGER;
    tf_count            INTEGER;
    tl_count            INTEGER;
    lc_count            INTEGER;
    at_count            INTEGER;
    trade_delete_count  INTEGER;
BEGIN
    SELECT id INTO p457_id
      FROM portfolios
     WHERE name = '457B Plan';

    IF p457_id IS NULL THEN
        RAISE NOTICE 'No 457B Plan portfolio found — migration 058 is a no-op';
        RETURN;
    END IF;

    -- Step 1: snapshot ALL trade_ids for this portfolio (open AND closed).
    -- Divergence from migration 054's status='CLOSED' filter — user opted
    -- to include closed history in this reset.
    CREATE TEMP TABLE dead_trades ON COMMIT DROP AS
      SELECT trade_id
        FROM trades_summary
       WHERE portfolio_id = p457_id
         AND deleted_at IS NULL;

    -- Step 2: FK-less child tables scoped to the dead trade_ids.
    DELETE FROM trade_images
     WHERE portfolio_id = p457_id
       AND trade_id IN (SELECT trade_id FROM dead_trades);
    GET DIAGNOSTICS ti_count = ROW_COUNT;

    DELETE FROM trade_fundamentals
     WHERE portfolio_id = p457_id
       AND trade_id IN (SELECT trade_id FROM dead_trades);
    GET DIAGNOSTICS tf_count = ROW_COUNT;

    DELETE FROM trade_lessons
     WHERE portfolio_id = p457_id
       AND trade_id IN (SELECT trade_id FROM dead_trades);
    GET DIAGNOSTICS tl_count = ROW_COUNT;

    DELETE FROM lot_closures
     WHERE portfolio_id = p457_id
       AND trade_id IN (SELECT trade_id FROM dead_trades);
    GET DIAGNOSTICS lc_count = ROW_COUNT;

    DELETE FROM audit_trail
     WHERE portfolio_id = p457_id
       AND trade_id IN (SELECT trade_id FROM dead_trades);
    GET DIAGNOSTICS at_count = ROW_COUNT;

    -- Step 3: delete every trade. Cascades to trades_details via the
    -- (portfolio_id, trade_id) FK. cash_transactions.trade_detail_id
    -- SET NULL cascades silently — cash rows survive.
    DELETE FROM trades_summary
     WHERE portfolio_id = p457_id
       AND deleted_at IS NULL;
    GET DIAGNOSTICS trade_delete_count = ROW_COUNT;

    -- NO wipe of trading_journal (user: "journal log and NLV entries are
    -- all good"). NO wipe of cash_transactions (Position Sizer / balance
    -- readouts depend on it). NO update to portfolios (no NLV baseline
    -- reset — the user is clearing the trade log, not restarting capital).

    RAISE NOTICE 'Migration 058 complete for 457B Plan';
    RAISE NOTICE '  FK-less child rows deleted: trade_images=%, trade_fundamentals=%, trade_lessons=%, lot_closures=%, audit_trail=%',
                 ti_count, tf_count, tl_count, lc_count, at_count;
    RAISE NOTICE '  trades_summary deleted (cascading to trades_details): %', trade_delete_count;
    RAISE NOTICE '  PRESERVED: trading_journal, cash_transactions, portfolios.starting_capital';
END $$;

-- ============================================================================
-- Verification (manual, after COMMIT)
-- ============================================================================
--   -- expect: 0 rows (fresh trade log)
--   SELECT status, COUNT(*) FROM trades_summary
--    WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = '457B Plan')
--      AND deleted_at IS NULL
--    GROUP BY status;
--
--   -- expect: 0 rows
--   SELECT COUNT(*) FROM trades_details
--    WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = '457B Plan');
--
--   -- expect: 300 rows (unchanged — journal preserved)
--   SELECT COUNT(*) FROM trading_journal
--    WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = '457B Plan');
--
--   -- expect: 32 rows (unchanged — cash chain preserved). Some may now
--   -- carry trade_detail_id = NULL (auto-cleared by ON DELETE SET NULL).
--   SELECT COUNT(*) FROM cash_transactions
--    WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = '457B Plan');
-- ============================================================================
