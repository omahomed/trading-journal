-- migrations/040_reset_long_term_growth_baseline.sql
--
-- Reset Long-Term Growth portfolio to a fresh 2026-03-31 baseline.
--
-- Context: the portfolio was repurposed from the old TQQQ Strategy
-- via migration 037 (rename + trade wipe), patched by migration 039
-- (orphan-table cleanup + $4,500 reseed). Subsequently the Robinhood
-- and Daily Journal importers backfilled 2026-01-02..2026-05-21 of
-- historical activity. The user has now decided the Q1 reconstruction
-- is too noisy (90 day-pair NLV continuity gaps in the manually-
-- reconstructed xlsx → TWR collapse to -52% LTD vs Modified Dietz
-- ~-5%) and wants to start the portfolio fresh on 2026-03-31 with
-- starting_capital = $14,681.62 (the closing NLV on that date per
-- the revised xlsx).
--
-- Scope: pre-3/31 trades, all pre-3/31 cash, full trading_journal.
-- Trades opened on or after 3/31 survive (their cash mirrors + child
-- rows are preserved).
--
-- Cascade map relied on:
--   - trades_summary → trades_details  ON DELETE CASCADE (schema.sql:147)
--   - cash_transactions.trade_detail_id → trades_details
--     ON DELETE SET NULL (migrations/009:43) — post-cutoff mirrors
--     of pre-cutoff trades are nulled, not deleted. Step 5 sweeps
--     them via the `trade_detail_id IS NULL AND source IN
--     ('buy','sell')` filter.
--   - FK-less tables (trade_images, trade_fundamentals, trade_lessons,
--     lot_closures, audit_trail) carry trade_id as a soft reference;
--     deleting trades_summary does NOT cascade. Step 2 wipes them
--     explicitly BEFORE the trades_summary delete so dead_trades is
--     still populated.
--   - trading_journal → daily_journal_captures CASCADE (migration 031).
--   - weekly_retros → weekly_retro_snapshots CASCADE (migration 028).
--
-- Idempotency: re-run finds dead_trades empty, every DELETE returns
-- 0, the IF EXISTS branch UPDATEs the existing initial-deposit row
-- (no-op when amount+date already match), and the orphan-sweep
-- sanity check still prints 0. Safe to apply twice.
--
-- Founder UUID + app.user_id GUC: migrations/run.py SET LOCALs both
-- per migration (defense in depth), so the cash_transactions DEFAULT
-- column for user_id resolves correctly without needing an explicit
-- value in the INSERT here.

DO $$
DECLARE
    ltg_id              INTEGER;
    cutoff              DATE     := '2026-03-31';
    new_capital         NUMERIC  := 14681.62;
    dead_count          INTEGER;
    ti_count            INTEGER;
    tf_count            INTEGER;
    tl_count            INTEGER;
    lc_count            INTEGER;
    at_count            INTEGER;
    cash_date_count     INTEGER;
    trade_delete_count  INTEGER;
    orphan_count        INTEGER;
    tj_count            INTEGER;
    wr_count            INTEGER;
    orphan_remaining    INTEGER;
    reseed_action       TEXT;
BEGIN
    SELECT id INTO ltg_id
      FROM portfolios
     WHERE name = 'Long-Term Growth';

    IF ltg_id IS NULL THEN
        RAISE NOTICE 'No Long-Term Growth portfolio found — migration 040 is a no-op';
        RETURN;
    END IF;

    -- ----------------------------------------------------------------
    -- Step 1: snapshot the dead trade IDs.
    -- Materialized as a temp table because the same set is consumed
    -- by every FK-less sweep below; after step 4 the trade_id strings
    -- would still exist in the child tables, but the trades_summary
    -- rows would be gone, so re-deriving the set would be empty.
    -- ON COMMIT DROP keeps the temp table scoped to this migration.
    -- ----------------------------------------------------------------
    CREATE TEMP TABLE dead_trades ON COMMIT DROP AS
      SELECT trade_id
        FROM trades_summary
       WHERE portfolio_id = ltg_id
         AND deleted_at IS NULL
         AND open_date < cutoff;
    SELECT COUNT(*) INTO dead_count FROM dead_trades;

    -- ----------------------------------------------------------------
    -- Step 2: wipe FK-less child tables BEFORE the trades_summary
    -- cascade. trade_images, trade_fundamentals, trade_lessons,
    -- lot_closures, audit_trail all reference trade_id as a string
    -- without a FK, so they survive cascade and need explicit
    -- DELETEs. Order within step 2 doesn't matter (no inter-table FKs).
    -- ----------------------------------------------------------------
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

    DELETE FROM audit_trail
     WHERE portfolio_id = ltg_id
       AND trade_id IN (SELECT trade_id FROM dead_trades);
    GET DIAGNOSTICS at_count = ROW_COUNT;

    -- ----------------------------------------------------------------
    -- Step 3: pre-cutoff cash. Bulk delete by date filter catches all
    -- old deposits + buy/sell mirrors whose date is < cutoff. Post-
    -- cutoff mirrors of pre-cutoff trades survive this step (their
    -- date is post-cutoff); they're handled in step 5.
    -- ----------------------------------------------------------------
    DELETE FROM cash_transactions
     WHERE portfolio_id = ltg_id
       AND date < cutoff;
    GET DIAGNOSTICS cash_date_count = ROW_COUNT;

    -- ----------------------------------------------------------------
    -- Step 4: trade delete. Cascades to trades_details via the
    -- (portfolio_id, trade_id) FK chain. Any surviving cash_transactions
    -- mirrors of these deleted detail rows have their trade_detail_id
    -- SET NULLed (the FK's ON DELETE behavior) — those orphans are
    -- swept in step 5.
    -- ----------------------------------------------------------------
    DELETE FROM trades_summary
     WHERE portfolio_id = ltg_id
       AND open_date < cutoff;
    GET DIAGNOSTICS trade_delete_count = ROW_COUNT;

    -- ----------------------------------------------------------------
    -- Step 5: orphan cash sweep. trade_detail_id IS NULL combined with
    -- source IN ('buy','sell') uniquely identifies mirror rows whose
    -- backing detail was deleted by step 4. (Real deposit/withdraw/
    -- reconcile rows always have trade_detail_id NULL but never have
    -- source 'buy' or 'sell' — the CHECK in migrations/009:46-48
    -- guarantees the source vocab partition.)
    -- ----------------------------------------------------------------
    DELETE FROM cash_transactions
     WHERE portfolio_id = ltg_id
       AND source IN ('buy', 'sell')
       AND trade_detail_id IS NULL;
    GET DIAGNOSTICS orphan_count = ROW_COUNT;

    -- ----------------------------------------------------------------
    -- Step 6: full trading_journal wipe. User is re-importing from a
    -- revised xlsx that starts 3/31; partial-wipe would leave the
    -- imported rows to clash with prior post-3/31 entries (the
    -- UNIQUE(portfolio_id, day) constraint + the chained beg_nlv
    -- semantics make full-wipe + re-import cleaner). Cascades to
    -- daily_journal_captures via migration 031's FK.
    -- ----------------------------------------------------------------
    DELETE FROM trading_journal WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS tj_count = ROW_COUNT;

    -- ----------------------------------------------------------------
    -- Step 6b: weekly_retros defensive cleanup. Currently 0 rows
    -- (migration 039 wiped them), but a partial-cutoff filter
    -- future-proofs the migration if retros land before next deploy.
    -- Cascades to weekly_retro_snapshots via migration 028's FK.
    -- ----------------------------------------------------------------
    DELETE FROM weekly_retros
     WHERE portfolio_id = ltg_id
       AND week_start < cutoff;
    GET DIAGNOSTICS wr_count = ROW_COUNT;

    -- ----------------------------------------------------------------
    -- Step 7: portfolio metadata. starting_capital + reset_date drive
    -- Settings UI display and the Risk Manager's drawdown-from-peak
    -- baseline.
    -- ----------------------------------------------------------------
    UPDATE portfolios
       SET starting_capital = new_capital,
           reset_date       = cutoff
     WHERE id = ltg_id;

    -- ----------------------------------------------------------------
    -- Step 8: re-seed the canonical 'Initial capital' deposit. Matches
    -- db_layer._sync_initial_deposit's invariant (one row per portfolio
    -- where source='deposit' AND note='Initial capital'). On first run
    -- this UPDATEs migration 039's $4,500 / 2026-01-01 reseed to
    -- $14,681.62 / 2026-03-31. On re-run it's a no-op (values match).
    -- ----------------------------------------------------------------
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

    -- ----------------------------------------------------------------
    -- Sanity check: after step 5 the orphan-mirror count must be 0.
    -- A non-zero count would indicate either a step-ordering bug or a
    -- pre-existing orphan that should be investigated separately.
    -- ----------------------------------------------------------------
    SELECT COUNT(*) INTO orphan_remaining
      FROM cash_transactions
     WHERE portfolio_id = ltg_id
       AND source IN ('buy', 'sell')
       AND trade_detail_id IS NULL;

    RAISE NOTICE 'Migration 040 complete';
    RAISE NOTICE '  Dead trades identified: %', dead_count;
    RAISE NOTICE '  FK-less child rows deleted: trade_images=%, trade_fundamentals=%, trade_lessons=%, lot_closures=%, audit_trail=%',
                 ti_count, tf_count, tl_count, lc_count, at_count;
    RAISE NOTICE '  Pre-cutoff cash deleted: %', cash_date_count;
    RAISE NOTICE '  Trades deleted (cascading to trades_details): %', trade_delete_count;
    RAISE NOTICE '  Orphan post-cutoff cash mirrors deleted: %', orphan_count;
    RAISE NOTICE '  trading_journal rows wiped: %', tj_count;
    RAISE NOTICE '  weekly_retros wiped: %', wr_count;
    RAISE NOTICE '  Portfolio updated: starting_capital=%, reset_date=%', new_capital, cutoff;
    RAISE NOTICE '  Initial deposit row %', reseed_action;
    RAISE NOTICE '  Sanity check: remaining orphan buy/sell cash rows = % (expected 0)', orphan_remaining;

    IF orphan_remaining > 0 THEN
        RAISE WARNING 'Migration 040: % orphan cash rows remain after sweep — investigate', orphan_remaining;
    END IF;
END $$;
