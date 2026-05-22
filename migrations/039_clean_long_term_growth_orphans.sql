-- migrations/039_clean_long_term_growth_orphans.sql
--
-- Cleanup of portfolio-scoped data that survived migration 037.
--
-- Migration 037 renamed TQQQ Strategy → Long-Term Growth and ran
-- DELETE FROM trades_summary, which cascaded to trades_details via
-- the (portfolio_id, trade_id) FK chain. Everything else with a
-- portfolio_id FK to portfolios(id) — trading_journal, audit_trail,
-- trade_images, etc. — survived the rename because their FK points
-- to the portfolio (whose id was preserved) rather than to
-- trades_summary. Result: 48 historical journal entries + 18 cash
-- transactions + 3 audit-trail rows + assorted empties remained
-- attached to the renamed portfolio.
--
-- This migration finishes the job: hard-deletes every portfolio-
-- scoped row, then re-seeds the canonical $4,500 / 2026-01-01
-- initial deposit so the cash ledger isn't empty. Audit (2026-05-21)
-- confirmed zero post-rename rows on any table, so the wipe doesn't
-- destroy any legitimate user activity.
--
-- Cascade behavior the migration relies on:
--   - DELETE trading_journal cascades to daily_journal_captures
--     (migration 031 FK on daily_journal_id).
--   - DELETE weekly_retros cascades to weekly_retro_snapshots
--     (migration 028 FK on weekly_retro_id).
--   - trade_fundamentals.image_id uses ON DELETE SET NULL — those
--     rows survive trade_images deletion. Explicit DELETE required.
--
-- Idempotency:
--   - All DELETEs are no-ops on re-run (0 rows remain to match).
--   - Re-seed INSERT guarded by NOT EXISTS on the app's invariant:
--     one (portfolio_id, source='deposit', note='Initial capital')
--     row per portfolio (see db_layer._sync_initial_deposit, which
--     enforces the same shape when starting_capital is edited via
--     Settings). The guard matches by (portfolio, source, note) —
--     NOT by date or amount — so if the user later edits the
--     starting_capital via Settings, this migration's re-run still
--     skips the insert and doesn't create a duplicate "Initial
--     capital" row.
--
-- Scope: only the 'Long-Term Growth' portfolio. All WHERE clauses
-- bind to the id resolved at the top; no other portfolio (CanSlim,
-- 457B Plan) can be affected. If the portfolio is somehow missing
-- (manual cleanup, fresh deploy), the migration RAISE NOTICEs and
-- exits cleanly.

DO $$
DECLARE
    ltg_id        INTEGER;
    tj_count      INTEGER;
    wr_count      INTEGER;
    at_count      INTEGER;
    ti_count      INTEGER;
    tf_count      INTEGER;
    dn_count      INTEGER;
    tl_count      INTEGER;
    rn_count      INTEGER;
    ct_count      INTEGER;
    lc_count      INTEGER;
    ta_count      INTEGER;
    tag_count     INTEGER;
BEGIN
    SELECT id INTO ltg_id
      FROM portfolios
     WHERE name = 'Long-Term Growth';

    IF ltg_id IS NULL THEN
        RAISE NOTICE 'No Long-Term Growth portfolio found — migration 039 is a no-op';
        RETURN;
    END IF;

    -- Parent tables (cascades handle dependent child tables).
    DELETE FROM trading_journal WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS tj_count = ROW_COUNT;

    DELETE FROM weekly_retros WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS wr_count = ROW_COUNT;

    -- Flat portfolio-scoped tables.
    DELETE FROM audit_trail WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS at_count = ROW_COUNT;

    DELETE FROM trade_images WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS ti_count = ROW_COUNT;

    -- trade_fundamentals.image_id uses ON DELETE SET NULL — these
    -- rows do NOT cascade from trade_images. Explicit DELETE needed.
    DELETE FROM trade_fundamentals WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS tf_count = ROW_COUNT;

    DELETE FROM drawdown_notes WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS dn_count = ROW_COUNT;

    DELETE FROM trade_lessons WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS tl_count = ROW_COUNT;

    DELETE FROM rule_notes WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS rn_count = ROW_COUNT;

    DELETE FROM cash_transactions WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS ct_count = ROW_COUNT;

    DELETE FROM lot_closures WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS lc_count = ROW_COUNT;

    -- tag_assignments deleted before tags so the FK from
    -- tag_assignments → tags doesn't block. (Both are portfolio-
    -- scoped so the cascade from portfolios would have handled it
    -- in any order, but explicit-ordered is clearer.)
    DELETE FROM tag_assignments WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS ta_count = ROW_COUNT;

    DELETE FROM tags WHERE portfolio_id = ltg_id;
    GET DIAGNOSTICS tag_count = ROW_COUNT;

    -- Re-seed the canonical $4,500 / 2026-01-01 deposit. Shape mirrors
    -- db_layer._sync_initial_deposit's INSERT exactly (source='deposit',
    -- note='Initial capital') so any subsequent Settings edit by the
    -- user flows through the same row.
    IF NOT EXISTS (
        SELECT 1 FROM cash_transactions
         WHERE portfolio_id = ltg_id
           AND source = 'deposit'
           AND note = 'Initial capital'
    ) THEN
        INSERT INTO cash_transactions (portfolio_id, date, amount, source, note)
        VALUES (ltg_id, '2026-01-01'::timestamp, 4500.00, 'deposit', 'Initial capital');
        RAISE NOTICE 'Re-seeded initial $4,500 deposit at 2026-01-01';
    ELSE
        RAISE NOTICE 'Initial capital deposit already exists; skipped re-seed';
    END IF;

    RAISE NOTICE 'Migration 039 cleanup counts: trading_journal=%, weekly_retros=%, audit_trail=%, trade_images=%, trade_fundamentals=%, drawdown_notes=%, trade_lessons=%, rule_notes=%, cash_transactions=%, lot_closures=%, tag_assignments=%, tags=%',
                 tj_count, wr_count, at_count, ti_count, tf_count,
                 dn_count, tl_count, rn_count, ct_count, lc_count,
                 ta_count, tag_count;
END $$;
