-- ============================================================================
-- Migration 005: Flip user_id to NOT NULL on every tenant table
-- ============================================================================
-- Belt-and-suspenders cleanup after Steps 1-3. Today the RLS policy's
-- WITH CHECK clause already rejects any INSERT where user_id is NULL (the
-- comparison user_id = <session uuid> is false when user_id is null). This
-- migration moves that guarantee one level deeper — Postgres itself refuses
-- NULL at the column level, so even an operator running raw SQL outside the
-- app-runtime role can't insert an orphan row.
--
-- Sanity precondition: migration 001's backfill filled user_id on every
-- existing row, and every INSERT since then has gone through get_db_connection
-- which SETs app.user_id (defaulted on the column). So by the time this runs,
-- no NULLs should exist and SET NOT NULL is a clean flip.
--
-- Locks: each ALTER ... SET NOT NULL briefly takes ACCESS EXCLUSIVE on the
-- table, but on tables of this size (worst case audit_trail ~3700 rows) it's
-- sub-second — safe to run against live prod.
-- ============================================================================

DO $$
DECLARE
    t TEXT;
    tenant_tables TEXT[] := ARRAY[
        'portfolios', 'trades_summary', 'trades_details', 'trading_journal',
        'audit_trail', 'trade_images', 'trade_fundamentals', 'drawdown_notes',
        'trade_lessons', 'rule_notes', 'app_config', 'dashboard_events'
    ];
BEGIN
    FOREACH t IN ARRAY tenant_tables LOOP
        EXECUTE format('ALTER TABLE %I ALTER COLUMN user_id SET NOT NULL', t);
    END LOOP;
END $$;


-- ============================================================================
-- VERIFICATION (after COMMIT)
-- ============================================================================
--   SELECT table_name, column_name, is_nullable
--   FROM information_schema.columns
--   WHERE column_name = 'user_id'
--     AND table_schema = 'public'
--     AND table_name IN ('portfolios','trades_summary','trades_details','trading_journal',
--                        'audit_trail','trade_images','trade_fundamentals','drawdown_notes',
--                        'trade_lessons','rule_notes','app_config','dashboard_events')
--   ORDER BY table_name;
-- Expected: every row has is_nullable = 'NO'.
