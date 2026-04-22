-- ============================================================================
-- Migration 003: Postgres Row-Level Security for tenant isolation (Tier 1, step 3)
-- ============================================================================
-- Flips the data layer from "hope every query filters by user_id" to "Postgres
-- physically refuses to return rows that don't match the session's user_id".
-- Defense in depth — a missed WHERE clause in application code is no longer a
-- data leak.
--
-- For each of the 12 tenant-scoped tables:
--   1. ALTER COLUMN user_id DEFAULT = the session variable `app.user_id`.
--      Makes INSERTs auto-tag the new row with the current user, so we don't
--      have to thread user_id through every INSERT statement in Python.
--   2. ENABLE + FORCE ROW LEVEL SECURITY. ENABLE turns it on; FORCE makes it
--      apply even to the table owner (neondb_owner on Neon) — without FORCE,
--      owners bypass RLS silently and the policy is ineffective.
--   3. A single `tenant_isolation` policy that both filters reads (USING) and
--      guards writes (WITH CHECK). A row is visible and writable iff
--      row.user_id = current_setting('app.user_id').
--
-- NULLIF + `true` on current_setting makes the whole chain safe:
--   - If app.user_id is not set: NULLIF returns NULL → cast to UUID stays NULL
--     → policy compares user_id = NULL → always false → row is invisible and
--     unwritable. Safe fail.
--   - If app.user_id is set to the owner UUID: policy matches normally.
--
-- market_signals is excluded — shared global market data (IBD signals on
-- SPY/IXIC). Every tenant reads the same rows.
--
-- How Python sets the session var (see db_layer.py):
--   Each connection does `SET app.user_id = '<uuid>'` based on a contextvar
--   set by the FastAPI middleware after JWT verification.
--
-- Idempotent via DROP POLICY IF EXISTS + ENABLE/FORCE being no-ops when
-- already applied.
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
        -- 1. DEFAULT user_id from session var on INSERT.
        EXECUTE format(
            'ALTER TABLE %I ALTER COLUMN user_id '
            'SET DEFAULT NULLIF(current_setting(%L, true), '''')::uuid',
            t, 'app.user_id'
        );

        -- 2. Turn RLS on and force it for table owners too.
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);

        -- 3. Rewriteable policy (drop-and-recreate for idempotency).
        EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', t);
        EXECUTE format(
            'CREATE POLICY tenant_isolation ON %I FOR ALL '
            'USING      (user_id = NULLIF(current_setting(%L, true), '''')::uuid) '
            'WITH CHECK (user_id = NULLIF(current_setting(%L, true), '''')::uuid)',
            t, 'app.user_id', 'app.user_id'
        );
    END LOOP;
END $$;


-- ============================================================================
-- VERIFICATION (after COMMIT)
-- ============================================================================
-- Expected: 12 rows, all with relrowsecurity = true and relforcerowsecurity = true.
--   SELECT relname, relrowsecurity, relforcerowsecurity
--   FROM pg_class
--   WHERE relname IN ('portfolios','trades_summary','trades_details','trading_journal',
--                     'audit_trail','trade_images','trade_fundamentals','drawdown_notes',
--                     'trade_lessons','rule_notes','app_config','dashboard_events')
--   ORDER BY relname;
--
-- Functional test from psql (replace UUID with founder):
--   SET app.user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a';
--   SELECT count(*) FROM trades_details;  -- expected: 1309 (founder's data)
--   SET app.user_id = '00000000-0000-0000-0000-000000000000';
--   SELECT count(*) FROM trades_details;  -- expected: 0 (different user)
--   RESET app.user_id;
--   SELECT count(*) FROM trades_details;  -- expected: 0 (no user set)
