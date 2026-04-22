-- ============================================================================
-- Migration 006: Soft-delete columns on customer-data tables (Tier 3, step 1)
-- ============================================================================
-- Adds `deleted_at TIMESTAMPTZ` to the three tables whose data has real
-- recovery value if accidentally deleted:
--   - trades_summary   (campaign-level records)
--   - trades_details   (individual buy/sell transactions)
--   - trading_journal  (daily journal entries)
--
-- After db_layer is updated (separate commit with this migration):
--   - DELETE endpoints UPDATE deleted_at = now() instead of running DELETE
--   - SELECT queries append `AND deleted_at IS NULL` so soft-deleted rows
--     are invisible to normal reads
--   - A future admin "restore" flow can set deleted_at = NULL to undelete
--
-- Why only these three: the other tenant tables either
--   (a) record history and shouldn't be deleted at all (audit_trail), or
--   (b) are settings that have no recovery value (rule_notes, app_config), or
--   (c) already cascade from the above (trade_images via FK).
--
-- Does not interact with RLS: the tenant_isolation policy still filters by
-- user_id; application code layers the deleted_at IS NULL filter on top.
-- ============================================================================

DO $$
DECLARE
    t TEXT;
    tables TEXT[] := ARRAY['trades_summary', 'trades_details', 'trading_journal'];
BEGIN
    FOREACH t IN ARRAY tables LOOP
        EXECUTE format(
            'ALTER TABLE %I ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ',
            t
        );
        -- Partial index — only indexes live rows, since that's what reads
        -- filter on. Keeps the index small even as soft-deleted rows grow.
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %I (user_id) WHERE deleted_at IS NULL',
            'idx_' || t || '_live', t
        );
    END LOOP;
END $$;


-- ============================================================================
-- VERIFICATION (after COMMIT)
-- ============================================================================
--   SELECT table_name, column_name, is_nullable
--   FROM information_schema.columns
--   WHERE column_name = 'deleted_at' AND table_schema = 'public'
--   ORDER BY table_name;
-- Expected: three rows, all is_nullable = YES (deleted_at is null = live row).
