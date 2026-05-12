-- ============================================================================
-- Migration 024: trigger_audit_trades_summary — migration-safe rewrite
-- ============================================================================
-- Purpose:
--   Make audit_trades_summary_changes() safe to fire from inside a migration
--   session. The previous version (migration 022) INSERTed into audit_trail
--   without listing user_id, relying on the column DEFAULT
--       NULLIF(current_setting('app.user_id', true), '')::uuid
--   which evaluates to NULL when the session variable isn't set. Combined
--   with audit_trail.user_id being NOT NULL (migration 005), any system-level
--   UPDATE on trades_summary fired the trigger, the trigger tried to INSERT a
--   NULL user_id, and the migration aborted.
--
--   Group 7-3's migration 023 was the first to hit this in production: it
--   issues three UPDATE statements on trades_summary that toggled watched
--   fields, the trigger fired, the NOT NULL fired, the runner rolled back.
--
-- Convention (apply to all future audit-style triggers):
--   Any AFTER UPDATE / INSERT / DELETE trigger that INSERTs into a tenant-
--   scoped table (one with NOT NULL user_id + the migration-003 DEFAULT)
--   MUST source user_id explicitly via:
--       COALESCE(
--           NULLIF(current_setting('app.user_id', true), '')::uuid,
--           'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid  -- founder fallback
--       )
--   Migration sessions don't have app.user_id set; without the fallback the
--   trigger aborts the migration that fired it.
--
-- Defense in depth: migrations/run.py also does `SET LOCAL app.user_id` to
-- the founder UUID per migration, so the column DEFAULT also resolves
-- correctly. The COALESCE here protects the trigger against any caller that
-- forgets to set the session variable (cron jobs, ad-hoc psql sessions,
-- replication tooling, future tools).
--
-- TODO(multi-tenant): when Tier 2 multi-tenancy lands, replace the founder
-- fallback with a dedicated 'system' user row + an RLS policy carve-out
-- (USING / WITH CHECK exception for user_id = '<system-uuid>') so migration-
-- attributed audit rows are visible to every tenant's admin viewer. Today
-- the founder UUID works because the app is single-tenant and the founder
-- is the only admin reader.
--
-- Provenance distinguisher (audit viewer):
--   username = 'pg_trigger'            when app.user_id IS set (app context)
--   username = 'pg_trigger:migration'  when app.user_id is unset (system context)
-- /api/audit and the frontend don't filter on username, so existing readers
-- are unaffected.
--
-- Bootstrap: CREATE OR REPLACE FUNCTION is DDL and does not fire the trigger
-- on itself. No recursion risk.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT
-- statements here.
-- ============================================================================

CREATE OR REPLACE FUNCTION audit_trades_summary_changes()
RETURNS TRIGGER AS $$
DECLARE
    diffs TEXT := '';
    uid   UUID;
    sess  TEXT;
BEGIN
    IF NEW.rule IS DISTINCT FROM OLD.rule THEN
        diffs := diffs || format('rule: %L → %L; ', OLD.rule, NEW.rule);
    END IF;
    IF NEW.buy_notes IS DISTINCT FROM OLD.buy_notes THEN
        diffs := diffs || format('buy_notes: %L → %L; ', OLD.buy_notes, NEW.buy_notes);
    END IF;
    IF NEW.sell_rule IS DISTINCT FROM OLD.sell_rule THEN
        diffs := diffs || format('sell_rule: %L → %L; ', OLD.sell_rule, NEW.sell_rule);
    END IF;
    IF NEW.sell_notes IS DISTINCT FROM OLD.sell_notes THEN
        diffs := diffs || format('sell_notes: %L → %L; ', OLD.sell_notes, NEW.sell_notes);
    END IF;
    IF NEW.notes IS DISTINCT FROM OLD.notes THEN
        diffs := diffs || format('notes: %L → %L; ', OLD.notes, NEW.notes);
    END IF;
    IF NEW.stop_loss IS DISTINCT FROM OLD.stop_loss THEN
        diffs := diffs || format('stop_loss: %L → %L; ', OLD.stop_loss, NEW.stop_loss);
    END IF;
    IF NEW.risk_budget IS DISTINCT FROM OLD.risk_budget THEN
        diffs := diffs || format('risk_budget: %L → %L; ', OLD.risk_budget, NEW.risk_budget);
    END IF;
    IF NEW.status IS DISTINCT FROM OLD.status THEN
        diffs := diffs || format('status: %L → %L; ', OLD.status, NEW.status);
    END IF;

    IF diffs <> '' THEN
        sess := current_setting('app.user_id', true);
        uid  := COALESCE(
                    NULLIF(sess, '')::uuid,
                    'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
                );

        INSERT INTO audit_trail (portfolio_id, user_id, username, action, trade_id, ticker, details)
        VALUES (
            NEW.portfolio_id,
            uid,
            CASE WHEN sess IS NULL OR sess = ''
                 THEN 'pg_trigger:migration'
                 ELSE 'pg_trigger' END,
            'SUMMARY_UPDATE',
            NEW.trade_id,
            NEW.ticker,
            LEFT(diffs, LENGTH(diffs) - 2)  -- strip trailing '; '
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- Verification
-- ============================================================================
-- Smoke test: the updated function body must contain the COALESCE fallback.
-- We grep pg_proc.prosrc rather than re-installing — a re-install would mask
-- whether THIS migration's CREATE OR REPLACE actually took effect.
-- ============================================================================
DO $$
DECLARE
    body TEXT;
BEGIN
    SELECT prosrc INTO body
      FROM pg_proc
     WHERE proname = 'audit_trades_summary_changes'
     LIMIT 1;

    IF body IS NULL THEN
        RAISE EXCEPTION 'Migration 024: audit_trades_summary_changes function not found after CREATE OR REPLACE';
    END IF;

    IF body NOT LIKE '%COALESCE%' THEN
        RAISE EXCEPTION 'Migration 024: function body missing COALESCE fallback — update did not take effect';
    END IF;

    IF body NOT LIKE '%pg_trigger:migration%' THEN
        RAISE EXCEPTION 'Migration 024: function body missing migration-context username distinguisher';
    END IF;

    RAISE NOTICE 'Migration 024 complete: audit trigger now migration-safe (COALESCE fallback + provenance distinguisher).';
END $$;


-- ============================================================================
-- ROLLBACK (if needed — copy into a new migration or run in psql)
-- ============================================================================
-- Re-install the migration-022 version of the function (omit user_id from the
-- INSERT, rely on column DEFAULT). Trigger definition itself is unchanged so
-- no DROP TRIGGER / CREATE TRIGGER needed.
-- ============================================================================
