-- ============================================================================
-- Migration 022: trades_summary CHECK constraints + audit_trail trigger
-- ============================================================================
-- Adds schema-level enforcement of invariants the application now upholds
-- (Phase 2 Commits 1-6) plus an AFTER UPDATE trigger that records who-
-- changed-what-when on critical user-prose / risk fields.
--
-- Defense-in-depth layer: the application has been hardened, but the DB
-- now also rejects bad writes regardless of where they originate (DB
-- console, future code paths, future tools). The audit trigger catches
-- changes that bypass the application's log_audit calls.
--
-- Diagnostic queries from the Commit 7 investigation report (run before
-- applying) confirmed the corpus is clean for all 10 constraints. No
-- pre-cleanup needed.
--
-- Deferred to a follow-up: chk_summary_closed_shares_zero
--   (status='CLOSED' OR shares=0). 20+ pre-existing violations from
--   historical bugs; needs separate investigation/cleanup commit.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT
-- statements here.
-- ============================================================================

-- ============================================================================
-- 1. CHECK constraints (idempotent via DO $$ EXCEPTION wrapper)
-- ============================================================================
-- Postgres CHECK constraints don't support IF NOT EXISTS natively. The
-- duplicate_object exception arm makes each ALTER TABLE safe to re-run.

DO $$ BEGIN
    BEGIN
        ALTER TABLE trades_summary ADD CONSTRAINT chk_summary_status_enum
            CHECK (status IN ('OPEN', 'CLOSED'));
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END $$;

DO $$ BEGIN
    BEGIN
        ALTER TABLE trades_summary ADD CONSTRAINT chk_summary_shares_nonneg
            CHECK (shares >= 0);
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END $$;

DO $$ BEGIN
    BEGIN
        ALTER TABLE trades_summary ADD CONSTRAINT chk_summary_risk_budget_nonneg
            CHECK (risk_budget IS NULL OR risk_budget >= 0);
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END $$;

DO $$ BEGIN
    BEGIN
        ALTER TABLE trades_summary ADD CONSTRAINT chk_summary_multiplier_positive
            CHECK (multiplier > 0);
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END $$;

DO $$ BEGIN
    BEGIN
        ALTER TABLE trades_summary ADD CONSTRAINT chk_summary_closed_date_consistency
            CHECK (status = 'CLOSED' OR closed_date IS NULL);
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END $$;

-- 5 separate text-sentinel CHECKs — one per column. Cleaner error messages
-- (the violating column shows up in the constraint name) and easier
-- independent rollback than a single bundled CHECK with OR clauses.

DO $$ BEGIN
    BEGIN
        ALTER TABLE trades_summary ADD CONSTRAINT chk_summary_rule_no_sentinel
            CHECK (rule IS NULL OR LOWER(TRIM(rule)) NOT IN ('nan', 'none', 'null'));
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END $$;

DO $$ BEGIN
    BEGIN
        ALTER TABLE trades_summary ADD CONSTRAINT chk_summary_buy_notes_no_sentinel
            CHECK (buy_notes IS NULL OR LOWER(TRIM(buy_notes)) NOT IN ('nan', 'none', 'null'));
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END $$;

DO $$ BEGIN
    BEGIN
        ALTER TABLE trades_summary ADD CONSTRAINT chk_summary_sell_rule_no_sentinel
            CHECK (sell_rule IS NULL OR LOWER(TRIM(sell_rule)) NOT IN ('nan', 'none', 'null'));
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END $$;

DO $$ BEGIN
    BEGIN
        ALTER TABLE trades_summary ADD CONSTRAINT chk_summary_sell_notes_no_sentinel
            CHECK (sell_notes IS NULL OR LOWER(TRIM(sell_notes)) NOT IN ('nan', 'none', 'null'));
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END $$;

DO $$ BEGIN
    BEGIN
        ALTER TABLE trades_summary ADD CONSTRAINT chk_summary_notes_no_sentinel
            CHECK (notes IS NULL OR LOWER(TRIM(notes)) NOT IN ('nan', 'none', 'null'));
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END $$;


-- ============================================================================
-- 2. Audit trigger
-- ============================================================================
-- Fires AFTER UPDATE on trades_summary. Watches 8 user-prose / risk fields
-- and writes a 'SUMMARY_UPDATE' row to audit_trail when any changed.
-- Skips numeric/LIFO-derived fields (total_cost, realized_pl, shares,
-- avg_entry, etc.) which change on every recompute and would flood the log.
-- Inserts nothing when no watched field changed (recompute-only updates).
--
-- Username = 'pg_trigger' literal so audit-trail viewers can distinguish
-- trigger-recorded changes from app-level log_audit() rows.

CREATE OR REPLACE FUNCTION audit_trades_summary_changes()
RETURNS TRIGGER AS $$
DECLARE
    diffs TEXT := '';
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
        INSERT INTO audit_trail (portfolio_id, username, action, trade_id, ticker, details)
        VALUES (
            NEW.portfolio_id,
            'pg_trigger',
            'SUMMARY_UPDATE',
            NEW.trade_id,
            NEW.ticker,
            LEFT(diffs, LENGTH(diffs) - 2)  -- strip trailing '; '
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_audit_trades_summary ON trades_summary;
CREATE TRIGGER trigger_audit_trades_summary
AFTER UPDATE ON trades_summary
FOR EACH ROW
EXECUTE FUNCTION audit_trades_summary_changes();


-- ============================================================================
-- 3. Verification block
-- ============================================================================
-- Counts the 10 CHECK constraints and confirms the trigger exists. RAISE
-- EXCEPTION on any miss so the migration runner aborts and the user
-- immediately knows something failed silently.

DO $$
DECLARE
    chk_count INTEGER;
    trig_exists BOOLEAN;
BEGIN
    SELECT COUNT(*) INTO chk_count
      FROM pg_constraint
     WHERE conrelid = 'trades_summary'::regclass
       AND contype = 'c'
       AND conname IN (
         'chk_summary_status_enum',
         'chk_summary_shares_nonneg',
         'chk_summary_risk_budget_nonneg',
         'chk_summary_multiplier_positive',
         'chk_summary_closed_date_consistency',
         'chk_summary_rule_no_sentinel',
         'chk_summary_buy_notes_no_sentinel',
         'chk_summary_sell_rule_no_sentinel',
         'chk_summary_sell_notes_no_sentinel',
         'chk_summary_notes_no_sentinel'
       );

    SELECT EXISTS (
        SELECT 1 FROM pg_trigger
         WHERE tgname = 'trigger_audit_trades_summary'
           AND tgrelid = 'trades_summary'::regclass
    ) INTO trig_exists;

    RAISE NOTICE 'Migration 022: % CHECK constraints in place; audit trigger %',
                 chk_count,
                 CASE WHEN trig_exists THEN 'installed' ELSE 'NOT installed' END;

    IF chk_count <> 10 THEN
        RAISE EXCEPTION 'Migration 022: expected 10 CHECK constraints, found %', chk_count;
    END IF;
    IF NOT trig_exists THEN
        RAISE EXCEPTION 'Migration 022: audit trigger not installed';
    END IF;
END $$;


-- ============================================================================
-- ROLLBACK (if needed — copy into a new migration or run in psql)
-- ============================================================================
-- DROP TRIGGER IF EXISTS trigger_audit_trades_summary ON trades_summary;
-- DROP FUNCTION IF EXISTS audit_trades_summary_changes();
--
-- ALTER TABLE trades_summary
--   DROP CONSTRAINT IF EXISTS chk_summary_status_enum,
--   DROP CONSTRAINT IF EXISTS chk_summary_shares_nonneg,
--   DROP CONSTRAINT IF EXISTS chk_summary_risk_budget_nonneg,
--   DROP CONSTRAINT IF EXISTS chk_summary_multiplier_positive,
--   DROP CONSTRAINT IF EXISTS chk_summary_closed_date_consistency,
--   DROP CONSTRAINT IF EXISTS chk_summary_rule_no_sentinel,
--   DROP CONSTRAINT IF EXISTS chk_summary_buy_notes_no_sentinel,
--   DROP CONSTRAINT IF EXISTS chk_summary_sell_rule_no_sentinel,
--   DROP CONSTRAINT IF EXISTS chk_summary_sell_notes_no_sentinel,
--   DROP CONSTRAINT IF EXISTS chk_summary_notes_no_sentinel;
-- ============================================================================
