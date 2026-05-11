-- ============================================================================
-- Migration 023: Backfill Trade Risk $ = premium (cost) for all option trades
-- ============================================================================
-- Group 7-3 policy correction. Migration 021 backfilled NULL/0 risk_budget
-- rows with `sizing_mode_pct × prior_day_end_nlv`, which is unrelated to
-- option max-loss math. For long options (calls + protective puts), max
-- loss = premium paid = qty × avg_entry × multiplier. The
-- trades_summary.total_cost column already holds exactly this value
-- (LIFO-current cost basis for OPEN trades, full BUY cost for CLOSED).
--
-- A second, related practice retired by Group 7-3: setting decorative
-- "50%-of-premium" stops on option BUY detail rows. Those stops produced
-- half-cost Trade Risk $ values under the pre-Group-7-3 formula and never
-- corresponded to executable stop logic. This migration zeroes them out
-- so the data layer reflects current policy.
--
-- Three steps in one transaction:
--   1. Backfill trades_summary.risk_budget = total_cost for all options.
--   2. Zero out decorative stop_loss values on option BUY detail rows.
--   3. Clear summary stop_loss on option summaries where set.
--
-- Idempotent: all three UPDATEs use `IS DISTINCT FROM` / `IS NOT NULL` /
-- `> 0` guards so a re-run is a no-op.
--
-- Provenance: the Migration 022 audit trigger on trades_summary captures
-- OLD/NEW risk_budget and stop_loss values to audit_trail for every row
-- changed by steps 1 and 3, so reviewers can trace exactly which values
-- this migration overwrote.
--
-- Expected affected (per pre-flight diagnostics on prod, 2026-05):
--   Step 1: 15 rows in CanSlim (6 OPEN + 9 CLOSED), 0 in other portfolios.
--   Step 2: 6 rows in CanSlim (the legacy 50% stops).
--   Step 3: ≤6 rows in CanSlim.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT
-- statements here.
-- ============================================================================

-- Step 1: trades_summary.risk_budget = total_cost for all options.
UPDATE trades_summary s
   SET risk_budget = s.total_cost
 WHERE s.instrument_type = 'OPTION'
   AND s.deleted_at IS NULL
   AND s.risk_budget IS DISTINCT FROM s.total_cost;


-- Step 2: zero out decorative stop_loss values on option BUY detail rows.
UPDATE trades_details
   SET stop_loss = 0
 WHERE instrument_type = 'OPTION'
   AND action          = 'BUY'
   AND stop_loss       > 0
   AND deleted_at     IS NULL;


-- Step 3: clear trades_summary.stop_loss for options where present.
UPDATE trades_summary
   SET stop_loss = NULL
 WHERE instrument_type = 'OPTION'
   AND stop_loss      IS NOT NULL
   AND deleted_at     IS NULL;


-- ============================================================================
-- Verification
-- ============================================================================
-- All three conditions must hold after this migration. RAISE EXCEPTION on
-- any miss so the runner aborts and the user immediately knows something
-- went wrong.
-- ============================================================================
DO $$
DECLARE
    mismatched_risk_budget    INTEGER;
    remaining_option_stops    INTEGER;
    remaining_summary_stops   INTEGER;
BEGIN
    SELECT COUNT(*) INTO mismatched_risk_budget
      FROM trades_summary s
     WHERE s.instrument_type = 'OPTION'
       AND s.deleted_at IS NULL
       AND s.risk_budget IS DISTINCT FROM s.total_cost;

    SELECT COUNT(*) INTO remaining_option_stops
      FROM trades_details
     WHERE instrument_type = 'OPTION'
       AND action          = 'BUY'
       AND stop_loss       > 0
       AND deleted_at     IS NULL;

    SELECT COUNT(*) INTO remaining_summary_stops
      FROM trades_summary
     WHERE instrument_type = 'OPTION'
       AND stop_loss      IS NOT NULL
       AND deleted_at     IS NULL;

    IF mismatched_risk_budget > 0 THEN
        RAISE EXCEPTION 'Migration 023 verification failed: % option summary row(s) still have risk_budget != total_cost',
                        mismatched_risk_budget;
    END IF;

    IF remaining_option_stops > 0 THEN
        RAISE EXCEPTION 'Migration 023 verification failed: % option BUY detail row(s) still have stop_loss > 0',
                        remaining_option_stops;
    END IF;

    IF remaining_summary_stops > 0 THEN
        RAISE EXCEPTION 'Migration 023 verification failed: % option summary row(s) still have stop_loss IS NOT NULL',
                        remaining_summary_stops;
    END IF;

    RAISE NOTICE 'Migration 023 complete: option Trade Risk $ aligned with premium for all rows.';
END $$;
