-- ============================================================================
-- Migration 021: Backfill risk_budget using sizing-mode × prior-day NLV
-- ============================================================================
-- Migration 020 backfilled risk_budget using
--   SUM_over_BUYs(shares × (entry - detail.stop_loss) × multiplier)
-- but detail.stop_loss is mutated by update_trade_stops, so post-stop-move
-- trades got post-move risk values (typically too low; R values inflated).
--
-- This migration restores the user's actual position-sizing rule for trades
-- that migration 020 left at 0/NULL (either because no BUY had a stop, or
-- the formula floored to 0):
--
--   risk_budget = sizing_mode_pct × prior_day_end_nlv
--
-- where sizing_mode_pct is derived from the most-recent prior journal
-- entry's market_cycle (MCT state), per frontend/src/lib/sizing-mode.ts:
--
--   POWERTREND / UPTREND  → Offense  → 1.00%  (0.01)
--   RALLY MODE            → Normal   → 0.75%  (0.0075)
--   CORRECTION            → Defense  → 0.50%  (0.005)
--   unknown / null        → Normal   → 0.75%  (safe-middle default;
--                                              matches mctStateToSizingMode's
--                                              DEFAULT_INDEX = 1)
--
-- prior_day_end_nlv is from the most-recent trading_journal row strictly
-- before the trade's open_date (handles weekends/holidays naturally — the
-- LATERAL JOIN walks back to whichever day had a journal entry with NLV).
-- Same prior entry sources both market_cycle AND end_nlv: one lookup,
-- deterministic, matches the user's "I sized using yesterday's known
-- regime and yesterday's known equity" workflow.
--
-- Filter: only updates rows where risk_budget IS NULL or = 0. Existing
-- non-zero values are preserved (per phase-2 sweep convention; user
-- explicitly opted out of touching trades whose stops were moved).
-- The =0 case includes legitimate "no stop set, floored by calc_risk_budget"
-- entries — overwriting is intentional, since the user's actual rule says
-- every trade has sizing_mode × NLV regardless of stop status.
--
-- UPPER(market_cycle) in the CASE arms for casing tolerance. Legacy data
-- isn't guaranteed to use the canonical 'POWERTREND' casing; misses default
-- to Normal (safe-middle) which is the same fallback as mctStateToSizingMode.
--
-- Idempotent: re-running is a no-op once all candidate rows have a value.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT
-- statements here.
-- ============================================================================

WITH candidates AS (
    SELECT s.id            AS summary_id,
           s.portfolio_id,
           s.trade_id,
           s.open_date,
           prior.market_cycle,
           prior.end_nlv
      FROM trades_summary s
      JOIN LATERAL (
           SELECT j.market_cycle, j.end_nlv
             FROM trading_journal j
            WHERE j.portfolio_id = s.portfolio_id
              AND j.day < s.open_date::date
              AND j.deleted_at IS NULL
              AND j.end_nlv IS NOT NULL
              AND j.end_nlv > 0
            ORDER BY j.day DESC
            LIMIT 1
      ) prior ON TRUE
     WHERE s.deleted_at IS NULL
       AND (s.risk_budget IS NULL OR s.risk_budget = 0)
)
UPDATE trades_summary s
   SET risk_budget = ROUND((
       CASE
         WHEN UPPER(c.market_cycle) IN ('POWERTREND', 'UPTREND') THEN 0.01
         WHEN UPPER(c.market_cycle) = 'RALLY MODE'               THEN 0.0075
         WHEN UPPER(c.market_cycle) = 'CORRECTION'               THEN 0.005
         ELSE 0.0075
       END * c.end_nlv
   )::numeric, 2)
  FROM candidates c
 WHERE s.id = c.summary_id;


-- ============================================================================
-- Verification
-- ============================================================================
-- After the UPDATE above, no trade should remain at risk_budget=0/NULL
-- if a prior journal entry was available. If `remaining_with_journal > 0`,
-- the migration missed something and the runner aborts via RAISE EXCEPTION.
--
-- Trades with `remaining_no_journal` > 0 are expected (no source data —
-- e.g. trades from before the journal started). Reported via NOTICE only.
-- ============================================================================
DO $$
DECLARE
    remaining_no_journal     INTEGER;
    remaining_with_journal   INTEGER;
BEGIN
    SELECT COUNT(*) INTO remaining_no_journal
      FROM trades_summary s
     WHERE s.deleted_at IS NULL
       AND (s.risk_budget IS NULL OR s.risk_budget = 0)
       AND NOT EXISTS (
           SELECT 1 FROM trading_journal j
            WHERE j.portfolio_id = s.portfolio_id
              AND j.day < s.open_date::date
              AND j.deleted_at IS NULL
              AND j.end_nlv IS NOT NULL
              AND j.end_nlv > 0
       );

    SELECT COUNT(*) INTO remaining_with_journal
      FROM trades_summary s
     WHERE s.deleted_at IS NULL
       AND (s.risk_budget IS NULL OR s.risk_budget = 0)
       AND EXISTS (
           SELECT 1 FROM trading_journal j
            WHERE j.portfolio_id = s.portfolio_id
              AND j.day < s.open_date::date
              AND j.deleted_at IS NULL
              AND j.end_nlv IS NOT NULL
              AND j.end_nlv > 0
       );

    RAISE NOTICE 'Migration 021: % trade(s) remain unfilled because no prior journal entry exists (expected, no source data).',
                 remaining_no_journal;

    IF remaining_with_journal > 0 THEN
        RAISE EXCEPTION 'Migration 021 verification failed: % trade(s) remain with risk_budget=0/NULL despite a prior journal entry being available.',
                        remaining_with_journal;
    END IF;
END $$;
