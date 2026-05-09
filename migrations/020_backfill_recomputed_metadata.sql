-- ============================================================================
-- Migration 020: Backfill recompute-wiped non-LIFO summary fields
-- ============================================================================
-- The bug fixed in this commit: api/main.py _recompute_summary_lifo passed
-- only LIFO-derived fields (Status, Shares, Avg_Entry, Realized_PL, ...) to
-- db.save_summary_with_closures. The DB layer's UPDATE binds every column
-- including the ones not in the input dict — so rule/buy_notes/sell_rule/
-- sell_notes got NULL and risk_budget got 0 (column DEFAULT). Triggered on
-- every sell, edit, delete, and rebuild.
--
-- This migration restores the wiped data from trades_details, which is the
-- source of truth for transaction-level rule/notes/stop_loss values.
--
-- Recoverable from detail rows:
--   1. rule        ← earliest BUY detail's rule (typically B1)
--   2. buy_notes   ← earliest BUY detail's notes
--   3. sell_rule   ← final SELL detail's rule (closed campaigns only)
--   4. sell_notes  ← final SELL detail's notes (closed campaigns only)
--   5. risk_budget ← SUM over BUY details: shares * (amount - stop_loss) * multiplier
--   6. stop_loss   ← latest BUY detail's stop_loss. Safe because the edit-stop
--                    UI (api/main.py update_trade_stops → db_layer.update_trade_stops
--                    line 1750-1757) writes to BOTH summary.stop_loss AND every
--                    open BUY detail.stop_loss in lockstep — so detail value
--                    reflects the user's current stop, not a frozen entry-time
--                    value. Only backfill where summary.stop_loss IS NULL (the
--                    wipe path). 0 is left untouched: 0 is the legitimate
--                    "no stop set" signal from log_buy.
--
-- Risk_budget caveat: detail.stop_loss can have been mutated by
-- update_trade_stops post-entry. For trades whose stops were never moved,
-- the formula gives the original entry risk. For trades with raised stops
-- (e.g. +10% BE rule), it gives the post-move risk — typically smaller
-- than the original. The R column (= realized_pl / risk_budget) will
-- therefore overstate R for those trades. Acceptable approximation since
-- the alternative is broken R for ~46 campaigns.
--
-- Risk_budget filter: only backfill rows where the bug definitely fired
-- (current value is NULL OR 0) AND the user did set a stop at entry
-- (a BUY detail row has stop_loss > 0). This excludes legitimate
-- "no stop set" entries — calc_risk_budget (trade_calc.py:56) returns 0
-- for those, which is a meaningful "unsized entry" signal, not a wipe.
--
-- Idempotent: each UPDATE has a WHERE clause that skips rows already
-- holding a value, so re-running is a no-op.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT
-- statements here.
-- ============================================================================

-- ============================================================================
-- Pre-flight diagnostics — run these BEFORE applying to confirm the scope.
-- They're SELECTs only (no side effects); copy them into psql to inspect.
-- ============================================================================
-- -- 1A. Campaigns where summary.rule is NULL/empty but a BUY detail has rule:
-- SELECT s.portfolio_id, s.trade_id, s.ticker,
--        (SELECT d.rule FROM trades_details d
--          WHERE d.portfolio_id=s.portfolio_id AND d.trade_id=s.trade_id
--            AND d.action='BUY' AND d.deleted_at IS NULL
--            AND d.rule IS NOT NULL AND TRIM(d.rule)<>''
--          ORDER BY d.date ASC LIMIT 1) AS recoverable_rule
--   FROM trades_summary s
--  WHERE s.deleted_at IS NULL
--    AND (s.rule IS NULL OR TRIM(s.rule)='')
--  ORDER BY s.portfolio_id, s.trade_id;
--
-- -- 1B. Same for buy_notes / sell_rule / sell_notes — substitute column names.
--
-- -- 2. Campaigns where summary.risk_budget is NULL or 0 AND a BUY detail
-- --    has stop_loss > 0 (likely-wipe candidates):
-- SELECT s.portfolio_id, s.trade_id, s.ticker, s.status, s.risk_budget,
--        (SELECT SUM(d.shares * (d.amount - d.stop_loss) * d.multiplier)
--           FROM trades_details d
--          WHERE d.portfolio_id=s.portfolio_id AND d.trade_id=s.trade_id
--            AND d.action='BUY' AND d.deleted_at IS NULL
--            AND d.stop_loss IS NOT NULL AND d.stop_loss > 0) AS computed_rb
--   FROM trades_summary s
--  WHERE s.deleted_at IS NULL
--    AND (s.risk_budget IS NULL OR s.risk_budget = 0)
--  ORDER BY s.portfolio_id, s.trade_id;

-- ============================================================================
-- 1. Backfill rule from earliest BUY detail's rule
-- ============================================================================
WITH first_buy_rule AS (
    SELECT DISTINCT ON (d.portfolio_id, d.trade_id)
           d.portfolio_id, d.trade_id, d.rule
      FROM trades_details d
     WHERE d.action = 'BUY'
       AND d.deleted_at IS NULL
       AND d.rule IS NOT NULL
       AND TRIM(d.rule) <> ''
     ORDER BY d.portfolio_id, d.trade_id, d.date ASC
)
UPDATE trades_summary s
   SET rule = fbr.rule
  FROM first_buy_rule fbr
 WHERE s.portfolio_id = fbr.portfolio_id
   AND s.trade_id = fbr.trade_id
   AND s.deleted_at IS NULL
   AND (s.rule IS NULL OR TRIM(s.rule) = '');

-- ============================================================================
-- 2. Backfill buy_notes from earliest BUY detail's notes
-- ============================================================================
WITH first_buy_notes AS (
    SELECT DISTINCT ON (d.portfolio_id, d.trade_id)
           d.portfolio_id, d.trade_id, d.notes
      FROM trades_details d
     WHERE d.action = 'BUY'
       AND d.deleted_at IS NULL
       AND d.notes IS NOT NULL
       AND TRIM(d.notes) <> ''
     ORDER BY d.portfolio_id, d.trade_id, d.date ASC
)
UPDATE trades_summary s
   SET buy_notes = fbn.notes
  FROM first_buy_notes fbn
 WHERE s.portfolio_id = fbn.portfolio_id
   AND s.trade_id = fbn.trade_id
   AND s.deleted_at IS NULL
   AND (s.buy_notes IS NULL OR TRIM(s.buy_notes) = '');

-- ============================================================================
-- 3. Backfill sell_rule from final SELL detail's rule (closed campaigns)
-- ============================================================================
WITH last_sell_rule AS (
    SELECT DISTINCT ON (d.portfolio_id, d.trade_id)
           d.portfolio_id, d.trade_id, d.rule
      FROM trades_details d
     WHERE d.action = 'SELL'
       AND d.deleted_at IS NULL
       AND d.rule IS NOT NULL
       AND TRIM(d.rule) <> ''
     ORDER BY d.portfolio_id, d.trade_id, d.date DESC
)
UPDATE trades_summary s
   SET sell_rule = lsr.rule
  FROM last_sell_rule lsr
 WHERE s.portfolio_id = lsr.portfolio_id
   AND s.trade_id = lsr.trade_id
   AND s.deleted_at IS NULL
   AND s.status = 'CLOSED'
   AND (s.sell_rule IS NULL OR TRIM(s.sell_rule) = '');

-- ============================================================================
-- 4. Backfill sell_notes from final SELL detail's notes (closed campaigns)
-- ============================================================================
WITH last_sell_notes AS (
    SELECT DISTINCT ON (d.portfolio_id, d.trade_id)
           d.portfolio_id, d.trade_id, d.notes
      FROM trades_details d
     WHERE d.action = 'SELL'
       AND d.deleted_at IS NULL
       AND d.notes IS NOT NULL
       AND TRIM(d.notes) <> ''
     ORDER BY d.portfolio_id, d.trade_id, d.date DESC
)
UPDATE trades_summary s
   SET sell_notes = lsn.notes
  FROM last_sell_notes lsn
 WHERE s.portfolio_id = lsn.portfolio_id
   AND s.trade_id = lsn.trade_id
   AND s.deleted_at IS NULL
   AND s.status = 'CLOSED'
   AND (s.sell_notes IS NULL OR TRIM(s.sell_notes) = '');

-- ============================================================================
-- 5. Backfill risk_budget from SUM over BUY details
-- ============================================================================
-- Formula matches calc_risk_budget(trade_calc.py:56):
--   shares * (entry - stop_loss) * multiplier, floored at 0.
-- Here we floor at 0 by filtering the inner rows to stop_loss > 0 AND
-- amount > stop_loss; calc_risk_budget would otherwise return 0 for those
-- entries and their contribution is irrelevant.
--
-- Filter: only backfill rows where the bug definitely fired AND the user
-- did set a stop at entry (excludes legitimate "no stop" entries).
WITH buy_risk AS (
    SELECT d.portfolio_id, d.trade_id,
           ROUND(SUM(d.shares * (d.amount - d.stop_loss) * d.multiplier)::numeric, 2)
               AS computed_rb
      FROM trades_details d
     WHERE d.action = 'BUY'
       AND d.deleted_at IS NULL
       AND d.stop_loss IS NOT NULL
       AND d.stop_loss > 0
       AND d.amount > d.stop_loss
     GROUP BY d.portfolio_id, d.trade_id
    HAVING SUM(d.shares * (d.amount - d.stop_loss) * d.multiplier) > 0
)
UPDATE trades_summary s
   SET risk_budget = br.computed_rb
  FROM buy_risk br
 WHERE s.portfolio_id = br.portfolio_id
   AND s.trade_id = br.trade_id
   AND s.deleted_at IS NULL
   AND (s.risk_budget IS NULL OR s.risk_budget = 0);

-- ============================================================================
-- 6. Backfill stop_loss from latest BUY detail's stop_loss
-- ============================================================================
-- Source: the most-recent BUY detail (by date) for each campaign. update_trade_stops
-- (db_layer.py:1750-1757) updates every open BUY detail's stop_loss in lockstep
-- with the summary, so any BUY detail's stop_loss reflects the user's most
-- recent adjustment. Picking the latest BUY by date is a tie-break for scale-in
-- campaigns where update_trade_stops may not have run since the last add-on.
--
-- Filter: only backfill where summary.stop_loss IS NULL (the wipe path). 0 is
-- left untouched — it's the legitimate "no stop set" signal from log_buy.
WITH last_buy_stop AS (
    SELECT DISTINCT ON (d.portfolio_id, d.trade_id)
           d.portfolio_id, d.trade_id, d.stop_loss
      FROM trades_details d
     WHERE d.action = 'BUY'
       AND d.deleted_at IS NULL
       AND d.stop_loss IS NOT NULL
       AND d.stop_loss > 0
     ORDER BY d.portfolio_id, d.trade_id, d.date DESC
)
UPDATE trades_summary s
   SET stop_loss = lbs.stop_loss
  FROM last_buy_stop lbs
 WHERE s.portfolio_id = lbs.portfolio_id
   AND s.trade_id = lbs.trade_id
   AND s.deleted_at IS NULL
   AND s.stop_loss IS NULL;

-- ============================================================================
-- Verification: report (do not raise) on any remaining gaps so an operator
-- running the migration sees the residual at-a-glance. We do NOT raise here
-- because some campaigns legitimately have no recoverable rule (e.g. a
-- direct DB import without rules ever populated).
-- ============================================================================
DO $$
DECLARE
    n_rule_gap     INTEGER;
    n_buy_gap      INTEGER;
    n_risk_gap     INTEGER;
    n_stop_gap     INTEGER;
BEGIN
    -- Rule still NULL/empty but a BUY detail has a rule we could have backfilled.
    SELECT COUNT(*) INTO n_rule_gap
      FROM trades_summary s
     WHERE s.deleted_at IS NULL
       AND (s.rule IS NULL OR TRIM(s.rule) = '')
       AND EXISTS (
           SELECT 1 FROM trades_details d
            WHERE d.portfolio_id = s.portfolio_id
              AND d.trade_id = s.trade_id
              AND d.action = 'BUY'
              AND d.deleted_at IS NULL
              AND d.rule IS NOT NULL
              AND TRIM(d.rule) <> ''
       );

    SELECT COUNT(*) INTO n_buy_gap
      FROM trades_summary s
     WHERE s.deleted_at IS NULL
       AND (s.buy_notes IS NULL OR TRIM(s.buy_notes) = '')
       AND EXISTS (
           SELECT 1 FROM trades_details d
            WHERE d.portfolio_id = s.portfolio_id
              AND d.trade_id = s.trade_id
              AND d.action = 'BUY'
              AND d.deleted_at IS NULL
              AND d.notes IS NOT NULL
              AND TRIM(d.notes) <> ''
       );

    SELECT COUNT(*) INTO n_risk_gap
      FROM trades_summary s
     WHERE s.deleted_at IS NULL
       AND (s.risk_budget IS NULL OR s.risk_budget = 0)
       AND EXISTS (
           SELECT 1 FROM trades_details d
            WHERE d.portfolio_id = s.portfolio_id
              AND d.trade_id = s.trade_id
              AND d.action = 'BUY'
              AND d.deleted_at IS NULL
              AND d.stop_loss IS NOT NULL
              AND d.stop_loss > 0
              AND d.amount > d.stop_loss
       );

    -- Stop_loss gap: summary still NULL but a BUY detail has stop_loss > 0.
    SELECT COUNT(*) INTO n_stop_gap
      FROM trades_summary s
     WHERE s.deleted_at IS NULL
       AND s.stop_loss IS NULL
       AND EXISTS (
           SELECT 1 FROM trades_details d
            WHERE d.portfolio_id = s.portfolio_id
              AND d.trade_id = s.trade_id
              AND d.action = 'BUY'
              AND d.deleted_at IS NULL
              AND d.stop_loss IS NOT NULL
              AND d.stop_loss > 0
       );

    IF n_rule_gap > 0 OR n_buy_gap > 0 OR n_risk_gap > 0 OR n_stop_gap > 0 THEN
        RAISE EXCEPTION
            'Migration 020 verification failed: rule_gap=% buy_notes_gap=% risk_budget_gap=% stop_loss_gap=%',
            n_rule_gap, n_buy_gap, n_risk_gap, n_stop_gap;
    END IF;
END $$;
