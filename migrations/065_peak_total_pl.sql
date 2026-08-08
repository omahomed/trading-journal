-- ============================================================================
-- Migration 065: peak_total_pl — SR12 MCP anchor rewrite
-- ============================================================================
-- Follow-up to migration 064. The original SR12 implementation anchored the
-- Ratcheting Profit Floor to B1 fill × peak_return_pct/2. The 2026-08-07
-- DELL walkthrough exposed the doctrinal bug: for scaled-in positions, the
-- B1 anchor doesn't protect what MCP was ever supposed to protect — "half
-- of what I've earned through this trip." A B1-anchored stop on DELL
-- (avg_entry $331 after 15 add-ons, B1 $176) fires at $322 and locks in
-- a NET LOSS on the aggregate position vs. the seemingly-safe "half of B1
-- peak gain" it promises on paper.
--
-- New anchor: `peak_total_pl` = the maximum value of
--   (realized_bank_at_that_time + shares_at_that_time × (day_high - avg_cost))
-- ever observed since B1, using end-of-day state on each bar. This is
-- literally "the biggest total P&L this campaign ever showed" — the number
-- MCP was written to protect.
--
-- Frontend recomputes target broker stop as:
--   target_stop = avg_entry + (peak_total_pl/2 - realized_bank) / shares
-- so the "half of peak" invariant holds on the aggregate. When
-- current_realized already exceeds peak_total_pl/2 (e.g. after a big trim),
-- the nudge auto-clears — no floor to enforce beyond the bank you've
-- already locked in.
--
-- Column semantics:
--   peak_total_pl — sticky max of total P&L (realized + unrealized-at-high).
--                   Ratcheted UP only. NULL until first backfill / reconcile.
--
-- Backfill: scripts/backfill_peak_total_pl.py walks daily bars from B1 date
-- for every armed campaign (b1_max_return_pct >= 50), reconstructing shares
-- / avg_cost / realized_bank per bar and picking the day-high compute. Run
-- separately after this migration commits.
--
-- Predecessor: sr12_floor_pct (migration 064) stays on the table but is no
-- longer read by anything after this migration + frontend/backend cutover.
-- Left in place so downstream tooling that references the column doesn't
-- crash mid-deploy. Retire in a later migration once cutover is stable.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

ALTER TABLE trades_summary
    ADD COLUMN IF NOT EXISTS peak_total_pl DECIMAL(14, 2);

COMMENT ON COLUMN trades_summary.peak_total_pl IS
    'SR12 anchor (post-migration-065): the maximum of (realized_bank + '
    'shares × (day_high - avg_cost)) ever observed since B1, using '
    'end-of-day state per bar. Sticky max, ratcheted up only by the '
    'daily b1_reconcile sweep. Frontend derives the target broker stop as '
    'avg_entry + (peak_total_pl/2 - realized_bank) / shares. NULL = not '
    'yet backfilled or campaign never cushion-qualified. Supersedes '
    'sr12_floor_pct (migration 064) which was B1-anchored and produced '
    'wrong targets on scaled-in positions. See migration 065.';


-- ============================================================================
-- Verification queries (manual, after COMMIT + backfill script)
-- ============================================================================
--   SELECT column_name, data_type, is_nullable, column_default
--     FROM information_schema.columns
--    WHERE table_name = 'trades_summary' AND column_name = 'peak_total_pl';
--   → numeric(14,2), YES, null
--
--   SELECT p.name, s.trade_id, s.ticker, s.b1_max_return_pct,
--          s.realized_pl, s.peak_total_pl,
--          ROUND(s.avg_entry +
--                (s.peak_total_pl / 2 - s.realized_pl) / NULLIF(s.shares, 0),
--                2) AS derived_target_stop
--     FROM trades_summary s
--     JOIN portfolios p ON p.id = s.portfolio_id
--    WHERE s.deleted_at IS NULL AND s.peak_total_pl IS NOT NULL
--   ORDER BY s.peak_total_pl DESC;
--   → DELL (CanSlim / 202604-013) should top the list; derived_target_stop
--     should land ~$400 vs. the pre-065 $322.90.
