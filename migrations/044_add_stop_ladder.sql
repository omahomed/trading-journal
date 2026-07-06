-- ============================================================================
-- Migration 044: add stop_ladder JSONB column to trades_details
-- ============================================================================
-- Position Sizer Phase 1: persist a 3-leg staged exit ("scale-out ladder")
-- alongside the existing single stop_loss column. When a buy is logged
-- with the Ladder stop mode, the ladder is stored here; the primary
-- stop_loss column continues to hold the first-firing leg's price so
-- every legacy read path (Trade Journal, Risk Manager, Portfolio Heat)
-- stays coherent without a null-check on every read.
--
-- Storage shape:
--   {
--     "legs": [
--       {"pct": 3, "shares": 5},
--       {"pct": 5, "shares": 5},
--       {"pct": 7, "shares": 6}
--     ]
--   }
--
-- Percentages are locked at [3, 5, 7] and validated in the log_buy
-- handler. Store pcts (not absolute prices) so the ladder stays coherent
-- if entry price is edited later — prices are derivable from
-- entry × (1 - pct/100).
--
-- Nullable: existing single-stop buys leave this column NULL and behave
-- exactly as they always have.
-- ============================================================================

ALTER TABLE trades_details
    ADD COLUMN IF NOT EXISTS stop_ladder JSONB NULL;
