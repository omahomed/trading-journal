-- migrations/046_mae_mfe_tracking.sql
--
-- Per-trade excursion metrics: Maximum Adverse Excursion, Maximum
-- Favorable Excursion, and mid-run retracement — all measured against
-- the B-series entry price on daily bars. Feeds two things:
--
--   1. Stop-multiple calibration for the New Entry sizer (is 1.5× ATR21
--      the right initial-stop distance?). Requires MAE across a
--      material sample of closed campaigns.
--   2. Retracement-tolerance analytics on winners (how far did leaders
--      pull back mid-run before making new highs?).
--
-- All excursions measured vs the B1 (initial buy) fill price, NOT the
-- blended cost. Blended cost moves with add-ons and would corrupt the
-- stop-calibration signal (a leader whose average is dragged up by a
-- late scale-in would falsely register as never having drawn down).
--
-- atr21_entry_pct is a FROZEN snapshot: ATR21% computed on the ~30
-- daily bars ending the day BEFORE entry. Once populated, never
-- recomputed. This is what the ×ATR multiples divide by, so it has to
-- stay stable across daily reconciles or the stored ratios drift.
--
-- Daily reconcile updates the running min-low / max-high derived
-- fields; only atr21_entry_pct is one-shot. All fields skipped for
-- options (consistent with b1_reconcile's `COALESCE(instrument_type,
-- 'STOCK') = 'STOCK'` filter — yfinance doesn't serve OCC option
-- symbols and the concept doesn't map cleanly to options anyway).
--
-- Storage rationale: columns on trades_summary rather than a 1:1 side
-- table. Matches the b1_max_return_pct precedent (migration 036);
-- avoids joining every load_summary reader; 7 NULL-defaulted columns
-- on a 30+ column table is cheap. Reads flow through the existing
-- load_summary path automatically.
--
-- ×ATR multiples (mae_atr, mfe_atr, max_retrace_atr) are NOT stored —
-- they're one division at read time. Storing them would risk drift
-- between the raw % and the multiple after any retro edit of
-- atr21_entry_pct; deriving keeps a single source of truth.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS is a no-op on re-run. Existing
-- rows initialize to NULL — the daily reconciler picks them up on its
-- next sweep, and the one-time backfill script does the same for the
-- current open positions.

DO $$
BEGIN
    ALTER TABLE trades_summary
      ADD COLUMN IF NOT EXISTS mae_pct                 NUMERIC(10,4) NULL,
      ADD COLUMN IF NOT EXISTS mfe_pct                 NUMERIC(10,4) NULL,
      ADD COLUMN IF NOT EXISTS atr21_entry_pct         NUMERIC(10,4) NULL,
      ADD COLUMN IF NOT EXISTS days_to_mae             INTEGER       NULL,
      ADD COLUMN IF NOT EXISTS days_to_mfe             INTEGER       NULL,
      ADD COLUMN IF NOT EXISTS max_retrace_pct         NUMERIC(10,4) NULL,
      ADD COLUMN IF NOT EXISTS mae_mfe_last_updated    DATE          NULL;
    RAISE NOTICE 'Added MAE/MFE tracking columns to trades_summary';
END $$;

-- app_runtime is the read-only role RLS enforces per-user visibility
-- through. Explicit SELECT grant on the new columns follows the same
-- pattern migration 010 uses for market_data.
GRANT SELECT (
    mae_pct, mfe_pct, atr21_entry_pct,
    days_to_mae, days_to_mfe, max_retrace_pct,
    mae_mfe_last_updated
) ON trades_summary TO app_runtime;

-- No new index. The daily reconciler filters via the existing
-- (status='OPEN' AND deleted_at IS NULL) predicate, which already has
-- coverage; per-trade reads route through load_summary and hit the
-- existing PK-adjacent path.
