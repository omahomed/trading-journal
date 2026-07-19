-- migrations/048_sr8_activation_anchor.sql
--
-- SR8 activation anchor: three fields on trades_summary that lock the
-- SR8 cascade's trim targets to the campaign's activation moment.
--
-- The bug this fixes:
--   SR8 Quick / Quicksand trim targets used to compute against LIVE
--   NLV. Core share count was fixed at activation; targets grew with
--   NAV appreciation. When NAV outpaced the core, Quick/QS targets in
--   shares exceeded held shares → trim signals no-op'd → cores went
--   undefended on valid signals. Confirmed live on BE (core set at
--   $430K NAV, Quick fired against $806K → target 319 shs vs 224
--   held → zero trim on a valid signal).
--
-- The fix:
--   Persist activation-day NLV + activation date + core shares. Trim
--   target formulas become:
--     quick_target_shares     = 0.10 × sr8_activation_nlv / current_px
--     quicksand_target_shares = 0.05 × sr8_activation_nlv / current_px
--     grateful_dead           = 0 (unchanged)
--
--   core_shares captures shares held the moment cushion first crossed
--   +50% from B1 entry price. Adds during SR8 belong to the trim-first
--   cohort — core stays fixed.
--
--   Cap-restore (rebuild ceiling on RS reclaim) STAYS live-NAV — it's
--   a downside defense (can't rebuild past what current NAV supports).
--   When live-NAV cap < core_shares, cap wins; logged for audit.
--
--   Cleared on full exit / new campaign (natural: closing a campaign
--   sets these to NULL implicitly since the row status flips to CLOSED
--   and any re-open creates a fresh row).
--
-- Backfill: a separate one-shot script (scripts/sr8_activation_backfill.py)
-- walks the daily-price + trading_journal history for each currently-open
-- SR8-tier campaign, finds the day cushion first crossed +50%, and
-- populates the trio. Legacy positions that already breached +50% get
-- historically-accurate anchors; positions still below +50% stay NULL
-- until they naturally cross.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS is a no-op on re-run. NULL
-- defaults are correct — positions that haven't hit +50% yet legitimately
-- have no activation.

DO $$
BEGIN
    ALTER TABLE trades_summary
      ADD COLUMN IF NOT EXISTS sr8_activation_date DATE          NULL,
      ADD COLUMN IF NOT EXISTS sr8_activation_nlv  NUMERIC(15,2) NULL,
      ADD COLUMN IF NOT EXISTS sr8_core_shares     NUMERIC(12,4) NULL;
    RAISE NOTICE 'Added SR8 activation-anchor columns to trades_summary';
END $$;

-- Same RLS pattern as migration 046 — the app_runtime role needs
-- explicit SELECT on new columns.
GRANT SELECT (
    sr8_activation_date, sr8_activation_nlv, sr8_core_shares
) ON trades_summary TO app_runtime;

-- Sanity check: the three columns must be NULL-able. If a future
-- ALTER accidentally sets NOT NULL, migration re-runs will fail loudly
-- via this constraint check.
DO $$
DECLARE
    v_nullable_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO v_nullable_count
      FROM information_schema.columns
     WHERE table_name = 'trades_summary'
       AND column_name IN ('sr8_activation_date', 'sr8_activation_nlv', 'sr8_core_shares')
       AND is_nullable = 'YES';
    IF v_nullable_count <> 3 THEN
        RAISE EXCEPTION 'SR8 activation columns must all be NULL-able; found % nullable', v_nullable_count;
    END IF;
END $$;
