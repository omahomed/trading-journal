-- migrations/036_b1_max_return_pct.sql
--
-- Persistent Sell Rule tier classification. The Active Campaign Summary's
-- Sell Rule column (SR1 / SR11 / SR8) is fundamentally STATE, not a pure
-- derivation from current price:
--   * SR11 disengages once SR8 activates (one-way absorption)
--   * SR8 cores never auto-demote on pullback — they only retire on SR8's
--     own signals (Quick / Quicksand / Grateful Dead) or SR13
--
-- The previous implementation classified from CURRENT B1 return %, which
-- mis-tiered any leader on a pullback (e.g. a position that peaked at +70%
-- and pulled back to +30% would mis-render as SR11 instead of SR8).
--
-- Fix: store the MAX B1 return % ever observed per OPEN campaign. Classify
-- from that max. Auto-promote on observation; never auto-demote.
--
-- Storage: single NUMERIC column on trades_summary, nullable. Negatives are
-- valid (positions that peaked at a loss). NULL means "no observation yet"
-- — frontend falls back to current B1 return for classification, and the
-- first POST /api/trades/{trade_id}/update-b1-max writes the seed value.
-- A separate one-time backfill script populates NULLs for existing OPEN
-- equity campaigns from yfinance daily-close history.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS is a no-op on re-run.

DO $$
BEGIN
    ALTER TABLE trades_summary
      ADD COLUMN IF NOT EXISTS b1_max_return_pct NUMERIC(10,4) NULL;
    RAISE NOTICE 'Added b1_max_return_pct to trades_summary';
END $$;
