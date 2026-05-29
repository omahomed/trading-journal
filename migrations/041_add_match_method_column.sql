-- migrations/041_add_match_method_column.sql
--
-- Phase 2 B-0: add the match_method column to trades_details.
--
-- The column carries a per-SELL stamp recording which matching method
-- consumed inventory at the time the sell was logged. After this
-- migration the column is populated but unused by application code —
-- nothing in api/ or scripts/ reads it yet. Phase 2 B-1 wires up
-- HCFO and the per-call MATCH_METHOD resolution; B-2 consolidates
-- the three inline matching paths into compute_match_summary; B-3
-- renames the now-method-agnostic helpers.
--
-- Stamps:
--   'LIFO' — Last-In-First-Out (current historical behavior).
--   'HCFO' — Highest-Cost-First-Out (tax-optimization, B-1+).
--   NULL   — BUY rows and any future non-matching action type.
--
-- Backfill scope:
--   trades_details.action vocabulary is exactly {'BUY','SELL'} per
--   schema.sql:119 and code (api/main.py + trade_calc.py both
--   uppercase and compare against those two literals only). Stocks
--   and options use the same vocab — an exercised option contract
--   writes a SELL row on the option leg + a BUY row on the stock
--   leg, both at the canonical literal. SELL is the inventory-
--   reducing action; every historical SELL was matched LIFO, so
--   every existing SELL gets stamped 'LIFO'. Soft-deleted SELLs
--   are stamped too: their historical match was LIFO at write
--   time, and the stamp survives any later un-delete.
--
-- Idempotency:
--   - ADD COLUMN IF NOT EXISTS
--   - CHECK constraint guarded by pg_constraint existence probe
--   - UPDATE filters on match_method IS NULL, so re-running
--     matches zero rows.
--
-- Verification: the trailing DO block fails loudly if any row with
-- action='SELL' is still NULL after backfill. Founder UUID is set
-- per-migration by migrations/run.py, so RLS / DEFAULT lookups
-- resolve cleanly inside this transaction.

-- ----------------------------------------------------------------
-- Step 1: ADD COLUMN (TEXT, nullable). Nullable so BUY rows stay
-- valid; the CHECK constraint allows NULL explicitly.
-- ----------------------------------------------------------------
ALTER TABLE trades_details
  ADD COLUMN IF NOT EXISTS match_method TEXT;

-- ----------------------------------------------------------------
-- Step 2: CHECK constraint. Named so re-runs can skip cleanly.
-- ----------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'trades_details_match_method_check'
  ) THEN
    ALTER TABLE trades_details
      ADD CONSTRAINT trades_details_match_method_check
      CHECK (match_method IN ('LIFO', 'HCFO') OR match_method IS NULL);
  END IF;
END $$;

-- ----------------------------------------------------------------
-- Step 3: backfill historical SELLs with 'LIFO'. Explicit IN list,
-- not LIKE / substring — the action vocab is pinned to {BUY,SELL}.
-- Filter on match_method IS NULL so this is a no-op on re-run (and
-- on any rows already migrated by hand).
-- ----------------------------------------------------------------
UPDATE trades_details
   SET match_method = 'LIFO'
 WHERE match_method IS NULL
   AND action IN ('SELL');

-- ----------------------------------------------------------------
-- Step 4: verification. Fail loudly if any SELL is still unstamped;
-- emit a NOTICE counter on success. expected_count is the total
-- number of SELL rows (including soft-deleted) and must equal the
-- post-backfill stamped count.
-- ----------------------------------------------------------------
DO $$
DECLARE
  unstamped_count  INTEGER;
  expected_count   INTEGER;
  stamped_count    INTEGER;
BEGIN
  SELECT COUNT(*) INTO unstamped_count
    FROM trades_details
   WHERE action IN ('SELL')
     AND match_method IS NULL;

  SELECT COUNT(*) INTO expected_count
    FROM trades_details
   WHERE action IN ('SELL');

  SELECT COUNT(*) INTO stamped_count
    FROM trades_details
   WHERE action IN ('SELL')
     AND match_method = 'LIFO';

  IF unstamped_count > 0 THEN
    RAISE EXCEPTION
      'Migration 041 backfill incomplete: % SELL row(s) still '
      'have NULL match_method (expected 0 of % total SELL rows). '
      'Investigate trades_details.action values for unexpected '
      'inventory-reducing types before re-running.',
      unstamped_count, expected_count;
  END IF;

  RAISE NOTICE
    'Migration 041 complete: % of % SELL rows stamped with LIFO',
    stamped_count, expected_count;
END $$;
