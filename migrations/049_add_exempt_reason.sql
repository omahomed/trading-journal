-- migrations/049_add_exempt_reason.sql
--
-- Add-on exempt-reason capture for the §2 Window rule (Pyramid v6).
--
-- The rule this supports:
--   Pyramid Sizer Rule 7 WINDOW rejects at-level add tranches once the
--   current price is > 15% above the B1 fill price. Two exemptions
--   allow the sizer to proceed anyway:
--     'sr8_rebuild' — RS-governed rebuild after an SR8 fire
--     'fresh_base'  — §3 structural breakout from a qualifying new base
--
-- Both exemptions are USER-DECLARED at commit time (not auto-detected).
-- Structural-breakout and RS-rebuild judgments are context-dependent
-- and mis-classify at the edges; a false-positive silently lets an
-- unqualified add through the gate. Forcing the trader to tick a box +
-- pick a reason is the codification — the audit trail is the point.
--
-- Storage rationale: column on trades_details (per-lot, since each
-- BUY row is either exempt or not — exemptions don't propagate to
-- siblings). VARCHAR(20) fits both current tokens + room to grow;
-- CHECK constraint enforces the closed enum so downstream analyzers
-- can trust the values. NULL is the default (not exempt / not
-- declared / pre-v6 add).
--
-- Post-30-adds review: analysts filter export/CSV by add_exempt_reason
-- to bucket by declared reason and outcome. That's the whole reason
-- the field exists — the sizer already gates without persistence, but
-- without persistence there's no way to answer "how did the exempt
-- calls actually pan out?"
--
-- Idempotent: ADD COLUMN IF NOT EXISTS is a no-op on re-run. NULL
-- default is correct — every historical row is legitimately null (no
-- one declared exempt before the rule existed).

DO $$
BEGIN
    ALTER TABLE trades_details
      ADD COLUMN IF NOT EXISTS add_exempt_reason VARCHAR(20) NULL;
    RAISE NOTICE 'Added add_exempt_reason column to trades_details';
END $$;

-- Enum enforcement. Add via DO block so re-running the migration
-- doesn't error on "constraint already exists" — the pattern used by
-- migration 016 for trades_summary_instrument_type_check.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.check_constraints
         WHERE constraint_name = 'trades_details_add_exempt_reason_check'
    ) THEN
        ALTER TABLE trades_details
          ADD CONSTRAINT trades_details_add_exempt_reason_check
          CHECK (add_exempt_reason IN ('sr8_rebuild', 'fresh_base') OR add_exempt_reason IS NULL);
        RAISE NOTICE 'Added CHECK constraint on trades_details.add_exempt_reason';
    END IF;
END $$;

-- RLS: same explicit-SELECT grant pattern as migration 048.
GRANT SELECT (add_exempt_reason) ON trades_details TO app_runtime;

-- Sanity check: column must be nullable. Symmetric with migration 048.
DO $$
DECLARE
    v_nullable TEXT;
BEGIN
    SELECT is_nullable INTO v_nullable
      FROM information_schema.columns
     WHERE table_name = 'trades_details'
       AND column_name = 'add_exempt_reason';
    IF v_nullable <> 'YES' THEN
        RAISE EXCEPTION 'add_exempt_reason must be NULL-able; found %', v_nullable;
    END IF;
END $$;
