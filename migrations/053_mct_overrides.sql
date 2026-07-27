-- ============================================================================
-- Migration 053: mct_overrides — user-declared CORRECTION override for M Factor
-- ============================================================================
-- The systematic M Factor state (POWERTREND / UPTREND / UUP / RALLY MODE /
-- CORRECTION) is engine-derived from market data; the CORRECTION
-- declaration in particular requires 2 closes < 50 SMA AND a 10% drawdown
-- from the reference high (see api/mct_engine.py:CORRECTION_DRAWDOWN).
--
-- Real-world discipline needs an escape hatch: when authoritative external
-- signals (e.g., IBD moving exposure to 0–20%, NAAIM survey collapse)
-- indicate a correction is here BEFORE the depth threshold is hit, the
-- user should be able to force the state into CORRECTION so risk
-- management downstream reacts. But un-taxed discretion is a rationalization
-- tool, so this table intentionally forces:
--
--   * a mandatory reason (min 40 chars) so cursor-blank clicks are hard
--   * a stamped activated_at + activated_date_ct so retro review can
--     compare override date to when the systematic rule finally fired
--   * cleared_at + cleared_by ('auto' | 'user') to distinguish overrides
--     the rule caught up with vs. ones the user manually retracted
--
-- Auto-clear policy lives in the /api/market/rally-prefix endpoint: when
-- the systematic state returns to POWERTREND / UPTREND (market recovered)
-- OR itself becomes CORRECTION (rule caught up), the active override is
-- cleared with cleared_by='auto'.
--
-- Partial unique index on (user_id) WHERE cleared_at IS NULL enforces one
-- active override per user. History rows (cleared_at NOT NULL) are
-- unlimited — the whole point is quarterly review.
--
-- RLS + FORCE mirrors migration 050 pattern; user_id DEFAULT from session.
-- No audit trigger (personal discipline data, same precedent as
-- pinned_routes / weekly_retros / routine_items).
-- ============================================================================

CREATE TABLE IF NOT EXISTS mct_overrides (
    id                  SERIAL          PRIMARY KEY,
    user_id             UUID            NOT NULL REFERENCES users(id) ON DELETE RESTRICT
                                         DEFAULT (
                                             COALESCE(
                                                 NULLIF(current_setting('app.user_id', true), '')::uuid,
                                                 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
                                             )
                                         ),
    activated_at        TIMESTAMPTZ     NOT NULL DEFAULT now(),
    activated_date_ct   DATE            NOT NULL,
    cleared_at          TIMESTAMPTZ,
    cleared_date_ct     DATE,
    cleared_by          TEXT            CHECK (cleared_by IS NULL OR cleared_by IN ('auto', 'user')),
    reason              TEXT            NOT NULL CHECK (length(reason) >= 40)
);

-- Same CT-day trigger pattern as routine_log (migration 050): the trigger
-- stamps activated_date_ct on INSERT and cleared_date_ct on UPDATE when
-- cleared_at is set. Postgres won't accept AT TIME ZONE in a stored
-- expression, so the trigger materializes the values.
CREATE OR REPLACE FUNCTION mct_overrides_set_dates_ct()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') THEN
        NEW.activated_date_ct := (NEW.activated_at AT TIME ZONE 'America/Chicago')::date;
    END IF;
    IF NEW.cleared_at IS NOT NULL AND (OLD IS NULL OR OLD.cleared_at IS DISTINCT FROM NEW.cleared_at) THEN
        NEW.cleared_date_ct := (NEW.cleared_at AT TIME ZONE 'America/Chicago')::date;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_mct_overrides_set_dates_ct ON mct_overrides;
CREATE TRIGGER trg_mct_overrides_set_dates_ct
    BEFORE INSERT OR UPDATE OF cleared_at ON mct_overrides
    FOR EACH ROW EXECUTE FUNCTION mct_overrides_set_dates_ct();

-- One active override per user at a time. Cleared rows are unrestricted.
CREATE UNIQUE INDEX IF NOT EXISTS idx_mct_overrides_one_active
    ON mct_overrides (user_id)
    WHERE cleared_at IS NULL;

-- Timeline read for review (newest first).
CREATE INDEX IF NOT EXISTS idx_mct_overrides_user_activated
    ON mct_overrides (user_id, activated_at DESC);

ALTER TABLE mct_overrides ENABLE ROW LEVEL SECURITY;
ALTER TABLE mct_overrides FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS mct_overrides_isolation ON mct_overrides;
CREATE POLICY mct_overrides_isolation ON mct_overrides FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ============================================================================
-- Verification (manual, after COMMIT)
-- ============================================================================
--   SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class
--    WHERE relname = 'mct_overrides';
--   -- expect: mct_overrides | true | true
--
--   -- Sanity insert (via session with app.user_id set):
--   INSERT INTO mct_overrides (reason) VALUES ('short') ;  -- expect error (<40)
--   INSERT INTO mct_overrides (reason) VALUES ('IBD moved exposure to 0-20% today, correction call');
--   -- expect success; activated_date_ct auto-stamped.
--
--   -- Attempt second active — should violate partial unique index:
--   INSERT INTO mct_overrides (reason) VALUES ('another one that is also more than forty chars long here');
--   -- expect duplicate key error.
--
--   -- Clear the first:
--   UPDATE mct_overrides SET cleared_at = now(), cleared_by = 'user'
--    WHERE cleared_at IS NULL;
--   -- expect: cleared_date_ct auto-stamped.
-- ============================================================================
