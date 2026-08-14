-- ============================================================================
-- Migration 068: cycle_references — per-cycle NLV anchor for L1
-- ============================================================================
-- Replaces the ATH-anchored L1 (-7.5% from all-time HWM) with a cycle-anchored
-- L1 (-7.5% from cycle_reference_nlv). The trend cycle is defined by IXIC
-- 21 EMA: 3 consecutive closes above the 21 EMA with the 3rd an up day flips
-- the cycle POSITIVE. On the flip date, cycle_reference_nlv is initialized to
-- that day's closing NLV. Inside a positive cycle the reference RATCHETS
-- (cummax over subsequent end_nlv values); it never decreases. When the trend
-- flips NEGATIVE (trend_count < 0), the reference FREEZES at the last high of
-- the prior positive leg and stays frozen for the duration of the negative
-- leg. A new positive flip starts a new row.
--
-- Why a table instead of computing on-demand from journal_history + trend_count:
-- the user wants to test the efficacy of the cycle-anchored reference vs
-- alternate anchoring schemes later. Persisting the actual ratchet path
-- (flip_date + initial_nlv + ratcheted_nlv + freeze state) makes those
-- retrospective studies deterministic — the compute-on-demand alternative
-- would silently drift if trend-count detection rules change.
--
-- Seed policy: this migration seeds ONE row for the current active cycle
-- (CanSlim, flip 2026-08-07, cycle #22) using real trading_journal values.
-- No backfill of prior cycles — those references are not needed today and
-- can be reconstructed offline if a later efficacy study needs them.
--
-- Automatic maintenance lives in api/main.py: journal_edit and
-- journal_batch_edit both call _ratchet_cycle_reference after a successful
-- save. That helper (a) creates a new row on a detected positive-trend flip,
-- (b) ratchets ratcheted_nlv on a new end_nlv high inside the active cycle,
-- (c) freezes the active row on a first-negative-trend day.
--
-- Tenant isolation follows migration 003 (user_id DEFAULT + RLS FORCE +
-- tenant_isolation policy). One row per (user, portfolio, flip_date).
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

CREATE TABLE IF NOT EXISTS cycle_references (
    id                  SERIAL PRIMARY KEY,
    user_id             UUID NOT NULL DEFAULT (
        COALESCE(
            NULLIF(current_setting('app.user_id', true), '')::uuid,
            'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
        )
    ),
    portfolio_id        INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    flip_date           DATE NOT NULL,
    initial_nlv         NUMERIC(14, 2) NOT NULL CHECK (initial_nlv > 0),
    ratcheted_nlv       NUMERIC(14, 2) NOT NULL CHECK (ratcheted_nlv > 0),
    ratcheted_on_date   DATE NOT NULL,
    is_frozen           BOOLEAN NOT NULL DEFAULT FALSE,
    frozen_at_date      DATE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, portfolio_id, flip_date),
    CHECK (ratcheted_nlv >= initial_nlv),
    CHECK ((is_frozen = FALSE AND frozen_at_date IS NULL)
        OR (is_frozen = TRUE  AND frozen_at_date IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_cycle_references_active
    ON cycle_references (user_id, portfolio_id, flip_date DESC)
    WHERE is_frozen = FALSE;

CREATE INDEX IF NOT EXISTS idx_cycle_references_lookup
    ON cycle_references (user_id, portfolio_id, flip_date DESC);

COMMENT ON TABLE cycle_references IS
    'Per-trend-cycle NLV anchor for the L1 exposure level. One row per '
    'positive-trend-cycle flip (per user, per portfolio). ratcheted_nlv '
    'is a cummax of end_nlv values from flip_date forward; freezes when '
    'the trend flips negative.';

COMMENT ON COLUMN cycle_references.flip_date IS
    'The IXIC 21EMA trend-cycle flip date (3 consecutive closes above 21 '
    'EMA, 3rd an up day). Global signal, but the reference NLV is per-portfolio.';
COMMENT ON COLUMN cycle_references.initial_nlv IS
    'The portfolio''s end_nlv on flip_date. Frozen once written.';
COMMENT ON COLUMN cycle_references.ratcheted_nlv IS
    'max(end_nlv) over the positive cycle. L1 = ratcheted_nlv × 0.925.';
COMMENT ON COLUMN cycle_references.is_frozen IS
    'TRUE when the positive cycle has ended (trend flipped negative). '
    'Frozen rows are never modified; a new positive flip creates a new row.';


-- updated_at maintenance.
CREATE OR REPLACE FUNCTION _cycle_references_touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_cycle_references_touch_updated_at ON cycle_references;
CREATE TRIGGER trg_cycle_references_touch_updated_at
    BEFORE UPDATE ON cycle_references
    FOR EACH ROW EXECUTE FUNCTION _cycle_references_touch_updated_at();


-- Row-level security — same pattern as migration 003.
ALTER TABLE cycle_references ENABLE ROW LEVEL SECURITY;
ALTER TABLE cycle_references FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON cycle_references;
CREATE POLICY tenant_isolation ON cycle_references FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ============================================================================
-- Seed: the current active cycle (CanSlim cycle #22, flip 2026-08-07).
--
-- Data-driven — reads trading_journal for the founder UUID's CanSlim rows
-- from flip_date forward. Values as of migration authoring: initial_nlv
-- ~$589,400, ratcheted ~$605,337, ratcheted_on ~2026-08-13. The seed adapts
-- to whatever the latest saved end_nlv actually is at migration run time.
--
-- Idempotent — the WHERE NOT EXISTS clause skips the insert if the row is
-- already present. The migration runner SETs app.user_id to the founder UUID
-- per its defense-in-depth policy, so RLS still lets this insert through.
--
-- No-op cases:
--   * portfolio CanSlim missing (fresh DB) — SELECT returns no row; skipped.
--   * no journal rows on/after flip_date — subquery returns NULL; skipped.
--   * row already exists — WHERE NOT EXISTS skips.
-- ============================================================================

DO $$
DECLARE
    v_user_id UUID := 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid;
    v_portfolio_id INTEGER;
    v_flip_date DATE := '2026-08-07';
    v_initial_nlv NUMERIC(14, 2);
    v_ratcheted_nlv NUMERIC(14, 2);
    v_ratcheted_on_date DATE;
BEGIN
    SELECT id INTO v_portfolio_id
      FROM portfolios
     WHERE name = 'CanSlim' AND user_id = v_user_id
     LIMIT 1;

    IF v_portfolio_id IS NULL THEN
        RAISE NOTICE 'Migration 068: no CanSlim portfolio for founder user, skipping seed';
        RETURN;
    END IF;

    -- Initial NLV: the flip-day's end_nlv (or the earliest available on/after
    -- flip_date if the flip-day row is missing for any reason).
    SELECT end_nlv INTO v_initial_nlv
      FROM trading_journal
     WHERE user_id = v_user_id
       AND portfolio_id = v_portfolio_id
       AND day >= v_flip_date
       AND end_nlv IS NOT NULL
       AND end_nlv > 0
       AND deleted_at IS NULL
     ORDER BY day ASC
     LIMIT 1;

    IF v_initial_nlv IS NULL THEN
        RAISE NOTICE 'Migration 068: no CanSlim end_nlv on/after %, skipping seed', v_flip_date;
        RETURN;
    END IF;

    -- Ratcheted NLV: cummax over the cycle so far.
    SELECT end_nlv, day
      INTO v_ratcheted_nlv, v_ratcheted_on_date
      FROM trading_journal
     WHERE user_id = v_user_id
       AND portfolio_id = v_portfolio_id
       AND day >= v_flip_date
       AND end_nlv IS NOT NULL
       AND end_nlv > 0
       AND deleted_at IS NULL
     ORDER BY end_nlv DESC, day ASC
     LIMIT 1;

    INSERT INTO cycle_references (
        user_id, portfolio_id, flip_date,
        initial_nlv, ratcheted_nlv, ratcheted_on_date
    ) VALUES (
        v_user_id, v_portfolio_id, v_flip_date,
        v_initial_nlv, v_ratcheted_nlv, v_ratcheted_on_date
    )
    ON CONFLICT (user_id, portfolio_id, flip_date) DO NOTHING;

    RAISE NOTICE 'Migration 068: seeded CanSlim cycle_reference flip=% initial=% ratcheted=% on=%',
                 v_flip_date, v_initial_nlv, v_ratcheted_nlv, v_ratcheted_on_date;
END $$;


-- ============================================================================
-- Verification (manual, post-COMMIT)
-- ============================================================================
--   \d cycle_references
--   SELECT relname, relrowsecurity, relforcerowsecurity
--     FROM pg_class WHERE relname = 'cycle_references';
--   -- relrowsecurity = t, relforcerowsecurity = t
--
--   -- confirm seed (founder session):
--   SELECT flip_date, initial_nlv, ratcheted_nlv, ratcheted_on_date, is_frozen
--     FROM cycle_references
--    WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = 'CanSlim')
--    ORDER BY flip_date DESC;
