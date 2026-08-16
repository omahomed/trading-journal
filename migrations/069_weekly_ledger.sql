-- ============================================================================
-- Migration 069: weekly_ledger — per-week transaction review
-- ============================================================================
-- Adds a new "Weekly Ledger" page under Daily Workflow. The page lists every
-- BUY + SELL detail row that landed Mon–Fri of a selected week, one row per
-- transaction (adds, partial trims, full closes all included). Distinct from
-- Weekly Retro (prose/reflection, untouched by this migration) and from
-- Campaign Review (aggregates details into one campaign row per trade).
--
-- Two schema changes:
--
--   1. New table `weekly_ledger_notes` — one free-text note per
--      (user, portfolio, week_start). Page-level; NOT per-row. Autosaved
--      from the UI. week_start is stored as a Monday-anchored DATE — the
--      backend normalizes any DATE input to that week's Monday before
--      write, so `week_start` is always a valid ISO Monday.
--
--   2. Extend `tag_assignments.entity_type` CHECK constraint to accept
--      "trades_details". Enables per-row lesson-tag assignment on the
--      Weekly Ledger via the existing TagPicker component — same
--      catalog + autocomplete + colored pills the Journal + Retro use.
--      No new column on trades_details; no new tag catalog. Tag assignments
--      already carry `entity_type` + `entity_id`; the new entity type just
--      needs the constraint's ALLOWED set widened.
--
-- Existing tag catalog stays portfolio-scoped and is shared across all
-- surfaces. A separate migration / bootstrap script seeds the 14
-- LESSON_CATEGORIES ("Followed Rules", "Chased Entry", "Bought Too Early",
-- etc.) as tags so autocomplete has them available from day 1.
--
-- Tenant isolation: RLS + FORCE + tenant_isolation policy on
-- weekly_ledger_notes (same pattern as migration 003 / 060 / 068).
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

-- ── 1. weekly_ledger_notes ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS weekly_ledger_notes (
    id              SERIAL PRIMARY KEY,
    user_id         UUID NOT NULL DEFAULT (
        COALESCE(
            NULLIF(current_setting('app.user_id', true), '')::uuid,
            'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
        )
    ),
    portfolio_id    INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    week_start      DATE NOT NULL,
    note            TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, portfolio_id, week_start)
);

CREATE INDEX IF NOT EXISTS idx_weekly_ledger_notes_lookup
    ON weekly_ledger_notes (user_id, portfolio_id, week_start DESC);

COMMENT ON TABLE weekly_ledger_notes IS
    'Free-text weekly note attached to the Weekly Ledger page. One row per '
    '(user, portfolio, week_start). Autosaved. week_start must be a Monday '
    '(caller-enforced; backend normalizes DATE input to the week''s Monday).';

COMMENT ON COLUMN weekly_ledger_notes.week_start IS
    'Monday of the week the note belongs to (ISO week convention).';


-- updated_at maintenance trigger.
CREATE OR REPLACE FUNCTION _weekly_ledger_notes_touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_weekly_ledger_notes_touch_updated_at
    ON weekly_ledger_notes;
CREATE TRIGGER trg_weekly_ledger_notes_touch_updated_at
    BEFORE UPDATE ON weekly_ledger_notes
    FOR EACH ROW EXECUTE FUNCTION _weekly_ledger_notes_touch_updated_at();


-- RLS — same pattern as migration 003.
ALTER TABLE weekly_ledger_notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_ledger_notes FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON weekly_ledger_notes;
CREATE POLICY tenant_isolation ON weekly_ledger_notes FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ── 2. Extend tag_assignments.entity_type for trades_details ──────────
-- The existing CHECK allows: weekly_retro, daily_journal, trades_summary.
-- Widen to also allow trades_details so the Weekly Ledger can attach
-- lesson tags per transaction row.
ALTER TABLE tag_assignments
    DROP CONSTRAINT IF EXISTS tag_assignments_entity_type_check;

ALTER TABLE tag_assignments
    ADD CONSTRAINT tag_assignments_entity_type_check
    CHECK (entity_type IN (
        'weekly_retro',
        'daily_journal',
        'trades_summary',
        'trades_details'
    ));


-- ============================================================================
-- Verification (manual, post-COMMIT)
-- ============================================================================
--   \d weekly_ledger_notes
--   SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class
--    WHERE relname = 'weekly_ledger_notes';
--
--   -- Confirm the widened CHECK accepts trades_details:
--   SELECT pg_get_constraintdef(oid)
--     FROM pg_constraint
--    WHERE conname = 'tag_assignments_entity_type_check';
