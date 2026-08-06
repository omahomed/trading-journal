-- ============================================================================
-- Migration 061: portfolios.slices_enabled — per-portfolio Slices opt-out
-- ============================================================================
-- Every portfolio defaults to Slices ON. The user can flip it OFF per
-- portfolio (e.g. CanSlim runs on a campaign-by-campaign basis, not a
-- pie-based allocation model, so the Slices page + strict-mode banners
-- are noise there).
--
-- What flips when this is FALSE:
--   * `/api/slices?portfolio=X` returns `{disabled: true, portfolio, portfolio_id}`
--     instead of a normal payload — the frontend renders a
--     "Slices disabled for this portfolio · [Enable]" empty state
--   * The Slices page's Manage button + edit paths are hidden
--   * Any future slice-consumer we add (an ACS badge, a nudge on Log
--     Buy, etc.) will filter on this flag
--
-- Default TRUE — turning slices on for existing portfolios is the
-- expected behavior; the toggle is for explicit opt-out. Backfilling
-- FALSE would be a policy call the user hasn't made.
--
-- NOT NULL DEFAULT keeps the column ergonomic — every existing row
-- gets TRUE without needing a follow-up UPDATE, and callers never have
-- to handle NULL semantics.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

ALTER TABLE portfolios
    ADD COLUMN IF NOT EXISTS slices_enabled BOOLEAN NOT NULL DEFAULT TRUE;


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
--   SELECT column_name, data_type, column_default
--     FROM information_schema.columns
--    WHERE table_name = 'portfolios' AND column_name = 'slices_enabled';
--
--   SELECT id, name, slices_enabled FROM portfolios ORDER BY id;
--   -- every existing portfolio should show slices_enabled = true.
