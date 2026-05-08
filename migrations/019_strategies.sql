-- ============================================================================
-- Migration 019: Strategy tagging — Phase 1
-- ============================================================================
-- Adds a `strategies` lookup table and a `strategy` column on `trades_summary`
-- so each campaign can be tagged with the strategy it belongs to. Phase 1 only
-- wires the schema, the seed rows, the GET /api/strategies endpoint, and the
-- Log Buy form. Retroactive tagging (Phase 2) and analytics filters (Phase 3)
-- ship in follow-up commits.
--
-- Why text-keyed FK (vs. integer surrogate key, the project's usual style):
--   Every other FK in schema.sql references an `id INTEGER`, but here we
--   reference `strategies(name)` directly. Three reasons make this a deliberate
--   exception — not an oversight:
--     1. The lookup is tiny (3 rows today, ~5 ever) and human-readable.
--        Storing the name in trades_summary saves the join when filtering.
--     2. Phase 2/3 surface filters work on the name string in URL/state, so
--        keeping the column as text removes one mapping layer in the UI.
--     3. ON UPDATE CASCADE handles the only mutation case (renaming a
--        strategy in the future Admin UI) without an integer indirection.
--   ON DELETE RESTRICT prevents accidentally orphaning trades by deleting an
--   in-use strategy — Phase 2's Admin UI will surface this as a friendly
--   "this strategy is in use by N trades" error.
--
-- Migration ordering (critical):
--   1. CREATE TABLE strategies (with seeds)
--   2. INSERT three seed rows
--   3. ALTER trades_summary ADD COLUMN strategy ... DEFAULT 'CanSlim'
--      → all 489 existing rows get tagged in this step
--   4. ALTER trades_summary ADD CONSTRAINT FK strategy → strategies(name)
--      → can only be enforced once every row has a valid value; the FK's
--        existence-check would fail mid-step if added in (3).
--
-- Idempotent: re-running on a populated DB is a no-op (IF NOT EXISTS on the
-- table + column, ON CONFLICT DO NOTHING on the seeds, pg_constraint guard
-- on the FK).
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT
-- statements here.
-- ============================================================================

-- 1. Lookup table
CREATE TABLE IF NOT EXISTS strategies (
    name        TEXT PRIMARY KEY,
    description TEXT,
    color       TEXT NOT NULL,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2. Seed rows. created_at uses the column DEFAULT so the three rows land in
-- INSERT order, which is what we want — CanSlim first, StockTalk second,
-- 21eStrategy third — for the GET /api/strategies "ORDER BY created_at ASC".
-- CanSlim's color (#6366f1) matches the existing dashboard primary accent
-- (--color-g-dash in frontend/src/app/globals.css) so the user's default
-- strategy reuses the visual language they already see throughout the app.
INSERT INTO strategies (name, description, color) VALUES
  ('CanSlim',     'O''Neil''s CanSlim methodology — high RS, strong fundamentals, technical breakouts', '#6366f1'),
  ('StockTalk',   'Small-cap fundamentals-heavy strategy with light technical analysis',                 '#d97706'),
  ('21eStrategy', '21 EMA-based technical strategy',                                                     '#0d9488')
ON CONFLICT (name) DO NOTHING;

-- 3. Strategy column with DEFAULT — auto-fills all existing trades_summary
-- rows (including soft-deleted ones) with 'CanSlim'. NOT NULL is safe to
-- declare in the same statement because the DEFAULT applies before the
-- constraint is checked.
ALTER TABLE trades_summary
  ADD COLUMN IF NOT EXISTS strategy TEXT NOT NULL DEFAULT 'CanSlim';

-- 4. FK constraint. PostgreSQL has no IF NOT EXISTS form for ADD CONSTRAINT,
-- so we guard via pg_constraint to keep the migration idempotent.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'trades_summary_strategy_fkey'
    ) THEN
        ALTER TABLE trades_summary
          ADD CONSTRAINT trades_summary_strategy_fkey
          FOREIGN KEY (strategy) REFERENCES strategies(name)
          ON UPDATE CASCADE ON DELETE RESTRICT;
    END IF;
END $$;

-- ============================================================================
-- Verification: refuse to commit if any active row still has a missing or
-- empty strategy. The DEFAULT in step 3 should make this impossible — this
-- check is belt-and-braces in case a future schema drift sneaks through.
-- ============================================================================
DO $$
DECLARE
    n_untagged INTEGER;
BEGIN
    SELECT COUNT(*) INTO n_untagged
    FROM trades_summary
    WHERE strategy IS NULL OR strategy = '';

    IF n_untagged > 0 THEN
        RAISE EXCEPTION
            'Migration 019: % trades_summary rows have NULL or empty strategy after backfill — aborting',
            n_untagged;
    END IF;
END $$;
