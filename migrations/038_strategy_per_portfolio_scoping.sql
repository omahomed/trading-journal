-- migrations/038_strategy_per_portfolio_scoping.sql
--
-- Per-portfolio scoping for strategies. Adds an optional
-- allowed_portfolio_names TEXT[] column to the strategies table:
--   NULL          → strategy is visible in ALL portfolios (default for
--                   universal strategies like 21eStrategy).
--   ARRAY[names]  → strategy is visible ONLY in the listed portfolios.
--
-- Reads via db.load_strategies(portfolio_name=...) compose the filter
-- as: WHERE allowed_portfolio_names IS NULL OR %s = ANY(...).
--
-- Why NAMES (not IDs): migrations are easier to write/read; portfolio
-- names are UNIQUE per user; renames are rare. If we ever rename a
-- portfolio in the future, the strategies array must be updated too
-- (acceptable cost vs. the migration ergonomics win here).
--
-- This migration depends on migration 037 having applied first
-- ('Long-Term Growth' didn't exist before that rename). Numbered
-- ordering enforces it.
--
-- Idempotent:
--   - ADD COLUMN IF NOT EXISTS
--   - UPDATEs are absolute (not delta) so re-running converges
--   - INSERT ... ON CONFLICT DO UPDATE keeps allowed_portfolio_names
--     in sync without touching unrelated columns

ALTER TABLE strategies
  ADD COLUMN IF NOT EXISTS allowed_portfolio_names TEXT[] NULL;

-- Existing strategy scoping
UPDATE strategies
   SET allowed_portfolio_names = ARRAY['CanSlim']
 WHERE name = 'CanSlim';

UPDATE strategies
   SET allowed_portfolio_names = ARRAY['CanSlim', 'Long-Term Growth']
 WHERE name = 'StockTalk';

-- 21eStrategy stays NULL (visible everywhere). Set explicitly in case
-- a prior backfill misset it.
UPDATE strategies
   SET allowed_portfolio_names = NULL
 WHERE name = '21eStrategy';

-- New strategies. Sky-blue + pink chosen to be distinct from the three
-- existing colors (CanSlim #6366f1 indigo, StockTalk #d97706 orange,
-- 21eStrategy #0d9488 teal).
INSERT INTO strategies (name, description, color, is_active, allowed_portfolio_names)
VALUES
  ('LongTerm',    'Long-term position trading',                  '#0284c7', TRUE,
   ARRAY['457B Plan', 'Long-Term Growth']),
  ('50sStrategy', 'Trades anchored to the 50-day SMA',           '#be185d', TRUE,
   ARRAY['457B Plan', 'Long-Term Growth'])
ON CONFLICT (name) DO UPDATE
  SET allowed_portfolio_names = EXCLUDED.allowed_portfolio_names,
      color                   = EXCLUDED.color,
      description             = EXCLUDED.description,
      is_active               = EXCLUDED.is_active;
