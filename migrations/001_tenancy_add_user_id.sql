-- ============================================================================
-- Migration 001: Add user_id to tenant-scoped tables (Tier 1, step 1)
-- ============================================================================
-- Adds a nullable user_id UUID column to every table that holds per-user data,
-- backfills all existing rows to the founder UUID, and adds a (user_id) index.
--
-- Scope:
--   - 12 tenant-scoped tables get user_id
--   - market_signals is intentionally excluded (shared global market data)
--   - NO foreign key yet — users table will be introduced with next-auth and
--     the FK will be added in a follow-up migration
--   - NOT NULL flip is DEFERRED to migration 003, after backend code has been
--     updated to write user_id on every insert
--
-- Idempotency: uses ADD COLUMN IF NOT EXISTS and WHERE user_id IS NULL on
-- backfills, so the migration can be re-run safely.
--
-- Owner/founder UUID (hardcoded for determinism across environments):
--   d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a
--
-- Transaction handling: the runner (migrations/run.py) wraps the whole file in
-- a transaction alongside the schema_migrations tracking insert, so BEGIN/COMMIT
-- are intentionally omitted here.
-- ============================================================================

-- ---------- portfolios ----------
ALTER TABLE portfolios ADD COLUMN IF NOT EXISTS user_id UUID;
UPDATE portfolios SET user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
  WHERE user_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_portfolios_user ON portfolios (user_id);

-- ---------- trades_summary ----------
ALTER TABLE trades_summary ADD COLUMN IF NOT EXISTS user_id UUID;
UPDATE trades_summary SET user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
  WHERE user_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_trades_summary_user ON trades_summary (user_id);

-- ---------- trades_details ----------
ALTER TABLE trades_details ADD COLUMN IF NOT EXISTS user_id UUID;
UPDATE trades_details SET user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
  WHERE user_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_trades_details_user ON trades_details (user_id);

-- ---------- trading_journal ----------
ALTER TABLE trading_journal ADD COLUMN IF NOT EXISTS user_id UUID;
UPDATE trading_journal SET user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
  WHERE user_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_trading_journal_user ON trading_journal (user_id);

-- ---------- audit_trail ----------
ALTER TABLE audit_trail ADD COLUMN IF NOT EXISTS user_id UUID;
UPDATE audit_trail SET user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
  WHERE user_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_audit_trail_user ON audit_trail (user_id);

-- ---------- trade_images ----------
ALTER TABLE trade_images ADD COLUMN IF NOT EXISTS user_id UUID;
UPDATE trade_images SET user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
  WHERE user_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_trade_images_user ON trade_images (user_id);

-- ---------- trade_fundamentals ----------
ALTER TABLE trade_fundamentals ADD COLUMN IF NOT EXISTS user_id UUID;
UPDATE trade_fundamentals SET user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
  WHERE user_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_trade_fundamentals_user ON trade_fundamentals (user_id);

-- ---------- drawdown_notes ----------
ALTER TABLE drawdown_notes ADD COLUMN IF NOT EXISTS user_id UUID;
UPDATE drawdown_notes SET user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
  WHERE user_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_drawdown_notes_user ON drawdown_notes (user_id);

-- ---------- trade_lessons ----------
ALTER TABLE trade_lessons ADD COLUMN IF NOT EXISTS user_id UUID;
UPDATE trade_lessons SET user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
  WHERE user_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_trade_lessons_user ON trade_lessons (user_id);

-- ---------- rule_notes ----------
ALTER TABLE rule_notes ADD COLUMN IF NOT EXISTS user_id UUID;
UPDATE rule_notes SET user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
  WHERE user_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_rule_notes_user ON rule_notes (user_id);

-- ---------- app_config ----------
ALTER TABLE app_config ADD COLUMN IF NOT EXISTS user_id UUID;
UPDATE app_config SET user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
  WHERE user_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_app_config_user ON app_config (user_id);

-- ---------- dashboard_events ----------
ALTER TABLE dashboard_events ADD COLUMN IF NOT EXISTS user_id UUID;
UPDATE dashboard_events SET user_id = 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
  WHERE user_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_dashboard_events_user ON dashboard_events (user_id);

-- ============================================================================
-- VERIFICATION (run separately after COMMIT)
-- ============================================================================
-- Every row in every tenant-scoped table should have user_id set:
--
--   SELECT 'portfolios'         AS table_name, count(*) FILTER (WHERE user_id IS NULL) AS nulls FROM portfolios
--   UNION ALL SELECT 'trades_summary',     count(*) FILTER (WHERE user_id IS NULL) FROM trades_summary
--   UNION ALL SELECT 'trades_details',     count(*) FILTER (WHERE user_id IS NULL) FROM trades_details
--   UNION ALL SELECT 'trading_journal',    count(*) FILTER (WHERE user_id IS NULL) FROM trading_journal
--   UNION ALL SELECT 'audit_trail',        count(*) FILTER (WHERE user_id IS NULL) FROM audit_trail
--   UNION ALL SELECT 'trade_images',       count(*) FILTER (WHERE user_id IS NULL) FROM trade_images
--   UNION ALL SELECT 'trade_fundamentals', count(*) FILTER (WHERE user_id IS NULL) FROM trade_fundamentals
--   UNION ALL SELECT 'drawdown_notes',     count(*) FILTER (WHERE user_id IS NULL) FROM drawdown_notes
--   UNION ALL SELECT 'trade_lessons',      count(*) FILTER (WHERE user_id IS NULL) FROM trade_lessons
--   UNION ALL SELECT 'rule_notes',         count(*) FILTER (WHERE user_id IS NULL) FROM rule_notes
--   UNION ALL SELECT 'app_config',         count(*) FILTER (WHERE user_id IS NULL) FROM app_config
--   UNION ALL SELECT 'dashboard_events',   count(*) FILTER (WHERE user_id IS NULL) FROM dashboard_events;
--
-- Expected: every `nulls` count = 0.
