-- ============================================================================
-- Migration 017: lot_closures — persist LIFO BUY × SELL pairings
-- ============================================================================
-- Today the LIFO accounting pass in api/main.py runs entirely in memory:
-- _recompute_summary_lifo() walks the trades_details rows for a campaign,
-- pairs each SELL against the most-recent open BUY lots, computes Realized_PL,
-- and writes one aggregated number to trades_summary.realized_pl. The
-- intermediate per-pair output (which BUY lot closed which SELL, how many
-- shares, at what prices) is recomputed from scratch every time the API or
-- frontend wants to display closure detail — and the same walk lives twice
-- (Python recompute, TypeScript trade-journal.tsx fallback in commit a796e6d).
--
-- Persisting the pairings gives us:
--   1. A single source of truth — drop the frontend LIFO walk entirely.
--   2. Per-closure forensics (which entry produced which exit P&L) without
--      re-walking the whole transaction list on every render.
--   3. Foundation for future analytics (holding-period histograms, exit-rule
--      attribution, partial-close tracking).
--
-- Step 4 will wire the writes (recompute deletes + reinserts these rows
-- atomically). This step only adds the table.
--
-- Shape mirrors the plan-sketch fields, with three corrections to match the
-- actual schema:
--   - trade_id is VARCHAR(50), not UUID (matches trades_summary/trades_details)
--   - sell_trx_id / buy_trx_id are VARCHAR(50), not TEXT (matches trades_details.trx_id)
--   - Numeric precisions match the source columns (shares 12,4 / price 12,4 /
--     pl 15,2 / multiplier 8,2)
--
-- Plus tenant-scoping the sketch omitted but the codebase requires:
--   - user_id  for RLS (cross-tenant isolation; see migration 003)
--   - portfolio_id for the standard per-portfolio scoping every domain table has
--
-- No FK to trades_summary on trade_id — matches the pattern in audit_trail,
-- trade_images, and trade_lessons. Deletion semantics (what happens to
-- closure rows when a SELL or whole trade is deleted) will be settled in
-- step 4 once the writes are wired up.
--
-- UNIQUE (portfolio_id, trade_id, sell_trx_id, buy_trx_id):
--   Recompute is delete-then-insert, so the constraint won't trip in normal
--   operation. If a recompute bug ever skips the delete or two recomputes
--   race, the constraint guarantees we fail loudly instead of silently
--   accumulating duplicate closure rows. portfolio_id is included because
--   trade_id is unique-per-portfolio, not globally — same scoping as
--   unique_trade_per_portfolio on trades_summary.
-- ============================================================================

CREATE TABLE IF NOT EXISTS lot_closures (
    id              SERIAL         PRIMARY KEY,
    user_id         UUID           NOT NULL REFERENCES users(id) ON DELETE RESTRICT
                                   DEFAULT NULLIF(current_setting('app.user_id', true), '')::uuid,
    portfolio_id    INTEGER        NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    trade_id        VARCHAR(50)    NOT NULL,
    sell_trx_id     VARCHAR(50)    NOT NULL,
    buy_trx_id      VARCHAR(50)    NOT NULL,
    shares          NUMERIC(12, 4) NOT NULL,
    buy_price       NUMERIC(12, 4) NOT NULL,
    sell_price      NUMERIC(12, 4) NOT NULL,
    multiplier      NUMERIC(8, 2)  NOT NULL DEFAULT 1,
    realized_pl     NUMERIC(15, 2) NOT NULL,
    closed_at       TIMESTAMP      NOT NULL,
    created_at      TIMESTAMPTZ    NOT NULL DEFAULT now(),

    CONSTRAINT unique_lot_closure UNIQUE (portfolio_id, trade_id, sell_trx_id, buy_trx_id)
);

CREATE INDEX IF NOT EXISTS idx_lot_closures_trade        ON lot_closures (portfolio_id, trade_id);
CREATE INDEX IF NOT EXISTS idx_lot_closures_sell_trx_id  ON lot_closures (sell_trx_id);
CREATE INDEX IF NOT EXISTS idx_lot_closures_buy_trx_id   ON lot_closures (buy_trx_id);
CREATE INDEX IF NOT EXISTS idx_lot_closures_user         ON lot_closures (user_id);

-- RLS: same pattern as migration 003 / 009 for every other tenant table.
ALTER TABLE lot_closures ENABLE ROW LEVEL SECURITY;
ALTER TABLE lot_closures FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON lot_closures;
CREATE POLICY tenant_isolation ON lot_closures FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
-- Expect: empty (no writes wired yet — that's step 4).
--   SELECT count(*) FROM lot_closures;
--
-- Expect: indexes present.
--   SELECT indexname FROM pg_indexes WHERE tablename = 'lot_closures' ORDER BY indexname;
--
-- Expect: RLS enabled with tenant_isolation policy.
--   SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = 'lot_closures';
--   SELECT polname FROM pg_policy WHERE polrelid = 'lot_closures'::regclass;
--
-- Expect: unique constraint present.
--   SELECT conname FROM pg_constraint WHERE conrelid = 'lot_closures'::regclass AND contype = 'u';
