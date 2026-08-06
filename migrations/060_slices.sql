-- ============================================================================
-- Migration 060: slices + slice_holdings — M1-Finance-style thematic buckets
-- ============================================================================
-- User-defined thematic pies per portfolio. Distinct from sector/theme
-- (ticker_taxonomy, migration 056) which is a fixed GICS-style taxonomy;
-- slices are opinionated allocation buckets with target percentages and
-- optional nesting (e.g. "AI Five Layers" -> "Chips" -> {AMD, MU, SNDK}).
--
-- Implicit root: `parent_id IS NULL` means "top-level slice in this
-- portfolio." No dedicated root row per portfolio — the portfolios table
-- already names it.
--
-- Two invariants enforced in the API layer (not as CHECKs, so bulk-load
-- + re-parent flows aren't fighting SQL-side constraints):
--   1. Leaf-only holdings — a slice with any slice_holdings row cannot
--      also have child slices, and vice versa.
--   2. Sums of target_pct across children of the same parent should be
--      100 — surfaced as a warning banner, not blocked (users configure
--      iteratively).
--
-- Strict-mode coverage: every OPEN-position ticker in a portfolio SHOULD
-- have a slice_holdings row. Missing tickers surface as an "unassigned"
-- banner in the frontend; not blocked at write time (you buy first, you
-- categorize when you get to the desk).
--
-- slice_holdings.portfolio_id is DENORMALIZED from the parent slice's
-- portfolio_id. Needed for the `UNIQUE(user_id, portfolio_id, ticker)`
-- constraint that guarantees a ticker belongs to at most one leaf slice
-- per portfolio. A trigger keeps it consistent with the parent slice.
--
-- Tenant isolation follows migration 003 (user_id DEFAULT + RLS FORCE +
-- tenant_isolation policy). RLS applies to both tables.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

CREATE TABLE IF NOT EXISTS slices (
    id              SERIAL PRIMARY KEY,
    user_id         UUID NOT NULL DEFAULT (
        COALESCE(
            NULLIF(current_setting('app.user_id', true), '')::uuid,
            'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
        )
    ),
    portfolio_id    INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    parent_id       INTEGER REFERENCES slices(id) ON DELETE CASCADE,
    name            TEXT NOT NULL CHECK (length(trim(name)) > 0),
    target_pct      NUMERIC(6, 3) NOT NULL DEFAULT 0
                    CHECK (target_pct >= 0 AND target_pct <= 100),
    sort_order      INTEGER NOT NULL DEFAULT 0,
    color           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, portfolio_id, parent_id, name)
);

CREATE INDEX IF NOT EXISTS idx_slices_user_portfolio
    ON slices (user_id, portfolio_id);
CREATE INDEX IF NOT EXISTS idx_slices_parent
    ON slices (parent_id);

COMMENT ON TABLE slices IS
    'User-defined allocation buckets per portfolio, M1-Finance-style. '
    'Nested via parent_id (NULL = implicit root). One row per bucket. '
    'target_pct is % of parent (root slices: % of portfolio).';


CREATE TABLE IF NOT EXISTS slice_holdings (
    id              SERIAL PRIMARY KEY,
    user_id         UUID NOT NULL DEFAULT (
        COALESCE(
            NULLIF(current_setting('app.user_id', true), '')::uuid,
            'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
        )
    ),
    portfolio_id    INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    slice_id        INTEGER NOT NULL REFERENCES slices(id) ON DELETE RESTRICT,
    ticker          TEXT NOT NULL CHECK (length(trim(ticker)) > 0),
    target_pct      NUMERIC(6, 3) NOT NULL DEFAULT 0
                    CHECK (target_pct >= 0 AND target_pct <= 100),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, portfolio_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_slice_holdings_slice
    ON slice_holdings (slice_id);
CREATE INDEX IF NOT EXISTS idx_slice_holdings_user_portfolio
    ON slice_holdings (user_id, portfolio_id);

COMMENT ON TABLE slice_holdings IS
    'Ticker-to-leaf-slice assignment. UNIQUE(user_id, portfolio_id, ticker) '
    'enforces one leaf per ticker per portfolio. portfolio_id is denormalized '
    'from the parent slice; a trigger keeps them in sync.';


-- Keep slice_holdings.portfolio_id aligned with its parent slice.
-- Runs on INSERT and on UPDATE of slice_id — the source of truth for
-- portfolio membership is the slice, not the holding.
CREATE OR REPLACE FUNCTION _slice_holdings_sync_portfolio()
RETURNS TRIGGER AS $$
BEGIN
    SELECT portfolio_id INTO NEW.portfolio_id
      FROM slices WHERE id = NEW.slice_id;
    IF NEW.portfolio_id IS NULL THEN
        RAISE EXCEPTION 'slice_holdings.slice_id % has no valid slice row', NEW.slice_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_slice_holdings_sync_portfolio ON slice_holdings;
CREATE TRIGGER trg_slice_holdings_sync_portfolio
    BEFORE INSERT OR UPDATE OF slice_id ON slice_holdings
    FOR EACH ROW EXECUTE FUNCTION _slice_holdings_sync_portfolio();


-- updated_at maintenance triggers.
CREATE OR REPLACE FUNCTION _slices_touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_slices_touch_updated_at ON slices;
CREATE TRIGGER trg_slices_touch_updated_at
    BEFORE UPDATE ON slices
    FOR EACH ROW EXECUTE FUNCTION _slices_touch_updated_at();

DROP TRIGGER IF EXISTS trg_slice_holdings_touch_updated_at ON slice_holdings;
CREATE TRIGGER trg_slice_holdings_touch_updated_at
    BEFORE UPDATE ON slice_holdings
    FOR EACH ROW EXECUTE FUNCTION _slices_touch_updated_at();


-- Row-level security — same pattern as migration 003.
ALTER TABLE slices ENABLE ROW LEVEL SECURITY;
ALTER TABLE slices FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON slices;
CREATE POLICY tenant_isolation ON slices FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);

ALTER TABLE slice_holdings ENABLE ROW LEVEL SECURITY;
ALTER TABLE slice_holdings FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON slice_holdings;
CREATE POLICY tenant_isolation ON slice_holdings FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
--   \d slices
--   \d slice_holdings
--
--   -- confirm RLS is on:
--   SELECT relname, relrowsecurity, relforcerowsecurity
--     FROM pg_class WHERE relname IN ('slices', 'slice_holdings');
--   -- both rows should have relrowsecurity = t, relforcerowsecurity = t
--
--   -- confirm the sync trigger fires:
--   -- (in a session with app.user_id set)
--   INSERT INTO slices (portfolio_id, name, target_pct)
--     VALUES ((SELECT id FROM portfolios WHERE name = 'Long-Term Growth'),
--             'Chips', 20) RETURNING id;
--   INSERT INTO slice_holdings (slice_id, ticker, target_pct)
--     VALUES (<id above>, 'AMD', 25);
--   -- slice_holdings.portfolio_id should equal the LTG portfolio id.
