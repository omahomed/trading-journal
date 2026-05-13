-- ============================================================================
-- Migration 025: weekly_retros + weekly_retro_ticker_grades — Phase 0
-- ============================================================================
-- Move weekly retros out of localStorage into Postgres so subsequent phases
-- (tags, snapshots, cross-entity search, the shared notes rail) have a real
-- server-side entity to attach to. Fresh start — no backfill (user confirmed
-- no localStorage data needs preserving).
--
-- Parent + child pattern mirroring trades_summary/trades_details. Children
-- are delete-then-insert on every parent save, the same idiom lot_closures
-- (migration 017) established: the helper deletes all child rows for the
-- given parent and re-inserts the current set, so the wire shape stays a
-- simple {ticker: {grade, behavior, notes}} dict instead of asking the
-- frontend to diff additions vs removals.
--
-- New pattern note: this is the first table in the codebase that combines
-- soft-delete (deleted_at) with a partial unique index restricted to live
-- rows (WHERE deleted_at IS NULL). The existing soft-deleted tables
-- (trades_summary, trades_details, trading_journal) use full unique
-- constraints because their natural keys are never recycled. Weekly retros
-- need recycling support: a user can soft-delete a week's retro and later
-- create a fresh one for the same Monday. The partial-unique index allows
-- that, and the upsert helper additionally REVIVES soft-deleted rows on
-- conflict so any tags attached to the original row keep their FK target.
--
-- No audit trigger — matches migrations 009 (cash_transactions) and 017
-- (lot_closures), which also skip audit on the grounds that the audit_trail
-- table is trade-centric.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT
-- statements here.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. weekly_retros — one row per (portfolio, Monday)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weekly_retros (
    id                  SERIAL          PRIMARY KEY,
    user_id             UUID            NOT NULL REFERENCES users(id) ON DELETE RESTRICT
                                        DEFAULT NULLIF(current_setting('app.user_id', true), '')::uuid,
    portfolio_id        INTEGER         NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    week_start          DATE            NOT NULL,
    week_grade          VARCHAR(3),
    best_decision       TEXT            NOT NULL DEFAULT '',
    worst_decision      TEXT            NOT NULL DEFAULT '',
    rule_change         BOOLEAN         NOT NULL DEFAULT FALSE,
    rule_change_text    TEXT            NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    deleted_at          TIMESTAMPTZ,

    CONSTRAINT weekly_retros_week_grade_vocab CHECK (
        week_grade IS NULL OR week_grade IN
        ('A+','A','A-','B+','B','B-','C+','C','C-','D','F')
    ),
    CONSTRAINT weekly_retros_week_start_monday CHECK (
        EXTRACT(ISODOW FROM week_start) = 1
    )
);

-- Partial unique on live rows only. A soft-deleted retro coexists with a
-- new live retro for the same week (the upsert helper prefers to revive the
-- soft-deleted row, but this index doesn't enforce that — it's the safety
-- net if a future code path inserts directly).
CREATE UNIQUE INDEX IF NOT EXISTS uq_weekly_retros_portfolio_week_live
    ON weekly_retros (portfolio_id, week_start)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_weekly_retros_user
    ON weekly_retros (user_id);
CREATE INDEX IF NOT EXISTS idx_weekly_retros_portfolio_week
    ON weekly_retros (portfolio_id, week_start DESC) WHERE deleted_at IS NULL;

ALTER TABLE weekly_retros ENABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_retros FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON weekly_retros;
CREATE POLICY tenant_isolation ON weekly_retros FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ---------------------------------------------------------------------------
-- 2. weekly_retro_ticker_grades — one row per (retro, ticker)
-- ---------------------------------------------------------------------------
-- Hard delete on retro cascade (CASCADE), not soft. The replace-all helper
-- DELETEs + INSERTs every save, so soft-delete on this table would just
-- accumulate tombstones. If a retro itself is soft-deleted, its child rows
-- are left intact so a revival restores the full picture.
CREATE TABLE IF NOT EXISTS weekly_retro_ticker_grades (
    id                  SERIAL          PRIMARY KEY,
    user_id             UUID            NOT NULL REFERENCES users(id) ON DELETE RESTRICT
                                        DEFAULT NULLIF(current_setting('app.user_id', true), '')::uuid,
    weekly_retro_id     INTEGER         NOT NULL REFERENCES weekly_retros(id) ON DELETE CASCADE,
    ticker              VARCHAR(20)     NOT NULL,
    grade               VARCHAR(20),
    behavior            VARCHAR(40),
    notes               TEXT            NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_retro_ticker UNIQUE (weekly_retro_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_retro_grades_retro
    ON weekly_retro_ticker_grades (weekly_retro_id);
CREATE INDEX IF NOT EXISTS idx_retro_grades_user
    ON weekly_retro_ticker_grades (user_id);
CREATE INDEX IF NOT EXISTS idx_retro_grades_ticker
    ON weekly_retro_ticker_grades (ticker);

ALTER TABLE weekly_retro_ticker_grades ENABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_retro_ticker_grades FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON weekly_retro_ticker_grades;
CREATE POLICY tenant_isolation ON weekly_retro_ticker_grades FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
-- Expect: empty (no writes wired yet — frontend swap lands in the same PR).
--   SELECT count(*) FROM weekly_retros;
--   SELECT count(*) FROM weekly_retro_ticker_grades;
--
-- Expect: RLS enabled with tenant_isolation policy on both.
--   SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class
--     WHERE relname IN ('weekly_retros','weekly_retro_ticker_grades');
--
-- Expect: Monday CHECK rejects a non-Monday insert.
--   INSERT INTO weekly_retros (portfolio_id, week_start) VALUES (1, '2026-05-13');
--   → ERROR:  new row for relation "weekly_retros" violates check constraint
--             "weekly_retros_week_start_monday"
