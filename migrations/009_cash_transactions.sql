-- ============================================================================
-- Migration 009: cash_transactions — ledger for derived NLV
-- ============================================================================
-- Today NLV is user-entered once per day via the Daily Routine. That
-- conflates "what I'm journaling" with "what the system knows about my
-- money." This migration lays the foundation for system-derived NLV:
--   NLV = cash_balance + Σ(open_position.shares × live_price)
-- where cash_balance is computed from the signed sum of cash_transactions
-- rows for the portfolio.
--
-- Design: one append-only ledger. Every trade emits a row (-shares*price for
-- BUY, +shares*price for SELL). Deposits/withdrawals are first-class rows
-- (source='deposit'/'withdraw'). A periodic "reconcile" row absorbs the drift
-- between system cash and the user's actual broker balance — this is the
-- escape hatch for the "journal, not ledger" philosophy: we don't track fees
-- or margin interest per-trade, users reconcile once a week/month instead.
--
-- Shape:
--   - amount is signed (+ for money in, - for money out)
--   - source constrained to a small vocabulary via CHECK
--   - trade_detail_id is nullable so deposit/withdraw/reconcile rows aren't
--     forced to reference a trade; SET NULL on trade delete so cash history
--     survives even if the originating trade is purged
--
-- RLS: tenant_isolation policy mirroring every other per-user table so
-- cross-tenant reads return zero rows, INSERTs auto-tag user_id from the
-- session variable.
--
-- Backfill:
--   - One 'deposit' row per portfolio that has starting_capital set
--   - One row per existing trades_details BUY/SELL so cash history extends
--     back through the founder's entire trading record from day one
-- ============================================================================

CREATE TABLE IF NOT EXISTS cash_transactions (
    id              SERIAL       PRIMARY KEY,
    user_id         UUID         NOT NULL REFERENCES users(id) ON DELETE RESTRICT
                                 DEFAULT NULLIF(current_setting('app.user_id', true), '')::uuid,
    portfolio_id    INTEGER      NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    date            TIMESTAMPTZ  NOT NULL,
    amount          NUMERIC(15, 2) NOT NULL,  -- signed: + in, - out
    source          VARCHAR(30)  NOT NULL,
    trade_detail_id INTEGER      REFERENCES trades_details(id) ON DELETE SET NULL,
    note            TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT cash_source_vocab CHECK (source IN (
        'deposit', 'withdraw', 'buy', 'sell', 'reconcile'
    ))
);

CREATE INDEX IF NOT EXISTS idx_cash_tx_portfolio_date ON cash_transactions (portfolio_id, date);
CREATE INDEX IF NOT EXISTS idx_cash_tx_user           ON cash_transactions (user_id);
CREATE INDEX IF NOT EXISTS idx_cash_tx_trade_detail   ON cash_transactions (trade_detail_id)
    WHERE trade_detail_id IS NOT NULL;

-- RLS: same pattern as migration 003 for every other tenant table.
ALTER TABLE cash_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE cash_transactions FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON cash_transactions;
CREATE POLICY tenant_isolation ON cash_transactions FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ============================================================================
-- Backfill — runs as neondb_owner (BYPASSRLS), so we must pass user_id
-- explicitly on INSERT instead of letting the column default fire (the
-- default reads app.user_id which isn't set during migrations).
-- ============================================================================

-- 1. Initial-capital deposits. Only for portfolios that already have a
--    starting_capital value set. Dates use reset_date when present, else
--    falls back to portfolio created_at.
INSERT INTO cash_transactions (user_id, portfolio_id, date, amount, source, note)
SELECT
    p.user_id,
    p.id,
    COALESCE(p.reset_date::timestamptz, p.created_at),
    p.starting_capital,
    'deposit',
    'Initial capital (backfilled from portfolio settings)'
FROM portfolios p
WHERE p.starting_capital IS NOT NULL
  AND p.starting_capital > 0
  AND NOT EXISTS (
      -- Idempotency guard: don't double-seed if migration re-runs
      SELECT 1 FROM cash_transactions c
      WHERE c.portfolio_id = p.id
        AND c.source       = 'deposit'
        AND c.note LIKE 'Initial capital%'
  );

-- 2. BUY/SELL cash flows from every existing trade detail row.
--    Negative for buys (cash out), positive for sells (cash in).
INSERT INTO cash_transactions (user_id, portfolio_id, date, amount, source, trade_detail_id, note)
SELECT
    td.user_id,
    td.portfolio_id,
    td.date,
    CASE UPPER(td.action)
        WHEN 'BUY'  THEN -(td.shares * td.amount)
        WHEN 'SELL' THEN  (td.shares * td.amount)
    END,
    LOWER(td.action),
    td.id,
    'Backfilled from trades_details'
FROM trades_details td
WHERE UPPER(td.action) IN ('BUY', 'SELL')
  AND NOT EXISTS (
      -- Idempotency guard: don't duplicate on re-run
      SELECT 1 FROM cash_transactions c
      WHERE c.trade_detail_id = td.id
  );


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
-- Expect: one row per existing BUY/SELL + one per portfolio with starting_capital.
--   SELECT source, count(*) FROM cash_transactions GROUP BY source ORDER BY source;
--
-- Expect: cash_balance matches intuition for each portfolio (may be negative
-- for portfolios without starting_capital set — that's expected; user sets
-- capital via Settings and a deposit row materializes via the API).
--   SELECT portfolio_id, SUM(amount) AS cash_balance
--   FROM cash_transactions GROUP BY portfolio_id ORDER BY portfolio_id;
