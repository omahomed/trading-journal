-- 010: MCT V11 — Data Foundation (Phase 1)
-- ============================================================================
-- Establishes the persistent data layer for the new Market Cycle Tracker (V11).
-- Adds market_data (canonical NDX/SPY OHLC + indicators) and replaces the
-- orphaned V9 market_signals table with V11's signal vocabulary.
--
-- Both tables are app-wide (no user_id, no RLS). The runtime drops to
-- app_runtime after authentication, so SELECT is granted explicitly to that
-- role. INSERT/UPDATE go through the migration owner (backfill, daily
-- updater, and the Phase 2 engine writer all run with elevated privileges).
--
-- Existing market_signals contents are discarded. The V9 schema was orphaned
-- at the API level (no readers) and the legacy writers in sync_market_data.py
-- and bootstrap_fresh_start.py have been marked deprecated. Phase 2 ships
-- the V11 writer that targets the new schema below.

-- ----------------------------------------------------------------------------
-- market_data: one row per (symbol, trade_date) with OHLC + computed indicators
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS market_data (
    id          BIGSERIAL PRIMARY KEY,
    symbol      TEXT NOT NULL,
    trade_date  DATE NOT NULL,
    open        NUMERIC(12, 4) NOT NULL,
    high        NUMERIC(12, 4) NOT NULL,
    low         NUMERIC(12, 4) NOT NULL,
    close       NUMERIC(12, 4) NOT NULL,
    volume      BIGINT,
    ema_8       NUMERIC(12, 4),
    ema_21      NUMERIC(12, 4),
    sma_50      NUMERIC(12, 4),
    sma_200     NUMERIC(12, 4),
    source      TEXT NOT NULL DEFAULT 'yfinance',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT market_data_symbol_date_unique UNIQUE (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date
    ON market_data (symbol, trade_date DESC);

GRANT SELECT ON market_data TO app_runtime;
GRANT USAGE, SELECT ON SEQUENCE market_data_id_seq TO app_runtime;

-- ----------------------------------------------------------------------------
-- market_signals: V11 signal log (drop & recreate; old schema was orphaned)
-- ----------------------------------------------------------------------------
-- signal_type is enforced by api.market_signals_vocab.ALLOWED_SIGNAL_TYPES at
-- the application layer rather than via a CHECK constraint, so the vocabulary
-- can evolve during Phase 2 without DROP/ADD CONSTRAINT churn.

DROP TABLE IF EXISTS market_signals;

CREATE TABLE market_signals (
    id              BIGSERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL,
    signal_type     TEXT NOT NULL,
    signal_label    TEXT NOT NULL,
    exposure_before INTEGER,
    exposure_after  INTEGER,
    state_before    TEXT,
    state_after     TEXT,
    meta            JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_market_signals_date ON market_signals (trade_date DESC);
CREATE INDEX idx_market_signals_type ON market_signals (signal_type);

GRANT SELECT ON market_signals TO app_runtime;
GRANT USAGE, SELECT ON SEQUENCE market_signals_id_seq TO app_runtime;
