-- ============================================
-- TRADING JOURNAL DATABASE SCHEMA
-- PostgreSQL 16+
-- ============================================

-- ============================================
-- TABLE: portfolios
-- PURPOSE: Reference table for portfolio names
-- ============================================
CREATE TABLE IF NOT EXISTS portfolios (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seed data
INSERT INTO portfolios (name) VALUES
    ('CanSlim'),
    ('TQQQ Strategy'),
    ('457B Plan')
ON CONFLICT (name) DO NOTHING;


-- ============================================
-- TABLE: trades_summary
-- PURPOSE: Campaign-level trade summaries (CSV: Trade_Log_Summary.csv)
-- SYNC: Calculated from trades_details via Python LIFO engine
-- ============================================
CREATE TABLE IF NOT EXISTS trades_summary (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    trade_id VARCHAR(50) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'OPEN',  -- 'OPEN' or 'CLOSED'
    open_date TIMESTAMP,
    closed_date TIMESTAMP,
    shares NUMERIC(12, 4) DEFAULT 0,  -- Current remaining shares
    avg_entry NUMERIC(12, 4) DEFAULT 0,
    avg_exit NUMERIC(12, 4) DEFAULT 0,
    total_cost NUMERIC(15, 2) DEFAULT 0,
    realized_pl NUMERIC(15, 2) DEFAULT 0,
    unrealized_pl NUMERIC(15, 2) DEFAULT 0,
    return_pct NUMERIC(10, 4) DEFAULT 0,
    sell_rule VARCHAR(100),
    notes TEXT,
    amount NUMERIC(15, 2) DEFAULT 0,  -- Legacy field
    value NUMERIC(15, 2) DEFAULT 0,   -- Current market value
    stop_loss NUMERIC(12, 4),
    rule VARCHAR(100),  -- Buy rule
    buy_notes TEXT,
    sell_notes TEXT,
    risk_budget NUMERIC(15, 2) DEFAULT 0,
    grade SMALLINT CHECK (grade IS NULL OR (grade BETWEEN 1 AND 5)),
    -- Manual live-price override (Migration 012). NLV + ACS prefer this
    -- over the yfinance result when set; NULL means 'fall through to the
    -- live price provider'. Primarily a workaround for OCC option symbols
    -- yfinance can't resolve.
    manual_price NUMERIC(15, 4),
    manual_price_set_at TIMESTAMPTZ,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_trade_per_portfolio UNIQUE (portfolio_id, trade_id)
);

CREATE INDEX IF NOT EXISTS idx_summary_status ON trades_summary (portfolio_id, status);
CREATE INDEX IF NOT EXISTS idx_summary_ticker ON trades_summary (ticker);
CREATE INDEX IF NOT EXISTS idx_summary_trade_id ON trades_summary (trade_id);


-- ============================================
-- TABLE: trades_details
-- PURPOSE: Transaction-level log (CSV: Trade_Log_Details.csv)
-- AUTHORITY: Source of truth for all trade calculations
-- ============================================
CREATE TABLE IF NOT EXISTS trades_details (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    trade_id VARCHAR(50) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,  -- 'BUY' or 'SELL'
    date TIMESTAMP NOT NULL,
    shares NUMERIC(12, 4) NOT NULL,
    amount NUMERIC(12, 4) NOT NULL,  -- Price per share
    value NUMERIC(15, 2) NOT NULL,   -- Total transaction value
    rule VARCHAR(100),
    notes TEXT,
    realized_pl NUMERIC(15, 2) DEFAULT 0,  -- Calculated by LIFO engine
    stop_loss NUMERIC(12, 4),
    trx_id VARCHAR(50),  -- Transaction ID (e.g., 'B1', 'S2')
    total_cost NUMERIC(15, 2),
    avg_entry NUMERIC(12, 4),
    avg_exit NUMERIC(12, 4),
    exec_grade VARCHAR(10),
    behavior_tag VARCHAR(100),
    retro_notes TEXT,
    date_dt TIMESTAMP,  -- Duplicate date field (legacy)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Foreign key constraint - must be added after trades_summary exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'fk_trade_summary'
    ) THEN
        ALTER TABLE trades_details
        ADD CONSTRAINT fk_trade_summary
        FOREIGN KEY (portfolio_id, trade_id)
        REFERENCES trades_summary(portfolio_id, trade_id)
        ON DELETE CASCADE;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_details_trade_id ON trades_details (portfolio_id, trade_id);
CREATE INDEX IF NOT EXISTS idx_details_date ON trades_details (date);
CREATE INDEX IF NOT EXISTS idx_details_action ON trades_details (action);


-- ============================================
-- TABLE: trading_journal
-- PURPOSE: Daily trading journal entries (CSV: Trading_Journal_Clean.csv)
-- INDEPENDENCE: Not linked to trades (separate daily tracking)
-- ============================================
CREATE TABLE IF NOT EXISTS trading_journal (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    day DATE NOT NULL,
    status VARCHAR(50),
    market_window VARCHAR(50),
    market_cycle VARCHAR(50),
    mct_display_day_num INTEGER,
    above_21ema INTEGER DEFAULT 0,  -- Column name: "> 21e"
    cash_change NUMERIC(15, 2) DEFAULT 0,  -- Column name: "Cash -/+"
    beg_nlv NUMERIC(15, 2) DEFAULT 0,
    end_nlv NUMERIC(15, 2) DEFAULT 0,
    daily_dollar_change NUMERIC(15, 2) DEFAULT 0,
    daily_pct_change NUMERIC(10, 4) DEFAULT 0,
    pct_invested NUMERIC(10, 4) DEFAULT 0,
    spy NUMERIC(10, 2) DEFAULT 0,
    nasdaq NUMERIC(10, 2) DEFAULT 0,
    market_notes TEXT,
    market_action TEXT,
    portfolio_heat NUMERIC(10, 4) DEFAULT 0,
    spy_atr NUMERIC(10, 4) DEFAULT 0,
    nasdaq_atr NUMERIC(10, 4) DEFAULT 0,
    score INTEGER DEFAULT 0,
    highlights TEXT,
    lowlights TEXT,
    mistakes TEXT,
    top_lesson TEXT,
    nlv_source VARCHAR(20) NOT NULL DEFAULT 'manual',
    holdings_source VARCHAR(20) NOT NULL DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_journal_day UNIQUE (portfolio_id, day),
    CONSTRAINT trading_journal_nlv_source_check
        CHECK (nlv_source IN ('manual', 'ibkr_auto', 'ibkr_override')),
    CONSTRAINT trading_journal_holdings_source_check
        CHECK (holdings_source IN ('manual', 'ibkr_auto', 'ibkr_override'))
);

CREATE INDEX IF NOT EXISTS idx_journal_date ON trading_journal (portfolio_id, day DESC);


-- ============================================
-- TABLE: audit_trail
-- PURPOSE: Audit log for all trade operations (CSV: Audit_Trail.csv)
-- ============================================
CREATE TABLE IF NOT EXISTS audit_trail (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    username VARCHAR(100) DEFAULT 'User',
    action VARCHAR(50) NOT NULL,  -- 'BUY', 'SELL', 'DELETE', 'REBUILD', etc.
    trade_id VARCHAR(50),
    ticker VARCHAR(20),
    details TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_trail (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trade_id ON audit_trail (trade_id);


-- ============================================
-- TRIGGER: Update last_updated on trades_summary
-- ============================================
CREATE OR REPLACE FUNCTION update_summary_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_summary_timestamp ON trades_summary;
CREATE TRIGGER trigger_update_summary_timestamp
BEFORE UPDATE ON trades_summary
FOR EACH ROW
EXECUTE FUNCTION update_summary_timestamp();


-- ============================================
-- TRIGGER: Auto-update trading_journal.updated_at
-- ============================================
CREATE OR REPLACE FUNCTION update_journal_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_journal_timestamp ON trading_journal;
CREATE TRIGGER trigger_update_journal_timestamp
BEFORE UPDATE ON trading_journal
FOR EACH ROW
EXECUTE FUNCTION update_journal_timestamp();


-- ============================================
-- MARKET SIGNALS (V10 schema — superseded by migration 010)
-- ============================================
-- The V10 table definition below is kept here for historical reference but
-- is OUT OF DATE: migration 010 dropped + recreated market_signals with the
-- V11 schema (trade_date, signal_type, signal_label, exposure_before,
-- exposure_after, state_before, state_after, meta). This block should be
-- replaced with the V11 definition in a follow-up cleanup.
CREATE TABLE IF NOT EXISTS market_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    signal_date DATE NOT NULL,
    close_price NUMERIC(12, 2) NOT NULL,
    daily_change_pct NUMERIC(10, 4) NOT NULL,
    market_exposure INTEGER DEFAULT 0,
    position_allocation NUMERIC(10, 4) DEFAULT 0,
    buy_switch BOOLEAN DEFAULT FALSE,
    distribution_count INTEGER DEFAULT 0,
    above_21ema BOOLEAN DEFAULT FALSE,
    above_50ma BOOLEAN DEFAULT FALSE,
    buy_signals VARCHAR(100),
    sell_signals VARCHAR(100),
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_signal_per_day UNIQUE (symbol, signal_date)
);

CREATE INDEX idx_signals_symbol_date ON market_signals (symbol, signal_date DESC);
CREATE INDEX idx_signals_date ON market_signals (signal_date DESC);


-- ============================================
-- TABLE: trade_images
-- PURPOSE: Store trade chart images (Weekly, Daily, Exit)
-- ============================================
CREATE TABLE IF NOT EXISTS trade_images (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    trade_id VARCHAR(50) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    image_type VARCHAR(20) NOT NULL,  -- 'weekly', 'daily', 'exit'
    image_url TEXT NOT NULL,  -- R2 object key or full URL
    file_name VARCHAR(255),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trade_images_trade ON trade_images (portfolio_id, trade_id);
CREATE INDEX IF NOT EXISTS idx_trade_images_type ON trade_images (image_type);


-- ============================================
-- TABLE: trade_fundamentals
-- PURPOSE: Extracted fundamental data from MarketSurge screenshots (via Claude Vision API)
-- ============================================
CREATE TABLE IF NOT EXISTS trade_fundamentals (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    trade_id VARCHAR(50) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    image_id INTEGER REFERENCES trade_images(id) ON DELETE SET NULL,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- IBD Ratings
    composite_rating INTEGER,
    eps_rating INTEGER,
    rs_rating INTEGER,
    group_rs_rating VARCHAR(10),
    smr_rating VARCHAR(10),
    acc_dis_rating VARCHAR(10),
    timeliness_rating VARCHAR(10),
    sponsorship_rating VARCHAR(10),

    -- Growth & Volume
    eps_growth_rate NUMERIC(10, 2),
    ud_vol_ratio NUMERIC(10, 2),

    -- Ownership
    mgmt_own_pct NUMERIC(10, 2),
    banks_own_pct NUMERIC(10, 2),
    funds_own_pct NUMERIC(10, 2),
    num_funds INTEGER,

    -- Market Data (at time of screenshot)
    price NUMERIC(12, 4),
    market_cap VARCHAR(50),
    industry_group VARCHAR(100),
    industry_group_rank INTEGER,

    -- Raw JSON (full extraction for future use)
    raw_json JSONB,

    CONSTRAINT unique_fundamental_per_image UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_trade ON trade_fundamentals (portfolio_id, trade_id);
CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker ON trade_fundamentals (ticker);


-- ============================================
-- TABLE: drawdown_notes
-- PURPOSE: User notes on historical deck crossings (Drawdown Discipline tab)
-- KEY: (portfolio_id, deck_level, crossing_date) — one note per crossing
-- ============================================
CREATE TABLE IF NOT EXISTS drawdown_notes (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    deck_level VARCHAR(10) NOT NULL,  -- 'L1', 'L2', 'L3'
    crossing_date DATE NOT NULL,
    note TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_drawdown_note UNIQUE (portfolio_id, deck_level, crossing_date)
);

CREATE INDEX IF NOT EXISTS idx_drawdown_notes ON drawdown_notes (portfolio_id, crossing_date DESC);


-- ============================================
-- TABLE: trade_lessons
-- PURPOSE: User lesson-learned notes per trade (Trade Review tab)
-- KEY: (portfolio_id, trade_id) — one lesson per campaign
-- ============================================
CREATE TABLE IF NOT EXISTS trade_lessons (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    trade_id VARCHAR(50) NOT NULL,
    note TEXT DEFAULT '',
    category VARCHAR(500) DEFAULT '',  -- pipe-separated list of category tags
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_trade_lesson UNIQUE (portfolio_id, trade_id)
);

CREATE INDEX IF NOT EXISTS idx_trade_lessons ON trade_lessons (portfolio_id, trade_id);


-- ============================================
-- TABLE: rule_notes
-- PURPOSE: User observation notes per buy/sell rule (Rule Studio tabs)
-- KEY: (portfolio_id, rule_side, rule_name) — one note per rule per side
-- ============================================
CREATE TABLE IF NOT EXISTS rule_notes (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    rule_side VARCHAR(10) NOT NULL,   -- 'buy' or 'sell'
    rule_name VARCHAR(200) NOT NULL,  -- e.g. 'br3.1 Reclaim 21e'
    note TEXT DEFAULT '',
    status VARCHAR(30) DEFAULT '',    -- 'Validated' / 'Modify' / 'Review' / 'Avoid' / ''
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_rule_note UNIQUE (portfolio_id, rule_side, rule_name)
);

CREATE INDEX IF NOT EXISTS idx_rule_notes ON rule_notes (portfolio_id, rule_side);


-- ============================================
-- TABLE: app_config
-- PURPOSE: Runtime-editable configuration (replaces hardcoded constants).
-- KEY: unique `key` (e.g. 'reset_date', 'hard_decks', 'heat_threshold')
-- VALUE: JSONB so we can store numbers, strings, dates, lists, or objects.
-- ============================================
CREATE TABLE IF NOT EXISTS app_config (
    id SERIAL PRIMARY KEY,
    key VARCHAR(100) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    value_type VARCHAR(20) NOT NULL,  -- 'number', 'string', 'date', 'json'
    category VARCHAR(50) NOT NULL,    -- 'risk', 'sizing', 'heat', 'earnings', etc.
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100) DEFAULT 'User'
);

CREATE INDEX IF NOT EXISTS idx_app_config_category ON app_config (category);


-- ============================================
-- TABLE: dashboard_events
-- PURPOSE: Event/milestone markers rendered on the Dashboard equity curve.
-- SCOPE: CanSlim only (Phase 1). `portfolio_scope` is reserved for Phase 2.
-- ============================================
CREATE TABLE IF NOT EXISTS dashboard_events (
    id SERIAL PRIMARY KEY,
    event_date DATE NOT NULL,
    label VARCHAR(200) NOT NULL,
    category VARCHAR(20) NOT NULL,    -- 'market' or 'personal'
    notes TEXT,
    color_override VARCHAR(20),       -- optional hex color (overrides category default)
    portfolio_scope VARCHAR(50) DEFAULT 'CanSlim',  -- 'CanSlim', 'All' (Phase 2)
    auto_generated BOOLEAN DEFAULT FALSE,  -- true when synced from app_config (e.g. RESET_DATE)
    source_key VARCHAR(100),          -- e.g. 'reset_date' for auto-gen events
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_event_per_date_label UNIQUE (event_date, label)
);

CREATE INDEX IF NOT EXISTS idx_dashboard_events_date ON dashboard_events (event_date);
CREATE INDEX IF NOT EXISTS idx_dashboard_events_scope ON dashboard_events (portfolio_scope);


-- ============================================
-- VERIFICATION QUERIES
-- ============================================

-- Verify all tables created
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

-- Verify portfolios seeded
SELECT * FROM portfolios;
