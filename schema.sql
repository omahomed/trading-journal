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
    market_action VARCHAR(100),
    score INTEGER DEFAULT 0,
    highlights TEXT,
    lowlights TEXT,
    mistakes TEXT,
    top_lesson TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_journal_day UNIQUE (portfolio_id, day)
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
-- VERIFICATION QUERIES
-- ============================================

-- Verify all tables created
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

-- Verify portfolios seeded
SELECT * FROM portfolios;
