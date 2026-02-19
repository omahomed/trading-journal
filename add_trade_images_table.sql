-- ============================================
-- MIGRATION: Add trade_images table
-- Run this once to add image storage capability
-- ============================================

CREATE TABLE IF NOT EXISTS trade_images (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    trade_id VARCHAR(50) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    image_type VARCHAR(20) NOT NULL,  -- 'weekly', 'daily', 'exit'
    image_url TEXT NOT NULL,  -- R2 object key or full URL
    file_name VARCHAR(255),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_trade_image UNIQUE (portfolio_id, trade_id, image_type)
);

CREATE INDEX IF NOT EXISTS idx_trade_images_trade ON trade_images (portfolio_id, trade_id);
CREATE INDEX IF NOT EXISTS idx_trade_images_type ON trade_images (image_type);

-- Verify table was created
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_name = 'trade_images'
ORDER BY ordinal_position;
