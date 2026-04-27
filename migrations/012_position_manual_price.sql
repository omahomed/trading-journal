-- ============================================================================
-- Migration 012: trades_summary.manual_price — manual override for live price
-- ============================================================================
-- Options pricing via yfinance is unreliable (OCC symbols frequently fail to
-- resolve), so positions fall back to cost basis and the dashboard NLV +
-- exposure tiles show stale numbers. This adds a per-position price override:
-- when manual_price is set, the NLV computation and Active Campaign Summary
-- prefer it over the yfinance result.
--
-- Shape:
--   - manual_price: per-share/per-contract price (NULL = no override)
--   - manual_price_set_at: timestamp the override was last written (used by
--     the UI to show how stale the manual value is)
--
-- Both columns are nullable and default NULL — no impact on existing rows.
-- Equity positions can use the same field too if a user ever wants to pin a
-- price (e.g. broker-side halt, after-hours), but the immediate driver is
-- options. The override is cleared automatically when the campaign closes
-- (trades_summary.status flips to CLOSED) — handled by save_summary_row.
-- ============================================================================

ALTER TABLE trades_summary
    ADD COLUMN IF NOT EXISTS manual_price        NUMERIC(15, 4),
    ADD COLUMN IF NOT EXISTS manual_price_set_at TIMESTAMPTZ;

-- No backfill: existing rows get NULL, which the read paths interpret as
-- 'fall through to live price provider'.
