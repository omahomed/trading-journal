-- ============================================================================
-- Migration 056: ticker_taxonomy — user-owned sector/theme mapping per ticker
-- ============================================================================
-- Concentration Risk needs a per-ticker sector + sub-category ("theme") to
-- roll positions up by. yfinance is unreliable for this: SNDK gets
-- "Computer Hardware" (should be Semiconductors — it competes with MU on
-- NAND); NBIS gets "Internet Content" (feels wrong for an AI infra play);
-- ETFs return blank taxonomy entirely. Any concentration report built on
-- raw yfinance would silently under-count exposure.
--
-- This table is the user's own vocabulary. yfinance can *suggest* at Log
-- Buy time; the user classifies. Per-ticker + per-user (Tier 1 multi-
-- tenancy) — not per-campaign, because SNDK's classification shouldn't
-- change across two campaigns of the same name.
--
-- `theme` is the user's chosen name for the sub-bucket ("Memory", "AI
-- Infra", "Leveraged Index", etc.). Free-text so the vocabulary grows
-- organically; the Mapping page autocompletes against existing values so
-- typos don't fragment buckets.
--
-- notes is optional freeform — e.g. "NAND competitor to MU, moves with
-- memory cycle" — surfaced as a tooltip on Concentration Risk hover.
--
-- Follows the standard tenant pattern (user_id DEFAULT + RLS FORCE +
-- tenant_isolation policy) so every user only sees their own mappings.
-- Reference: migration 003.
-- ============================================================================

CREATE TABLE IF NOT EXISTS ticker_taxonomy (
    id          SERIAL PRIMARY KEY,
    user_id     UUID NOT NULL DEFAULT NULLIF(current_setting('app.user_id', true), '')::uuid,
    ticker      TEXT NOT NULL,
    sector      TEXT NOT NULL,
    theme       TEXT,
    notes       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_ticker_taxonomy_user_sector
    ON ticker_taxonomy (user_id, sector);
CREATE INDEX IF NOT EXISTS idx_ticker_taxonomy_user_theme
    ON ticker_taxonomy (user_id, theme);

COMMENT ON TABLE ticker_taxonomy IS
    'User-owned sector + theme mapping per ticker for Concentration Risk. '
    'One row per (user, ticker). yfinance is a suggestion at Log Buy time '
    'only; classification is user-controlled to fix mis-classes (SNDK, '
    'NBIS) and to bucket ETFs (which yfinance leaves blank).';

COMMENT ON COLUMN ticker_taxonomy.theme IS
    'User-chosen sub-bucket name ("Memory", "AI Infra", "Leveraged Index"). '
    'Free-text; Mapping page autocompletes against existing values.';

-- Row-level security — same pattern as migration 003.
ALTER TABLE ticker_taxonomy ENABLE ROW LEVEL SECURITY;
ALTER TABLE ticker_taxonomy FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON ticker_taxonomy;
CREATE POLICY tenant_isolation ON ticker_taxonomy FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);

-- updated_at maintenance trigger.
CREATE OR REPLACE FUNCTION _ticker_taxonomy_touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_ticker_taxonomy_touch_updated_at ON ticker_taxonomy;
CREATE TRIGGER trg_ticker_taxonomy_touch_updated_at
    BEFORE UPDATE ON ticker_taxonomy
    FOR EACH ROW EXECUTE FUNCTION _ticker_taxonomy_touch_updated_at();
