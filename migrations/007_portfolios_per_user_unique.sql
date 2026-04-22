-- ============================================================================
-- Migration 007: Scope portfolios.name UNIQUE to per-user
-- ============================================================================
-- The original schema had `name VARCHAR(50) UNIQUE NOT NULL` on portfolios,
-- which made portfolio names globally unique across the whole database. That
-- worked for a single-tenant Streamlit app but blocks multi-tenancy — a beta
-- user couldn't name their portfolio "CanSlim" (or anything else the founder
-- already owns).
--
-- Fix: drop the global UNIQUE, replace with UNIQUE (user_id, name) so each
-- user gets their own namespace. Combined with the user_id column added in
-- migration 001 + RLS from migration 003, this closes the per-user portfolio
-- gap that was blocking end-to-end beta signups.
--
-- Postgres auto-generates the old constraint name as `portfolios_name_key`
-- (from the inline UNIQUE). We drop it by that name; IF EXISTS makes the
-- migration idempotent.
--
-- Idempotency: IF EXISTS on the drop, IF NOT EXISTS on the new index.
-- ============================================================================

-- Drop the global UNIQUE constraint
ALTER TABLE portfolios DROP CONSTRAINT IF EXISTS portfolios_name_key;

-- Add a per-user UNIQUE constraint — safe because migration 005 already
-- flipped user_id to NOT NULL, so every row has a non-null user_id.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_portfolios_user_name
    ON portfolios (user_id, name);
