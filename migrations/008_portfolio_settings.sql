-- ============================================================================
-- Migration 008: Portfolio-level settings (starting_capital, reset_date)
-- ============================================================================
-- Per-user portfolios need their own starting equity baseline and drawdown-
-- reset anchor. The founder tenant has these values baked into app code as
-- constants (e.g. RESET_DATE = 2025-12-16); multi-tenant beta needs them
-- stored per portfolio so every user can set their own.
--
-- Both columns are NULLABLE — a portfolio without a starting_capital is
-- treated as "no baseline yet" (UI shows zeros until the user sets one).
-- reset_date NULL means "use the portfolio created_at" as the anchor.
--
-- NUMERIC(15,2) matches the precision used elsewhere for dollar amounts
-- (trades_summary.total_cost, etc.) so equity math stays consistent.
--
-- Idempotent via ADD COLUMN IF NOT EXISTS.
-- ============================================================================

ALTER TABLE portfolios
    ADD COLUMN IF NOT EXISTS starting_capital NUMERIC(15, 2),
    ADD COLUMN IF NOT EXISTS reset_date       DATE;
