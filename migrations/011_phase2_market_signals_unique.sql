-- 011: MCT V11 Phase 2 — market_signals uniqueness
-- ============================================================================
-- Adds a UNIQUE constraint on (trade_date, signal_type) so the engine writer
-- can use INSERT ... ON CONFLICT DO NOTHING for idempotency. Phase 1 left the
-- table empty, so this is non-destructive: no duplicates exist to block the
-- constraint addition.
--
-- The engine fires at most one of each signal_type per trade_date — multiple
-- canonical events per day are always distinct types (e.g., RALLY_INVALIDATED
-- and STEP_0_RALLY_DAY both firing on 2025-04-09). This constraint enforces
-- that contract at the database layer.

ALTER TABLE market_signals
    ADD CONSTRAINT uq_market_signals_date_type UNIQUE (trade_date, signal_type);
