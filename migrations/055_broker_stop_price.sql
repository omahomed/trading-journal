-- ============================================================================
-- Migration 055: trades_summary.broker_stop_price — two-stop model flag
-- ============================================================================
-- Position Sizer's two-stop model parks a physical order at −0.75× ATR21
-- from the B1 fill (SR14 "0.75× ATR Stop") ALONGSIDE the composite/thesis
-- stop (SR1 territory). The composite is math scaffolding for share count;
-- the broker stop is what actually fires in the market first if the
-- 0.75× ATR line is breached intraday.
--
-- Presence of `broker_stop_price` is the flag that a position is on the
-- two-stop model — no separate boolean column needed. Read semantics:
--
--   * NULL → single-stop model (classic SR1 capital protection at deep
--     thesis stop; no physical order tighter than that)
--   * NUMERIC > 0 → two-stop model active. Sell-rule tier classifier
--     shows SR14 instead of SR1 for B1_return_pct < 10% positions, so
--     the ACS Sell Rule column reflects the actual first-firing stop.
--
-- Wire-up: Log Buy accepts an optional broker_stop_price at entry;
-- Position Sizer's "Send to Log Buy" prefills it when the two-stop
-- model was used; Trade Manager + Trade Journal + ACS right-click each
-- offer an edit surface for after-the-fact backfill. Setting to NULL
-- clears the flag and drops the position back to SR1.
--
-- No default value — legacy rows stay NULL (single-stop), which is
-- semantically correct: we didn't have a broker stop parked for them.
-- ============================================================================

ALTER TABLE trades_summary
    ADD COLUMN IF NOT EXISTS broker_stop_price NUMERIC;

COMMENT ON COLUMN trades_summary.broker_stop_price IS
    'Physical broker stop price at −0.75× ATR21 from B1 fill (SR14). '
    'NULL = single-stop model. NUMERIC > 0 = two-stop model active, '
    'ACS Sell Rule column shows SR14 instead of SR1 for <10% B1 return.';
