-- 009: trades_summary.be_stop_moved_at
-- ============================================================================
-- Adds a per-campaign timestamp recording when the user moved their stop to
-- breakeven under the "+10% BE rule". Set automatically by the
-- PUT /api/trades/update-stops endpoint when new_stop is within 0.5% of
-- avg_entry AND current price ≥ avg_entry × 1.10; can be cleared by the
-- same endpoint if the user pushes the stop back off BE.
--
-- Combined with sr15 (BE Stop Out) on the sell side, this gives us the
-- denominator (trades where the rule was applied) AND numerator (trades
-- that stopped at BE vs continued higher) needed to evaluate the rule.

ALTER TABLE trades_summary
  ADD COLUMN IF NOT EXISTS be_stop_moved_at TIMESTAMPTZ;

COMMENT ON COLUMN trades_summary.be_stop_moved_at IS
  'Timestamp when stop was moved to breakeven under the +10% BE rule. NULL = not applied.';
