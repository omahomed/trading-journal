-- ============================================================================
-- Migration 067: suggested_exposure_pct — MCT engine snapshot at save time
-- ============================================================================
-- Journal Log needs a per-row reference for "what exposure did the MCT
-- engine suggest on this day" so the operator can compare their actual
-- `% invested` against the model's guidance retrospectively. The value
-- comes from the same run_engine + to_rally_prefix_response pipeline the
-- M Factor page renders, extracting `entry_exposure` (the current-cycle
-- ceiling from state["exposure"] after all ratchets, cuts, and step-
-- ladder progression have been applied).
--
-- Stamped at save time on every trading_journal insert (NLV Entry,
-- journal_edit, journal_batch_edit) alongside market_cycle /
-- mct_display_day_num / trend_count — same discipline: the value
-- captured at save is the truth of what the engine reported that
-- moment, and it stays that way (see _heal_recent_mct_stamps for the
-- NULL-only healing rule).
--
-- No backfill: pre-migration rows keep NULL and the Journal Log column
-- renders "—" for those days. Going-forward stamping starts as soon as
-- migration 067 runs. Historical suggested-exposure values are lost
-- because the engine's state["exposure"] is stateful (ratchets, cuts)
-- — reconstructing a past day's suggested exposure would require
-- replaying every engine run from cycle start to that day, and we
-- don't have the intermediate ratchet history stored.
--
-- Widened to DECIMAL(6, 2) so values up to 9999.99% fit — the engine
-- can produce 100 / 120 / 140 / 160 in POWERTREND with an SR8-driven
-- ratchet ceiling; leaving room for future doctrine tweaks.

ALTER TABLE trading_journal
ADD COLUMN IF NOT EXISTS suggested_exposure_pct DECIMAL(6, 2);

COMMENT ON COLUMN trading_journal.suggested_exposure_pct IS
'MCT engine entry_exposure at save time (mirrors /api/market/rally-prefix). '
'NULL for pre-067 rows. Stamped once at save; never overwritten by heal (per '
'the 2026-07-29 immutability rule).';
