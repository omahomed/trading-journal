-- ============================================================================
-- Migration 043: add trend_count column to trading_journal
-- ============================================================================
-- Persist the signed Trend Count (length of the current 21e leg) alongside
-- the existing market_cycle (Migration 010) and mct_display_day_num
-- (Migration 015) columns, so the Daily Journal page can render the value
-- directly from the row without replaying the engine on every visit.
--
-- Value semantics (mirrors the live banner logic in api/mct_engine.py):
--   NULL  — pre-first-Step-4 in the replay (blank), or no engine bar for
--           this journal date (holiday / pre-ingest save). NULL is a
--           first-class value; do not default to 0 which would collide
--           with a legit arm bar.
--    +N   — sessions since the last Step-4 arm; positive leg holding
--    -N   — sessions since the last Tier-1 confirmed break; negative leg
--
-- Type choice — SMALLINT (−32,768…32,767). Comfortably fits the widest
-- possible leg (~130 trading years) with room to spare. No CHECK: 0 is a
-- valid arm bar, negative is a valid broken-leg count, positive is a
-- valid holding-leg count.
--
-- Backfill: scripts/backfill_trend_count.py replays the engine once over
-- ^IXIC history, extracts (trend_anchor_idx, trend_sign) per bar record,
-- and stamps every journal row whose date matches a bar. Rows without a
-- matching bar (weekends / holidays / pre-2010) stay NULL — the lazy-heal
-- pattern used for mct_display_day_num is not yet mirrored here (Half 2).
-- ============================================================================

ALTER TABLE trading_journal
    ADD COLUMN IF NOT EXISTS trend_count SMALLINT NULL;
