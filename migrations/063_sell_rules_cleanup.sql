-- ============================================================================
-- Migration 063: sell rules taxonomy cleanup — retag historical SR12→SR7, SR14→SR1
-- ============================================================================
-- Per 2026-08-07 review notes on the sell-rules audit table:
--
--   * SR4 (Time Stop) — retired. SR3 semantics cover portfolio-time trims.
--     No historical stamps to retag (0 rows).
--   * SR6 (8e Momentum Trim) — retired. Already doctrinally retired per
--     the canonical handoff (0-for-5 fire quality). 0 rows in the DB.
--   * SR7 — description shortened to "21e Violation" in the app glossary.
--     Historical stamps prefixed "sr7 ..." keep their existing text; the
--     display layer resolves them by the "sr7" code, not the full string.
--   * SR8.2 — glossary text was stale ("drifts further below 8w MA");
--     engine (mors/mors_backtest.py) already fires Quicksand on the 13w
--     MA break per DEFAULT_WEEKLY_EMAS = (8, 13, 21). Doc-only fix in the
--     app glossary; no engine or data change.
--   * SR12 (TQQQ Strategy Exit) — collapsed into SR7 (same 21 EMA
--     violation on a different index). 3 historical stamps retagged.
--   * SR14 (0.75× ATR Stop) — collapsed into SR1. Broker-stop presence
--     becomes a chip on the row instead of a tier promotion. 1
--     historical stamp retagged.
--
-- Retag strategy: match on the lowercase "sr14" / "sr12" prefix so both
-- the freshly-normalized short-form ("SR12") and the legacy full-form
-- ("sr12 tqqq strategy exit") are caught. Rewritten values keep the
-- current app's stamp shape "sr7 21e violation" / "sr1 capital
-- protection" so the display layer stays consistent.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

-- SR12 → SR7: 3 historical rows in the current DB (all CanSlim / TQQQ).
UPDATE trades_summary
   SET sell_rule = 'sr7 21e violation'
 WHERE deleted_at IS NULL
   AND sell_rule IS NOT NULL
   AND LOWER(TRIM(sell_rule)) LIKE 'sr12%';

-- SR14 → SR1: 1 historical row in the current DB.
UPDATE trades_summary
   SET sell_rule = 'sr1 capital protection'
 WHERE deleted_at IS NULL
   AND sell_rule IS NOT NULL
   AND LOWER(TRIM(sell_rule)) LIKE 'sr14%';


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
--   SELECT LOWER(TRIM(sell_rule)) AS rule, COUNT(*)
--     FROM trades_summary
--    WHERE deleted_at IS NULL
--      AND sell_rule IS NOT NULL AND TRIM(sell_rule) <> ''
--    GROUP BY 1 ORDER BY 1;
--   → No rows starting with "sr12" or "sr14" should remain.
--   → SR7 + SR1 counts should have bumped by 3 and 1 respectively.
