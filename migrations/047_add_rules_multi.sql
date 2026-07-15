-- ============================================================================
-- Migration 047: add rules JSONB column to trades_summary + trades_details
-- ============================================================================
-- Buy-Rule confluence support: convert the single-value `rule` VARCHAR(100)
-- into a multi-value ordered array so a trade can capture the primary
-- buy setup PLUS any confluence rules that fired at entry.
--
-- Storage shape (rules):
--   ["br8.1 Daily STL Break", "br1.2 Cup w Handle", "br3.1 Reclaim 21e"]
--   ["br5.2 Upside Reversal"]
--   []
--
-- Contract:
--   * rules[0] is the PRIMARY rule. All analytics / setup scorecard
--     stats continue reading `rule` (kept in sync as rules[0]).
--   * rules[1..] are CONFLUENCE rules — display-only context, no PF
--     weighting change. Future "Confluence" analytics will read the
--     full array.
--
-- Rollback-friendly split (not an in-place type change):
--   * Add `rules JSONB NOT NULL DEFAULT '[]'::jsonb` on both tables.
--   * Backfill: every existing single-value `rule` becomes a
--     one-element array in `rules` (or `[]` if NULL/empty).
--   * `rule` column stays alongside. Backend writes both columns
--     during the transition — `rule` = rules[0] (or '' when the array
--     is empty), `rules` = the full array. A follow-up migration will
--     drop `rule` once the frontend has been on the array shape for a
--     stable window.
--
-- Applies to BOTH trades_summary and trades_details. Summary's rules
-- mirrors the B1 (earliest BUY) detail row's rules — same convention
-- as summary.rule mirrors first BUY's rule (see
-- db_layer.mirror_detail_edit_to_summary). Convention preserved.
--
-- SELL rows in trades_details will have rules arrays too (single-value
-- for now — sr8.1 etc.), but the multi-select UI is BUY-side only.
-- Sells stay single-select in the UI.
-- ============================================================================

ALTER TABLE trades_summary
    ADD COLUMN IF NOT EXISTS rules JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE trades_details
    ADD COLUMN IF NOT EXISTS rules JSONB NOT NULL DEFAULT '[]'::jsonb;

-- Backfill: coerce every existing single-value `rule` into a
-- one-element JSONB array. NULL / empty strings become '[]'.
UPDATE trades_summary
   SET rules = jsonb_build_array(rule)
 WHERE rule IS NOT NULL
   AND rule <> ''
   AND rules = '[]'::jsonb;

UPDATE trades_details
   SET rules = jsonb_build_array(rule)
 WHERE rule IS NOT NULL
   AND rule <> ''
   AND rules = '[]'::jsonb;

-- GIN indexes so "which trades used br8.1?" filter queries stay fast
-- once analytics starts using `rules ? 'br8.1'` or `rules @> '["brX"]'`.
CREATE INDEX IF NOT EXISTS idx_summary_rules
    ON trades_summary USING gin (rules);

CREATE INDEX IF NOT EXISTS idx_details_rules
    ON trades_details USING gin (rules);
