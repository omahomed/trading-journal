-- ============================================================================
-- Migration 032: daily_captures_backfill — Phase 7 data move
-- ============================================================================
-- Move existing user-uploaded note images from trade_images into the new
-- daily_journal_captures table. Before Phase 7, the daily report's
-- "Daily Thoughts" card routed user uploads into trade_images with a
-- synthetic trade_id="EOD-{day}" and image_type='eod_note'. Phase 7
-- gives those uploads their own first-class table; this migration
-- relocates the existing rows so the new "Daily Captures" section shows
-- them on first render.
--
-- The two other 'eod_*' image_type values — 'eod_dashboard' and
-- 'eod_campaign' — stay in trade_images. They're labeled "auto-generated
-- EOD content" and continue to render in the End Of Day Snapshots
-- collapsible (which keeps its existing behavior post-Phase 7).
--
-- IDEMPOTENCY: re-running this migration produces zero changes.
--   - The INSERT uses WHERE NOT EXISTS (matching on storage_ref) — once a
--     trade_images row has been migrated, the equivalent capture row
--     blocks re-insert.
--   - We do NOT DELETE the source trade_images rows. The table has no
--     deleted_at column (per the design note in migration 006), so a
--     hard-delete would be irreversible. Instead, the frontend reads
--     trade_images via /api/snapshots/{day} and now filters out
--     image_type='eod_note' at the client (and the GET endpoint also
--     excludes them — see api/main.py changes). The legacy rows survive
--     in trade_images as harmless historical data.
--
-- WHY HARD-FILTER INSTEAD OF SOFT-DELETE: trade_images deliberately lacks
-- a deleted_at column (migration 006 documents the rationale — it
-- hard-deletes via FK cascade from trades). Adding deleted_at just for
-- this single backfill is over-engineered; the cleaner path is to read
-- around the legacy rows.
--
-- TRADE_ID PARSING: trade_id is the literal string "EOD-{YYYY-MM-DD}".
-- SUBSTRING extracts the date suffix; CAST to ::date and JOIN to
-- trading_journal on (portfolio_id, day). The defensive `tj.deleted_at
-- IS NULL` guards against soft-deleted journal rows (full unique
-- constraint on (portfolio_id, day) means deleted rows linger until
-- hard-deleted or restored — see Phase 7 audit Section 1, Concern 2).
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

INSERT INTO daily_journal_captures (
    user_id, daily_journal_id, storage_ref, file_name,
    mime_type, file_size_bytes, sort_order, caption, created_at
)
SELECT
    ti.user_id,
    tj.id,
    ti.image_url,                                  -- legacy column == R2 object key
    ti.file_name,
    NULL,                                          -- legacy rows didn't record mime
    NULL,                                          -- legacy rows didn't record byte size
    0,                                             -- default sort_order (sort by created_at)
    '',                                            -- empty caption
    ti.uploaded_at
FROM trade_images ti
JOIN trading_journal tj
  ON tj.portfolio_id = ti.portfolio_id
 AND tj.day = (SUBSTRING(ti.trade_id FROM 5))::date
WHERE ti.image_type = 'eod_note'
  AND tj.deleted_at IS NULL
  AND NOT EXISTS (
      SELECT 1 FROM daily_journal_captures djc
       WHERE djc.storage_ref = ti.image_url
        AND djc.deleted_at IS NULL
  );


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
--   -- Counts should match: every eod_note row with a live parent journal
--   -- row gets a daily_journal_captures row.
--   SELECT
--     (SELECT count(*) FROM daily_journal_captures) AS captures_count,
--     (SELECT count(*) FROM trade_images ti JOIN trading_journal tj
--        ON tj.portfolio_id = ti.portfolio_id
--       AND tj.day = (SUBSTRING(ti.trade_id FROM 5))::date
--      WHERE ti.image_type = 'eod_note'
--        AND tj.deleted_at IS NULL) AS expected_count;
--
--   -- Spot-check: same storage_ref + same created_at on both sides.
--   SELECT ti.image_url, ti.uploaded_at,
--          djc.storage_ref, djc.created_at
--     FROM trade_images ti
--     JOIN daily_journal_captures djc ON djc.storage_ref = ti.image_url
--    WHERE ti.image_type = 'eod_note'
--    LIMIT 5;
--
--   -- Re-running the migration should INSERT zero rows.
