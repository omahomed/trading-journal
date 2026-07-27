-- ============================================================================
-- Migration 051: retire the "Discretionary action taken today?" system counter
-- ============================================================================
-- The rule-break tally is already covered by the Daily Report Card scorecard,
-- so the redundant checklist counter is being removed to reduce noise on the
-- Trading Checklist page. Companion change: the seed is also removed from
-- db_layer._ROUTINE_SYSTEM_ITEMS so newly-provisioned tenants never see it.
--
-- Approach — soft-delete, not DELETE:
--   Setting active=false hides the item from list_routine_items() (which
--   filters WHERE ri.active = true) without cascading routine_log rows.
--   Any historical incident ticks stay queryable for later analysis; if the
--   decision is ever reversed, flipping active=true restores the item plus
--   all its history.
--
-- RLS bypass: neondb_owner (MIGRATIONS_DATABASE_URL) has BYPASSRLS, so this
-- UPDATE walks every tenant's row despite the routine_items FORCE ROW LEVEL
-- SECURITY policy. Same pattern as migration 009 backfill.
-- ============================================================================

UPDATE routine_items
   SET active = false,
       updated_at = now()
 WHERE is_system = true
   AND name = 'Discretionary action taken today?';

-- ============================================================================
-- Verification (manual, after COMMIT)
-- ============================================================================
--   SELECT user_id, active, updated_at
--     FROM routine_items
--    WHERE is_system = true
--      AND name = 'Discretionary action taken today?';
--   -- expect: every row active=false with a fresh updated_at.
-- ============================================================================
