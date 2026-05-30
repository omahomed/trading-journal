-- ============================================================================
-- Migration 042: pinned_routes — sidebar "Pinned" section
-- ============================================================================
-- Per-user pinned route paths for the desktop sidebar's Pinned section.
-- A separate table from pinned_entities (Migration 029) because route paths
-- are strings, not integer FK refs to existing rows. pinned_entities.entity_id
-- is INTEGER NOT NULL and its CHECK constraint limits entity_type to a fixed
-- vocabulary keyed on rows in other tables; bending it polymorphic to fit
-- routes would have touched every toggle_pin / list_pinned_entity_ids caller.
-- Routes are a different "kind" of entity (URL path, not row id), so a
-- purpose-built table is cleaner — same persistence idioms (soft-delete +
-- revive, partial-unique live index, RLS) without polymorphic gymnastics.
--
-- Soft-delete + idempotent revival is the persistence contract:
--   pin   → row exists with deleted_at IS NULL
--   unpin → UPDATE deleted_at = NOW() (same row, soft-deleted)
--   re-pin same route → UPDATE deleted_at = NULL (revives the row,
--                       same id; preserves the original pinned_at for
--                       FIFO-by-pinned_at stability across toggles).
-- toggle_pin_route() in db_layer implements the SELECT-then-branch logic.
--
-- The CHECK constraint on route_path is conservative — matches the current
-- nav.ts inventory (kebab-case lowercase paths like "/log-buy",
-- "/active-campaign") and rejects uppercase letters, underscores, and
-- characters outside [a-z0-9-]. A future nav.ts route that needs uppercase
-- (e.g., "/AI") would require relaxing this constraint via a follow-up
-- migration. Defensive against arbitrary text (XSS attempts, malformed
-- paths) at the DB boundary.
--
-- No audit trigger — pins are personal UI preferences, not trade data.
-- Same precedent as pinned_entities (029), strategies (019),
-- cash_transactions (009), tag_assignments (026).
--
-- RLS enabled + FORCE'd per the canonical migration-024+ pattern for
-- user-scoped tables. NULLIF wrapper on app.user_id prevents the empty-
-- string GUC default from masquerading as a legitimate UUID.
--
-- Reversible: DROP TABLE pinned_routes CASCADE — no foreign-key inbound
-- references, no dependent triggers, no callers outside this migration's
-- companion db_layer helpers (which gracefully tolerate the table being
-- absent if rolled back).
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

CREATE TABLE IF NOT EXISTS pinned_routes (
    id          SERIAL          PRIMARY KEY,
    user_id     UUID            NOT NULL REFERENCES users(id) ON DELETE RESTRICT
                                 DEFAULT (
                                     COALESCE(
                                         NULLIF(current_setting('app.user_id', true), '')::uuid,
                                         'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
                                     )
                                 ),
    route_path  VARCHAR(120)    NOT NULL
                                 CHECK (route_path ~ '^/[a-z0-9-]+(/[a-z0-9-]+)*$'),
    pinned_at   TIMESTAMPTZ     NOT NULL DEFAULT now(),
    deleted_at  TIMESTAMPTZ
);

-- One live pin per user × route. Revival via toggle_pin_route preserves
-- pinned_at across pin/unpin/repin so FIFO ordering is stable. Mirrors
-- idx_pinned_entities_live (Migration 029).
CREATE UNIQUE INDEX IF NOT EXISTS idx_pinned_routes_live
    ON pinned_routes (user_id, route_path)
    WHERE deleted_at IS NULL;

-- Primary read path: "what does this user have pinned, in FIFO order?"
-- list_pinned_routes orders by pinned_at ASC and filters deleted_at IS NULL.
CREATE INDEX IF NOT EXISTS idx_pinned_routes_user_live
    ON pinned_routes (user_id, pinned_at)
    WHERE deleted_at IS NULL;

ALTER TABLE pinned_routes ENABLE ROW LEVEL SECURITY;
ALTER TABLE pinned_routes FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS pinned_routes_isolation ON pinned_routes;
CREATE POLICY pinned_routes_isolation ON pinned_routes FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
-- Expect: empty table.
--   SELECT count(*) FROM pinned_routes;
--
-- Expect: RLS enabled with pinned_routes_isolation policy.
--   SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class
--     WHERE relname = 'pinned_routes';
--
-- Expect: CHECK rejects uppercase, underscores, missing leading slash.
--   INSERT INTO pinned_routes (route_path) VALUES ('/Log-Buy');    -- rejected
--   INSERT INTO pinned_routes (route_path) VALUES ('/log_buy');    -- rejected
--   INSERT INTO pinned_routes (route_path) VALUES ('log-buy');     -- rejected
--
-- Expect: CHECK accepts canonical nav.ts paths.
--   INSERT INTO pinned_routes (route_path) VALUES ('/log-buy');           -- ok
--   INSERT INTO pinned_routes (route_path) VALUES ('/active-campaign');   -- ok
--   INSERT INTO pinned_routes (route_path) VALUES ('/m-factor');          -- ok
--
-- Expect: partial-unique allows re-pinning after soft-delete (same pair).
--   INSERT INTO pinned_routes (route_path) VALUES ('/log-buy');
--   UPDATE pinned_routes SET deleted_at = NOW() WHERE route_path = '/log-buy';
--   INSERT INTO pinned_routes (route_path) VALUES ('/log-buy');           -- ok
