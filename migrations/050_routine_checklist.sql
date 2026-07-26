-- ============================================================================
-- Migration 050: routine_items + routine_log — Trading Checklist Phase 1
-- ============================================================================
-- Two tables backing the new "Trading Checklist" page (under Daily Workflow,
-- distinct from Daily Routine). User-configurable checklist with frequency
-- (daily / weekly / monthly / quarterly) and slot (premarket, intraday,
-- end_of_shift, after_close, weekend) as INDEPENDENT fields. Slot is
-- nullable so monthly + quarterly items can live without a time-of-day slot
-- (they're rendered as their own sections).
--
-- Item types:
--   task    — the normal check-when-done model. Participates in overdue.
--   counter — an incident counter (e.g. "discretionary action taken today").
--             No-tick is the good state; each tick logs an incident.
--             Never renders as overdue.
--
-- Same-day undo is enforced by a UNIQUE constraint on
--   (item_id, ((completed_at AT TIME ZONE 'America/Chicago')::date))
-- but Postgres won't accept that expression directly in a unique index
-- because AT TIME ZONE with a text literal is STABLE, not IMMUTABLE.
-- The workaround: a maintained `completed_date_ct DATE` column, set by a
-- BEFORE INSERT/UPDATE trigger, then the unique index is on plain columns.
--
-- Seed strategy: no seed inserts in this migration. System items are
-- auto-provisioned on first GET /api/routine/items per user (upsert with
-- ON CONFLICT DO NOTHING). This handles the founder plus any TEST_ACCOUNTS
-- tenants uniformly and avoids the migration-time RLS gymnastics of
-- inserting rows for multiple user_ids from a single app.user_id session.
--
-- No audit trigger — personal checklist state, not trade data. Same
-- precedent as pinned_routes (042), weekly_retros (025), tag_assignments
-- (026), cash_transactions (009).
--
-- RLS enabled + FORCE'd per the canonical migration-024+ pattern.
-- ============================================================================

CREATE TABLE IF NOT EXISTS routine_items (
    id          SERIAL          PRIMARY KEY,
    user_id     UUID            NOT NULL REFERENCES users(id) ON DELETE RESTRICT
                                 DEFAULT (
                                     COALESCE(
                                         NULLIF(current_setting('app.user_id', true), '')::uuid,
                                         'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
                                     )
                                 ),
    name        TEXT            NOT NULL CHECK (length(name) BETWEEN 1 AND 120),
    frequency   TEXT            NOT NULL CHECK (frequency IN ('daily','weekly','monthly','quarterly')),
    slot        TEXT            CHECK (slot IS NULL OR slot IN ('premarket','intraday','end_of_shift','after_close','weekend')),
    item_type   TEXT            NOT NULL DEFAULT 'task' CHECK (item_type IN ('task','counter')),
    link        TEXT            CHECK (link IS NULL OR link ~ '^https?://'),
    is_system   BOOLEAN         NOT NULL DEFAULT false,
    sort_order  INTEGER         NOT NULL DEFAULT 0,
    active      BOOLEAN         NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- Primary read path: "give me this user's active items, ordered."
CREATE INDEX IF NOT EXISTS idx_routine_items_user_read
    ON routine_items (user_id, active, frequency, slot, sort_order);

-- Uniqueness for system items only — a user can't have two "Journal" seeds.
-- Custom items may share names (rename friction shouldn't be a hard block).
CREATE UNIQUE INDEX IF NOT EXISTS idx_routine_items_user_system_name
    ON routine_items (user_id, name)
    WHERE is_system = true;

ALTER TABLE routine_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE routine_items FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS routine_items_isolation ON routine_items;
CREATE POLICY routine_items_isolation ON routine_items FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


CREATE TABLE IF NOT EXISTS routine_log (
    id                  SERIAL          PRIMARY KEY,
    item_id             INTEGER         NOT NULL REFERENCES routine_items(id) ON DELETE CASCADE,
    user_id             UUID            NOT NULL REFERENCES users(id) ON DELETE RESTRICT
                                         DEFAULT (
                                             COALESCE(
                                                 NULLIF(current_setting('app.user_id', true), '')::uuid,
                                                 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
                                             )
                                         ),
    completed_at        TIMESTAMPTZ     NOT NULL DEFAULT now(),
    completed_date_ct   DATE            NOT NULL
);

-- Trigger maintains completed_date_ct = completed_at in America/Chicago.
-- Must be a real column because AT TIME ZONE 'X' is STABLE and can't
-- appear in a UNIQUE index expression directly.
CREATE OR REPLACE FUNCTION routine_log_set_date_ct()
RETURNS TRIGGER AS $$
BEGIN
    NEW.completed_date_ct := (NEW.completed_at AT TIME ZONE 'America/Chicago')::date;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_routine_log_set_date_ct ON routine_log;
CREATE TRIGGER trg_routine_log_set_date_ct
    BEFORE INSERT OR UPDATE OF completed_at ON routine_log
    FOR EACH ROW EXECUTE FUNCTION routine_log_set_date_ct();

-- One tick per item per CT day. Same-day untick is legal via
-- DELETE /api/routine/log/{id}; cross-day untick is 409'd in the endpoint.
CREATE UNIQUE INDEX IF NOT EXISTS idx_routine_log_item_day
    ON routine_log (item_id, completed_date_ct);

-- "Give me the most recent tick per item" — powers the last_run derivation
-- and the per-item history query.
CREATE INDEX IF NOT EXISTS idx_routine_log_item_completed
    ON routine_log (item_id, completed_at DESC);

-- Per-user timeline for future analytics (Phase 4).
CREATE INDEX IF NOT EXISTS idx_routine_log_user_completed
    ON routine_log (user_id, completed_at DESC);

ALTER TABLE routine_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE routine_log FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS routine_log_isolation ON routine_log;
CREATE POLICY routine_log_isolation ON routine_log FOR ALL
    USING      (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid)
    WITH CHECK (user_id = NULLIF(current_setting('app.user_id', true), '')::uuid);


-- ============================================================================
-- Verification queries (manual, after COMMIT)
-- ============================================================================
--   SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class
--    WHERE relname IN ('routine_items','routine_log');
--   -- expect: both rows with true / true
--
--   -- Sanity: same-day trigger populates completed_date_ct.
--   INSERT INTO routine_items (name, frequency, slot) VALUES ('t','daily','after_close') RETURNING id;
--   INSERT INTO routine_log (item_id) VALUES (<id>) RETURNING completed_at, completed_date_ct;
--   -- expect: completed_date_ct = current date in America/Chicago.
--
--   -- Same-day double-tick blocked:
--   INSERT INTO routine_log (item_id) VALUES (<id>);   -- errors on unique
-- ============================================================================
