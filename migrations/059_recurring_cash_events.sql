-- ============================================================================
-- Migration 059: recurring_cash_events — configurable bi-weekly deposits
-- ============================================================================
-- Set-and-forget recurring cash deposits (initially: 457B Plan bi-weekly
-- contribution at 90% × $980 = $882). Not an auto-poster — the NLV Entry
-- page renders a reminder card when a configured event's next_due_date
-- has arrived, and the user clicks Post (with optional amount override)
-- or Skip. State only advances on that click; there is no cron.
--
-- Design notes:
--   * ONE row per configured event. In practice most portfolios will have
--     0 or 1 rows; the schema supports N so the same 457B could later add
--     "employer match" as a separate event without another table.
--   * amount is captured as base_amount + percent so the reminder card
--     can render "Base $980 × 90% = $882". Percent defaults to 100 for
--     the fixed-dollar case (base_amount is the deposit directly).
--   * next_due_date carries the CURRENT cycle's target. Post/Skip advance
--     it by cadence_days. NO last_fired_at, NO history table — the
--     cash_transactions row IS the audit trail (note prefixed with the
--     event's note field for traceability).
--   * anchor_date is the reference start date. Present-day utility is
--     the initial next_due_date seed (`next_due_date = anchor_date` on
--     create). Retained on the row for future features (e.g. "regenerate
--     the schedule from anchor" if next_due drift is ever a concern).
--
-- Tenant isolation: user_id defaults from app.user_id (migration 003
-- pattern) so the daily-workflow surfaces see only the caller's events.
-- RLS enabled + the SELECT/INSERT/UPDATE/DELETE policies mirror
-- cash_transactions.
--
-- Seed: one row for 457B Plan with base_amount=980, percent=90,
-- anchor_date=next_due_date=2026-08-07 (the operator's next payday, from
-- the payroll sheet). Idempotent guard — re-run finds the row and skips.
--
-- The migration runner wraps this file in a transaction; no BEGIN/COMMIT.
-- ============================================================================

CREATE TABLE IF NOT EXISTS recurring_cash_events (
    id              SERIAL PRIMARY KEY,
    user_id         UUID NOT NULL DEFAULT (
        COALESCE(
            NULLIF(current_setting('app.user_id', true), '')::uuid,
            'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
        )
    ),
    portfolio_id    INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    anchor_date     DATE NOT NULL,
    cadence_days    INTEGER NOT NULL DEFAULT 14 CHECK (cadence_days > 0),
    base_amount     NUMERIC(12, 2) NOT NULL CHECK (base_amount >= 0),
    percent         NUMERIC(6, 2) NOT NULL DEFAULT 100.00
                    CHECK (percent >= 0 AND percent <= 200),
    note            TEXT NOT NULL DEFAULT '',
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    next_due_date   DATE NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_recurring_cash_events_portfolio
    ON recurring_cash_events (portfolio_id, active);
CREATE INDEX IF NOT EXISTS idx_recurring_cash_events_user
    ON recurring_cash_events (user_id);

ALTER TABLE recurring_cash_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS recurring_cash_events_tenant_isolation ON recurring_cash_events;
CREATE POLICY recurring_cash_events_tenant_isolation
    ON recurring_cash_events
    USING (user_id = COALESCE(
        NULLIF(current_setting('app.user_id', true), '')::uuid,
        'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
    ))
    WITH CHECK (user_id = COALESCE(
        NULLIF(current_setting('app.user_id', true), '')::uuid,
        'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid
    ));

-- Seed the founder's 457B bi-weekly config. Idempotent guard on
-- (user_id, portfolio_id, note) — re-run finds the row and skips.
DO $$
DECLARE
    p457_id      INTEGER;
    founder_id   UUID := 'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a';
    existing_id  INTEGER;
BEGIN
    SELECT id INTO p457_id FROM portfolios WHERE name = '457B Plan';
    IF p457_id IS NULL THEN
        RAISE NOTICE 'No 457B Plan portfolio found — seed skipped';
        RETURN;
    END IF;

    SELECT id INTO existing_id
      FROM recurring_cash_events
     WHERE user_id     = founder_id
       AND portfolio_id = p457_id
       AND note = '457B bi-weekly contribution';

    IF existing_id IS NOT NULL THEN
        RAISE NOTICE 'Recurring 457B config already exists (id=%) — seed skipped', existing_id;
        RETURN;
    END IF;

    INSERT INTO recurring_cash_events
        (user_id, portfolio_id, anchor_date, cadence_days,
         base_amount, percent, note, active, next_due_date)
    VALUES
        (founder_id, p457_id, DATE '2026-08-07', 14,
         980.00, 90.00, '457B bi-weekly contribution',
         TRUE, DATE '2026-08-07');

    RAISE NOTICE 'Seeded recurring 457B contribution config';
END $$;

-- ============================================================================
-- Verification (manual, after COMMIT)
-- ============================================================================
--   -- expect: table exists with the expected columns
--   \d+ recurring_cash_events
--
--   -- expect: 1 seeded row for the founder's 457B account
--   SELECT r.id, p.name, r.anchor_date, r.cadence_days, r.base_amount,
--          r.percent, r.next_due_date, r.active, r.note
--     FROM recurring_cash_events r
--     JOIN portfolios p ON p.id = r.portfolio_id
--    WHERE p.name = '457B Plan';
-- ============================================================================
