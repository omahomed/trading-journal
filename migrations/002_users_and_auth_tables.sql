-- ============================================================================
-- Migration 002: Users table + Auth.js schema + FK constraints (Tier 1, step 2a)
-- ============================================================================
-- Creates the four tables required by Auth.js (next-auth v5) pg adapter:
--   users, accounts, sessions, verification_token
-- All tables use UUID primary keys to match the user_id columns added in
-- migration 001. Column names that Auth.js expects in camelCase ("userId",
-- "emailVerified", "providerAccountId", "sessionToken") are quoted.
--
-- Seeds the founder row so every existing tenant row (backfilled to
-- d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a in migration 001) has a matching user.
-- Founder email pulled from CLAUDE.md user profile.
--
-- Finally, adds FOREIGN KEY constraints from every tenant-scoped table to
-- users(id) with ON DELETE RESTRICT — user deletion must go through explicit
-- data cleanup (prevents accidental catastrophic deletes; can soften later
-- when a "delete my account" flow with soft-deletes is built).
--
-- Idempotency: uses IF NOT EXISTS / ON CONFLICT everywhere; safe to re-run.
-- ============================================================================

-- pgcrypto provides gen_random_uuid() (Neon has it available by default)
CREATE EXTENSION IF NOT EXISTS pgcrypto;


-- ---------- users ----------
CREATE TABLE IF NOT EXISTS users (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT,
    email           TEXT         UNIQUE,
    "emailVerified" TIMESTAMPTZ,
    image           TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Seed the founder row. ON CONFLICT = idempotent across re-runs.
INSERT INTO users (id, email, name)
VALUES (
    'd7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a'::uuid,
    'omahomed@gmail.com',
    'Mahomed Ouedraogo'
)
ON CONFLICT (id) DO NOTHING;


-- ---------- accounts (OAuth provider links) ----------
CREATE TABLE IF NOT EXISTS accounts (
    id                  UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    "userId"            UUID    NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type                TEXT    NOT NULL,
    provider            TEXT    NOT NULL,
    "providerAccountId" TEXT    NOT NULL,
    refresh_token       TEXT,
    access_token        TEXT,
    expires_at          BIGINT,
    id_token            TEXT,
    scope               TEXT,
    session_state       TEXT,
    token_type          TEXT,
    UNIQUE (provider, "providerAccountId")
);

CREATE INDEX IF NOT EXISTS idx_accounts_user ON accounts ("userId");


-- ---------- sessions (used if session strategy = database; JWT = unused) ----------
CREATE TABLE IF NOT EXISTS sessions (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    "userId"       UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires        TIMESTAMPTZ NOT NULL,
    "sessionToken" TEXT        UNIQUE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions ("userId");


-- ---------- verification_token (magic-link one-time tokens) ----------
-- Auth.js expects the singular name `verification_token`, even though the rest
-- of the schema uses plural. Don't rename without updating the adapter config.
CREATE TABLE IF NOT EXISTS verification_token (
    identifier TEXT         NOT NULL,
    expires    TIMESTAMPTZ  NOT NULL,
    token      TEXT         NOT NULL,
    PRIMARY KEY (identifier, token)
);


-- ============================================================================
-- Foreign keys: tenant-scoped tables → users(id)
-- ON DELETE RESTRICT so a user cannot be deleted while their data exists.
-- The IF NOT EXISTS logic via information_schema makes this idempotent.
-- ============================================================================

DO $$
DECLARE
    t TEXT;
    fk_name TEXT;
    tenant_tables TEXT[] := ARRAY[
        'portfolios', 'trades_summary', 'trades_details', 'trading_journal',
        'audit_trail', 'trade_images', 'trade_fundamentals', 'drawdown_notes',
        'trade_lessons', 'rule_notes', 'app_config', 'dashboard_events'
    ];
BEGIN
    FOREACH t IN ARRAY tenant_tables LOOP
        fk_name := 'fk_' || t || '_user';
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = fk_name
        ) THEN
            EXECUTE format(
                'ALTER TABLE %I ADD CONSTRAINT %I '
                'FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE RESTRICT',
                t, fk_name
            );
        END IF;
    END LOOP;
END $$;


-- ============================================================================
-- VERIFICATION (after COMMIT)
-- ============================================================================
--   SELECT id, email, name FROM users;
--   -- expected: exactly 1 row with id = d7e8f9a0-...
--
--   SELECT conname, conrelid::regclass AS table_name
--   FROM pg_constraint
--   WHERE contype = 'f' AND confrelid = 'users'::regclass
--   ORDER BY conrelid::regclass::text;
--   -- expected: 12 FK rows, one per tenant table
