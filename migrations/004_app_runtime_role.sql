-- ============================================================================
-- Migration 004: app_runtime role — makes migration 003's RLS actually enforce
-- ============================================================================
-- Migration 003 enabled + forced RLS on all 12 tenant tables, but Neon's
-- primary role (neondb_owner) ships with the `BYPASSRLS` attribute which
-- silently short-circuits every policy. Neon doesn't allow neondb_owner to
-- strip that attribute from itself (migration 004's earlier attempt failed
-- with "permission denied to alter role").
--
-- Work-around: create a second role `app_runtime` with NOBYPASSRLS, grant it
-- CRUD on the schema, and `GRANT app_runtime TO neondb_owner` so the owner
-- can `SET ROLE app_runtime` on demand. In db_layer.get_db_connection() the
-- app now runs:
--      SET app.user_id = <uuid>;
--      SET ROLE app_runtime;
-- Once ROLE is set, Postgres checks RLS like any ordinary user — policies
-- kick in, cross-tenant reads return zero rows, INSERTs auto-tag user_id.
--
-- Migrations continue to run as neondb_owner (SET ROLE isn't applied there),
-- so DDL and owner-level admin still work. When we introduce a proper
-- managed-role setup in Tier 2, this bridge is easy to replace.
--
-- Idempotent: CREATE ROLE IF NOT EXISTS is not a thing in Postgres, so we
-- wrap in a DO block with pg_roles check.
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_runtime') THEN
        CREATE ROLE app_runtime NOINHERIT NOLOGIN NOBYPASSRLS;
    END IF;
END $$;

-- Let neondb_owner switch into app_runtime without a password.
GRANT app_runtime TO neondb_owner;

-- app_runtime needs the usual CRUD surface on the public schema.
GRANT USAGE ON SCHEMA public TO app_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES    IN SCHEMA public TO app_runtime;
GRANT USAGE, SELECT                 ON ALL SEQUENCES IN SCHEMA public TO app_runtime;

-- Future tables and sequences auto-inherit the same grants.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES    TO app_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT                 ON SEQUENCES TO app_runtime;
