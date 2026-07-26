#!/usr/bin/env python3
"""
Migration runner.

Applies any `NNN_*.sql` file in this directory that has not yet been recorded
in the `schema_migrations` tracking table, in filename-sorted order. Each
migration runs in its own transaction together with the tracking insert, so a
partial failure cannot leave the database half-migrated.

Usage:
    python migrations/run.py

Connection (checked in order):
    1. MIGRATIONS_DATABASE_URL — dedicated role with CREATE on the public
       schema (e.g. neondb_owner). Preferred on Railway so the runtime
       DATABASE_URL can stay locked to a USAGE-only role like app_runtime.
    2. DATABASE_URL — falls back here when no separate migration URL is
       set. Fine locally where the same role does both.
    3. `.streamlit/secrets.toml` → [database] url (local dev fallback).

If MIGRATIONS_DATABASE_URL is set but connects as a role without CREATE
privilege on the public schema, ensure_tracking_table() will fail on the
first deploy — see [migrations/README.md] for role/grant setup.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

import psycopg2

MIGRATIONS_DIR = Path(__file__).resolve().parent
REPO_ROOT = MIGRATIONS_DIR.parent
SECRETS_PATH = REPO_ROOT / ".streamlit" / "secrets.toml"


def get_database_url() -> tuple[str, str]:
    """Return (url, source_label). Source label surfaces which env var
    supplied the URL so the deploy log makes it obvious when we're on
    the split-role setup vs. the shared-role fallback."""
    url = os.environ.get("MIGRATIONS_DATABASE_URL")
    if url:
        return url, "MIGRATIONS_DATABASE_URL"
    url = os.environ.get("DATABASE_URL")
    if url:
        return url, "DATABASE_URL"
    if SECRETS_PATH.exists():
        with open(SECRETS_PATH, "rb") as f:
            return tomllib.load(f)["database"]["url"], "secrets.toml"
    raise RuntimeError(
        "MIGRATIONS_DATABASE_URL / DATABASE_URL not set and "
        ".streamlit/secrets.toml not found. Export one of the URLs "
        "or populate the secrets file before running."
    )


def mask_url(url: str) -> str:
    return url.split("@", 1)[-1] if "@" in url else url


def ensure_tracking_table(conn) -> None:
    """Ensure schema_migrations exists, without needing CREATE on schema
    public when it already does.

    Postgres checks CREATE privilege on the target schema for
    `CREATE TABLE IF NOT EXISTS` *before* it notices the table already
    exists — so the bare IF NOT EXISTS raises "permission denied for schema
    public" for a role that can run DML but lacks CREATE (e.g. the pooled
    Railway connection). We probe with to_regclass first (needs only
    USAGE + SELECT, which such a role has) and only issue CREATE when the
    table is genuinely absent. Fresh databases still get the table; existing
    ones (all migrations already applied) sail past without a privilege wall.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.schema_migrations')")
        if cur.fetchone()[0] is None:
            cur.execute(
                """
                CREATE TABLE schema_migrations (
                    filename    VARCHAR(255) PRIMARY KEY,
                    applied_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
                )
                """
            )
    conn.commit()


def already_applied(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def pending_files(applied: set[str]) -> list[Path]:
    return sorted(
        p for p in MIGRATIONS_DIR.glob("*.sql") if p.name not in applied
    )


# Founder UUID — single-tenant attribution for audit_trail rows written by
# triggers that fire during a migration. Mirrors the literal in migration 024.
# When Tier 2 multi-tenancy lands, replace with a dedicated 'system' user.
_FOUNDER_USER_ID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"


def apply_one(conn, path: Path) -> None:
    sql = path.read_text()
    try:
        with conn.cursor() as cur:
            # Migration self-attribution: causes audit_trail.user_id DEFAULT
            # to resolve to founder UUID when triggers fire during the
            # migration. SET LOCAL scopes to this transaction only.
            cur.execute("SET LOCAL app.user_id = %s", (_FOUNDER_USER_ID,))
            cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations (filename) VALUES (%s)",
                (path.name,),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(
            f"  ✗ FAILED: {path.name} — {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        raise


def main() -> int:
    db_url, source = get_database_url()
    print(f"Connecting to {mask_url(db_url)} (from {source}) ...")

    conn = psycopg2.connect(db_url)
    try:
        ensure_tracking_table(conn)
        applied = already_applied(conn)
        pending = pending_files(applied)

        if not pending:
            print("No pending migrations.")
            return 0

        print(f"Pending ({len(pending)}):")
        for p in pending:
            print(f"  - {p.name}")

        for path in pending:
            print(f"\n→ {path.name}")
            apply_one(conn, path)
            print(f"  applied")

        print(
            f"\nDone. {len(pending)} applied: "
            f"{', '.join(p.name for p in pending)}."
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
