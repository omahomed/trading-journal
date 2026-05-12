#!/usr/bin/env python3
"""
Migration runner.

Applies any `NNN_*.sql` file in this directory that has not yet been recorded
in the `schema_migrations` tracking table, in filename-sorted order. Each
migration runs in its own transaction together with the tracking insert, so a
partial failure cannot leave the database half-migrated.

Usage:
    python migrations/run.py

Connection:
    1. DATABASE_URL environment variable (preferred)
    2. `.streamlit/secrets.toml` → [database] url (local fallback)
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


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    if SECRETS_PATH.exists():
        with open(SECRETS_PATH, "rb") as f:
            return tomllib.load(f)["database"]["url"]
    raise RuntimeError(
        "DATABASE_URL not set and .streamlit/secrets.toml not found. "
        "Export DATABASE_URL or populate the secrets file before running."
    )


def mask_url(url: str) -> str:
    return url.split("@", 1)[-1] if "@" in url else url


def ensure_tracking_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
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
    db_url = get_database_url()
    print(f"Connecting to {mask_url(db_url)} ...")

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
