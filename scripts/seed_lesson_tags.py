#!/usr/bin/env python3
"""One-time seed: create LESSON_CATEGORIES as tags in a user's portfolios.

Populates the shared tag catalog with the 14 lesson categories the app
already ships (frontend/src/lib/lesson-categories.ts::LESSON_CATEGORIES)
so the new Weekly Ledger's per-row lesson picker has them available in
autocomplete from day 1. The user can still add new lessons freely
(they're just tags) or delete any they don't use.

Tags are portfolio-scoped, so we insert one row per (portfolio, name).
Idempotent: skips a name that already exists in that portfolio's palette
(case-insensitive collision via the existing UNIQUE index).

Usage:
    # Seed the founder user across every portfolio they own (default):
    python scripts/seed_lesson_tags.py

    # Seed a specific portfolio only:
    python scripts/seed_lesson_tags.py --portfolio "Long-Term Growth"

    # Dry-run (default: writes). Add --dry-run to preview.
    python scripts/seed_lesson_tags.py --dry-run

Color mapping (uses the closed tag palette — rose/amber/emerald/sky/violet):
    emerald  : Followed Rules
    amber    : Entry timing, FOMO, Chased Entry, Bought Too Early,
               Scaled in too fast
    rose     : Exit too early, Exit too late, Rule deviation
    sky      : Market conditions, Portfolio Management, Undersized,
               Oversized, Other
    violet   : Stop placement
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import db_layer as db  # noqa: E402
import psycopg2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("seed_lesson_tags")


# Mirrors frontend/src/lib/lesson-categories.ts::LESSON_CATEGORIES and its
# semantic color grouping. Keep in sync with the LESSON_CATEGORIES catalog.
LESSON_TAG_SEEDS: list[tuple[str, str]] = [
    ("Followed Rules",     "emerald"),
    ("Entry timing",       "amber"),
    ("FOMO",               "amber"),
    ("Chased Entry",       "amber"),
    ("Bought Too Early",   "amber"),
    ("Stop placement",     "violet"),
    ("Undersized",         "sky"),
    ("Oversized",          "sky"),
    ("Scaled in too fast", "amber"),
    ("Exit too early",     "rose"),
    ("Exit too late",      "rose"),
    ("Market conditions",  "sky"),
    ("Portfolio Management", "sky"),
    ("Rule deviation",     "rose"),
    ("Other",              "sky"),
]

FOUNDER_UUID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"


def seed_one_portfolio(portfolio_name: str, dry_run: bool) -> tuple[int, int]:
    """Insert every lesson tag into `portfolio_name`. Returns
    (inserted, skipped) counts. Skips names that already exist in the
    portfolio's palette (case-insensitive)."""
    inserted, skipped = 0, 0
    for name, color in LESSON_TAG_SEEDS:
        if dry_run:
            log.info(f"  [dry-run] would create '{name}' ({color}) in {portfolio_name}")
            inserted += 1
            continue
        try:
            db.create_tag(portfolio_name, name, color)
            log.info(f"  + created '{name}' ({color}) in {portfolio_name}")
            inserted += 1
        except psycopg2.errors.UniqueViolation:
            log.info(f"    · '{name}' already exists in {portfolio_name} — skipped")
            skipped += 1
    return inserted, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--portfolio", action="append",
                    help="Seed a specific portfolio (repeatable). "
                         "Default: seed every portfolio the user owns.")
    ap.add_argument("--user-id", default=FOUNDER_UUID,
                    help=f"UUID of the user to seed against. Default: founder.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview without writing.")
    args = ap.parse_args()

    db.current_user_id.set(args.user_id)

    # Resolve target portfolios. Falls back to "every portfolio the user
    # currently owns" when --portfolio isn't specified; that's the common
    # first-run case and matches how the seed is expected to be used.
    if args.portfolio:
        portfolios = [p for p in args.portfolio]
    else:
        with db.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT name FROM portfolios "
                    " WHERE user_id = %s ORDER BY name",
                    (args.user_id,),
                )
                portfolios = [r[0] for r in cur.fetchall()]

    if not portfolios:
        log.error(f"No portfolios found for user {args.user_id}")
        return 1

    log.info("=" * 60)
    log.info(f"LESSON TAG SEED  ({'DRY-RUN' if args.dry_run else 'COMMIT'})")
    log.info(f"User: {args.user_id}")
    log.info(f"Portfolios: {', '.join(portfolios)}")
    log.info("=" * 60)

    total_inserted, total_skipped = 0, 0
    for pname in portfolios:
        log.info(f"\n>> {pname}")
        try:
            i, s = seed_one_portfolio(pname, args.dry_run)
            total_inserted += i
            total_skipped += s
        except ValueError as e:
            log.error(f"   FAILED {pname}: {e}")

    log.info("=" * 60)
    log.info(f"Total: {total_inserted} inserted, {total_skipped} already existed")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
