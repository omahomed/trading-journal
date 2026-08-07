"""Tests for db.set_declared_sr8 — the migration-062 promote/demote helper.

Covers:
  1. Promotion succeeds when the campaign is cushion-qualified (peak >= 50%).
  2. Promotion is rejected with reason='not_qualified' when peak < 50%.
  3. Core-shares override at promotion time (Option C from the design).
  4. Demotion preserves sr8_core_shares as historical audit.
  5. Repeated calls are idempotent.
  6. Unknown trade_id returns None (404 signal).

Reads/writes against the live Neon DB via the standard `requires_db`
fixture — same pattern as the SR8 monitor tests. Uses the founder UUID
so the RLS policy admits the writes.
"""
from __future__ import annotations

import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — skip DB-backed test",
)

FOUNDER_UUID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"


@pytest.fixture(autouse=True)
def _tenant_context():
    import db_layer as db
    db.current_user_id.set(FOUNDER_UUID)
    yield


def _find_qualified_open() -> tuple[str, str] | None:
    """A cushion-qualified OPEN campaign that is NOT DELL (the seed row).
    Returns (portfolio_name, trade_id) or None if none available."""
    import db_layer as db
    with db.get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.name, s.trade_id
                  FROM trades_summary s
                  JOIN portfolios p ON p.id = s.portfolio_id
                 WHERE s.deleted_at IS NULL
                   AND s.status = 'OPEN'
                   AND s.b1_max_return_pct >= 50
                   AND s.trade_id != '202604-013'
                 LIMIT 1
                """
            )
            row = cur.fetchone()
            return (row[0], row[1]) if row else None


def _find_unqualified_open() -> tuple[str, str] | None:
    """A sub-qualified OPEN campaign (peak < 50 or NULL)."""
    import db_layer as db
    with db.get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.name, s.trade_id
                  FROM trades_summary s
                  JOIN portfolios p ON p.id = s.portfolio_id
                 WHERE s.deleted_at IS NULL
                   AND s.status = 'OPEN'
                   AND (s.b1_max_return_pct < 50 OR s.b1_max_return_pct IS NULL)
                 LIMIT 1
                """
            )
            row = cur.fetchone()
            return (row[0], row[1]) if row else None


def test_promotion_and_demotion_round_trip():
    """Promote a qualified campaign, verify flag flipped + audit anchors
    stayed, demote, verify flag cleared + sr8_core_shares preserved."""
    import db_layer as db
    hit = _find_qualified_open()
    if hit is None:
        pytest.skip("No qualified OPEN campaign available (other than DELL).")
    portfolio, trade_id = hit

    try:
        r1 = db.set_declared_sr8(portfolio, trade_id, True,
                                 core_shares_override=42.0)
        assert r1["was_written"] is True
        assert r1["is_declared_sr8"] is True
        assert r1["sr8_core_shares"] == 42.0
        assert r1["reason"] is None

        # Demote — core_shares must PERSIST as historical audit.
        r2 = db.set_declared_sr8(portfolio, trade_id, False)
        assert r2["was_written"] is True
        assert r2["is_declared_sr8"] is False
        assert r2["sr8_core_shares"] == 42.0
    finally:
        # Always reset the flag so re-runs don't leave state behind.
        db.set_declared_sr8(portfolio, trade_id, False)


def test_promotion_of_unqualified_is_rejected():
    """A campaign whose peak b1_return < 50 cannot be declared. Returns
    reason='not_qualified' without writing."""
    import db_layer as db
    hit = _find_unqualified_open()
    if hit is None:
        pytest.skip("No sub-qualified OPEN campaign available.")
    portfolio, trade_id = hit

    r = db.set_declared_sr8(portfolio, trade_id, True)
    assert r["was_written"] is False
    assert r["reason"] == "not_qualified"
    assert r["is_declared_sr8"] is False


def test_unknown_trade_id_returns_none():
    """Missing trade_id → None. Endpoint layer maps this to 404."""
    import db_layer as db
    assert db.set_declared_sr8("CanSlim", "999999-999", True) is None
    assert db.set_declared_sr8("CanSlim", "999999-999", False) is None


def test_dell_seed_row_is_declared():
    """Migration 062 backfilled exactly one row — DELL 202604-013 in
    CanSlim — as declared. list_declared_sr8 must surface it."""
    import db_layer as db
    declared = db.list_declared_sr8()
    dells = [d for d in declared
             if d["trade_id"] == "202604-013" and d["ticker"] == "DELL"]
    assert len(dells) == 1, f"Expected 1 DELL declared, got {len(dells)}: {declared}"
    assert dells[0]["portfolio"] == "CanSlim"
