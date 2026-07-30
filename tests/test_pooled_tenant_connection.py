"""Guards `_PooledTenantConnection` — the wrapper that re-applies
`SET app.user_id` + `SET ROLE app_runtime` after every commit/rollback so
Neon's transaction-mode pooler can't drop tenant context between
transactions.

Regression prevention for the 2026-07-30 sync bug where the same script
inserted ARM successfully and then NOT NULL-violated on NBIS because the
follow-up transaction landed on a pool-swapped backend with no
`app.user_id` set. See docstring on `_PooledTenantConnection` in
db_layer.py for the full story.
"""
from __future__ import annotations
from unittest.mock import MagicMock, call

import db_layer


def _mock_conn():
    conn = MagicMock()
    cur_cm = MagicMock()
    cur = MagicMock()
    cur_cm.__enter__ = MagicMock(return_value=cur)
    cur_cm.__exit__ = MagicMock(return_value=False)
    conn.cursor = MagicMock(return_value=cur_cm)
    return conn, cur


def test_apply_runs_on_construction():
    conn, cur = _mock_conn()
    db_layer._PooledTenantConnection(conn, "abc-uuid")
    cur.execute.assert_any_call("SET app.user_id = %s", ("abc-uuid",))
    cur.execute.assert_any_call("SET ROLE app_runtime")


def test_commit_reapplies_session_vars():
    conn, cur = _mock_conn()
    w = db_layer._PooledTenantConnection(conn, "u1")
    cur.execute.reset_mock()
    conn.commit.reset_mock()

    w.commit()

    # commit fires exactly once on the underlying conn
    conn.commit.assert_called_once_with()
    # and the SETs re-run afterwards for the next transaction
    assert cur.execute.call_args_list == [
        call("SET app.user_id = %s", ("u1",)),
        call("SET ROLE app_runtime"),
    ]


def test_rollback_reapplies_session_vars():
    conn, cur = _mock_conn()
    w = db_layer._PooledTenantConnection(conn, "u1")
    cur.execute.reset_mock()
    conn.rollback.reset_mock()

    w.rollback()

    conn.rollback.assert_called_once_with()
    assert cur.execute.call_args_list == [
        call("SET app.user_id = %s", ("u1",)),
        call("SET ROLE app_runtime"),
    ]


def test_repeated_commits_each_reapply():
    """The bug fires when many transactions run back-to-back — every one
    of them needs the SETs, not just the first."""
    conn, cur = _mock_conn()
    w = db_layer._PooledTenantConnection(conn, "u1")
    cur.execute.reset_mock()

    for _ in range(5):
        w.commit()

    # Each commit re-runs the two SETs → 5 * 2 = 10 execute calls
    assert cur.execute.call_count == 10
    assert conn.commit.call_count == 5


def test_attributes_pass_through_to_underlying():
    conn, _cur = _mock_conn()
    conn.autocommit = False
    conn.notices = []
    w = db_layer._PooledTenantConnection(conn, "u1")
    assert w.autocommit is False
    assert w.notices == []
    assert w.cursor is conn.cursor


def test_no_wrapping_when_uid_absent():
    """Migrations run without a tenant context; get_db_connection must
    hand them the raw connection so `SET ROLE app_runtime` doesn't fire
    (they run as neondb_owner BYPASSRLS on purpose)."""
    # Sanity: the wrapper class is only meaningful when uid is set.
    # get_db_connection() branches on `uid` — we're not going to spin up
    # a real DB in unit tests, but we can lock the branch by grepping the
    # source. That's fragile; the smoke test in test_smoke_pooler.py
    # covers the live behavior.
    import inspect
    src = inspect.getsource(db_layer.get_db_connection)
    assert "_PooledTenantConnection(raw_conn, uid) if uid else raw_conn" in src, (
        "get_db_connection must skip the wrapper when no tenant uid is set"
    )
