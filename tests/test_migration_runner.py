"""Unit tests for the migration runner's failure surface and self-attribution.

These are shape tests, not integration tests — they mock psycopg2 connection
and cursor so we can assert what the runner DOES (SETs app.user_id, prints
the FAILED banner, re-raises) without needing a live Postgres.

Group 7-4: hardens migrations/run.py against the Group 7-3 silent-skip mode
where a failing migration printed only a Python traceback amid 'applied'
lines. The runner must now:
  - SET LOCAL app.user_id to the founder UUID per migration (so trigger-
    fired audit_trail inserts can resolve user_id via DEFAULT).
  - Print an explicit '✗ FAILED: <file> — <error>' banner to stderr before
    re-raising, so failures are visible to deploy-log scanners.
  - Continue to re-raise the underlying exception so the runner exits non-
    zero and Railway's release step aborts the deploy.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# Load migrations/run.py as a module without importing through the package
# system (the file lives next to .sql files, no __init__.py).
_RUN_PATH = Path(__file__).resolve().parent.parent / "migrations" / "run.py"
_spec = importlib.util.spec_from_file_location("migration_runner", _RUN_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


FOUNDER_UUID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"


def _make_conn():
    """Build a mock psycopg2 connection whose cursor() is a context manager."""
    conn = MagicMock()
    cur = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = cur
    cm.__exit__.return_value = False
    conn.cursor.return_value = cm
    return conn, cur


def _fake_migration(tmp_path: Path, name: str, body: str = "SELECT 1;") -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


class TestApplyOneSuccess:
    def test_sets_app_user_id_before_executing_migration(self, tmp_path):
        """SET LOCAL app.user_id must run BEFORE the migration SQL so any
        trigger that fires during the migration sees the founder UUID."""
        conn, cur = _make_conn()
        path = _fake_migration(tmp_path, "999_test.sql", "SELECT 1;")

        runner.apply_one(conn, path)

        # First execute: SET LOCAL app.user_id = <founder>
        first_call = cur.execute.call_args_list[0]
        assert "SET LOCAL app.user_id" in first_call.args[0]
        assert first_call.args[1] == (FOUNDER_UUID,)

    def test_executes_sql_and_records_in_schema_migrations(self, tmp_path):
        conn, cur = _make_conn()
        path = _fake_migration(tmp_path, "999_test.sql", "SELECT 42;")

        runner.apply_one(conn, path)

        # Calls: SET LOCAL, migration SQL, INSERT INTO schema_migrations.
        assert cur.execute.call_count == 3
        assert cur.execute.call_args_list[1].args[0] == "SELECT 42;"
        insert_call = cur.execute.call_args_list[2]
        assert "INSERT INTO schema_migrations" in insert_call.args[0]
        assert insert_call.args[1] == ("999_test.sql",)
        conn.commit.assert_called_once()
        conn.rollback.assert_not_called()


class TestApplyOneFailure:
    def test_prints_failed_banner_to_stderr(self, tmp_path, capsys):
        """On exception, the runner must print '✗ FAILED: <file> — <error>'
        to stderr so deploy-log scanners see it amid the 'applied' lines."""
        conn, cur = _make_conn()
        cur.execute.side_effect = [None, RuntimeError("boom"), None]
        path = _fake_migration(tmp_path, "999_test.sql")

        with pytest.raises(RuntimeError, match="boom"):
            runner.apply_one(conn, path)

        captured = capsys.readouterr()
        assert "FAILED: 999_test.sql" in captured.err
        assert "RuntimeError" in captured.err
        assert "boom" in captured.err

    def test_rolls_back_on_failure(self, tmp_path):
        conn, cur = _make_conn()
        cur.execute.side_effect = [None, RuntimeError("boom"), None]
        path = _fake_migration(tmp_path, "999_test.sql")

        with pytest.raises(RuntimeError):
            runner.apply_one(conn, path)

        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()

    def test_reraises_underlying_exception(self, tmp_path):
        """The runner must NOT swallow — Railway's release step relies on a
        non-zero exit code to abort the deploy."""
        conn, cur = _make_conn()
        cur.execute.side_effect = [None, ValueError("specific error"), None]
        path = _fake_migration(tmp_path, "999_test.sql")

        with pytest.raises(ValueError, match="specific error"):
            runner.apply_one(conn, path)


class TestFounderUuidConstant:
    def test_matches_migration_024_literal(self):
        """The constant must equal the literal in migrations/024 — they're
        both load-bearing for the same attribution behaviour."""
        assert runner._FOUNDER_USER_ID == FOUNDER_UUID
        m024 = (
            Path(__file__).resolve().parent.parent
            / "migrations"
            / "024_audit_trigger_migration_safe.sql"
        )
        assert FOUNDER_UUID in m024.read_text()
