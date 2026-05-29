"""Validator coverage for migrate_csv_to_postgres.import_details.

The CSV import path used to accept any string in the Trx_ID column,
which is how production accumulated ~10 rows with free-form text like
"Market selloff and reducing exposure". Strict-mode now rejects
malformed trx_ids at the import boundary against the canonical
db_layer.TRX_ID_PATTERN.

Empty Trx_ID is intentionally still allowed — legacy data pre-dates
the field and a blank cell shouldn't fail the migration.

These tests stub the DB connection (psycopg2 cursor) so the validator
fires before any SQL runs. We assert on the ValueError raised + the
error-message contents that an operator would use to find the bad
CSV row.
"""
from __future__ import annotations

import io
from contextlib import contextmanager
from unittest import mock

import pandas as pd
import pytest

from migrate_csv_to_postgres import import_details


class _FakeCursor:
    """Minimal psycopg2-like cursor: returns a portfolio_id on the
    `SELECT id FROM portfolios` lookup, records executions otherwise."""

    def __init__(self, portfolio_id: int = 1) -> None:
        self.portfolio_id = portfolio_id
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params=()) -> None:
        self.executed.append((sql, params))

    def fetchone(self):
        return (self.portfolio_id,)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    """psycopg2-connection stand-in with a cursor() context manager."""

    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True


@pytest.fixture
def _patched_csv(monkeypatch):
    """Patch pd.read_csv to return the dataframe passed to the fixture.

    Yields a function `set_df(df)` that the test calls to install the
    DataFrame the import will see.
    """
    container: dict[str, pd.DataFrame] = {}

    def fake_read_csv(_path, *a, **kw) -> pd.DataFrame:
        return container["df"]

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    # The import also calls execute_values — stub it to a no-op so we don't
    # depend on a real DB session.
    monkeypatch.setattr(
        "migrate_csv_to_postgres.execute_values",
        lambda cur, sql, rows: None,
    )

    # The os.path.exists guard short-circuits if the path doesn't exist;
    # always say it does so the function reaches the validation loop.
    monkeypatch.setattr("os.path.exists", lambda _p: True)

    def set_df(df: pd.DataFrame) -> None:
        container["df"] = df

    return set_df


def _detail_csv(trx_id: str, trade_id: str = "202605-001") -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Trade_ID": trade_id, "Ticker": "AAPL", "Action": "BUY",
            "Date": "2026-05-01", "Shares": 100, "Amount": 200, "Value": 20000,
            "Rule": "br1.1", "Notes": "", "Realized_PL": 0, "Stop_Loss": 195,
            "Trx_ID": trx_id, "Exec_Grade": "", "Behavior_Tag": "",
            "Retro_Notes": "",
        }
    ])


# ---------------------------------------------------------------------------
# Negative: malformed trx_id raises ValueError pointing at the CSV row
# ---------------------------------------------------------------------------


def test_freeform_text_in_trx_id_raises(_patched_csv):
    """The free-form-text corruption pattern (e.g. 'Market selloff...')
    that already polluted ~10 production rows is rejected at import."""
    _patched_csv(_detail_csv("Market selloff and reducing exposure"))
    conn = _FakeConn()
    with pytest.raises(ValueError) as excinfo:
        import_details(conn, "CanSlim", "/fake/path/details.csv")
    msg = str(excinfo.value)
    assert "Market selloff and reducing exposure" in msg
    assert "row 2" in msg               # human-readable CSV line number
    assert "Trade_ID=202605-001" in msg # row identification for the operator
    assert not conn.committed           # no partial commit


def test_garbage_trx_id_raises(_patched_csv):
    """A trx_id that's vaguely formatted but doesn't match the pattern
    (e.g. typo 'XY42') still raises."""
    _patched_csv(_detail_csv("XY42"))
    conn = _FakeConn()
    with pytest.raises(ValueError) as excinfo:
        import_details(conn, "CanSlim", "/fake/path/details.csv")
    assert "XY42" in str(excinfo.value)


def test_leading_zero_trx_id_raises(_patched_csv):
    """B01 / A02 / S0 are rejected — generator counts from 1 and the
    regex tightens to [1-9]\\d* to match."""
    _patched_csv(_detail_csv("B01"))
    conn = _FakeConn()
    with pytest.raises(ValueError):
        import_details(conn, "CanSlim", "/fake/path/details.csv")


# ---------------------------------------------------------------------------
# Positive: well-formed variants pass the validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trx_id", [
    "B1", "A1", "S1", "SA1", "SB1",
    "B42", "A100", "S99",
    "B0-Pre",
    "A4-2", "B1-2", "SA3-5",        # dedupe survivors
    "B1-Auto", "S1-Auto", "SB7-Auto",  # IBKR markers
])
def test_well_formed_trx_id_passes(_patched_csv, trx_id):
    """Every variant currently present in production data validates."""
    _patched_csv(_detail_csv(trx_id))
    conn = _FakeConn()
    # Should not raise.
    import_details(conn, "CanSlim", "/fake/path/details.csv")
    assert conn.committed


def test_empty_trx_id_passes(_patched_csv):
    """Empty Trx_ID is intentionally allowed (legacy data pre-dates the
    field). Validator only fires on a non-empty cell."""
    _patched_csv(_detail_csv(""))
    conn = _FakeConn()
    import_details(conn, "CanSlim", "/fake/path/details.csv")
    assert conn.committed


# ---------------------------------------------------------------------------
# Error message ergonomics — the operator needs row number + value
# ---------------------------------------------------------------------------


def test_error_message_points_at_specific_row(_patched_csv):
    """A multi-row CSV with the bad row at the third data row reports
    'row 4' (header = row 1, three data rows = rows 2..4)."""
    df = pd.concat([
        _detail_csv("B1", trade_id="202605-001"),
        _detail_csv("A1", trade_id="202605-001"),
        _detail_csv("not a valid trx_id", trade_id="202605-002"),
    ], ignore_index=True)
    _patched_csv(df)
    conn = _FakeConn()
    with pytest.raises(ValueError) as excinfo:
        import_details(conn, "CanSlim", "/fake/path/details.csv")
    msg = str(excinfo.value)
    assert "row 4" in msg
    assert "Trade_ID=202605-002" in msg
    assert "not a valid trx_id" in msg
