"""Regression: /api/journal/latest must return the last row with a REAL
end_nlv, not just the chronologically latest row.

Before this filter, a same-day journal row saved with just a checklist /
notes / recap (no NLV logged yet) would come back as the "latest," its
end_nlv would be None, and every downstream consumer (Portfolio Heat's
equity basis, Position Sizer's account-equity prefill, Log Buy's read-
only Account Equity display + submit validation, Trade Journal's POS
SIZE %, NLV Entry's `prev_end_nlv` baseline for the daily-change diff)
would either fall back to a fake $100k default or blank out entirely.

The correct behavior: journal_latest walks back to the most recent row
whose end_nlv is finite and > 0, so "latest available NLV" always means
the true last real NLV — regardless of any newer NLV-less rows the
trader may have opened for the day.
"""
from __future__ import annotations

from typing import Any

import jwt
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import db_layer


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(monkeypatch):
    """Patch db.load_journal so we can inject arbitrary journal history
    without touching Postgres. Same shape assumption as the snapshot-
    semantics test: bypass column normalization, feed snake_case rows."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {"journal_df": pd.DataFrame()}
    monkeypatch.setattr(db_layer, "load_journal",
                        lambda *a, **kw: state["journal_df"])
    monkeypatch.setattr(main, "_normalize_journal", lambda df: df)

    tc = TestClient(main.app, headers=_auth_headers())

    def set_history(rows: list[dict]):
        state["journal_df"] = pd.DataFrame(rows)
    tc.set_history = set_history  # type: ignore[attr-defined]
    return tc


def _row(day: str, end_nlv, **extras):
    return {"day": pd.Timestamp(day), "end_nlv": end_nlv, **extras}


def test_latest_row_with_nlv_wins_over_newer_row_without_nlv(client):
    """The regression itself: two rows exist, the newer one has no NLV
    (e.g. today's row saved with just a checklist / recap), the older
    one has $52,450. Endpoint must return the older row so downstream
    consumers see a real equity basis."""
    client.set_history([  # type: ignore[attr-defined]
        _row("2026-07-25", 52450.0),
        _row("2026-07-27", None),  # today's checklist-only row
    ])
    resp = client.get("/api/journal/latest?portfolio=CanSlim")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("end_nlv") == 52450.0
    assert body.get("day") == "2026-07-25"


def test_latest_row_with_nlv_wins_over_newer_row_with_zero_nlv(client):
    """A same-day row saved with end_nlv=0 (rare — usually a data
    accident) shouldn't count either; walk back to the last positive
    NLV. Zero is not a real equity basis."""
    client.set_history([  # type: ignore[attr-defined]
        _row("2026-07-25", 52450.0),
        _row("2026-07-27", 0.0),
    ])
    body = client.get("/api/journal/latest?portfolio=CanSlim").json()
    assert body.get("end_nlv") == 52450.0


def test_returns_the_actual_latest_when_it_has_a_valid_nlv(client):
    """Non-regression case — when the latest row has a real NLV, that
    row wins as expected."""
    client.set_history([  # type: ignore[attr-defined]
        _row("2026-07-25", 52450.0),
        _row("2026-07-27", 53100.0),
    ])
    body = client.get("/api/journal/latest?portfolio=CanSlim").json()
    assert body.get("end_nlv") == 53100.0
    assert body.get("day") == "2026-07-27"


def test_returns_error_when_no_row_has_nlv(client):
    """If the entire journal exists but every row's end_nlv is null,
    that IS the empty-history case — endpoint returns the same shape
    as no-journal-at-all so frontend can render one empty state.
    (This is what fires the Portfolio Heat "No NLV history" bail.)"""
    client.set_history([  # type: ignore[attr-defined]
        _row("2026-07-25", None),
        _row("2026-07-27", None),
    ])
    body = client.get("/api/journal/latest?portfolio=CanSlim").json()
    assert body == {"error": "No journal data"}


def test_before_param_still_walks_back_past_nlv_less_row(client):
    """The `before` cutoff and the NLV filter compose: NLV Entry's
    baseline-lookup for editing 2026-07-27's row must skip both the
    edited row itself AND any earlier NLV-less rows to find the true
    prior baseline."""
    client.set_history([  # type: ignore[attr-defined]
        _row("2026-07-20", 51900.0),
        _row("2026-07-25", None),  # skipped by NLV filter
        _row("2026-07-27", 53100.0),  # skipped by `before` cutoff
    ])
    body = client.get(
        "/api/journal/latest?portfolio=CanSlim&before=2026-07-27"
    ).json()
    assert body.get("end_nlv") == 51900.0
    assert body.get("day") == "2026-07-20"


def test_returns_error_when_journal_totally_empty(client):
    """Baseline sanity check: no rows at all → the classic no-data
    response. This is unchanged behavior; locking it so a future
    refactor doesn't accidentally return a 500 or a partial row."""
    client.set_history([])  # type: ignore[attr-defined]
    body = client.get("/api/journal/latest?portfolio=CanSlim").json()
    assert body == {"error": "No journal data"}
