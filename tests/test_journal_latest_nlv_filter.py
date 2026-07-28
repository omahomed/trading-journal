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
    # beg_nlv + cash_change are what journal_history reads to build the
    # adjusted_beg / daily_return / twr_curve chain. Default them to 0
    # so tests don't have to specify every column just to hit the filter.
    base = {
        "day": pd.Timestamp(day),
        "end_nlv": end_nlv,
        "beg_nlv": end_nlv if end_nlv is not None else 0,
        "cash_change": 0,
    }
    base.update(extras)
    return base


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


# ---------------------------------------------------------------------------
# /api/journal/history — same NLV-bearing filter, applied to the list view
# ---------------------------------------------------------------------------

def test_history_hides_hollow_game_plan_rows(client):
    """Regression: Journal Log shouldn't show a phantom row for today
    when the only writer that touched today's journal was
    save_journal_game_plan (draft auto-save with no NLV logged yet).
    Backend filters at /api/journal/history so the frontend doesn't
    have to re-implement the "is this a real logged day" check."""
    client.set_history([  # type: ignore[attr-defined]
        _row("2026-07-25", 52450.0),
        _row("2026-07-27", None),  # hollow Game Plan row
    ])
    body = client.get("/api/journal/history?portfolio=CanSlim&days=0").json()
    assert isinstance(body, list)
    days = [str(r["day"])[:10] for r in body]
    assert "2026-07-27" not in days
    assert "2026-07-25" in days


def test_history_hides_zero_nlv_rows(client):
    """Zero-NLV rows are also treated as unlogged — same rule as the
    latest endpoint, so an accidental $0 save doesn't sneak into the
    historical browse."""
    client.set_history([  # type: ignore[attr-defined]
        _row("2026-07-25", 52450.0),
        _row("2026-07-27", 0.0),
    ])
    body = client.get("/api/journal/history?portfolio=CanSlim&days=0").json()
    days = [str(r["day"])[:10] for r in body]
    assert "2026-07-27" not in days


def test_history_returns_empty_when_no_row_has_nlv(client):
    """All rows hollow (draft-only portfolio) → empty list, not a mix
    of $0 rows the frontend has to guess-filter."""
    client.set_history([  # type: ignore[attr-defined]
        _row("2026-07-25", None),
        _row("2026-07-27", None),
    ])
    body = client.get("/api/journal/history?portfolio=CanSlim&days=0").json()
    assert body == []


# ---------------------------------------------------------------------------
# /api/journal/entry — INVERSE contract: unfiltered single-row fetch for the
# Daily Journal write shell. The other three endpoints filter to end_nlv>0
# so hollow rows stay invisible to display consumers; this one must NOT
# filter, because the write shell hydrates today's Game Plan / Daily
# Thoughts from a row that may not have an NLV yet.
# ---------------------------------------------------------------------------

def test_entry_returns_hollow_row_that_history_hides(client):
    """Regression from 2026-07-28: after journal_history started filtering
    hollow rows for Journal Log's browse view, Daily Journal's write shell
    (which shared the same endpoint) started blanking today's editors on
    refresh — the row was there but the filtered response hid it. The
    entry endpoint reads unfiltered so hydration works either way."""
    client.set_history([  # type: ignore[attr-defined]
        _row("2026-07-25", 52450.0, daily_thoughts="<p>day 25 thoughts</p>"),
        _row("2026-07-28", None, game_plan="<p>plan for today</p>",
             daily_thoughts="<p>typed but no NLV yet</p>"),
    ])
    body = client.get(
        "/api/journal/entry?portfolio=CanSlim&day=2026-07-28"
    ).json()
    assert body.get("day") == "2026-07-28"
    assert body.get("daily_thoughts") == "<p>typed but no NLV yet</p>"
    assert body.get("game_plan") == "<p>plan for today</p>"


def test_entry_returns_error_when_day_missing():
    """Belt-and-suspenders: no `day` param → clear error instead of a
    surprise "give me the whole journal" behavior."""
    from fastapi.testclient import TestClient
    import api.main as main
    # No DB patching here — endpoint should short-circuit on the missing
    # arg before touching load_journal, so the real db.load_journal never
    # fires. This test proves the guard runs first.
    tc = TestClient(main.app, headers=_auth_headers())
    body = tc.get("/api/journal/entry?portfolio=CanSlim").json()
    assert body == {"error": "day required"}


def test_entry_returns_error_when_no_row_for_day(client):
    """Non-existent day → {"error": "No entry"}. Frontend treats this as
    "clean slate" and initializes empty editors — the daily_journal
    hydration effect ignores the error shape and starts fresh."""
    client.set_history([_row("2026-07-25", 52450.0)])  # type: ignore[attr-defined]
    body = client.get(
        "/api/journal/entry?portfolio=CanSlim&day=2026-07-28"
    ).json()
    assert body == {"error": "No entry"}


def test_entry_does_NOT_filter_zero_nlv_rows(client):
    """journal_latest / journal_history filter end_nlv=0 as "unlogged."
    journal_entry does NOT filter — Daily Journal must still see the
    row to hydrate its editors from the game_plan / daily_thoughts
    columns, even if end_nlv is 0 or null."""
    client.set_history([  # type: ignore[attr-defined]
        _row("2026-07-28", 0.0, game_plan="<p>drafted</p>"),
    ])
    body = client.get(
        "/api/journal/entry?portfolio=CanSlim&day=2026-07-28"
    ).json()
    assert body.get("day") == "2026-07-28"
    assert body.get("game_plan") == "<p>drafted</p>"
    assert body.get("end_nlv") == 0.0


# ---------------------------------------------------------------------------
# /api/portfolio/heat-preview — same NLV-bearing filter
# ---------------------------------------------------------------------------

def test_heat_preview_walks_back_past_hollow_game_plan_row(client, monkeypatch):
    """Regression: NLV Entry's heat tile was reading 0.00% because a
    same-day hollow Game Plan row was picked as the "latest" and its
    end_nlv=None became 0. Walk back to the last NLV-bearing row so the
    preview matches the Portfolio Heat page — both endpoints agree on
    what "current equity" means."""
    import api.main as main
    # Stub the heat computation so the test is deterministic and doesn't
    # depend on yfinance. Assert the endpoint passed our real NLV through.
    captured = {}
    def fake_heat(portfolio, as_of, equity):
        captured["equity"] = equity
        return 12.34
    monkeypatch.setattr(main, "_compute_portfolio_heat", fake_heat)

    client.set_history([  # type: ignore[attr-defined]
        _row("2026-07-25", 52450.0),
        _row("2026-07-27", None),  # hollow Game Plan row would blank preview
    ])
    body = client.get("/api/portfolio/heat-preview?portfolio=CanSlim").json()
    assert body["nlv_used"] == 52450.0
    assert body["heat"] == 12.34
    assert captured["equity"] == 52450.0


def test_heat_preview_returns_zero_when_no_row_has_nlv(client, monkeypatch):
    """Truly-empty portfolio → 0/0 response. Belt-and-suspenders: the
    heat computation must not be invoked (no basis to compute against)."""
    import api.main as main
    heat_called = []
    monkeypatch.setattr(main, "_compute_portfolio_heat",
                        lambda *a, **kw: (heat_called.append(1), 99.0)[1])

    client.set_history([_row("2026-07-27", None)])  # type: ignore[attr-defined]
    body = client.get("/api/portfolio/heat-preview?portfolio=CanSlim").json()
    assert body == {"heat": 0.0, "nlv_used": 0.0, "portfolio": "CanSlim"}
    assert heat_called == []
