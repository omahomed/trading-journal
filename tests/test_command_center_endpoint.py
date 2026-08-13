"""Tests for GET /api/command-center — the cross-portfolio landing page.

Stubs db.list_portfolios / db.load_journal / db.load_summary /
db.get_net_contributions so the tests exercise the aggregation loop
without a database. Mirrors the fixture pattern in
test_contract_invariants.py.
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


def _journal_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    cols = ["day", "end_nlv", "beg_nlv", "cash_change", "pct_invested",
            "daily_dollar_change", "daily_pct_change"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def _summary_df(open_count: int, closed_count: int = 0) -> pd.DataFrame:
    """Fake trades_summary: open_count OPEN rows, closed_count CLOSED rows."""
    rows: list[dict[str, Any]] = []
    for i in range(open_count):
        rows.append({"trade_id": f"OPEN-{i}", "status": "OPEN", "ticker": "AAA"})
    for i in range(closed_count):
        rows.append({"trade_id": f"CLOSED-{i}", "status": "CLOSED", "ticker": "AAA"})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["trade_id", "status", "ticker"])


@pytest.fixture
def cc_client(monkeypatch):
    """TestClient with all four db calls stubbed so tests can inject
    per-portfolio journals + summaries. State mutated via configure()."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "portfolios": [],
        "journal_by_name": {},   # name -> DataFrame
        "summary_by_name": {},   # name -> DataFrame
        "contrib_by_pid": {},    # id -> float
    }

    monkeypatch.setattr(db_layer, "list_portfolios",
                        lambda: state["portfolios"])
    monkeypatch.setattr(db_layer, "load_journal",
                        lambda name: state["journal_by_name"].get(name, pd.DataFrame()))
    monkeypatch.setattr(db_layer, "load_summary",
                        lambda name: state["summary_by_name"].get(name, pd.DataFrame()))
    monkeypatch.setattr(db_layer, "get_net_contributions",
                        lambda pid: state["contrib_by_pid"].get(pid, 0.0))
    # Neutralize heat compute (touched by _normalize_trades sometimes)
    monkeypatch.setattr(main, "_compute_portfolio_heat",
                        lambda *a, **kw: 0.0)

    tc = TestClient(main.app, headers=_auth_headers())

    def configure(portfolios: list[dict], journals: dict, summaries: dict,
                  contribs: dict | None = None):
        state["portfolios"] = portfolios
        state["journal_by_name"] = journals
        state["summary_by_name"] = summaries
        state["contrib_by_pid"] = contribs or {}

    tc.configure = configure  # type: ignore[attr-defined]
    return tc


def test_empty_portfolios_returns_empty_rows(cc_client):
    """No portfolios → {rows: []}. Frontend renders the onboarding empty
    state; must not 500 or return a plain list."""
    cc_client.configure(portfolios=[], journals={}, summaries={})  # type: ignore[attr-defined]
    res = cc_client.get("/api/command-center").json()
    assert res == {"rows": []}


def test_multi_portfolio_returns_row_per_portfolio(cc_client):
    """Three portfolios → three rows. Each row carries its own KPIs from
    the portfolio-scoped journal + summary. The endpoint's job is per-
    portfolio isolation — no cross-contamination between them."""
    cc_client.configure(  # type: ignore[attr-defined]
        portfolios=[
            {"id": 1, "name": "CanSlim"},
            {"id": 2, "name": "LTG"},
            {"id": 3, "name": "Diva"},
        ],
        journals={
            "CanSlim": _journal_df([
                {"day": "2026-08-11", "end_nlv": 100000.0, "beg_nlv": 100000.0,
                 "cash_change": 0, "pct_invested": 50.0,
                 "daily_dollar_change": 0, "daily_pct_change": 0},
                {"day": "2026-08-12", "end_nlv": 92500.0, "beg_nlv": 100000.0,
                 "cash_change": 0, "pct_invested": 45.0,
                 "daily_dollar_change": -7500.0, "daily_pct_change": -7.5},
            ]),
            "LTG": _journal_df([
                {"day": "2026-08-12", "end_nlv": 250000.0, "beg_nlv": 250000.0,
                 "cash_change": 0, "pct_invested": 80.0,
                 "daily_dollar_change": 0, "daily_pct_change": 0},
            ]),
            "Diva": _journal_df([
                {"day": "2026-08-11", "end_nlv": 50000.0, "beg_nlv": 50000.0,
                 "cash_change": 0, "pct_invested": 100.0,
                 "daily_dollar_change": 0, "daily_pct_change": 0},
                {"day": "2026-08-12", "end_nlv": 40000.0, "beg_nlv": 50000.0,
                 "cash_change": 0, "pct_invested": 90.0,
                 "daily_dollar_change": -10000.0, "daily_pct_change": -20.0},
            ]),
        },
        summaries={
            "CanSlim": _summary_df(open_count=8, closed_count=42),
            "LTG":     _summary_df(open_count=3, closed_count=10),
            "Diva":    _summary_df(open_count=0, closed_count=5),
        },
    )

    res = cc_client.get("/api/command-center").json()
    rows = res["rows"]
    assert len(rows) == 3

    by_name = {r["portfolio_name"]: r for r in rows}

    # CanSlim: at 7.5% drawdown, exactly on L1 threshold
    canslim = by_name["CanSlim"]
    assert canslim["portfolio_id"] == 1
    assert canslim["journal_available"] is True
    assert canslim["nlv"] == 92500.0
    assert canslim["exposure_pct"] == 45.0
    assert canslim["open_position_count"] == 8
    # peak=100000, current=92500 → -7.5%
    assert abs(canslim["drawdown_current_pct"] - (-7.5)) < 0.01
    assert canslim["drawdown_peak_nlv"] == 100000.0

    # LTG: only one journal row → 0% drawdown, no day delta yet
    ltg = by_name["LTG"]
    assert ltg["journal_available"] is True
    assert ltg["nlv"] == 250000.0
    assert ltg["exposure_pct"] == 80.0
    assert ltg["open_position_count"] == 3
    assert ltg["drawdown_current_pct"] == 0.0
    # First-entry portfolios: no prior day, no delta
    assert ltg["nlv_delta_dollar"] is None
    assert ltg["nlv_delta_pct"] is None

    # Diva: at 20% drawdown, past L3
    diva = by_name["Diva"]
    assert diva["open_position_count"] == 0
    assert abs(diva["drawdown_current_pct"] - (-20.0)) < 0.01


def test_portfolio_with_no_journal_still_returns_row(cc_client):
    """A brand-new portfolio (no NLV logged yet) must still appear on the
    Command Center — with journal_available=false and null KPIs. The row
    itself is the affordance: "you have this portfolio, log an NLV to
    populate it." Dropping the row would silently hide the portfolio."""
    cc_client.configure(  # type: ignore[attr-defined]
        portfolios=[
            {"id": 1, "name": "CanSlim"},
            {"id": 99, "name": "Fresh"},
        ],
        journals={
            "CanSlim": _journal_df([
                {"day": "2026-08-12", "end_nlv": 100000.0, "beg_nlv": 100000.0,
                 "cash_change": 0, "pct_invested": 50.0,
                 "daily_dollar_change": 0, "daily_pct_change": 0},
            ]),
            # "Fresh" intentionally omitted → empty df
        },
        summaries={},  # neither portfolio has trades yet
    )

    res = cc_client.get("/api/command-center").json()
    rows = res["rows"]
    assert len(rows) == 2

    fresh = next(r for r in rows if r["portfolio_name"] == "Fresh")
    assert fresh["journal_available"] is False
    assert fresh["nlv"] is None
    assert fresh["drawdown_current_pct"] is None
    assert fresh["open_position_count"] == 0
