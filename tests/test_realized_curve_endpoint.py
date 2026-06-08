"""Tests for GET /api/realized/curve — realized-equity-curve endpoint.

Source of truth: lot_closures.realized_pl (already multiplier-scaled at
write time) summed by lot_closures.closed_at::date, prefix-summed into
a cumulative series. Baseline NLV for the % anchor: trading_journal NLV
at/just before `start`, falling back to portfolios.starting_capital.

Tests mock the loaders (db.load_lot_closures, db.load_journal, and
db.get_db_connection for the starting_capital fallback) so we exercise
the endpoint's grouping / prefix-sum / baseline-resolution logic without
hitting the DB.

Coverage:
  1. Prefix-sum correctness — multiple closures on the same day sum into
     one point; cum_realized_pl ascends across days in date order.
  2. start filter — closures before start are excluded (no carry into a
     starting point); the curve begins at 0 on start_date.
  3. % anchor math — cum_realized_pct == cum_realized_pl / start_nlv * 100
     against a known baseline.
  4. Baseline preference — journal row ON start uses beg_nlv; no row on
     start but a row before → end_nlv of latest prior row; no journal
     rows at all → portfolios.starting_capital with baseline_source set.
  5. Options closures — realized_pl is already multiplier-scaled at the
     lot_closures write path; the curve must NOT re-scale.
  6. Response envelope — summary fields populated correctly (total,
     closed_count, start_date echo, baseline_source value).
  7. Empty input — no closures returns series=[], totals zeroed.
  8. Bad date — start="not-a-date" returns {error: ...} not a 500.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import jwt
import pandas as pd
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def _closures_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a frame matching db.load_lot_closures()'s shape. Only the
    columns the endpoint reads (closed_at, realized_pl) are populated;
    the rest are filler so the frame matches the real loader's columns."""
    base_cols = ["trade_id", "buy_trx_id", "sell_trx_id",
                 "shares", "buy_price", "sell_price",
                 "multiplier", "realized_pl", "closed_at"]
    return pd.DataFrame([
        {c: r.get(c) for c in base_cols} for r in rows
    ])


def _journal_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """trading_journal frame — only day / beg_nlv / end_nlv read here."""
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["day", "beg_nlv", "end_nlv"])


@pytest.fixture
def stubbed(monkeypatch):
    """Patch the loaders the endpoint depends on so each test can drive
    closures and journal independently. Also patches db.get_db_connection
    for the starting_capital fallback path."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "closures": pd.DataFrame(),
        "journal":  pd.DataFrame(),
        "starting_capital": None,   # what SELECT starting_capital returns
        "starting_capital_raises": False,
    }

    monkeypatch.setattr(main.db, "load_lot_closures",
                        lambda portfolio: state["closures"].copy())
    monkeypatch.setattr(main.db, "load_journal",
                        lambda portfolio: state["journal"].copy())

    # Mock the get_db_connection context manager used by the
    # starting_capital fallback. cur.fetchone() returns (starting_capital,)
    # or None to simulate "no portfolio row." If `starting_capital_raises`
    # is True, the connection itself raises — exercises the except path.
    class _FakeCursor:
        def execute(self, *args, **kwargs): pass
        def fetchone(self):
            sc = state["starting_capital"]
            return (sc,) if sc is not None else None
        def __enter__(self): return self
        def __exit__(self, *args): return False

    class _FakeConn:
        def cursor(self): return _FakeCursor()
        def __enter__(self): return self
        def __exit__(self, *args): return False

    def fake_get_db_connection():
        if state["starting_capital_raises"]:
            raise RuntimeError("connection refused")
        return _FakeConn()

    monkeypatch.setattr(main.db, "get_db_connection", fake_get_db_connection)

    if hasattr(main.limiter, "enabled"):
        original_enabled = main.limiter.enabled
        main.limiter.enabled = False
    else:
        original_enabled = None

    client = TestClient(main.app, headers=_auth_headers())
    try:
        yield state, client
    finally:
        if original_enabled is not None:
            main.limiter.enabled = original_enabled


# ─────────────────────────────────────────────────────────────────────
# 1. Prefix-sum correctness
# ─────────────────────────────────────────────────────────────────────

def test_prefix_sum_same_day_summed_into_one_point(stubbed):
    """Two closures on 2026-01-15 → one series point with the sum."""
    state, client = stubbed
    state["closures"] = _closures_df([
        {"realized_pl": 100.0, "closed_at": "2026-01-15 10:30:00"},
        {"realized_pl": 250.0, "closed_at": "2026-01-15 14:45:00"},
    ])
    state["journal"] = _journal_df([
        {"day": "2026-01-01", "beg_nlv": 100000.0, "end_nlv": 100000.0},
    ])

    r = client.get("/api/realized/curve?portfolio=CanSlim&start=2026-01-01")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["series"]) == 1
    assert body["series"][0]["day"] == "2026-01-15"
    assert body["series"][0]["cum_realized_pl"] == 350.0
    assert body["summary"]["closed_count"] == 2  # raw closure rows in range


def test_prefix_sum_ascends_across_days(stubbed):
    """Three days of closures → cum_realized_pl ascends in date order."""
    state, client = stubbed
    state["closures"] = _closures_df([
        {"realized_pl":  500.0, "closed_at": "2026-01-10 09:30:00"},
        {"realized_pl": -200.0, "closed_at": "2026-02-05 11:00:00"},
        {"realized_pl":  150.0, "closed_at": "2026-03-12 15:00:00"},
    ])
    state["journal"] = _journal_df([
        {"day": "2026-01-01", "beg_nlv": 100000.0, "end_nlv": 100000.0},
    ])

    r = client.get("/api/realized/curve?portfolio=CanSlim&start=2026-01-01")
    body = r.json()
    assert [p["day"] for p in body["series"]] == ["2026-01-10", "2026-02-05", "2026-03-12"]
    assert [p["cum_realized_pl"] for p in body["series"]] == [500.0, 300.0, 450.0]
    assert body["summary"]["total_realized_pl"] == 450.0
    assert body["summary"]["closed_count"] == 3


# ─────────────────────────────────────────────────────────────────────
# 2. start filter
# ─────────────────────────────────────────────────────────────────────

def test_start_filter_excludes_pre_start_closures(stubbed):
    """A 2025-12-31 closure must NOT show up in a curve started 2026-01-01,
    and must NOT be folded into a starting point — the curve begins at 0."""
    state, client = stubbed
    state["closures"] = _closures_df([
        {"realized_pl": 10000.0, "closed_at": "2025-12-31 15:30:00"},
        {"realized_pl":   400.0, "closed_at": "2026-01-15 10:00:00"},
    ])
    state["journal"] = _journal_df([
        {"day": "2026-01-01", "beg_nlv": 100000.0, "end_nlv": 100000.0},
    ])

    r = client.get("/api/realized/curve?portfolio=CanSlim&start=2026-01-01")
    body = r.json()
    assert len(body["series"]) == 1
    assert body["series"][0]["day"] == "2026-01-15"
    assert body["series"][0]["cum_realized_pl"] == 400.0  # NOT 10400
    assert body["summary"]["closed_count"] == 1


# ─────────────────────────────────────────────────────────────────────
# 3. % anchor math
# ─────────────────────────────────────────────────────────────────────

def test_pct_anchor_against_known_start_nlv(stubbed):
    """cum_realized_pct = cum_realized_pl / start_nlv * 100, per point."""
    state, client = stubbed
    state["closures"] = _closures_df([
        {"realized_pl": 1000.0, "closed_at": "2026-01-10 10:00:00"},
        {"realized_pl": 1500.0, "closed_at": "2026-01-20 10:00:00"},
    ])
    state["journal"] = _journal_df([
        # beg_nlv on start_date is the canonical anchor.
        {"day": "2026-01-01", "beg_nlv": 50000.0, "end_nlv": 50500.0},
    ])

    r = client.get("/api/realized/curve?portfolio=CanSlim&start=2026-01-01")
    body = r.json()
    pts = body["series"]
    # Point 1: 1000 / 50000 * 100 = 2.0
    # Point 2: 2500 / 50000 * 100 = 5.0
    assert pts[0]["cum_realized_pct"] == 2.0
    assert pts[1]["cum_realized_pct"] == 5.0
    assert body["summary"]["start_nlv"] == 50000.0
    assert body["summary"]["realized_pct"] == 5.0
    assert body["summary"]["baseline_source"] == "journal"


# ─────────────────────────────────────────────────────────────────────
# 4. Baseline preference / fallback chain
# ─────────────────────────────────────────────────────────────────────

def test_baseline_uses_journal_beg_nlv_on_start_date(stubbed):
    """When a journal row exists ON start_date, use its beg_nlv (the
    open-of-day NLV). End_nlv of that same row must NOT be used."""
    state, client = stubbed
    state["closures"] = _closures_df([
        {"realized_pl": 100.0, "closed_at": "2026-01-15 10:00:00"},
    ])
    state["journal"] = _journal_df([
        {"day": "2026-01-01", "beg_nlv": 80000.0, "end_nlv": 85000.0},
    ])
    r = client.get("/api/realized/curve?portfolio=CanSlim&start=2026-01-01")
    body = r.json()
    assert body["summary"]["start_nlv"] == 80000.0  # beg, not end
    assert body["summary"]["baseline_source"] == "journal"


def test_baseline_uses_journal_end_nlv_when_only_prior_rows_exist(stubbed):
    """No row ON start; latest row BEFORE start → use that row's end_nlv."""
    state, client = stubbed
    state["closures"] = _closures_df([
        {"realized_pl": 100.0, "closed_at": "2026-01-15 10:00:00"},
    ])
    state["journal"] = _journal_df([
        {"day": "2025-12-30", "beg_nlv": 70000.0, "end_nlv": 71000.0},
        {"day": "2025-12-31", "beg_nlv": 71000.0, "end_nlv": 72500.0},
    ])
    r = client.get("/api/realized/curve?portfolio=CanSlim&start=2026-01-01")
    body = r.json()
    assert body["summary"]["start_nlv"] == 72500.0  # end_nlv of latest prior row
    assert body["summary"]["baseline_source"] == "journal"


def test_baseline_falls_back_to_starting_capital(stubbed):
    """No journal rows → use portfolios.starting_capital, with
    baseline_source flipped to 'starting_capital'."""
    state, client = stubbed
    state["closures"] = _closures_df([
        {"realized_pl": 1000.0, "closed_at": "2026-01-15 10:00:00"},
    ])
    state["journal"] = pd.DataFrame()  # empty
    state["starting_capital"] = 60000.0

    r = client.get("/api/realized/curve?portfolio=CanSlim&start=2026-01-01")
    body = r.json()
    assert body["summary"]["baseline_source"] == "starting_capital"
    assert body["summary"]["start_nlv"] == 60000.0
    # 1000 / 60000 * 100 = 1.666… → rounded to 2dp.
    assert body["series"][0]["cum_realized_pct"] == pytest.approx(1.67, abs=0.01)


def test_baseline_none_when_no_journal_and_no_starting_capital(stubbed):
    """Both sources empty → baseline_source='none', start_nlv=0, all
    pct values 0 (no division-by-zero)."""
    state, client = stubbed
    state["closures"] = _closures_df([
        {"realized_pl": 1000.0, "closed_at": "2026-01-15 10:00:00"},
    ])
    state["journal"] = pd.DataFrame()
    state["starting_capital"] = None

    r = client.get("/api/realized/curve?portfolio=CanSlim&start=2026-01-01")
    body = r.json()
    assert body["summary"]["baseline_source"] == "none"
    assert body["summary"]["start_nlv"] == 0.0
    assert body["series"][0]["cum_realized_pl"] == 1000.0
    assert body["series"][0]["cum_realized_pct"] == 0.0
    assert body["summary"]["realized_pct"] == 0.0


# ─────────────────────────────────────────────────────────────────────
# 5. Multiplier path — already scaled in lot_closures
# ─────────────────────────────────────────────────────────────────────

def test_options_closure_realized_pl_is_not_rescaled(stubbed):
    """An options closure stored as 1500 (per-contract pl × multiplier
    × shares already baked in by trade_calc.py:189) must appear as
    1500 in the cum series — NOT 150000 (would be a double-scale)."""
    state, client = stubbed
    state["closures"] = _closures_df([
        # Realistic options closure: 5 contracts, $3 pl per contract,
        # multiplier=100 → trade_calc writes realized_pl = 5 * 3 * 100 = 1500.
        {"realized_pl": 1500.0, "closed_at": "2026-01-15 10:00:00",
         "multiplier": 100, "shares": 5},
    ])
    state["journal"] = _journal_df([
        {"day": "2026-01-01", "beg_nlv": 100000.0, "end_nlv": 100000.0},
    ])

    r = client.get("/api/realized/curve?portfolio=CanSlim&start=2026-01-01")
    body = r.json()
    assert body["series"][0]["cum_realized_pl"] == 1500.0
    assert body["summary"]["total_realized_pl"] == 1500.0


# ─────────────────────────────────────────────────────────────────────
# 6. Response envelope
# ─────────────────────────────────────────────────────────────────────

def test_response_envelope_summary_shape(stubbed):
    """summary carries every documented field with the expected types."""
    state, client = stubbed
    state["closures"] = _closures_df([
        {"realized_pl": 100.0, "closed_at": "2026-01-15 10:00:00"},
    ])
    state["journal"] = _journal_df([
        {"day": "2026-01-01", "beg_nlv": 100000.0, "end_nlv": 100000.0},
    ])

    r = client.get("/api/realized/curve?portfolio=CanSlim&start=2026-01-01")
    body = r.json()
    summary = body["summary"]
    assert set(summary.keys()) == {
        "total_realized_pl", "realized_pct", "closed_count",
        "start_nlv", "start_date", "baseline_source",
    }
    assert summary["start_date"] == "2026-01-01"
    assert summary["closed_count"] == 1
    assert summary["total_realized_pl"] == 100.0


# ─────────────────────────────────────────────────────────────────────
# 7. Empty input
# ─────────────────────────────────────────────────────────────────────

def test_no_closures_returns_empty_series_zero_summary(stubbed):
    """Portfolio with no closures → series=[], total/count=0; the
    summary baseline + start_date echo are still populated normally."""
    state, client = stubbed
    state["closures"] = pd.DataFrame()
    state["journal"] = _journal_df([
        {"day": "2026-01-01", "beg_nlv": 100000.0, "end_nlv": 100000.0},
    ])

    r = client.get("/api/realized/curve?portfolio=CanSlim&start=2026-01-01")
    body = r.json()
    assert body["series"] == []
    assert body["summary"]["total_realized_pl"] == 0.0
    assert body["summary"]["closed_count"] == 0
    assert body["summary"]["realized_pct"] == 0.0
    assert body["summary"]["start_nlv"] == 100000.0
    assert body["summary"]["baseline_source"] == "journal"


# ─────────────────────────────────────────────────────────────────────
# 8. Bad input
# ─────────────────────────────────────────────────────────────────────

def test_bad_start_date_returns_error_not_500(stubbed):
    state, client = stubbed
    r = client.get("/api/realized/curve?portfolio=CanSlim&start=not-a-date")
    assert r.status_code == 200
    assert "error" in r.json()


def test_default_start_is_2026_01_01_when_omitted(stubbed):
    """Spec: default start = 2026-01-01."""
    state, client = stubbed
    state["closures"] = pd.DataFrame()
    state["journal"] = _journal_df([
        {"day": "2026-01-01", "beg_nlv": 100000.0, "end_nlv": 100000.0},
    ])
    r = client.get("/api/realized/curve?portfolio=CanSlim")
    body = r.json()
    assert body["summary"]["start_date"] == "2026-01-01"
