"""Tests for GET /api/analytics/trailing-avg-loss.

Locks the aggregation contract feeding the New Entry page's sizing model:
  denominator_% = max(4.0%, |avg_loss_pct|)
  position_size_% = risk_unit_% / denominator_%

The endpoint returns raw aggregates so the client can distinguish empty
from tighter-than-floor. Coverage:

  * Empty-portfolio → sample_size=0, both pcts None
  * Winners + zero-return trades → excluded from mean/median
  * Options excluded (protects the aggregate from the -99% option-to-zero
    outlier)
  * OPEN campaigns excluded
  * Soft-deleted rows excluded (via load_summary — deleted_at IS NULL
    already filtered at the DB layer)
  * Rows outside the trailing window (default 12 months) excluded
  * Median = statistical median (PERCENTILE_CONT(0.5) equivalent)
  * window_months bounds validation (rejects <=0 and >120)
  * Missing schema columns → empty response (migration-tolerant)

The fixture stubs db_layer.load_summary the same way
test_add_effectiveness.py does, so no live DB is required.
"""
from __future__ import annotations

from typing import Any

import jwt
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _row(
    trade_id: str,
    *,
    return_pct: float,
    status: str = "CLOSED",
    instrument_type: str = "STOCK",
    closed_date: str = "2026-06-01",
    ticker: str = "TEST",
) -> dict[str, Any]:
    return {
        "trade_id": trade_id,
        "ticker": ticker,
        "status": status,
        "instrument_type": instrument_type,
        "return_pct": return_pct,
        "closed_date": pd.Timestamp(closed_date),
        # Fields _normalize_trades touches; kept present to avoid KeyError.
        "open_date": pd.Timestamp("2026-01-02"),
        "shares": 0.0,
        "avg_entry": 0.0,
        "avg_exit": 0.0,
        "total_cost": 0.0,
        "realized_pl": 0.0,
        "unrealized_pl": 0.0,
        "rule": "",
        "stop_loss": 0.0,
        "multiplier": 1.0,
    }


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    import db_layer

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {"rows": []}

    def fake_load_summary(portfolio, status=None):
        df = pd.DataFrame(state["rows"])
        if df.empty:
            return df
        if status:
            df = df[df["status"].astype(str).str.upper() == status.upper()]
        return df.copy()

    monkeypatch.setattr(db_layer, "load_summary", fake_load_summary)

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


def _GET(client, **params):
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v != "")
    return client.get(f"/api/analytics/trailing-avg-loss?{qs}")


# ---------------------------------------------------------------------------
# 1. Empty portfolio → sample_size=0
# ---------------------------------------------------------------------------


def test_empty_portfolio_returns_null_pcts_and_zero_sample(stubbed):
    state, client = stubbed
    state["rows"] = []
    r = _GET(client, portfolio="CanSlim")
    assert r.status_code == 200
    body = r.json()
    assert body["sample_size"] == 0
    assert body["avg_loss_pct"] is None
    assert body["median_loss_pct"] is None
    assert body["window_months"] == 12
    assert body["portfolio"] == "CanSlim"


# ---------------------------------------------------------------------------
# 2. Winners + zero-return excluded
# ---------------------------------------------------------------------------


def test_only_negative_return_pct_counted(stubbed):
    state, client = stubbed
    state["rows"] = [
        _row("t1", return_pct=-3.0),
        _row("t2", return_pct=-5.0),
        _row("t3", return_pct=0.0),   # exclude
        _row("t4", return_pct=12.5),  # exclude
    ]
    r = _GET(client, portfolio="CanSlim")
    body = r.json()
    assert body["sample_size"] == 2
    assert body["avg_loss_pct"] == pytest.approx(-4.0)
    assert body["median_loss_pct"] == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# 3. Options excluded — the whole reason we chose the STOCK filter
# ---------------------------------------------------------------------------


def test_options_excluded_from_aggregate(stubbed):
    """Bought long calls going to zero register as ~-99% return; those
    don't reflect the trader's actual exit behavior on equity positions.
    The aggregate must ignore them entirely."""
    state, client = stubbed
    state["rows"] = [
        _row("t1", return_pct=-3.0, instrument_type="STOCK"),
        _row("t2", return_pct=-5.0, instrument_type="STOCK"),
        _row("opt-total-loss", return_pct=-99.76, instrument_type="OPTION"),
        _row("opt-partial",    return_pct=-45.0,  instrument_type="OPTION"),
    ]
    r = _GET(client, portfolio="CanSlim")
    body = r.json()
    assert body["sample_size"] == 2, "options must not contribute"
    assert body["avg_loss_pct"] == pytest.approx(-4.0)
    # If options had leaked in the mean would be ~-38%, an obvious tell.
    assert body["avg_loss_pct"] > -10.0


# ---------------------------------------------------------------------------
# 4. OPEN campaigns excluded
# ---------------------------------------------------------------------------


def test_open_campaigns_excluded(stubbed):
    state, client = stubbed
    state["rows"] = [
        _row("closed", return_pct=-4.0, status="CLOSED"),
        _row("open-loser", return_pct=-30.0, status="OPEN"),  # exclude
    ]
    r = _GET(client, portfolio="CanSlim")
    body = r.json()
    assert body["sample_size"] == 1
    assert body["avg_loss_pct"] == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# 5. Window boundary: rows outside trailing 12 months excluded
# ---------------------------------------------------------------------------


def test_rows_outside_trailing_window_excluded(stubbed):
    """A losing trade closed 15 months ago must NOT count toward the
    trailing 12-month aggregate. Boundary at CURRENT_DATE - N months."""
    state, client = stubbed
    today = pd.Timestamp.today().normalize()
    in_window = (today - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    outside   = (today - pd.DateOffset(months=15)).strftime("%Y-%m-%d")
    state["rows"] = [
        _row("recent", return_pct=-4.0, closed_date=in_window),
        _row("stale",  return_pct=-99.0, closed_date=outside),
    ]
    r = _GET(client, portfolio="CanSlim", window_months=12)
    body = r.json()
    assert body["sample_size"] == 1
    assert body["avg_loss_pct"] == pytest.approx(-4.0)


def test_custom_window_months_reshapes_sample(stubbed):
    state, client = stubbed
    today = pd.Timestamp.today().normalize()
    d3   = (today - pd.DateOffset(months=2)).strftime("%Y-%m-%d")
    d6   = (today - pd.DateOffset(months=5)).strftime("%Y-%m-%d")
    d12  = (today - pd.DateOffset(months=11)).strftime("%Y-%m-%d")
    state["rows"] = [
        _row("r3",  return_pct=-2.0, closed_date=d3),
        _row("r6",  return_pct=-6.0, closed_date=d6),
        _row("r12", return_pct=-12.0, closed_date=d12),
    ]
    # 3-month window: only r3
    r3 = _GET(client, portfolio="CanSlim", window_months=3).json()
    assert r3["sample_size"] == 1
    assert r3["avg_loss_pct"] == pytest.approx(-2.0)
    # 6-month window: r3 + r6
    r6 = _GET(client, portfolio="CanSlim", window_months=6).json()
    assert r6["sample_size"] == 2
    assert r6["avg_loss_pct"] == pytest.approx(-4.0)
    # 12-month window: all three
    r12 = _GET(client, portfolio="CanSlim", window_months=12).json()
    assert r12["sample_size"] == 3
    assert r12["avg_loss_pct"] == pytest.approx(-6.6666, abs=1e-3)


# ---------------------------------------------------------------------------
# 6. Median contract: matches numpy.median on an asymmetric sample
# ---------------------------------------------------------------------------


def test_median_matches_statistical_definition(stubbed):
    state, client = stubbed
    pcts = [-2.0, -3.0, -5.0, -8.0, -25.0]  # asymmetric tail
    state["rows"] = [_row(f"t{i}", return_pct=p) for i, p in enumerate(pcts)]
    r = _GET(client, portfolio="CanSlim").json()
    assert r["sample_size"] == 5
    assert r["median_loss_pct"] == pytest.approx(float(np.median(pcts)))
    assert r["avg_loss_pct"] == pytest.approx(float(np.mean(pcts)))
    # Mean is dragged by -25%; median stays at -5% — the whole reason
    # both values are exposed to the client.
    assert r["median_loss_pct"] > r["avg_loss_pct"]


# ---------------------------------------------------------------------------
# 7. window_months bounds validation
# ---------------------------------------------------------------------------


def test_rejects_non_positive_window(stubbed):
    _, client = stubbed
    assert _GET(client, portfolio="CanSlim", window_months=0).status_code == 400
    assert _GET(client, portfolio="CanSlim", window_months=-3).status_code == 400


def test_rejects_absurdly_large_window(stubbed):
    _, client = stubbed
    r = _GET(client, portfolio="CanSlim", window_months=121)
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 8. Migration-tolerance — schema missing return_pct or closed_date columns
# ---------------------------------------------------------------------------


def test_schema_missing_required_columns_returns_empty(stubbed, monkeypatch):
    """If the DB is mid-migration and load_summary returns rows without
    the aggregation columns, the endpoint must return the empty-response
    shape (sample_size=0, both pcts None) rather than 500."""
    import db_layer

    def _load_ancient(portfolio, status=None):
        return pd.DataFrame([{
            "trade_id": "old", "ticker": "T", "status": "CLOSED",
            "open_date": pd.Timestamp("2026-01-01"),
            # No return_pct, no closed_date, no instrument_type
        }])
    monkeypatch.setattr(db_layer, "load_summary", _load_ancient)
    _, client = stubbed
    r = _GET(client, portfolio="CanSlim")
    assert r.status_code == 200
    body = r.json()
    assert body["sample_size"] == 0
    assert body["avg_loss_pct"] is None


# ---------------------------------------------------------------------------
# 9. Golden path — mirrors the endpoint's documented example
# ---------------------------------------------------------------------------


def test_response_shape_matches_contract(stubbed):
    """Freeze the exact response keys the frontend consumer parses."""
    state, client = stubbed
    state["rows"] = [_row("t1", return_pct=-4.58)]
    body = _GET(client, portfolio="CanSlim").json()
    for k in ("portfolio", "window_months", "avg_loss_pct",
              "median_loss_pct", "sample_size", "as_of"):
        assert k in body, f"missing key: {k}"
    assert body["portfolio"] == "CanSlim"
    assert body["window_months"] == 12
    assert body["sample_size"] == 1
    assert body["avg_loss_pct"] == pytest.approx(-4.58)
    assert body["median_loss_pct"] == pytest.approx(-4.58)
    # as_of is ISO date; not asserting exact value (moves with test time).
    assert isinstance(body["as_of"], str) and len(body["as_of"]) == 10
