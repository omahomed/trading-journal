"""Tests for the /api/market/rally-data endpoint.

Covers two related fixes:
1. Day numbering: Day 1 must be the FTD itself (not the day after);
   baseline for the % gain column is the day-before-FTD close.
2. The 4/22/2025 historical reference line (`historical_rally_2025`)
   is computed live from ^IXIC closes via yfinance, cached for the
   worker lifetime, and degrades gracefully if yfinance is down.

yfinance is monkeypatched throughout — no network calls.
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


def _make_history_df(dates: list[str], closes: list[float],
                     lows: list[float] | None = None) -> pd.DataFrame:
    """Build a yfinance-shaped history DataFrame indexed by date."""
    if lows is None:
        # Default: low ≈ close − 0.5% (just needs to be < close for the
        # endpoint's rally-fail check to behave sensibly; tests don't
        # exercise that path here).
        lows = [c * 0.995 for c in closes]
    df = pd.DataFrame({"Close": closes, "Low": lows},
                      index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates]))
    return df


# ---------------------------------------------------------------------------
# Synthetic price series anchored on a fictional 2026-04-08 FTD. The
# day-before-FTD close (2026-04-07) is $20,000 and the FTD itself (Day 1)
# closes at $20,560 — a +2.80% session gain that should appear verbatim
# in points[0].pct under the new convention.
# ---------------------------------------------------------------------------

USER_FTD_DATES = [
    "2026-04-03", "2026-04-06", "2026-04-07",
    "2026-04-08",  # ← FTD (Day 1)
    "2026-04-09", "2026-04-10",
]
USER_FTD_CLOSES = [
    19800.00, 19900.00, 20000.00,    # pre-FTD: baseline = 20000.00
    20560.00,                          # Day 1: +2.80%
    20720.00, 21000.00,                # Day 2: +3.60%, Day 3: +5.00%
]
USER_DAY_BEFORE_FTD_CLOSE = 20000.00
USER_FTD_PCT_DAY1 = 2.80


# Mirror of the verified live ^IXIC values (2025-04-21 + post-FTD trading
# days) so the historical-line test asserts on real numbers, not made-up.
HISTORICAL_2025_DATES = [
    "2025-04-15", "2025-04-16", "2025-04-17",
    "2025-04-21",                      # day before FTD (2025-04-18 was Good Friday)
    "2025-04-22",                      # ← Day 1 FTD
    "2025-04-23", "2025-04-24", "2025-04-25",
]
HISTORICAL_2025_CLOSES = [
    16823.17, 16307.16, 16286.45,
    15870.90,                          # day_before_ftd_close
    16300.42,                          # Day 1: +2.71%
    16708.05, 17166.04, 17382.94,      # Day 2..4
]


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient with auth + rate-limiter disabled, plus a clean
    @lru_cache so each test starts with the historical-rally helper unprimed."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    # Reset the historical helper's cache between tests so each test
    # observes the yfinance mock for the first call (otherwise ordering
    # matters and the caching test poisons later runs).
    main._get_2025_historical_rally.cache_clear()

    original_enabled = getattr(main.limiter, "enabled", True)
    if hasattr(main.limiter, "enabled"):
        main.limiter.enabled = False

    c = TestClient(main.app, headers=_auth_headers())
    try:
        yield c
    finally:
        if hasattr(main.limiter, "enabled"):
            main.limiter.enabled = original_enabled


def _patch_yfinance(monkeypatch, *,
                    user_df: pd.DataFrame | None = None,
                    historical_df: pd.DataFrame | None = None,
                    historical_raises: bool = False,
                    return_call_counter: bool = False):
    """Install a yfinance.Ticker mock that returns different DataFrames
    depending on which call it's serving. Both the user line and the
    historical line use ^IXIC, so we discriminate by call order:
    the endpoint calls the user-line history first, then the historical
    helper. (When the helper is cached from an earlier test in the same
    process, only the user-line call lands.)

    Pass return_call_counter=True to also get back a list whose length
    is the number of `.history()` invocations, for the caching assertion.
    """
    calls: list[dict] = []

    def fake_history(self_inner, **kwargs):
        # Discriminate by start date: the historical helper passes
        # start="2025-04-15"; everything else is the user line.
        start = str(kwargs.get("start", ""))
        if start == "2025-04-15":
            calls.append({"kind": "historical", "kwargs": dict(kwargs)})
            if historical_raises:
                raise RuntimeError("yfinance unreachable")
            return historical_df if historical_df is not None else pd.DataFrame()
        else:
            calls.append({"kind": "user", "kwargs": dict(kwargs)})
            return user_df if user_df is not None else pd.DataFrame()

    # Patch yf.Ticker so its .history() method returns our fake.
    fake_ticker = MagicMock()
    fake_ticker.history = lambda **kwargs: fake_history(fake_ticker, **kwargs)

    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", lambda symbol: fake_ticker)
    return calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_day_1_is_ftd_itself(client, monkeypatch):
    """Day 1 in the response must be the FTD itself, not the day after."""
    user_df = _make_history_df(USER_FTD_DATES, USER_FTD_CLOSES)
    historical_df = _make_history_df(HISTORICAL_2025_DATES, HISTORICAL_2025_CLOSES)
    _patch_yfinance(monkeypatch, user_df=user_df, historical_df=historical_df)

    r = client.get("/api/market/rally-data?ftd_date=2026-04-08&index=^IXIC")
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body, body

    points = body["points"]
    # Day 1 is the FTD's own row.
    assert points[0]["day"] == 1
    assert points[0]["date"] == "2026-04-08"
    assert points[0]["close"] == 20560.00
    # Day 1 pct = FTD's own session gain (vs day-before-FTD close).
    assert points[0]["pct"] == pytest.approx(USER_FTD_PCT_DAY1, abs=0.01)


def test_baseline_is_day_before_ftd_close(client, monkeypatch):
    """The pct for every row uses day-before-FTD close as baseline,
    not FTD close. Verify Day 2's value matches the explicit math."""
    user_df = _make_history_df(USER_FTD_DATES, USER_FTD_CLOSES)
    historical_df = _make_history_df(HISTORICAL_2025_DATES, HISTORICAL_2025_CLOSES)
    _patch_yfinance(monkeypatch, user_df=user_df, historical_df=historical_df)

    r = client.get("/api/market/rally-data?ftd_date=2026-04-08&index=^IXIC")
    body = r.json()
    assert body["day_before_ftd_close"] == USER_DAY_BEFORE_FTD_CLOSE

    # Day 2: 20720 / 20000 - 1 = +3.60%
    assert body["points"][1]["day"] == 2
    assert body["points"][1]["pct"] == pytest.approx(3.60, abs=0.01)


def test_response_field_renamed(client, monkeypatch):
    """The legacy `day0_close` field is gone; new field is `day_before_ftd_close`."""
    user_df = _make_history_df(USER_FTD_DATES, USER_FTD_CLOSES)
    historical_df = _make_history_df(HISTORICAL_2025_DATES, HISTORICAL_2025_CLOSES)
    _patch_yfinance(monkeypatch, user_df=user_df, historical_df=historical_df)

    r = client.get("/api/market/rally-data?ftd_date=2026-04-08&index=^IXIC")
    body = r.json()
    assert "day_before_ftd_close" in body
    assert "day0_close" not in body


def test_historical_2025_first_day_is_ftd(client, monkeypatch):
    """The historical_rally_2025 line's Day 1 is 2025-04-22 (the FTD)
    with pct ≈ +2.71% computed against the 2025-04-21 close."""
    user_df = _make_history_df(USER_FTD_DATES, USER_FTD_CLOSES)
    historical_df = _make_history_df(HISTORICAL_2025_DATES, HISTORICAL_2025_CLOSES)
    _patch_yfinance(monkeypatch, user_df=user_df, historical_df=historical_df)

    r = client.get("/api/market/rally-data?ftd_date=2026-04-08&index=^IXIC")
    body = r.json()
    hist = body["historical_rally_2025"]
    assert hist is not None
    assert hist[0]["day"] == 1
    assert hist[0]["date"] == "2025-04-22"
    assert hist[0]["close"] == 16300.42
    # +2.71% computed in the design report from the verified live IXIC values.
    assert hist[0]["pct"] == pytest.approx(2.71, abs=0.01)


def test_historical_2025_caches_after_first_call(client, monkeypatch):
    """The historical helper is @lru_cache(maxsize=1) — calling the
    endpoint twice should fetch the historical series once. The
    user-line fetch happens on every call (not cached)."""
    user_df = _make_history_df(USER_FTD_DATES, USER_FTD_CLOSES)
    historical_df = _make_history_df(HISTORICAL_2025_DATES, HISTORICAL_2025_CLOSES)
    calls = _patch_yfinance(monkeypatch, user_df=user_df, historical_df=historical_df)

    r1 = client.get("/api/market/rally-data?ftd_date=2026-04-08&index=^IXIC")
    r2 = client.get("/api/market/rally-data?ftd_date=2026-04-08&index=^IXIC")
    assert r1.status_code == 200 and r2.status_code == 200

    user_calls = [c for c in calls if c["kind"] == "user"]
    historical_calls = [c for c in calls if c["kind"] == "historical"]
    # User-line history fetched per request (not cached at this layer).
    assert len(user_calls) == 2
    # Historical-line history fetched once and reused thereafter.
    assert len(historical_calls) == 1


def test_historical_fetch_failure_degrades_gracefully(client, monkeypatch):
    """If the historical-line yfinance fetch raises, the endpoint still
    returns the user line — just with historical_rally_2025 = None."""
    user_df = _make_history_df(USER_FTD_DATES, USER_FTD_CLOSES)
    _patch_yfinance(monkeypatch, user_df=user_df, historical_raises=True)

    r = client.get("/api/market/rally-data?ftd_date=2026-04-08&index=^IXIC")
    body = r.json()
    assert "error" not in body
    # User line is still populated.
    assert len(body["points"]) >= 1
    assert body["points"][0]["day"] == 1
    # Historical line gracefully None.
    assert body["historical_rally_2025"] is None
