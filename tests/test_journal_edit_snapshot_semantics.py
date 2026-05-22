"""Snapshot semantics for /api/journal/edit auto-compute branches.

Regression guard for the bug where the auto-compute branches in
api.main.journal_edit (market_cycle/mct_display_day_num, spy_atr,
nasdaq_atr, portfolio_heat) silently recomputed against "today's"
data when an edit landed on a pre-existing row whose stored value
was 0 or NULL. The truthiness gate (``if not journal_entry[field]:``)
couldn't distinguish "this row never had a value" from "this row's
value is legitimately 0," so a Manage-Logs edit of an existing row
where any of those fields hadn't been backfilled would overwrite
the persisted snapshot with current-date inputs (wrong open
positions, wrong ATR window).

The fix gates all four auto-compute branches on
``existing_row_present``. On a fresh insert the auto-compute fires
normally (snapshot at as_of=today, correct by construction). On an
edit of an existing row, the auto-compute is skipped entirely —
preservation wins. Explicit payload values still flow through to
DB on both paths.

Four scenarios per field (4 fields × 4 = 16 cases total):
  (a) Fresh insert: no row exists → auto-compute fires
  (b) Edit existing non-zero, payload omits → preserved
  (c) Edit existing zero, payload omits → preserved at 0 (the bug
      case — previously the auto-compute would trigger and replace 0
      with a recomputed value)
  (d) Edit with explicit payload override → payload value wins
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


# Deterministic auto-compute return values. Pick numbers that
# unambiguously distinguish "fired" from "preserved 0" or "preserved
# non-zero" in assertions.
_AUTO_CYCLE = "POWERTREND"
_AUTO_DAY_NUM = 12
_AUTO_HEAT = 7.5
_AUTO_SPY_ATR = 1.234
_AUTO_NDX_ATR = 1.567


def _existing_row(day="2026-04-24", *, market_cycle="POWERTREND",
                  mct_display_day_num=12,
                  spy_atr=1.0, nasdaq_atr=1.0, portfolio_heat=5.0):
    """Synthetic post-_normalize_journal row with snake_case columns.
    Defaults represent a fully-populated row; tests override the
    field-under-test to 0 or non-zero as the scenario requires."""
    return {
        "day": pd.Timestamp(day),
        "status": "F",
        "market_window": "",
        "market_cycle": market_cycle,
        "mct_display_day_num": mct_display_day_num,
        "above_21ema": 1,
        "cash_change": 0.0,
        "beg_nlv": 14681.62,
        "end_nlv": 15000.00,
        "daily_dollar_change": 318.38,
        "daily_pct_change": 2.17,
        "pct_invested": 0.0,
        "spy": 715.14,
        "nasdaq": 24880.21,
        "market_notes": "",
        "market_action": "",
        "portfolio_heat": portfolio_heat,
        "spy_atr": spy_atr,
        "nasdaq_atr": nasdaq_atr,
        "score": 5,
        "highlights": "",
        "lowlights": "",
        "mistakes": "",
        "top_lesson": "",
        "daily_thoughts": "",
        "nlv_source": "manual",
        "holdings_source": "manual",
    }


@pytest.fixture
def edit_stubs(monkeypatch):
    """Yield (state, client). state.journal_df controls whether the
    /api/journal/edit handler perceives an existing row; state.saved
    captures every dict passed to db.save_journal_entry so tests can
    assert on the final persisted shape."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {"journal_df": pd.DataFrame(), "saved": []}

    monkeypatch.setattr(db_layer, "load_journal",
                        lambda *a, **kw: state["journal_df"])
    # Bypass column-rename normalization; our fixtures use snake_case.
    monkeypatch.setattr(main, "_normalize_journal", lambda df: df)

    def fake_save(journal_entry):
        state["saved"].append(dict(journal_entry))
        return 1
    monkeypatch.setattr(db_layer, "save_journal_entry", fake_save)

    monkeypatch.setattr(main, "_compute_mct_state_with_day_num",
                        lambda *a, **kw: (_AUTO_CYCLE, _AUTO_DAY_NUM))
    monkeypatch.setattr(main, "_compute_portfolio_heat",
                        lambda *a, **kw: _AUTO_HEAT)

    def fake_atr(ticker, *a, **kw):
        return _AUTO_NDX_ATR if "IXIC" in str(ticker) else _AUTO_SPY_ATR
    monkeypatch.setattr(main, "_compute_ticker_atr_pct", fake_atr)

    client = TestClient(main.app, headers=_auth_headers())
    yield state, client


def _post_edit(client, **payload_overrides):
    body = {"portfolio": "CanSlim", "day": "2026-04-24", **payload_overrides}
    r = client.post("/api/journal/edit", json=body)
    assert r.status_code == 200, r.text
    return r


def _last_saved(state):
    assert state["saved"], "Expected db.save_journal_entry to fire"
    return state["saved"][-1]


# ─────────────────────────────────────────────────────────────────────────────
# portfolio_heat — 4 cases
# ─────────────────────────────────────────────────────────────────────────────


class TestPortfolioHeatSnapshot:
    def test_fresh_insert_fires_auto_compute(self, edit_stubs):
        """No existing row → auto-compute fires → persisted value is the
        deterministic _AUTO_HEAT."""
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame()                 # no row
        _post_edit(client, end_nlv=15000)
        assert _last_saved(state)["portfolio_heat"] == pytest.approx(_AUTO_HEAT)

    def test_edit_existing_nonzero_preserved(self, edit_stubs):
        """Existing row has portfolio_heat=5.0, payload omits → preserved
        at 5.0 (not recomputed to _AUTO_HEAT)."""
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame([_existing_row(portfolio_heat=5.0)])
        _post_edit(client)
        assert _last_saved(state)["portfolio_heat"] == pytest.approx(5.0)

    def test_edit_existing_zero_preserved(self, edit_stubs):
        """REGRESSION: existing row has portfolio_heat=0 (e.g., imported
        without backfill yet). Edit must NOT recompute against today's
        positions — the 0 must stay 0 until an explicit gap-fill is run."""
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame([_existing_row(portfolio_heat=0.0)])
        _post_edit(client)
        assert _last_saved(state)["portfolio_heat"] == pytest.approx(0.0)

    def test_edit_explicit_override_wins(self, edit_stubs):
        """Explicit payload value wins regardless of prior state."""
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame([_existing_row(portfolio_heat=5.0)])
        _post_edit(client, portfolio_heat=12.5)
        assert _last_saved(state)["portfolio_heat"] == pytest.approx(12.5)


# ─────────────────────────────────────────────────────────────────────────────
# spy_atr — 4 cases
# ─────────────────────────────────────────────────────────────────────────────


class TestSpyAtrSnapshot:
    def test_fresh_insert_fires_auto_compute(self, edit_stubs):
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame()
        _post_edit(client, end_nlv=15000)
        assert _last_saved(state)["spy_atr"] == pytest.approx(_AUTO_SPY_ATR)

    def test_edit_existing_nonzero_preserved(self, edit_stubs):
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame([_existing_row(spy_atr=5.0)])
        _post_edit(client)
        assert _last_saved(state)["spy_atr"] == pytest.approx(5.0)

    def test_edit_existing_zero_preserved(self, edit_stubs):
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame([_existing_row(spy_atr=0.0)])
        _post_edit(client)
        assert _last_saved(state)["spy_atr"] == pytest.approx(0.0)

    def test_edit_explicit_override_wins(self, edit_stubs):
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame([_existing_row(spy_atr=5.0)])
        _post_edit(client, spy_atr=12.5)
        assert _last_saved(state)["spy_atr"] == pytest.approx(12.5)


# ─────────────────────────────────────────────────────────────────────────────
# nasdaq_atr — 4 cases
# ─────────────────────────────────────────────────────────────────────────────


class TestNasdaqAtrSnapshot:
    def test_fresh_insert_fires_auto_compute(self, edit_stubs):
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame()
        _post_edit(client, end_nlv=15000)
        assert _last_saved(state)["nasdaq_atr"] == pytest.approx(_AUTO_NDX_ATR)

    def test_edit_existing_nonzero_preserved(self, edit_stubs):
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame([_existing_row(nasdaq_atr=5.0)])
        _post_edit(client)
        assert _last_saved(state)["nasdaq_atr"] == pytest.approx(5.0)

    def test_edit_existing_zero_preserved(self, edit_stubs):
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame([_existing_row(nasdaq_atr=0.0)])
        _post_edit(client)
        assert _last_saved(state)["nasdaq_atr"] == pytest.approx(0.0)

    def test_edit_explicit_override_wins(self, edit_stubs):
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame([_existing_row(nasdaq_atr=5.0)])
        _post_edit(client, nasdaq_atr=12.5)
        assert _last_saved(state)["nasdaq_atr"] == pytest.approx(12.5)


# ─────────────────────────────────────────────────────────────────────────────
# market_cycle — 4 cases
# (mct_display_day_num is paired in the same auto-compute branch; the
# cycle cases also exercise the integer field's preservation by side
# effect of the shared `if not existing_row_present` gate.)
# ─────────────────────────────────────────────────────────────────────────────


class TestMarketCycleSnapshot:
    def test_fresh_insert_fires_auto_compute(self, edit_stubs):
        """No existing row → MCT engine replay fires → cycle persisted
        as deterministic _AUTO_CYCLE, day_num as _AUTO_DAY_NUM."""
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame()
        _post_edit(client, end_nlv=15000)
        saved = _last_saved(state)
        assert saved["market_cycle"] == _AUTO_CYCLE
        assert saved["mct_display_day_num"] == _AUTO_DAY_NUM

    def test_edit_existing_nonempty_preserved(self, edit_stubs):
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame([_existing_row(
            market_cycle="CORRECTION", mct_display_day_num=3,
        )])
        _post_edit(client)
        saved = _last_saved(state)
        assert saved["market_cycle"] == "CORRECTION"
        assert saved["mct_display_day_num"] == 3

    def test_edit_existing_empty_preserved(self, edit_stubs):
        """REGRESSION: row exists with market_cycle='' and
        mct_display_day_num=None. Edit must NOT re-stamp from the
        engine — preservation wins."""
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame([_existing_row(
            market_cycle="", mct_display_day_num=None,
        )])
        _post_edit(client)
        saved = _last_saved(state)
        assert saved["market_cycle"] == ""
        # save_journal_entry receives None unchanged; the column allows NULL.
        assert saved["mct_display_day_num"] is None

    def test_edit_explicit_override_wins(self, edit_stubs):
        state, client = edit_stubs
        state["journal_df"] = pd.DataFrame([_existing_row(
            market_cycle="POWERTREND", mct_display_day_num=12,
        )])
        _post_edit(client, market_cycle="RALLY_ATTEMPT", mct_display_day_num=7)
        saved = _last_saved(state)
        assert saved["market_cycle"] == "RALLY_ATTEMPT"
        assert saved["mct_display_day_num"] == 7
