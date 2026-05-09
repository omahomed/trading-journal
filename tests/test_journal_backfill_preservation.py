"""Regression tests for journal write paths preserving omitted fields (Phase 2, Commit 3).

The bug: db_layer.save_journal_entry rewrites every column on UPDATE/INSERT.
If a caller omits a key from the journal_entry dict, save_journal_entry
defaults that field — wiping any prior user-entered value.

Two callers had the bug:
  J1 — api/main.py journal_edit (line 706-): omits `status` and `above_21ema`
  J2 — api/main.py journal_backfill_metrics (line 851-): omits `status`,
       `above_21ema`, `mct_display_day_num`, `nlv_source`, `holdings_source`

Fix: both callers now pass-through the existing-row value for each omitted
field, mirroring the recompute-preservation pattern from c0435ee. The shared
load_journal helper was extended to include nlv_source + holdings_source in
its SELECT projection so the preservation read works.

Tests guard each preservation site:
  - J2 backfill: 6 tests (5 preservation + 1 computed-fields-still-work)
  - J1 edit:    4 tests (status preservation x2 — body and existing-row;
                          above_21ema preservation x2 — body and existing-row)
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


# ---------------------------------------------------------------------------
# Shared scaffolding for both J1 and J2 tests
# ---------------------------------------------------------------------------


def _journal_row(day="2026-04-24", *, status="F", above_21ema=1,
                 mct_display_day_num=12,
                 nlv_source="ibkr_auto", holdings_source="ibkr_override",
                 portfolio_heat=0.0, spy_atr=0.0, nasdaq_atr=0.0,
                 market_cycle="POWERTREND"):
    """Build a synthetic journal row in the post-_normalize_journal shape.

    Defaults make the row look like a populated, "all five preservation
    fields set to non-default values" entry — exactly the shape that exposes
    the bug.
    """
    return {
        "day": pd.Timestamp(day),
        "status": status,
        "market_window": "",
        "market_cycle": market_cycle,
        "mct_display_day_num": mct_display_day_num,
        "above_21ema": above_21ema,
        "cash_change": 0.0,
        "beg_nlv": 487703.85,
        "end_nlv": 486630.39,
        "daily_dollar_change": -1073.46,
        "daily_pct_change": -0.22,
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
        "nlv_source": nlv_source,
        "holdings_source": holdings_source,
    }


@pytest.fixture
def journal_stubs(monkeypatch):
    """Yield (state, client) for both J1 and J2 tests.

    state knobs:
      journal_df  — what db.load_journal returns (post-normalize)
      compute_*   — return values for stubbed _compute_* helpers

    state observations:
      saved  — list of dicts passed to db.save_journal_entry
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state: dict[str, Any] = {
        "journal_df": pd.DataFrame(),
        "saved": [],
        "compute_cycle": "POWERTREND",
        "compute_day_num": 12,
        "compute_heat": 7.5,
        "compute_atr_spy": 1.234,
        "compute_atr_ndx": 1.567,
    }

    monkeypatch.setattr(db_layer, "load_journal",
                        lambda *a, **kw: state["journal_df"])
    # Bypass _normalize_journal — the test fixtures use snake_case directly.
    monkeypatch.setattr(main, "_normalize_journal", lambda df: df)

    def fake_save(journal_entry):
        state["saved"].append(dict(journal_entry))
        return 1
    monkeypatch.setattr(db_layer, "save_journal_entry", fake_save)

    # Stub auto-compute helpers so they don't hit yfinance / DB.
    monkeypatch.setattr(main, "_compute_cycle_state",
                        lambda *a, **kw: state["compute_cycle"])
    monkeypatch.setattr(main, "_compute_mct_state_with_day_num",
                        lambda *a, **kw: (state["compute_cycle"],
                                          state["compute_day_num"]))
    monkeypatch.setattr(main, "_compute_portfolio_heat",
                        lambda *a, **kw: state["compute_heat"])
    def fake_atr(ticker, *a, **kw):
        return state["compute_atr_ndx"] if "IXIC" in str(ticker) else state["compute_atr_spy"]
    monkeypatch.setattr(main, "_compute_ticker_atr_pct", fake_atr)

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


# ---------------------------------------------------------------------------
# J2: journal_backfill_metrics preservation (6 tests)
# ---------------------------------------------------------------------------


def _make_backfill_df(**row_overrides):
    """Single-row journal DataFrame in the post-normalize shape.

    By default, sets `portfolio_heat=0` so the backfill loop will mark this
    row as "needs update" and run the save path. (If the row has portfolio_heat
    > 0, backfill skips it as no-op.)
    """
    row_overrides.setdefault("portfolio_heat", 0.0)
    row = _journal_row(**row_overrides)
    return pd.DataFrame([row])


def _post_backfill(client, **body_overrides):
    body = {
        "portfolio": "CanSlim",
        "start_date": "2026-04-24",
        "end_date": "2026-04-24",
        **body_overrides,
    }
    return client.post("/api/journal/backfill-metrics", json=body)


def test_j2_backfill_preserves_user_entered_nlv_source(journal_stubs):
    """Existing row has nlv_source='ibkr_auto'. Backfill runs (need_heat=True
    because portfolio_heat=0). Captured save dict must show 'ibkr_auto', NOT
    the default 'manual'."""
    state, client = journal_stubs
    state["journal_df"] = _make_backfill_df(nlv_source="ibkr_auto",
                                            holdings_source="ibkr_override")

    r = _post_backfill(client)
    assert r.status_code == 200, r.text
    assert state["saved"], "Expected save_journal_entry to fire"

    saved = state["saved"][-1]
    assert saved["nlv_source"] == "ibkr_auto"
    assert saved["holdings_source"] == "ibkr_override"


def test_j2_backfill_preserves_status(journal_stubs):
    """Existing row has status='F' (final). Captured save shows 'F', not 'U'."""
    state, client = journal_stubs
    state["journal_df"] = _make_backfill_df(status="F")

    r = _post_backfill(client)
    assert r.status_code == 200, r.text
    saved = state["saved"][-1]
    assert saved["status"] == "F"


def test_j2_backfill_preserves_above_21ema(journal_stubs):
    """Existing row has above_21ema=1. Captured save shows 1, not 0."""
    state, client = journal_stubs
    state["journal_df"] = _make_backfill_df(above_21ema=1)

    r = _post_backfill(client)
    assert r.status_code == 200, r.text
    saved = state["saved"][-1]
    assert saved["above_21ema"] == 1


def test_j2_backfill_preserves_mct_display_day_num(journal_stubs):
    """Existing row has mct_display_day_num=42. Captured save shows 42, not None."""
    state, client = journal_stubs
    state["journal_df"] = _make_backfill_df(mct_display_day_num=42)

    r = _post_backfill(client)
    assert r.status_code == 200, r.text
    saved = state["saved"][-1]
    assert saved["mct_display_day_num"] == 42


def test_j2_backfill_correctly_updates_computed_fields(journal_stubs):
    """Preservation must not break the actual purpose of the endpoint.
    Existing row has portfolio_heat=0 and spy_atr=0; computed values fire."""
    state, client = journal_stubs
    state["journal_df"] = _make_backfill_df(
        portfolio_heat=0.0, spy_atr=0.0, nasdaq_atr=0.0,
    )
    state["compute_heat"] = 7.5
    state["compute_atr_spy"] = 1.234
    state["compute_atr_ndx"] = 1.567

    r = _post_backfill(client)
    assert r.status_code == 200, r.text
    saved = state["saved"][-1]
    # Computed fields fired
    assert saved["portfolio_heat"] == 7.5
    assert saved["spy_atr"] == 1.234
    assert saved["nasdaq_atr"] == 1.567


def test_j2_backfill_invalid_source_collapses_to_manual(journal_stubs):
    """Defensive guard: if a row somehow has a non-enum nlv_source value
    (shouldn't happen given the CHECK constraint, but defensive at the
    write boundary), the sanitizer collapses it to 'manual'."""
    state, client = journal_stubs
    state["journal_df"] = _make_backfill_df(nlv_source="bogus_value",
                                            holdings_source="also_bogus")

    r = _post_backfill(client)
    assert r.status_code == 200, r.text
    saved = state["saved"][-1]
    assert saved["nlv_source"] == "manual"
    assert saved["holdings_source"] == "manual"


# ---------------------------------------------------------------------------
# J1: journal_edit preservation (4 tests)
# ---------------------------------------------------------------------------


def _post_edit(client, **body_overrides):
    body = {
        "portfolio": "CanSlim",
        "day": "2026-04-24",
        "end_nlv": 486630.39,
        "beg_nlv": 487703.85,
        "cash_change": 0,
        "pct_invested": 0,
        "spy": 715.14,
        "nasdaq": 24880.21,
        "score": 5,
        "highlights": "",
        "mistakes": "",
        "market_window": "",
        **body_overrides,
    }
    return client.post("/api/journal/edit", json=body)


def test_j1_edit_preserves_status_when_body_omits(journal_stubs):
    """Existing row has status='F'. Body omits status. Captured save dict
    shows 'F', not the default 'U'.
    """
    state, client = journal_stubs
    state["journal_df"] = pd.DataFrame([_journal_row(status="F")])

    r = _post_edit(client)
    assert r.status_code == 200, r.text
    assert state["saved"], "Expected save_journal_entry to fire"
    saved = state["saved"][-1]
    assert saved["status"] == "F"


def test_j1_edit_status_body_value_wins(journal_stubs):
    """Body provides status='D'. Existing row has status='F'. Body wins."""
    state, client = journal_stubs
    state["journal_df"] = pd.DataFrame([_journal_row(status="F")])

    r = _post_edit(client, status="D")
    assert r.status_code == 200, r.text
    saved = state["saved"][-1]
    assert saved["status"] == "D"


def test_j1_edit_preserves_above_21ema_when_body_omits(journal_stubs):
    """Existing row has above_21ema=1. Body omits it. Captured save shows 1,
    not the default 0."""
    state, client = journal_stubs
    state["journal_df"] = pd.DataFrame([_journal_row(above_21ema=1)])

    r = _post_edit(client)
    assert r.status_code == 200, r.text
    saved = state["saved"][-1]
    assert saved["above_21ema"] == 1


def test_j1_edit_above_21ema_body_value_wins(journal_stubs):
    """Body provides above_21ema=0. Existing row has 1. Body wins (legitimate
    override — user is recording that price is no longer above 21EMA)."""
    state, client = journal_stubs
    state["journal_df"] = pd.DataFrame([_journal_row(above_21ema=1)])

    r = _post_edit(client, above_21ema=0)
    assert r.status_code == 200, r.text
    saved = state["saved"][-1]
    assert saved["above_21ema"] == 0


def test_j1_edit_first_time_entry_uses_defaults(journal_stubs):
    """No existing row (empty DataFrame). Body omits status + above_21ema.
    Captured save uses safe defaults: status='U' (current schema-aligned
    default), above_21ema=0."""
    state, client = journal_stubs
    state["journal_df"] = pd.DataFrame()  # no existing row

    r = _post_edit(client)
    assert r.status_code == 200, r.text
    saved = state["saved"][-1]
    assert saved["status"] == "U"
    assert saved["above_21ema"] == 0
