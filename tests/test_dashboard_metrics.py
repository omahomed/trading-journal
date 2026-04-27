"""Tests for nlv_service.dashboard_metrics — the read view powering the
dashboard refactor.

Stubs db.load_journal + compute_nlv (via monkeypatch) so the tests run
without a database. Mirrors the pattern in test_manual_price_override.py.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _journal_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a journal DataFrame in the shape db.load_journal returns.

    Column names match what normalize_journal_columns produces — the
    function under test re-normalises but only the standard names are
    needed since none of these tests exercise alternate-column compat."""
    cols = ["day", "end_nlv", "beg_nlv", "cash_change", "pct_invested",
            "daily_dollar_change", "daily_pct_change"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def _live_snapshot(nlv: float = 487291.22, cash: float = -430868.40,
                   market_value: float = 918159.62) -> dict[str, Any]:
    """Canonical compute_nlv return shape for the happy path."""
    return {
        "cash": cash,
        "market_value": market_value,
        "nlv": nlv,
        "positions": [],
        "as_of": "2026-04-27T13:30:00",
    }


@pytest.fixture
def stubbed(monkeypatch):
    """Yield (configure, run) — configure stubs, then run dashboard_metrics."""
    import nlv_service

    state: dict[str, Any] = {
        "journal": None, "live": None, "live_raises": False,
        "net_contributions": 0.0, "net_contrib_raises": False,
    }

    monkeypatch.setattr(nlv_service.db, "load_journal",
                        lambda name: state["journal"])

    def fake_compute_nlv(pid, name):
        if state["live_raises"]:
            raise RuntimeError("yfinance unreachable")
        return state["live"]
    monkeypatch.setattr(nlv_service, "compute_nlv", fake_compute_nlv)

    def fake_net_contrib(pid):
        if state["net_contrib_raises"]:
            raise RuntimeError("ledger unreachable")
        return state["net_contributions"]
    monkeypatch.setattr(nlv_service.db, "get_net_contributions", fake_net_contrib)

    def configure(*, journal=None, live=None, live_raises: bool = False,
                  net_contributions: float = 0.0,
                  net_contrib_raises: bool = False):
        state["journal"] = journal
        state["live"] = live
        state["live_raises"] = live_raises
        state["net_contributions"] = net_contributions
        state["net_contrib_raises"] = net_contrib_raises

    return configure, lambda: nlv_service.dashboard_metrics(1, "CanSlim")


# ---------------------------------------------------------------------------
# Real prod-data round-trip
#
# This is the load-bearing test: with the user's actual prod values,
# compute-on-read must reconstruct total_holdings within $0.01 of the
# broker's number ($917,498.79). If this drifts, the whole "no schema
# change" refinement falls apart.
# ---------------------------------------------------------------------------


def test_total_holdings_round_trips_within_one_cent(stubbed):
    """The frontend computes pct_invested = total_holdings / end_nlv * 100
    in JS float64 (yielding 188.5412026980066) and Postgres NUMERIC(10,4)
    stores it as 188.5412. Reading back: 486630.39 × 188.5412 / 100 =
    917498.776... → rounded to $917,498.78. Net round-trip loss is one
    cent for a $900k+ holding — invisible at the dashboard's whole-dollar
    display precision. This test pins that contract: any future change
    to pct_invested storage must preserve cent-level round-trip."""
    end_nlv = 486630.39
    total_holdings_intended = 917498.79
    # Mirror the frontend's JS float computation, then Postgres' NUMERIC(10,4)
    # rounding. This is the value that actually lives in the DB.
    pct_invested_stored = round(total_holdings_intended / end_nlv * 100, 4)
    assert pct_invested_stored == 188.5412  # sanity-check the fixture itself

    configure, run = stubbed
    configure(
        journal=_journal_df([
            {"day": "2026-04-23", "end_nlv": 471004.89, "beg_nlv": 471004.89,
             "cash_change": 0, "pct_invested": pct_invested_stored,
             "daily_dollar_change": 0, "daily_pct_change": 0},
            {"day": "2026-04-24", "end_nlv": end_nlv, "beg_nlv": 471004.89,
             "cash_change": 0, "pct_invested": pct_invested_stored,
             "daily_dollar_change": 13496.0, "daily_pct_change": 2.85},
        ]),
        live=_live_snapshot(),
    )

    out = run()

    # Within one cent of the broker-stated value. Compare as integer cents
    # so we don't trip on float noise (abs(0.01 - 0.01) lands near 9e-12).
    assert round(abs(out["total_holdings"] - total_holdings_intended) * 100) <= 1
    assert round(abs(out["cash"] - (end_nlv - total_holdings_intended)) * 100) <= 1
    # exposure_pct mirrors the stored pct_invested verbatim (no extra round-trip)
    assert out["exposure_pct"] == pct_invested_stored


# ---------------------------------------------------------------------------
# Happy path — every journal-derived field populated
# ---------------------------------------------------------------------------


def test_full_journal_returns_all_journal_fields(stubbed):
    configure, run = stubbed
    configure(
        journal=_journal_df([
            {"day": "2026-04-23", "end_nlv": 480000.00, "beg_nlv": 470000.00,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 10000.0, "daily_pct_change": 2.13},
            {"day": "2026-04-24", "end_nlv": 486630.39, "beg_nlv": 480000.00,
             "cash_change": 0, "pct_invested": 188.5413,
             "daily_dollar_change": 6630.39, "daily_pct_change": 1.38},
        ]),
        live=_live_snapshot(),
    )

    out = run()

    assert out["journal_available"] is True
    assert out["as_of_date"] == "2026-04-24"
    assert out["nlv"] == 486630.39
    # Deltas come straight off the latest row's stored values.
    assert out["nlv_delta_dollar"] == 6630.39
    assert out["nlv_delta_pct"] == 1.38
    # Drawdown: peak is the latest value (it's the highest), current = peak,
    # so drawdown is 0%.
    assert out["drawdown_peak_nlv"] == 486630.39
    assert out["drawdown_peak_date"] == "2026-04-24"
    assert out["drawdown_current_pct"] == 0.0


def test_drawdown_when_current_below_peak(stubbed):
    """A real drawdown — peak Friday, current Monday is lower."""
    configure, run = stubbed
    configure(
        journal=_journal_df([
            {"day": "2026-04-24", "end_nlv": 500000.00, "beg_nlv": 0,
             "cash_change": 0, "pct_invested": 0,
             "daily_dollar_change": 0, "daily_pct_change": 0},
            {"day": "2026-04-27", "end_nlv": 475000.00, "beg_nlv": 500000.00,
             "cash_change": 0, "pct_invested": 0,
             "daily_dollar_change": -25000.00, "daily_pct_change": -5.0},
        ]),
        live=_live_snapshot(),
    )

    out = run()

    assert out["drawdown_peak_nlv"] == 500000.00
    assert out["drawdown_peak_date"] == "2026-04-24"
    # (475000 - 500000) / 500000 * 100 = -5.0
    assert out["drawdown_current_pct"] == -5.0


# ---------------------------------------------------------------------------
# Edge cases the spec calls out explicitly
# ---------------------------------------------------------------------------


def test_no_journal_entries_returns_empty_shape_with_flag(stubbed):
    """Brand-new portfolio: nothing to display from the journal. The
    response keys still exist (frontend can render with optional chains)
    but every journal field is None and `journal_available` is false."""
    configure, run = stubbed
    configure(journal=pd.DataFrame(), live=_live_snapshot())

    out = run()

    assert out["journal_available"] is False
    assert out["as_of_date"] is None
    assert out["nlv"] is None
    assert out["nlv_delta_dollar"] is None
    assert out["total_holdings"] is None
    assert out["exposure_pct"] is None
    assert out["cash"] is None
    assert out["drawdown_current_pct"] is None
    assert out["drawdown_peak_nlv"] is None
    assert out["ltd_pct"] is None
    assert out["ytd_pct"] is None


def test_first_journal_entry_has_null_deltas(stubbed):
    """One row → no previous day to compare against → deltas are null
    (the spec disallows showing '+$0.00 (+0.00%)' in that case)."""
    configure, run = stubbed
    configure(
        journal=_journal_df([
            {"day": "2026-04-24", "end_nlv": 486630.39, "beg_nlv": 0,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 0, "daily_pct_change": 0},
        ]),
        live=_live_snapshot(),
    )

    out = run()

    assert out["journal_available"] is True
    assert out["nlv"] == 486630.39
    assert out["nlv_delta_dollar"] is None
    assert out["nlv_delta_pct"] is None


def test_live_estimate_fields_when_compute_nlv_succeeds(stubbed):
    """Diff = live - journal; diff_pct = diff / journal * 100. The field
    is what the NLV tile's small grey sub-label renders below the headline."""
    configure, run = stubbed
    configure(
        journal=_journal_df([
            {"day": "2026-04-24", "end_nlv": 486630.39, "beg_nlv": 0,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 0, "daily_pct_change": 0},
        ]),
        live=_live_snapshot(nlv=487291.22),
    )

    out = run()

    assert out["live_estimate_unavailable"] is False
    assert out["live_estimate_nlv"] == 487291.22
    assert out["live_estimate_diff"] == round(487291.22 - 486630.39, 2)
    # 660.83 / 486630.39 * 100 ≈ 0.1358%
    assert abs(out["live_estimate_diff_pct"] - 0.1358) < 0.001


def test_live_estimate_unavailable_when_compute_nlv_raises(stubbed):
    """yfinance / DB / network blow-up in compute_nlv must not break the
    journal-derived fields. live_estimate_unavailable=true is the contract."""
    configure, run = stubbed
    configure(
        journal=_journal_df([
            {"day": "2026-04-24", "end_nlv": 486630.39, "beg_nlv": 0,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 0, "daily_pct_change": 0},
        ]),
        live_raises=True,
    )

    out = run()

    # Journal fields still populated
    assert out["journal_available"] is True
    assert out["nlv"] == 486630.39
    # Live fields all None + the unavailable flag set
    assert out["live_estimate_unavailable"] is True
    assert out["live_estimate_nlv"] is None
    assert out["live_estimate_diff"] is None
    assert out["live_estimate_diff_pct"] is None


def test_live_estimate_unavailable_does_not_block_no_journal_case(stubbed):
    """No journal AND compute_nlv blew up — both classes of failure can
    coexist; response is just the 'all fields null + flags true' shape."""
    configure, run = stubbed
    configure(journal=pd.DataFrame(), live_raises=True)

    out = run()

    assert out["journal_available"] is False
    assert out["live_estimate_unavailable"] is True
    assert out["nlv"] is None
    assert out["live_estimate_nlv"] is None


def test_live_estimate_no_anchor_when_journal_nlv_zero(stubbed):
    """Defensive — a journal entry with end_nlv=0 (corrupted? test data?)
    would divide-by-zero on diff_pct. Surface live nlv alone, no diff."""
    configure, run = stubbed
    configure(
        journal=_journal_df([
            {"day": "2026-04-24", "end_nlv": 0, "beg_nlv": 0,
             "cash_change": 0, "pct_invested": 0,
             "daily_dollar_change": 0, "daily_pct_change": 0},
        ]),
        live=_live_snapshot(nlv=487291.22),
    )

    out = run()

    assert out["live_estimate_nlv"] == 487291.22
    assert out["live_estimate_diff"] is None
    assert out["live_estimate_diff_pct"] is None
    assert out["live_estimate_unavailable"] is False


# ---------------------------------------------------------------------------
# Drawdown edge — peak hit twice, report earlier date (more conservative)
# ---------------------------------------------------------------------------


def test_drawdown_peak_date_uses_first_occurrence(stubbed):
    """If the max NLV occurs on multiple days, peak_date is the earliest.
    Makes 'days since peak' longer / more conservative."""
    configure, run = stubbed
    configure(
        journal=_journal_df([
            {"day": "2026-04-22", "end_nlv": 500000, "beg_nlv": 0,
             "cash_change": 0, "pct_invested": 0,
             "daily_dollar_change": 0, "daily_pct_change": 0},
            {"day": "2026-04-23", "end_nlv": 490000, "beg_nlv": 500000,
             "cash_change": 0, "pct_invested": 0,
             "daily_dollar_change": -10000, "daily_pct_change": -2.0},
            {"day": "2026-04-24", "end_nlv": 500000, "beg_nlv": 490000,
             "cash_change": 0, "pct_invested": 0,
             "daily_dollar_change": 10000, "daily_pct_change": 2.04},
        ]),
        live=_live_snapshot(),
    )

    out = run()

    assert out["drawdown_peak_nlv"] == 500000
    # First time we hit 500k
    assert out["drawdown_peak_date"] == "2026-04-22"
    # Current matches peak so drawdown is 0
    assert out["drawdown_current_pct"] == 0.0


# ---------------------------------------------------------------------------
# Dollar P&L (LTD + YTD) — sub-label values for the LTD/YTD tiles
# ---------------------------------------------------------------------------


def test_ltd_pl_dollar_uses_journal_nlv_minus_net_contributions(stubbed):
    """LTD P&L dollar = current journal NLV − cash-ledger net contributions.
    Spec: '$+339,829'. Anchored on journal NLV (the dashboard headline)
    not live NLV — keeps the tile internally consistent."""
    configure, run = stubbed
    configure(
        journal=_journal_df([
            {"day": "2026-04-24", "end_nlv": 486630.39, "beg_nlv": 0,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 0, "daily_pct_change": 0},
        ]),
        live=_live_snapshot(),
        net_contributions=146801.39,  # so ltd_pl_dollar = 339,829.00
    )

    out = run()

    assert out["ltd_pl_dollar"] == 339829.00


def test_ltd_pl_dollar_none_when_ledger_lookup_fails(stubbed):
    """db.get_net_contributions raising shouldn't break the dashboard.
    Surface the journal/TWR fields as normal, set ltd_pl_dollar=None,
    and the frontend falls back to a static sub-label."""
    configure, run = stubbed
    configure(
        journal=_journal_df([
            {"day": "2026-04-24", "end_nlv": 486630.39, "beg_nlv": 0,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 0, "daily_pct_change": 0},
        ]),
        live=_live_snapshot(),
        net_contrib_raises=True,
    )

    out = run()

    assert out["ltd_pl_dollar"] is None
    # Other journal fields still present
    assert out["nlv"] == 486630.39
    assert out["ltd_pct"] is not None


def test_ytd_pl_dollar_uses_prior_year_end_baseline(stubbed):
    """YTD baseline preference #1: end_nlv of the last journal entry of
    the prior year. Captures the broker-confirmed Dec 31 close cleanly."""
    configure, run = stubbed
    # Use the actual current year so the YTD filter inside dashboard_metrics
    # picks up the rows we expect — relative dates would mis-match in 2027+.
    from datetime import datetime as _dt
    yr = _dt.now().year
    configure(
        journal=_journal_df([
            # Prior year — last row is the YTD baseline
            {"day": f"{yr-1}-12-30", "end_nlv": 350000, "beg_nlv": 348000,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 2000, "daily_pct_change": 0.57},
            {"day": f"{yr-1}-12-31", "end_nlv": 362267.39, "beg_nlv": 350000,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 12267.39, "daily_pct_change": 3.5},
            # Current year — current NLV plus the cash flows we paid in
            {"day": f"{yr}-04-24", "end_nlv": 486630.39, "beg_nlv": 482000,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 4630.39, "daily_pct_change": 0.96},
        ]),
        live=_live_snapshot(),
        net_contributions=146801.39,
    )

    out = run()

    # 486630.39 - 362267.39 - 0 = 124,363.00
    assert out["ytd_pl_dollar"] == 124363.00
    assert out["ytd_available"] is True


def test_ytd_pl_dollar_subtracts_intra_year_cash_flows(stubbed):
    """If the user deposited or withdrew during the year, YTD P&L must
    net those out — otherwise a $50k mid-year deposit would inflate
    YTD P&L by $50k. cash_change column is summed across year rows."""
    from datetime import datetime as _dt
    yr = _dt.now().year
    configure, run = stubbed
    configure(
        journal=_journal_df([
            {"day": f"{yr-1}-12-31", "end_nlv": 400000, "beg_nlv": 0,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 0, "daily_pct_change": 0},
            {"day": f"{yr}-02-01", "end_nlv": 460000, "beg_nlv": 410000,
             "cash_change": 50000,  # $50k deposit
             "pct_invested": 100.0,
             "daily_dollar_change": 0, "daily_pct_change": 0},
            {"day": f"{yr}-04-24", "end_nlv": 540000, "beg_nlv": 500000,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 0, "daily_pct_change": 0},
        ]),
        live=_live_snapshot(),
        net_contributions=450000,
    )

    out = run()

    # 540000 - 400000 (baseline) - 50000 (cash flow) = 90000 of true gain
    assert out["ytd_pl_dollar"] == 90000.00


def test_ytd_pl_dollar_falls_back_to_first_current_year_beg_nlv(stubbed):
    """Portfolio that opened mid-year has no prior-year row. Use the
    first current-year row's beg_nlv as the baseline (yesterday's close,
    since beg_nlv is yesterday's end_nlv)."""
    from datetime import datetime as _dt
    yr = _dt.now().year
    configure, run = stubbed
    configure(
        journal=_journal_df([
            # Only current-year rows — opened this year
            {"day": f"{yr}-03-15", "end_nlv": 100000, "beg_nlv": 95000,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 5000, "daily_pct_change": 5.26},
            {"day": f"{yr}-04-24", "end_nlv": 150000, "beg_nlv": 145000,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 5000, "daily_pct_change": 3.45},
        ]),
        live=_live_snapshot(),
        net_contributions=95000,
    )

    out = run()

    # baseline = first row's beg_nlv = 95000; current = 150000
    # 150000 - 95000 - 0 = 55000
    assert out["ytd_pl_dollar"] == 55000.00


def test_ytd_pl_dollar_none_when_no_current_year_rows(stubbed):
    """Defensive — if every journal row predates the current year, YTD
    isn't meaningful. dashboard_metrics already gates on twr_ytd_available
    but let the helper prove it independently for unit-test confidence."""
    from datetime import datetime as _dt
    yr = _dt.now().year
    configure, run = stubbed
    configure(
        journal=_journal_df([
            {"day": f"{yr-1}-06-15", "end_nlv": 100000, "beg_nlv": 0,
             "cash_change": 0, "pct_invested": 100.0,
             "daily_dollar_change": 0, "daily_pct_change": 0},
        ]),
        live=_live_snapshot(),
        net_contributions=100000,
    )

    out = run()

    # ytd_available is False, so the helper short-circuits to None.
    assert out["ytd_available"] is False
    assert out["ytd_pl_dollar"] is None


def test_empty_journal_dollar_pl_fields_are_null(stubbed):
    """No-journal case: confirm the new fields appear in the empty
    response shape (frontend reads with optional chains; the keys must
    exist or we'd get TS narrowing surprises)."""
    configure, run = stubbed
    configure(journal=pd.DataFrame(), live=_live_snapshot())

    out = run()

    assert "ltd_pl_dollar" in out and out["ltd_pl_dollar"] is None
    assert "ytd_pl_dollar" in out and out["ytd_pl_dollar"] is None
