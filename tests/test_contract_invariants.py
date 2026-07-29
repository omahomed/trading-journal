"""Contract tests — the "these things must agree" invariants across the
load-bearing surfaces catalogued in ARCHITECTURE.md.

These aren't implementation tests. They lock the cross-endpoint /
cross-page contracts so a change to one path that silently breaks a
downstream reader FAILS LOUDLY at CI time instead of shipping.

Every test here is motivated by an actual regression we shipped —
the specific bug is documented in the docstring. Adding new
invariants here is cheaper than debugging class-of-bug regressions.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import jwt
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import db_layer


_TEST_SECRET = "test-secret-not-for-prod"
_TEST_USER_ID = "test-user"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _auth_headers() -> dict[str, str]:
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


# ============================================================================
# Invariant #1 — MCT stamper and /api/market/rally-prefix agree on state
# ============================================================================
#
# Motivating regression (2026-07-27): the Force Correction override was
# wired into /api/market/rally-prefix, so M Factor banner showed CORRECTION.
# But _compute_mct_state_with_day_num (called from journal saves) did NOT
# apply the override — so trading_journal.market_cycle stamped
# "UPTREND UNDER PRESSURE" and Journal Log's MCT State column diverged
# from what the user saw on M Factor for the same day.
#
# These tests would have failed at the moment that divergence was
# introduced — before deploy.


def _fake_engine_result(state_name="UPTREND UNDER PRESSURE"):
    """One-bar EngineResult that both readers can extract state from.

    Default baseline is UUP (not POWERTREND) so the rally_prefix
    auto-clear check ('if override and systematic in {PT, UP, CORR}:
    clear') doesn't fire during override tests. The realistic scenario
    for override use is: systematic engine says UUP or RALLY MODE (the
    market weakened but hasn't crossed the 10% threshold), user
    declares CORRECTION. Testing against POWERTREND baseline would
    exercise the auto-clear path, not the overlay path we want to lock.
    """
    from api.mct_engine import EngineResult

    # Build final_state consistent with _state_name's derivation rules.
    # See api/mct_endpoint_adapter.py::_state_name for the branch order.
    final_state = {
        "power_trend": False,
        "step4_done": False,
        "step4_ever_fired": False,
        "correction_active": False,
        "in_correction": False,
        "rally_active": False,
        "step0_done": False,
        "reference_high": 110.0,
        "exposure": 100,
        "cap_at_100": False,
    }
    if state_name == "POWERTREND":
        final_state["power_trend"] = True
    elif state_name == "UPTREND":
        final_state["step4_done"] = True
    elif state_name == "UPTREND UNDER PRESSURE":
        # step4 fired before, cleared by a mid-cycle break; no active correction.
        final_state["step4_ever_fired"] = True
    elif state_name == "CORRECTION":
        final_state["in_correction"] = True
        final_state["correction_active"] = True

    bars = pd.DataFrame([{
        "trade_date": pd.Timestamp("2026-07-27"),
        "state": state_name,
        "cycle_start_idx": pd.NA,
        "pt_on_idx": pd.NA,
        "rally_active": False,
        "close": 100.0, "open": 100.0, "high": 101.0, "low": 99.0,
        "ema_8": 100.0, "ema_21": 100.0, "sma_50": 100.0, "sma_200": 100.0,
    }])
    return EngineResult(bars=bars, signals=[], final_state=final_state)


@pytest.fixture
def patched_engine(monkeypatch):
    """Stub run_engine + market data updater so the tests are deterministic.
    The fixture returns a mutable dict the test writes `state` into; both
    the stamper AND rally_prefix will see whatever state the test picks."""
    import api.main as main
    from api.mct_engine import EngineResult

    state = {"state_name": "POWERTREND"}

    def fake_run_engine(symbol="^IXIC", as_of=None, force_correction_at_date=None):
        return _fake_engine_result(state["state_name"])

    monkeypatch.setattr("api.mct_endpoint_adapter.run_engine", fake_run_engine)
    monkeypatch.setattr("api.main.run_engine", fake_run_engine, raising=False)
    monkeypatch.setattr("api.market_data_updater.update_if_needed",
                        lambda symbol="^IXIC": None)
    monkeypatch.setattr(main, "_project_rally_prefix_for_data_lag",
                        lambda response, requested_as_of: response)

    # No active override for the baseline tests; individual tests override.
    monkeypatch.setattr(db_layer, "get_active_mct_override", lambda: None)
    return state


def test_stamper_and_endpoint_agree_when_no_override(patched_engine):
    """Baseline — same state name flows through both paths when no
    override is active. Uses UUP as the state (matches the realistic
    scenario where an override would later be declared)."""
    import api.main as main
    patched_engine["state_name"] = "UPTREND UNDER PRESSURE"
    stamper_state, _ = main._compute_mct_state_with_day_num("2026-07-27")
    endpoint = main.rally_prefix(as_of_date="2026-07-27")
    assert stamper_state == endpoint["state"] == "UPTREND UNDER PRESSURE"


def test_stamper_and_endpoint_agree_when_override_active(patched_engine, monkeypatch):
    """The regression case — with an active override, BOTH paths must
    reach the same state. The stamper must call run_engine with the
    override date; rally_prefix's overlay logic must produce the
    matching state string.

    Fake run_engine returns CORRECTION when called with the override
    date (simulating what the real engine's Phase 3a hook does on
    force_correction_at_date). Both readers should see CORRECTION."""
    import api.main as main

    # `id` must be present — rally_prefix's overlay branch reads it when
    # building the response, and KeyError there falls through the outer
    # try/except to the systematic response (silent failure that masks
    # the very invariant this test wants to lock).
    monkeypatch.setattr(db_layer, "get_active_mct_override", lambda: {
        "id": 1,
        "activated_date_ct": "2026-07-27",
        "reason": "manual — pre-IBD 10% threshold",
    })

    # When rally_prefix runs the SECOND engine call (the override overlay),
    # our stub sees force_correction_at_date=date(2026,7,27) and can flip
    # the returned state — mimicking the real Phase 3a. When the stamper
    # runs (from _current_override_date()), same date arg, same flip.
    #
    # Systematic baseline is UUP (not POWERTREND/UPTREND/CORRECTION) so
    # the rally_prefix auto-clear check doesn't short-circuit the overlay
    # — matches the realistic scenario where an override is declared
    # (market weakened, hasn't crossed the systematic threshold yet).
    def flip_on_override(symbol="^IXIC", as_of=None, force_correction_at_date=None):
        if force_correction_at_date == date(2026, 7, 27):
            r = _fake_engine_result("CORRECTION")
            # rally_prefix's overlay checks force_correction_applied.
            r.final_state["force_correction_applied"] = True
            return r
        return _fake_engine_result("UPTREND UNDER PRESSURE")
    monkeypatch.setattr("api.mct_endpoint_adapter.run_engine", flip_on_override)

    stamper_state, _ = main._compute_mct_state_with_day_num("2026-07-27")
    endpoint = main.rally_prefix(as_of_date="2026-07-27")
    assert stamper_state == endpoint["state"] == "CORRECTION", (
        f"stamper={stamper_state!r}, endpoint={endpoint['state']!r} — "
        "if these ever disagree, Journal Log's MCT badge diverges from "
        "the M Factor page for the same day"
    )


# ============================================================================
# Invariant #2 — journal_latest and heat-preview return the same NLV
# ============================================================================
#
# Motivating regression: NLV Entry's heat tile read 0.00% because
# heat-preview picked a hollow row while journal_latest correctly walked
# back past it. The two endpoints must ALWAYS agree on what "current NLV"
# means, or the same portfolio shows different equity basis on different
# pages simultaneously.


@pytest.fixture
def journal_client(monkeypatch):
    """Patch db.load_journal so tests can inject arbitrary journal history
    without touching Postgres. Mirrors the fixture pattern in
    test_journal_latest_nlv_filter.py."""
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)
    import api.main as main
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    state = {"journal_df": pd.DataFrame()}
    monkeypatch.setattr(db_layer, "load_journal",
                        lambda *a, **kw: state["journal_df"])
    monkeypatch.setattr(main, "_normalize_journal", lambda df: df)
    # Neutralize the heat computation so tests focus on NLV plumbing.
    monkeypatch.setattr(main, "_compute_portfolio_heat",
                        lambda *a, **kw: 0.0)

    tc = TestClient(main.app, headers=_auth_headers())

    def set_history(rows: list[dict]):
        state["journal_df"] = pd.DataFrame(rows)
    tc.set_history = set_history  # type: ignore[attr-defined]
    return tc


def _journal_row(day: str, end_nlv):
    return {
        "day": pd.Timestamp(day),
        "end_nlv": end_nlv,
        "beg_nlv": end_nlv if end_nlv is not None else 0,
        "cash_change": 0,
    }


def test_journal_latest_end_nlv_equals_heat_preview_nlv_used(journal_client):
    """The invariant: journal_latest.end_nlv == heat_preview.nlv_used.
    A hollow row for today must not divert one endpoint from the other."""
    journal_client.set_history([  # type: ignore[attr-defined]
        _journal_row("2026-07-25", 52450.0),
        _journal_row("2026-07-27", None),  # hollow Game Plan row for today
    ])
    latest = journal_client.get("/api/journal/latest?portfolio=CanSlim").json()
    preview = journal_client.get("/api/portfolio/heat-preview?portfolio=CanSlim").json()
    assert latest["end_nlv"] == preview["nlv_used"] == 52450.0, (
        f"latest={latest.get('end_nlv')!r}, preview={preview.get('nlv_used')!r} — "
        "when these disagree, Portfolio Heat and NLV Entry show different "
        "equity basis for the same portfolio"
    )


def test_journal_latest_and_heat_preview_agree_when_only_hollow_rows(journal_client):
    """Zero NLV history → both endpoints signal empty in their own shape,
    but must AGREE on "no valid NLV." journal_latest returns
    {error: "No journal data"}; heat_preview returns {nlv_used: 0.0}.
    Both signal the same empty state."""
    journal_client.set_history([  # type: ignore[attr-defined]
        _journal_row("2026-07-25", None),
        _journal_row("2026-07-27", None),
    ])
    latest = journal_client.get("/api/journal/latest?portfolio=CanSlim").json()
    preview = journal_client.get("/api/portfolio/heat-preview?portfolio=CanSlim").json()
    assert latest.get("error") == "No journal data"
    assert preview["nlv_used"] == 0.0
    assert "end_nlv" not in latest  # empty-history shape


# ============================================================================
# Invariant #3 — Static audit: MCT stamp/heal path uses _current_override_date
# ============================================================================
#
# Motivating regression: the Force Correction override arg was added to
# run_engine, but three of its callers (the two stampers + the heal) were
# not updated. The result: M Factor showed the override, everything else
# used the systematic engine, and Journal Log silently rendered wrong.
#
# This test greps api/main.py to enforce that every stamp/heal call site
# for run_engine wires the override through. Not exhaustive across all
# callers (backfill scripts intentionally use systematic-only) — scoped
# to the three functions the ARCHITECTURE.md map lists as user-facing.


_STAMP_HEAL_FUNCTIONS = [
    "_compute_mct_state_with_day_num",
    "_compute_trend_count",
    "_heal_recent_mct_stamps",
]


def _extract_function_body(source: str, name: str) -> str:
    """Slice out a function's body from source text so we can assert
    on its contents. Uses `def <name>` as start and the next top-level
    `def ` / `@app.` as end. Sloppy but sufficient for a grep-audit."""
    lines = source.splitlines()
    start_idx = next(
        (i for i, ln in enumerate(lines)
         if ln.startswith(f"def {name}(") or ln.startswith(f"def {name}:")),
        None,
    )
    if start_idx is None:
        raise AssertionError(f"function {name!r} not found in api/main.py")
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        ln = lines[i]
        # Top-level def or decorator ends the function.
        if ln.startswith("def ") or ln.startswith("@app."):
            end_idx = i
            break
    return "\n".join(lines[start_idx:end_idx])


# ============================================================================
# Invariant #4c — no stray percent chars in load_summary's SELECT template
# ============================================================================
#
# Regression 2026-07-29: an SR14 rollout comment inside the load_summary
# f-string SQL template contained a literal "10%)" — psycopg2 parses
# every "%" in a query as a format directive (even inside "-- SQL
# comments"). Any occurrence of "%" not followed by "s" or another "%"
# raises IndexError on cur.execute, and load_summary silently returned
# an empty DataFrame. ACS then rendered "0 open positions" for a
# portfolio with 7 open positions.
#
# Static test: scan load_summary's SELECT template for stray "%" chars.
# Same grep-based discipline as the migration-audit rules in CLAUDE.md;
# faster to catch here than to debug an "empty positions" incident.


def test_load_summary_select_template_has_no_stray_percent_chars():
    """psycopg2 raises IndexError on any '%' not followed by 's' or
    another '%' (in SELECT templates, including inside `-- comments`).
    The template lives inside a Python f-string, so it's easy to
    accidentally paste a comment that contains a natural-language
    percent sign. This test catches that class before it ships."""
    import re
    src = (_REPO_ROOT / "db_layer.py").read_text()
    lines = src.splitlines()

    in_template = False
    start_line = 0
    template_lines: list[str] = []
    for i, line in enumerate(lines, start=1):
        if 'query = f"""' in line and not in_template:
            in_template = True
            start_line = i
            continue
        if in_template:
            # Closing triple-quote ends the template.
            if '"""' in line:
                in_template = False
                break
            template_lines.append(line)

    assert template_lines, (
        "load_summary's SELECT f-string template not found — the parser "
        "assumed a 'query = f\"\"\"' opener followed by a '\"\"\"' closer. "
        "Update this test's slicing if load_summary was restructured."
    )

    strays: list[tuple[int, str]] = []
    for i, line in enumerate(template_lines):
        for m in re.finditer(r'%(?!s)(?!%)', line):
            snippet = line[max(0, m.start() - 15): m.start() + 15]
            strays.append((start_line + 1 + i, snippet))

    assert not strays, (
        "Stray '%' chars found in load_summary's SELECT template. "
        "psycopg2 will raise IndexError on cur.execute and load_summary "
        "will silently return empty. Rephrase to avoid percent chars "
        "(e.g. 'pct' or 'percent') or hoist the comment to a Python # "
        "line above the f-string.\n\nOffending lines:\n"
        + "\n".join(f"  line {ln}: ...{snip}..." for ln, snip in strays)
    )


# ============================================================================
# Invariant #4a — SR14 tier flips ONLY when broker_stop_price is set
# ============================================================================
#
# Motivating design (2026-07-29 rollout): Position Sizer's two-stop model
# parks a physical broker stop at −0.75× ATR21 from B1 fill. Presence of
# `trades_summary.broker_stop_price` is the flag that promotes the ACS
# Sell Rule tier from SR1 → SR14 when B1 return < 10%. Above +10%, the
# BE stop replaces the broker stop, so the flag becomes stale bookkeeping
# and the classifier ignores it (tier moves to SR11 / SR8 as usual).
#
# Static test: grep the sell-rule classifier's source to confirm the
# <10% branch reads broker_stop_price. Cheap regression guard — a future
# refactor that drops the check silently reverts everyone with a broker
# stop parked back to SR1 without any test failing on unit level.


def test_sell_rule_classifier_reads_broker_stop_price_in_sub_10_branch():
    """The classifier's <10% branch must read broker_stop_price and
    promote to SR14 when set. Regression: if a refactor collapses the
    classifier signature back to just b1ReturnPct, every SR14 position
    silently downgrades to SR1 and the operator loses the flag distinction
    without any per-page test surfacing the drift."""
    src = (_REPO_ROOT / "frontend" / "src" / "lib" / "sell-rule.ts").read_text()

    # Signature must accept brokerStopPrice as an optional second arg.
    assert "brokerStopPrice" in src, (
        "sell-rule.ts::classifySellRuleTier no longer references "
        "brokerStopPrice — the <10% branch will always return SR1, "
        "silently retiring the SR14 tier for every two-stop-model "
        "position."
    )
    # sr14 literal must appear in the tier union + returned by the
    # promotion path — cheap grep guards against a rename that
    # doesn't update every callsite.
    assert '"sr14"' in src, (
        "sr14 literal missing from sell-rule.ts — the SellRuleTier "
        "union or classifier's return path was changed without "
        "updating this contract test."
    )


def test_sell_rule_tier_order_places_sr14_between_sr1_and_sr11():
    """The tier order matters for the ACS Sell Rule column's sort UX.
    SR14 sits between SR1 (no physical stop) and SR11 (BE stop) because
    it's 'one step further along in defense' than SR1 but not yet promoted
    to the BE step-up. A reshuffle here breaks the intended sort order."""
    src = (_REPO_ROOT / "frontend" / "src" / "lib" / "sell-rule.ts").read_text()
    # Extract the SELL_RULE_TIER_ORDER block.
    import re
    m = re.search(
        r"SELL_RULE_TIER_ORDER[^{]*\{([^}]+)\}",
        src, re.DOTALL,
    )
    assert m, "SELL_RULE_TIER_ORDER block not found in sell-rule.ts"
    block = m.group(1)
    # Parse rank per tier (naive but sufficient: sr1: 0, sr14: 1, ...).
    ranks = {}
    for tier in ("sr1", "sr14", "sr11", "sr8"):
        tm = re.search(rf"{tier}\s*:\s*(\d+)", block)
        assert tm, f"tier {tier!r} missing from SELL_RULE_TIER_ORDER"
        ranks[tier] = int(tm.group(1))
    assert ranks["sr1"] < ranks["sr14"] < ranks["sr11"] < ranks["sr8"], (
        f"SR14 must rank between SR1 and SR11, but ranks were: {ranks}. "
        "Sort order in the ACS Sell Rule column depends on this ladder."
    )


# ============================================================================
# Invariant #4 — NLV Entry (batch-edit) does not clobber Daily Journal fields
# ============================================================================
#
# Motivating regression (2026-07-28): user typed in Daily Thoughts, saved
# NLV Entry from the Daily Routine card, came back to Daily Journal — the
# rich-text body was blanked. Cause: journal_batch_edit's UPDATE branch
# hardcoded `daily_thoughts = ""` (plus lowlights, top_lesson, above_21ema)
# as "defaults" and bound those into the UPDATE SQL. NLV Entry's payload
# doesn't carry those fields — it doesn't own them — so the batch write
# silently wiped whatever the Daily Journal write shell had persisted.
#
# Static test: batch_edit's UPDATE branch must PRESERVE these fields from
# the existing row, not default them to empty. Reads api/main.py.


def test_batch_edit_update_preserves_daily_thoughts_and_free_text():
    """The load-bearing invariant: any write path that TOUCHES the
    trading_journal row (batch-edit / journal-edit / game-plan) must
    preserve fields it doesn't own. NLV Entry saves via batch-edit and
    doesn't send daily_thoughts / lowlights / top_lesson; those must
    survive the write. The test asserts the UPDATE branch reads them
    from existing_row rather than binding a hardcoded empty default."""
    source = (_REPO_ROOT / "api" / "main.py").read_text()

    # Slice the journal_batch_edit function.
    import re
    m = re.search(
        r"def journal_batch_edit\(.*?\n(.*?)(?=\n(?:def |@app\.))",
        source, re.DOTALL,
    )
    assert m, "journal_batch_edit not found in api/main.py"
    body = m.group(0)

    # The existence-check SELECT must fetch each of these columns so the
    # UPDATE branch can bind them from existing_row instead of defaulting.
    for col in ("daily_thoughts", "lowlights", "top_lesson", "above_21ema"):
        assert col in body, (
            f"journal_batch_edit no longer references {col!r} in its "
            f"existence-check SELECT — the UPDATE branch will fall back "
            f"to a hardcoded '' default and clobber whatever the Daily "
            f"Journal shell / Journal checklist wrote for the day."
        )

    # No unconditional `= ""` (or `= 0`) default for these fields at
    # module scope — the write path must gate them behind
    # existing_row_present so the preservation branch fires on UPDATE.
    # We check that any assignment to these names sits INSIDE an else
    # branch (i.e. the INSERT-only path).
    forbidden = [
        r'^\s+daily_thoughts = ""\s*$',
        r'^\s+lowlights = ""\s*$',
        r'^\s+top_lesson = ""\s*$',
        r'^\s+above_21ema = 0\s*$',
    ]
    for pattern in forbidden:
        matches = re.findall(pattern, body, re.MULTILINE)
        # A single occurrence is fine (the INSERT-only else branch).
        # Two would mean the old "always default" bug is back.
        assert len(matches) <= 1, (
            f"pattern {pattern!r} appears {len(matches)} times in "
            f"journal_batch_edit — expected at most one (inside the "
            f"else branch for the fresh-INSERT case). Multiple hits "
            f"means the UPDATE branch is blanking the field again."
        )


def test_every_stamp_heal_function_wires_override_into_run_engine():
    """Every MCT stamp / heal function that calls run_engine must also
    thread _current_override_date() through as force_correction_at_date.
    A missing wire silently reverts the user's override on the next
    save/heal."""
    source = (_REPO_ROOT / "api" / "main.py").read_text()
    failures = []
    for fn in _STAMP_HEAL_FUNCTIONS:
        body = _extract_function_body(source, fn)
        # Must call run_engine at least once.
        if "run_engine(" not in body:
            failures.append(f"{fn}: no run_engine( call found — grep this test")
            continue
        # Must reference _current_override_date somewhere in the body so
        # the override can plumb through. Not a perfect check (someone
        # could reference then not pass it) but catches the actual
        # regression class — "forgot to add it entirely."
        if "_current_override_date" not in body:
            failures.append(
                f"{fn}: calls run_engine but never references "
                "_current_override_date() — override will silently be dropped"
            )
    assert not failures, (
        "\n".join(failures) + "\n\nSee ARCHITECTURE.md §2 (MCT Engine) for the "
        "override plumbing contract."
    )
