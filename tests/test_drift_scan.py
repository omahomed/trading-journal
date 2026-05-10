"""Tests for the drift-scan admin endpoint and the underlying check runner.

DESIGN
======
The drift checks are SQL — testing them ideally would mean running them
against a real Postgres instance with synthetic-fixture data. We don't
require DATABASE_URL for the default test run though, so we use a
FakeCursor harness that:
  - Pattern-matches each SQL the runner issues (count vs sample) and
    returns canned rows.
  - Lets a single test specify "for check X, count = N and sample = [...]"
    via a small dict.

This catches the surfaces the user prompt called out:
  - Admin gate (founder-only contract)
  - Param routing (?portfolio= / ?check_id= / ?limit_samples=)
  - Response shape & summary tile arithmetic
  - Statement-timeout on one check doesn't fail the whole scan
  - Per-check synthetic violations surface count + samples correctly

It does NOT verify the SQL strings are themselves correct against the
schema — that's reserved for manual eyeball during code review and (if
desired later) a `pytestmark_db` integration suite.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import jwt
import psycopg2
import pytest
from fastapi.testclient import TestClient


_TEST_SECRET = "test-secret-not-for-prod"
_FOUNDER_ID = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"
_OTHER_USER_ID = "11111111-2222-3333-4444-555555555555"


def _token_for(user_id: str) -> str:
    return jwt.encode({"sub": user_id}, _TEST_SECRET, algorithm="HS256")


def _founder_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_token_for(_FOUNDER_ID)}"}


def _non_founder_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_token_for(_OTHER_USER_ID)}"}


# ---------------------------------------------------------------------------
# FakeCursor harness — matches each SQL the runner issues to a check_id and
# returns the canned (count, samples) the test wants.
# ---------------------------------------------------------------------------


class FakeCursor:
    """Minimal cursor that recognises the runner's two SQL shapes.

    The runner issues, per check:
      1. SELECT 1                                 (txn warm-up)
      2. SET LOCAL statement_timeout = 30000
      3. SELECT COUNT(*) FROM (<check.sql>) AS v
      4. SELECT * FROM (<check.sql>) AS v LIMIT %(limit)s

    We pattern-match on the count vs sample wrapper and the substring of
    the inner SQL to figure out which check is being run. Each test
    primes `state` with check_id -> {"count": int, "samples": list}.
    """

    def __init__(self, state: dict[str, Any]):
        self._state = state
        self._description: list[Any] = []
        self._last_rows: list[Any] = []
        self._cur_check_id: str | None = None
        self._timeout_ids: set[str] = set(state.get("timeout_check_ids", set()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql: str, params: dict | None = None) -> None:
        # Identify which check (if any) this SQL targets by scanning the
        # canned drift_checks for a registered SQL fragment present in
        # the wrapped query.
        from api.drift_checks import DRIFT_CHECKS

        # Paranoia: simulate a statement-timeout for any check whose id
        # is in the timeout set. The runner's except block catches this
        # and turns it into a check-level error — that's what we test.
        for c in DRIFT_CHECKS:
            if c.check_id in self._timeout_ids and c.sql in sql:
                raise psycopg2.errors.QueryCanceled("statement timeout")

        if sql.strip() == "SELECT 1":
            self._last_rows = [(1,)]
            self._description = [_FakeDesc("col0")]
            return
        if sql.strip().startswith("SET LOCAL"):
            self._last_rows = []
            return

        # Otherwise it's the count or sample wrapper — figure out which
        # check based on the inner SQL fragment. The runner wraps the
        # registered SQL verbatim inside `SELECT ... FROM ({sql}) AS v`,
        # so the entire registered SQL appears as a substring of `sql`.
        # Matching the WHOLE registered SQL avoids ambiguity between
        # checks that share the same opening column projection (e.g.
        # rule_mismatch vs buy_notes_mismatch).
        matched = None
        for c in DRIFT_CHECKS:
            if c.sql in sql:
                matched = c.check_id
                break
        self._cur_check_id = matched

        canned = self._state["canned"].get(matched, {"count": 0, "samples": []})

        if "SELECT COUNT(*)" in sql:
            self._last_rows = [(canned["count"],)]
            self._description = [_FakeDesc("count")]
            return

        # Sample query: return the canned sample list, respecting LIMIT.
        limit = (params or {}).get("limit", 10)
        rows = canned["samples"][:limit]
        if rows:
            self._description = [_FakeDesc(k) for k in rows[0].keys()]
            self._last_rows = [tuple(r.values()) for r in rows]
        else:
            # Even when there are no rows, the runner reads cur.description
            # — fall back to a single placeholder column so .description
            # is non-empty.
            self._description = [_FakeDesc("trade_id")]
            self._last_rows = []

    @property
    def description(self):
        return self._description

    def fetchone(self):
        return self._last_rows[0] if self._last_rows else None

    def fetchall(self):
        return list(self._last_rows)

    def close(self):
        pass


class _FakeDesc:
    def __init__(self, name: str):
        self.name = name

    def __getitem__(self, idx):
        # psycopg2's cursor.description is a tuple-like; the runner reads
        # d[0] for the name. Keep both index and attribute access working.
        if idx == 0:
            return self.name
        raise IndexError(idx)


class FakeConnection:
    def __init__(self, state: dict[str, Any]):
        self._state = state

    def cursor(self):
        return FakeCursor(self._state)

    def rollback(self):
        pass

    def commit(self):
        pass

    # context manager protocol — db.get_db_connection() yields via with-stmt.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed(monkeypatch):
    """Yield (state, client). state.canned is a dict[check_id -> {count, samples}].

    Default canned state = empty (every check returns 0 violations).
    Tests mutate state["canned"] before issuing the request to inject
    known drift for specific checks.
    """
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    import api.main as main
    import db_layer

    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)
    monkeypatch.setattr(main, "FOUNDER_USER_ID", _FOUNDER_ID)

    state: dict[str, Any] = {
        "canned": {},  # check_id -> {"count": int, "samples": list[dict]}
        "timeout_check_ids": set(),
        "portfolio_lookup": {  # name -> id; the endpoint does this lookup
            "CanSlim": 1,
            "TQQQ Strategy": 2,
            "457B Plan": 3,
        },
    }

    # Stub db.get_db_connection to yield a FakeConnection. The endpoint
    # uses two connections (one for portfolio lookup, one for the scan).
    # The portfolio lookup hits a SELECT id FROM portfolios — handle that
    # specially.
    class _PortfolioLookupCursor:
        def __init__(self, st):
            self._st = st
            self._row = None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params):
            assert "SELECT id FROM portfolios" in sql
            name = params[0] if isinstance(params, tuple) else params
            self._row = (self._st["portfolio_lookup"].get(name),) if self._st["portfolio_lookup"].get(name) else None

        def fetchone(self):
            return self._row

        def close(self):
            pass

    class _PortfolioLookupConn:
        def __init__(self, st):
            self._st = st
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self): return _PortfolioLookupCursor(self._st)
        def rollback(self): pass
        def commit(self): pass

    # Track which call we're on — first call = portfolio lookup (if filter
    # set), second call = scan. Easier to identify: portfolio lookup
    # cursor's first execute starts with "SELECT id FROM portfolios".
    # Just hand back a connection that handles both shapes.
    class _MultiPurposeCursor:
        def __init__(self, st):
            self._st = st
            self._mode: str | None = None
            self._inner: Any = None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            if self._inner is not None:
                self._inner.__exit__(*a)
            return False

        def execute(self, sql, params=None):
            if self._mode is None:
                if "SELECT id FROM portfolios" in sql:
                    self._mode = "lookup"
                    self._inner = _PortfolioLookupCursor(self._st).__enter__()
                else:
                    self._mode = "drift"
                    self._inner = FakeCursor(self._st).__enter__()
            return self._inner.execute(sql, params)

        @property
        def description(self):
            return getattr(self._inner, "description", [])

        def fetchone(self):
            return self._inner.fetchone()

        def fetchall(self):
            return self._inner.fetchall() if hasattr(self._inner, "fetchall") else []

        def close(self):
            if self._inner is not None and hasattr(self._inner, "close"):
                self._inner.close()

    class _MultiPurposeConn:
        def __init__(self, st):
            self._st = st
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self): return _MultiPurposeCursor(self._st)
        def rollback(self): pass
        def commit(self): pass

    monkeypatch.setattr(db_layer, "get_db_connection",
                        lambda *a, **kw: _MultiPurposeConn(state))
    # The endpoint imports `db` (= db_layer) directly; same target.
    monkeypatch.setattr("api.main.db.get_db_connection",
                        lambda *a, **kw: _MultiPurposeConn(state))

    original_enabled = getattr(main.limiter, "enabled", True)
    if hasattr(main.limiter, "enabled"):
        main.limiter.enabled = False

    client = TestClient(main.app)
    try:
        yield state, client
    finally:
        if hasattr(main.limiter, "enabled"):
            main.limiter.enabled = original_enabled


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_registry_has_eleven_checks():
    """11 checks live in the registry. (Originally 12; closed_with_nonzero_shares
    was dropped after investigation showed trades_summary.shares is dual-semantic
    by design — for CLOSED trades it carries lifetime total_buy_shs, not 0, so
    the check's premise was invalid. See trade_calc.py:177 for the canonical
    write semantics.) Guard against accidental drift in the registry size."""
    from api.drift_checks import DRIFT_CHECKS
    assert len(DRIFT_CHECKS) == 11


def test_registry_check_ids_unique():
    """Duplicate check_ids would silently shadow each other in the
    by-id lookup — assert ids are unique up-front."""
    from api.drift_checks import DRIFT_CHECKS
    ids = [c.check_id for c in DRIFT_CHECKS]
    assert len(ids) == len(set(ids)), f"Duplicate check_ids: {ids}"


def test_registry_severities_are_valid():
    """Severity must be 'warning' or 'error' — frontend has hard-coded
    color logic on these two values."""
    from api.drift_checks import DRIFT_CHECKS
    for c in DRIFT_CHECKS:
        assert c.severity in ("warning", "error"), c


def test_registry_sql_uses_portfolio_filter_param():
    """Every check must reference %(portfolio_id)s so the endpoint's
    portfolio filter actually scopes the query. A check that forgets
    this would silently scan all portfolios on a portfolio-scoped run."""
    from api.drift_checks import DRIFT_CHECKS
    for c in DRIFT_CHECKS:
        assert "%(portfolio_id)s" in c.sql, (
            f"{c.check_id} missing %(portfolio_id)s — would not honor "
            f"the ?portfolio= filter."
        )


def test_registry_trx_id_joins_exclude_empty_trades():
    """Checks #8/#9/#12 must exclude trades with any empty/NULL trx_id
    from their evaluation. Otherwise the LIFO join produces false
    positives — see check #10 (lot_closures_empty_trx_id) for the
    underlying data shape. Empty-trx trades get fixed via recompute
    LIFO; once fixed, they rejoin the eligible set naturally."""
    from api.drift_checks import DRIFT_CHECKS_BY_ID
    for cid in ("open_summary_no_open_buys",
                "closed_summary_with_open_buys",
                "summary_shares_vs_open_buy_remaining"):
        sql = DRIFT_CHECKS_BY_ID[cid].sql
        # Each check must exclude empty-trx-id trades via NOT EXISTS
        assert "NOT EXISTS" in sql, f"{cid} missing trx-id exclusion"
        assert "trx_id IS NULL OR" in sql or "trx_id = ''" in sql, \
            f"{cid} missing empty-string trx-id check"


def test_registry_summary_shares_check_is_open_only():
    """Regression guard: summary_shares_vs_open_buy_remaining (#12) must
    filter to status='OPEN'. trades_summary.shares is dual-semantic — for
    CLOSED trades it carries lifetime total_buy_shs by design (see
    trade_calc.py:177 + api/main.py:3142), so comparing it against
    SUM(open_remaining)=0 over-flags every closed campaign that ever
    bought shares (478-of-480 false positives on first prod scan).
    Without this filter, the check is invalid; with it, the check evaluates
    only the population where the comparison is meaningful."""
    from api.drift_checks import DRIFT_CHECKS_BY_ID
    sql = DRIFT_CHECKS_BY_ID["summary_shares_vs_open_buy_remaining"].sql
    assert "s.status = 'OPEN'" in sql, (
        "summary_shares_vs_open_buy_remaining missing status='OPEN' "
        "filter — would re-introduce the dual-semantic false positive."
    )


def test_registry_no_closed_with_nonzero_shares_check():
    """Regression guard: closed_with_nonzero_shares was dropped from the
    registry. It compared status='CLOSED' AND shares>0 as drift, but
    trades_summary.shares carries lifetime total_buy_shs on closed
    trades by design — every closed campaign that ever bought shares
    would (correctly) trip the check. Permanently invalid; not just
    'temporarily disabled'. If anyone re-adds it, this test fires."""
    from api.drift_checks import DRIFT_CHECKS_BY_ID
    assert "closed_with_nonzero_shares" not in DRIFT_CHECKS_BY_ID, (
        "closed_with_nonzero_shares re-added to registry — its premise is "
        "invalid given the dual-semantic of trades_summary.shares. See "
        "trade_calc.py:177 and the rationale in this commit's message."
    )


def test_registry_filters_soft_deletes():
    """Every check that references a soft-delete-bearing table must
    include the corresponding `deleted_at IS NULL` filter. Regression
    guard for the systematic gap caught on 5/10/26 (DOCN orphan
    summary, NBIS 40-share orphan BUY, FPS 175-share orphan BUYs
    produced 3 false positives in #12 because the LIFO joins didn't
    filter d.deleted_at).

    Approach: count references to each soft-delete table in the SQL
    (after exemptions), and require at least that many `deleted_at
    IS NULL` filter tokens. Each table reference must be paired with
    a filter — over-filtering is harmless. This catches a check that
    references the table but skips the filter, without forcing a
    full SQL parser into the test harness.

    Soft-delete-bearing tables (per migrations/006_soft_deletes.sql):
    trades_summary, trades_details, trading_journal. lot_closures
    and portfolios are NOT in this list — they have no deleted_at
    column."""
    from api.drift_checks import DRIFT_CHECKS

    SOFT_DELETE_TABLES = ("trades_summary", "trades_details", "trading_journal")

    EXEMPTIONS = {
        # #10 LEFT JOINs trades_summary purely for ticker resolution on
        # orphan closures. We WANT orphan closures to surface even when
        # their parent summary is soft-deleted — the soft-deleted parent
        # IS the data integrity signal worth investigating. Audit Part B
        # special-case decision.
        ("lot_closures_empty_trx_id", "trades_summary"),
    }

    for c in DRIFT_CHECKS:
        sql_lower = c.sql.lower()
        expected = 0
        referenced_tables = []
        for table in SOFT_DELETE_TABLES:
            if (c.check_id, table) in EXEMPTIONS:
                continue
            count = sql_lower.count(table)
            if count > 0:
                referenced_tables.append((table, count))
            expected += count
        actual = sql_lower.count("deleted_at is null")
        assert actual >= expected, (
            f"{c.check_id}: references soft-delete tables "
            f"{referenced_tables} ({expected} total references after "
            f"exemptions) but only has {actual} `deleted_at IS NULL` "
            f"filter(s). At least one reference is unfiltered — "
            f"soft-deleted rows would be included in the check's scope."
        )


# ---------------------------------------------------------------------------
# Admin gate
# ---------------------------------------------------------------------------


def test_drift_scan_rejects_non_founder(stubbed):
    state, client = stubbed
    r = client.get("/api/admin/drift-scan", headers=_non_founder_headers())
    assert r.status_code == 200
    assert r.json() == {"error": "forbidden_not_admin"}


def test_drift_scan_rejects_missing_token(stubbed):
    """No bearer at all — middleware rejects with 401 before the handler runs."""
    _, client = stubbed
    r = client.get("/api/admin/drift-scan")
    assert r.status_code == 401


def test_drift_scan_accepts_founder(stubbed):
    """Founder gets the response shape (clean state)."""
    _, client = stubbed
    r = client.get("/api/admin/drift-scan", headers=_founder_headers())
    assert r.status_code == 200
    body = r.json()
    assert "checks" in body
    assert "summary" in body
    assert body["summary"]["total_checks"] == 11


# ---------------------------------------------------------------------------
# Clean state
# ---------------------------------------------------------------------------


def test_drift_scan_clean_state_passes_all(stubbed):
    """Empty fixture → all 11 checks report 0 violations and summary
    counts add up correctly."""
    _, client = stubbed
    r = client.get("/api/admin/drift-scan", headers=_founder_headers())
    body = r.json()
    assert body["summary"] == {
        "total_checks": 11,
        "passed":   11,
        "warnings": 0,
        "errors":   0,
    }
    for check in body["checks"]:
        assert check["violation_count"] == 0
        assert check["samples"] == []
        assert check["error"] is None


# ---------------------------------------------------------------------------
# Per-check synthetic violations
# ---------------------------------------------------------------------------


def _seed(state, check_id: str, count: int, samples: list[dict]) -> None:
    """Test helper — seed the FakeCursor's canned rows for one check."""
    state["canned"][check_id] = {"count": count, "samples": samples}


def test_check_summary_detail_rule_mismatch(stubbed):
    state, client = stubbed
    _seed(state, "summary_detail_rule_mismatch", 1, [{
        "trade_id": "202604-001", "ticker": "NVDA", "portfolio": "CanSlim",
        "summary_rule": "br3.1", "detail_rule": "br3.2",
    }])
    r = client.get(
        "/api/admin/drift-scan?check_id=summary_detail_rule_mismatch",
        headers=_founder_headers(),
    )
    body = r.json()
    assert len(body["checks"]) == 1
    assert body["checks"][0]["violation_count"] == 1
    assert body["checks"][0]["samples"][0]["trade_id"] == "202604-001"
    assert body["summary"]["warnings"] == 1
    assert body["summary"]["passed"] == 0


def test_check_summary_detail_buy_notes_mismatch(stubbed):
    state, client = stubbed
    _seed(state, "summary_detail_buy_notes_mismatch", 2, [
        {"trade_id": "202604-001", "ticker": "AVGO", "portfolio": "CanSlim",
         "summary_buy_notes": "old", "detail_buy_notes": "new"},
        {"trade_id": "202604-002", "ticker": "META", "portfolio": "CanSlim",
         "summary_buy_notes": "x", "detail_buy_notes": "y"},
    ])
    r = client.get(
        "/api/admin/drift-scan?check_id=summary_detail_buy_notes_mismatch",
        headers=_founder_headers(),
    )
    body = r.json()
    assert body["checks"][0]["violation_count"] == 2
    assert len(body["checks"][0]["samples"]) == 2


def test_check_risk_budget_null_or_zero_post_021_is_error_severity(stubbed):
    """Tripwire — declared severity is 'error', should bucket into
    summary.errors when violated."""
    state, client = stubbed
    _seed(state, "risk_budget_null_or_zero_post_021", 1, [{
        "trade_id": "202602-005", "ticker": "TSLA", "portfolio": "CanSlim",
        "open_date": "2026-02-15T00:00:00", "risk_budget": 0,
    }])
    r = client.get(
        "/api/admin/drift-scan?check_id=risk_budget_null_or_zero_post_021",
        headers=_founder_headers(),
    )
    body = r.json()
    assert body["checks"][0]["severity"] == "error"
    assert body["summary"]["errors"] == 1


def test_check_string_nan_in_prose_tripwire(stubbed):
    """Migration 022 should prevent any new sentinels — tripwire 'error'."""
    state, client = stubbed
    _seed(state, "string_nan_in_prose", 1, [{
        "trade_id": "202604-001", "ticker": "NVDA", "portfolio": "CanSlim",
        "column_name": "buy_notes", "bad_value": "nan",
    }])
    r = client.get(
        "/api/admin/drift-scan?check_id=string_nan_in_prose",
        headers=_founder_headers(),
    )
    body = r.json()
    assert body["checks"][0]["severity"] == "error"
    assert body["summary"]["errors"] == 1


def test_check_open_with_closed_date_tripwire(stubbed):
    state, client = stubbed
    _seed(state, "open_with_closed_date", 1, [{
        "trade_id": "202604-001", "ticker": "NVDA", "portfolio": "CanSlim",
        "status": "OPEN", "closed_date": "2026-04-15T00:00:00",
    }])
    r = client.get(
        "/api/admin/drift-scan?check_id=open_with_closed_date",
        headers=_founder_headers(),
    )
    body = r.json()
    assert body["checks"][0]["severity"] == "error"


def test_check_invalid_journal_source_values(stubbed):
    state, client = stubbed
    _seed(state, "invalid_journal_source_values", 1, [{
        "portfolio": "CanSlim", "day": "2026-04-01",
        "nlv_source": "bogus", "holdings_source": "manual",
    }])
    r = client.get(
        "/api/admin/drift-scan?check_id=invalid_journal_source_values",
        headers=_founder_headers(),
    )
    body = r.json()
    assert body["checks"][0]["violation_count"] == 1
    assert body["checks"][0]["samples"][0]["nlv_source"] == "bogus"


def test_check_open_summary_no_open_buys(stubbed):
    state, client = stubbed
    _seed(state, "open_summary_no_open_buys", 1, [{
        "trade_id": "202602-009", "ticker": "AMD", "portfolio": "CanSlim",
        "open_remaining": 0,
    }])
    r = client.get(
        "/api/admin/drift-scan?check_id=open_summary_no_open_buys",
        headers=_founder_headers(),
    )
    body = r.json()
    assert body["checks"][0]["severity"] == "error"


def test_check_closed_summary_with_open_buys(stubbed):
    state, client = stubbed
    _seed(state, "closed_summary_with_open_buys", 1, [{
        "trade_id": "202601-005", "ticker": "ARM", "portfolio": "CanSlim",
        "open_remaining": 50,
    }])
    r = client.get(
        "/api/admin/drift-scan?check_id=closed_summary_with_open_buys",
        headers=_founder_headers(),
    )
    body = r.json()
    assert body["checks"][0]["violation_count"] == 1
    assert body["checks"][0]["samples"][0]["open_remaining"] == 50


def test_check_lot_closures_empty_trx_id(stubbed):
    state, client = stubbed
    _seed(state, "lot_closures_empty_trx_id", 3, [{
        "trade_id": "202604-001", "ticker": "NVDA", "portfolio": "CanSlim",
        "sell_trx_id": "S1", "buy_trx_id": "", "closure_id": 42,
    }])
    r = client.get(
        "/api/admin/drift-scan?check_id=lot_closures_empty_trx_id",
        headers=_founder_headers(),
    )
    body = r.json()
    assert body["checks"][0]["violation_count"] == 3
    assert body["checks"][0]["samples"][0]["buy_trx_id"] == ""


def test_check_summary_realized_pl_vs_lot_closures_sum(stubbed):
    state, client = stubbed
    _seed(state, "summary_realized_pl_vs_lot_closures_sum", 1, [{
        "trade_id": "202602-003", "ticker": "GOOG", "portfolio": "CanSlim",
        "summary_realized_pl": 1234.56,
        "lot_closures_sum": 1230.00,
        "diff": 4.56,
    }])
    r = client.get(
        "/api/admin/drift-scan?check_id=summary_realized_pl_vs_lot_closures_sum",
        headers=_founder_headers(),
    )
    body = r.json()
    assert body["checks"][0]["severity"] == "warning"
    assert body["checks"][0]["samples"][0]["diff"] == 4.56


def test_check_summary_shares_vs_open_buy_remaining(stubbed):
    """OPEN trade where summary.shares (200) doesn't match the per-buy
    LIFO remainder (100) — the actionable drift this check catches.
    Sample explicitly carries status='OPEN' to document the scope:
    only OPEN trades are eligible after the dual-semantic correction."""
    state, client = stubbed
    _seed(state, "summary_shares_vs_open_buy_remaining", 1, [{
        "trade_id": "202604-001", "ticker": "NVDA", "portfolio": "CanSlim",
        "status": "OPEN",
        "summary_shares": 200, "detail_remaining": 100, "diff": 100,
    }])
    r = client.get(
        "/api/admin/drift-scan?check_id=summary_shares_vs_open_buy_remaining",
        headers=_founder_headers(),
    )
    body = r.json()
    assert body["checks"][0]["severity"] == "error"
    assert body["checks"][0]["samples"][0]["diff"] == 100
    assert body["checks"][0]["samples"][0]["status"] == "OPEN"


# ---------------------------------------------------------------------------
# Param routing
# ---------------------------------------------------------------------------


def test_check_id_filter_runs_only_that_check(stubbed):
    """?check_id=X → response contains exactly that one check."""
    _, client = stubbed
    r = client.get(
        "/api/admin/drift-scan?check_id=open_with_closed_date",
        headers=_founder_headers(),
    )
    body = r.json()
    assert len(body["checks"]) == 1
    assert body["checks"][0]["check_id"] == "open_with_closed_date"
    assert body["check_filter"] == "open_with_closed_date"


def test_unknown_check_id_returns_400(stubbed):
    _, client = stubbed
    r = client.get(
        "/api/admin/drift-scan?check_id=does_not_exist",
        headers=_founder_headers(),
    )
    assert r.status_code == 400
    assert "Unknown check_id" in r.json()["detail"]


def test_unknown_portfolio_returns_400(stubbed):
    _, client = stubbed
    r = client.get(
        "/api/admin/drift-scan?portfolio=NotARealPortfolio",
        headers=_founder_headers(),
    )
    assert r.status_code == 400
    assert "Unknown portfolio" in r.json()["detail"]


def test_portfolio_filter_echoes_in_response(stubbed):
    _, client = stubbed
    r = client.get(
        "/api/admin/drift-scan?portfolio=CanSlim",
        headers=_founder_headers(),
    )
    body = r.json()
    assert body["portfolio_filter"] == "CanSlim"


def test_limit_samples_ceiling_enforced_by_query_validator(stubbed):
    """FastAPI's Query(le=50) rejects 51+ at the validator layer with a 422."""
    _, client = stubbed
    r = client.get(
        "/api/admin/drift-scan?limit_samples=999",
        headers=_founder_headers(),
    )
    assert r.status_code == 422


def test_limit_samples_floor_enforced_by_query_validator(stubbed):
    """ge=1 rejects 0 with 422."""
    _, client = stubbed
    r = client.get(
        "/api/admin/drift-scan?limit_samples=0",
        headers=_founder_headers(),
    )
    assert r.status_code == 422


def test_limit_samples_default_echoed_in_response(stubbed):
    _, client = stubbed
    r = client.get("/api/admin/drift-scan", headers=_founder_headers())
    body = r.json()
    assert body["sample_limit"] == 10  # default


# ---------------------------------------------------------------------------
# Statement timeout — one check failing must not abort the whole scan
# ---------------------------------------------------------------------------


def test_statement_timeout_one_check_does_not_fail_scan(stubbed):
    """Simulate a statement timeout on one check; assert (a) the other
    10 still ran and reported, (b) the timed-out check is bucketed as
    'error' regardless of its declared severity, and (c) the response
    surfaces the underlying error message in the check entry."""
    state, client = stubbed
    state["timeout_check_ids"] = {"summary_realized_pl_vs_lot_closures_sum"}

    r = client.get("/api/admin/drift-scan", headers=_founder_headers())
    body = r.json()

    # 11 entries total — none missing.
    assert len(body["checks"]) == 11

    # Find the timed-out check.
    failed = next(
        c for c in body["checks"]
        if c["check_id"] == "summary_realized_pl_vs_lot_closures_sum"
    )
    assert failed["error"] is not None
    assert "timeout" in failed["error"].lower() or "canceled" in failed["error"].lower()
    # Even though declared 'warning', a check that errored is bucketed as 'error'.
    assert failed["severity"] == "error"
    assert "Check failed to run" in failed["remediation"]

    # Other 10 still passed (no errors injected for them).
    other_passed = sum(
        1 for c in body["checks"]
        if c["check_id"] != "summary_realized_pl_vs_lot_closures_sum"
        and c["violation_count"] == 0
        and c["error"] is None
    )
    assert other_passed == 10


# ---------------------------------------------------------------------------
# Summary arithmetic
# ---------------------------------------------------------------------------


def test_summary_buckets_warnings_and_errors_separately(stubbed):
    """Mix one warning-class violation, one error-class violation, and
    one timeout. Assert the summary tile counts each bucket correctly."""
    state, client = stubbed
    # Warning-class: rule mismatch
    _seed(state, "summary_detail_rule_mismatch", 1, [{
        "trade_id": "202604-001", "ticker": "NVDA", "portfolio": "CanSlim",
        "summary_rule": "a", "detail_rule": "b",
    }])
    # Error-class: lot_closures with empty trx_id
    _seed(state, "lot_closures_empty_trx_id", 1, [{
        "trade_id": "202601-003", "ticker": "AAPL", "portfolio": "CanSlim",
        "sell_trx_id": "S1", "buy_trx_id": "", "closure_id": 99,
    }])
    # Force a timeout on one more check → counted as 'error'.
    state["timeout_check_ids"] = {"open_summary_no_open_buys"}

    r = client.get("/api/admin/drift-scan", headers=_founder_headers())
    body = r.json()

    s = body["summary"]
    assert s["total_checks"] == 11
    assert s["warnings"] == 1
    assert s["errors"] == 2  # lot_closures_empty_trx_id + open_summary_no_open_buys (timeout)
    assert s["passed"] == 8
    assert s["warnings"] + s["errors"] + s["passed"] == s["total_checks"]
