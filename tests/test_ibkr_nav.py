"""Tests for ibkr_flex.fetch_nav_for_date + /api/ibkr/nav-for-date endpoint.

The puller does HTTP, so all network is stubbed via monkeypatching
requests.get. The endpoint returns 200-OK-with-success-flag on every path
(per spec: the frontend must always be able to read body.success and
body.error / body.message rather than juggling HTTP error codes), and the
tests assert that contract for both happy and failure paths.
"""
from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest

import ibkr_flex
from ibkr_flex import FlexQueryError, fetch_nav_for_date


# ---------------------------------------------------------------------------
# HTTP fixtures — IBKR's two-step Send/Get protocol simulated as canned XML
# ---------------------------------------------------------------------------


def _send_success_xml(ref: str = "REF123") -> str:
    return (
        "<?xml version='1.0'?>"
        "<FlexStatementResponse>"
        "<Status>Success</Status>"
        f"<ReferenceCode>{ref}</ReferenceCode>"
        "<Url>https://example/Get</Url>"
        "</FlexStatementResponse>"
    )


def _send_error_xml(code: str, msg: str) -> str:
    return (
        "<?xml version='1.0'?>"
        "<FlexStatementResponse>"
        "<Status>Fail</Status>"
        f"<ErrorCode>{code}</ErrorCode>"
        f"<ErrorMessage>{msg}</ErrorMessage>"
        "</FlexStatementResponse>"
    )


def _nav_report_xml(rows: list[dict]) -> str:
    """Build a Flex Query NAV report with one or more EquitySummary rows."""
    body = "".join(
        "<EquitySummaryByReportDateInBase "
        + " ".join(f'{k}="{v}"' for k, v in r.items())
        + " />"
        for r in rows
    )
    return (
        "<?xml version='1.0'?>"
        "<FlexQueryResponse>"
        "<FlexStatements count='1'>"
        "<FlexStatement accountId='U1234567' fromDate='20260427' toDate='20260427'>"
        f"<EquitySummaryInBase>{body}</EquitySummaryInBase>"
        "</FlexStatement>"
        "</FlexStatements>"
        "</FlexQueryResponse>"
    )


def _make_resp(text: str, status_code: int = 200):
    """Build a fake `requests.Response` shape that satisfies the ibkr_flex
    code path: .text, .status_code, .raise_for_status()."""
    r = MagicMock()
    r.text = text
    r.status_code = status_code
    r.raise_for_status = MagicMock()
    return r


def _stub_requests(monkeypatch, *, send_xml: str, get_xml: str):
    """Wire requests.get to return send_xml on the SendRequest URL and
    get_xml on the GetStatement URL. Sequence-independent."""
    def fake_get(url, params=None, timeout=None):
        if "SendRequest" in url:
            return _make_resp(send_xml)
        if "GetStatement" in url:
            return _make_resp(get_xml)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(ibkr_flex.requests, "get", fake_get)


# ---------------------------------------------------------------------------
# fetch_nav_for_date — happy path
# ---------------------------------------------------------------------------


def test_fetch_nav_for_date_success(monkeypatch):
    """Match the requested date, parse total/cash/stock from attributes."""
    _stub_requests(
        monkeypatch,
        send_xml=_send_success_xml(),
        get_xml=_nav_report_xml([{
            "reportDate": "20260427",
            "total": "487264.50",
            "cash": "-431003.85",
            "stock": "918268.35",
            "currency": "USD",
            "accountId": "U1234567",
        }]),
    )

    result = fetch_nav_for_date(
        query_id="QID", target_date=date(2026, 4, 27), token="TOK",
    )

    assert result["nav"] == 487264.50
    assert result["cash_balance"] == -431003.85
    assert result["position_value"] == 918268.35
    assert result["currency"] == "USD"
    assert result["account"] == "U1234567"
    assert result["date"] == "2026-04-27"


def test_fetch_nav_accepts_iso_date_string(monkeypatch):
    """target_date='YYYY-MM-DD' is the form the HTTP endpoint passes through."""
    _stub_requests(
        monkeypatch,
        send_xml=_send_success_xml(),
        get_xml=_nav_report_xml([{
            "reportDate": "20260427",
            "total": "100000",
            "cash": "20000",
            "stock": "80000",
        }]),
    )

    result = fetch_nav_for_date(query_id="QID", target_date="2026-04-27", token="TOK")
    assert result["nav"] == 100000.0
    assert result["date"] == "2026-04-27"


def test_fetch_nav_picks_correct_row_when_report_has_multiple_dates(monkeypatch):
    """A multi-day report must return the row matching the requested date."""
    _stub_requests(
        monkeypatch,
        send_xml=_send_success_xml(),
        get_xml=_nav_report_xml([
            {"reportDate": "20260425", "total": "480000", "cash": "-100", "stock": "480100"},
            {"reportDate": "20260426", "total": "485000", "cash": "-200", "stock": "485200"},
            {"reportDate": "20260427", "total": "487264.50", "cash": "-431003.85", "stock": "918268.35"},
        ]),
    )

    result = fetch_nav_for_date(query_id="QID", target_date="2026-04-27", token="TOK")
    assert result["nav"] == 487264.50


def test_fetch_nav_falls_back_to_single_rollup_row_by_toDate(monkeypatch):
    """If a Flex Query is configured to emit one rollup row, accept it when
    the parent toDate matches the request — covers the spec's 'Last Business
    Day' period mode that doesn't tag reportDate on the row itself."""
    _stub_requests(
        monkeypatch,
        send_xml=_send_success_xml(),
        get_xml=_nav_report_xml([{
            "toDate": "20260427",
            "total": "487264.50",
            "cash": "-431003.85",
            "stock": "918268.35",
        }]),
    )

    result = fetch_nav_for_date(query_id="QID", target_date="2026-04-27", token="TOK")
    assert result["nav"] == 487264.50


# ---------------------------------------------------------------------------
# fetch_nav_for_date — failure paths
# ---------------------------------------------------------------------------


def test_fetch_nav_no_data_for_date(monkeypatch):
    """Report parsed but no row matches the requested date → no_data_for_date."""
    _stub_requests(
        monkeypatch,
        send_xml=_send_success_xml(),
        get_xml=_nav_report_xml([
            {"reportDate": "20260425", "total": "480000", "cash": "0", "stock": "480000"},
        ]),
    )

    with pytest.raises(FlexQueryError) as exc:
        fetch_nav_for_date(query_id="QID", target_date="2026-04-27", token="TOK")

    assert exc.value.code == "no_data_for_date"
    # Available dates surfaced for debuggability
    assert "20260425" in str(exc.value)


def test_fetch_nav_auth_failure(monkeypatch):
    """1003 = invalid token. Maps to ibkr_auth_failed (not generic)."""
    _stub_requests(
        monkeypatch,
        send_xml=_send_error_xml("1003", "Invalid token"),
        get_xml="<unused/>",
    )

    with pytest.raises(FlexQueryError) as exc:
        fetch_nav_for_date(query_id="QID", target_date="2026-04-27", token="TOK")

    assert exc.value.code == "ibkr_auth_failed"
    assert "Invalid token" in str(exc.value)


def test_fetch_nav_missing_credentials(monkeypatch):
    """No env vars + no explicit args → ibkr_not_configured before any HTTP."""
    monkeypatch.delenv("IBKR_FLEX_TOKEN", raising=False)
    monkeypatch.delenv("IBKR_NAV_FLEX_QUERY_ID", raising=False)

    with pytest.raises(FlexQueryError) as exc:
        fetch_nav_for_date()

    assert exc.value.code == "ibkr_not_configured"


def test_fetch_nav_parse_error_when_report_has_no_equity_section(monkeypatch):
    """If the Flex Query is misconfigured (missing NAV section) the report
    parses but has no EquitySummaryByReportDateInBase rows → parse_error,
    not no_data_for_date. The user needs to fix the query, not retry."""
    _stub_requests(
        monkeypatch,
        send_xml=_send_success_xml(),
        get_xml=(
            "<?xml version='1.0'?>"
            "<FlexQueryResponse><FlexStatements count='1'>"
            "<FlexStatement accountId='U1234567' fromDate='20260427' toDate='20260427' />"
            "</FlexStatements></FlexQueryResponse>"
        ),
    )

    with pytest.raises(FlexQueryError) as exc:
        fetch_nav_for_date(query_id="QID", target_date="2026-04-27", token="TOK")

    assert exc.value.code == "parse_error"
    assert "Net Asset Value" in str(exc.value)


def test_fetch_nav_network_timeout(monkeypatch):
    """requests.Timeout → network_timeout (distinguished from auth/data errors)."""
    import requests as _requests

    def boom(*a, **kw):
        raise _requests.Timeout("timed out after 30s")

    monkeypatch.setattr(ibkr_flex.requests, "get", boom)

    with pytest.raises(FlexQueryError) as exc:
        fetch_nav_for_date(query_id="QID", target_date="2026-04-27", token="TOK")

    assert exc.value.code == "network_timeout"


# ---------------------------------------------------------------------------
# Endpoint shape — always 200 OK with success flag
# ---------------------------------------------------------------------------


_TEST_SECRET = "test-secret-key-for-pytest-only-not-prod"
_TEST_USER_ID = "00000000-0000-4000-8000-000000000000"


def _make_auth_headers() -> dict:
    """Generate a valid bearer header for the test JWT secret."""
    import jwt
    token = jwt.encode({"sub": _TEST_USER_ID}, _TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient bound to the live `app` from api.main.

    Configures AUTH_SECRET + valid JWT so the middleware lets requests through,
    and disables rate limiting so high-volume test loops don't 429. Cleans
    IBKR env vars per-test to avoid cross-test bleed.
    """
    monkeypatch.delenv("IBKR_FLEX_TOKEN", raising=False)
    monkeypatch.delenv("IBKR_NAV_FLEX_QUERY_ID", raising=False)
    monkeypatch.setenv("AUTH_SECRET", _TEST_SECRET)

    from fastapi.testclient import TestClient
    import api.main as main

    # Patch the module-level AUTH_SECRET that the middleware closed over at
    # import time — env var alone isn't enough.
    monkeypatch.setattr(main, "AUTH_SECRET", _TEST_SECRET)

    # Disable rate limiting for the duration of the test
    original_enabled = getattr(main.limiter, "enabled", True)
    if hasattr(main.limiter, "enabled"):
        main.limiter.enabled = False

    c = TestClient(main.app, headers=_make_auth_headers())
    try:
        yield c
    finally:
        if hasattr(main.limiter, "enabled"):
            main.limiter.enabled = original_enabled


def test_endpoint_returns_success_shape(client, monkeypatch):
    """Happy path: success=true with nav, cash_balance, position_value,
    report_date, source. The frontend reads exactly these field names."""
    fake_result = {
        "date": "2026-04-27",
        "nav": 487264.50,
        "cash_balance": -431003.85,
        "position_value": 918268.35,
        "currency": "USD",
        "account": "U1234567",
    }
    monkeypatch.setattr(ibkr_flex, "fetch_nav_for_date",
                        lambda **kw: fake_result)

    r = client.get("/api/ibkr/nav-for-date?date=2026-04-27")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["nav"] == 487264.50
    assert body["cash_balance"] == -431003.85
    assert body["position_value"] == 918268.35
    assert body["report_date"] == "2026-04-27"
    assert body["source"] == "ibkr_flex_query"


def test_endpoint_returns_error_shape_on_no_data(client, monkeypatch):
    """no_data_for_date is the most likely failure (user opens Daily Routine
    before IBKR has finalised the day). Endpoint must keep 200 OK and surface
    {success:false, error:'no_data_for_date', message:...}."""
    def raise_no_data(**kw):
        raise FlexQueryError("no_data_for_date", "No NAV data for 2026-04-27")
    monkeypatch.setattr(ibkr_flex, "fetch_nav_for_date", raise_no_data)

    r = client.get("/api/ibkr/nav-for-date?date=2026-04-27")
    assert r.status_code == 200, "must always return 200 so frontend can render banner"
    body = r.json()
    assert body["success"] is False
    assert body["error"] == "no_data_for_date"
    assert "No NAV data" in body["message"]


def test_endpoint_returns_error_shape_on_auth_failure(client, monkeypatch):
    """Auth failures still return 200 OK with the error code/message."""
    def raise_auth(**kw):
        raise FlexQueryError("ibkr_auth_failed", "IBKR rejected the request (1003): Invalid token")
    monkeypatch.setattr(ibkr_flex, "fetch_nav_for_date", raise_auth)

    r = client.get("/api/ibkr/nav-for-date?date=2026-04-27")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error"] == "ibkr_auth_failed"


def test_endpoint_returns_error_shape_on_unknown_exception(client, monkeypatch):
    """A non-FlexQueryError exception (e.g. an unexpected import error) still
    yields 200 OK with error='unknown_error', so the frontend never crashes."""
    def boom(**kw):
        raise RuntimeError("totally unexpected")
    monkeypatch.setattr(ibkr_flex, "fetch_nav_for_date", boom)

    r = client.get("/api/ibkr/nav-for-date?date=2026-04-27")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error"] == "unknown_error"
    assert "totally unexpected" in body["message"]


def test_endpoint_no_date_param_uses_default(client, monkeypatch):
    """Omitting ?date= must call the puller with target_date=None (which uses
    _last_completed_trading_day). Verifies the wiring rather than the date
    logic itself (that's covered by the puller's own tests)."""
    captured: dict[str, Any] = {}

    def capture(**kw):
        captured.update(kw)
        return {
            "date": "2026-04-25", "nav": 100.0, "cash_balance": 0.0,
            "position_value": 100.0, "currency": "USD", "account": "U1",
        }
    monkeypatch.setattr(ibkr_flex, "fetch_nav_for_date", capture)

    r = client.get("/api/ibkr/nav-for-date")
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert captured.get("target_date") is None


# ---------------------------------------------------------------------------
# Diagnostic helpers — inspect_nav_xml + redact_nav_xml
# ---------------------------------------------------------------------------


def _root(xml_str: str):
    """Parse an XML string and return the root element."""
    import xml.etree.ElementTree as _ET
    return _ET.fromstring(xml_str)


def test_inspect_nav_xml_reports_attrs_and_dates():
    """The whole point of the debug endpoint: surface every attr key on the
    NAV rows + every date-shaped value, so we can spot a key mismatch
    (e.g. parser keys on `reportDate` but report uses `toDate`)."""
    xml = _nav_report_xml([
        {"toDate": "20260424", "fromDate": "20260424", "total": "100",
         "cash": "20", "stock": "80", "currency": "USD"},
    ])
    info = ibkr_flex.inspect_nav_xml(_root(xml))

    assert info["row_count"] == 1
    # Every attribute key should be reported, sorted
    assert "toDate" in info["all_attrs_on_nav_row"]
    assert "fromDate" in info["all_attrs_on_nav_row"]
    assert "total" in info["all_attrs_on_nav_row"]
    # Date-shaped values surface as key=value pairs — this is what answers
    # "what date attribute is the report actually using?"
    assert "toDate=20260424" in info["all_date_attrs_found"]
    assert "fromDate=20260424" in info["all_date_attrs_found"]
    # Caller can compare what the parser looks for against what's available
    assert info["parser_searches_for_attr"] == "reportDate"


def test_inspect_nav_xml_handles_zero_rows_with_candidate_tags():
    """When the NAV section is missing entirely, surface any tag that *might*
    be the renamed equivalent (Equity*, NAV*, NetAsset*) so debugging can
    follow the trail without a second round trip."""
    xml = (
        "<?xml version='1.0'?>"
        "<FlexQueryResponse><FlexStatements count='1'>"
        "<FlexStatement accountId='U1' fromDate='20260427' toDate='20260427'>"
        "<EquitySummaryByReportDate reportDate='20260427' total='100'/>"
        "</FlexStatement></FlexStatements></FlexQueryResponse>"
    )
    info = ibkr_flex.inspect_nav_xml(_root(xml))

    assert info["row_count"] == 0
    assert "EquitySummaryByReportDate" in info["candidate_nav_tags_when_zero_rows"]


def test_inspect_nav_xml_dedupes_date_pairs_across_rows():
    """A multi-row report shouldn't list the same key=value pair twice."""
    xml = _nav_report_xml([
        {"reportDate": "20260424", "total": "100", "cash": "0", "stock": "100"},
        {"reportDate": "20260425", "total": "101", "cash": "0", "stock": "101"},
        {"reportDate": "20260424", "total": "100", "cash": "0", "stock": "100"},  # dup
    ])
    info = ibkr_flex.inspect_nav_xml(_root(xml))

    # Two unique dates despite three rows
    date_pairs = [p for p in info["all_date_attrs_found"] if p.startswith("reportDate=")]
    assert sorted(date_pairs) == ["reportDate=20260424", "reportDate=20260425"]


def test_redact_nav_xml_masks_account_id():
    """accountId in attribute form (both quote styles) must not leak."""
    xml = '<FlexStatement accountId="U1234567" fromDate="20260427"><Foo masterAccountId=\'U7654321\'/></FlexStatement>'
    out = ibkr_flex.redact_nav_xml(xml)
    assert "U1234567" not in out
    assert "U7654321" not in out
    assert 'accountId="<REDACTED>"' in out
    assert "masterAccountId='<REDACTED>'" in out
    # Non-sensitive attributes are preserved verbatim
    assert 'fromDate="20260427"' in out


def test_redact_nav_xml_handles_empty_input():
    """Defensive: no crash on empty / None-equivalent input."""
    assert ibkr_flex.redact_nav_xml("") == ""


# ---------------------------------------------------------------------------
# /api/admin/ibkr/raw-nav-debug — founder gate + diagnostic shape
# ---------------------------------------------------------------------------


def test_admin_debug_blocks_non_founder(client):
    """Bearer auth alone isn't enough — the user must be the founder.
    Returns 200 OK + forbidden_not_admin so the response shape is uniform."""
    r = client.get("/api/admin/ibkr/raw-nav-debug")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error"] == "forbidden_not_admin"


def test_admin_debug_returns_diagnostic_shape_for_founder(client, monkeypatch):
    """Founder access: success=true, redacted XML snippet, attribute dump,
    date pairs. This is the response the user inspects in the browser."""
    import api.main as main
    # Make the test JWT count as founder for this scenario
    monkeypatch.setattr(main, "FOUNDER_USER_ID", _TEST_USER_ID)
    monkeypatch.setenv("IBKR_FLEX_TOKEN", "TOK")
    monkeypatch.setenv("IBKR_NAV_FLEX_QUERY_ID", "QID")

    # Mirror the real "Last Business Day" rollup case the user is hitting:
    # one row with toDate (no reportDate) — exactly the bug they're chasing.
    _stub_requests(
        monkeypatch,
        send_xml=_send_success_xml(),
        get_xml=_nav_report_xml([
            {"toDate": "20260424", "fromDate": "20260424",
             "accountId": "U1234567", "total": "487264.50",
             "cash": "-431003.85", "stock": "918268.35", "currency": "USD"},
        ]),
    )

    r = client.get("/api/admin/ibkr/raw-nav-debug")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True

    # The exact diagnostic that answers "why does no_data_for_date keep firing?"
    assert "toDate" in body["all_attrs_on_nav_row"]
    assert "reportDate" not in body["all_attrs_on_nav_row"]
    assert "toDate=20260424" in body["all_date_attrs_found"]
    assert body["parser_searches_for_attr"] == "reportDate"
    assert body["row_count"] == 1

    # Account ID is scrubbed from the raw snippet
    assert "U1234567" not in body["raw_xml_first_2000_chars"]
    assert "<REDACTED>" in body["raw_xml_first_2000_chars"]


def test_admin_debug_reports_missing_creds_for_founder(client, monkeypatch):
    """No env vars set → ibkr_not_configured (does not 500). Same shape."""
    import api.main as main
    monkeypatch.setattr(main, "FOUNDER_USER_ID", _TEST_USER_ID)
    # creds intentionally absent

    r = client.get("/api/admin/ibkr/raw-nav-debug")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error"] == "ibkr_not_configured"


def test_admin_debug_surfaces_ibkr_auth_failure(client, monkeypatch):
    """When IBKR rejects the credentials, the debug endpoint surfaces the
    same FlexQueryError code as the user-facing endpoint — but at 200 OK."""
    import api.main as main
    monkeypatch.setattr(main, "FOUNDER_USER_ID", _TEST_USER_ID)
    monkeypatch.setenv("IBKR_FLEX_TOKEN", "TOK")
    monkeypatch.setenv("IBKR_NAV_FLEX_QUERY_ID", "QID")

    _stub_requests(
        monkeypatch,
        send_xml=_send_error_xml("1003", "Invalid token"),
        get_xml="<unused/>",
    )

    r = client.get("/api/admin/ibkr/raw-nav-debug")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error"] == "ibkr_auth_failed"
