"""IBKR Client Portal API client for real-time account summary.

Counterpart to ibkr_flex.py — that module uses IBKR Flex Query (T+1, official
EOD numbers). This module talks to the local Client Portal Gateway for
SAME-DAY real-time NAV. Used by the EOD journal save script so the user
doesn't have to wait until next morning for finalized broker data.

The gateway is a Java app that must be running locally (typically on the
Mac mini at https://localhost:5050) and the user must have logged in via
browser. Sessions last ~24 hours before re-auth is required.
"""
from __future__ import annotations

from typing import Any

import requests
import urllib3

# Gateway uses a self-signed cert — suppress the urllib3 noise. We're talking
# to localhost only; cert verification adds nothing here.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_GATEWAY_URL = "https://localhost:5050"


class ClientPortalError(Exception):
    """Raised when the gateway is unreachable, unauthenticated, or returns
    an unexpected response. Caller should treat as a soft failure — fall
    back to manual entry rather than crashing the EOD save."""


def _check_auth(gateway_url: str) -> bool:
    """Return True if the gateway has an active authenticated session.

    Raises ClientPortalError if the gateway itself is unreachable (network
    error, gateway not running). Returns False (no exception) when the
    gateway answers but reports no active session — that's a "user needs
    to re-login" condition, distinguishable from "gateway is down."
    """
    try:
        r = requests.get(
            f"{gateway_url}/v1/api/iserver/auth/status", verify=False, timeout=5,
        )
        r.raise_for_status()
        return bool(r.json().get("authenticated"))
    except requests.exceptions.RequestException as e:
        raise ClientPortalError(f"Gateway unreachable at {gateway_url}: {e}")


def _init_portfolio(gateway_url: str) -> list[dict[str, Any]]:
    """Initialize the portfolio session.

    /v1/api/portfolio/{id}/summary returns empty until /portfolio/accounts
    has been called at least once per session — IBKR's quirk. We always
    call it before summary fetch even if we don't need the account list,
    because skipping it produces a silent empty response.
    """
    try:
        r = requests.get(
            f"{gateway_url}/v1/api/portfolio/accounts", verify=False, timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise ClientPortalError(f"portfolio/accounts init failed: {e}")


def fetch_account_summary(
    account_id: str, gateway_url: str = DEFAULT_GATEWAY_URL,
) -> dict[str, Any]:
    """Fetch a real-time account summary from the local Client Portal gateway.

    Returns a dict with normalized fields the journal save flow needs:
        account_id           — string IBKR account identifier
        nlv                  — Net Liquidation Value (the headline NAV)
        cash                 — Total cash value, negative when on margin
        position_value       — Gross market value of all positions
        equity_with_loan     — Equity with loan value (alt NAV measure)
        previous_day_equity  — Prior trading day's equity per IBKR
        currency             — ISO currency code (typically "USD")
        as_of_timestamp      — Server timestamp (epoch millis) the
                               numbers were valid as-of

    Raises ClientPortalError on:
        - Gateway unreachable (not running, port wrong)
        - Gateway running but not authenticated (user needs to log in)
        - HTTP error from the summary endpoint
    """
    if not _check_auth(gateway_url):
        raise ClientPortalError(
            f"Gateway is running but not authenticated. Open "
            f"{gateway_url} in a browser and log in to IBKR."
        )

    _init_portfolio(gateway_url)

    try:
        r = requests.get(
            f"{gateway_url}/v1/api/portfolio/{account_id}/summary",
            verify=False, timeout=10,
        )
        r.raise_for_status()
        raw = r.json()
    except requests.exceptions.RequestException as e:
        raise ClientPortalError(f"summary fetch for {account_id} failed: {e}")

    if not isinstance(raw, dict) or "netliquidation" not in raw:
        raise ClientPortalError(
            f"Unexpected summary response for {account_id}: missing "
            f"netliquidation field. Init may not have completed — try again."
        )

    def _amount(key: str) -> float:
        """Pull the numeric `amount` out of one of IBKR's wrapped value
        objects. Each summary field is shaped like
        {amount, currency, isNull, timestamp, value, severity}."""
        wrapper = raw.get(key)
        if not isinstance(wrapper, dict):
            return 0.0
        try:
            return float(wrapper.get("amount") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    nlv_wrapper = raw.get("netliquidation", {}) or {}
    return {
        "account_id": account_id,
        "nlv": round(_amount("netliquidation"), 2),
        "cash": round(_amount("totalcashvalue"), 2),
        "position_value": round(_amount("grosspositionvalue"), 2),
        "equity_with_loan": round(_amount("equitywithloanvalue"), 2),
        "previous_day_equity": round(_amount("previousdayequitywithloanvalue"), 2),
        "currency": nlv_wrapper.get("currency") or "USD",
        "as_of_timestamp": int(nlv_wrapper.get("timestamp") or 0),
    }
