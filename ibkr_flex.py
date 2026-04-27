# ibkr_flex.py — IBKR Flex Query client for pulling trade confirmations
#
# Two-step HTTP flow:
#   1. Request report → get a reference code
#   2. Download report XML using the reference code
#   3. Parse XML into a clean DataFrame of executions
#
# Requires: IBKR_FLEX_TOKEN + IBKR_FLEX_QUERY_ID env vars
# (set on Railway in production; export locally for dev).

import re
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import os
from datetime import datetime, date, timedelta
try:
    from zoneinfo import ZoneInfo
    _TZ_ET = ZoneInfo("America/New_York")
    _TZ_CT = ZoneInfo("America/Chicago")
except ImportError:
    _TZ_ET = None
    _TZ_CT = None


class FlexQueryError(Exception):
    """Raised by NAV-pull helpers. `code` is a stable machine identifier the
    HTTP layer maps to the response body's `error` field; `message` is human-
    readable. The endpoint always returns 200 OK and embeds these in the JSON
    so the frontend can render a fallback banner without parsing HTTP errors.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _et_to_ct(date_str: str, time_str: str):
    """Convert (YYYY-MM-DD, HH:MM:SS) from Eastern to Central time.
    IBKR Flex reports are published in the account's configured timezone,
    which for US accounts defaults to ET. We normalise to CT to match the
    rest of the trading journal (per CLAUDE.md, all timestamps are
    America/Chicago). Handles DST via zoneinfo. Returns ("", "") on error.
    """
    if not _TZ_ET or not _TZ_CT or not date_str or not time_str:
        return ("", "")
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=_TZ_ET).astimezone(_TZ_CT)
        return (dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"))
    except Exception:
        return ("", "")


# IBKR Flex Web Service endpoints
_BASE = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService"
_SEND_URL = f"{_BASE}.SendRequest"
_GET_URL = f"{_BASE}.GetStatement"


def _get_credentials():
    """Read IBKR Flex credentials from env vars (Railway service config)."""
    token = os.environ.get("IBKR_FLEX_TOKEN", "")
    query_id = os.environ.get("IBKR_FLEX_QUERY_ID", "")
    if token and query_id:
        return token.strip(), query_id.strip(), None
    return None, None, "IBKR credentials not configured. Set IBKR_FLEX_TOKEN and IBKR_FLEX_QUERY_ID env vars."


def fetch_flex_report(token=None, query_id=None, max_retries=5, retry_delay=3):
    """
    Pull a Flex Query report from IBKR.

    Args:
        token: Flex Web Service token (reads from secrets if None)
        query_id: Flex Query ID (reads from secrets if None)
        max_retries: How many times to retry if report is still generating
        retry_delay: Seconds between retries

    Returns:
        (xml_root, error_message) — xml_root is an ElementTree Element on
        success, None on failure. error_message is None on success.
    """
    if token is None or query_id is None:
        token, query_id, err = _get_credentials()
        if err:
            return None, err

    # Step 1: Request the report
    try:
        resp = requests.get(
            _SEND_URL,
            params={"t": token, "q": query_id, "v": "3"},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return None, f"IBKR request failed: {e}"

    # Parse the response to get the reference code
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return None, f"IBKR returned invalid XML: {resp.text[:500]}"

    status = root.findtext("Status", "")
    if status != "Success":
        code = root.findtext("ErrorCode", "")
        msg = root.findtext("ErrorMessage", "Unknown error")
        return None, f"IBKR request error ({code}): {msg}"

    ref_code = root.findtext("ReferenceCode", "")
    if not ref_code:
        return None, "IBKR returned success but no reference code"

    # Step 2: Download the report (with retry for generation delay)
    for attempt in range(max_retries):
        try:
            resp2 = requests.get(
                _GET_URL,
                params={"q": ref_code, "t": token, "v": "3"},
                timeout=30,
            )
            resp2.raise_for_status()
        except requests.RequestException as e:
            return None, f"IBKR download failed: {e}"

        # Check if it's a "still generating" warning
        try:
            check_root = ET.fromstring(resp2.text)
            check_status = check_root.findtext("Status", "")
            if check_status == "Warn":
                error_code = check_root.findtext("ErrorCode", "")
                if error_code == "1019":
                    # Report still generating — retry
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return None, "IBKR report generation timed out. Try again in a few seconds."
                else:
                    msg = check_root.findtext("ErrorMessage", "Unknown warning")
                    return None, f"IBKR warning ({error_code}): {msg}"
        except ET.ParseError:
            # Not XML error response — might be the actual report. Continue.
            pass

        # Parse the actual report
        try:
            report_root = ET.fromstring(resp2.text)
            return report_root, None
        except ET.ParseError:
            return None, f"IBKR returned unparseable report: {resp2.text[:500]}"

    return None, "IBKR report download failed after retries"


def parse_trade_confirms(xml_root):
    """
    Parse a Flex Query XML report and extract trade confirmations.

    Args:
        xml_root: ElementTree root element from fetch_flex_report()

    Returns:
        pandas DataFrame with columns:
          account, trade_date, settle_date, symbol, description,
          asset_class, action, quantity, price, amount, commission,
          net_cash, currency, order_time, put_call, strike, expiry
    """
    trades = []

    # Navigate: FlexQueryResponse > FlexStatements > FlexStatement > TradeConfirms > TradeConfirm
    # The exact nesting can vary; search broadly
    for confirm in xml_root.iter("TradeConfirm"):
        attribs = confirm.attrib

        # Map IBKR buySell to our action convention
        raw_action = attribs.get("buySell", "").strip().upper()
        if raw_action in ("BUY", "BOT"):
            action = "BUY"
        elif raw_action in ("SELL", "SLD"):
            action = "SELL"
        else:
            action = raw_action  # keep as-is for unknowns

        # Parse quantity (IBKR sometimes uses negative for sells)
        qty = abs(float(attribs.get("quantity", 0)))

        # Parse price — IBKR uses different field names across report types
        price = 0.0
        for price_key in ("tradePrice", "price", "tradeMoney", "closePrice"):
            if price_key in attribs and float(attribs[price_key] or 0) != 0:
                price = float(attribs[price_key])
                break

        # Parse amount / commission / net_cash — try multiple field names
        amount = 0.0
        for amt_key in ("amount", "proceeds", "cost", "tradeMoney"):
            if amt_key in attribs:
                amount = abs(float(attribs[amt_key] or 0))
                if amount > 0:
                    break
        # Fallback: compute from qty * price
        if amount == 0 and qty > 0 and price > 0:
            amount = qty * price

        commission = float(attribs.get("commission", attribs.get("ibCommission", 0)) or 0)
        net_cash = float(attribs.get("netCash", attribs.get("net", 0)) or 0)

        # Format trade date
        trade_date_raw = attribs.get("tradeDate", "")
        trade_date = ""
        if len(trade_date_raw) == 8:
            trade_date = f"{trade_date_raw[:4]}-{trade_date_raw[4:6]}-{trade_date_raw[6:8]}"
        else:
            trade_date = trade_date_raw

        # Format order time — IBKR's Flex XML uses several possible field
        # names depending on how the query is configured. Try in priority
        # order and normalise to HH:MM:SS.
        def _fmt_time(raw: str) -> str:
            raw = (raw or "").strip()
            if not raw:
                return ""
            # Combined "YYYYMMDD;HHMMSS" or "YYYY-MM-DD HH:MM:SS" timestamps
            if ";" in raw:
                raw = raw.split(";", 1)[1]
            elif "T" in raw:
                raw = raw.split("T", 1)[1]
            elif " " in raw:
                raw = raw.split(" ", 1)[1]
            # Strip timezone suffixes like "+0000"
            for sep in ("+", "-", "Z"):
                if sep in raw[3:]:
                    raw = raw.split(sep)[0]
                    break
            raw = raw.strip()
            # Already HH:MM:SS or HH:MM — pass through (pad seconds if needed)
            if ":" in raw:
                parts = raw.split(":")
                if len(parts) >= 3:
                    return f"{parts[0][:2]}:{parts[1][:2]}:{parts[2][:2]}"
                if len(parts) == 2:
                    return f"{parts[0][:2]}:{parts[1][:2]}:00"
                return raw
            # Bare HHMMSS or HHMM
            if len(raw) >= 6 and raw.isdigit():
                return f"{raw[:2]}:{raw[2:4]}:{raw[4:6]}"
            if len(raw) == 4 and raw.isdigit():
                return f"{raw[:2]}:{raw[2:4]}:00"
            return raw

        # IMPORTANT: do not include report-wide timestamps (reportDate,
        # whenGenerated) — those are the report-generation time and would
        # paint every row with the same fetch timestamp.
        order_time = ""
        for time_key in ("orderTime", "tradeTime", "executionTime", "transactionTime", "dateTime"):
            candidate = _fmt_time(attribs.get(time_key, ""))
            if candidate and ":" in candidate:
                order_time = candidate
                break

        # Convert ET → CT. IBKR Flex reports stamp times in the account's
        # configured timezone (ET for US accounts); the rest of the app runs
        # in America/Chicago. Also update trade_date if the conversion rolls
        # it over to an adjacent day (rare but possible for very early /
        # very late fills).
        if order_time and trade_date:
            ct_date, ct_time = _et_to_ct(trade_date, order_time)
            if ct_time:
                order_time = ct_time
                trade_date = ct_date

        # Option fields
        put_call = attribs.get("putCall", "").strip()
        strike = attribs.get("strike", "").strip()
        expiry_raw = attribs.get("expiry", "").strip()
        expiry = ""
        if len(expiry_raw) == 8:
            expiry = f"{expiry_raw[:4]}-{expiry_raw[4:6]}-{expiry_raw[6:8]}"
        else:
            expiry = expiry_raw

        # Order ID — same for all partial fills of one order. Primary key
        # for consolidation (more reliable than price which can differ by
        # fractions across fills).
        order_id = attribs.get("orderID", attribs.get("ibOrderID", attribs.get("orderReference", "")))

        trades.append({
            "account": attribs.get("accountId", ""),
            "trade_date": trade_date,
            "settle_date": attribs.get("settleDateTarget", ""),
            "symbol": attribs.get("symbol", "").strip().upper(),
            "description": attribs.get("description", "").strip(),
            "asset_class": attribs.get("assetCategory", "").strip(),
            "action": action,
            "quantity": qty,
            "price": price,
            "amount": amount,
            "commission": commission,
            "net_cash": net_cash,
            "currency": attribs.get("currency", "USD"),
            "order_time": order_time,
            "put_call": put_call,
            "strike": strike,
            "expiry": expiry,
            "order_id": str(order_id).strip(),
        })

    if not trades:
        return pd.DataFrame(columns=[
            "account", "trade_date", "settle_date", "symbol", "description",
            "asset_class", "action", "quantity", "price", "amount",
            "commission", "net_cash", "currency", "order_time",
            "put_call", "strike", "expiry", "order_id",
        ])

    df = pd.DataFrame(trades)
    # Sort by date + time
    df = df.sort_values(["trade_date", "order_time"], ascending=[False, False])
    return df


def consolidate_partial_fills(df):
    """
    IBKR often fills a single order in multiple lots (e.g., SELL 435 TQQQ
    gets filled as 35 + 200 + 200 at slightly different prices). Consolidate
    these into one row per logical order.

    Primary grouping key: order_id (IBKR assigns the same order ID to all
    partial fills of one order). Falls back to symbol + action + trade_date +
    order_time if order_id is missing.

    Price is computed as a weighted average: sum(qty * price) / sum(qty).
    """
    if df.empty:
        return df

    # Determine grouping strategy
    has_order_id = 'order_id' in df.columns and df['order_id'].notna().any() and (df['order_id'] != '').any()

    if has_order_id:
        # Primary: group by order_id (most reliable — same order, different fills)
        group_keys = ['order_id']
    else:
        # Fallback: group by symbol + action + date + time
        group_keys = ['symbol', 'action', 'trade_date', 'order_time',
                      'asset_class', 'put_call', 'strike', 'expiry']

    group_keys = [k for k in group_keys if k in df.columns]

    # Compute weighted price before grouping
    df = df.copy()
    df['_value'] = df['quantity'] * df['price']

    agg_rules = {
        'quantity': 'sum',
        '_value': 'sum',
        'amount': 'sum',
        'commission': 'sum',
        'net_cash': 'sum',
        'account': 'first',
        'trade_date': 'first',
        'order_time': 'first',
        'settle_date': 'first',
        'symbol': 'first',
        'description': 'first',
        'asset_class': 'first',
        'action': 'first',
        'currency': 'first',
        'put_call': 'first',
        'strike': 'first',
        'expiry': 'first',
    }
    if has_order_id:
        agg_rules['order_id'] = 'first'
    agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

    consolidated = df.groupby(group_keys, as_index=False).agg(agg_rules)

    # Weighted average price
    consolidated['price'] = consolidated['_value'] / consolidated['quantity']
    consolidated['price'] = consolidated['price'].round(4)
    consolidated = consolidated.drop(columns=['_value'], errors='ignore')

    consolidated = consolidated.sort_values(['trade_date', 'order_time'], ascending=[False, False])
    return consolidated


def get_raw_debug_info(xml_root):
    """Extract raw XML attributes from the first TradeConfirm for debugging.
    Returns dict with 'first' (first trade) and 'first_opt' (first option trade)."""
    result = {}
    try:
        confirms = list(xml_root.iter("TradeConfirm"))
        if confirms:
            result['first'] = dict(confirms[0].attrib)
            # Also find first option trade specifically
            for c in confirms:
                if c.attrib.get('assetCategory', '') == 'OPT':
                    result['first_opt'] = dict(c.attrib)
                    break
    except Exception:
        pass
    return result


def group_same_day_orders(df):
    """
    Second-stage consolidation: after partial fills are merged by order_id,
    also group multiple distinct orders placed the same day for the same
    ticker + side into a single logical transaction. Prevents the log flow
    from creating B1+B2 (or A1+A2+A3) when the user really made one decision
    that got executed in multiple installments over the course of the day.

    Grouping key: trade_date + symbol + action + asset_class + put_call +
    strike + expiry (options need the contract specifics too so different
    strikes on the same underlying don't collapse).

    Price is the share-weighted average across all orders. Earliest
    order_time is kept (represents when the user first committed to the
    position that day). order_id becomes a comma-joined list so the grouping
    is traceable; downstream code only uses it as a passthrough string.
    """
    if df.empty:
        return df

    group_keys = ['trade_date', 'symbol', 'action']
    for optional in ('asset_class', 'put_call', 'strike', 'expiry'):
        if optional in df.columns:
            group_keys.append(optional)

    df = df.copy()
    df['_value'] = df['quantity'] * df['price']

    agg_rules = {
        'quantity': 'sum',
        '_value': 'sum',
        'amount': 'sum',
        'commission': 'sum',
        'net_cash': 'sum',
        'account': 'first',
        'order_time': 'min',
        'settle_date': 'first',
        'description': 'first',
        'currency': 'first',
        'order_id': lambda s: ",".join(sorted({str(x) for x in s if str(x).strip() and str(x) != 'nan'})),
    }
    agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

    # dropna=False keeps rows whose option-specific keys (put_call, strike,
    # expiry) are NaN for stocks — otherwise pandas silently drops them.
    grouped = df.groupby(group_keys, as_index=False, dropna=False).agg(agg_rules)
    grouped['price'] = grouped['_value'] / grouped['quantity']
    grouped['price'] = grouped['price'].round(4)
    grouped = grouped.drop(columns=['_value'], errors='ignore')
    grouped = grouped.sort_values(['trade_date', 'order_time'], ascending=[False, False])
    return grouped


def _format_date_yyyymmdd(d) -> str:
    """Accepts date | str(YYYY-MM-DD) | str(YYYYMMDD). Returns YYYYMMDD."""
    if isinstance(d, date):
        return d.strftime("%Y%m%d")
    s = str(d).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s.replace("-", "")
    return s


def _to_iso_date(s: str) -> str:
    """YYYYMMDD → YYYY-MM-DD. Returns the input untouched if it doesn't match."""
    s = (s or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def _read_field(elem, *names) -> str:
    """Read a value from either an element attribute or a child element. IBKR
    Flex Queries serialise NAV data as XML attributes (e.g. <EquitySummaryByReportDateInBase
    reportDate="20260427" total="..." />), but custom queries can also produce
    child-element form. Try attributes first (the common case), then children.
    """
    for n in names:
        if n in elem.attrib:
            v = elem.attrib.get(n, "").strip()
            if v:
                return v
    for n in names:
        child = elem.find(n)
        if child is not None and child.text:
            v = child.text.strip()
            if v:
                return v
    return ""


def _last_completed_trading_day(now_et: datetime = None) -> date:
    """Default for the endpoint when no `date` param is supplied.

    Rule: if it's a weekday and past 6 PM ET, IBKR has likely finalised today's
    NAV — return today. Otherwise step back to the most recent weekday. Doesn't
    know about market holidays; the puller will raise no_data_for_date if IBKR
    hasn't closed the books for the requested day, which is the right signal.
    """
    if _TZ_ET is None:
        now_et = now_et or datetime.now()
    else:
        now_et = now_et or datetime.now(_TZ_ET)

    d = now_et.date()
    is_weekday = d.weekday() < 5
    if is_weekday and now_et.hour >= 18:
        return d
    # Step back to previous weekday
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d


def _fetch_nav_xml(query_id: str, token: str):
    """Run the IBKR Flex Query Send → Get → retry protocol for the NAV query
    and return the raw report XML root. Pure transport — no parsing of the
    report's payload schema. Raises FlexQueryError on any failure with the
    same machine-readable codes documented on fetch_nav_for_date().

    Extracted so the admin debug endpoint can introspect the raw XML without
    going through schema-specific parsing.
    """
    # ── Step 1: request the report ────────────────────────────────────────
    try:
        resp = requests.get(
            _SEND_URL,
            params={"t": token, "q": query_id, "v": "3"},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.Timeout as e:
        raise FlexQueryError("network_timeout", f"Timed out reaching IBKR: {e}")
    except requests.RequestException as e:
        raise FlexQueryError("network_timeout", f"Network error reaching IBKR: {e}")

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        raise FlexQueryError(
            "ibkr_report_error",
            f"IBKR returned invalid XML: {resp.text[:300]}",
        )

    status = root.findtext("Status", "")
    if status != "Success":
        code = root.findtext("ErrorCode", "")
        msg = root.findtext("ErrorMessage", "Unknown error")
        # Auth-style errors: bad token, invalid query ID, account not enabled
        if code in ("1003", "1004", "1005", "1011", "1012"):
            raise FlexQueryError(
                "ibkr_auth_failed",
                f"IBKR rejected the request ({code}): {msg}",
            )
        raise FlexQueryError(
            "ibkr_report_error",
            f"IBKR request error ({code}): {msg}",
        )

    ref_code = root.findtext("ReferenceCode", "")
    if not ref_code:
        raise FlexQueryError(
            "ibkr_report_error",
            "IBKR returned success but no reference code",
        )

    # ── Step 2: download the report (with retry for generation delay) ─────
    max_retries = 5
    retry_delay = 3
    for attempt in range(max_retries):
        try:
            resp2 = requests.get(
                _GET_URL,
                params={"q": ref_code, "t": token, "v": "3"},
                timeout=30,
            )
            resp2.raise_for_status()
        except requests.Timeout as e:
            raise FlexQueryError("network_timeout", f"Timed out downloading report: {e}")
        except requests.RequestException as e:
            raise FlexQueryError("network_timeout", f"Network error downloading report: {e}")

        try:
            check_root = ET.fromstring(resp2.text)
        except ET.ParseError:
            raise FlexQueryError(
                "ibkr_report_error",
                f"IBKR returned unparseable report: {resp2.text[:300]}",
            )

        # "Still generating" warning vs the actual report
        check_status = check_root.findtext("Status", "")
        if check_status == "Warn":
            error_code = check_root.findtext("ErrorCode", "")
            if error_code == "1019":
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise FlexQueryError(
                    "report_generation_timeout",
                    "IBKR report is still generating after retries — try again in a few seconds",
                )
            msg = check_root.findtext("ErrorMessage", "Unknown warning")
            raise FlexQueryError(
                "ibkr_report_error",
                f"IBKR warning ({error_code}): {msg}",
            )
        # Status empty/absent means this is the actual data payload.
        return check_root

    raise FlexQueryError(
        "report_generation_timeout",
        "IBKR report download failed after retries",
    )


def fetch_nav_for_date(query_id: str = None, target_date=None, token: str = None) -> dict:
    """Pull NAV from IBKR Flex Query for a single trading day.

    The NAV puller is a *separate* Flex Query from the trade-confirms one — it
    has its own query_id (env var IBKR_NAV_FLEX_QUERY_ID) and reports
    EquitySummaryByReportDateInBase rows. Token is shared with the trade puller
    (IBKR_FLEX_TOKEN).

    Args:
        query_id: NAV Flex Query ID (defaults to IBKR_NAV_FLEX_QUERY_ID env var)
        target_date: date | str. Defaults to last completed trading day.
        token: Flex token (defaults to IBKR_FLEX_TOKEN env var)

    Returns: dict with keys: date (YYYY-MM-DD str), nav (float), cash_balance
        (float), position_value (float), currency (str), account (str).

    Raises FlexQueryError with one of these `code`s:
        - ibkr_not_configured: missing token or query_id env vars
        - ibkr_auth_failed: IBKR rejected the request (bad token / query_id)
        - network_timeout: connection / timeout error reaching IBKR
        - ibkr_report_error: IBKR returned a Status=Error or unexpected XML
        - report_generation_timeout: report still generating after retries
        - no_data_for_date: report parsed but no NAV row matches target_date
        - parse_error: XML didn't contain expected EquitySummary structure
    """
    # Resolve credentials — env-var defaults are intentional so callers don't
    # have to plumb them through; explicit args are the test-injection seam.
    if token is None:
        token = os.environ.get("IBKR_FLEX_TOKEN", "").strip()
    if query_id is None:
        query_id = os.environ.get("IBKR_NAV_FLEX_QUERY_ID", "").strip()

    if not token or not query_id:
        raise FlexQueryError(
            "ibkr_not_configured",
            "IBKR NAV puller not configured. Set IBKR_FLEX_TOKEN and "
            "IBKR_NAV_FLEX_QUERY_ID env vars.",
        )

    # Resolve target date
    if target_date is None:
        target_date = _last_completed_trading_day()
    if isinstance(target_date, str):
        try:
            target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise FlexQueryError(
                "invalid_date",
                f"target_date must be YYYY-MM-DD, got: {target_date!r}",
            )
    target_yyyymmdd = _format_date_yyyymmdd(target_date)

    report_xml = _fetch_nav_xml(query_id, token)
    return _parse_nav_report(report_xml, target_yyyymmdd)


def _parse_nav_report(xml_root, target_yyyymmdd: str) -> dict:
    """Extract a single day's NAV row from the EquitySummary section.

    Real IBKR shape (verified against MO_NAV_Daily, period=LastBusinessDay):

        <FlexStatement fromDate="20260424" toDate="20260424"
                       period="LastBusinessDay" whenGenerated="..." accountId="...">
          <AccountInformation accountId="..." />
          <EquitySummaryInBase>
            <EquitySummaryByReportDateInBase cash=".." stock=".." total=".." ... />
            <EquitySummaryByReportDateInBase cash=".." stock=".." total=".." ... />
          </EquitySummaryInBase>
        </FlexStatement>

    Two facts the original parser had wrong, both surfaced by the admin
    debug endpoint:
      1. The data date lives on the *parent* `FlexStatement.toDate` —
         the EquitySummary rows themselves carry no date attribute.
      2. Period=LastBusinessDay produces *two* rows: opening + closing.
         The closing one (last in document order) is what NLV semantics
         demand. Earlier output `total=471004.89` was the open; correct
         EOD value is `total=486630.39` (the second row).
    """
    all_rows = list(xml_root.iter("EquitySummaryByReportDateInBase"))
    matched_row = None
    matched_iso_date = None

    # Two distinct schemas — pick the matching strategy from the data shape:
    #
    # (A) Rows carry their own reportDate (custom Flex Query configurations,
    #     range queries, or legacy setups). Row-level dates are
    #     authoritative — we don't fall back to FlexStatement matching when
    #     they're present, because mixing the two could silently return a
    #     row tagged "4/25" when the wrapper says "4/27" — that's a config
    #     mismatch, not data we should claim is the requested day's NAV.
    # (B) Rows have no date attribute; date lives on the parent
    #     <FlexStatement toDate=...>. This is the *real* shape for the
    #     standard Net Asset Value (NAV) section with period=LastBusinessDay
    #     and similar single-day periods. Two rows (open + close) appear
    #     inside one FlexStatement; take the last (EOD).
    rows_have_report_date = any(_read_field(r, "reportDate") for r in all_rows)
    if rows_have_report_date:
        for row in all_rows:
            if _read_field(row, "reportDate") == target_yyyymmdd:
                matched_row = row
                matched_iso_date = _to_iso_date(target_yyyymmdd)
                break
    else:
        for fs in xml_root.iter("FlexStatement"):
            fs_to = (fs.attrib.get("toDate") or "").strip()
            fs_from = (fs.attrib.get("fromDate") or "").strip()
            if target_yyyymmdd in (fs_to, fs_from):
                inner = list(fs.iter("EquitySummaryByReportDateInBase"))
                if inner:
                    matched_row = inner[-1]
                    matched_iso_date = _to_iso_date(fs_to or fs_from)
                    break
        # Single-row rollup fallback — toDate lives on the row itself
        # (rare configuration). Only triggered when rows have no reportDate.
        if matched_row is None and len(all_rows) == 1:
            only = all_rows[0]
            row_to_date = _read_field(only, "toDate", "fromDate", "reportDate")
            if row_to_date == target_yyyymmdd:
                matched_row = only
                matched_iso_date = _to_iso_date(row_to_date)

    if matched_row is None:
        all_rows = list(xml_root.iter("EquitySummaryByReportDateInBase"))
        if not all_rows:
            raise FlexQueryError(
                "parse_error",
                "Flex Query response did not contain EquitySummaryByReportDateInBase "
                "rows. Make sure 'Net Asset Value (NAV)' is enabled in the query "
                "configuration.",
            )
        # Build the available-dates suffix from BOTH FlexStatement toDates
        # (the real IBKR shape) and any reportDate on rows (for compat
        # configurations). De-dup, preserve insertion order.
        avail = []
        seen = set()
        for fs in xml_root.iter("FlexStatement"):
            d = (fs.attrib.get("toDate") or "").strip()
            if d and d not in seen:
                seen.add(d)
                avail.append(d)
        for row in all_rows:
            d = _read_field(row, "reportDate")
            if d and d not in seen:
                seen.add(d)
                avail.append(d)

        msg = (
            f"No NAV data for {_to_iso_date(target_yyyymmdd)} — possibly "
            f"market not yet closed or IBKR not yet finalised."
        )
        if avail:
            msg += f" Available dates in report: {', '.join(avail[:5])}"
        raise FlexQueryError("no_data_for_date", msg)

    def _num(v) -> float:
        try:
            return float(v) if v not in (None, "", "null") else 0.0
        except ValueError:
            return 0.0

    nav = _num(_read_field(matched_row, "total"))
    cash = _num(_read_field(matched_row, "cash"))
    stock = _num(_read_field(matched_row, "stock"))
    # Sum non-stock position categories. Field names are PLURAL in IBKR's
    # actual responses (bonds/options/commodities/notes) — earlier code
    # used singular forms and silently summed to zero. interestAccruals is
    # the field on the row; older guess was just "interest".
    extras = sum(
        _num(_read_field(matched_row, f))
        for f in ("bonds", "options", "commodities", "notes",
                  "crypto", "interestAccruals", "marginFinancingChargeAccruals")
    )
    position_value = stock + extras
    if position_value == 0 and nav != 0:
        # Final fallback: if we couldn't parse positions, derive from nav-cash
        position_value = nav - cash

    if nav == 0:
        raise FlexQueryError(
            "parse_error",
            "Matched NAV row had total=0 — unexpected. Check Flex Query "
            "configuration includes the NAV section.",
        )

    # Account ID — pull from the row, fall back to the surrounding
    # FlexStatement element. Currency defaults to USD if the report uses base
    # currency without explicit tagging.
    account = _read_field(matched_row, "accountId")
    if not account:
        for fs in xml_root.iter("FlexStatement"):
            account = fs.attrib.get("accountId", "")
            if account:
                break

    currency = _read_field(matched_row, "currency") or "USD"

    return {
        "date": matched_iso_date or _to_iso_date(target_yyyymmdd),
        "nav": round(nav, 2),
        "cash_balance": round(cash, 2),
        "position_value": round(position_value, 2),
        "currency": currency,
        "account": account,
    }


# ── NAV diagnostic helpers ─────────────────────────────────────────────────
# Powering /api/admin/ibkr/raw-nav-debug. The parser keys off `reportDate`
# but real IBKR Flex Queries can use other date attribute names depending
# on the configured period (Last Business Day vs Range vs Today). When the
# parser raises no_data_for_date and the user can't see why, these helpers
# expose the raw shape so we can identify the mismatch.

# Attribute names that have ever surfaced in Flex Query NAV responses across
# the period configurations IBKR offers. The inspector reports any of these
# that are present so we can update the parser if a new one shows up.
_KNOWN_DATE_ATTR_NAMES = (
    "reportDate", "toDate", "fromDate", "asOfDate", "date", "statementDate",
    "periodEnd", "periodStart", "tradeDate", "settleDate",
)


def _yyyymmdd_shape(s: str) -> bool:
    """True if `s` looks like an IBKR YYYYMMDD date stamp."""
    return bool(s) and len(s) == 8 and s.isdigit()


def inspect_nav_xml(xml_root) -> dict:
    """Extract structural info from a NAV Flex Query XML root for diagnostics.

    Returns a dict with the shape the admin debug endpoint serves. No PII
    in the output — account IDs are reported as a presence flag, not the
    actual value (the redacted XML snippet handles them separately).

    Reports:
      - row_count: number of EquitySummaryByReportDateInBase rows
      - all_attrs_on_nav_row: union of attribute keys across all rows (sorted)
      - all_date_attrs_found: every (key, value) pair across rows where the
        value is YYYYMMDD-shaped — discriminates `toDate=20260424` from
        `reportDate=20260424`, which is the whole point of this endpoint
      - parser_searches_for_attr: what the parser currently keys on (so the
        user can confirm the mismatch at a glance)
      - alternate_top_level_tags: other tag names that appear at the top
        level under FlexStatement (helps identify if NAV section name has
        drifted, e.g. EquitySummaryByReportDate vs EquitySummaryInBase)
    """
    rows = list(xml_root.iter("EquitySummaryByReportDateInBase"))

    # Union of all attribute keys
    all_attrs = set()
    for r in rows:
        all_attrs.update(r.attrib.keys())

    # Date-shaped attribute values, dedup on (k, v) so the same date appearing
    # under multiple keys gets surfaced under each one.
    date_pairs = []
    seen = set()
    for r in rows:
        for k, v in r.attrib.items():
            v_stripped = (v or "").strip()
            if _yyyymmdd_shape(v_stripped):
                pair = f"{k}={v_stripped}"
                if pair not in seen:
                    seen.add(pair)
                    date_pairs.append(pair)

    # Top-level structural breadcrumbs — what tags are inside FlexStatement?
    top_tags = set()
    for fs in xml_root.iter("FlexStatement"):
        for child in list(fs):
            top_tags.add(child.tag)
        # Some configurations nest the data directly under FlexStatement
        # without an intermediate wrapper; in that case top_tags will already
        # show the wrapper-or-data tag mix.
        break  # FlexStatement is single-account; first one is enough

    # If there are zero EquitySummaryByReportDateInBase rows, surface any
    # close-name candidates (e.g. EquitySummaryByReportDate, NetAssetValue)
    # so the user can spot a renaming.
    candidate_tags = []
    if not rows:
        for elem in xml_root.iter():
            tag = elem.tag
            if "Equity" in tag or "NAV" in tag.upper() or "NetAsset" in tag:
                if tag not in candidate_tags:
                    candidate_tags.append(tag)

    # FlexStatement-level date attrs — the real IBKR shape stores the data
    # date on the parent <FlexStatement>, not on the rows. Reporting both
    # is what made the original bug visible at a glance.
    flex_statement_dates = []
    fs_seen = set()
    for fs in xml_root.iter("FlexStatement"):
        for k in ("fromDate", "toDate", "period"):
            v = (fs.attrib.get(k) or "").strip()
            if v:
                pair = f"{k}={v}"
                if pair not in fs_seen:
                    fs_seen.add(pair)
                    flex_statement_dates.append(pair)

    return {
        "row_count": len(rows),
        "all_attrs_on_nav_row": sorted(all_attrs),
        "all_date_attrs_found": date_pairs,
        "flex_statement_attrs": flex_statement_dates,
        "parser_searches_for_attr": "FlexStatement.toDate (primary), reportDate on row (legacy)",
        "alternate_top_level_tags": sorted(top_tags),
        "candidate_nav_tags_when_zero_rows": candidate_tags,
    }


def redact_nav_xml(text: str) -> str:
    """Strip account-identifying values from raw IBKR XML before showing it
    in a JSON response. Three classes of redaction:

    1. accountId attributes — anywhere they appear (FlexStatement,
       EquitySummary rows, etc.). Replaced with "<REDACTED>".
    2. Other id-shaped attributes that could leak account scope:
       masterAccountId, custodianAccountId, brokerageAccountId.
    3. Bare account-number-shaped values inside element text (defence-in-depth).

    The token never appears in the response body (it's only a query parameter
    on the request), but we'd add token redaction here if that ever changes.
    """
    if not text:
        return text
    out = text
    # Attribute redactions — match k="..." or k='...'
    for attr in ("accountId", "masterAccountId", "custodianAccountId",
                 "brokerageAccountId", "ibAccount"):
        out = re.sub(
            rf'{attr}="[^"]*"', f'{attr}="<REDACTED>"', out,
        )
        out = re.sub(
            rf"{attr}='[^']*'", f"{attr}='<REDACTED>'", out,
        )
    return out


def pull_ibkr_trades(consolidate=True, group_same_day=True):
    """
    Convenience function: fetch + parse + optionally consolidate in one call.

    Args:
        consolidate: If True, merge partial fills into single rows (order_id).
        group_same_day: If True, additionally group same-day same-ticker same-side
            orders into one row (prevents spurious B1/B2 or A1/A2/A3 from the
            log flow when the user made one logical buy in multiple installments).

    Returns:
        (DataFrame, raw_debug_dict, error_message)
    """
    xml_root, err = fetch_flex_report()
    if err:
        return pd.DataFrame(), {}, err

    raw_debug = get_raw_debug_info(xml_root)
    df = parse_trade_confirms(xml_root)
    if consolidate and not df.empty:
        df = consolidate_partial_fills(df)
    if group_same_day and not df.empty:
        df = group_same_day_orders(df)
    return df, raw_debug, None
