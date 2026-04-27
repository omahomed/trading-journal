# ibkr_flex.py — IBKR Flex Query client for pulling trade confirmations
#
# Two-step HTTP flow:
#   1. Request report → get a reference code
#   2. Download report XML using the reference code
#   3. Parse XML into a clean DataFrame of executions
#
# Requires: IBKR_FLEX_TOKEN + IBKR_FLEX_QUERY_ID env vars
# (set on Railway in production; export locally for dev).

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
    report_xml = None
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

        # Check for "still generating" warning vs the actual report
        try:
            check_root = ET.fromstring(resp2.text)
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
            # Real report — Status will be empty or absent for the data XML
            report_xml = check_root
            break
        except ET.ParseError:
            raise FlexQueryError(
                "ibkr_report_error",
                f"IBKR returned unparseable report: {resp2.text[:300]}",
            )

    if report_xml is None:
        raise FlexQueryError(
            "report_generation_timeout",
            "IBKR report download failed after retries",
        )

    return _parse_nav_report(report_xml, target_yyyymmdd)


def _parse_nav_report(xml_root, target_yyyymmdd: str) -> dict:
    """Extract a single day's NAV row from the EquitySummary section.

    IBKR Flex Queries with the "Net Asset Value (NAV)" section enabled emit
    EquitySummaryByReportDateInBase rows — one per report date. We match on
    reportDate, with a fallback to the parent FlexStatement's toDate when the
    summary is rolled up rather than per-day.
    """
    rows = list(xml_root.iter("EquitySummaryByReportDateInBase"))

    matched = None
    for row in rows:
        report_date = _read_field(row, "reportDate")
        if report_date == target_yyyymmdd:
            matched = row
            break

    # Fallback: some Flex Queries emit a single rollup row (no reportDate or
    # only fromDate/toDate). If exactly one row exists and its toDate matches
    # the requested day, accept it.
    if matched is None and len(rows) == 1:
        only = rows[0]
        to_date = _read_field(only, "toDate", "fromDate", "reportDate")
        if to_date == target_yyyymmdd:
            matched = only

    if matched is None:
        avail = [_read_field(r, "reportDate") for r in rows]
        avail = [a for a in avail if a]
        msg = (
            f"No NAV data for {_to_iso_date(target_yyyymmdd)} — possibly "
            f"market not yet closed or IBKR not yet finalised."
        )
        if avail:
            msg += f" Available dates in report: {', '.join(avail[:5])}"
        # If the report parsed but had zero rows of the expected element, that's
        # a parse-shape mismatch, not a missing-data error.
        if not rows:
            raise FlexQueryError(
                "parse_error",
                "Flex Query response did not contain EquitySummaryByReportDateInBase "
                "rows. Make sure 'Net Asset Value (NAV)' is enabled in the query "
                "configuration.",
            )
        raise FlexQueryError("no_data_for_date", msg)

    def _num(v) -> float:
        try:
            return float(v) if v not in (None, "", "null") else 0.0
        except ValueError:
            return 0.0

    nav = _num(_read_field(matched, "total"))
    cash = _num(_read_field(matched, "cash"))
    stock = _num(_read_field(matched, "stock"))
    # Some account configurations split stock value across stock + bond +
    # options + commodities. If we have those, total them as position_value.
    extras = sum(
        _num(_read_field(matched, f))
        for f in ("bond", "option", "commodity", "fund", "notes", "warrant", "interest")
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
    account = _read_field(matched, "accountId")
    if not account:
        for fs in xml_root.iter("FlexStatement"):
            account = fs.attrib.get("accountId", "")
            if account:
                break

    currency = _read_field(matched, "currency") or "USD"
    report_date_iso = _to_iso_date(_read_field(matched, "reportDate", "toDate") or target_yyyymmdd)

    return {
        "date": report_date_iso,
        "nav": round(nav, 2),
        "cash_balance": round(cash, 2),
        "position_value": round(position_value, 2),
        "currency": currency,
        "account": account,
    }


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
