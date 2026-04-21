# ibkr_flex.py — IBKR Flex Query client for pulling trade confirmations
#
# Two-step HTTP flow:
#   1. Request report → get a reference code
#   2. Download report XML using the reference code
#   3. Parse XML into a clean DataFrame of executions
#
# Requires: flex_token + flex_query_id in st.secrets['ibkr']

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import os
try:
    import streamlit as st
except ImportError:
    st = None


# IBKR Flex Web Service endpoints
_BASE = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService"
_SEND_URL = f"{_BASE}.SendRequest"
_GET_URL = f"{_BASE}.GetStatement"


def _get_credentials():
    """Read IBKR Flex credentials from env vars or Streamlit secrets."""
    # 1. Check environment variables (FastAPI / Railway)
    token = os.environ.get("IBKR_FLEX_TOKEN", "")
    query_id = os.environ.get("IBKR_FLEX_QUERY_ID", "")
    if token and query_id:
        return token.strip(), query_id.strip(), None

    # 2. Fallback to Streamlit secrets
    try:
        if st and hasattr(st, 'secrets'):
            ibkr = st.secrets.get("ibkr", {})
            token = ibkr.get("flex_token", "")
            query_id = ibkr.get("flex_query_id", "")
            if token and query_id:
                return str(token).strip(), str(query_id).strip(), None
    except Exception:
        pass

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

        order_time = ""
        for time_key in ("orderTime", "dateTime", "tradeTime", "executionTime", "reportDate"):
            candidate = _fmt_time(attribs.get(time_key, ""))
            if candidate and ":" in candidate:
                order_time = candidate
                break

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


def pull_ibkr_trades(consolidate=True):
    """
    Convenience function: fetch + parse + optionally consolidate in one call.

    Args:
        consolidate: If True, merge partial fills into single rows.

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
    return df, raw_debug, None
