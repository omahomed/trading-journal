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
import streamlit as st


# IBKR Flex Web Service endpoints
_BASE = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService"
_SEND_URL = f"{_BASE}.SendRequest"
_GET_URL = f"{_BASE}.GetStatement"


def _get_credentials():
    """Read IBKR Flex credentials from Streamlit secrets."""
    try:
        ibkr = st.secrets.get("ibkr", {})
        token = ibkr.get("flex_token", "")
        query_id = ibkr.get("flex_query_id", "")
        if not token or not query_id:
            return None, None, "IBKR credentials not configured. Add [ibkr] flex_token and flex_query_id to secrets."
        return str(token).strip(), str(query_id).strip(), None
    except Exception as e:
        return None, None, f"Could not read IBKR secrets: {e}"


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

        # Parse price
        price = float(attribs.get("tradePrice", 0))

        # Parse amount / commission / net_cash
        amount = abs(float(attribs.get("amount", 0)))
        commission = float(attribs.get("commission", 0))
        net_cash = float(attribs.get("netCash", 0))

        # Format trade date
        trade_date_raw = attribs.get("tradeDate", "")
        trade_date = ""
        if len(trade_date_raw) == 8:
            trade_date = f"{trade_date_raw[:4]}-{trade_date_raw[4:6]}-{trade_date_raw[6:8]}"
        else:
            trade_date = trade_date_raw

        # Format order time
        order_time_raw = attribs.get("orderTime", "")
        order_time = ""
        if len(order_time_raw) >= 6:
            order_time = f"{order_time_raw[:2]}:{order_time_raw[2:4]}:{order_time_raw[4:6]}"
        else:
            order_time = order_time_raw

        # Option fields
        put_call = attribs.get("putCall", "").strip()
        strike = attribs.get("strike", "").strip()
        expiry_raw = attribs.get("expiry", "").strip()
        expiry = ""
        if len(expiry_raw) == 8:
            expiry = f"{expiry_raw[:4]}-{expiry_raw[4:6]}-{expiry_raw[6:8]}"
        else:
            expiry = expiry_raw

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
        })

    if not trades:
        return pd.DataFrame(columns=[
            "account", "trade_date", "settle_date", "symbol", "description",
            "asset_class", "action", "quantity", "price", "amount",
            "commission", "net_cash", "currency", "order_time",
            "put_call", "strike", "expiry",
        ])

    df = pd.DataFrame(trades)
    # Sort by date + time
    df = df.sort_values(["trade_date", "order_time"], ascending=[False, False])
    return df


def pull_ibkr_trades():
    """
    Convenience function: fetch + parse in one call.

    Returns:
        (DataFrame, error_message) — DataFrame of trades on success,
        empty DataFrame + error string on failure.
    """
    xml_root, err = fetch_flex_report()
    if err:
        return pd.DataFrame(), err

    df = parse_trade_confirms(xml_root)
    return df, None
