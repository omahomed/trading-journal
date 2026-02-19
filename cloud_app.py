import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math
import numpy as np
from datetime import datetime, time, timedelta
import gspread
from google.oauth2.service_account import Credentials
import os
import shutil

# --- SAFE IMPORT FOR PLOTLY ---
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="CAN SLIM PRO (Cloud)", layout="wide", page_icon="☁️")
APP_VERSION = "18.0 (Global Scope Architecture)"

# --- GOOGLE SHEETS CONFIG ---
SPREADSHEET_NAME = "Master_Trading_Journal"
TAB_MAP = {
    'Trading_Journal_Clean.csv': 'Journal',
    'Trade_Log_Details.csv': 'Details',
    'Trade_Log_Summary.csv': 'Summary'
}
# Keys
JOURNAL_FILE = 'Trading_Journal_Clean.csv'
DETAILS_FILE = 'Trade_Log_Details.csv'
SUMMARY_FILE = 'Trade_Log_Summary.csv'

# Risk Management
RISK_START_DATE = '2025-11-14' 

# --- CUSTOM RULES ---
BUY_RULES = [
    "br1.1 Consolidation", "br1.2 Cup w Handle", "br1.3 Cup w/o Handle", "br1.4 Double Bottom",
    "br1.5 IPO Base", "br1.6 Flat Base", "br2.1 HVE", "br2.2 HVSI", "br2.3 HV1",
    "br3.1 Reclaim 21e", "br3.2 Reclaim 50s", "br3.3 Reclaim 200s", "br3.4 Reclaim 10W", 
    "br4.1 PB 21e", "br4.2 PB 50s", "br4.3 PB 10w", "br4.4 PB 200s", "br4.5 VWAP", 
    "br5.1 Undercut & Rally", "br6.1 Gapper", "br7.1 TQQQ Strategy", "br8.1 Daily STL Break", 
    "br8.2 Weekly STL Break", "br8.3 Monthly STL Break", "br9.1 21e Strategy", "ns No Setup"
]
ADD_RULES = [
    "ar1 New High after Gentle PB", "ar2 KMA Pullback", "ar3 KMA Reclaim", "ar4 JL Century",
    "ar5 Continuation Gap Up", "ar6 High Low Support", "ar7 3 Weeks Tight",
    "ar8 Adding to initial position"
]
SELL_RULES = [
    "sr1 Capital Protection", "sr2 Selling into Strength", "sr3 Portfolio Management",
    "sr4 Change of Character", "sr5 Equator Line Break", "sr6 Webby RS Rule",
    "sr7 Selling before Earnings", "sr8 TQQQ Strategy Exit", "sr9 Breakout Failure"
]
ALL_RULES = sorted(list(set(BUY_RULES + ADD_RULES + SELL_RULES)))

# --- CLOUD DATA LAYER (CACHED & ROBUST) ---
@st.cache_resource
def get_gspread_client():
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        return gspread.authorize(credentials)
    except Exception as e:
        st.error(f"❌ Cloud Auth Failed: {e}")
        return None

@st.cache_data(ttl=60) 
def load_data(file_key):
    """Loads data, caches it, and normalizes types strictly."""
    SCHEMAS = {
        JOURNAL_FILE: ['Day', 'Beg NLV', 'End NLV', 'Cash -/+', 'Daily $ Change', 'SPY', 'Nasdaq', '% Invested', 'Market_Action', 'Keywords', 'Score'],
        DETAILS_FILE: ['Trade_ID', 'Trx_ID', 'Ticker', 'Action', 'Date', 'Shares', 'Amount', 'Value', 'Rule', 'Notes', 'Realized_PL', 'Stop_Loss'],
        SUMMARY_FILE: ['Trade_ID', 'Ticker', 'Status', 'Open_Date', 'Closed_Date', 'Shares', 'Avg_Entry', 'Avg_Exit', 'Total_Cost', 'Realized_PL', 'Unrealized_PL', 'Return_Pct', 'Buy_Rule', 'Notes']
    }

    client = get_gspread_client()
    if not client: return pd.DataFrame(columns=SCHEMAS.get(file_key, []))
    
    try:
        sh = client.open(SPREADSHEET_NAME)
        worksheet = sh.worksheet(TAB_MAP[file_key])
        data = worksheet.get_all_records()
        
        if not data: return pd.DataFrame(columns=SCHEMAS.get(file_key, []))
            
        df = pd.DataFrame(data)
        
        # Cleanup Headers
        df.columns = [c.strip().replace(',', '').replace('"', '') for c in df.columns]
        
        # De-Dupe Columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Ensure Schema Columns Exist
        for col in SCHEMAS.get(file_key, []):
            if col not in df.columns: df[col] = np.nan

        # Numeric Cleaning
        cols_to_clean = ['Beg NLV', 'End NLV', 'Cash -/+', 'Daily $ Change', 'SPY', 'Nasdaq', 
                        'Shares', 'Amount', 'Value', 'Total_Cost', 'Realized_PL', 
                        'Avg_Entry', 'Avg_Exit', 'Stop_Loss']
        
        def smart_float(x):
            if pd.isna(x) or x == "": return 0.0
            try:
                s = str(x).strip().replace('$','').replace(',','').replace('%','')
                if '(' in s: s = s.replace('(', '-').replace(')', '')
                return float(s)
            except: return 0.0

        for col in cols_to_clean:
            if col in df.columns: df[col] = df[col].apply(smart_float)

        # Date Conversion
        date_cols = ['Day', 'Date', 'Open_Date', 'Closed_Date']
        for col in date_cols:
            if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')

        # String Cleanup
        str_cols = ['Trade_ID', 'Trx_ID', 'Ticker', 'Action', 'Rule', 'Status', '% Invested']
        for col in str_cols:
            if col in df.columns: df[col] = df[col].fillna('').astype(str).str.strip()
        
        if 'Total_Shares' in df.columns: df.rename(columns={'Total_Shares': 'Shares'}, inplace=True)
        if 'Nsadaq' in df.columns: df.rename(columns={'Nsadaq': 'Nasdaq'}, inplace=True)
            
        return df
    except Exception as e:
        return pd.DataFrame(columns=SCHEMAS.get(file_key, []))

def secure_save(df, file_key):
    """Writes to Cloud and clears cache."""
    client = get_gspread_client()
    if not client: return
    try:
        sh = client.open(SPREADSHEET_NAME)
        worksheet = sh.worksheet(TAB_MAP[file_key])
        
        export_df = df.copy()
        # De-dupe before saving
        export_df = export_df.loc[:, ~export_df.columns.duplicated()]
        
        date_cols = ['Day', 'Date', 'Open_Date', 'Closed_Date']
        for col in date_cols:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else "")

        export_df = export_df.fillna("")
        data_to_write = [export_df.columns.values.tolist()] + export_df.values.tolist()
        
        worksheet.clear()
        worksheet.update(data_to_write)
        load_data.clear()
    except Exception as e:
        st.error(f"Failed to save to Cloud: {e}")

# --- LOGIC HELPERS ---
def generate_trx_id(df_d, trade_id, action, date_str):
    if df_d.empty: return "B1" if action == 'BUY' else "S1"
    txs = df_d[df_d['Trade_ID'] == trade_id].copy()
    if txs.empty: return "B1" if action == 'BUY' else "S1"
    buys = txs[txs['Action'] == 'BUY'].sort_values('Date')
    if buys.empty: return "B1"
    try:
        start_date = pd.to_datetime(buys.iloc[0]['Date']).date()
        curr_date = pd.to_datetime(date_str).date()
    except: return "X1"
    if action == 'BUY':
        if curr_date == start_date:
            b_count = len([x for x in txs['Trx_ID'] if str(x).startswith('B') and 'S' not in str(x)])
            return f"B{b_count + 1}"
        else:
            a_count = len([x for x in txs['Trx_ID'] if str(x).startswith('A')])
            return f"A{a_count + 1}"
    elif action == 'SELL':
        lots = [] 
        for _, row in txs.sort_values('Date').iterrows():
            if row['Action'] == 'BUY': lots.append({'id': row.get('Trx_ID', 'B1'), 'qty': row['Shares']})
            elif row['Action'] == 'SELL':
                sold = row['Shares']
                for i in range(len(lots)-1, -1, -1):
                    if sold <= 0: break
                    take = min(lots[i]['qty'], sold); lots[i]['qty'] -= take; sold -= take
                lots = [L for L in lots if L['qty'] > 0]
        if lots: return f"S{lots[-1]['id']}"
        else: return "S1"

def update_campaign_summary(trade_id, df_d, df_s):
    try:
        txs = df_d[df_d['Trade_ID'] == trade_id].copy()
        if txs.empty: return df_d, df_s
        txs['Date'] = pd.to_datetime(txs['Date'], errors='coerce')
        txs = txs.dropna(subset=['Date'])
        txs['Sort_Date'] = txs['Date'].dt.normalize()
        txs['Type_Rank'] = txs['Action'].apply(lambda x: 0 if x == 'BUY' else 1)
        txs = txs.sort_values(['Sort_Date', 'Type_Rank', 'Date']) 
        inventory = []; total_realized_pl = 0.0
        for _, row in txs.iterrows():
            idx = row.name 
            if row['Action'] == 'BUY':
                inventory.append({'price': row['Amount'], 'shares': row['Shares']})
                if idx in df_d.index: df_d.at[idx, 'Realized_PL'] = 0.0
            elif row['Action'] == 'SELL':
                shares_to_sell = row['Shares']
                sell_price = row['Amount']
                trx_pnl = 0.0
                while shares_to_sell > 0 and inventory:
                    last_lot = inventory.pop()
                    take = min(shares_to_sell, last_lot['shares'])
                    pnl = (sell_price - last_lot['price']) * take
                    trx_pnl += pnl
                    shares_to_sell -= take
                    last_lot['shares'] -= take
                    if last_lot['shares'] > 0.0001: inventory.append(last_lot)
                if idx in df_d.index: df_d.at[idx, 'Realized_PL'] = trx_pnl
                total_realized_pl += trx_pnl
        buys = txs[txs['Action'] == 'BUY']; sells = txs[txs['Action'] == 'SELL']
        total_buy_shs = buys['Shares'].sum(); total_buy_val = buys['Value'].sum()
        avg_entry = total_buy_val / total_buy_shs if total_buy_shs > 0 else 0.0
        curr_shares = sum(item['shares'] for item in inventory)
        if abs(curr_shares) < 0.001: curr_shares = 0
        curr_cost = sum(item['shares'] * item['price'] for item in inventory)
        first_buy = buys.iloc[0] if not buys.empty else None
        last_sell = sells.iloc[-1] if not sells.empty else None
        
        # De-dupe columns before checking index
        df_s = df_s.loc[:, ~df_s.columns.duplicated()]
        
        idx = df_s[df_s['Trade_ID'] == trade_id].index
        if not idx.empty:
            i = idx[0]
            if first_buy is not None: df_s.at[i, 'Open_Date'] = first_buy['Date']
            df_s.at[i, 'Avg_Entry'] = avg_entry
            if not sells.empty: 
                tot_sold = sells['Shares'].sum()
                df_s.at[i, 'Avg_Exit'] = (sells['Value'].sum() / tot_sold) if tot_sold > 0 else 0.0
            df_s.at[i, 'Realized_PL'] = total_realized_pl
            if curr_shares < 1:
                df_s.at[i, 'Status'] = 'CLOSED'; df_s.at[i, 'Shares'] = total_buy_shs; df_s.at[i, 'Total_Cost'] = total_buy_val
                df_s.at[i, 'Return_Pct'] = (total_realized_pl / total_buy_val * 100) if total_buy_val != 0 else 0.0
                if last_sell is not None: df_s.at[i, 'Closed_Date'] = last_sell['Date']
            else:
                df_s.at[i, 'Status'] = 'OPEN'; df_s.at[i, 'Shares'] = curr_shares; df_s.at[i, 'Total_Cost'] = curr_cost
                df_s.at[i, 'Closed_Date'] = pd.NaT
        else:
            ticker = first_buy['Ticker'] if first_buy is not None else "UNKNOWN"
            open_d = first_buy['Date'] if first_buy is not None else datetime.now()
            buy_rule = first_buy['Rule'] if first_buy is not None else ""
            new_row = {
                'Trade_ID': trade_id, 'Ticker': ticker, 'Status': 'OPEN' if curr_shares > 0 else 'CLOSED',
                'Open_Date': open_d, 'Closed_Date': last_sell['Date'] if (curr_shares == 0 and last_sell is not None) else pd.NaT,
                'Shares': curr_shares if curr_shares > 0 else total_buy_shs,
                'Avg_Entry': avg_entry, 'Avg_Exit': 0.0, 'Total_Cost': curr_cost if curr_shares > 0 else total_buy_val,
                'Realized_PL': total_realized_pl, 'Unrealized_PL': 0.0, 'Return_Pct': 0.0,
                'Buy_Rule': buy_rule, 'Notes': ''
            }
            df_s = pd.concat([df_s, pd.DataFrame([new_row])], ignore_index=True)
        return df_d, df_s
    except Exception as e:
        st.error(f"Logic Error in Trade {trade_id}: {e}"); return df_d, df_s

def color_pnl(val): return 'color: #ff4b4b' if isinstance(val, (int, float)) and val < 0 else 'color: #2ca02c'
def color_neg_value(val): return 'color: #ff4b4b' if isinstance(val, (int, float)) and val < 0 else ''
def color_result(val):
    if val == 'WIN': return 'color: #2ca02c; font-weight: bold'
    elif val == 'LOSS': return 'color: #ff4b4b; font-weight: bold'
    return 'color: gray'
def color_score(val):
    try:
        v = float(val)
        if v >= 4: return 'color: #2ca02c; font-weight: bold'; 
        if v <= 2: return 'color: #ff4b4b; font-weight: bold'
    except: pass
    return ''

def calculate_open_risk(df_d, df_s, nlv):
    if df_d.empty or df_s.empty or nlv == 0: return 0.0, 0.0
    total_risk_dollars = 0.0
    if 'Status' in df_s.columns:
        open_campaigns = df_s[df_s['Status'] == 'OPEN']
        for _, campaign in open_campaigns.iterrows():
            tid = campaign['Trade_ID']; 
            try: shares_held = float(campaign.get('Shares', 0))
            except: shares_held = 0.0
            try:
                stops = df_d[df_d['Trade_ID'] == tid]['Stop_Loss']
                valid_stops = stops[stops > 0]
                if not valid_stops.empty: stop_val = float(valid_stops.iloc[-1]) 
                else: stop_val = 0.0
            except: stop_val = 0.0
            if shares_held > 0 and stop_val > 0:
                unreal = float(campaign.get('Unrealized_PL', 0.0))
                current_val = float(campaign['Total_Cost']) + unreal; current_price = current_val / shares_held
                risk = max(0, (current_price - stop_val) * shares_held)
                total_risk_dollars += risk
    return total_risk_dollars, (total_risk_dollars / nlv) * 100

def analyze_market_trend(ticker_symbol):
    try:
        tick = yf.Ticker(ticker_symbol)
        df = tick.history(period="2y")
        if df.empty: return None
        df['21EMA'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['50SMA'] = df['Close'].rolling(window=50).mean()
        df['200SMA'] = df['Close'].rolling(window=200).mean()
        df['Prev_Close'] = df['Close'].shift(1)
        def get_status_state(ma_col):
            if len(df) < 60: return "SIDEWAYS", "orange", "➡"
            state = "YELLOW"; color = "#ffcc00"; symbol = "➡"; violation_low = None; pt_active = False; pt_streak = 0
            subset = df.iloc[-60:].copy()
            for i in range(1, len(subset)):
                row = subset.iloc[i]; close = row['Close']; low = row['Low']; ma = row[ma_col]; is_up = close > row['Prev_Close']
                if state == "GREEN" or state == "POWERTREND":
                    if close < ma: state = "YELLOW"; color = "#ffcc00"; symbol = "➡"; violation_low = low; pt_active = False
                elif state == "YELLOW":
                    if violation_low is not None and low < violation_low: state = "RED"; color = "#ff3333"; symbol = "⬇"; violation_low = None
                    elif close > ma: state = "GREEN"; color = "#00cc00"; symbol = "⬆"; violation_low = None
                elif state == "RED":
                    if close > ma: state = "YELLOW"; color = "#ffcc00"; symbol = "➡"; violation_low = None
                if ma_col == '21EMA':
                    if low > ma:
                        pt_streak += 1
                        if pt_streak >= 3 and is_up: pt_active = True
                    else: pt_streak = 0; pt_active = False
                    if pt_active and state != "RED": state = "POWERTREND"; color = "#8A2BE2"; symbol = "🚀"
            return state, color, symbol
        s_stat, s_col, s_sym = get_status_state('21EMA'); m_stat, m_col, m_sym = get_status_state('50SMA'); l_stat, l_col, l_sym = get_status_state('200SMA')
        def get_streak(ma_col):
            curr_close = df['Close'].iloc[-1]; curr_ma = df[ma_col].iloc[-1]
            is_above = curr_close > curr_ma; count = 0
            for i in range(len(df)-1, -1, -1):
                c = df['Close'].iloc[i]; m = df[ma_col].iloc[i]
                if is_above:
                    if c > m: count += 1; 
                    else: break
                else:
                    if c < m: count += 1; 
                    else: break
            direction = "⬆" if is_above else "⬇"; dcol = "#00cc00" if is_above else "#ff3333"
            diff_pct = ((curr_close - curr_ma) / curr_ma) * 100
            return count, direction, dcol, diff_pct, curr_ma
        s_days, s_dir, s_dcol, s_diff, s_val = get_streak('21EMA'); m_days, m_dir, m_dcol, m_diff, m_val = get_streak('50SMA'); l_days, l_dir, l_dcol, l_diff, l_val = get_streak('200SMA')
        return {'price': df['Close'].iloc[-1], 'short': {'stat':s_stat, 'col':s_col, 'sym':s_sym, 'val':s_val, 'days':s_days, 'dir':s_dir, 'dcol':s_dcol, 'diff':s_diff}, 'med': {'stat':m_stat, 'col':m_col, 'sym':m_sym, 'val':m_val, 'days':m_days, 'dir':m_dir, 'dcol':m_dcol, 'diff':m_diff}, 'long': {'stat':l_stat, 'col':l_col, 'sym':l_sym, 'val':l_val, 'days':l_days, 'dir':l_dir, 'dcol':l_dcol, 'diff':l_diff}}
    except: return None

# --- SIDEBAR ---
st.sidebar.title("CAN SLIM SYSTEM")
st.sidebar.caption(f"v{APP_VERSION}")
if st.sidebar.button("🔄 Force Refresh Data"):
    load_data.clear()
    st.rerun()
page = st.sidebar.radio("Go to", ["Dashboard", "Daily Journal", "M Factor", "Risk Manager", "Ticker Forensics", "Period Review", "Daily Routine", "Position Sizer", "Trade Manager", "Analytics"])

# --- GLOBAL DATA LOAD ---
df_j = load_data(JOURNAL_FILE); df_d = load_data(DETAILS_FILE); df_s = load_data(SUMMARY_FILE)

# --- GLOBAL VARIABLES INITIALIZATION ---
current_equity = 100000.0 # Default
if not df_j.empty:
    df_j = df_j.sort_values('Day')
    current_equity = float(df_j['End NLV'].iloc[-1])
    
    # Pre-calculate TWR for global usage
    df_j['Adjusted_Beg'] = df_j['Beg NLV'] + df_j['Cash -/+'].fillna(0)
    df_j['Daily_Return'] = df_j.apply(lambda row: (row['End NLV'] - row['Adjusted_Beg']) / row['Adjusted_Beg'] if row['Adjusted_Beg'] != 0 else 0, axis=1)
    df_j['TWR_Curve'] = (1 + df_j['Daily_Return']).cumprod()
    df_j['LTD_Pct'] = (df_j['TWR_Curve'] - 1) * 100
    
    # Pre-calculate Daily Pct
    denom = df_j['Beg NLV'] + df_j['Cash -/+'].fillna(0.0)
    df_j['Daily_Pct'] = 0.0; valid_mask = denom != 0
    df_j.loc[valid_mask, 'Daily_Pct'] = (df_j['End NLV'] - denom) / denom * 100

# --- PAGES ---

if page == "Dashboard":
    st.header("MARKET DASHBOARD")
    if not df_j.empty:
        if 'Nsadaq' in df_j.columns: df_j.rename(columns={'Nsadaq': 'Nasdaq'}, inplace=True)
        risk_dol, risk_pct = calculate_open_risk(df_d, df_s, current_equity)
        curr_year = datetime.now().year; df_ytd = df_j[df_j['Day'].dt.year == curr_year].copy()
        ytd_val = ((1 + df_ytd['Daily_Pct']/100).cumprod().iloc[-1] - 1) * 100 if not df_ytd.empty else 0.0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net Liq Value", f"${current_equity:,.2f}", f"{df_j['Daily $ Change'].iloc[-1]:,.2f}")
        c2.metric("LTD Return", f"{df_j['LTD_Pct'].iloc[-1]:.2f}%")
        c3.metric("YTD Return", f"{ytd_val:.2f}%")
        exposure = df_j['% Invested'].iloc[-1] if '% Invested' in df_j.columns else 0
        c4.metric("Exposure", f"{exposure}")
        st.markdown("---"); r1, r2, r3, r4 = st.columns(4)
        r1.metric("Open Risk ($)", f"${risk_dol:,.2f}"); r2.metric("Open Risk (%)", f"{risk_pct:.2f}%", delta="-Target < 1.5%" if risk_pct > 1.5 else "Safe", delta_color="inverse")
        df_j['EC_8EMA'] = df_j['LTD_Pct'].ewm(span=8, adjust=False).mean()
        df_j['EC_21EMA'] = df_j['LTD_Pct'].ewm(span=21, adjust=False).mean()
        df_j['EC_50SMA'] = df_j['LTD_Pct'].rolling(window=50).mean()
        if 'Nasdaq' in df_j.columns: df_j['NDX_21EMA'] = df_j['Nasdaq'].ewm(span=21, adjust=False).mean()
        plt.style.use('bmh'); fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        if 'Nasdaq' in df_j.columns:
            ax1.fill_between(df_j['Day'], 0.97, 1.0, transform=ax1.transAxes, where=(df_j['Nasdaq']>=df_j['NDX_21EMA']), color='green', alpha=0.4, zorder=0)
            ax1.fill_between(df_j['Day'], 0.97, 1.0, transform=ax1.transAxes, where=(df_j['Nasdaq']<df_j['NDX_21EMA']), color='red', alpha=0.4, zorder=0)
        ax1b = ax1.twinx(); ax1b.set_ylim(0, 600); ax1b.set_yticks([])
        if '% Invested' in df_j.columns: 
             try: inv_float = df_j['% Invested'].str.rstrip('%').astype('float')
             except: inv_float = df_j['% Invested']
             ax1b.fill_between(df_j['Day'], inv_float, color='orange', alpha=0.2, label='% Invested')
        if 'SPY' in df_j.columns: 
            df_j['SPY_Pct'] = df_j['SPY'].pct_change().fillna(0); df_j['SPY_LTD'] = ((1 + df_j['SPY_Pct']).cumprod() - 1) * 100
            ax1.plot(df_j['Day'], df_j['SPY_LTD'], color='gray', alpha=0.4, label='SPY')
        ax1.plot(df_j['Day'], df_j['LTD_Pct'], color='darkblue', linewidth=2.5, label='Portfolio')
        ax1.plot(df_j['Day'], df_j['EC_8EMA'], color='purple', linewidth=1.2, label='8 EMA'); ax1.plot(df_j['Day'], df_j['EC_21EMA'], color='green', linewidth=1.2, label='21 EMA'); ax1.plot(df_j['Day'], df_j['EC_50SMA'], color='red', linewidth=1.2, label='50 SMA')
        ax1.legend(loc='upper left'); ax1.set_title("Equity Curve")
        colors = ['green' if x >= 0 else 'red' for x in df_j['Daily $ Change']]
        ax2.bar(df_j['Day'], df_j['Daily $ Change'], color=colors)
        st.pyplot(fig)
    else: st.info("Connect Google Sheets to see Dashboard.")

elif page == "M Factor":
    st.header("MARKET HEALTH (M FACTOR)")
    if st.button("Refresh Market Data"): st.cache_data.clear()
    nasdaq_data = analyze_market_trend("^IXIC"); spy_data = analyze_market_trend("SPY")
    if nasdaq_data and spy_data:
        s_stat = nasdaq_data['short']['stat']; m_stat = nasdaq_data['med']['stat']
        if m_stat == "RED": win_stat, win_col = "CLOSED", "#ff3333"
        elif s_stat == "POWERTREND": win_stat, win_col = "POWER TREND", "#8A2BE2"
        elif s_stat == "GREEN": win_stat, win_col = "OPEN", "#00cc00"
        else: win_stat, win_col = "NEUTRAL", "orange"
        st.markdown(f"""<div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;"><h2 style="margin:0; color: #333;">MARKET WINDOW</h2><h1 style="margin:0; font-size: 3.5em; color: {win_col};">{win_stat}</h1></div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        def display_trend_box(title, data):
            st.subheader(f"{title} (${data['price']:.2f})")
            def arrow(sym, col, val, label, st_txt): st.markdown(f"**{label}**: <span style='color:{col}; font-size:1.5em; font-weight:bold;'>{st_txt} {sym}</span> ({val:.2f})", unsafe_allow_html=True)
            arrow(data['short']['sym'], data['short']['col'], data['short']['val'], "Short (21e)", data['short']['stat']); arrow(data['med']['sym'], data['med']['col'], data['med']['val'], "Med (50s)", data['med']['stat']); arrow(data['long']['sym'], data['long']['col'], data['long']['val'], "Long (200s)", data['long']['stat'])
            st.markdown("---"); m1, m2, m3 = st.columns(3)
            def trend_metric(col, item, label): 
                diff_col = "#00cc00" if item['diff'] >= 0 else "#ff3333"
                col.markdown(f"""<div style="background-color: #fafafa; padding: 10px; border-radius: 5px; border: 1px solid #eee; margin-bottom: 5px;"><div style="font-size: 0.8rem; color: #666;">{label}</div><div style="font-size: 1.2rem; font-weight: bold; color: {item['dcol']};">{item['dir']} {item['days']}d</div><div style="font-size: 1.0rem; color: {diff_col}; font-weight: 500;">{item['diff']:+.2f}%</div></div>""", unsafe_allow_html=True)
            with m1: trend_metric(st, data['short'], "21e Trend")
            with m2: trend_metric(st, data['med'], "50s Trend")
            with m3: trend_metric(st, data['long'], "200s Trend")
        with c1: display_trend_box("NASDAQ COMPOSITE", nasdaq_data)
        with c2: display_trend_box("S&P 500", spy_data)
    else: st.error("Could not fetch market data.")

elif page == "Risk Manager":
    st.header("HIGHER LOW PROTOCOL (Risk Manager)")
    if not df_j.empty:
        start_dt = pd.to_datetime(RISK_START_DATE)
        df_risk = df_j[df_j['Day'] >= start_dt].copy()
        if df_risk.empty: st.warning(f"No journal entries found on or after Risk Start Date ({RISK_START_DATE}).")
        else:
            curr_nlv = current_equity
            lookback_window = df_risk.tail(50)
            low_nlv = lookback_window['End NLV'].min()
            hard_floor = low_nlv * 1.01 
            risk_budget = curr_nlv - hard_floor
            current_open_risk, risk_pct = calculate_open_risk(df_d, df_s, curr_nlv)
            st.markdown(f"### Risk Budget: **${risk_budget:,.2f}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current NLV", f"${curr_nlv:,.2f}")
            col2.metric("Hard Floor (Low + 1%)", f"${hard_floor:,.2f}", help="Based on 50-day low since reset")
            col3.metric("Current Open Risk", f"${current_open_risk:,.2f}", 
                        delta=f"${risk_budget - current_open_risk:,.2f} Remaining" if risk_budget > current_open_risk else f"OVER BUDGET by ${current_open_risk - risk_budget:,.2f}",
                        delta_color="normal" if risk_budget > current_open_risk else "inverse")
            if risk_budget > 0:
                utilization = min(1.0, current_open_risk / risk_budget)
                st.write(f"Risk Utilization: {utilization:.1%}")
                st.progress(utilization)
                if utilization > 1.0: st.error("🚨 CRITICAL: Risk budget exceeded!")
            else: st.error("⛔ STOP TRADING: You are below your higher-low floor.")
            st.markdown("---")
            st.subheader(f"Equity Structure (Since {RISK_START_DATE})")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(lookback_window['Day'], lookback_window['End NLV'], label="Equity Curve", color="#1f77b4", linewidth=2)
            ax.axhline(hard_floor, color='red', linestyle='--', linewidth=1.5, label="Hard Floor")
            ax.fill_between(lookback_window['Day'], hard_floor, lookback_window['End NLV'], where=(lookback_window['End NLV'] >= hard_floor), color='green', alpha=0.1)
            ax.fill_between(lookback_window['Day'], hard_floor, lookback_window['End NLV'], where=(lookback_window['End NLV'] < hard_floor), color='red', alpha=0.1)
            ax.legend(); ax.grid(True, alpha=0.3); st.pyplot(fig)
    else: st.info("No journal data found.")

elif page == "Daily Journal":
    st.header("DAILY TRADING JOURNAL")
    tab_view, tab_manage = st.tabs(["View Logs", "Manage Logs"])
    with tab_view:
        if not df_j.empty:
            view_opt = st.radio("Filter View", ["Current Week", "By Month", "All History"], horizontal=True)
            df_view = df_j.copy()
            if view_opt == "Current Week":
                today = datetime.now().date(); start_week = today - timedelta(days=today.weekday())
                df_view = df_view[df_view['Day'].dt.date >= start_week]
            elif view_opt == "By Month":
                df_view['Month_Str'] = df_view['Day'].dt.strftime('%B %Y')
                months = sorted(df_view['Month_Str'].unique().tolist(), key=lambda x: datetime.strptime(x, '%B %Y'), reverse=True)
                if months:
                    sel_month = st.selectbox("Select Month", months)
                    df_view = df_view[df_view['Month_Str'] == sel_month]
            st.dataframe(df_view.sort_values('Day', ascending=False)[['Day', 'Daily_Pct', 'Market_Action', 'Keywords', 'Score']].style.format({'Day': '{:%m/%d/%y}', 'Daily_Pct': '{:+.2f}%', 'Score': '{:.0f}'}).applymap(color_pnl, subset=['Daily_Pct']).applymap(color_score, subset=['Score']), hide_index=True)
        else: st.info("No journal entries found.")
    with tab_manage:
        st.subheader("Delete Incorrect Entries")
        if not df_j.empty:
            df_j_del = df_j.sort_values('Day', ascending=False).copy()
            options = [f"{row['Day'].strftime('%Y-%m-%d')} | End NLV: ${row['End NLV']:,.2f}" for i, row in df_j_del.iterrows()]
            sel_del = st.selectbox("Select Log to Delete", options)
            if st.button("DELETE ENTRY"):
                date_to_del = sel_del.split("|")[0].strip(); df_j = df_j[df_j['Day'].dt.strftime('%Y-%m-%d') != date_to_del]
                secure_save(df_j, JOURNAL_FILE); st.success(f"Deleted entry for {date_to_del}"); st.rerun()

elif page == "Ticker Forensics":
    st.header("TICKER FORENSICS")
    if not df_s.empty and 'Status' in df_s.columns:
        closed = df_s[df_s['Status']=='CLOSED'].copy()
        if not closed.empty:
            st.subheader("1. Ticker Leaderboard (Top 20 Movers)")
            ticker_stats = closed.groupby('Ticker').agg(Total_PL=('Realized_PL', 'sum'), Trade_Count=('Trade_ID', 'count'), Win_Rate=('Realized_PL', lambda x: (x>0).mean())).sort_values('Total_PL', ascending=True)
            top_movers = pd.concat([ticker_stats.head(10), ticker_stats.tail(10)])
            top_movers = top_movers[~top_movers.index.duplicated()]
            fig, ax = plt.subplots(figsize=(10, 8)); colors = ['#2ca02c' if x >= 0 else '#ff4b4b' for x in top_movers['Total_PL']]
            top_movers['Total_PL'].plot(kind='barh', ax=ax, color=colors); ax.axvline(0, color='black', linewidth=1); ax.set_title("Total P&L by Ticker"); ax.set_xlabel("P&L ($)"); st.pyplot(fig)
            st.dataframe(ticker_stats.sort_values('Total_PL', ascending=False).style.format({'Total_PL': '${:,.2f}', 'Win_Rate': '{:.1%}'}).applymap(color_pnl, subset=['Total_PL']))
            st.markdown("---"); st.subheader("2. Trade Distribution")
            selected_ticker = st.selectbox("Filter Distribution", ["All"] + sorted(closed['Ticker'].unique().tolist()))
            dist_data = closed if selected_ticker == "All" else closed[closed['Ticker'] == selected_ticker]
            if not dist_data.empty:
                fig2, ax2 = plt.subplots(figsize=(10, 5)); n, bins, patches = ax2.hist(dist_data['Realized_PL'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                ax2.axvline(0, color='black', linewidth=2, linestyle='--'); st.pyplot(fig2)
                c1, c2 = st.columns(2); c1.metric("Average Trade", f"${dist_data['Realized_PL'].mean():,.2f}"); c2.metric("Total P&L", f"${dist_data['Realized_PL'].sum():,.2f}")
        else: st.info("No closed trades to analyze.")
    else: st.info("Summary data not found.")

elif page == "Period Review":
    st.header("PERIOD REVIEW")
    view_mode = st.radio("Select Period", ["Weekly", "Monthly"], horizontal=True)
    if not df_j.empty:
        if 'SPY' in df_j.columns: df_j['SPY_Pct'] = df_j['SPY'].pct_change().fillna(0); df_j['SPY_LTD'] = ((1 + df_j['SPY_Pct']).cumprod() - 1) * 100
        else: df_j['SPY_LTD'] = 0.0
        if 'Nasdaq' in df_j.columns: df_j['NDX_Pct'] = df_j['Nasdaq'].pct_change().fillna(0); df_j['NDX_LTD'] = ((1 + df_j['NDX_Pct']).cumprod() - 1) * 100
        else: df_j['NDX_LTD'] = 0.0
        resample_code = 'W-FRI' if view_mode == "Weekly" else 'ME'
        agg_dict = {'Beg NLV': 'first', 'End NLV': 'last', 'Daily $ Change': 'sum', 'Cash -/+': 'sum', 'LTD_Pct': 'last'}
        if 'SPY_LTD' in df_j.columns: agg_dict['SPY_LTD'] = 'last'
        if 'NDX_LTD' in df_j.columns: agg_dict['NDX_LTD'] = 'last'
        df_p = df_j.set_index('Day').resample(resample_code).agg(agg_dict).dropna()
        if not df_p.empty:
            df_p['Period P&L ($)'] = df_p['End NLV'] - (df_p['Beg NLV'] + df_p['Cash -/+']); denom = df_p['Beg NLV'] + df_p['Cash -/+']; df_p['Period Return %'] = 0.0; df_p.loc[denom != 0, 'Period Return %'] = (df_p['Period P&L ($)'] / denom) * 100
            df_p['MA_10'] = df_p['LTD_Pct'].rolling(window=10).mean(); df_table = df_p.sort_index(ascending=False)
            latest = df_table.iloc[0]; c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Latest {view_mode} P&L", f"${latest['Period P&L ($)']:,.2f}", delta_color="normal"); c2.metric("Period Return %", f"{latest['Period Return %']:.2f}%", delta_color="normal")
            p_end = latest.name; p_start = p_end - timedelta(days=6) if view_mode == "Weekly" else p_end.replace(day=1)
            count = 0; wr = 0.0
            if not df_s.empty and 'Closed_Date' in df_s.columns:
                mask = (df_s['Status'] == 'CLOSED') & (df_s['Closed_Date'] >= p_start) & (df_s['Closed_Date'] <= p_end + timedelta(days=1))
                trades = df_s[mask]; count = len(trades); wins = len(trades[trades['Realized_PL'] > 0]); wr = wins/count if count > 0 else 0.0
            c3.metric("Trades Closed", count); c4.metric("Win Rate", f"{wr:.1%}")
            st.markdown("---"); st.subheader("Performance Curve")
            fig, ax = plt.subplots(figsize=(12, 6))
            last_port = df_p['LTD_Pct'].iloc[-1]; last_spy = df_p['SPY_LTD'].iloc[-1]; last_ndx = df_p['NDX_LTD'].iloc[-1]
            ax.plot(df_p.index, df_p['LTD_Pct'], label=f"Portfolio ({last_port:+.1f}%)", color="#1f77b4", linewidth=2.5)
            if 'SPY_LTD' in df_p.columns: ax.plot(df_p.index, df_p['SPY_LTD'], label=f"SPY ({last_spy:+.1f}%)", color="gray", alpha=0.5)
            if 'NDX_LTD' in df_p.columns: ax.plot(df_p.index, df_p['NDX_LTD'], label=f"NDX ({last_ndx:+.1f}%)", color="orange", alpha=0.5)
            if 'MA_10' in df_p.columns: ax.plot(df_p.index, df_p['MA_10'], label="10 MA", color="purple", linestyle="--")
            ax.legend(); st.pyplot(fig)
            st.subheader("Financial Statement")
            st.dataframe(df_table[['Beg NLV', 'Cash -/+', 'End NLV', 'Period P&L ($)', 'Period Return %']].style.format({'Beg NLV': '${:,.2f}', 'Cash -/+': '${:,.2f}', 'End NLV': '${:,.2f}', 'Period P&L ($)': '${:,.2f}', 'Period Return %': '{:+.2f}%'}).applymap(color_pnl, subset=['Period P&L ($)', 'Period Return %']))
        else: st.info("Not enough data for period review.")
    else: st.info("Journal empty.")

elif page == "Daily Routine":
    st.header("END OF DAY LOG")
    tab1, tab2, tab3 = st.tabs(["Log Today", "View Full Log", "Manage Logs"])
    with tab1:
        with st.form("daily_form", clear_on_submit=True):
            c1, c2 = st.columns(2); log_date = c1.date_input("Date", datetime.now()); nlv = c2.number_input("Closing NLV", min_value=0.0, step=100.0)
            c3, c4 = st.columns(2); holdings = c3.number_input("Total Holdings Value", min_value=0.0, step=100.0); cash_flow = c4.number_input("Cash Added/Removed", value=0.0, step=100.0)
            st.markdown("### Qualitative"); mkt_note = st.text_area("Market Action Notes"); kw = st.text_input("Keywords"); score = st.slider("Discipline Score", 1, 5, 3)
            if st.form_submit_button("Submit Log"):
                if nlv > 0:
                    try: spy_val = yf.Ticker("SPY").history(period="1d")['Close'].iloc[-1]; ndx_val = yf.Ticker("^IXIC").history(period="1d")['Close'].iloc[-1]
                    except: spy_val, ndx_val = 0, 0
                    prev_nlv = nlv
                    if not df_j.empty: prev_nlv = df_j.sort_values('Day')['End NLV'].iloc[-1]
                    daily_dol = nlv - (prev_nlv + cash_flow); invested_pct = (holdings / nlv) if nlv != 0 else 0
                    new_row = {'Day': log_date, 'Beg NLV': prev_nlv, 'End NLV': nlv, 'Cash -/+': cash_flow, 'Daily $ Change': daily_dol, 'SPY': spy_val, 'Nasdaq': ndx_val, '% Invested': f"{invested_pct:.1%}", 'Market_Action': mkt_note, 'Keywords': kw, 'Score': score}
                    df_j = pd.concat([df_j, pd.DataFrame([new_row])], ignore_index=True)
                    secure_save(df_j, JOURNAL_FILE); st.success("Logged successfully!")
                else: st.error("NLV must be greater than 0")
    with tab2:
        if not df_j.empty: st.dataframe(df_j.sort_values('Day', ascending=False))
    with tab3:
        if not df_j.empty:
            opts = [f"{r['Day'].strftime('%Y-%m-%d')} | NLV: ${r['End NLV']:,.2f}" for i, r in df_j.iterrows()]
            sel = st.selectbox("Select Log to Delete", opts)
            if st.button("DELETE ENTRY"):
                d_str = sel.split("|")[0].strip(); df_j = df_j[df_j['Day'].dt.strftime('%Y-%m-%d') != d_str]
                secure_save(df_j, JOURNAL_FILE); st.success("Deleted."); st.rerun()

elif page == "Position Sizer":
    st.header("POSITION SIZING CALCULATOR")
    equity = current_equity
    size_map = {"Shotgun (2.5%)": 2.5, "Half (5%)": 5.0, "Full (10%)": 10.0, "Full+1 (15%)": 15.0, "Full+2 (20%)": 20.0, "Max (25%)": 25.0, "30%": 30.0, "35%": 35.0, "40%": 40.0, "45%": 45.0, "50%": 50.0}

    tab_new, tab_add = st.tabs(["🆕 New Position", "➕ Add to Position (Pyramid)"])
    
    with tab_new:
        st.caption("Standard Sizing for Initial Entry")
        colA, colB = st.columns(2)
        with colA: acct_val = st.number_input("Account Equity ($)", value=equity, step=1000.0, key="np_eq"); ticker = st.text_input("Ticker Symbol", key="np_tk").upper(); entry = st.number_input("Entry Price ($)", min_value=0.01, step=0.1, key="np_ep")
        with colB:
            stop_mode = st.radio("Stop Input Mode", ["Manual Price", "Percentage (%)"], horizontal=True, key="np_mode")
            if stop_mode == "Percentage (%)":
                stop_pct_in = st.number_input("Stop Loss %", value=8.0, step=0.5, key="np_pct")
                stop_val = entry * (1 - (stop_pct_in/100))
                st.info(f"Calculated Stop: ${stop_val:.2f}"); stop_note = f"{stop_pct_in}% Trailing"
            else:
                stop_val = st.number_input("Stop Price ($)", min_value=0.0, max_value=entry if entry > 0 else 10000.0, key="np_sv"); stop_note = "Tech Level"
        st.markdown("---"); c1, c2 = st.columns(2)
        with c1: risk_pct = st.slider("Risk % of Capital", 0.25, 1.50, 0.75, 0.05, key="np_rp")
        with c2:
            size_mode = st.select_slider("Position Size Scale", options=list(size_map.keys()), value="Half (5%)", key="np_sm")
        max_pos_pct = size_map[size_mode]
        if st.button("Calculate Trade", key="np_btn"):
            if entry > 0 and stop_val > 0 and stop_val < entry:
                risk_share = entry - stop_val; risk_budget = acct_val * (risk_pct/100); shares_risk = math.floor(risk_budget / risk_share); shares_cap = math.floor((acct_val * (max_pos_pct/100)) / entry); rec_shares = min(shares_risk, shares_cap); cost = rec_shares * entry; weight = (cost / acct_val) * 100; risk_dol = rec_shares * risk_share; stop_pct = (risk_share/entry)*100
                st.markdown("### 🎫 TRADE TICKET"); t1, t2, t3 = st.columns(3); t1.metric("BUY SHARES", rec_shares); t2.metric("TOTAL COST", f"${cost:,.0f}"); t3.metric("WEIGHT", f"{weight:.1f}%")
                st.caption(f"Risk: ${risk_dol:.0f} ({ (risk_dol/acct_val)*100 :.2f}%) | Stop Width: {stop_pct:.2f}%")
                if stop_pct > 8.0: st.error("⚠️ STOP LOSS > 8% Violation!")
                if rec_shares == shares_cap: st.info(f"ℹ️ Capped by Max Size ({max_pos_pct}%)")
            else: st.error("Check your prices.")

    with tab_add:
        st.caption("Calculate shares to add and new stop to protect total equity.")
        if 'Status' in df_s.columns and not df_s.empty: open_positions = df_s[df_s['Status'] == 'OPEN'].sort_values('Ticker')
        else: open_positions = pd.DataFrame()
        if not open_positions.empty:
            sel_pos = st.selectbox("Select Holding", options=open_positions['Ticker'].unique().tolist())
            row = open_positions[open_positions['Ticker'] == sel_pos].iloc[0]
            try: live_price = yf.Ticker(row['Ticker']).history(period="1d")['Close'].iloc[-1]
            except: live_price = row.get('Avg_Entry', 100)
            c1, c2 = st.columns(2)
            curr_price = c1.number_input("Current Price ($)", min_value=0.01, value=float(live_price), step=0.1, key=f"add_cp_{row['Ticker']}")
            acct_val_add = c2.number_input("Account Equity ($)", value=equity, disabled=True, key="add_av")
            safe_shares = int(float(row['Shares'])) if pd.notnull(row['Shares']) else 0
            safe_avg = float(row['Avg_Entry']) if pd.notnull(row['Avg_Entry']) else 0.0
            safe_cost = float(row['Total_Cost']) if pd.notnull(row['Total_Cost']) else 0.0
            safe_weight = (safe_cost/equity)*100 if equity != 0 else 0.0
            st.markdown(f"**Current Status:** {safe_shares} shares @ ${safe_avg:.2f} ({safe_weight:.1f}% Weight)")
            st.markdown("---"); c3, c4 = st.columns(2)
            target_mode_add = c3.select_slider("Target Total Position Size", options=list(size_map.keys()), value="Full (10%)", key="add_ts_mode")
            target_size_pct = size_map[target_mode_add]
            max_risk_pct = c4.slider("Max Total Risk % (Capital)", 0.25, 1.25, 0.75, 0.05, key="add_mr")
            if st.button("Calculate Add-On", key="add_btn"):
                target_value = acct_val_add * (target_size_pct / 100)
                current_value = safe_shares * curr_price 
                value_to_add = target_value - current_value
                if value_to_add <= 0: st.error(f"You are already over the target weight! (Current: ${current_value:,.0f} vs Target: ${target_value:,.0f})")
                else:
                    shares_to_add = math.floor(value_to_add / curr_price)
                    total_shares = safe_shares + shares_to_add
                    new_avg_cost = ((safe_shares * safe_avg) + (shares_to_add * curr_price)) / total_shares
                    cost_of_add = shares_to_add * curr_price
                    allowed_risk_dollars = acct_val_add * (max_risk_pct / 100)
                    required_cushion_per_share = allowed_risk_dollars / total_shares
                    new_stop_price = curr_price - required_cushion_per_share
                    stop_dist_pct = (required_cushion_per_share / curr_price) * 100
                    st.markdown("### ➕ PYRAMID TICKET"); k1, k2, k3, k4 = st.columns(4)
                    k1.metric("ADD SHARES", f"+{shares_to_add}"); k2.metric("EST. COST", f"${cost_of_add:,.2f}"); k3.metric("NEW TOTAL", f"{int(total_shares)}"); k4.metric("AVG COST (Est)", f"${new_avg_cost:.2f}")
                    st.markdown("### 🛡️ RISK MANAGEMENT"); r1, r2 = st.columns(2)
                    r1.metric("SUGGESTED NEW STOP", f"${new_stop_price:.2f}", delta=f"-{stop_dist_pct:.2f}% from Current")
                    r2.metric("TOTAL RISK", f"${allowed_risk_dollars:,.0f}", f"{max_risk_pct}% of Equity")
                    if new_stop_price > curr_price: st.error("❌ MATHEMATICALLY IMPOSSIBLE: You are risking too much!")
                    elif new_stop_price > safe_avg: st.success("✅ PROFIT LOCK: This stop is above your original entry.")
                    elif stop_dist_pct < 3.0: st.warning("⚠️ TIGHT STOP: Stop is < 3% away.")
        else: st.info("No Open Positions found in Summary file.")

elif page == "Trade Manager":
    st.header("TRADE MANAGER")
    if not df_s.empty:
        valid_sum_cols = ['Trade_ID', 'Ticker', 'Status', 'Open_Date', 'Shares', 'Avg_Entry', 'Total_Cost', 'Unrealized_PL', 'Return_Pct', 'Buy_Rule']
        valid_sum_cols = [c for c in valid_sum_cols if c in df_s.columns]
    else: valid_sum_cols = []

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Log Buy", "Log Sell", "Update Prices", "Edit Transaction", "Database Health", "Delete Trade", "Active Campaign Summary", "Active Campaign Detailed", "Detailed Trade Log", "All Campaigns"])
    
    # --- TAB 1: LOG BUY (LIVE CALCULATOR - NO FORM) ---
    with tab1:
        st.caption("Live Entry Calculator")
        c_top1, c_top2 = st.columns(2)
        trade_type = c_top1.radio("Action Type", ["Start New Campaign", "Scale In (Add to Existing)"], horizontal=True)
        b_date = c_top2.date_input("Date", datetime.now(), key="b_date_input")
        b_time = c_top2.time_input("Time", datetime.now().time(), step=60, key="b_time_input")
        st.markdown("---")
        c1, c2 = st.columns(2)
        if trade_type == "Start New Campaign":
            b_tick = c1.text_input("Ticker Symbol").upper()
            default_id = "2025-001"
            if not df_s.empty:
                try: 
                    last_id = str(df_s['Trade_ID'].iloc[-1])
                    num = int(last_id.split('-')[1]) + 1
                    default_id = f"2025-{num:03d}"
                except: pass
            b_id = c2.text_input("Trade ID", value=default_id)
            b_rule = st.selectbox("Buy Rule", BUY_RULES)
        else:
            # Scale In Logic
            if 'Status' in df_s.columns and not df_s.empty: open_opts = df_s[df_s['Status']=='OPEN'].copy()
            else: open_opts = pd.DataFrame()
            b_tick, b_id = "", ""
            if not open_opts.empty:
                open_opts = open_opts.sort_values('Ticker')
                opts = ["Select..."] + [f"{r['Ticker']} | {r['Trade_ID']}" for _, r in open_opts.iterrows()]
                sel_camp = c1.selectbox("Select Existing Campaign", opts)
                if sel_camp and sel_camp != "Select...":
                    b_tick, b_id = sel_camp.split(" | ")
                    curr_row = open_opts[open_opts['Trade_ID']==b_id].iloc[0]
                    c2.info(f"Holding: {int(float(curr_row['Shares']))} shs @ ${float(curr_row['Avg_Entry']):.2f}")
            else:
                c1.warning("No Open Campaigns to add to.")
            b_rule = st.selectbox("Add Rule", ADD_RULES)

        c3, c4 = st.columns(2)
        b_shs = c3.number_input("Shares", min_value=1, step=1, key="b_shs_live")
        b_px = c4.number_input("Entry Price ($)", min_value=0.0, step=0.1, key="b_px_live")
        
        st.markdown("#### 🛡️ Risk Management")
        c_stop1, c_stop2 = st.columns(2)
        with c_stop1:
            stop_mode = st.radio("Stop Loss Mode", ["Price Level ($)", "Percentage (%)"], horizontal=True)
        with c_stop2:
            if stop_mode == "Percentage (%)":
                sl_pct = st.number_input("Stop Loss %", value=8.0, step=0.5, format="%.1f")
                if b_px > 0:
                    b_stop = b_px * (1 - (sl_pct/100))
                    st.metric("Calculated Stop Price", f"${b_stop:.2f}", delta=f"-{sl_pct}%")
                else:
                    b_stop = 0.0
                    st.caption("Enter Entry Price to see Stop")
            else:
                b_stop = st.number_input("Stop Price ($)", min_value=0.0, step=0.1, value=float(b_px*0.92) if b_px>0 else 0.0)
                if b_px > 0 and b_stop > 0:
                    actual_pct = ((b_px - b_stop) / b_px) * 100
                    st.caption(f"Implied Risk: {actual_pct:.2f}%")

        st.markdown("---")
        c_note1, c_note2 = st.columns(2)
        b_note = c_note1.text_input("Trade Notes")
        b_trx = c_note2.text_input("Manual Trx ID (Optional)")
        
        if st.button("LOG BUY ORDER", type="primary", use_container_width=True):
            if b_tick and b_id and b_shs > 0 and b_px > 0:
                ts = datetime.combine(b_date, b_time).strftime("%Y-%m-%d %H:%M")
                cost = b_shs * b_px
                if not b_trx: b_trx = generate_trx_id(df_d, b_id, 'BUY', ts)
                new_d = {'Trade_ID': b_id, 'Trx_ID': b_trx, 'Ticker': b_tick, 'Action': 'BUY', 'Date': ts, 'Shares': b_shs, 'Amount': b_px, 'Value': cost, 'Rule': b_rule, 'Notes': b_note, 'Realized_PL': 0, 'Stop_Loss': b_stop}
                df_d = pd.concat([df_d, pd.DataFrame([new_d])], ignore_index=True)
                if trade_type == "Start New Campaign":
                    new_s = {'Trade_ID': b_id, 'Ticker': b_tick, 'Status': 'OPEN', 'Open_Date': ts, 'Shares': 0, 'Avg_Entry': 0, 'Total_Cost': 0, 'Realized_PL': 0, 'Unrealized_PL': 0, 'Buy_Rule': b_rule, 'Notes': b_note}
                    df_s = pd.concat([df_s, pd.DataFrame([new_s])], ignore_index=True)
                secure_save(df_d, DETAILS_FILE)
                df_d, df_s = update_campaign_summary(b_id, df_d, df_s)
                secure_save(df_d, DETAILS_FILE); secure_save(df_s, SUMMARY_FILE)
                st.success(f"✅ EXECUTED: Bought {b_shs} {b_tick} @ ${b_px}"); st.balloons(); st.rerun()
            else: st.error("⚠️ Missing Data: Ensure Ticker, ID, Shares, and Price are filled.")

    with tab2:
        if 'Status' in df_s.columns and not df_s.empty: open_opts = df_s[df_s['Status']=='OPEN'].copy()
        else: open_opts = pd.DataFrame()
        if not open_opts.empty:
             open_opts = open_opts.sort_values('Ticker')
             s_opts = [f"{r['Ticker']} | {r['Trade_ID']}" for _, r in open_opts.iterrows()]
             sel_sell = st.selectbox("Select Trade to Sell", s_opts)
             if sel_sell:
                 s_tick, s_id = sel_sell.split(" | ")
                 row = open_opts[open_opts['Trade_ID']==s_id].iloc[0]
                 safe_shares_held = int(float(row['Shares']))
                 st.info(f"Selling {s_tick} (Own {safe_shares_held} shs)")
                 c1, c2 = st.columns(2)
                 s_date = c1.date_input("Date", datetime.now(), key='s_date')
                 s_time = c2.time_input("Time", datetime.now().time(), step=60, key='s_time')
                 s_shs = st.number_input("Shares", min_value=1, max_value=safe_shares_held, step=1)
                 s_px = st.number_input("Price", min_value=0.0, step=0.1)
                 s_rule = st.selectbox("Sell Rule", SELL_RULES)
                 s_trx = st.text_input("Manual Trx ID (Optional)", key='s_trx')
                 if st.button("LOG SELL ORDER", type="primary"):
                    ts = datetime.combine(s_date, s_time).strftime("%Y-%m-%d %H:%M")
                    proc = s_shs * s_px
                    if not s_trx: s_trx = generate_trx_id(df_d, s_id, 'SELL', ts)
                    new_d = {'Trade_ID':s_id, 'Trx_ID': s_trx, 'Ticker':s_tick, 'Action':'SELL', 'Date':ts, 'Shares':s_shs, 'Amount':s_px, 'Value':proc, 'Rule':s_rule, 'Realized_PL': 0}
                    df_d = pd.concat([df_d, pd.DataFrame([new_d])], ignore_index=True)
                    secure_save(df_d, DETAILS_FILE)
                    df_d, df_s = update_campaign_summary(s_id, df_d, df_s)
                    secure_save(df_d, DETAILS_FILE); secure_save(df_s, SUMMARY_FILE)
                    st.success(f"Sold. Transaction ID: {s_trx}"); st.rerun()
        else: st.info("No positions to sell.")

    with tab3:
        if st.button("REFRESH PRICES"):
            if 'Status' in df_s.columns:
                open_rows = df_s[df_s['Status']=='OPEN']
                if not open_rows.empty:
                    p = st.progress(0); n=0
                    for i, r in open_rows.iterrows():
                        try:
                            tk = r['Ticker'] if r['Ticker']!='COMP' else '^IXIC'; curr = yf.Ticker(tk).history(period='1d')['Close'].iloc[-1]
                            mkt = r['Shares'] * curr; unreal = mkt - r['Total_Cost']; df_s.at[i, 'Unrealized_PL'] = unreal; df_s.at[i, 'Return_Pct'] = (unreal/r['Total_Cost'])*100 if r['Total_Cost'] else 0
                        except: pass
                        n+=1; p.progress(n/len(open_rows))
                    secure_save(df_s, SUMMARY_FILE); st.success("Updated!"); st.rerun()
                else: st.info("No open positions.")

    with tab4:
        if not df_d.empty and 'Trade_ID' in df_d.columns:
            all_ids = [str(x) for x in df_d['Trade_ID'].unique()]
            edit_id = st.selectbox("Select Trade ID to Edit", all_ids, format_func=lambda x: f"{x} | {df_d[df_d['Trade_ID']==x]['Ticker'].iloc[0]}")
            if edit_id:
                txs = df_d[df_d['Trade_ID'].astype(str) == edit_id].reset_index()
                if not txs.empty:
                    search_term = st.text_input("Filter Transactions", "")
                    if search_term: txs = txs[txs.astype(str).apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)]
                    if not txs.empty:
                        tx_options = [f"{row.get('Trx_ID','')} | {row['Date']} | {row['Action']} {row['Shares']} @ {row['Amount']}" for idx, row in txs.iterrows()]
                        selected_tx_str = st.selectbox("Select Transaction Line", tx_options)
                        if selected_tx_str:
                            sel_idx = tx_options.index(selected_tx_str); row_idx = int(txs.iloc[sel_idx]['index']); current_row = df_d.loc[row_idx]
                            cA, cB = st.columns(2)
                            with cA:
                                with st.form("edit_form"):
                                    c1, c2 = st.columns(2)
                                    with c1: 
                                        try: 
                                            dt_obj = pd.to_datetime(current_row['Date'])
                                            if pd.isna(dt_obj): dt_obj = datetime.now()
                                        except: dt_obj = datetime.now()
                                        e_date = st.date_input("Date", dt_obj); e_time = st.time_input("Time", dt_obj.time(), step=60)
                                    with c2: 
                                        curr_rule = current_row['Rule']
                                        e_rule = st.selectbox("Rule", ALL_RULES, index=ALL_RULES.index(curr_rule) if curr_rule in ALL_RULES else 0)
                                    e_trx = st.text_input("Trx ID", value=str(current_row.get('Trx_ID', '')))
                                    c3, c4 = st.columns(2); sl_val = float(current_row['Stop_Loss']) if pd.notna(current_row.get('Stop_Loss')) else 0.0
                                    e_stop = st.number_input("Stop Loss", value=sl_val, min_value=0.0) 
                                    e_note = st.text_input("Notes", str(current_row.get('Notes', '')))
                                    st.markdown("---"); c5, c6 = st.columns(2)
                                    e_shs = st.number_input("Shares", value=int(current_row['Shares']), step=1)
                                    e_amt = st.number_input("Price ($)", value=float(current_row['Amount']), step=0.01)
                                    if st.form_submit_button("Save Changes"):
                                        new_ts = datetime.combine(e_date, e_time).strftime("%Y-%m-%d %H:%M")
                                        df_d.at[row_idx, 'Date'] = new_ts; df_d.at[row_idx, 'Rule'] = e_rule; df_d.at[row_idx, 'Stop_Loss'] = e_stop; df_d.at[row_idx, 'Notes'] = e_note; df_d.at[row_idx, 'Shares'] = e_shs; df_d.at[row_idx, 'Amount'] = e_amt; df_d.at[row_idx, 'Value'] = e_shs * e_amt; df_d.at[row_idx, 'Trx_ID'] = e_trx
                                        secure_save(df_d, DETAILS_FILE); df_d, df_s = update_campaign_summary(edit_id, df_d, df_s); secure_save(df_d, DETAILS_FILE); secure_save(df_s, SUMMARY_FILE); st.success("Updated & Resynced!"); st.rerun()
                            with cB:
                                st.write("  "); st.write("  "); st.write("  ")
                                if st.button("DELETE TRANSACTION", type="primary"):
                                    df_d = df_d.drop(row_idx); secure_save(df_d, DETAILS_FILE); df_d, df_s = update_campaign_summary(edit_id, df_d, df_s); secure_save(df_d, DETAILS_FILE); secure_save(df_s, SUMMARY_FILE); st.success("Transaction Deleted."); st.rerun()
                else: st.warning("No transactions found.")
        else: st.info("No transaction data available.")

    with tab5:
        st.subheader("Database Maintenance")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("FORCE REPAIR (Recalculate All)"):
                if not df_d.empty:
                    progress_bar = st.progress(0); all_ids = df_d['Trade_ID'].unique()
                    for idx, tid in enumerate(all_ids):
                        df_d, df_s = update_campaign_summary(tid, df_d, df_s); progress_bar.progress((idx + 1) / len(all_ids))
                    secure_save(df_d, DETAILS_FILE); secure_save(df_s, SUMMARY_FILE); st.success(f"Rebuilt {len(all_ids)} Campaigns."); st.rerun()
        with c2:
            if st.button("PATCH MISSING DATES"):
                count = 0
                for idx, row in df_d.iterrows():
                    if pd.isna(row['Date']):
                        tid = row['Trade_ID']; s_row = df_s[df_s['Trade_ID'] == tid]
                        if not s_row.empty and pd.notna(s_row.iloc[0]['Open_Date']): df_d.at[idx, 'Date'] = s_row.iloc[0]['Open_Date']
                        else: df_d.at[idx, 'Date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                        count += 1
                if count > 0: secure_save(df_d, DETAILS_FILE); st.success(f"Patched {count} rows."); st.rerun()
                else: st.info("No missing dates found.")

    with tab6:
        if not df_s.empty:
            del_id = st.selectbox("ID to Delete", df_s['Trade_ID'].tolist())
            if st.button("DELETE PERMANENTLY"):
                df_s = df_s[df_s['Trade_ID']!=del_id]; df_d = df_d[df_d['Trade_ID']!=del_id]; secure_save(df_s, SUMMARY_FILE); secure_save(df_d, DETAILS_FILE); st.success("Deleted."); st.rerun()

    with tab7:
        st.subheader("Active Campaign Summary")
        if not df_s.empty and valid_sum_cols and 'Status' in df_s.columns:
             df_open = df_s[df_s['Status'] == 'OPEN'].copy()
             def get_last_stop(tid):
                 try:
                     stops = df_d[df_d['Trade_ID'] == tid]['Stop_Loss']
                     return stops.iloc[-1] if not stops.empty else 0.0
                 except: return 0.0
             df_open['Stop Loss'] = df_open['Trade_ID'].apply(get_last_stop)
             df_open['Current Value'] = df_open['Total_Cost'] + df_open.get('Unrealized_PL', 0.0).fillna(0.0)
             df_open['Current Price'] = df_open.apply(lambda x: (x['Current Value']/x['Shares']) if x['Shares']>0 else 0, axis=1)
             df_open['Risk $'] = (df_open['Current Price'] - df_open['Stop Loss']) * df_open['Shares']
             df_open['Risk $'] = df_open['Risk $'].apply(lambda x: x if x > 0 else 0.0)
             df_open['Risk %'] = (df_open['Risk $'] / current_equity) * 100
             df_open['Pos Size %'] = (df_open['Current Value'] / current_equity) * 100
             df_open['Max SL (0.5%)'] = df_open.apply(lambda row: row['Avg_Entry'] - ((current_equity * 0.005) / row['Shares']) if row['Shares'] > 0 else 0, axis=1)
             m1, m2, m3, m4 = st.columns(4)
             m1.metric("Open Positions", len(df_open))
             m2.metric("Total Unrealized P&L", f"${df_open['Unrealized_PL'].sum():,.2f}")
             m3.metric("Total Open Risk ($)", f"${df_open['Risk $'].sum():,.2f}")
             m4.metric("Total Portfolio Heat (%)", f"{df_open['Risk %'].sum():.2f}%")
             cols_final = ['Trade_ID', 'Ticker', 'Open_Date', 'Shares', 'Avg_Entry', 'Total_Cost', 'Current Value', 'Unrealized_PL', 'Return_Pct', 'Pos Size %', 'Stop Loss', 'Max SL (0.5%)', 'Risk $', 'Risk %', 'Buy_Rule']
             st.dataframe(df_open[cols_final].style.format({'Shares':'{:.0f}', 'Total_Cost':'${:,.2f}', 'Unrealized_PL':'${:,.2f}', 'Avg_Entry':'${:,.2f}', 'Return_Pct':'{:.2f}%', 'Current Value': '${:,.2f}', 'Pos Size %': '{:.1f}%', 'Stop Loss': '${:,.2f}', 'Max SL (0.5%)': '${:,.2f}', 'Risk $': '${:,.2f}', 'Risk %': '{:.2f}%', 'Open_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else ''}).applymap(color_pnl, subset=['Unrealized_PL']))
        else: st.info("No open positions.")

    with tab8:
        st.subheader("Active Campaign Detailed (Transactions)")
        if not df_d.empty and not df_s.empty and 'Status' in df_s.columns:
            open_ids = df_s[df_s['Status'] == 'OPEN']['Trade_ID'].unique()
            view_df = df_d[df_d['Trade_ID'].isin(open_ids)].copy()
            start_map = df_s.set_index('Trade_ID')['Open_Date'].to_dict()
            view_df['Campaign_Start'] = view_df['Trade_ID'].map(start_map)
            tick_filter = st.selectbox("Filter Open Ticker", ["All"] + sorted(view_df['Ticker'].unique().tolist()), key='act_det')
            if tick_filter != "All": view_df = view_df[view_df['Ticker'] == tick_filter]
            if not view_df.empty:
                curr_prices = {}
                for idx, row in df_s[df_s['Status']=='OPEN'].iterrows():
                    if row['Shares'] > 0: curr_prices[row['Trade_ID']] = (row['Total_Cost'] + row.get('Unrealized_PL', 0)) / row['Shares']
                display_df = view_df.copy()
                remaining_map = {}
                for tid in display_df['Trade_ID'].unique():
                    subset = display_df[display_df['Trade_ID'] == tid].copy()
                    subset['Type_Rank'] = subset['Action'].apply(lambda x: 0 if x == 'BUY' else 1)
                    subset = subset.sort_values(['Date', 'Type_Rank'])
                    inventory = [] 
                    for idx, row in subset.iterrows():
                        if row['Action'] == 'BUY':
                            inventory.append({'idx': idx, 'qty': row['Shares']})
                            remaining_map[idx] = row['Shares']
                        elif row['Action'] == 'SELL':
                            to_sell = row['Shares']
                            while to_sell > 0 and inventory:
                                last = inventory.pop()
                                take = min(to_sell, last['qty'])
                                last['qty'] -= take
                                to_sell -= take
                                remaining_map[last['idx']] = last['qty']
                                if last['qty'] > 0: inventory.append(last)
                display_df['Remaining_Shares'] = display_df.index.map(remaining_map).fillna(0)
                def calc_unrealized(row): 
                     if row['Action'] == 'BUY' and row['Remaining_Shares'] > 0:
                         price = curr_prices.get(row['Trade_ID'], 0)
                         return (price - row['Amount']) * row['Remaining_Shares']
                     return 0.0
                display_df['Unrealized_PL'] = display_df.apply(calc_unrealized, axis=1)
                display_df['Realized_PL'] = display_df.apply(lambda x: x['Realized_PL'] if x['Action'] == 'SELL' else 0, axis=1)
                display_df['Shares'] = display_df.apply(lambda x: -x['Shares'] if x['Action'] == 'SELL' else x['Shares'], axis=1)
                display_df['Value'] = display_df.apply(lambda x: -x['Value'] if x['Action'] == 'SELL' else x['Value'], axis=1)
                final_cols = ['Trade_ID', 'Trx_ID', 'Campaign_Start', 'Date', 'Ticker', 'Action', 'Shares', 'Remaining_Shares', 'Amount', 'Stop_Loss', 'Value', 'Realized_PL', 'Unrealized_PL', 'Rule', 'Notes']
                show_cols = [c for c in final_cols if c in display_df.columns]
                st.dataframe(display_df[show_cols].sort_values(['Trade_ID', 'Date']).style.format({'Date': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'None', 'Campaign_Start': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else '', 'Amount':'${:,.2f}', 'Stop_Loss':'${:,.2f}', 'Value':'${:,.2f}', 'Realized_PL':'${:,.2f}', 'Unrealized_PL':'${:,.2f}', 'Remaining_Shares':'{:.0f}'}).applymap(color_pnl, subset=['Value','Realized_PL','Unrealized_PL']).applymap(color_neg_value, subset=['Shares']))
            else: st.info("No open transactions.")
        else: st.info("No data.")

    with tab9:
        st.subheader("Detailed Trade Log (History)")
        tick_filter = st.selectbox("Filter History Ticker", ["All"] + sorted(df_d['Ticker'].unique().tolist()), key='hist_det')
        view_df = df_d if tick_filter=="All" else df_d[df_d['Ticker']==tick_filter]
        if not view_df.empty:
            display_df = view_df.copy()
            display_df['Shares'] = display_df.apply(lambda x: -x['Shares'] if x['Action'] == 'SELL' else x['Shares'], axis=1)
            display_df['Value'] = display_df.apply(lambda x: -x['Value'] if x['Action'] == 'SELL' else x['Value'], axis=1)
            final_cols = ['Trade_ID', 'Trx_ID', 'Date', 'Ticker', 'Action', 'Shares', 'Amount', 'Stop_Loss', 'Value', 'Realized_PL', 'Rule', 'Notes']
            show_cols = [c for c in final_cols if c in display_df.columns]
            st.dataframe(display_df[show_cols].sort_values(['Trade_ID', 'Date']).style.format({'Date': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'None', 'Shares':'{:.0f}', 'Amount':'${:,.2f}', 'Stop_Loss':'${:,.2f}', 'Value':'${:,.2f}', 'Realized_PL':'${:,.2f}'}).applymap(color_pnl, subset=['Value','Realized_PL']).applymap(color_neg_value, subset=['Shares']))

    with tab10:
        st.subheader("All Campaigns (Summary)")
        if not df_s.empty:
            df_s_view = df_s.reset_index().rename(columns={'index': 'Seq_ID'})
            def get_result(row):
                if row['Status'] == 'OPEN': return "OPEN"
                pct = row['Return_Pct']; return "BE" if -0.5 <= pct <= 0.5 else ("WIN" if pct > 0.5 else "LOSS")
            df_s_view['Result'] = df_s_view.apply(get_result, axis=1)
            tick_filter_all = st.selectbox("Filter Campaign Ticker", ["All"] + sorted(df_s['Ticker'].unique().tolist()))
            if tick_filter_all != "All": df_s_view = df_s_view[df_s_view['Ticker'] == tick_filter_all]
            
            st.dataframe(df_s_view[['Trade_ID', 'Ticker', 'Status', 'Result', 'Open_Date', 'Closed_Date', 'Realized_PL', 'Return_Pct', 'Buy_Rule']].style.format({'Realized_PL': '${:,.2f}', 'Return_Pct': '{:.2f}%', 'Open_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else ''}).applymap(color_pnl, subset=['Realized_PL']).applymap(color_result, subset=['Result']), hide_index=True)
        else: st.info("No campaigns found.")

elif page == "Analytics":
    st.header("PERFORMANCE AUDIT")
    if not df_s.empty:
        closed = df_s[df_s['Status']=='CLOSED'].copy()
        wins = closed[closed['Realized_PL'] > 0]
        losses = closed[closed['Realized_PL'] <= 0]
        
        gross_profit = wins['Realized_PL'].sum() if not wins.empty else 0
        gross_loss = abs(losses['Realized_PL'].sum()) if not losses.empty else 0
        pf_val = gross_profit/gross_loss if gross_loss != 0 else 0
        bat_avg = (len(wins)/len(closed) * 100) if not closed.empty else 0
        avg_win = wins['Realized_PL'].mean() if not wins.empty else 0
        avg_loss = losses['Realized_PL'].mean() if not losses.empty else 0
        wl_ratio = abs(avg_win/avg_loss) if avg_loss!=0 else 0.0
        
        if PLOTLY_AVAILABLE:
            st.subheader("1. The Scoreboard")
            fig_pf = go.Figure(go.Indicator(mode="gauge+number", value=pf_val, title={'text': "Profit Factor"}, gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "#2ca02c" if pf_val > 1.5 else "orange"}}))
            fig_pf.update_layout(height=250)
            st.plotly_chart(fig_pf, use_container_width=True)
        else:
            st.metric("Profit Factor", f"{pf_val:.2f}")

        if not closed.empty:
            strat = closed.groupby('Buy_Rule').agg(Trades=('Trade_ID','count'), PL=('Realized_PL','sum'), WinRate=('Realized_PL', lambda x: (x>0).mean())).sort_values('PL', ascending=False)
            st.markdown("---"); st.subheader("3. Strategy Breakdown")
            st.dataframe(strat.style.format({'PL':'${:,.2f}', 'WinRate':'{:.1%}'}).applymap(color_pnl, subset=['PL']))
    else: st.info("No data available.")