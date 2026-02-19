import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math
from datetime import datetime, time, timedelta
import os
import shutil



# --- CONFIGURATION ---
st.set_page_config(page_title="CAN SLIM COMMAND CENTER", layout="wide", page_icon="📈")
APP_VERSION = "16.0 (Clean Workflow)"

# --- CONSTANTS & PATHS ---
DATA_ROOT = "portfolios"
PORT_CANSLIM = "CanSlim"
PORT_TQQQ = "TQQQ Strategy"
PORT_457B = "457B Plan"
ALL_PORTFOLIOS = [PORT_CANSLIM, PORT_TQQQ, PORT_457B]

# --- SAFETY CHECK: Ensure Folders Exist ---
if not os.path.exists(DATA_ROOT): os.makedirs(DATA_ROOT)
for p in ALL_PORTFOLIOS:
    path = os.path.join(DATA_ROOT, p)
    if not os.path.exists(path): os.makedirs(path)

# Risk Settings
RISK_START_DATE = '2025-11-14'

# --- PLOTLY CHECK ---
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --- RULES LISTS ---
BUY_RULES = [
    # 1. Base Breakouts
    "br1.1 Consolidation", "br1.2 Cup w Handle", "br1.3 Cup w/o Handle", "br1.4 Double Bottom",
    "br1.5 IPO Base", "br1.6 Flat Base", 
    # 2. Volume & Volatility
    "br2.1 HVE", "br2.2 HVSI", "br2.3 HV1",
    # 3. Moving Average Reclaims
    "br3.1 Reclaim 21e", "br3.2 Reclaim 50s", "br3.3 Reclaim 200s", "br3.4 Reclaim 10W", 
    # 4. Pullbacks
    "br4.1 PB 21e", "br4.2 PB 50s", "br4.3 PB 10w", "br4.4 PB 200s", "br4.5 VWAP", 
    # 5. Reversals
    "br5.1 Undercut & Rally", "br5.2 Upside Reversal", 
    # 6. Gaps
    "br6.1 Gapper", 
    # 7. Strategies
    "br7.1 TQQQ Strategy", 
    # 8. Trendline Breaks
    "br8.1 Daily STL Break", "br8.2 Weekly STL Break", "br8.3 Monthly STL Break", 
    # 9. Moving Average Strategies
    "br9.1 21e Strategy", 
    # 10. Pyramiding / Adds (Formerly Add Rules)
    "br10.1 Add: New High after Gentle PB", "br10.2 Add: KMA Pullback", 
    "br10.3 Add: KMA Reclaim", "br10.4 Add: JL Century",
    "br10.5 Add: Continuation Gap Up", "br10.6 Add: High Low Support", 
    "br10.7 Add: 3 Weeks Tight", "br10.8 Add: Generic Scale-In",
    # Misc
    "ns No Setup"
]

# Note: ADD_RULES list is deprecated/merged into BUY_RULES

SELL_RULES = [
    "sr1 Capital Protection", # Renamed from sr1.1 (Hard Stop implied)
    "sr2 Selling into Strength", 
    "sr3 Portfolio Management",
    "sr4 Change of Character", 
    "sr5 Equator Line Break", 
    "sr6 Webby RS Rule",
    "sr7 Selling before Earnings", 
    "sr8 TQQQ Strategy Exit", 
    "sr9 Breakout Failure"
]

# Combined for Dropdowns
ALL_RULES = sorted(list(set(BUY_RULES + SELL_RULES)))

# --- HELPER FUNCTIONS ---
def clean_num(x):
    try:
        if pd.isna(x) or str(x).strip() == '': return 0.0
        s = str(x).strip().replace('$','').replace(',','').replace('%','')
        if '(' in s: s = s.replace('(', '-').replace(')', '')
        return float(s)
    except: return 0.0

def clean_dataframe(df):
    df.columns = [c.strip().replace(',', '').replace('"', '') for c in df.columns]
    valid_cols = [c for c in df.columns if 'Unnamed' not in c and c != '']
    df = df[valid_cols]
    return df

def secure_save(df, filename):
    if not os.path.exists(os.path.dirname(filename)): os.makedirs(os.path.dirname(filename))
    if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)
    if os.path.exists(filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"{os.path.basename(filename).replace('.csv', '')}_{timestamp}.csv")
        try: shutil.copy(filename, backup_path)
        except: pass
    
    # Date formatting for CSV
    if filename in [DETAILS_FILE, SUMMARY_FILE]:
        date_cols = ['Date', 'Open_Date', 'Closed_Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
    df.to_csv(filename, index=False)

def load_data(file):
    if not os.path.exists(file): return pd.DataFrame()
    try:
        df = pd.read_csv(file)
        df = clean_dataframe(df)
        
        if file.endswith('Trading_Journal_Clean.csv'):
            expected_cols = ['Beg NLV', 'End NLV', 'Cash -/+', 'Daily $ Change', 'SPY', 'Nasdaq', '% Invested']
            for c in expected_cols:
                if c in df.columns: df[c] = df[c].apply(clean_num)
            if 'Nsadaq' in df.columns: df.rename(columns={'Nsadaq': 'Nasdaq'}, inplace=True)
            if 'Day' in df.columns: df['Day'] = pd.to_datetime(df['Day'], errors='coerce')
            if 'Market_Action' not in df.columns: df['Market_Action'] = ""
            if 'Keywords' not in df.columns: df['Keywords'] = ""
            if 'Score' not in df.columns: df['Score'] = 0
            
        if file.endswith('Details.csv') or file.endswith('Summary.csv'):
            rename_map = {'Total_Shares': 'Shares', 'Close_Date': 'Closed_Date', 'Cost': 'Amount', 'Price': 'Amount', 'Net': 'Value'}
            df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}, inplace=True)
            date_cols = ['Date', 'Open_Date', 'Closed_Date']
            for col in date_cols:
                if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
            num_cols = ['Shares', 'Amount', 'Value', 'Total_Cost', 'Realized_PL', 'Avg_Entry', 'Avg_Exit', 'Stop_Loss']
            for num_col in num_cols:
                if num_col not in df.columns: df[num_col] = 0.0
                else: df[num_col] = df[num_col].apply(clean_num)
            if 'Trade_ID' in df.columns: 
                df['Trade_ID'] = df['Trade_ID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            if 'Trx_ID' in df.columns: 
                df['Trx_ID'] = df['Trx_ID'].astype(str).str.strip()
            if 'Action' in df.columns:
                df['Action'] = df['Action'].astype(str).str.strip().str.upper()
            if 'Ticker' in df.columns:
                df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
                df = df[df['Ticker'] != 'NAN']
                df = df[df['Ticker'] != '']

        return df
    except: return pd.DataFrame()

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
        # 1. Get Transactions for this Trade ID
        txs = df_d[df_d['Trade_ID'] == trade_id].copy()
        if txs.empty: return df_d, df_s
        
        # 2. Sort Chronologically
        txs['Date'] = pd.to_datetime(txs['Date'], errors='coerce')
        txs = txs.dropna(subset=['Date'])
        txs['Sort_Date'] = txs['Date'].dt.normalize()
        txs['Type_Rank'] = txs['Action'].apply(lambda x: 0 if x == 'BUY' else 1)
        txs = txs.sort_values(['Sort_Date', 'Type_Rank', 'Date']) 
        
        inventory = []
        total_realized_pl = 0.0
        
        # 3. LIFO Math Engine
        for _, row in txs.iterrows():
            idx = row.name 
            if row['Action'] == 'BUY':
                inventory.append({'price': row['Amount'], 'shares': row['Shares']})
                df_d.at[idx, 'Realized_PL'] = 0.0
            elif row['Action'] == 'SELL':
                shares_to_sell = row['Shares']
                sell_price = row['Amount']
                trx_pnl = 0.0
                while shares_to_sell > 0 and inventory:
                    last_lot = inventory[-1] 
                    take = min(shares_to_sell, last_lot['shares'])
                    pnl = (sell_price - last_lot['price']) * take
                    trx_pnl += pnl
                    shares_to_sell -= take
                    last_lot['shares'] -= take
                    if last_lot['shares'] < 0.0001: inventory.pop() 
                df_d.at[idx, 'Realized_PL'] = trx_pnl
                total_realized_pl += trx_pnl

        # 4. Calculate Inventory Stats
        curr_shares = sum(item['shares'] for item in inventory)
        curr_cost_total = sum(item['shares'] * item['price'] for item in inventory)
        
        if curr_shares > 0: avg_entry = curr_cost_total / curr_shares
        else:
            buys = txs[txs['Action'] == 'BUY']
            avg_entry = buys['Value'].sum() / buys['Shares'].sum() if not buys.empty else 0.0

        # 5. UPDATE SUMMARY (THE SYNC STEP)
        # Convert both to string to ensure match
        idx = df_s[df_s['Trade_ID'].astype(str).str.replace('.0','') == str(trade_id).replace('.0','')].index
        if not idx.empty:
            i = idx[0]
            buys = txs[txs['Action'] == 'BUY']
            sells = txs[txs['Action'] == 'SELL']
            
            # --- THE MISSING LINK: FORCE RULE SYNC ---
            if not buys.empty: 
                # Ensure 'Rule' column exists in Summary
                if 'Rule' not in df_s.columns: df_s['Rule'] = ''
                
                # Sync Rule from the FIRST Buy Transaction
                first_buy_rule = buys.iloc[0].get('Rule', '')
                df_s.at[i, 'Rule'] = first_buy_rule
                
                # Sync Open Date
                df_s.at[i, 'Open_Date'] = buys.iloc[0]['Date'].strftime('%Y-%m-%d %H:%M')
            
            df_s.at[i, 'Avg_Entry'] = avg_entry
            df_s.at[i, 'Realized_PL'] = total_realized_pl
            
            if curr_shares < 1:
                df_s.at[i, 'Status'] = 'CLOSED'
                df_s.at[i, 'Shares'] = buys['Shares'].sum()
                df_s.at[i, 'Total_Cost'] = buys['Value'].sum()
                if not sells.empty: df_s.at[i, 'Avg_Exit'] = sells['Value'].sum() / sells['Shares'].sum()
                df_s.at[i, 'Closed_Date'] = sells.iloc[-1]['Date'].strftime('%Y-%m-%d %H:%M') if not sells.empty else None
                df_s.at[i, 'Return_Pct'] = (total_realized_pl / df_s.at[i, 'Total_Cost'] * 100) if df_s.at[i, 'Total_Cost'] != 0 else 0.0
            else:
                df_s.at[i, 'Status'] = 'OPEN'
                df_s.at[i, 'Shares'] = curr_shares
                df_s.at[i, 'Total_Cost'] = curr_cost_total
                df_s.at[i, 'Closed_Date'] = None

        return df_d, df_s
    except Exception as e: 
        print(f"Error updating campaign: {e}")
        return df_d, df_s
        return df_d, df_s

def color_pnl(val): return 'color: #ff4b4b' if isinstance(val, (int, float)) and val < 0 else 'color: #2ca02c'
def color_neg_value(val): return 'color: #ff4b4b' if isinstance(val, (int, float)) and val < 0 else ''
def color_result(val):
    if val == 'WIN': return 'color: #2ca02c; font-weight: bold'
    elif val == 'LOSS': return 'color: #ff4b4b; font-weight: bold'
    return 'color: gray'
def color_score(val):
    try:
        v = float(val)
        if v >= 4: return 'color: #2ca02c; font-weight: bold'
        if v <= 2: return 'color: #ff4b4b; font-weight: bold'
    except: pass
    return ''

def calculate_open_risk(df_d, df_s, nlv):
    if df_d.empty or df_s.empty or nlv == 0: return 0.0, 0.0
    total_risk_dollars = 0.0
    if 'Status' in df_s.columns:
        open_campaigns = df_s[df_s['Status'] == 'OPEN']
        for _, campaign in open_campaigns.iterrows():
            tid = campaign['Trade_ID']; shares_held = campaign.get('Shares', 0)
            try:
                stops = df_d[df_d['Trade_ID'] == tid]['Stop_Loss']
                valid_stops = stops[stops > 0]
                stop_val = valid_stops.iloc[-1] if not valid_stops.empty else 0.0
            except: stop_val = 0.0
            if shares_held > 0 and stop_val > 0:
                current_val = campaign['Total_Cost'] + campaign.get('Unrealized_PL', 0.0)
                current_price = current_val / shares_held if shares_held > 0 else 0
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
                    if c > m: count += 1
                    else: break
                else:
                    if c < m: count += 1
                    else: break
            direction = "⬆" if is_above else "⬇"; dcol = "#00cc00" if is_above else "#ff3333"
            diff_pct = ((curr_close - curr_ma) / curr_ma) * 100
            return count, direction, dcol, diff_pct, curr_ma
        s_days, s_dir, s_dcol, s_diff, s_val = get_streak('21EMA'); m_days, m_dir, m_dcol, m_diff, m_val = get_streak('50SMA'); l_days, l_dir, l_dcol, l_diff, l_val = get_streak('200SMA')
        return {'price': df['Close'].iloc[-1], 'short': {'stat':s_stat, 'col':s_col, 'sym':s_sym, 'val':s_val, 'days':s_days, 'dir':s_dir, 'dcol':s_dcol, 'diff':s_diff}, 'med': {'stat':m_stat, 'col':m_col, 'sym':m_sym, 'val':m_val, 'days':m_days, 'dir':m_dir, 'dcol':m_dcol, 'diff':m_diff}, 'long': {'stat':l_stat, 'col':l_col, 'sym':l_sym, 'val':l_val, 'days':l_days, 'dir':l_dir, 'dcol':l_dcol, 'diff':l_diff}}
    except: return None

# ==============================================================================
# 2. SIDEBAR NAVIGATION (THE ONE SOURCE OF TRUTH)
# ==============================================================================
st.sidebar.title("🚀 SYSTEM O'NEIL")
st.sidebar.markdown("---")

# A. SINGLE STRATEGY SELECTOR
# This variable 'portfolio' controls the entire app context.
portfolio = st.sidebar.selectbox(
    "🔥 Active Strategy", 
    [PORT_CANSLIM, PORT_TQQQ, PORT_457B],
    index=0,
    help="Select the account you want to manage."
)

# B. DYNAMIC PATH CONFIGURATION
# We define the paths IMMEDIATELY so every page knows where to look.
ACTIVE_DIR = os.path.join(DATA_ROOT, portfolio)
BACKUP_DIR = os.path.join(ACTIVE_DIR, 'backups') # <--- Added this back for safety

if portfolio == PORT_CANSLIM:
    CURR_PORT_NAME = "CanSlim (Main)"
elif portfolio == PORT_TQQQ:
    CURR_PORT_NAME = "TQQQ Strategy"
else:
    CURR_PORT_NAME = "457B (Retirement)"

# Standardized Filenames (Since they are the same for all folders now)
JOURNAL_FILE = os.path.join(ACTIVE_DIR, 'Trading_Journal_Clean.csv')
SUMMARY_FILE = os.path.join(ACTIVE_DIR, 'Trade_Log_Summary.csv')
DETAILS_FILE = os.path.join(ACTIVE_DIR, 'Trade_Log_Details.csv')

st.sidebar.markdown("---")

# C. PAGE NAVIGATION
# One menu to rule them all.
page = st.sidebar.radio("Go to Module", [
    "Command Center", 
    "Dashboard", 
    "Daily Routine", 
    "Daily Journal", 
    "Daily Report Card",  # <--- NEW FLIGHT RECORDER ADDED HERE
    "M Factor", 
    "Risk Manager", 
    "Ticker Forensics", 
    "Period Review", 
    "Position Sizer", 
    "Trade Manager", 
    "Analytics"
])

st.sidebar.markdown("---")
st.sidebar.caption(f"📂 **Active:** {CURR_PORT_NAME}")

# ==============================================================================
# PAGE 1: COMMAND CENTER (PRECISION YTD FIX)
# ==============================================================================
if page == "Command Center":
    st.title("COMMAND CENTER")

    # --- CONFIGURATION ---
    CANSLIM_RESET_DATE = pd.Timestamp("2025-12-16") 

    # --- 1. ROBUST DATA LOADER ---
    def clean_num_local(x):
        try:
            if isinstance(x, str):
                return float(x.replace('$', '').replace(',', '').replace('%', '').strip())
            return float(x)
        except: return 0.0

    p_c_j = os.path.join(DATA_ROOT, PORT_CANSLIM, 'Trading_Journal_Clean.csv')
    p_c_s = os.path.join(DATA_ROOT, PORT_CANSLIM, 'Trade_Log_Summary.csv')
    p_c_d = os.path.join(DATA_ROOT, PORT_CANSLIM, 'Trade_Log_Details.csv')
    p_t_j = os.path.join(DATA_ROOT, PORT_TQQQ, 'Trading_Journal_Clean.csv')
    p_t_s = os.path.join(DATA_ROOT, PORT_TQQQ, 'Trade_Log_Summary.csv')
    p_t_d = os.path.join(DATA_ROOT, PORT_TQQQ, 'Trade_Log_Details.csv')
    p_r_j = os.path.join(DATA_ROOT, PORT_457B, 'Trading_Journal_Clean.csv')

    def load_clean(p):
        d = load_data(p)
        if not d.empty and 'Day' in d.columns:
            d['Day'] = pd.to_datetime(d['Day'], errors='coerce')
            d.sort_values('Day', inplace=True) 
            
            for c in ['Beg NLV', 'End NLV', 'Cash -/+', 'Daily $ Change', 'SPY', 'Nasdaq']:
                if c in d.columns: d[c] = d[c].apply(clean_num_local)
            
            # Calculates Daily Return strictly from PL/Beg to avoid cash flow distortion
            if 'Daily $ Change' in d.columns:
                d['Trusted_PL'] = d['Daily $ Change'].fillna(0.0)
            else:
                d['Trusted_PL'] = d['End NLV'] - d['Beg NLV'] - d['Cash -/+']

            d['Adjusted_Beg'] = d['Beg NLV'] + d['Cash -/+']
            d['Trusted_Ret'] = 0.0
            mask = d['Adjusted_Beg'] != 0
            d.loc[mask, 'Trusted_Ret'] = d.loc[mask, 'Trusted_PL'] / d.loc[mask, 'Adjusted_Beg']
                
        return d

    df_cj, df_cs, df_cd = load_clean(p_c_j), load_data(p_c_s), load_data(p_c_d)
    df_tj, df_ts, df_td = load_clean(p_t_j), load_data(p_t_s), load_data(p_t_d)
    df_rj = load_clean(p_r_j)

    # --- 2. METRICS ENGINE (PRECISION TWR) ---
    def get_smart_metrics(df):
        if df.empty: return 0, 0, 0, 0, 0, df
        
        df['Daily_PL'] = df.get('Trusted_PL', 0.0)
        df['Daily_Ret'] = df.get('Trusted_Ret', 0.0)
        
        # Build Equity Curve (Base 1.0)
        df['EC'] = (1 + df['Daily_Ret']).cumprod()
        
        hwm_ec = df['EC'].max()
        curr_ec = df['EC'].iloc[-1]
        dd = ((curr_ec - hwm_ec) / hwm_ec) * 100 if hwm_ec > 0 else 0
        
        curr_nlv = df['End NLV'].iloc[-1]
        day_chg = df['Daily_PL'].iloc[-1]
        ec_21 = df['EC'].ewm(span=21, adjust=False).mean().iloc[-1]
        
        return curr_nlv, day_chg, dd, curr_ec, ec_21, df

    can_nlv, can_chg, can_dd, can_ec, can_21, df_cj_c = get_smart_metrics(df_cj.copy())
    tqq_nlv, tqq_chg, tqq_dd, tqq_ec, tqq_21, df_tj_c = get_smart_metrics(df_tj.copy())
    ret_nlv, ret_chg, ret_dd, ret_ec, ret_21, df_rj_c = get_smart_metrics(df_rj.copy())

    # --- 3. RECALCULATE CANSLIM DD (POST-SPLIT RESET) ---
    if not df_cj_c.empty:
        df_post_split = df_cj_c[df_cj_c['Day'] > CANSLIM_RESET_DATE]
        if not df_post_split.empty:
            # Re-base HWM calculation to the split date
            hwm_split = df_post_split['End NLV'].max()
            curr_split = df_post_split['End NLV'].iloc[-1]
            can_dd = ((curr_split - hwm_split) / hwm_split) * 100 if hwm_split > 0 else 0.0
        else:
            can_dd = 0.0

    # Totals
    tot_nlv = can_nlv + tqq_nlv + ret_nlv
    tot_day_chg = can_chg + tqq_chg + ret_chg

    # --- 4. COMBINED CORE (2 ACCOUNTS) ---
    df_core = pd.DataFrame()
    d1 = df_cj_c['Day'].tolist() if not df_cj_c.empty and 'Day' in df_cj_c.columns else []
    d2 = df_tj_c['Day'].tolist() if not df_tj_c.empty and 'Day' in df_tj_c.columns else []
    core_dates = sorted(list(set(d1 + d2)))

    if core_dates:
        def reindex_core(df, dates):
            if df.empty: return pd.DataFrame(index=dates, columns=['End NLV', 'Trusted_PL', 'Beg NLV', 'Cash -/+']).fillna(0)
            df = df.set_index('Day').reindex(dates)
            df['End NLV'] = df['End NLV'].fillna(method='ffill').fillna(0)
            df['Trusted_PL'] = df['Trusted_PL'].fillna(0.0) 
            df['Cash -/+'] = df['Cash -/+'].fillna(0.0)
            df['Beg NLV'] = df['End NLV'] - df['Trusted_PL'] - df['Cash -/+']
            return df
        
        c_core = reindex_core(df_cj_c, core_dates)
        t_core = reindex_core(df_tj_c, core_dates)
        
        df_core = pd.DataFrame(index=core_dates)
        df_core['End NLV'] = c_core['End NLV'] + t_core['End NLV']
        df_core['Daily_PL'] = c_core['Trusted_PL'] + t_core['Trusted_PL']
        df_core['Cash -/+'] = c_core['Cash -/+'] + t_core['Cash -/+']
        df_core['Beg NLV'] = c_core['Beg NLV'] + t_core['Beg NLV']
        
        denom_core = df_core['Beg NLV'] + df_core['Cash -/+']
        df_core['Daily_Pct'] = 0.0
        mask_c = denom_core != 0
        df_core.loc[mask_c, 'Daily_Pct'] = df_core.loc[mask_c, 'Daily_PL'] / denom_core.loc[mask_c]
        
        df_core['EC'] = (1 + df_core['Daily_Pct']).cumprod()
        df_core['LTD_Pct'] = (df_core['EC'] - 1) * 100

        bench_spy = df_cj_c.set_index('Day')['SPY'].reindex(core_dates).fillna(method='ffill')
        if not bench_spy.dropna().empty:
            start_spy = bench_spy.dropna().iloc[0]
            df_core['SPY_Bench'] = ((bench_spy / start_spy) - 1) * 100

    # --- TABS ---
    tab_dash, tab_core, tab_hist = st.tabs(["📊 Pilot's Panel", "⚔️ Trading Core (Combined)", "📜 Historical Data"])

    # --- TAB 1: PILOT'S PANEL (REDESIGNED) ---
    with tab_dash:
        st.markdown(f"### 🏦 Total Net Worth: **${tot_nlv:,.2f}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Daily Net P&L", f"${tot_day_chg:+,.2f}", delta_color="normal")
        
        # --- CANSLIM PRECISION YTD CALCULATION ---
        can_ytd_str = "0.00%"
        if not df_cj_c.empty:
            curr_yr = datetime.now().year
            df_yr = df_cj_c[df_cj_c['Day'].dt.year == curr_yr]
            if not df_yr.empty:
                # Math: (1 + r1) * (1 + r2) ... - 1
                # This ensures Day 1 is INCLUDED in the calculation
                ytd_raw = (1 + df_yr['Daily_Ret']).prod() - 1
                can_ytd_str = f"{ytd_raw * 100:.2f}%"

        c2.metric("CanSlim (Main)", f"${can_nlv:,.2f}", f"YTD: {can_ytd_str}")
        c3.metric("TQQQ Strat", f"${tqq_nlv:,.2f}", f"{tqq_dd:.2f}% DD")
        c4.metric("457B Plan", f"${ret_nlv:,.2f}", f"{ret_dd:.2f}% DD")
        st.markdown("---")

        col_L, col_R = st.columns(2)
        
        # === TQQQ STRATEGY (SMART LOGIC) ===
        with col_L:
            st.markdown("#### ⚡ TQQQ Strategy (QQQ Based)")
            tqqq_open = False
            if not df_ts.empty and 'Status' in df_ts.columns:
                if not df_ts[df_ts['Status'] == 'OPEN'].empty: tqqq_open = True
            
            try:
                q_hist = yf.Ticker("QQQ").history(period="1mo")
                if not q_hist.empty:
                    q_hist['21EMA'] = q_hist['Close'].ewm(span=21, adjust=False).mean()
                    curr_q = q_hist['Close'].iloc[-1]
                    curr_ema = q_hist['21EMA'].iloc[-1]
                    
                    st.metric("QQQ Price", f"${curr_q:.2f}", f"21e: ${curr_ema:.2f}")
                    
                    if not tqqq_open:
                        st.info("Status: 🛡️ CASH (Scanning for Entry)")
                        consecutive = 0
                        for i in range(len(q_hist)):
                            if q_hist.iloc[-(i+1)]['Low'] > q_hist.iloc[-(i+1)]['21EMA']: consecutive += 1
                            else: break
                        
                        is_up_day = False
                        if len(q_hist) >= 2: is_up_day = q_hist['Close'].iloc[-1] > q_hist['Close'].iloc[-2]
                            
                        st.write(f"**Setup Progress:** {consecutive} Day(s) Low > 21e")
                        if consecutive >= 3:
                            if is_up_day: st.success("✅ **BUY SIGNAL TRIGGERED** (3+ Days > 21e & Up Day)")
                            else: st.warning("⏳ **WAITING:** Setup Valid, Need Up Day.")
                        else: st.write(f"Need {3-consecutive} more days above 21e.")
                    else:
                        st.success("Status: 🚀 INVESTED")
                        is_below = curr_q < curr_ema
                        if is_below:
                            buffer_px = q_hist['Low'].iloc[-1] * 0.998
                            st.error("⚠️ **21e BROKEN (CLOSE)**")
                            st.write(f"Violation Low: ${q_hist['Low'].iloc[-1]:.2f}")
                            st.markdown(f"**SELL TRIGGER:** If Intraday < :red[**${buffer_px:.2f}**]")
                        else: st.write("✅ Price is above 21e. Hold.")
            except Exception as e: st.warning(f"Strategy Data Error: {e}")

        # === CANSLIM ALERTS (RESET) ===
        with col_R:
            st.markdown("#### 🌳 CanSlim (Main)")
            pr4_state = "UNKNOWN"; pr4_color = "gray"
            if not df_cj_c.empty:
                if can_ec >= can_21: pr4_state = "🚀 POWER TREND"; pr4_color = "green"
                else: pr4_state = "⚠️ UNDER PRESSURE"; pr4_color = "orange"
            st.markdown(f"**Market Status:** :{pr4_color}[**{pr4_state}**]")
            
            st.markdown(f"##### 🛡️ Circuit Breakers (Resets {CANSLIM_RESET_DATE.strftime('%m/%d')})")
            fuse_b = can_dd 
            cb1, cb2, cb3 = -3.0, -6.0, -10.0
            
            if fuse_b > cb1:
                st.success(f"🟢 **ALL CLEAR** (DD: {fuse_b:.2f}%) \n\n Cushion: **{abs(fuse_b - cb1):.2f}%** to Level 1.")
            elif cb1 >= fuse_b > cb2:
                st.warning(f"⚠️ **LEVEL 1 ACTIVE** (DD: {fuse_b:.2f}%)")
                st.warning("👉 **ACTION:** STOP BUYING")
            elif cb2 >= fuse_b > cb3:
                st.error(f"🛑 **LEVEL 2 ACTIVE** (DD: {fuse_b:.2f}%)")
                st.error("👉 **ACTION:** SELL LAGGARDS")
            else:
                st.error(f"☠️ **LEVEL 3 ACTIVE** (DD: {fuse_b:.2f}%)")
                st.error("👉 **ACTION:** LIQUIDATE / CASH")

    # --- TAB 2: TRADING CORE ---
    with tab_core:
        st.header("⚔️ TRADING CORE (CanSlim + TQQQ)")
        
        if not df_core.empty:
            core_nlv = df_core['End NLV'].iloc[-1]
            core_day_pl = df_core['Daily_PL'].iloc[-1]
            core_day_pct = df_core['Daily_Pct'].iloc[-1] * 100
            
            # Core YTD Calculation (Precision Mode)
            curr_yr = datetime.now().year
            df_core['Year'] = df_core.index.year
            df_core_yr = df_core[df_core['Year'] == curr_yr]
            ytd_val = 0.0
            if not df_core_yr.empty:
                ytd_val = ((1 + df_core_yr['Daily_Pct']).prod() - 1) * 100
            
            # Exposure Calc
            if not df_cs.empty: df_cs['Source'] = 'CanSlim'
            if not df_ts.empty: df_ts['Source'] = 'TQQQ'
            df_open_core = pd.concat([df_cs[df_cs['Status']=='OPEN'] if not df_cs.empty else pd.DataFrame(), df_ts[df_ts['Status']=='OPEN'] if not df_ts.empty else pd.DataFrame()], ignore_index=True)
            
            core_exp_pct = 0.0; core_pos_count = len(df_open_core)
            if not df_open_core.empty and core_nlv > 0:
                df_open_core['Mkt_Val'] = df_open_core['Total_Cost'] + df_open_core['Unrealized_PL']
                core_exp_pct = (df_open_core['Mkt_Val'].sum() / core_nlv) * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trading Core NLV", f"${core_nlv:,.2f}", f"{core_day_pl:+,.2f} ({core_day_pct:+.2f}%)")
            m2.metric("LTD Return", f"{df_core['LTD_Pct'].iloc[-1]:.2f}%")
            m3.metric("YTD Return", f"{ytd_val:.2f}%")
            m4.metric("Combined Exposure", f"{core_exp_pct:.1f}%", f"{core_pos_count} Pos")
            st.markdown("---")
            
            # PLOTS
            df_core['EC_8EMA'] = df_core['LTD_Pct'].ewm(span=8, adjust=False).mean()
            df_core['EC_21EMA'] = df_core['LTD_Pct'].ewm(span=21, adjust=False).mean()
            
            plt.style.use('bmh')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            last_port = df_core['LTD_Pct'].iloc[-1]
            ax1.plot(df_core.index, df_core['LTD_Pct'], color='darkblue', linewidth=2.5, label=f"Core ({last_port:+.1f}%)")
            ax1.plot(df_core.index, df_core['EC_8EMA'], color='purple', linewidth=1.2, label='8 EMA')
            ax1.plot(df_core.index, df_core['EC_21EMA'], color='green', linewidth=1.2, label='21 EMA')
            
            if 'SPY_Bench' in df_core.columns:
                last_spy = df_core['SPY_Bench'].iloc[-1]
                ax1.plot(df_core.index, df_core['SPY_Bench'], color='gray', alpha=0.5, label=f"SPY ({last_spy:+.1f}%)")
            
            ax1.fill_between(df_core.index, df_core['LTD_Pct'], df_core['EC_21EMA'], where=(df_core['LTD_Pct'] >= df_core['EC_21EMA']), interpolate=True, color='green', alpha=0.1)
            ax1.fill_between(df_core.index, df_core['LTD_Pct'], df_core['EC_21EMA'], where=(df_core['LTD_Pct'] < df_core['EC_21EMA']), interpolate=True, color='red', alpha=0.1)
            ax1.legend(loc='upper left'); ax1.set_title("Trading Core Equity Curve"); ax1.set_ylabel("Return %")
            
            colors = ['green' if x >= 0 else 'red' for x in df_core['Daily_PL']]
            ax2.bar(df_core.index, df_core['Daily_PL'], color=colors)
            ax2.set_title("Combined Daily P&L ($)")
            st.pyplot(fig)
        else: st.info("No data in Trading Accounts.")

    # --- TAB 3: HISTORY ---
    with tab_hist:
        st.header("📜 Trading Core History")
        if not df_core.empty:
            hist_view = df_core[['Beg NLV', 'Cash -/+', 'End NLV', 'Daily_PL', 'Daily_Pct']].sort_index(ascending=False).copy()
            hist_view.columns = ['Start Equity', 'Cash Flow', 'End Equity', 'Daily P&L ($)', 'Daily Return %']
            
            st.dataframe(
                hist_view.style.format({
                    'Start Equity': '${:,.2f}', 'Cash Flow': '${:,.2f}', 'End Equity': '${:,.2f}',
                    'Daily P&L ($)': '${:+,.2f}', 'Daily Return %': '{:+.2%}'
                })
                .applymap(lambda x: 'color: #2ca02c' if x >= 0 else 'color: #ff4b4b', subset=['Daily P&L ($)', 'Daily Return %'])
                , use_container_width=True, height=600
            )
        else: st.info("No history available.")

# ==============================================================================
# PAGE 2: DASHBOARD (HYBRID + SECTOR INTELLIGENCE - PRECISION FIX)
# ==============================================================================
if page == "Dashboard":
    # Fix Sidebar Variable Name
    CURR_PORT_NAME = portfolio 
    
    st.header(f"MARKET DASHBOARD: {CURR_PORT_NAME}")
    
    # --- HELPER: ROBUST FORMATTER ---
    def fmt_money(val):
        """Safely formats money, handling strings/floats/errors."""
        try:
            if isinstance(val, str):
                val = float(val.replace('$', '').replace(',', ''))
            return f"${val:,.2f}"
        except:
            return "$0.00"

    # --- HELPER: NUMERIC CLEANER ---
    def clean_num_local(x):
        try:
            if isinstance(x, str):
                return float(x.replace('$', '').replace(',', '').replace('%', '').strip())
            return float(x)
        except: return 0.0

    # --- HELPER: SECTOR CACHE ---
    @st.cache_data(ttl=3600*24)
    def get_sector_map(ticker_list):
        s_map = {}
        for t in ticker_list:
            if t == 'CASH': continue
            try:
                inf = yf.Ticker(t).info
                s_map[t] = inf.get('sector', 'Unknown')
            except:
                s_map[t] = 'Unknown'
        return s_map

    # Ensure we look for the Clean file first
    p_clean = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')
    p_legacy = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal.csv')
    final_j_path = p_clean if os.path.exists(p_clean) else p_legacy

    if not os.path.exists(final_j_path): st.error("Journal missing.")
    else:
        df_j = pd.read_csv(final_j_path)
        df_d = load_data(DETAILS_FILE)
        df_s = load_data(SUMMARY_FILE)
        
        # --- 1. PREPARE JOURNAL DATA ---
        if 'Nsadaq' in df_j.columns: df_j.rename(columns={'Nsadaq': 'Nasdaq'}, inplace=True)
        df_j['Day'] = pd.to_datetime(df_j['Day'], errors='coerce')
        df_j = df_j.dropna(subset=['Day']).sort_values('Day')
        
        # Clean Numerics
        for c in ['Beg NLV', 'End NLV', 'Cash -/+', 'Daily $ Change', 'SPY', 'Nasdaq', '% Invested']:
            if c in df_j.columns: df_j[c] = df_j[c].apply(clean_num_local)

        # --- 2. LIVE DATA ENGINE ---
        calc_exposure_pct = 0.0
        num_open_pos = 0
        risk_pct = 0.0
        df_open = pd.DataFrame()
        curr_nlv = df_j['End NLV'].iloc[-1] if not df_j.empty else 0.0

        if not df_s.empty and curr_nlv > 0:
            df_open = df_s[df_s['Status'] == 'OPEN'].copy()
            if not df_open.empty:
                num_open_pos = len(df_open)
                
                # A. LIVE PRICE
                tickers = df_open['Ticker'].unique().tolist()
                try:
                    live_batch = yf.download(tickers, period="1d", progress=False)['Close'].iloc[-1]
                except: live_batch = pd.Series()

                def get_live_price(r):
                    try: 
                        if len(tickers) == 1:
                            val = float(live_batch) if not pd.isna(live_batch) else float(r['Avg_Entry'])
                            return val
                        else:
                            val = live_batch.get(r['Ticker'])
                            return float(val) if not pd.isna(val) else float(r['Avg_Entry'])
                    except: return float(r['Avg_Entry'])

                df_open['Cur_Px'] = df_open.apply(get_live_price, axis=1)

                # B. TRUE STOP & SECTOR
                def get_true_stop(trade_id):
                    txs = df_d[df_d['Trade_ID'] == trade_id]
                    if txs.empty: return 0.0
                    if 'Date' in txs.columns:
                        txs['Date'] = pd.to_datetime(txs['Date'], errors='coerce')
                        txs = txs.sort_values('Date')
                    valid_stops = txs['Stop_Loss'].dropna()
                    valid_stops = valid_stops[valid_stops > 0.01]
                    if not valid_stops.empty: return float(valid_stops.iloc[-1]) 
                    return 0.0 

                df_open['Stop_Loss'] = df_open['Trade_ID'].apply(get_true_stop)
                sec_map = get_sector_map(tickers)
                df_open['Sector'] = df_open['Ticker'].map(sec_map).fillna('Unknown')
                
                # C. CALCULATIONS
                df_open['Mkt_Val'] = df_open['Shares'] * df_open['Cur_Px']
                df_open['Unrealized_PL'] = df_open['Mkt_Val'] - df_open['Total_Cost']
                df_open['Return_Pct'] = df_open.apply(lambda x: (x['Unrealized_PL'] / x['Total_Cost'] * 100) if x['Total_Cost'] != 0 else 0, axis=1)
                
                df_open['R_Dol'] = (df_open['Cur_Px'] - df_open['Stop_Loss']) * df_open['Shares']
                risk_dol = df_open[df_open['R_Dol'] > 0]['R_Dol'].sum()
                risk_pct = (risk_dol / curr_nlv) * 100
                calc_exposure_pct = (df_open['Mkt_Val'].sum() / curr_nlv) * 100

        # --- 3. PERFORMANCE METRICS (PRECISION FIX) ---
        if not df_j.empty:
            
            # --- THE FIX: RECALCULATE TWR FROM SCRATCH ---
            # We ignore df_j['Daily % Change'] because it might contain historic manual rounding errors.
            # We recalculate strictly from End, Beg, and Cash.
            
            df_j['Adjusted_Beg'] = df_j['Beg NLV'] + df_j['Cash -/+']
            df_j['Daily_Pct'] = 0.0
            
            # Formula: (End - Adj_Beg) / Adj_Beg
            mask = df_j['Adjusted_Beg'] != 0
            df_j.loc[mask, 'Daily_Pct'] = (df_j.loc[mask, 'End NLV'] - df_j.loc[mask, 'Adjusted_Beg']) / df_j.loc[mask, 'Adjusted_Beg']
            
            # Build Equity Curve (Base 1.0)
            df_j['Equity_Curve'] = (1 + df_j['Daily_Pct']).cumprod()
            df_j['LTD_Pct'] = (df_j['Equity_Curve'] - 1) * 100
            
            # Benchmarks
            if 'SPY' in df_j.columns and not df_j['SPY'].eq(0).all():
                 start_spy = df_j.loc[df_j['SPY'] > 0, 'SPY'].iloc[0]
                 df_j['SPY_Bench'] = ((df_j['SPY'] / start_spy) - 1) * 100
            if 'Nasdaq' in df_j.columns and not df_j['Nasdaq'].eq(0).all():
                 start_ndx = df_j.loc[df_j['Nasdaq'] > 0, 'Nasdaq'].iloc[0]
                 df_j['NDX_Bench'] = ((df_j['Nasdaq'] / start_ndx) - 1) * 100

            # YTD Calculation (Geometric Linking)
            curr_year = datetime.now().year
            df_ytd = df_j[df_j['Day'].dt.year == curr_year].copy()
            
            ytd_val = 0.0
            if not df_ytd.empty:
                # Precision Math: Product of (1+r) - 1
                ytd_val = ((1 + df_ytd['Daily_Pct']).prod() - 1) * 100
            
            # --- 4. TOP DISPLAY ---
            c1, c2, c3, c4 = st.columns(4)
            
            daily_dol = df_j['Daily $ Change'].iloc[-1]
            daily_pct_display = df_j['Daily_Pct'].iloc[-1] * 100
            
            c1.metric("Net Liq Value", fmt_money(curr_nlv), f"{daily_dol:+,.2f} ({daily_pct_display:+.2f}%)")
            c2.metric("LTD Return", f"{df_j['LTD_Pct'].iloc[-1]:.2f}%")
            c3.metric("YTD Return", f"{ytd_val:.2f}%") # Matches Period Review perfectly now
            
            limit = 12
            delta_msg = f"{num_open_pos}/{limit} Pos | Risk: {risk_pct:.2f}%"
            mode = "normal" if num_open_pos <= limit else "inverse"
            c4.metric("Live Exposure", f"{calc_exposure_pct:.1f}%", delta=delta_msg, delta_color=mode)
            
            st.markdown("---")
            
            # --- 5. PLOTS ---
            df_j['EC_8EMA'] = df_j['LTD_Pct'].ewm(span=8, adjust=False).mean()
            df_j['EC_21EMA'] = df_j['LTD_Pct'].ewm(span=21, adjust=False).mean()
            df_j['EC_50SMA'] = df_j['LTD_Pct'].rolling(window=50).mean()
            if 'Nasdaq' in df_j.columns: df_j['NDX_21EMA'] = df_j['Nasdaq'].ewm(span=21, adjust=False).mean()
            
            plt.style.use('bmh')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Top Plot
            if 'Nasdaq' in df_j.columns:
                ax1.fill_between(df_j['Day'], 0.97, 1.0, transform=ax1.transAxes, where=(df_j['Nasdaq']>=df_j['NDX_21EMA']), color='green', alpha=0.4, zorder=0)
                ax1.fill_between(df_j['Day'], 0.97, 1.0, transform=ax1.transAxes, where=(df_j['Nasdaq']<df_j['NDX_21EMA']), color='red', alpha=0.4, zorder=0)
                ax1.text(0.5, 0.985, "MARKET TREND (COMP vs 21e)", transform=ax1.transAxes, ha='center', fontsize=8, fontweight='bold')
            
            lbl_port = f"Portfolio ({df_j['LTD_Pct'].iloc[-1]:+.1f}%)"
            if 'SPY_Bench' in df_j.columns:
                lbl_spy = f"SPY ({df_j['SPY_Bench'].iloc[-1]:+.1f}%)"
                ax1.plot(df_j['Day'], df_j['SPY_Bench'], color='gray', linestyle='-', linewidth=1.5, alpha=0.7, label=lbl_spy)
            if 'NDX_Bench' in df_j.columns:
                lbl_ndx = f"Nasdaq ({df_j['NDX_Bench'].iloc[-1]:+.1f}%)"
                ax1.plot(df_j['Day'], df_j['NDX_Bench'], color='#1f77b4', linestyle='-', linewidth=1.5, alpha=0.7, label=lbl_ndx)
            
            ax1.plot(df_j['Day'], df_j['LTD_Pct'], color='darkblue', linewidth=2.5, label=lbl_port)
            ax1.plot(df_j['Day'], df_j['EC_8EMA'], color='purple', linewidth=1.2, label='8 EMA')
            ax1.plot(df_j['Day'], df_j['EC_21EMA'], color='green', linewidth=1.2, label='21 EMA')
            ax1.plot(df_j['Day'], df_j['EC_50SMA'], color='red', linewidth=1.2, label='50 SMA')
            
            ax1.fill_between(df_j['Day'], df_j['LTD_Pct'], df_j['EC_21EMA'], where=(df_j['LTD_Pct'] >= df_j['EC_21EMA']), interpolate=True, color='green', alpha=0.15)
            ax1.fill_between(df_j['Day'], df_j['LTD_Pct'], df_j['EC_21EMA'], where=(df_j['LTD_Pct'] < df_j['EC_21EMA']), interpolate=True, color='red', alpha=0.15)
            ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
            ax1.set_title("Equity Curve (LTD)")
            ax1.set_ylabel("Return %")

            # Right Axis (Exposure)
            ax1b = ax1.twinx()
            exp_color = '#e67e22' 
            ax1b.fill_between(df_j['Day'], df_j['% Invested'], 0, color=exp_color, alpha=0.3, label='Exposure %')
            ax1b.plot(df_j['Day'], df_j['% Invested'], color=exp_color, linewidth=1, alpha=0.6)
            ax1b.axhline(100, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
            ax1b.set_ylim(0, 1000) 
            ax1b.set_yticks([0, 100, 200]) 
            ax1b.set_ylabel("% Exposure", color=exp_color, fontsize=9)
            ax1b.tick_params(axis='y', labelcolor=exp_color)
            ax1b.grid(False) 

            # Bottom Plot
            colors = ['green' if x >= 0 else 'red' for x in df_j['Daily $ Change']]
            ax2.bar(df_j['Day'], df_j['Daily $ Change'], color=colors)
            y_min, y_max = df_j['Daily $ Change'].min(), df_j['Daily $ Change'].max()
            if y_min == y_max: y_min, y_max = -1, 1
            ax2.fill_between(df_j['Day'], y_min, y_min + (y_max-y_min)*0.05, where=(df_j['LTD_Pct'] >= df_j['EC_21EMA']), color='green', alpha=0.5)
            ax2.fill_between(df_j['Day'], y_min, y_min + (y_max-y_min)*0.05, where=(df_j['LTD_Pct'] < df_j['EC_21EMA']), color='red', alpha=0.5)
            ax2.set_title("Daily P&L ($) | Bottom Strip: Portfolio Trend")
            
            st.pyplot(fig)
            
            # --- 6. BOTTOM DECK: FORENSICS ---
            st.markdown("---")
            st.subheader("🔭 Position Forensics")
            
            tab_mon, tab_alloc, tab_sec = st.tabs(["📋 Holding Monitor", "🥧 Allocation", "🏭 Sector"])
            
            with tab_mon:
                if not df_open.empty:
                    display_cols = ['Ticker', 'Sector', 'Shares', 'Avg_Entry', 'Cur_Px', 'Stop_Loss', 'Return_Pct', 'R_Dol']
                    final_cols = [c for c in display_cols if c in df_open.columns]
                    
                    def color_risk(val):
                        if val > (curr_nlv * 0.01): return 'color: #d62728' 
                        elif val > 0: return 'color: #ff7f0e' 
                        else: return 'color: #2ca02c' 
                    
                    st.dataframe(
                        df_open[final_cols].style.format({
                            'Avg_Entry': fmt_money, 'Cur_Px': fmt_money, 'Stop_Loss': fmt_money, 
                            'Return_Pct': '{:+.2f}%', 'R_Dol': fmt_money
                        })
                        .applymap(lambda x: 'color: #2ca02c' if x > 0 else 'color: #ff4b4b', subset=['Return_Pct'])
                        .applymap(color_risk, subset=['R_Dol'])
                        , use_container_width=True
                    )
                else: st.info("No open positions.")
                
            with tab_alloc:
                if not df_open.empty:
                    import plotly.express as px
                    total_exp = df_open['Mkt_Val'].sum()
                    cash_val = curr_nlv - total_exp
                    alloc_data = df_open[['Ticker', 'Mkt_Val']].copy()
                    
                    if cash_val > 1:
                        new_row = pd.DataFrame([{'Ticker': 'CASH', 'Mkt_Val': cash_val}])
                        alloc_data = pd.concat([alloc_data, new_row], ignore_index=True)
                    
                    fig_pie = px.pie(alloc_data, values='Mkt_Val', names='Ticker', title=f"Capital Allocation", hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else: st.success("100% CASH")

            with tab_sec:
                if not df_open.empty:
                    import plotly.express as px
                    sec_data = df_open.groupby('Sector')['Mkt_Val'].sum().reset_index()
                    total_exp = df_open['Mkt_Val'].sum()
                    cash_val = curr_nlv - total_exp
                    if cash_val > 1:
                        new_row = pd.DataFrame([{'Sector': 'CASH', 'Mkt_Val': cash_val}])
                        sec_data = pd.concat([sec_data, new_row], ignore_index=True)

                    fig_sec = px.pie(sec_data, values='Mkt_Val', names='Sector', title="Sector Exposure", hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism)
                    st.plotly_chart(fig_sec, use_container_width=True)
                else: st.info("No positions to analyze.")

        else: st.info("Data loaded but empty rows.")

# ==============================================================================
# PAGE 3: DAILY JOURNAL (CLEAN & FINAL)
# ==============================================================================
elif page == "Daily Journal":
    st.header(f"DAILY TRADING JOURNAL ({CURR_PORT_NAME})")
    
    TARGET_FILE = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')
    if not os.path.exists(TARGET_FILE):
        TARGET_FILE = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal.csv')

    if not os.path.exists(TARGET_FILE):
        st.warning("No Journal File Found. Please go to 'Daily Routine' to log your first day.")
    else:
        df_j = pd.read_csv(TARGET_FILE)
        df_d = load_data(DETAILS_FILE)
        
        # --- 0. DATA SANITIZATION ---
        if not df_j.empty:
            df_j['Day'] = pd.to_datetime(df_j['Day'], errors='coerce')
            numeric_cols = ['End NLV', 'Beg NLV', 'Cash -/+', 'Daily $ Change', 'SPY', 'Nasdaq']
            for c in numeric_cols:
                if c in df_j.columns:
                    df_j[c] = pd.to_numeric(df_j[c].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0.0)

        # --- 1. MISSING ENTRY CHECK ---
        if not df_d.empty:
            cutoff = datetime(2025, 11, 21).date()
            valid_dates = pd.to_datetime(df_d['Date'], errors='coerce').dropna().dt.date.unique()
            trade_days = sorted([d for d in valid_dates if d >= cutoff], reverse=True)
            journal_days = df_j['Day'].dropna().dt.date.unique() if not df_j.empty else []
            missing = [d for d in trade_days if d not in journal_days]
            if missing: 
                missing_str = ', '.join([d.strftime('%m/%d') for d in missing[:5]])
                st.warning(f"⚠️ Missing Journal Entries for: {missing_str}")

        tab_view, tab_manage = st.tabs(["View Logs", "Manage Logs"])
        
        # --- TAB 1: VIEW LOGS ---
        with tab_view:
            if not df_j.empty:
                view_opt = st.radio("Filter View", ["Current Week", "By Month", "All History"], horizontal=True)
                
                df_calc = df_j.copy().dropna(subset=['Day']).sort_values('Day', ascending=True)
                
                # Math Fix for Viewer
                if 'Daily % Change' in df_calc.columns:
                    df_calc['Daily_Pct'] = pd.to_numeric(df_calc['Daily % Change'].astype(str).str.replace('%', '', regex=False), errors='coerce').fillna(0.0)
                else:
                    denom = df_calc['Beg NLV'] + df_calc['Cash -/+']
                    df_calc['Daily_Pct'] = 0.0
                    mask = denom != 0
                    df_calc.loc[mask, 'Daily_Pct'] = (df_calc['End NLV'] - denom) / denom * 100
                
                df_calc['SPY_Pct'] = df_calc['SPY'].pct_change() * 100
                df_calc['Nasdaq_Pct'] = df_calc['Nasdaq'].pct_change() * 100
                
                if view_opt == "Current Week":
                    today = datetime.now().date()
                    start_week = today - timedelta(days=today.weekday()) 
                    df_view = df_calc[df_calc['Day'].dt.date >= start_week]
                elif view_opt == "By Month":
                    df_calc['Month_Str'] = df_calc['Day'].dt.strftime('%B %Y')
                    months = sorted(df_calc['Month_Str'].unique().tolist(), key=lambda x: datetime.strptime(x, '%B %Y'), reverse=True)
                    sel_month = st.selectbox("Select Month", months) if months else None
                    df_view = df_calc[df_calc['Month_Str'] == sel_month] if sel_month else df_calc
                else:
                    df_view = df_calc

                # --- SHOW COLUMNS ---
                show_cols = [
                    'Day', 'End NLV', 'Daily_Pct', 
                    'SPY', 'SPY_Pct', 'Nasdaq', 'Nasdaq_Pct', 
                    'Market_Notes',   # Global Context
                    'Market_Action',  # Portfolio Moves
                    'Score'
                ]
                valid_cols = [c for c in show_cols if c in df_view.columns]
                
                st.dataframe(
                    df_view.sort_values('Day', ascending=False)[valid_cols]
                    .style.format({
                        'Day': '{:%m/%d/%y}', 'End NLV': '${:,.2f}', 'Daily_Pct': '{:+.2f}%', 
                        'Score': '{:.0f}', 'SPY': '{:.2f}', 'SPY_Pct': '{:+.2f}%', 'Nasdaq': '{:.2f}', 'Nasdaq_Pct': '{:+.2f}%'
                    })
                    .applymap(color_pnl, subset=[c for c in ['Daily_Pct', 'SPY_Pct', 'Nasdaq_Pct'] if c in df_view.columns]) 
                    .applymap(color_score, subset=['Score']), 
                    hide_index=True, use_container_width=True
                )
            else: st.info("No journal entries found.")

        # --- TAB 2: MANAGE LOGS ---
        with tab_manage:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.subheader("Delete Incorrect Entries")
                if not df_j.empty:
                    df_j_del = df_j.dropna(subset=['Day']).sort_values('Day', ascending=False).copy()
                    if not df_j_del.empty:
                        options = [f"{row['Day'].strftime('%Y-%m-%d')} | End NLV: ${float(row['End NLV']):,.2f}" for i, row in df_j_del.iterrows()]
                        sel_del = st.selectbox("Select Log to Delete", options)
                        if st.button("DELETE ENTRY"):
                            date_to_del = sel_del.split("|")[0].strip()
                            df_j = df_j[df_j['Day'].dt.strftime('%Y-%m-%d') != date_to_del]
                            secure_save(df_j, TARGET_FILE); st.success(f"Deleted entry for {date_to_del}"); st.rerun()
            
            with col_m2:
                st.subheader("Repair Market Data")
                force_update = st.checkbox("Force Overwrite Existing Data")
                if st.button("RUN MARKET DATA SYNC"):
                    if not df_j.empty:
                        try:
                            start_d = df_j['Day'].min(); end_d = df_j['Day'].max() + timedelta(days=1)
                            spy_hist = yf.Ticker("SPY").history(start=start_d, end=end_d)['Close'].tz_localize(None)
                            ndx_hist = yf.Ticker("^IXIC").history(start=start_d, end=end_d)['Close'].tz_localize(None)
                            
                            for idx, row in df_j.iterrows():
                                d = row['Day'].normalize().replace(tzinfo=None)
                                if force_update or row['SPY'] == 0:
                                    if d in spy_hist.index: df_j.at[idx, 'SPY'] = spy_hist.loc[d]
                                if force_update or row['Nasdaq'] == 0:
                                    if d in ndx_hist.index: df_j.at[idx, 'Nasdaq'] = ndx_hist.loc[d]
                            
                            secure_save(df_j, TARGET_FILE); st.success("✅ Sync Complete!"); st.rerun()
                        except Exception as e: st.error(f"Sync Error: {e}")

# ==============================================================================
# PAGE 4: M FACTOR (MARKET HEALTH) - VISUAL FIX
# ==============================================================================
elif page == "M Factor":
    st.header(f"MARKET HEALTH (M FACTOR)")
    # CSS Styles
    st.markdown("""<style>.market-banner {padding: 20px; border-radius: 12px; text-align: center; color: white; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);} .ticker-box {background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 10px; color: black;} .metric-row {display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 15px; color: #333; border-bottom: 1px dashed #eee; padding-bottom: 8px;} .metric-row:last-child {border-bottom: none; margin-bottom: 0; padding-bottom: 0;} .sub-text {font-size: 12px; color: #999; font-weight: 400; margin-left: 5px;}</style>""", unsafe_allow_html=True)
    
    if st.button("Refresh Market Data"): st.cache_data.clear()

    def get_market_state(ticker):
        try:
            # 1. FETCH DATA
            df = yf.Ticker(ticker).history(period="2y")
            if df.empty: return None
            
            # 2. CALCULATE INDICATORS
            df['21EMA'] = df['Close'].ewm(span=21, adjust=False).mean()
            df['50SMA'] = df['Close'].rolling(window=50).mean()
            df['200SMA'] = df['Close'].rolling(window=200).mean()
            df['Prev_Close'] = df['Close'].shift(1)
            df['Is_Up'] = df['Close'] > df['Prev_Close']
            
            def pct_diff(a, b): return ((a - b) / b) * 100
            
            # --- HELPER: CALCULATE STREAK FOR ANY MA ---
            def calc_streak(ma_col):
                curr_close = df['Close'].iloc[-1]
                curr_ma = df[ma_col].iloc[-1]
                is_above = curr_close > curr_ma
                count = 0
                for i in range(len(df)-1, -1, -1):
                    c = df['Close'].iloc[i]; m = df[ma_col].iloc[i]
                    if is_above:
                        if c > m: count += 1
                        else: break
                    else:
                        if c < m: count += 1
                        else: break
                return count

            streak_21 = calc_streak('21EMA')
            streak_50 = calc_streak('50SMA')
            streak_200 = calc_streak('200SMA')
            
            # 3. DETERMINE STATE (LOOP)
            subset = df.iloc[-60:].copy() 
            state = "OPEN" 
            setup_low_21 = None; setup_low_50 = None; pt_streak = 0
            
            for date, row in subset.iterrows():
                close = row['Close']; low = row['Low']
                ema21 = row['21EMA']; sma50 = row['50SMA']
                is_up = row['Is_Up']
                
                if low > ema21: pt_streak += 1
                else: pt_streak = 0
                
                if close < sma50:
                    if setup_low_50 is None: setup_low_50 = low
                    elif low < (setup_low_50 * 0.998): state = "CLOSED"
                else:
                    setup_low_50 = None
                    if state == "CLOSED": state = "NEUTRAL"
                
                if state != "CLOSED":
                    if close < ema21:
                        if state == "POWERTREND":
                            state = "NEUTRAL"; setup_low_21 = low; pt_streak = 0
                        else:
                            if setup_low_21 is None: setup_low_21 = low
                            elif low < (setup_low_21 * 0.998): state = "NEUTRAL"
                    else:
                        setup_low_21 = None
                        if state == "NEUTRAL": state = "OPEN"
                        if state in ["OPEN", "POWERTREND"]:
                            if pt_streak >= 3 and is_up: state = "POWERTREND"
                            elif state == "POWERTREND" and pt_streak == 0: state = "OPEN"

            curr = subset.iloc[-1]
            return {
                'price': curr['Close'], 'state': state, 
                'ema21': curr['21EMA'], 'd21': pct_diff(curr['Close'], curr['21EMA']), 
                'sma50': curr['50SMA'], 'd50': pct_diff(curr['Close'], curr['50SMA']), 
                'sma200': curr['200SMA'], 'd200': pct_diff(curr['Close'], curr['200SMA']),
                's21': streak_21, 's50': streak_50, 's200': streak_200
            }
        except: return None

    nasdaq = get_market_state("^IXIC")
    spy = get_market_state("SPY")

    if nasdaq and spy:
        status = nasdaq['state']
        
        if status == "POWERTREND": bg = "#8A2BE2"; exp = "200% (Margin Enabled)"
        elif status == "OPEN": bg = "#2ca02c"; exp = "100% (Full)"
        elif status == "NEUTRAL": bg = "#ffcc00"; exp = "Max 50% (CAUTION)"
        else: bg = "#ff4b4b"; exp = "0% (Cash / Defensive)"

        st.markdown(f"""<div class="market-banner" style="background-color: {bg};"><div style="font-size: 14px; opacity: 0.9;">MARKET WINDOW</div><div style="font-size: 48px; font-weight: 800; margin: 5px 0;">{status}</div><div style="font-size: 16px;">RECOMMENDED EXPOSURE: {exp}</div></div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        
        def make_card_html(title, d):
            def arr(v): return "⬆" if v>0 else "⬇"
            def col(v): return "#2ca02c" if v>0 else "#ff4b4b"
            
            # --- FLATTENED STRING TO PREVENT MARKDOWN CODE BLOCK ERROR ---
            html = f"""<div class="ticker-box"><div style="display:flex; justify-content:space-between; border-bottom: 2px solid #f0f0f0; padding-bottom:15px; margin-bottom:15px;"><span style="font-size:20px; font-weight:700;">{title}</span><span style="font-size:20px; color:#555;">${d['price']:,.2f}</span></div><div class="metric-row"><div><span style="font-weight:600;">Short (21e)</span> <span class="sub-text">({d['ema21']:,.2f})</span></div><div style="font-weight:700; color:{col(d['d21'])};">{arr(d['d21'])} {d['s21']} <span style="font-size:13px; opacity:0.8;">({abs(d['d21']):.2f}%)</span></div></div><div class="metric-row"><div><span style="font-weight:600;">Med (50s)</span> <span class="sub-text">({d['sma50']:,.2f})</span></div><div style="font-weight:700; color:{col(d['d50'])};">{arr(d['d50'])} {d['s50']} <span style="font-size:13px; opacity:0.8;">({abs(d['d50']):.2f}%)</span></div></div><div class="metric-row"><div><span style="font-weight:600;">Long (200s)</span> <span class="sub-text">({d['sma200']:,.2f})</span></div><div style="font-weight:700; color:{col(d['d200'])};">{arr(d['d200'])} {d['s200']} <span style="font-size:13px; opacity:0.8;">({abs(d['d200']):.2f}%)</span></div></div></div>"""
            return html
            
        with c1: st.markdown(make_card_html("NASDAQ", nasdaq), unsafe_allow_html=True)
        with c2: st.markdown(make_card_html("S&P 500", spy), unsafe_allow_html=True)
    else: st.error("Market Data Unavailable")

# ==============================================================================
# PAGE 4: RISK MANAGER (VISUAL HARD DECKS)
# ==============================================================================
elif page == "Risk Manager":
    st.header(f"RISK MANAGEMENT ({CURR_PORT_NAME})")
    
    # --- CONFIGURATION ---
    RESET_DATE = pd.Timestamp("2025-12-16")
    
    # 1. ROBUST LOAD (Same as Command Center)
    def clean_num_local(x):
        try:
            if isinstance(x, str):
                return float(x.replace('$', '').replace(',', '').replace('%', '').strip())
            return float(x)
        except: return 0.0

    p_path = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')
    
    if os.path.exists(p_path):
        df = pd.read_csv(p_path)
        
        # Clean Data
        if not df.empty and 'Day' in df.columns:
            df['Day'] = pd.to_datetime(df['Day'], errors='coerce')
            df.sort_values('Day', inplace=True) # Ascending for graph
            
            for c in ['End NLV', 'Beg NLV', 'Cash -/+']: 
                if c in df.columns: df[c] = df[c].apply(clean_num_local)
            
            # 2. FILTER FOR RESET DATE
            # We only look at performance AFTER the split to calculate the Peak
            df_active = df[df['Day'] >= RESET_DATE].copy()
            
            if not df_active.empty:
                curr_nlv = df_active['End NLV'].iloc[-1]
                
                # 3. CALCULATE PEAK (High Water Mark)
                # The highest closing balance achieved since the reset
                peak_nlv = df_active['End NLV'].max()
                
                # Drawdown Amount
                dd_dol = peak_nlv - curr_nlv
                dd_pct = (dd_dol / peak_nlv) * 100 if peak_nlv > 0 else 0.0
                
                # 4. DEFINE HARD DECKS (Dollar Levels)
                # Level 1: -5%
                deck_l1 = peak_nlv * 0.95
                # Level 2: -7%
                deck_l2 = peak_nlv * 0.93
                # Level 3: -10%
                deck_l3 = peak_nlv * 0.90
                
                # Distance to triggers
                dist_l1 = curr_nlv - deck_l1
                
                # --- DISPLAY: HEADS UP DISPLAY ---
                st.subheader(f"Current Status: ${curr_nlv:,.2f}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Peak (HWM)", f"${peak_nlv:,.2f}")
                col1.caption(f"Since {RESET_DATE.strftime('%m/%d/%y')}")
                
                col2.metric("Current Drawdown", f"-{dd_pct:.2f}%", f"-${dd_dol:,.2f}", delta_color="inverse")
                
                status_txt = "🟢 GREEN LIGHT"
                if dd_pct >= 10: status_txt = "☠️ RED LIGHT (CASH)"
                elif dd_pct >= 7: status_txt = "🟠 ORANGE LIGHT (30% MAX)"
                elif dd_pct >= 5: status_txt = "🟡 YELLOW LIGHT (NO MARGIN)"
                
                col3.metric("Risk State", status_txt)
                if dd_pct < 5:
                    col3.caption(f"Buffer: ${dist_l1:,.0f} to Level 1")

                st.markdown("---")

                # --- VISUALIZATION: THE HARD DECK CHART ---
                st.subheader("📉 The Hard Deck")
                
                # Prepare data for plotting
                # We want to show the NLV vs the 3 Trailing Stop Lines
                dates = df_active['Day']
                nlvs = df_active['End NLV']
                
                # Calculate the HWM series (Trailing Peak)
                hwm_series = df_active['End NLV'].cummax()
                
                # Calculate the Fuse Lines based on the HWM series
                l1_series = hwm_series * 0.95
                l2_series = hwm_series * 0.93
                l3_series = hwm_series * 0.90
                
                plt.style.use('bmh')
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot Lines
                ax.plot(dates, nlvs, color='black', linewidth=2.5, label='Net Liquidity')
                ax.plot(dates, hwm_series, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Peak (HWM)')
                
                # Plot Hard Decks
                ax.plot(dates, l1_series, color='#f1c40f', linewidth=1.5, alpha=0.8, label='L1: Caution (-5%)') # Yellow
                ax.plot(dates, l2_series, color='#e67e22', linewidth=1.5, alpha=0.8, label='L2: Defense (-7%)') # Orange
                ax.plot(dates, l3_series, color='#c0392b', linewidth=2, alpha=0.8, label='L3: Cash (-10%)')   # Red
                
                # Fill Zones
                ax.fill_between(dates, hwm_series, l1_series, color='green', alpha=0.05) # Safe Zone
                ax.fill_between(dates, l1_series, l2_series, color='yellow', alpha=0.1)  # Caution Zone
                ax.fill_between(dates, l2_series, l3_series, color='orange', alpha=0.1)  # Defense Zone
                ax.fill_between(dates, l3_series, min(l3_series.min(), nlvs.min()) * 0.98, color='red', alpha=0.05) # Danger Zone
                
                ax.set_title(f"Risk Levels relative to Peak (Dynamic)")
                ax.set_ylabel("Account Value ($)")
                ax.legend(loc='upper left')
                
                st.pyplot(fig)
                
                # --- FUSE BOX INSTRUCTIONS ---
                st.markdown("### 🧨 Fuse Box Protocols")
                
                f1, f2, f3 = st.columns(3)
                
                # FUSE 1
                f1.markdown("#### 🟡 LEVEL 1")
                f1.markdown(f"**Trigger:** -5% DD (**${deck_l1:,.0f}**)")
                if curr_nlv <= deck_l1:
                    f1.error("❌ FUSE BLOWN")
                else:
                    f1.success("✅ SECURE")
                f1.info("**Action:** Remove Margin.\n\nLockout New Buys until steady.")

                # FUSE 2
                f2.markdown("#### 🟠 LEVEL 2")
                f2.markdown(f"**Trigger:** -7% DD (**${deck_l2:,.0f}**)")
                if curr_nlv <= deck_l2:
                    f2.error("❌ FUSE BLOWN")
                else:
                    f2.success("✅ SECURE")
                f2.warning("**Action:** Max 30% Invested.\n\nManage winners only. Cut loose ends.")

                # FUSE 3
                f3.markdown("#### ☠️ LEVEL 3")
                f3.markdown(f"**Trigger:** -10% DD (**${deck_l3:,.0f}**)")
                if curr_nlv <= deck_l3:
                    f3.error("❌ FUSE BLOWN")
                else:
                    f3.success("✅ SECURE")
                f3.error("**Action:** GO TO CASH.\n\nProtection Mode. No trading for 48hrs.")

            else:
                st.info(f"No data available after the reset date ({RESET_DATE.strftime('%Y-%m-%d')}).")
        else:
            st.warning("Journal file is empty or missing 'Day' column.")
    else:
        st.error("No Journal File Found.")

elif page == "Ticker Forensics":
    st.header("TICKER FORENSICS")
    if os.path.exists(SUMMARY_FILE):
        df_s = load_data(SUMMARY_FILE); closed = df_s[df_s['Status']=='CLOSED'].copy()
        if not closed.empty:
            st.subheader("1. Ticker Leaderboard (Top 20 Movers)")
            ticker_stats = closed.groupby('Ticker').agg(Total_PL=('Realized_PL', 'sum'), Trade_Count=('Trade_ID', 'count'), Win_Rate=('Realized_PL', lambda x: (x>0).mean())).sort_values('Total_PL', ascending=True)
            top_movers = pd.concat([ticker_stats.head(10), ticker_stats.tail(10)]); top_movers = top_movers[~top_movers.index.duplicated()]
            fig, ax = plt.subplots(figsize=(10, 8)); colors = ['#2ca02c' if x >= 0 else '#ff4b4b' for x in top_movers['Total_PL']]
            top_movers['Total_PL'].plot(kind='barh', ax=ax, color=colors); ax.axvline(0, color='black', linewidth=1); ax.set_title("Total P&L by Ticker"); ax.set_xlabel("P&L ($)"); st.pyplot(fig)
            st.dataframe(ticker_stats.sort_values('Total_PL', ascending=False).style.format({'Total_PL': '${:,.2f}', 'Win_Rate': '{:.1%}'}).applymap(color_pnl, subset=['Total_PL']))
            st.markdown("---"); st.subheader("2. Trade Distribution (Bell Curve)")
            selected_ticker = st.selectbox("Filter Distribution", ["All"] + sorted(closed['Ticker'].unique().tolist()))
            dist_data = closed if selected_ticker == "All" else closed[closed['Ticker'] == selected_ticker]
            if not dist_data.empty:
                fig2, ax2 = plt.subplots(figsize=(10, 5)); n, bins, patches = ax2.hist(dist_data['Realized_PL'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                for bin_val, patch in zip(bins, patches): patch.set_facecolor('#ff4b4b' if bin_val < 0 else '#2ca02c'); patch.set_alpha(0.6)
                ax2.axvline(0, color='black', linewidth=2, linestyle='--'); ax2.set_title(f"P&L Distribution ({selected_ticker})"); ax2.set_xlabel("P&L ($)"); st.pyplot(fig2)
                c1, c2, c3 = st.columns(3); c1.metric("Average Trade", f"${dist_data['Realized_PL'].mean():,.2f}"); c2.metric("Median Trade", f"${dist_data['Realized_PL'].median():,.2f}"); c3.metric("Skew", f"{dist_data['Realized_PL'].skew():.2f}")
            else: st.warning("No trades.")
        else: st.info("No closed trades.")
    else: st.error("Summary file missing.")

# ==============================================================================
# PAGE 7: PERIOD REVIEW (MATH FIXED TO MATCH DASHBOARD)
# ==============================================================================
elif page == "Period Review":
    st.header("PERIODIC REVIEW")
    
    # 1. SCOPE SELECTOR
    review_scope = st.radio("Review Scope", ["⚔️ Combined Core (CanSlim + TQQQ)", f"🌳 {PORT_CANSLIM}", f"⚡ {PORT_TQQQ}"], horizontal=True)
    
    # --- DATA ENGINE ---
    def clean_num_local(x):
        try:
            if isinstance(x, str):
                return float(x.replace('$', '').replace(',', '').replace('%', '').strip())
            return float(x)
        except: return 0.0

    def get_df(p_name):
        path = os.path.join(DATA_ROOT, p_name, 'Trading_Journal_Clean.csv')
        summ_path = os.path.join(DATA_ROOT, p_name, 'Trade_Log_Summary.csv')
        
        if os.path.exists(path):
            d = pd.read_csv(path)
            if not d.empty and 'Day' in d.columns:
                d['Day'] = pd.to_datetime(d['Day'], errors='coerce')
                d = d.dropna(subset=['Day']).sort_values('Day')
                for c in ['Beg NLV', 'End NLV', 'Cash -/+', 'Daily $ Change', 'SPY', 'Nasdaq']:
                    if c in d.columns: d[c] = d[c].apply(clean_num_local)
        else: d = pd.DataFrame()
        
        if os.path.exists(summ_path):
            s = pd.read_csv(summ_path)
        else: s = pd.DataFrame()
        
        return d, s

    # LOAD DATA
    if "Combined" in review_scope:
        df_c, s_c = get_df(PORT_CANSLIM)
        df_t, s_t = get_df(PORT_TQQQ)
        
        if not df_c.empty or not df_t.empty:
            dates_c = df_c['Day'].tolist() if not df_c.empty else []
            dates_t = df_t['Day'].tolist() if not df_t.empty else []
            all_dates = sorted(list(set(dates_c + dates_t)))
            
            def reindex_port(df, dates):
                if df.empty: return pd.DataFrame(index=dates, columns=['End NLV', 'Beg NLV', 'Cash -/+', 'Daily $ Change']).fillna(0)
                df = df.set_index('Day').reindex(dates)
                df['End NLV'] = df['End NLV'].fillna(method='ffill').fillna(0)
                df['Beg NLV'] = df['Beg NLV'].fillna(method='ffill').fillna(0)
                df['Cash -/+'] = df['Cash -/+'].fillna(0)
                df['Daily $ Change'] = df['Daily $ Change'].fillna(0)
                df['Beg NLV'] = df['End NLV'] - df['Daily $ Change'] - df['Cash -/+']
                return df

            rc = reindex_port(df_c, all_dates)
            rt = reindex_port(df_t, all_dates)
            
            df_j = pd.DataFrame(index=all_dates)
            df_j['Day'] = pd.to_datetime(all_dates)
            df_j['End NLV'] = rc['End NLV'] + rt['End NLV']
            df_j['Beg NLV'] = rc['Beg NLV'] + rt['Beg NLV']
            df_j['Cash -/+'] = rc['Cash -/+'] + rt['Cash -/+']
            df_j['Daily $ Change'] = rc['Daily $ Change'] + rt['Daily $ Change']
            
            if not df_c.empty:
                bench = df_c.set_index('Day')[['SPY', 'Nasdaq']].reindex(all_dates).fillna(method='ffill')
                df_j['SPY'] = bench['SPY'].values
                df_j['Nasdaq'] = bench['Nasdaq'].values
            else:
                df_j['SPY'] = 0; df_j['Nasdaq'] = 0
        else: df_j = pd.DataFrame()
        df_s = pd.concat([s_c, s_t], ignore_index=True)
        
    else:
        target_port = PORT_CANSLIM if "CanSlim" in review_scope else PORT_TQQQ
        df_j, df_s = get_df(target_port)

    # --- RENDER ENGINE ---
    if not df_j.empty:
        # 1. CALCULATE DAILY TWR
        df_j['Adjusted_Beg'] = df_j['Beg NLV'] + df_j['Cash -/+']
        df_j['Daily_Return'] = 0.0
        mask = df_j['Adjusted_Beg'] != 0
        df_j.loc[mask, 'Daily_Return'] = (df_j.loc[mask, 'End NLV'] - df_j.loc[mask, 'Adjusted_Beg']) / df_j.loc[mask, 'Adjusted_Beg']
        
        # 2. CALCULATE EQUITY CURVES
        df_j['TWR_Curve'] = (1 + df_j['Daily_Return']).cumprod()
        df_j['Portfolio_LTD'] = (df_j['TWR_Curve'] - 1) * 100
        
        if 'SPY' in df_j.columns:
            df_j['SPY_Pct'] = df_j['SPY'].pct_change().fillna(0)
            df_j['SPY_LTD'] = ((1 + df_j['SPY_Pct']).cumprod() - 1) * 100
        
        if 'Nasdaq' in df_j.columns:
            df_j['NDX_Pct'] = df_j['Nasdaq'].pct_change().fillna(0)
            df_j['NDX_LTD'] = ((1 + df_j['NDX_Pct']).cumprod() - 1) * 100

        tab_w, tab_m, tab_y = st.tabs(["Weekly Review", "Monthly Review", "Annual & CAGR"])
        
        def render_period_review(mode, df_source):
            resample_code = 'W-FRI' if mode == "Weekly" else 'ME'
            if mode == "Annual": resample_code = 'YE'
            
            # --- CRITICAL MATH FIX: GEOMETRIC LINKING ---
            # We calculate Period Return by linking daily returns: (1+r1)*(1+r2)... - 1
            # This is TWR (Time Weighted Return) and immune to withdrawals.
            df_p = df_source.set_index('Day').resample(resample_code).agg({
                'Beg NLV': 'first', 'End NLV': 'last', 
                'Daily $ Change': 'sum', 'Cash -/+': 'sum', 
                'Portfolio_LTD': 'last', 'SPY_LTD': 'last', 'NDX_LTD': 'last',
                'Daily_Return': lambda x: (1 + x).prod() - 1  # <--- THE FIX
            }).dropna()
            
            if df_p.empty:
                st.info(f"Not enough data for {mode} review.")
                return

            df_p.rename(columns={'Daily_Return': 'Period Return %'}, inplace=True)
            df_p['Period Return %'] = df_p['Period Return %'] * 100 # Convert to %
            df_p['Period P&L ($)'] = df_p['End NLV'] - (df_p['Beg NLV'] + df_p['Cash -/+'])
            
            df_p['MA_10'] = df_p['Portfolio_LTD'].rolling(window=10).mean()
            df_table = df_p.sort_index(ascending=False)
            
            # Latest Metrics
            latest = df_table.iloc[0]
            curr_date = latest.name
            
            if mode == "Weekly": p_start = curr_date - timedelta(days=6)
            elif mode == "Monthly": p_start = curr_date.replace(day=1)
            else: p_start = curr_date.replace(month=1, day=1)
            p_end = curr_date
            
            count = 0; wr = 0.0
            if not df_s.empty and 'Closed_Date' in df_s.columns:
                df_s['Closed_Date'] = pd.to_datetime(df_s['Closed_Date'], errors='coerce')
                mask_t = (df_s['Status'] == 'CLOSED') & (df_s['Closed_Date'] >= p_start) & (df_s['Closed_Date'] <= p_end + timedelta(days=1))
                period_trades = df_s[mask_t]
                count = len(period_trades)
                wins = len(period_trades[period_trades['Realized_PL'] > 0])
                wr = wins/count if count > 0 else 0.0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Latest {mode} P&L", f"${latest['Period P&L ($)']:,.2f}", delta_color="normal")
            c2.metric(f"{mode} Return %", f"{latest['Period Return %']:.2f}%") # Matches Dashboard
            c3.metric("Trades Closed", count)
            c4.metric("Win Rate", f"{wr:.1%}")
            
            st.markdown("---")
            
            # PLOT 1: Performance
            st.subheader(f"{mode} Performance vs Benchmark")
            fig, ax = plt.subplots(figsize=(12, 6))
            last_port = df_p['Portfolio_LTD'].iloc[-1]
            
            ax.plot(df_p.index, df_p['Portfolio_LTD'], label=f"Portfolio ({last_port:+.1f}%)", color="#1f77b4", linewidth=2.5)
            if 'SPY_LTD' in df_p.columns:
                ax.plot(df_p.index, df_p['SPY_LTD'], label="S&P 500", color="gray", alpha=0.6, linewidth=1.5)
            if 'NDX_LTD' in df_p.columns:
                ax.plot(df_p.index, df_p['NDX_LTD'], label="Nasdaq", color="orange", alpha=0.6, linewidth=1.5)
            
            ax.set_ylabel("Total Return (%)"); ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # PLOT 2: Net P&L
            st.subheader(f"Net {mode} P&L ($)")
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            colors = ['#2ca02c' if x >= 0 else '#ff4b4b' for x in df_p['Period P&L ($)']]
            w_map = {"Weekly": 2, "Monthly": 20, "Annual": 200}
            ax2.bar(df_p.index, df_p['Period P&L ($)'], color=colors, width=w_map.get(mode, 10))
            ax2.axhline(0, color='black', linewidth=0.5); ax2.set_ylabel("Net P&L ($)")
            st.pyplot(fig2)
            
            # TABLE
            st.subheader(f"{mode} Financial Statement")
            financials = df_table[['Beg NLV', 'Cash -/+', 'End NLV', 'Period P&L ($)', 'Period Return %', 'Portfolio_LTD']].copy()
            financials.columns = ['Start Equity', 'Cash Flow', 'End Equity', 'Net P&L ($)', 'Return % (TWR)', 'LTD Return %']
            
            fmt_str = '%Y-%m-%d'
            if mode == "Monthly": fmt_str = '%B %Y'
            elif mode == "Annual": fmt_str = '%Y'
            financials.index = financials.index.strftime(fmt_str)
            
            st.dataframe(
                financials.style.format({
                    'Start Equity': '${:,.2f}', 'Cash Flow': '${:,.2f}', 'End Equity': '${:,.2f}', 
                    'Net P&L ($)': '${:,.2f}', 'Return % (TWR)': '{:.2f}%', 'LTD Return %': '{:.2f}%'
                }).applymap(color_pnl, subset=['Net P&L ($)', 'Return % (TWR)', 'LTD Return %']),
                use_container_width=True
            )

        with tab_w: render_period_review("Weekly", df_j)
        with tab_m: render_period_review("Monthly", df_j)
        with tab_y:
            total_days = (df_j['Day'].max() - df_j['Day'].min()).days
            final_twr = df_j['TWR_Curve'].iloc[-1]
            cagr = 0.0
            if total_days > 0:
                years = total_days / 365.25
                if years > 0: cagr = (final_twr ** (1 / years)) - 1
            
            st.markdown(f"### 📈 Compound Annual Growth Rate (CAGR): **{cagr:.2%}**")
            st.caption(f"Calculated using Time-Weighted Return over {years:.1f} years.")
            st.markdown("---")
            render_period_review("Annual", df_j)

    else:
        st.info("Insufficient data to generate review.")

# ==============================================================================
# PAGE 3: DAILY ROUTINE (SEPARATE COLUMNS FOR GLOBAL VS LOCAL NOTES)
# ==============================================================================
if page == "Daily Routine":
    st.header("🌅 DAILY ROUTINE (MASTER BLOTTER)")
    st.caption("Enter End-of-Day numbers. 'Cash Added/Removed' is for deposits/withdrawals only.")

    # 1. DEFINE PATHS
    PORTFOLIO_MAP = {
        PORT_CANSLIM: os.path.join(DATA_ROOT, PORT_CANSLIM, 'Trading_Journal_Clean.csv'),
        PORT_TQQQ:    os.path.join(DATA_ROOT, PORT_TQQQ, 'Trading_Journal_Clean.csv'),
        PORT_457B:    os.path.join(DATA_ROOT, PORT_457B, 'Trading_Journal_Clean.csv')
    }

    # THE GOLDEN STANDARD (Added 'Market_Notes' for Global Context)
    MASTER_ORDER = [
        'Day', 
        'Status', 
        'Market Window', 
        '> 21e', 
        'Cash -/+', 
        'Beg NLV', 
        'End NLV', 
        'Daily $ Change', 
        'Daily % Change', 
        '% Invested', 
        'SPY', 
        'Nasdaq', 
        'Market_Notes',   # <--- NEW: Global/Market Context
        'Market_Action',  # <--- Mapped to Portfolio Specifics (Buy/Sell)
        'Keywords', 
        'Score'
    ]

    # --- HELPER: ROBUST LOADER ---
    def load_and_prep_file(path):
        if not os.path.exists(path): return pd.DataFrame(columns=MASTER_ORDER)
        try:
            df = pd.read_csv(path)
            if len(df) == 0 or 'Day' not in df.columns: return pd.DataFrame(columns=MASTER_ORDER)
            
            df['Day'] = pd.to_datetime(df['Day'], errors='coerce').dt.strftime('%Y-%m-%d')
            df = df.dropna(subset=['Day'])
            
            # Ensure all Master Columns exist (This will create 'Market_Notes' if missing)
            for c in MASTER_ORDER:
                if c not in df.columns:
                    df[c] = '' if c in ['Status','Market Window','Keywords','Market_Action','Market_Notes','Daily % Change'] else 0.0
            
            return df[MASTER_ORDER]
        except: return pd.DataFrame(columns=MASTER_ORDER)

    # 2. MASTER FORM
    with st.form("master_routine_form"):
        st.subheader("1. General Market Data")
        c1, c2, c3, c4 = st.columns(4)
        entry_date = c1.date_input("Date", datetime.now())
        entry_date_str = entry_date.strftime("%Y-%m-%d")
        
        try:
            live_spy = yf.Ticker("SPY").history(period='1d')['Close'].iloc[-1]
            live_ndx = yf.Ticker("^IXIC").history(period='1d')['Close'].iloc[-1]
        except: live_spy = 0.0; live_ndx = 0.0

        spy_val = c3.number_input("SPY Close", value=float(live_spy), format="%.2f")
        ndx_val = c4.number_input("Nasdaq Close", value=float(live_ndx), format="%.2f")
        
        # GLOBAL NOTE INPUT
        market_notes = st.text_input("Market/Global Notes", placeholder="e.g., FTD on Nasdaq? Volatility Spike? CPI Data?")
        st.markdown("---")
        
        st.subheader("2. Portfolio Updates")
        input_keys = {} 
        for p_name in [PORT_CANSLIM, PORT_TQQQ, PORT_457B]:
            do_update = st.checkbox(f"Update {p_name}?", value=True, key=f"chk_{p_name}")
            if do_update:
                c_a, c_b, c_c, c_d = st.columns([1, 1, 1, 2])
                nlv_in = c_a.number_input(f"Closing NLV ({p_name})", value=0.0, step=100.0, format="%.2f", key=f"nlv_{p_name}")
                sec_in = c_b.number_input(f"Total Holdings ({p_name})", value=0.0, step=100.0, format="%.2f", key=f"sec_{p_name}")
                cash_flow_in = c_c.number_input(f"Cash Added/Removed ({p_name})", value=0.0, step=100.0, format="%.2f", key=f"cf_{p_name}")
                
                # PORTFOLIO SPECIFIC NOTE INPUT
                note_in = c_d.text_input(f"Actions ({p_name})", key=f"note_{p_name}", placeholder="e.g. BUY: NVDA, SELL: GOOG")
                
                input_keys[p_name] = {'nlv': nlv_in, 'sec': sec_in, 'cash_flow': cash_flow_in, 'note': note_in}
            st.divider()

        force_ovr = st.checkbox("⚠️ Force Overwrite Existing Entry", help="Check this if you get a 'Skipped' error but want to save anyway.")

        if st.form_submit_button("💾 LOG SELECTED ACCOUNTS", type="primary"):
            success_count = 0
            
            for p_name, inputs in input_keys.items():
                p_path = PORTFOLIO_MAP[p_name]
                end_nlv = inputs['nlv']
                sec_val = inputs['sec']
                cash_flow = inputs['cash_flow']
                
                if end_nlv > 0:
                    df_curr = load_and_prep_file(p_path)
                    
                    # 1. DUPLICATE CHECK
                    if not df_curr.empty:
                        existing_dates = df_curr['Day'].tolist()
                        if entry_date_str in existing_dates:
                            if not force_ovr:
                                st.error(f"⚠️ SKIPPED {p_name}: Entry for {entry_date_str} exists.")
                                continue 
                            else:
                                df_curr = df_curr[df_curr['Day'] != entry_date_str]
                                st.warning(f"♻️ Overwriting {p_name} entry for {entry_date_str}...")

                    # 2. GET PREVIOUS
                    prev_nlv = 0.0
                    if not df_curr.empty:
                        df_curr = df_curr.sort_values('Day', ascending=False)
                        try: prev_nlv = clean_num(df_curr.iloc[0]['End NLV'])
                        except: prev_nlv = 0.0
                        
                    # 3. MATH
                    if prev_nlv > 0:
                        daily_chg = end_nlv - prev_nlv - cash_flow
                        pct_val = (daily_chg / prev_nlv) * 100
                    elif cash_flow > 0:
                        daily_chg = end_nlv - cash_flow
                        pct_val = (daily_chg / cash_flow) * 100
                    else:
                        daily_chg = 0.0
                        pct_val = 0.0
                    
                    daily_pct_str = f"{pct_val:.2f}%"
                    invested_pct = (sec_val / end_nlv) * 100 if end_nlv > 0 else 0.0
                    
                    # 4. CREATE ROW (NEW MAPPING)
                    new_row = {
                        'Day': entry_date_str, 'Status': 'U', 'Market Window': 'Open', '> 21e': 0.0,
                        'Cash -/+': cash_flow, 'Beg NLV': prev_nlv, 'End NLV': end_nlv,
                        'Daily $ Change': daily_chg, 'Daily % Change': daily_pct_str,
                        '% Invested': invested_pct, 'SPY': spy_val, 'Nasdaq': ndx_val,
                        'Market_Notes': market_notes,   # <--- Global Notes go here
                        'Market_Action': inputs['note'], # <--- Portfolio Actions go here
                        'Keywords': '',
                        'Score': 0
                    }
                    
                    # 5. SAVE
                    new_df_row = pd.DataFrame([new_row])
                    for c in MASTER_ORDER: 
                        if c not in new_df_row.columns: new_df_row[c] = ''
                    
                    df_final = pd.concat([new_df_row[MASTER_ORDER], df_curr[MASTER_ORDER]], ignore_index=True)
                    df_final['Sort_Key'] = pd.to_datetime(df_final['Day'], errors='coerce')
                    df_final = df_final.sort_values('Sort_Key', ascending=False).drop(columns=['Sort_Key'])
                    
                    try:
                        df_final.to_csv(p_path, index=False)
                        success_count += 1
                    except Exception as e: st.error(f"❌ Write Failed {p_name}: {e}")

                else: st.warning(f"Skipped {p_name}: NLV is 0.00")

            if success_count > 0:
                st.success(f"✅ Successfully Updated {success_count} Portfolios!")
                st.balloons()

    # --- FILE AUDITOR ---
    st.markdown("---")
    with st.expander("🕵️‍♂️ View File Locations & Status"):
        for p_name, p_path in PORTFOLIO_MAP.items():
            exists = "✅ Found" if os.path.exists(p_path) else "❌ Missing"
            st.markdown(f"**{p_name}:** `{p_path}` ({exists})")

# ==============================================================================
# PAGE 9: POSITION SIZER (RESTORED ARCHIVE VERSION)
# ==============================================================================
elif page == "Position Sizer":
    st.header(f"POSITION SIZING CALCULATOR ({CURR_PORT_NAME})")
    
    # --- GLOBAL DATA ---
    equity = 100000.0
    if os.path.exists(JOURNAL_FILE):
        try:
            df = load_data(JOURNAL_FILE)
            if not df.empty:
                # robust cleaning of currency string
                val_str = str(df['End NLV'].iloc[0]).replace('$','').replace(',','')
                equity = float(val_str)
        except: pass
    
    df_s = load_data(SUMMARY_FILE)
    
    # --- SIZING MAP ---
    size_map = {
        "Shotgun (2.5%)": 2.5, 
        "Half (5%)": 5.0, 
        "Full (10%)": 10.0, 
        "Full+1 (15%)": 15.0, 
        "Full+2 (20%)": 20.0, 
        "Max (25%)": 25.0,
        "30%": 30.0,
        "35%": 35.0,
        "40%": 40.0,
        "45%": 45.0,
        "50%": 50.0
    }

    # UPDATED TABS: ADDED 4TH TAB (TRIM)
    tab_new, tab_risk, tab_add, tab_trim = st.tabs(["🆕 Standard Entry", "🧮 Risk-Based Sizing", "➕ Add (Pyramid)", "✂️ Trim (Sell Down)"])
    
    # --- TAB 1: NEW POSITION (STANDARD) ---
    with tab_new:
        st.caption("Size based on Portfolio Weight (e.g., 'I want a 10% Position')")
        colA, colB = st.columns(2)
        with colA: 
            acct_val = st.number_input("Account Equity ($)", value=equity, step=1000.0, key="np_eq")
            ticker = st.text_input("Ticker Symbol", key="np_tk").upper()
            entry = st.number_input("Entry Price ($)", min_value=0.01, step=0.1, key="np_ep")
        
        with colB:
            stop_mode = st.radio("Stop Input Mode", ["Manual Price", "Percentage (%)"], horizontal=True, key="np_mode")
            if stop_mode == "Percentage (%)":
                stop_pct_in = st.number_input("Stop Loss %", value=8.0, step=0.5, key="np_pct")
                stop_val = entry * (1 - (stop_pct_in/100))
                st.info(f"Calculated Stop: ${stop_val:.2f}")
            else:
                stop_val = st.number_input("Stop Price ($)", min_value=0.0, max_value=entry if entry > 0 else 10000.0, key="np_sv")
                if entry > 0: stop_pct_in = ((entry - stop_val)/entry)*100
                else: stop_pct_in = 0

        st.markdown("---"); c1, c2 = st.columns(2)
        with c1: risk_pct = st.slider("Risk % of Capital", 0.25, 1.50, 0.75, 0.05, key="np_rp")
        with c2:
            size_mode = st.select_slider("Position Size Scale", options=list(size_map.keys()), value="Half (5%)", key="np_sm")
        max_pos_pct = size_map[size_mode]
        
        if st.button("Calculate Standard Trade", key="np_btn"):
            if entry > 0 and stop_val > 0 and stop_val < entry:
                risk_share = entry - stop_val
                risk_budget = acct_val * (risk_pct/100)
                shares_risk = math.floor(risk_budget / risk_share)
                shares_cap = math.floor((acct_val * (max_pos_pct/100)) / entry)
                rec_shares = min(shares_risk, shares_cap)
                cost = rec_shares * entry
                weight = (cost / acct_val) * 100
                risk_dol = rec_shares * risk_share
                stop_pct = (risk_share/entry)*100
                
                st.markdown("### 🎫 TRADE TICKET")
                t1, t2, t3 = st.columns(3)
                t1.metric("BUY SHARES", rec_shares)
                t2.metric("TOTAL COST", f"${cost:,.0f}")
                t3.metric("WEIGHT", f"{weight:.1f}%")
                
                st.caption(f"Risk: ${risk_dol:.0f} ({ (risk_dol/acct_val)*100 :.2f}%) | Stop Width: {stop_pct:.2f}%")
                
                if stop_pct > 8.0: st.error("⚠️ STOP LOSS > 8% Violation!")
                if rec_shares == shares_cap: st.info(f"ℹ️ Capped by Max Size ({max_pos_pct}%)")
            else: st.error("Check your prices.")

    # --- TAB 2: RISK-BASED SIZING ---
    with tab_risk:
        st.caption("Size based on Risk Budget (e.g., 'I want to risk 0.75% of Equity')")
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            r_eq = st.number_input("Account Equity ($)", value=equity, step=1000.0, key="rb_eq")
            r_tick = st.text_input("Ticker Symbol", key="rb_tk").upper()
            r_entry = st.number_input("Entry Price ($)", min_value=0.01, step=0.1, key="rb_ep")
        with r_c2:
            st.markdown("**Stop Loss Settings**")
            r_stop_mode = st.radio("Stop Mode", ["Percentage Distance (%)", "Price Level ($)"], horizontal=True, key="rb_sm")
            if r_stop_mode == "Percentage Distance (%)":
                r_sl_pct = st.number_input("Stop Loss Width %", value=7.0, step=0.5, key="rb_slp")
                r_stop_price = r_entry * (1 - (r_sl_pct/100)) if r_entry > 0 else 0
                st.info(f"Implied Stop Price: ${r_stop_price:.2f}")
            else:
                r_stop_price = st.number_input("Stop Price Level ($)", min_value=0.01, step=0.1, key="rb_slv")
                if r_entry > 0 and r_stop_price < r_entry:
                    r_sl_pct = ((r_entry - r_stop_price) / r_entry) * 100
                    st.info(f"Implied Stop Width: {r_sl_pct:.2f}%")
                else: r_sl_pct = 0
        st.markdown("---")
        r_risk_pct = st.slider("Risk Budget (% of Equity)", 0.25, 2.0, 0.75, 0.05, key="rb_rp")
        risk_dollars = r_eq * (r_risk_pct / 100)
        st.write(f"**Risk Budget:** ${risk_dollars:,.2f}")
        
        if st.button("Calculate Risk-Based Size", type="primary", key="rb_btn"):
            if r_entry > 0 and r_stop_price > 0 and r_stop_price < r_entry:
                risk_per_share = r_entry - r_stop_price
                shares_allowable = math.floor(risk_dollars / risk_per_share)
                total_cost = shares_allowable * r_entry
                portfolio_weight = (total_cost / r_eq) * 100
                
                st.markdown("### 🎯 SIZING RESULTS")
                m1, m2, m3 = st.columns(3)
                m1.metric("BUY SHARES", f"{shares_allowable}")
                m2.metric("TOTAL COST", f"${total_cost:,.2f}")
                m3.metric("PORTFOLIO WEIGHT", f"{portfolio_weight:.1f}%")
                
                if r_sl_pct > 8.0: st.warning(f"⚠️ WIDE STOP ({r_sl_pct:.2f}%): Note that standard CAN SLIM stops are 7-8% max.")
                if portfolio_weight > 25.0: st.warning(f"⚠️ CONCENTRATION RISK: This position would be {portfolio_weight:.1f}% of your account.")
                st.success(f"Logic: Buying {shares_allowable} shares with a ${risk_per_share:.2f} stop width risks exactly ${risk_dollars:,.2f}.")
            else: st.error("Invalid Prices: Entry must be greater than Stop Price.")

    # --- TAB 3: PYRAMIDING (ADD ON) ---
    with tab_add:
        st.caption("Calculate shares to add using 'Financed Risk' (Open Profit).")
        open_positions = df_s[df_s['Status'] == 'OPEN'].sort_values('Ticker')
        
        if not open_positions.empty:
            sel_pos = st.selectbox("Select Holding", options=open_positions['Ticker'].unique().tolist(), key="add_sel")
            row = open_positions[open_positions['Ticker'] == sel_pos].iloc[0]
            
            # Fetch Price safely
            try: live_price = yf.Ticker(row['Ticker']).history(period="1d")['Close'].iloc[-1]
            except: live_price = row.get('Avg_Entry', 100)
            
            c1, c2 = st.columns(2)
            curr_price = c1.number_input("Current Price ($)", min_value=0.01, value=float(live_price), step=0.1, key=f"add_cp_{row['Ticker']}")
            acct_val_add = c2.number_input("Account Equity ($)", value=equity, disabled=True, key="add_av")
            
            st.markdown(f"**Current Status:** {int(row['Shares'])} shares @ ${row['Avg_Entry']:.2f} ({(row['Total_Cost']/equity)*100:.1f}% Weight)")
            st.markdown("---")
            
            # O'NEIL UPDATE: Simplified Input (Removed Risk Slider)
            target_mode_add = st.select_slider("Target Total Position Size", options=list(size_map.keys()), value="Full (10%)", key="add_ts_mode")
            target_size_pct = size_map[target_mode_add]
            
            if st.button("Calculate Add-On", key="add_btn"):
                # 1. Standard Math
                target_value = acct_val_add * (target_size_pct / 100)
                current_value = row['Shares'] * curr_price 
                value_to_add = target_value - current_value
                
                if value_to_add <= 0:
                    st.error(f"You are already over the target weight! (Current: ${current_value:,.0f} vs Target: ${target_value:,.0f})")
                else:
                    shares_to_add = math.floor(value_to_add / curr_price)
                    total_shares = row['Shares'] + shares_to_add
                    
                    # 2. Weighted Average Cost Calculation
                    new_avg_cost = ((row['Shares'] * row['Avg_Entry']) + (shares_to_add * curr_price)) / total_shares
                    
                    # 3. Financed Risk Logic (CORRECTED)
                    current_profit = (curr_price - row['Avg_Entry']) * row['Shares']
                    
                    # Breakeven Stop: Price where Net P&L = $0
                    be_stop = new_avg_cost 
                    
                    # House Money Stop: Price where Net P&L = +50% of Original Profit
                    # To KEEP profit, we must stop ABOVE the average cost.
                    profit_to_keep = current_profit * 0.5
                    house_money_stop = new_avg_cost + (profit_to_keep / total_shares)

                    st.markdown("### ➕ PYRAMID TICKET")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("ADD SHARES", f"+{shares_to_add}")
                    k2.metric("NEW TOTAL", f"{int(total_shares)}")
                    k3.metric("NEW AVG COST", f"${new_avg_cost:.2f}", help="Price where you break even (lose all profit).")
                    k4.metric("CUSHION", f"${current_profit:,.0f}", help="Total open profit available.")
                    
                    st.markdown("### 🛡️ SAFE STOPS (Financed Risk)")
                    
                    c_risk1, c_risk2 = st.columns(2)
                    with c_risk1:
                        st.info(f"**Breakeven Stop:** ${be_stop:.2f}\n\nAt this price, you give back 100% of profit but lose **$0 Principal**.")
                    with c_risk2:
                        st.info(f"**'Lock 50%' Stop:** ${house_money_stop:.2f}\n\nAt this price, you walk away with **${profit_to_keep:,.0f}** profit.")

                    st.markdown("---")
                    st.caption("🤖 **O'Neil Logic Check:** Compare these stops to your Chart Support (e.g., 21-Day Line).")
                    
                    if be_stop > curr_price:
                         st.error("⛔ **MATH ERROR:** New Average Cost is higher than Current Price. Check inputs.")
                    elif be_stop > (curr_price * 0.98):
                         st.warning("⚠️ **DANGER:** Your Breakeven Stop is less than 2% away. This is NOT a Free Roll.")
                    else:
                         st.success("✅ **GREEN LIGHT:** You have a solid cushion. If your Chart Stop is higher than your Breakeven Stop, you are safe.")
        else:
            st.info("No Open Positions found in Summary file.")

    # --- TAB 4: TRIM (SELL DOWN) ---
    with tab_trim:
        st.caption("Calculate shares to sell to reach a desired Standard Portfolio Size.")
        open_positions = df_s[df_s['Status'] == 'OPEN'].sort_values('Ticker')
        
        if not open_positions.empty:
            sel_trim = st.selectbox("Select Holding to Trim", options=open_positions['Ticker'].unique().tolist(), key="trim_sel")
            row_t = open_positions[open_positions['Ticker'] == sel_trim].iloc[0]
            
            # Fetch Price safely
            try: live_price_t = yf.Ticker(row_t['Ticker']).history(period="1d")['Close'].iloc[-1]
            except: live_price_t = row_t.get('Avg_Entry', 100)
            
            t1, t2, t3 = st.columns(3)
            curr_price_t = t1.number_input("Current Price ($)", value=float(live_price_t), step=0.1, key=f"trim_cp_{row_t['Ticker']}")
            curr_val_t = row_t['Shares'] * curr_price_t
            curr_weight_t = (curr_val_t / equity) * 100
            
            t2.metric("Current Weight", f"{curr_weight_t:.1f}%")
            t3.metric("Current Value", f"${curr_val_t:,.0f}")
            
            st.markdown("---")
            st.subheader("Target Allocation")
            target_mode_trim = st.select_slider("Target Standard Position Size", options=list(size_map.keys()), value="Half (5%)", key="trim_sm")
            target_weight_t = size_map[target_mode_trim]
            
            if st.button("Calculate Trim", type="primary", key="trim_btn"):
                if target_weight_t >= curr_weight_t:
                    st.warning(f"⚠️ Target ({target_weight_t}%) is higher than or equal to Current ({curr_weight_t:.1f}%). Use the 'Add' tab to increase size.")
                else:
                    target_val_t = equity * (target_weight_t / 100)
                    value_to_sell = curr_val_t - target_val_t
                    shares_to_sell = math.ceil(value_to_sell / curr_price_t) 
                    
                    remaining_shares = row_t['Shares'] - shares_to_sell
                    remaining_val = remaining_shares * curr_price_t
                    actual_new_weight = (remaining_val / equity) * 100
                    
                    st.markdown("### ✂️ SELL TICKET")
                    c_res1, c_res2, c_res3 = st.columns(3)
                    c_res1.metric("SELL SHARES", f"-{int(shares_to_sell)}", help=f"Proceeds: ${shares_to_sell*curr_price_t:,.0f}")
                    c_res2.metric("REMAINING SHARES", f"{int(remaining_shares)}")
                    c_res3.metric("NEW WEIGHT", f"{actual_new_weight:.1f}%", delta=f"{actual_new_weight - curr_weight_t:.1f}%")
                    
                    if remaining_shares < 0:
                        st.error("Error: Calculation resulted in negative shares. Check inputs.")
                    elif remaining_shares == 0:
                        st.info("ℹ️ This closes the position completely.")
                    else:
                        st.success(f"✅ Selling {int(shares_to_sell)} shares reduces position to {target_mode_trim} ({target_weight_t}%).")
        else:
            st.info("No active positions to trim.")

# ==============================================================================
# PAGE 10: TRADE MANAGER (FULL CONTEXT: BUY/SELL NOTES & RULES)
# ==============================================================================
elif page == "Trade Manager":
    st.header(f"TRADE MANAGER ({CURR_PORT_NAME})")
    
    # Initialize files if missing
    if not os.path.exists(DETAILS_FILE): 
        pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Amount','Value','Rule','Notes','Realized_PL','Stop_Loss','Trx_ID']).to_csv(DETAILS_FILE, index=False)
    if not os.path.exists(SUMMARY_FILE): 
        pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Total_Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Rule','Notes','Buy_Notes','Sell_Rule','Sell_Notes']).to_csv(SUMMARY_FILE, index=False)
    
    df_d = load_data(DETAILS_FILE)
    df_s = load_data(SUMMARY_FILE)
    
    # --- SCHEMA FIXES ---
    # 1. Rename legacy 'Buy_Rule' -> 'Rule'
    if 'Buy_Rule' in df_s.columns and 'Rule' not in df_s.columns:
        df_s.rename(columns={'Buy_Rule': 'Rule'}, inplace=True)
    if 'Rule' not in df_s.columns: df_s['Rule'] = ""

    # 2. Ensure new dedicated Note/Rule columns exist
    for col in ['Buy_Notes', 'Sell_Rule', 'Sell_Notes']:
        if col not in df_s.columns: df_s[col] = ""

    valid_sum_cols = ['Trade_ID', 'Ticker', 'Status', 'Open_Date', 'Shares', 'Avg_Entry', 'Total_Cost', 'Unrealized_PL', 'Return_Pct', 'Rule', 'Buy_Notes', 'Sell_Rule']
    valid_sum_cols = [c for c in valid_sum_cols if c in df_s.columns]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Log Buy", "Log Sell", "Update Prices", "Edit Transaction", "Database Health", "Delete Trade", "Active Campaign Summary", "Active Campaign Detailed", "Detailed Trade Log", "All Campaigns"])
    
    # --- TAB 1: LOG BUY ---
    with tab1:
        st.caption("Live Entry Calculator")
        
        # Session State Init
        if 'b_tick' not in st.session_state: st.session_state['b_tick'] = ""
        if 'b_id' not in st.session_state: st.session_state['b_id'] = ""
        if 'b_shs' not in st.session_state: st.session_state['b_shs'] = 0
        if 'b_px' not in st.session_state: st.session_state['b_px'] = 0.0
        if 'b_note' not in st.session_state: st.session_state['b_note'] = ""
        if 'b_trx' not in st.session_state: st.session_state['b_trx'] = ""
        if 'b_sl_pct' not in st.session_state: st.session_state['b_sl_pct'] = 8.0
        if 'b_stop_val' not in st.session_state: st.session_state['b_stop_val'] = 0.0

        c_top1, c_top2 = st.columns(2)
        trade_type = c_top1.radio("Action Type", ["Start New Campaign", "Scale In (Add to Existing)"], horizontal=True)
        
        now = datetime.now()
        b_date = c_top2.date_input("Date", now, key="b_date_input")
        b_time = c_top2.time_input("Time", now.time(), step=60, key="b_time_input")
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        if trade_type == "Start New Campaign":
            b_tick = c1.text_input("Ticker Symbol", key="b_tick")
            if b_tick: b_tick = b_tick.upper() 
            
            now_ym = datetime.now().strftime("%Y%m") 
            default_id = f"{now_ym}-001"
            if not df_s.empty:
                relevant_ids = [str(x) for x in df_s['Trade_ID'] if str(x).startswith(now_ym)]
                if relevant_ids:
                    try:
                        last_seq = max([int(x.split('-')[-1]) for x in relevant_ids if '-' in x])
                        new_seq = last_seq + 1
                        default_id = f"{now_ym}-{new_seq:03d}"
                    except: pass
            if st.session_state['b_id'] == "": st.session_state['b_id'] = default_id  
            b_id = c2.text_input("Trade ID", key="b_id")
            b_rule = st.selectbox("Buy Rule", BUY_RULES)
        else:
            open_opts = df_s[df_s['Status']=='OPEN'].copy()
            b_tick, b_id = "", ""
            if not open_opts.empty:
                open_opts = open_opts.sort_values('Ticker')
                opts = ["Select..."] + [f"{r['Ticker']} | {r['Trade_ID']}" for _, r in open_opts.iterrows()]
                sel_camp = c1.selectbox("Select Existing Campaign", opts, key="b_scale_sel")
                if sel_camp and sel_camp != "Select...":
                    b_tick, b_id = sel_camp.split(" | ")
                    curr_row = open_opts[open_opts['Trade_ID']==b_id].iloc[0]
                    c2.info(f"Holding: {int(curr_row['Shares'])} shs @ ${curr_row['Avg_Entry']:.2f}")
            else: c1.warning("No Open Campaigns.")
            b_rule = st.selectbox("Add Rule", ADD_RULES)

        c3, c4 = st.columns(2)
        b_shs = c3.number_input("Shares", min_value=0, step=1, key="b_shs")
        b_px = c4.number_input("Entry Price ($)", min_value=0.0, step=0.1, format="%.2f", key="b_px")
        
        st.markdown("#### 🛡️ Risk Management")
        c_stop1, c_stop2 = st.columns(2)
        with c_stop1: stop_mode = st.radio("Stop Loss Mode", ["Price Level ($)", "Percentage (%)"], horizontal=True)
        with c_stop2:
            if stop_mode == "Percentage (%)":
                sl_pct = st.number_input("Stop Loss %", value=8.0, step=0.5, format="%.1f", key="b_sl_pct")
                b_stop = b_px * (1 - (sl_pct/100)) if b_px > 0 else 0.0
                st.metric("Calculated Stop", f"${b_stop:.2f}", delta=f"-{sl_pct}%")
            else:
                def_val = float(b_px * 0.92) if (st.session_state['b_stop_val'] == 0.0 and b_px > 0) else st.session_state['b_stop_val']
                b_stop = st.number_input("Stop Price ($)", min_value=0.0, step=0.1, value=def_val, format="%.2f", key="b_stop_val")
                if b_px > 0 and b_stop > 0: 
                    actual_pct = ((b_px - b_stop) / b_px) * 100
                    st.caption(f"Implied Risk: {actual_pct:.2f}%")

        st.markdown("---")
        c_note1, c_note2 = st.columns(2)
        b_note = c_note1.text_input("Buy Rationale (Notes)", key="b_note")
        b_trx = c_note2.text_input("Manual Trx ID (Optional)", key="b_trx")

        if st.button("LOG BUY ORDER", type="primary", use_container_width=True):
            if b_tick and b_id and b_shs > 0 and b_px > 0:
                ts = datetime.combine(b_date, b_time).strftime("%Y-%m-%d %H:%M")
                cost = b_shs * b_px
                if not b_trx: b_trx = generate_trx_id(df_d, b_id, 'BUY', ts)
                
                # Save Detail
                new_d = {'Trade_ID': b_id, 'Trx_ID': b_trx, 'Ticker': b_tick, 'Action': 'BUY', 'Date': ts, 'Shares': b_shs, 'Amount': b_px, 'Value': cost, 'Rule': b_rule, 'Notes': b_note, 'Realized_PL': 0, 'Stop_Loss': b_stop}
                df_d = pd.concat([df_d, pd.DataFrame([new_d])], ignore_index=True)
                
                if trade_type == "Start New Campaign":
                    # --- NEW: EXPLICITLY SAVE 'Buy_Notes' ---
                    new_s = {
                        'Trade_ID': b_id, 'Ticker': b_tick, 'Status': 'OPEN', 'Open_Date': ts, 
                        'Shares': 0, 'Avg_Entry': 0, 'Total_Cost': 0, 'Realized_PL': 0, 'Unrealized_PL': 0, 
                        'Rule': b_rule, 
                        'Notes': b_note,       # General Note
                        'Buy_Notes': b_note,   # Explicit Buy Note
                        'Sell_Rule': '', 'Sell_Notes': '' # Init empty
                    }
                    df_s = pd.concat([df_s, pd.DataFrame([new_s])], ignore_index=True)
                
                secure_save(df_d, DETAILS_FILE)
                df_d, df_s = update_campaign_summary(b_id, df_d, df_s) # Syncs math
                secure_save(df_d, DETAILS_FILE)
                secure_save(df_s, SUMMARY_FILE)
                
                st.success(f"✅ EXECUTED: Bought {b_shs} {b_tick} @ ${b_px}")
                for k in ['b_tick','b_id','b_shs','b_px','b_note','b_trx','b_stop_val']:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()
            else: st.error("⚠️ Missing Data.")

    # --- TAB 2: LOG SELL ---
    with tab2:
        open_opts = df_s[df_s['Status']=='OPEN'].copy()
        if not open_opts.empty:
             open_opts = open_opts.sort_values('Ticker')
             s_opts = [f"{r['Ticker']} | {r['Trade_ID']}" for _, r in open_opts.iterrows()]
             sel_sell = st.selectbox("Select Trade to Sell", s_opts)
             if sel_sell:
                 s_tick, s_id = sel_sell.split(" | ")
                 row = open_opts[open_opts['Trade_ID']==s_id].iloc[0]
                 st.info(f"Selling {s_tick} (Own {int(row['Shares'])} shs)")
                 
                 c1, c2 = st.columns(2)
                 s_date = c1.date_input("Date", datetime.now(), key='s_date')
                 s_time = c2.time_input("Time", datetime.now().time(), step=60, key='s_time')
                 
                 c3, c4 = st.columns(2)
                 s_shs = c3.number_input("Shares", min_value=1, max_value=int(row['Shares']), step=1)
                 s_px = c4.number_input("Price", min_value=0.0, step=0.1)
                 
                 # --- NEW: EXPLICIT SELL RULE & NOTES ---
                 c5, c6 = st.columns(2)
                 s_rule = c5.selectbox("Sell Rule / Reason", SELL_RULES)
                 s_note = c6.text_input("Sell Context / Notes", key='s_note', placeholder="Why did you sell?")
                 s_trx = st.text_input("Manual Trx ID (Optional)", key='s_trx')
                 
                 if st.button("LOG SELL ORDER", type="primary"):
                    ts = datetime.combine(s_date, s_time).strftime("%Y-%m-%d %H:%M")
                    proc = s_shs * s_px
                    if not s_trx: s_trx = generate_trx_id(df_d, s_id, 'SELL', ts)
                    
                    # Log Detail (Rule = Sell Rule, Notes = Sell Note)
                    new_d = {'Trade_ID':s_id, 'Trx_ID': s_trx, 'Ticker':s_tick, 'Action':'SELL', 'Date':ts, 'Shares':s_shs, 'Amount':s_px, 'Value':proc, 'Rule':s_rule, 'Notes': s_note, 'Realized_PL': 0}
                    df_d = pd.concat([df_d, pd.DataFrame([new_d])], ignore_index=True)
                    secure_save(df_d, DETAILS_FILE)
                    
                    # Sync Math
                    df_d, df_s = update_campaign_summary(s_id, df_d, df_s)
                    
                    # --- CRITICAL FIX: FORCE WRITE SELL DATA TO SUMMARY ---
                    # We do this AFTER update_campaign_summary to ensure it doesn't get overwritten
                    idx = df_s[df_s['Trade_ID'] == s_id].index
                    if not idx.empty:
                        df_s.at[idx[0], 'Sell_Rule'] = s_rule
                        # Append or Overwrite notes? Overwrite is cleaner for "Last Action Reason"
                        df_s.at[idx[0], 'Sell_Notes'] = s_note
                    
                    secure_save(df_s, SUMMARY_FILE)
                    
                    st.success(f"Sold. Transaction ID: {s_trx}")
                    st.rerun()
        else: st.info("No positions to sell.")

    # --- TAB 3: UPDATE PRICES ---
    with tab3:
        st.subheader("🛡️ Risk Control Center")
        if st.button("REFRESH MARKET PRICES", type="primary"):
            open_rows = df_s[df_s['Status']=='OPEN']
            if not open_rows.empty:
                p = st.progress(0); n=0
                for i, r in open_rows.iterrows():
                    try:
                        tk = r['Ticker'] if r['Ticker']!='COMP' else '^IXIC'
                        curr = yf.Ticker(tk).history(period='1d')['Close'].iloc[-1]
                        mkt = r['Shares'] * curr
                        unreal = mkt - r['Total_Cost']
                        df_s.at[i, 'Unrealized_PL'] = unreal
                        df_s.at[i, 'Return_Pct'] = (unreal/r['Total_Cost'])*100 if r['Total_Cost'] else 0
                    except: pass
                    n+=1; p.progress(n/len(open_rows))
                secure_save(df_s, SUMMARY_FILE); st.success("✅ Prices Updated!"); st.rerun()
            else: st.warning("No open positions.")

        st.markdown("---")
        st.markdown("### 🛑 Rapid Stop Adjustment")
        open_pos = df_s[df_s['Status'] == 'OPEN'].sort_values('Ticker')
        if not open_pos.empty:
            def get_current_stop_display(tid):
                try:
                    stops = df_d[df_d['Trade_ID'] == tid]['Stop_Loss']
                    val = stops.iloc[-1] if not stops.empty else 0.0
                    return val
                except: return 0.0

            opts_dict = {f"{r['Ticker']} (Current: ${get_current_stop_display(r['Trade_ID']):.2f})": r['Trade_ID'] for _, r in open_pos.iterrows()}
            sel_label = st.selectbox("Select Position to Protect", list(opts_dict.keys()))
            sel_id = opts_dict[sel_label]
            curr_stop_val = get_current_stop_display(sel_id)
            
            c_up1, c_up2, c_up3 = st.columns(3)
            new_stop_price = c_up1.number_input("New Hard Stop Price ($)", value=float(curr_stop_val), min_value=0.0, step=0.01, format="%.2f")
            
            if c_up3.button("UPDATE STOP LOSS"):
                mask = (df_d['Trade_ID'] == sel_id) & (df_d['Action'] == 'BUY')
                if mask.any():
                    last_idx = df_d[mask].last_valid_index()
                    df_d.at[last_idx, 'Stop_Loss'] = new_stop_price
                    secure_save(df_d, DETAILS_FILE)
                    st.success(f"✅ Stop Updated to ${new_stop_price:.2f}")
                    st.rerun()
                else: st.error("Could not find a BUY transaction.")
        else: st.info("No active positions.")

    # --- TAB 4: EDIT TRANSACTION ---
    with tab4:
        st.header("📝 Edit Transaction")
        all_ids = sorted([str(x) for x in df_d['Trade_ID'].unique()], reverse=True)
        if not all_ids:
            st.info("No trades recorded yet.")
        else:
            def fmt_func(x):
                try: return f"{x} | {df_d[df_d['Trade_ID'].astype(str) == x]['Ticker'].iloc[0]}"
                except: return str(x)
            edit_id = st.selectbox("Select Trade ID to Edit", all_ids, format_func=fmt_func)
            if edit_id:
                txs = df_d[df_d['Trade_ID'].astype(str) == edit_id].reset_index().sort_values('Date', ascending=False)
                if not txs.empty:
                    tx_options = [f"{row.get('Trx_ID','')} | {row['Date']} | {row['Action']} {row['Shares']} @ {row['Amount']}" for idx, row in txs.iterrows()]
                    selected_tx_str = st.selectbox("Select Transaction Line", tx_options)
                    if selected_tx_str:
                        sel_idx = tx_options.index(selected_tx_str)
                        row_idx = int(txs.iloc[sel_idx]['index'])
                        current_row = df_d.loc[row_idx]
                        
                        st.markdown("---")
                        cA, cB = st.columns([2, 1])
                        with cA:
                            with st.form("edit_form"):
                                st.subheader(f"Editing: {selected_tx_str}")
                                c1, c2 = st.columns(2)
                                try: dt_obj = pd.to_datetime(current_row['Date'])
                                except: dt_obj = datetime.now()
                                e_date = c1.date_input("Date", dt_obj)
                                e_time = c1.time_input("Time", dt_obj.time(), step=60)
                                
                                # Rule Edit
                                curr_rule = current_row.get('Rule', '')
                                r_idx = ALL_RULES.index(curr_rule) if curr_rule in ALL_RULES else 0
                                e_rule = c2.selectbox("Strategy / Rule", ALL_RULES, index=r_idx)
                                
                                e_trx = st.text_input("Trx ID", value=str(current_row.get('Trx_ID', '')))
                                sl_val = float(current_row['Stop_Loss']) if pd.notna(current_row.get('Stop_Loss')) else 0.0
                                e_stop = c1.number_input("Stop Loss", value=sl_val, step=0.01) 
                                e_note = c2.text_input("Notes", str(current_row.get('Notes', '')))
                                
                                e_shs = c1.number_input("Shares", value=float(current_row['Shares']), step=1.0)
                                e_amt = c2.number_input("Price ($)", value=float(current_row['Amount']), step=0.01)
                                
                                if st.form_submit_button("💾 Save Changes"):
                                    new_ts = datetime.combine(e_date, e_time).strftime("%Y-%m-%d %H:%M")
                                    df_d.at[row_idx, 'Date'] = new_ts
                                    df_d.at[row_idx, 'Rule'] = e_rule
                                    df_d.at[row_idx, 'Stop_Loss'] = e_stop
                                    df_d.at[row_idx, 'Notes'] = e_note
                                    df_d.at[row_idx, 'Shares'] = e_shs
                                    df_d.at[row_idx, 'Amount'] = e_amt
                                    df_d.at[row_idx, 'Value'] = e_shs * e_amt
                                    df_d.at[row_idx, 'Trx_ID'] = e_trx
                                    
                                    secure_save(df_d, DETAILS_FILE)
                                    df_d, df_s = update_campaign_summary(edit_id, df_d, df_s)
                                    secure_save(df_s, SUMMARY_FILE)
                                    st.success("✅ Updated!"); st.rerun()

                        with cB:
                            st.write("### ⚠️ Danger Zone")
                            if st.button("🗑️ DELETE TRANSACTION", type="primary"):
                                df_d = df_d.drop(row_idx)
                                secure_save(df_d, DETAILS_FILE)
                                df_d, df_s = update_campaign_summary(edit_id, df_d, df_s)
                                secure_save(df_s, SUMMARY_FILE)
                                st.warning("Transaction Deleted."); st.rerun()

    # --- TAB 5: DATABASE HEALTH ---
    with tab5:
        st.subheader("Database Maintenance")
        if st.button("FULL REBUILD (Generate Missing Summaries)"):
            if df_d.empty: st.error("Details file is empty.")
            else:
                det_ids = df_d['Trade_ID'].unique()
                sum_ids = df_s['Trade_ID'].unique() if not df_s.empty else []
                missing = [tid for tid in det_ids if tid not in sum_ids]
                new_rows = []
                for tid in missing:
                    trade_txs = df_d[df_d['Trade_ID'] == tid]
                    buys = trade_txs[trade_txs['Action'] == 'BUY'].sort_values('Date')
                    first_tx = buys.iloc[0] if not buys.empty else trade_txs.sort_values('Date').iloc[0]
                    new_rows.append({'Trade_ID': str(tid), 'Ticker': first_tx['Ticker'], 'Status': 'OPEN', 'Open_Date': first_tx['Date'], 'Shares': 0, 'Total_Cost': 0, 'Realized_PL': 0})
                if new_rows:
                    df_s = pd.concat([df_s, pd.DataFrame(new_rows)], ignore_index=True)
                
                all_ids = df_d['Trade_ID'].unique()
                p=st.progress(0)
                for i, tid in enumerate(all_ids):
                    df_d, df_s = update_campaign_summary(tid, df_d, df_s)
                    p.progress((i+1)/len(all_ids))
                
                secure_save(df_d, DETAILS_FILE); secure_save(df_s, SUMMARY_FILE)
                st.success(f"Rebuilt {len(all_ids)} Campaigns."); st.rerun()

    # --- TAB 6: DELETE TRADE ---
    with tab6:
        del_id = st.selectbox("ID to Delete", df_s['Trade_ID'].tolist() if not df_s.empty else [])
        if st.button("DELETE PERMANENTLY"):
            df_s = df_s[df_s['Trade_ID']!=del_id]; df_d = df_d[df_d['Trade_ID']!=del_id]
            secure_save(df_s, SUMMARY_FILE); secure_save(df_d, DETAILS_FILE)
            st.success("Deleted."); st.rerun()

# --- TAB 7: ACTIVE CAMPAIGN SUMMARY ---
    with tab7:
        st.subheader("Active Campaign Summary")
        # Ensure we have data and columns
        if not df_s.empty:
             # Filter for OPEN trades
             df_open = df_s[df_s['Status'] == 'OPEN'].copy()
             
             if not df_open.empty:
                 # --- 1. DATA ENRICHMENT ---
                 def get_last_stop(tid):
                     try:
                         rows = df_d[df_d['Trade_ID'] == tid]
                         valid_stops = rows[rows['Stop_Loss'] > 0.01]['Stop_Loss']
                         return valid_stops.iloc[-1] if not valid_stops.empty else 0.0
                     except: return 0.0
                 
                 df_open['Stop Loss'] = df_open['Trade_ID'].apply(get_last_stop)
                 df_open['Current Value'] = df_open['Total_Cost'] + df_open.get('Unrealized_PL', 0.0).fillna(0.0)
                 
                 # Calculate Current Price based on Value/Shares to handle partial fills correctly
                 df_open['Current Price'] = df_open.apply(lambda x: (x['Current Value']/x['Shares']) if x['Shares'] > 0 else 0, axis=1)
                 
                 # Risk Math
                 df_open['Risk $'] = (df_open['Current Price'] - df_open['Stop Loss']) * df_open['Shares']
                 df_open['Risk $'] = df_open['Risk $'].apply(lambda x: x if x > 0 else 0.0)
                 
                 # Equity Fetch for % calculations
                 equity = 100000 # Fallback
                 if os.path.exists(JOURNAL_FILE):
                     try: 
                         j_df = pd.read_csv(JOURNAL_FILE)
                         if not j_df.empty:
                            equity = clean_num(j_df['End NLV'].iloc[0]) 
                     except: pass
                 
                 df_open['Risk %'] = (df_open['Risk $'] / equity) * 100
                 df_open['Pos Size %'] = (df_open['Current Value'] / equity) * 100
                 
                 # Max Stop Loss (0.5% Equity Rule)
                 df_open['Max SL (0.5%)'] = df_open.apply(lambda row: row['Avg_Entry'] - ((equity * 0.005) / row['Shares']) if row['Shares'] > 0 else 0, axis=1)
                 
                 # --- 2. THE DASHBOARD METRICS (RESTORED) ---
                 total_mkt_val = df_open['Current Value'].sum()
                 total_unreal = df_open['Unrealized_PL'].sum()
                 total_risk = df_open['Risk $'].sum()
                 
                 m1, m2, m3, m4, m5 = st.columns(5)
                 m1.metric("Open Positions", len(df_open))
                 m2.metric("Total Market Value", f"${total_mkt_val:,.2f}")
                 m3.metric("Total Unrealized P&L", f"${total_unreal:,.2f}", delta_color="normal")
                 m4.metric("Total Open Risk ($)", f"${total_risk:,.2f}")
                 m5.metric("Total Portfolio Heat (%)", f"{df_open['Risk %'].sum():.2f}%")
                 
                 # --- 3. THE DATAFRAME ---
                 # We use 'Rule' instead of 'Buy_Rule' (Schema fix) and add 'Buy_Notes'
                 cols_target = [
                     'Trade_ID', 'Ticker', 'Open_Date', 'Shares', 'Avg_Entry', 
                     'Current Price', 'Total_Cost', 'Current Value', 'Unrealized_PL', 
                     'Return_Pct', 'Pos Size %', 'Stop Loss', 'Max SL (0.5%)', 
                     'Risk $', 'Risk %', 'Rule', 'Buy_Notes'
                 ]
                 # Safety check: only show columns that actually exist
                 cols_final = [c for c in cols_target if c in df_open.columns]
                 
                 st.dataframe(
                     df_open[cols_final].style
                     .format({
                         'Shares':'{:.0f}', 'Total_Cost':'${:,.2f}', 'Unrealized_PL':'${:,.2f}', 
                         'Avg_Entry':'${:,.2f}', 'Current Price':'${:,.2f}', 'Return_Pct':'{:.2f}%', 
                         'Current Value': '${:,.2f}', 'Pos Size %': '{:.1f}%', 'Stop Loss': '${:,.2f}', 
                         'Max SL (0.5%)': '${:,.2f}', 'Risk $': '${:,.2f}', 'Risk %': '{:.2f}%',
                         'Open_Date': lambda x: x if isinstance(x, str) else (x.strftime('%Y-%m-%d') if pd.notnull(x) else '')
                     })
                     .applymap(color_pnl, subset=['Unrealized_PL', 'Return_Pct']),
                     height=(len(df_open) + 1) * 35 + 3,
                     use_container_width=True
                 )
             else: st.info("No open positions.")
        else: st.info("No data available.")

# --- TAB 8: ACTIVE CAMPAIGN DETAILED (WITH FLIGHT DECK) ---
    with tab8:
        st.subheader("Active Campaign Detailed (Transactions)")
        if not df_d.empty and not df_s.empty:
            open_ids = df_s[df_s['Status'] == 'OPEN']['Trade_ID'].unique().tolist()
            view_df = df_d[df_d['Trade_ID'].isin(open_ids)].copy()
            
            if not view_df.empty:
                unique_open_tickers = sorted(view_df['Ticker'].unique().tolist())
                tick_filter = st.selectbox("Filter Open Ticker", ["All"] + unique_open_tickers, key='act_det')
                
                # --- 1. THE FLIGHT DECK (SUMMARY METRICS) ---
                if tick_filter != "All":
                    # Get the Summary Row for this Ticker
                    summ_row = df_s[(df_s['Ticker'] == tick_filter) & (df_s['Status'] == 'OPEN')]
                    
                    if not summ_row.empty:
                        r = summ_row.iloc[0]
                        
                        # 1. Fetch Live Price (or calculate proxy)
                        try:
                            live_px = yf.Ticker(tick_filter).history(period="1d")['Close'].iloc[-1]
                        except:
                            # Fallback: Current Value / Shares
                            if r['Shares'] > 0:
                                val = r['Total_Cost'] + r.get('Unrealized_PL', 0)
                                live_px = val / r['Shares']
                            else: live_px = r['Avg_Entry']
                        
                        # 2. Calculate Metrics
                        shares = r['Shares']
                        avg_cost = r['Avg_Entry']
                        realized = r.get('Realized_PL', 0.0)
                        
                        # Live Unrealized Calculation
                        mkt_val = shares * live_px
                        unrealized = mkt_val - r['Total_Cost']
                        return_pct = (unrealized / r['Total_Cost']) * 100 if r['Total_Cost'] else 0.0
                        
                        # 3. Display Flight Deck
                        st.markdown(f"### 🚁 Flight Deck: {tick_filter}")
                        f1, f2, f3, f4, f5, f6 = st.columns(6)
                        f1.metric("Current Price", f"${live_px:,.2f}")
                        f2.metric("Avg Cost", f"${avg_cost:,.2f}")
                        f3.metric("Shares Held", f"{int(shares):,}")
                        f4.metric("Unrealized P&L", f"${unrealized:,.2f}", f"{return_pct:.2f}%")
                        f5.metric("Realized P&L", f"${realized:,.2f}", delta_color="normal")
                        f6.metric("Total Equity", f"${mkt_val:,.2f}")
                        st.markdown("---")
                    
                    # Filter Dataframe for Detail View
                    view_df = view_df[view_df['Ticker'] == tick_filter]

                # --- 2. DETAILED LIFO TRANSACTION LIST ---
                if not view_df.empty:
                    # Map Campaign Start Date
                    start_map = df_s.set_index('Trade_ID')['Open_Date'].to_dict()
                    view_df['Campaign_Start'] = view_df['Trade_ID'].map(start_map)
                    
                    # Calculate Current Price Proxy for LIFO Math
                    curr_prices = {}
                    for idx, row in df_s[df_s['Status']=='OPEN'].iterrows():
                        if row['Shares'] > 0: 
                            val = row['Total_Cost'] + row.get('Unrealized_PL', 0)
                            curr_prices[row['Trade_ID']] = val / row['Shares']
                    
                    # LIFO Logic for Remaining Shares
                    display_df = view_df.copy()
                    remaining_map = {}
                    
                    for tid in display_df['Trade_ID'].unique():
                        subset = display_df[display_df['Trade_ID'] == tid].copy()
                        # Sort: Date ascending, Buys (0) before Sells (1)
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
                    
                    # Per-Transaction Metrics
                    def calc_unrealized(row): 
                         if row['Action'] == 'BUY' and row['Remaining_Shares'] > 0:
                             price = curr_prices.get(row['Trade_ID'], 0)
                             return (price - row['Amount']) * row['Remaining_Shares']
                         return 0.0
                    display_df['Unrealized_PL'] = display_df.apply(calc_unrealized, axis=1)

                    def calc_return_pct(row):
                        if row['Action'] == 'BUY' and row['Remaining_Shares'] > 0:
                            current_price = curr_prices.get(row['Trade_ID'], 0)
                            entry_price = row['Amount']
                            if entry_price > 0:
                                return ((current_price - entry_price) / entry_price) * 100
                        return 0.0
                    display_df['Return_Pct'] = display_df.apply(calc_return_pct, axis=1)

                    # Visual Adjustments
                    display_df['Realized_PL'] = display_df.apply(lambda x: x['Realized_PL'] if x['Action'] == 'SELL' else 0, axis=1)
                    display_df['Shares'] = display_df.apply(lambda x: -x['Shares'] if x['Action'] == 'SELL' else x['Shares'], axis=1)
                    display_df['Value'] = display_df.apply(lambda x: -x['Value'] if x['Action'] == 'SELL' else x['Value'], axis=1)
                    
                    # Final Columns
                    final_cols = ['Trade_ID', 'Trx_ID', 'Campaign_Start', 'Date', 'Ticker', 'Action', 'Shares', 'Remaining_Shares', 'Amount', 'Stop_Loss', 'Value', 'Realized_PL', 'Unrealized_PL', 'Return_Pct', 'Rule', 'Notes']
                    show_cols = [c for c in final_cols if c in display_df.columns]
                    
                    st.dataframe(
                        display_df[show_cols].sort_values(['Trade_ID', 'Date']).style
                        .format({
                            'Date': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'None', 
                            'Campaign_Start': lambda x: x if isinstance(x, str) else (x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'None'), 
                            'Amount':'${:,.2f}', 'Stop_Loss':'${:,.2f}', 'Value':'${:,.2f}', 
                            'Realized_PL':'${:,.2f}', 'Unrealized_PL':'${:,.2f}', 
                            'Return_Pct':'{:.2f}%', 'Remaining_Shares':'{:.0f}'
                        })
                        .applymap(color_pnl, subset=['Value','Realized_PL','Unrealized_PL', 'Return_Pct'])
                        .applymap(color_neg_value, subset=['Shares']), 
                        height=(len(display_df) + 1) * 35 + 3, 
                        use_container_width=True
                    )
                else: st.info("No matching transactions.")
            else: st.info("No open transactions found.")
        else: st.info("No data available.")

# --- TAB 9: DETAILED TRADE LOG ---
    with tab9:
        st.subheader("Detailed Trade Log (History)")
        # Robust Ticker List Generation (Old Logic)
        unique_tickers_hist = sorted(df_d['Ticker'].dropna().astype(str).unique().tolist())
        
        tick_filter = st.selectbox("Filter History Ticker", ["All"] + unique_tickers_hist, key='hist_det')
        view_df = df_d if tick_filter=="All" else df_d[df_d['Ticker']==tick_filter]
        
        if not view_df.empty:
            display_df = view_df.copy()
            
            # Visual Logic: Make Sells Negative for coloring
            display_df['Shares'] = display_df.apply(lambda x: -x['Shares'] if x['Action'] == 'SELL' else x['Shares'], axis=1)
            display_df['Value'] = display_df.apply(lambda x: -x['Value'] if x['Action'] == 'SELL' else x['Value'], axis=1)
            
            # Columns to Show
            final_cols = ['Trade_ID', 'Trx_ID', 'Date', 'Ticker', 'Action', 'Shares', 'Amount', 'Stop_Loss', 'Value', 'Realized_PL', 'Rule', 'Notes']
            show_cols = [c for c in final_cols if c in display_df.columns]
            
            # Render with Full Formatting (Old Logic restored)
            st.dataframe(
                display_df[show_cols].sort_values(['Trade_ID', 'Date']).style
                .format({
                    'Date': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'None', 
                    'Shares':'{:.0f}', 'Amount':'${:,.2f}', 
                    'Stop_Loss':'${:,.2f}', 'Value':'${:,.2f}', 'Realized_PL':'${:,.2f}'
                })
                .applymap(color_pnl, subset=['Value','Realized_PL'])
                .applymap(color_neg_value, subset=['Shares']),
                use_container_width=True
            )
        else:
            st.info("No transactions found.")

# --- TAB 10: ALL CAMPAIGNS (PRO SCOREBOARD) ---
    with tab10:
        st.subheader("All Campaigns (Summary)")
        
        # 1. Prepare Data
        df_s_view = df_s.reset_index().rename(columns={'index': 'Seq_ID'})
        
        def get_result(row):
            if row['Status'] == 'OPEN': return "OPEN"
            pct = row['Return_Pct']
            return "BE" if -0.5 <= pct <= 0.5 else ("WIN" if pct > 0.5 else "LOSS")
        df_s_view['Result'] = df_s_view.apply(get_result, axis=1)
        
        # 2. Filters
        unique_tickers_sum = sorted(df_s['Ticker'].dropna().astype(str).unique().tolist())
        tick_filter_all = st.selectbox("Filter Campaign Ticker", ["All"] + unique_tickers_sum)
        
        unique_rules = sorted([str(x) for x in df_s['Rule'].unique() if pd.notnull(x)])
        rule_filter = st.multiselect("Filter by Buy Rule", unique_rules)
        res_filter = st.multiselect("Filter by Result", ["WIN", "LOSS", "BE", "OPEN"])
        
        view_all = df_s_view.copy()
        
        # Apply Filters
        if tick_filter_all != "All": view_all = view_all[view_all['Ticker'] == tick_filter_all]
        if rule_filter: view_all = view_all[view_all['Rule'].isin(rule_filter)]
        if res_filter: view_all = view_all[view_all['Result'].isin(res_filter)]
        
        if not view_all.empty:
            # --- 3. THE SCOREBOARD (METRICS ENGINE) ---
            closed_trades = view_all[view_all['Status'] == 'CLOSED']
            
            if not closed_trades.empty:
                # Basic Counts
                total_trades = len(closed_trades)
                wins = closed_trades[closed_trades['Result'] == 'WIN']
                losses = closed_trades[closed_trades['Result'] == 'LOSS']
                
                num_wins = len(wins)
                num_losses = len(losses)
                win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0.0
                
                # Dollar Stats
                gross_profit = wins['Realized_PL'].sum()
                gross_loss = abs(losses['Realized_PL'].sum())
                net_pl = gross_profit - gross_loss
                
                # Averages
                avg_win = gross_profit / num_wins if num_wins > 0 else 0.0
                avg_loss = gross_loss / num_losses if num_losses > 0 else 0.0
                
                # Key Ratios
                pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
                
                # Expectancy (Average R per trade)
                # Formula: (Win% * AvgWin) - (Loss% * AvgLoss)
                win_pct_dec = win_rate / 100
                loss_pct_dec = 1 - win_pct_dec
                expectancy = (win_pct_dec * avg_win) - (loss_pct_dec * avg_loss)
                
                # --- RENDER METRICS ---
                st.markdown("### 🏆 Performance Matrix (Closed Trades)")
                
                # Row 1: The Bottom Line
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Net Profit", f"${net_pl:,.2f}", f"{len(view_all)} Total Campaigns")
                m2.metric("Profit Factor", f"{pf:.2f}", delta="Excellent" if pf > 2.0 else "Needs Work" if pf < 1.0 else "Good")
                m3.metric("Win Rate", f"{win_rate:.1f}%", f"{num_wins}W - {num_losses}L")
                m4.metric("Expectancy", f"${expectancy:,.2f}", "Avg value per trade")
                
                st.markdown("---")
                
                # Row 2: The Edge
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Gross Profit", f"${gross_profit:,.2f}", delta_color="normal")
                k2.metric("Gross Loss", f"-${gross_loss:,.2f}", delta_color="inverse")
                k3.metric("Avg Win", f"${avg_win:,.2f}", delta_color="normal")
                k4.metric("Avg Loss", f"-${avg_loss:,.2f}", f"W/L Ratio: {wl_ratio:.2f}")
                
                st.markdown("---")
            else:
                st.info("No closed trades in this view to calculate metrics.")

            # 4. Duration Calculation
            def calc_days_open(row):
                try:
                    start = pd.to_datetime(row['Open_Date'])
                    end = pd.to_datetime(row['Closed_Date']) if row['Status'] == 'CLOSED' and pd.notna(row['Closed_Date']) else datetime.now()
                    return (end - start).days
                except: return 0
            view_all['Days_Open'] = view_all.apply(calc_days_open, axis=1)
            
            # 5. Table Display
            all_cols = [
                'Seq_ID', 'Trade_ID', 'Ticker', 'Status', 'Result', 
                'Open_Date', 'Closed_Date', 'Days_Open', 
                'Shares', 'Avg_Entry', 'Avg_Exit', 'Total_Cost', 
                'Realized_PL', 'Return_Pct', 
                'Rule', 'Buy_Notes', 'Sell_Rule', 'Sell_Notes'
            ]
            valid_all = [c for c in all_cols if c in df_s_view.columns]
            
            # Sort by Sequence (preserve original order)
            view_all = view_all.sort_values('Seq_ID', ascending=False)
            
            def highlight_status(val): return 'color: red' if val == 'CLOSED' else 'color: green'
            
            st.dataframe(
                view_all[valid_all].style.format({
                    'Open_Date': lambda x: x if isinstance(x, str) else (x.strftime('%Y-%m-%d') if pd.notnull(x) else 'None'), 
                    'Closed_Date': lambda x: x if isinstance(x, str) else (x.strftime('%Y-%m-%d') if pd.notnull(x) else 'None'), 
                    'Shares':'{:.0f}', 'Avg_Entry':'${:,.2f}', 'Avg_Exit':'${:,.2f}', 
                    'Total_Cost':'${:,.2f}', 'Realized_PL':'${:,.2f}', 'Return_Pct':'{:.2f}%'
                })
                .applymap(highlight_status, subset=['Status'])
                .applymap(color_pnl, subset=['Realized_PL'])
                .applymap(color_result, subset=['Result']), 
                hide_index=True,
                use_container_width=True
            )
        else: st.info("No campaigns match your filters.")

# ==============================================================================
# PAGE 11: ANALYTICS (THE BLEED FORENSICS)
# ==============================================================================
elif page == "Analytics":
    st.header(f"ANALYTICS & AUDIT ({CURR_PORT_NAME})")
    
    # 1. LOAD DATA
    if os.path.exists(SUMMARY_FILE):
        df_s = load_data(SUMMARY_FILE)
        
        df_j = pd.DataFrame()
        if os.path.exists(JOURNAL_FILE):
            df_j = load_data(JOURNAL_FILE)

        # --- DATA PREP ---
        df_s['Closed_Date'] = pd.to_datetime(df_s['Closed_Date'], errors='coerce')
        df_s['Open_Date_DT'] = pd.to_datetime(df_s['Open_Date'], errors='coerce')
        
        def clean_num_local(x):
            try:
                if isinstance(x, str): return float(x.replace('$', '').replace(',', '').replace('%', '').strip())
                return float(x)
            except: return 0.0

        if not df_j.empty:
            df_j['Day'] = pd.to_datetime(df_j['Day'], errors='coerce')
            df_j.sort_values('Day', inplace=True)
            for c in ['End NLV', 'Beg NLV', 'Cash -/+', 'Daily $ Change']: 
                if c in df_j.columns: df_j[c] = df_j[c].apply(clean_num_local)
        
        # --- M_TREND LOGIC ---
        @st.cache_data
        def get_mkt_hist_ana():
            try:
                df = yf.Ticker("^IXIC").history(period="5y")
                if df.empty: df = yf.Ticker("SPY").history(period="5y")
                df.index = df.index.tz_localize(None)
                df['21EMA'] = df['Close'].ewm(span=21, adjust=False).mean()
                return df
            except: return pd.DataFrame()
        
        mkt_df = get_mkt_hist_ana()
        
        def get_m_trend(dt_val):
            try:
                if pd.isna(dt_val): return "Unknown"
                d = dt_val.normalize().replace(tzinfo=None)
                if mkt_df.empty: return "No Data"
                idx = mkt_df.index.get_indexer([d], method='nearest')[0]
                if idx < 0 or idx >= len(mkt_df): return "Out of Range"
                if abs((mkt_df.index[idx] - d).days) > 5: return "Unknown"
                return "UP" if mkt_df.iloc[idx]['Close'] > mkt_df.iloc[idx]['21EMA'] else "DOWN"
            except: return "Unknown"
            
        def color_m_trend(val):
            if val == 'UP': return 'color: #2ca02c; font-weight: bold'
            elif val == 'DOWN': return 'color: #ff4b4b; font-weight: bold'
            return 'color: gray'

        if not mkt_df.empty:
            df_s['M_Trend'] = df_s.apply(lambda x: get_m_trend(x['Open_Date_DT']), axis=1)
        else:
            df_s['M_Trend'] = "Unknown" 
        
        # --- CALCULATIONS ---
        all_sorted = df_s.sort_values('Open_Date_DT', ascending=False)
        
        def get_slump_pl(row): 
            return row['Realized_PL'] if row['Status']=='CLOSED' else row.get('Unrealized_PL', 0.0)
        
        all_sorted['Slump_PL'] = all_sorted.apply(get_slump_pl, axis=1)
        
        closed = df_s[df_s['Status']=='CLOSED'].copy()
        
        if 'Rule' in closed.columns: closed['Strat_Rule'] = closed['Rule'].fillna("Unknown")
        elif 'Buy_Rule' in closed.columns: closed['Strat_Rule'] = closed['Buy_Rule'].fillna("Unknown")
        else: closed['Strat_Rule'] = "Unknown"
        if 'Sell_Rule' not in closed.columns: closed['Sell_Rule'] = "Unknown"

        wins = closed[closed['Realized_PL'] > 0]
        losses = closed[closed['Realized_PL'] <= 0]
        
        gross_profit = wins['Realized_PL'].sum() if not wins.empty else 0
        gross_loss = abs(losses['Realized_PL'].sum()) if not losses.empty else 0
        pf_val = gross_profit/gross_loss if gross_loss != 0 else 0
        
        bat_avg = (len(wins)/len(closed) * 100) if not closed.empty else 0
        avg_win = wins['Realized_PL'].mean() if not wins.empty else 0
        avg_loss = losses['Realized_PL'].mean() if not losses.empty else 0
        wl_ratio = abs(avg_win/avg_loss) if avg_loss!=0 else 0.0

        # ==============================================================================
        # TAB ARCHITECTURE
        # ==============================================================================
        tab_perf, tab_dd = st.tabs(["📊 Performance Audit", "📉 Drawdown Detective"])

        # --- TAB 1: EXISTING PERFORMANCE AUDIT ---
        with tab_perf:
            st.subheader("1. The Scoreboard")
            
            if PLOTLY_AVAILABLE:
                fig_pf = go.Figure(go.Indicator(
                    mode = "gauge+number", value = pf_val, title = {'text': "Profit Factor"},
                    gauge = {'axis': {'range': [0, 5]}, 'bar': {'color': "#2ca02c" if pf_val > 1.5 else "orange"},
                             'steps': [{'range': [0, 1], 'color': "lightgray"}, {'range': [1, 5], 'color': "white"}]}
                ))
                fig_pf.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                
                fig_wr = go.Figure(go.Indicator(
                    mode = "gauge+number", value = bat_avg, number = {'suffix': "%"}, title = {'text': "Win Rate"},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1f77b4"},
                             'steps': [{'range': [0, 40], 'color': "#ff4b4b"}, {'range': [40, 100], 'color': "#2ca02c"}]}
                ))
                fig_wr.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                
                fig_wl = go.Figure()
                fig_wl.add_trace(go.Bar(y=[''], x=[avg_win], name='Avg Win', orientation='h', marker=dict(color='#2ca02c'), text=f"${avg_win:,.0f}", textposition='auto'))
                fig_wl.add_trace(go.Bar(y=[''], x=[avg_loss], name='Avg Loss', orientation='h', marker=dict(color='#ff4b4b'), text=f"${avg_loss:,.0f}", textposition='auto'))
                fig_wl.update_layout(title=f"Win/Loss Ratio: {wl_ratio:.2f}", barmode='relative', height=250, margin=dict(l=20, r=20, t=50, b=20), xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='black'), yaxis=dict(showticklabels=False))

                c1, c2, c3 = st.columns(3)
                c1.plotly_chart(fig_pf, use_container_width=True)
                c2.plotly_chart(fig_wr, use_container_width=True)
                c3.plotly_chart(fig_wl, use_container_width=True)
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Batting Avg", f"{bat_avg:.1f}%")
                m2.metric("Profit Factor", f"{pf_val:.2f}")
                m3.metric("Avg Win", f"${avg_win:,.2f}", delta_color="normal")
                m4.metric("Avg Loss", f"${avg_loss:,.2f}", delta_color="inverse")
            
            if len(all_sorted) >= 5:
                st.markdown("---")
                st.subheader("2. Recent Form (Last Initiated)")
                
                def calc_form(n):
                    slice_df = all_sorted.head(n)
                    w = slice_df[slice_df['Slump_PL'] > 0]
                    l = slice_df[slice_df['Slump_PL'] <= 0]
                    win_rate = len(w) / len(slice_df) if len(slice_df) > 0 else 0.0
                    p_fac = w['Slump_PL'].sum() / abs(l['Slump_PL'].sum()) if abs(l['Slump_PL'].sum()) > 0 else 999
                    return win_rate, p_fac
                
                wr_10, pf_10 = calc_form(10)
                wr_20, pf_20 = calc_form(20)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Last 10 Win Rate", f"{wr_10:.1%}", delta="-COLD" if wr_10 < 0.4 else "+OK", delta_color="normal")
                c2.metric("Last 10 Profit Factor", f"{pf_10:.2f}")
                c3.metric("Last 20 Win Rate", f"{wr_20:.1%}")
                c4.metric("Last 20 Profit Factor", f"{pf_20:.2f}")
            
            if not closed.empty:
                strat = closed.groupby('Strat_Rule').agg(
                    Trades=('Trade_ID','count'), 
                    PL=('Realized_PL','sum'), 
                    WinRate=('Realized_PL', lambda x: (x>0).mean())
                ).sort_values('PL', ascending=False)
                
                st.markdown("---")
                st.subheader("3. Strategy Breakdown")
                st.dataframe(strat.style.format({'PL':'${:,.2f}', 'WinRate':'{:.1%}'}))

                st.markdown("---")
                st.subheader("4. Rule Forensics (Drill Down)")
                avail_rules = sorted(closed['Strat_Rule'].astype(str).unique().tolist())
                sel_rule = st.selectbox("Select Strategy Rule", ["None"] + avail_rules)
                
                if sel_rule != "None":
                    rule_trades = closed[closed['Strat_Rule'] == sel_rule].copy()
                    if not rule_trades.empty:
                        r_pl = rule_trades['Realized_PL'].sum()
                        r_cnt = len(rule_trades)
                        r_wr = len(rule_trades[rule_trades['Realized_PL']>0])/r_cnt
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Rule P&L", f"${r_pl:,.2f}", delta_color="normal")
                        c2.metric("Count", r_cnt)
                        c3.metric("Win Rate", f"{r_wr:.1%}")
                        
                        cols_show = ['Trade_ID', 'Ticker', 'M_Trend', 'Open_Date', 'Closed_Date', 'Realized_PL', 'Return_Pct', 'Sell_Rule']
                        valid_cols = [c for c in cols_show if c in rule_trades.columns]
                        
                        st.dataframe(
                            rule_trades[valid_cols].sort_values('Closed_Date', ascending=False)
                            .style.format({'Realized_PL': '${:,.2f}', 'Return_Pct': '{:.2f}%'})
                            .applymap(lambda x: 'color: #2ca02c' if x>0 else 'color: #ff4b4b', subset=['Realized_PL'])
                            .applymap(color_m_trend, subset=['M_Trend'])
                        )
            else:
                st.info("No closed trades yet.")

        # --- TAB 2: DRAWDOWN DETECTIVE (THE BLEED VERSION) ---
        with tab_dd:
            st.subheader("📉 Drawdown Forensics (Uncovering 'The Bleed')")
            st.caption("Separating Realized Losses (Sold) from Open Profit Giveback (Held).")
            
            if not df_j.empty and len(df_j) > 5:
                # 1. TWR CURVE (For Duration & Depth)
                df_j['Adjusted_Beg'] = df_j['Beg NLV'] + df_j['Cash -/+']
                df_j['Day_Ret'] = 0.0
                mask = df_j['Adjusted_Beg'] != 0
                df_j.loc[mask, 'Day_Ret'] = (df_j.loc[mask, 'End NLV'] - df_j.loc[mask, 'Adjusted_Beg']) / df_j.loc[mask, 'Adjusted_Beg']
                df_j['TWR_Curve'] = (1 + df_j['Day_Ret']).cumprod()
                
                # 2. DRAWDOWN PERIODS
                df_j['HWM_TWR'] = df_j['TWR_Curve'].cummax()
                df_j['DD_Pct'] = (df_j['TWR_Curve'] - df_j['HWM_TWR']) / df_j['HWM_TWR']
                
                # 3. IDENTIFY CLUSTERS
                df_j['Is_DD'] = df_j['DD_Pct'] < -0.01 
                df_j['DD_Group'] = (df_j['Is_DD'] != df_j['Is_DD'].shift()).cumsum()
                
                dd_periods = []
                for grp_id, data in df_j[df_j['Is_DD']].groupby('DD_Group'):
                    start_d = data['Day'].min()
                    end_d = data['Day'].max()
                    depth_pct = data['DD_Pct'].min()
                    duration = (end_d - start_d).days + 1
                    
                    # 4. CALCULATE "THE BLEED" (DOLLAR MATH)
                    # Total Dollar Drop = Sum of Daily P&L (Total fluctuations) during this period
                    # This captures market movement accurately, ignoring cash deposits/withdrawals
                    total_pain = data['Daily $ Change'].sum()
                    
                    dd_periods.append({
                        'Start': start_d, 'End': end_d, 
                        'Depth': depth_pct * 100, 
                        'Total_Pain': total_pain, # This is usually negative
                        'Days': duration
                    })
                
                # Sort by Depth
                worst_dds = sorted(dd_periods, key=lambda x: x['Depth'])[:3]
                
                if worst_dds:
                    fig_dd, ax_dd = plt.subplots(figsize=(10, 4))
                    ax_dd.fill_between(df_j['Day'], df_j['DD_Pct']*100, 0, color='red', alpha=0.3)
                    ax_dd.plot(df_j['Day'], df_j['DD_Pct']*100, color='red', linewidth=1)
                    ax_dd.set_title("Drawdown Depth (TWR Basis)")
                    ax_dd.set_ylabel("Drawdown %")
                    st.pyplot(fig_dd)

                    st.markdown("### The Top 3 Drawdown Sequences")
                    cols = st.columns(3)
                    for i, dd in enumerate(worst_dds):
                        with cols[i]:
                            st.error(f"Sequence #{i+1}")
                            st.metric("Max Depth", f"{dd['Depth']:.2f}%", f"{dd['Days']} Days")
                            st.metric("Total Dollar Drop", f"${dd['Total_Pain']:,.0f}", delta_color="inverse")
                            
                            # FORENSICS: Realized vs Giveback
                            # 1. Find Realized Losses in this window
                            mask_loss = (
                                (closed['Closed_Date'] >= dd['Start']) & 
                                (closed['Closed_Date'] <= dd['End'])
                            )
                            trades_in_window = closed[mask_loss]
                            realized_pain = trades_in_window['Realized_PL'].sum() # Net realized P&L
                            
                            # 2. Calculate "The Bleed" (Unrealized/Giveback)
                            # Total Pain = Realized + Bleed
                            # Bleed = Total Pain - Realized
                            # Note: Total_Pain is negative. Realized is usually negative (or small pos).
                            bleed_pain = dd['Total_Pain'] - realized_pain
                            
                            st.markdown("---")
                            st.caption("Breakdown of the Pain:")
                            st.markdown(f"**Realized P&L:** :red[${realized_pain:,.0f}]")
                            st.markdown(f"**The Bleed (Open Profit):** :red[${bleed_pain:,.0f}]")
                            
                            st.markdown("---")
                            # 3. Culprits (Stocks Sold)
                            culprits = trades_in_window[trades_in_window['Realized_PL'] < 0].sort_values('Realized_PL', ascending=True).head(3)
                            
                            if not culprits.empty:
                                st.caption("**Realized Losers:**")
                                for _, trade in culprits.iterrows():
                                    st.write(f"• **{trade['Ticker']}**: :red[${trade['Realized_PL']:,.0f}]")
                            else:
                                st.caption("No realized losses. Pain was purely giving back open profit.")
                else:
                    st.success("No significant drawdowns detected.")
            else:
                st.info("Insufficient daily data.")

    else:
        st.error("No Summary Data Found.")

# ==============================================================================
# PAGE 12: DAILY REPORT CARD (THE FLIGHT RECORDER)
# ==============================================================================
elif page == "Daily Report Card":
    st.header(f"📠 DAILY REPORT CARD ({CURR_PORT_NAME})")
    
    # 1. LOAD ALL DATA
    # We need everything to build a complete picture
    p_clean = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')
    p_legacy = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal.csv')
    path_j = p_clean if os.path.exists(p_clean) else p_legacy
    path_s = os.path.join(DATA_ROOT, portfolio, 'Trade_Log_Summary.csv')
    path_d = os.path.join(DATA_ROOT, portfolio, 'Trade_Log_Details.csv') # For stops/risk if needed
    
    if os.path.exists(path_j):
        df_j = pd.read_csv(path_j)
        df_s = load_data(path_s)
        
        # Data Prep
        df_j['Day'] = pd.to_datetime(df_j['Day'], errors='coerce')
        df_j = df_j.dropna(subset=['Day']).sort_values('Day', ascending=False)
        
        # Helper Clean
        def clean_num_local(x):
            try: return float(str(x).replace('$', '').replace(',', '').replace('%', '').strip())
            except: return 0.0

        for c in ['End NLV', 'Beg NLV', 'Cash -/+', 'Daily $ Change', 'SPY', 'Nasdaq']:
            if c in df_j.columns: df_j[c] = df_j[c].apply(clean_num_local)
        
        # 2. DATE SELECTOR
        # Default to the most recent entry
        available_dates = df_j['Day'].dt.date.unique()
        if len(available_dates) > 0:
            selected_date = st.selectbox("Select Date for Report", available_dates, index=0)
            
            # --- GET DAY'S DATA ---
            # Filter Journal for specific day
            day_stats = df_j[df_j['Day'].dt.date == selected_date].iloc[0]
            
            # Filter Trades for specific day (Closed Today)
            sold_today = pd.DataFrame()
            if not df_s.empty:
                df_s['Closed_Date'] = pd.to_datetime(df_s['Closed_Date'], errors='coerce')
                sold_today = df_s[
                    (df_s['Status'] == 'CLOSED') & 
                    (df_s['Closed_Date'].dt.date == selected_date)
                ]
            
            # Get Open Positions (Snapshot as of right now, or approximation)
            # Note: Historical open positions are hard to reconstruct perfectly without a snapshot table.
            # We will show Current Open Positions if date is Today, otherwise we show "End of Day" snapshot logic if possible.
            # For simplicity, we list *Current* Open Positions but label them clearly.
            open_pos = df_s[df_s['Status'] == 'OPEN'].copy() if not df_s.empty else pd.DataFrame()
            
            # --- CALCULATE RISK STATE (The Logic from Risk Manager) ---
            RESET_DATE = pd.Timestamp("2025-12-16")
            # Filter history up to the selected date for HWM calc
            hist_slice = df_j[df_j['Day'] <= pd.Timestamp(selected_date)].sort_values('Day')
            hist_slice_post = hist_slice[hist_slice['Day'] >= RESET_DATE]
            
            risk_msg = "⚪ NO DATA"
            risk_color = "black"
            dd_pct = 0.0
            
            if not hist_slice_post.empty:
                curr_nlv = hist_slice_post['End NLV'].iloc[-1]
                peak_nlv = hist_slice_post['End NLV'].max()
                dd_pct = ((curr_nlv - peak_nlv) / peak_nlv) * 100 if peak_nlv > 0 else 0.0
                
                if dd_pct >= -5:
                    risk_msg = "🟢 GREEN LIGHT (Normal Operations)"
                    risk_color = "green"
                elif -5 > dd_pct >= -7:
                    risk_msg = "🟡 YELLOW LIGHT (No Margin / Lockout New Buys)"
                    risk_color = "#f1c40f" # Gold
                elif -7 > dd_pct >= -10:
                    risk_msg = "🟠 ORANGE LIGHT (Defensive: Max 30% Invested)"
                    risk_color = "orange"
                else:
                    risk_msg = "🔴 RED LIGHT (CASH ONLY - Protection Mode)"
                    risk_color = "red"

            # --- CALCULATE M-FACTOR (Market State) ---
            # Simple check: Is Nasdaq above 21EMA?
            # We need historical data for this date
            m_state = "Unknown"
            try:
                # We use the SPY/Nasdaq values logged in the journal for that day
                log_spy = day_stats.get('SPY', 0)
                log_ndx = day_stats.get('Nasdaq', 0)
                m_state = f"SPY: ${log_spy:,.2f} | NDX: ${log_ndx:,.2f}"
            except: pass

            # --- GENERATE REPORT (MARKDOWN) ---
            
            # 1. Header & Financials
            nlv = day_stats['End NLV']
            day_dol = day_stats['Daily $ Change']
            # Recalculate Day % to be safe/precise
            prev_adj = day_stats['Beg NLV'] + day_stats['Cash -/+']
            day_pct = (day_dol / prev_adj * 100) if prev_adj != 0 else 0.0
            
            report = f"""
# 📜 DAILY TRADING RECORD
**Date:** {selected_date.strftime('%A, %B %d, %Y')}
**Account:** {CURR_PORT_NAME}
**Net Liquidity:** ${nlv:,.2f}

---

### 1. 🌍 STATE OF THE MARKET (M-FACTOR)
**Levels:** {m_state}
**Daily Context:**
> {day_stats.get('Market_Notes', 'No notes logged.')}

---

### 2. 📊 DAILY PERFORMANCE
| Metric | Value |
| :--- | :--- |
| **Daily P&L ($)** | ${day_dol:+,.2f} |
| **Daily Return** | {day_pct:+.2f}% |
| **Drawdown** | {dd_pct:.2f}% (from Post-Split Peak) |

**Portfolio Actions / Notes:**
> {day_stats.get('Market_Action', 'No actions logged.')}

---

### 3. 🛡️ RISK MANAGER PROTOCOL
**Current Status:** {risk_msg}
**Drawdown Depth:** {dd_pct:.2f}%

**Required Action:**
"""
            if dd_pct >= -5: report += "* **Maintain Discipline:** Follow standard buy/sell rules.\n"
            elif -5 > dd_pct >= -7: report += "* **CAUTION:** Remove Margin. Stop New Buys. Tighten stops.\n"
            elif -7 > dd_pct >= -10: report += "* **DEFENSE:** Reduce exposure to max 30%. Manage winners only.\n"
            else: report += "* **EMERGENCY:** Move to Cash. Cease trading for 48 hours.\n"

            report += "\n---\n\n### 4. 📉 TRADES CLOSED TODAY\n"
            if not sold_today.empty:
                report += "| Ticker | Result P&L | Return % | Reason |\n| :--- | :--- | :--- | :--- |\n"
                for _, row in sold_today.iterrows():
                    report += f"| **{row['Ticker']}** | ${row['Realized_PL']:,.2f} | {row['Return_Pct']:.2f}% | {row.get('Sell_Rule', '')} |\n"
            else:
                report += "*No positions closed today.*\n"

            report += "\n---\n\n### 5. ⚔️ ACTIVE CAMPAIGNS (OPEN POSITIONS)\n"
            if not open_pos.empty:
                report += "| Ticker | Market Value | Open P&L | Exposure % |\n| :--- | :--- | :--- | :--- |\n"
                total_exp = 0
                for _, row in open_pos.iterrows():
                    # Calculate live P&L estimate
                    mkt_val = row['Total_Cost'] + row['Unrealized_PL'] # Approx
                    exp_pct = (mkt_val / nlv) * 100 if nlv != 0 else 0
                    report += f"| **{row['Ticker']}** | ${mkt_val:,.2f} | ${row['Unrealized_PL']:,.2f} | {exp_pct:.1f}% |\n"
                    total_exp += exp_pct
                report += f"\n**Total Exposure:** {total_exp:.1f}%"
            else:
                report += "**100% CASH**"

            # --- RENDER ---
            
            # PREVIEW (Visual)
            st.markdown("### 👁️ Report Preview")
            with st.container(border=True):
                st.markdown(report)
            
            # COPY BLOCK (Raw Text)
            st.markdown("### 📋 Copy for Records")
            st.text_area("Select All + Copy", report, height=300, help="Copy this text into OneNote, Evernote, or your physical journal.")
            
        else:
            st.info("No journal entries found to generate a report.")
    else:
        st.error("Journal file not found.")