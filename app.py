import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math
from datetime import datetime, time, timedelta
import os
import shutil

# Helper function to get current date in Central Time
def get_current_date_ct():
    """Get current date in US Central Time (Chicago)."""
    # UTC to Central Time: -6 hours (CST) or -5 hours (CDT)
    # Using -6 for conservative approach
    from datetime import timezone
    utc_now = datetime.now(timezone.utc)
    ct_now = utc_now - timedelta(hours=6)
    return ct_now.date()

def get_current_time_ct():
    """Get current time in US Central Time (Chicago)."""
    from datetime import timezone
    utc_now = datetime.now(timezone.utc)
    ct_now = utc_now - timedelta(hours=6)
    return ct_now.time()

# Database layer (PostgreSQL support)
try:
    import db_layer as db
    DB_AVAILABLE = True
except (ImportError, KeyError, Exception) as e:
    DB_AVAILABLE = False
    print(f"⚠️  db_layer import failed: {type(e).__name__}: {e}")

# R2 Storage (Cloudflare R2 for images)
try:
    import r2_storage as r2
    R2_AVAILABLE = True
except (ImportError, KeyError, Exception) as e:
    R2_AVAILABLE = False
    print(f"⚠️  r2_storage import failed: {type(e).__name__}: {e}")

# Feature flag: Use database instead of CSV
# Auto-enable if running on Streamlit Cloud with database secrets
if DB_AVAILABLE and hasattr(st, 'secrets') and 'database' in st.secrets:
    USE_DATABASE = True  # Running on Streamlit Cloud with database configured
    print("✅ Database mode enabled (Streamlit Cloud)")
else:
    USE_DATABASE = os.getenv('USE_DATABASE', 'false').lower() == 'true' and DB_AVAILABLE
    if USE_DATABASE:
        print("✅ Database mode enabled (environment variable)")

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
    "br1.5 IPO Base", "br1.6 Flat Base", "br1.7 Consolidation Pivot",
    # 2. Volume & Volatility
    "br2.1 HVE", "br2.2 HVSI", "br2.3 HV1",
    # 3. Moving Average Reclaims
    "br3.1 Reclaim 21e", "br3.2 Reclaim 50s", "br3.3 Reclaim 200s", "br3.4 Reclaim 10W", 
    # 4. Pullbacks
    "br4.1 PB 21e", "br4.2 PB 50s", "br4.3 PB 10w", "br4.4 PB 200s", 
    "br4.5 PB 8e", "br4.6 VWAP",  # <--- UPDATED HERE
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
    "br10.9 Add: Upside Reversal", "br10.10 Add: Consolidation Pivot",
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
    """
    Save DataFrame with verification to prevent data loss.
    If USE_DATABASE is true, also saves to PostgreSQL for validation.
    Returns True if save succeeded, False otherwise.
    """
    # Database mode: Database saves are now handled directly in the calling code
    # This function only saves to CSV for backup purposes
    # (No longer iterates through all rows - that was causing 30+ second delays!)

    # CSV mode (always runs for now - parallel operation)
    try:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        # Determine backup directory (handles case where BACKUP_DIR not yet defined)
        backup_dir = globals().get('BACKUP_DIR', os.path.join(os.path.dirname(filename), 'backups'))
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        # Create backup of existing file
        if os.path.exists(filename):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"{os.path.basename(filename).replace('.csv', '')}_{timestamp}.csv")
            try:
                shutil.copy(filename, backup_path)
            except:
                pass

        # Date formatting for CSV
        if filename in [DETAILS_FILE, SUMMARY_FILE]:
            date_cols = ['Date', 'Open_Date', 'Closed_Date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')

        # Save to temporary file first
        temp_file = filename + '.tmp'
        df.to_csv(temp_file, index=False)

        # Verify file was written correctly
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            # Verify we can read it back
            test_df = pd.read_csv(temp_file)
            if len(test_df) == len(df):
                # Success! Replace the original
                shutil.move(temp_file, filename)
                return True
            else:
                st.error(f"⚠️ Save verification failed for {os.path.basename(filename)}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                return False
        else:
            st.error(f"⚠️ Failed to write {os.path.basename(filename)}")
            return False

    except Exception as e:
        st.error(f"❌ Save error: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False

def load_data(file):
    """
    Load data from CSV or database based on USE_DATABASE flag.
    Maintains backward compatibility with CSV-based code.
    """
    # If database mode is enabled, load from PostgreSQL
    if USE_DATABASE:
        try:
            # Extract portfolio name from file path
            # Path format: portfolios/CanSlim/Trade_Log_Summary.csv
            portfolio_name = portfolio  # Default to current portfolio
            if 'portfolios/' in file or 'portfolios\\' in file:
                # Extract portfolio name from path
                parts = file.replace('\\', '/').split('/')
                if 'portfolios' in parts:
                    idx = parts.index('portfolios')
                    if idx + 1 < len(parts):
                        portfolio_name = parts[idx + 1]

            # Determine which table to query based on filename
            if file.endswith('Trade_Log_Summary.csv') or file.endswith('Summary.csv'):
                df = db.load_summary(portfolio_name)
            elif file.endswith('Trade_Log_Details.csv') or file.endswith('Details.csv'):
                df = db.load_details(portfolio_name)
            elif file.endswith('Trading_Journal_Clean.csv') or file.endswith('Journal.csv'):
                df = db.load_journal(portfolio_name)
            else:
                # Fallback to CSV for unknown files
                if not os.path.exists(file): return pd.DataFrame()
                df = pd.read_csv(file)
                df = clean_dataframe(df)
                return df

            # Database returned data - minimal cleaning needed
            if df.empty:
                return df

            # Ensure Trade_ID is string without .0
            if 'Trade_ID' in df.columns:
                df['Trade_ID'] = df['Trade_ID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            if 'Trx_ID' in df.columns:
                df['Trx_ID'] = df['Trx_ID'].astype(str).str.strip()

            return df

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.error(f"⚠️ Database error: {e}. Falling back to CSV.")
            # Log full traceback for debugging
            print(f"DATABASE ERROR in load_data():\n{error_details}")
            # Fall through to CSV mode

    # CSV mode (original code)
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
            rename_map = {
                'Total_Shares': 'Shares',
                'Close_Date': 'Closed_Date',
                'Cost': 'Amount',
                'Price': 'Amount',
                'Net': 'Value',
                'Buy_Rule': 'Rule'  # Standardize: always use 'Rule'
            }
            df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}, inplace=True)

            # Ensure critical columns exist for Summary files
            if file.endswith('Summary.csv'):
                if 'Rule' not in df.columns:
                    df['Rule'] = ''
                if 'Buy_Notes' not in df.columns:
                    df['Buy_Notes'] = ''
                if 'Sell_Rule' not in df.columns:
                    df['Sell_Rule'] = ''
                if 'Sell_Notes' not in df.columns:
                    df['Sell_Notes'] = ''
                if 'Risk_Budget' not in df.columns:
                    df['Risk_Budget'] = 0.0
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

def load_trade_data():
    """Load and prepare trade detail/summary DataFrames with schema fixes.
    Used by Trade Manager, Log Buy, and Log Sell pages.
    Returns: (df_d, df_s)
    """
    if not os.path.exists(DETAILS_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Amount','Value','Rule','Notes','Realized_PL','Stop_Loss','Trx_ID']).to_csv(DETAILS_FILE, index=False)
    if not os.path.exists(SUMMARY_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Total_Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Rule','Notes','Buy_Notes','Sell_Rule','Sell_Notes']).to_csv(SUMMARY_FILE, index=False)

    df_d = load_data(DETAILS_FILE)
    df_s = load_data(SUMMARY_FILE)

    if 'Risk_Budget' not in df_s.columns:
        df_s['Risk_Budget'] = 0.0

    if 'Buy_Rule' in df_s.columns and 'Rule' not in df_s.columns:
        df_s.rename(columns={'Buy_Rule': 'Rule'}, inplace=True)
    if 'Rule' not in df_s.columns: df_s['Rule'] = ""

    for col in ['Buy_Notes', 'Sell_Rule', 'Sell_Notes']:
        if col not in df_s.columns: df_s[col] = ""

    return df_d, df_s

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

            # Sync to database if enabled
            if USE_DATABASE:
                try:
                    update_data = {
                        'shares': float(df_s.at[i, 'Shares']),
                        'avg_entry': float(df_s.at[i, 'Avg_Entry']),
                        'total_cost': float(df_s.at[i, 'Total_Cost']),
                        'realized_pl': float(df_s.at[i, 'Realized_PL']),
                        'status': df_s.at[i, 'Status']
                    }
                    db.sync_trade_summary(portfolio, trade_id, update_data)
                except Exception as db_err:
                    print(f"Database sync error: {db_err}")

        return df_d, df_s
    except Exception as e:
        print(f"Error updating campaign: {e}")
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

# ==============================================================================
# DATA VALIDATION MODULE
# ==============================================================================

def validate_trade_entry(action, ticker, shares, price, stop_loss=None, trade_id=None, df_s=None):
    """
    Validates trade entry data before saving.
    Returns: (is_valid, error_messages_list)
    """
    errors = []

    # 1. Basic validation
    if not ticker or ticker.strip() == '':
        errors.append("❌ Ticker cannot be empty")

    if shares <= 0:
        errors.append("❌ Shares must be greater than 0")

    if price <= 0:
        errors.append("❌ Price must be greater than 0")

    # 2. Action-specific validation
    if action == 'BUY':
        if stop_loss is not None and stop_loss > 0:
            if stop_loss >= price:
                errors.append(f"❌ Stop loss (${stop_loss:.2f}) must be below entry price (${price:.2f})")

            # Check stop width (warn if > 10%)
            stop_width = ((price - stop_loss) / price) * 100
            if stop_width > 10:
                errors.append(f"⚠️ Warning: Stop is {stop_width:.1f}% wide (recommend < 8%)")

    elif action == 'SELL':
        # Check if trying to sell more than owned
        if df_s is not None and trade_id:
            owned = df_s[df_s['Trade_ID'] == trade_id]['Shares'].sum() if not df_s.empty else 0
            if shares > owned:
                errors.append(f"❌ Cannot sell {shares} shares - you only own {int(owned)}")

    # 3. Duplicate Trade ID check (for new trades)
    if action == 'BUY' and trade_id and df_s is not None:
        if not df_s.empty and trade_id in df_s['Trade_ID'].values:
            errors.append(f"❌ Trade ID '{trade_id}' already exists")

    return len(errors) == 0, errors

def validate_position_size(shares, price, equity, max_pct=25.0):
    """
    Validates position size against equity.
    Returns: (is_valid, warning_message)
    """
    if equity <= 0:
        return True, ""

    position_value = shares * price
    position_pct = (position_value / equity) * 100

    if position_pct > max_pct:
        return False, f"⛔ Position size {position_pct:.1f}% exceeds {max_pct}% limit"
    elif position_pct > (max_pct * 0.8):  # Warn at 80% of limit
        return True, f"⚠️ Warning: Position size is {position_pct:.1f}% (near {max_pct}% limit)"

    return True, ""

def log_audit_trail(action, trade_id, ticker, details, username="User"):
    """
    Logs all trade actions to audit trail.
    Uses database if USE_DATABASE=true, otherwise uses CSV.
    """
    # Database mode
    if USE_DATABASE:
        try:
            db.log_audit(portfolio, action, trade_id, ticker, details, username)
            return True
        except Exception as e:
            print(f"Audit log error (DB): {e}")
            # Fall through to CSV mode

    # CSV mode
    try:
        audit_file = os.path.join(os.path.dirname(DETAILS_FILE), 'Audit_Trail.csv')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        audit_entry = {
            'Timestamp': timestamp,
            'User': username,
            'Action': action,
            'Trade_ID': trade_id,
            'Ticker': ticker,
            'Details': details
        }

        # Load existing audit log or create new
        if os.path.exists(audit_file):
            audit_df = pd.read_csv(audit_file)
        else:
            audit_df = pd.DataFrame(columns=['Timestamp', 'User', 'Action', 'Trade_ID', 'Ticker', 'Details'])

        # Append new entry
        audit_df = pd.concat([audit_df, pd.DataFrame([audit_entry])], ignore_index=True)

        # Keep last 1000 entries only
        if len(audit_df) > 1000:
            audit_df = audit_df.tail(1000)

        audit_df.to_csv(audit_file, index=False)
        return True
    except Exception as e:
        print(f"Audit log error: {e}")
        return False

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
st.sidebar.title("🚀 MO Money")
st.sidebar.markdown("---")

# A. SINGLE STRATEGY SELECTOR
# This variable 'portfolio' controls the entire app context.
portfolio = st.sidebar.selectbox(
    "🔥 Active Strategy",
    [PORT_CANSLIM, PORT_TQQQ, PORT_457B],
    index=0,
    help="Select the account you want to manage."
)

# DEBUG: Database status indicator (lightweight - no queries)
# B. DYNAMIC PATH CONFIGURATION
# We define the paths IMMEDIATELY so every page knows where to look.
ACTIVE_DIR = os.path.join(DATA_ROOT, portfolio)
BACKUP_DIR = os.path.join(ACTIVE_DIR, 'backups') # <--- Added this back for safety

if portfolio == PORT_CANSLIM:
    CURR_PORT_NAME = "CanSlim"
elif portfolio == PORT_TQQQ:
    CURR_PORT_NAME = "TQQQ Strategy"
else:
    CURR_PORT_NAME = "457B Plan"

# Standardized Filenames (Since they are the same for all folders now)
JOURNAL_FILE = os.path.join(ACTIVE_DIR, 'Trading_Journal_Clean.csv')
SUMMARY_FILE = os.path.join(ACTIVE_DIR, 'Trade_Log_Summary.csv')
DETAILS_FILE = os.path.join(ACTIVE_DIR, 'Trade_Log_Details.csv')

st.sidebar.markdown("---")

# C. PAGE NAVIGATION
# Collapsible navigation with expanders (matches Streamlit theme)

# Initialize session state for page selection
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

# Navigation UI
with st.sidebar:
    st.markdown("### 🧭 Navigation")

    # Helper function to create nav button
    def nav_button(label, icon=""):
        icon_text = f"{icon} " if icon else ""
        if st.button(f"{icon_text}{label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

    # 📊 DASHBOARDS (expanded by default)
    with st.expander("📊 Dashboards", expanded=True):
        nav_button("Dashboard", "📊")
        nav_button("Trading Overview", "📈")

    # 💼 TRADING OPERATIONS
    with st.expander("💼 Trading Ops", expanded=True):
        nav_button("Active Campaign Summary", "📋")
        nav_button("Log Buy", "🟢")
        nav_button("Log Sell", "🔴")
        nav_button("Position Sizer", "🔢")
        nav_button("Trade Journal", "📔")
        nav_button("Trade Manager", "📝")

    # 🛡️ RISK MANAGEMENT
    with st.expander("🛡️ Risk Management", expanded=False):
        nav_button("Earnings Planner", "💣")
        nav_button("Portfolio Heat", "🔥")
        nav_button("Risk Manager", "🛡️")

    # 📅 DAILY WORKFLOW
    with st.expander("📅 Daily Workflow", expanded=False):
        nav_button("Daily Journal", "📔")
        nav_button("Daily Report Card", "📊")
        nav_button("Daily Routine", "🌅")
        nav_button("Weekly Retro", "🔄")

    # 📈 MARKET INTELLIGENCE
    with st.expander("📈 Market Intel", expanded=False):
        nav_button("IBD Market School", "🏫")
        nav_button("M Factor", "📊")

    # 🔍 DEEP DIVE
    with st.expander("🔍 Deep Dive", expanded=False):
        nav_button("Analytics", "📈")
        nav_button("Performance Audit", "📊")
        nav_button("Performance Heat Map", "🔥")
        nav_button("Period Review", "⏱️")
        nav_button("Ticker Forensics", "🔬")

    # ⚙️ LEGACY
    with st.expander("⚙️ Legacy", expanded=False):
        nav_button("Dashboard (Legacy)", "⚙️")

# Get page from session state
page = st.session_state.page

# ====== OLD NAVIGATIONS (COMMENTED FOR EASY REVERT) ======
# Option 1: Original radio button
# page = st.sidebar.radio("Go to Module", [
#     "Dashboard", "Trading Overview", "Command Center", "Dashboard (Legacy)",
#     "Daily Routine", "Daily Journal", "Daily Report Card", "IBD Market School",
#     "M Factor", "Performance Heat Map", "Ticker Forensics", "Period Review",
#     "Position Sizer", "Trade Manager", "Analytics", "Weekly Retro"
# ])

# Option 2: option_menu (icon-based but not collapsible)
# from streamlit_option_menu import option_menu
# with st.sidebar:
#     page = option_menu(
#         menu_title="Navigation",
#         options=["Dashboard", "Trading Overview", ...],
#         icons=["speedometer2", "graph-up", ...],
#         default_index=0
#     )
# ====== END OLD NAVIGATIONS ======

st.sidebar.markdown("---")
st.sidebar.caption(f"📂 **Active:** {CURR_PORT_NAME}")


# ==============================================================================
# PAGE 2: DASHBOARD (NEW MODERN VERSION)
# ==============================================================================
if page == "Dashboard":
    st.title("📊 DASHBOARD")
    st.caption(f"Portfolio: {CURR_PORT_NAME} • {datetime.now().strftime('%B %d, %Y')}")

    # === HELPER FUNCTIONS ===
    def fmt_money(val):
        try:
            if isinstance(val, str):
                val = float(val.replace('$', '').replace(',', ''))
            return f"${val:,.2f}"
        except:
            return "$0.00"

    def clean_num_local(x):
        try:
            if isinstance(x, str):
                return float(x.replace('$', '').replace(',', '').replace('%', '').strip())
            return float(x)
        except: return 0.0

    # === LOAD DATA ===
    p_clean = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')
    df_j = load_data(p_clean)
    df_d = load_data(DETAILS_FILE)
    df_s = load_data(SUMMARY_FILE)

    if df_j.empty:
        st.error("Journal data missing. Please log trades in Trade Manager.")
    else:
        # === PREPARE DATA ===
        if 'Nsadaq' in df_j.columns: df_j.rename(columns={'Nsadaq': 'Nasdaq'}, inplace=True)
        df_j['Day'] = pd.to_datetime(df_j['Day'], errors='coerce')
        df_j = df_j.dropna(subset=['Day']).sort_values('Day')

        # Clean numeric columns
        for c in ['Beg NLV', 'End NLV', 'Cash -/+', 'Daily $ Change', 'SPY', 'Nasdaq', '% Invested']:
            if c in df_j.columns: df_j[c] = df_j[c].apply(clean_num_local)

        # Calculate equity curve
        df_j['Adjusted_Beg'] = df_j['Beg NLV'] + df_j['Cash -/+']
        df_j['Daily_Pct'] = 0.0
        mask = df_j['Adjusted_Beg'] != 0
        df_j.loc[mask, 'Daily_Pct'] = (df_j.loc[mask, 'End NLV'] - df_j.loc[mask, 'Adjusted_Beg']) / df_j.loc[mask, 'Adjusted_Beg']
        df_j['Equity_Curve'] = (1 + df_j['Daily_Pct']).cumprod()
        df_j['LTD_Pct'] = (df_j['Equity_Curve'] - 1) * 100

        # Benchmarks
        if 'SPY' in df_j.columns and not df_j['SPY'].eq(0).all():
            start_spy = df_j.loc[df_j['SPY'] > 0, 'SPY'].iloc[0]
            df_j['SPY_Bench'] = ((df_j['SPY'] / start_spy) - 1) * 100
        if 'Nasdaq' in df_j.columns and not df_j['Nasdaq'].eq(0).all():
            start_ndx = df_j.loc[df_j['Nasdaq'] > 0, 'Nasdaq'].iloc[0]
            df_j['NDX_Bench'] = ((df_j['Nasdaq'] / start_ndx) - 1) * 100

        # YTD calculation
        curr_year = datetime.now().year
        df_ytd = df_j[df_j['Day'].dt.year == curr_year].copy()
        ytd_val = 0.0
        ytd_spy = 0.0
        ytd_nasdaq = 0.0
        if not df_ytd.empty:
            ytd_val = ((1 + df_ytd['Daily_Pct']).prod() - 1) * 100

            # SPY YTD
            if 'SPY' in df_j.columns:
                prior_year_data = df_j[(df_j['Day'].dt.year < curr_year) & (df_j['SPY'] > 0)]
                if not prior_year_data.empty:
                    start_s = prior_year_data['SPY'].iloc[-1]
                elif not df_ytd.empty and not df_ytd['SPY'].eq(0).all():
                    start_s = df_ytd.loc[df_ytd['SPY'] > 0, 'SPY'].iloc[0]
                else:
                    start_s = 0.0
                curr_spy = df_j['SPY'].iloc[-1]
                if start_s > 0:
                    ytd_spy = ((curr_spy / start_s) - 1) * 100

            # Nasdaq YTD
            if 'Nasdaq' in df_j.columns:
                prior_year_data_ndx = df_j[(df_j['Day'].dt.year < curr_year) & (df_j['Nasdaq'] > 0)]
                if not prior_year_data_ndx.empty:
                    start_ndx = prior_year_data_ndx['Nasdaq'].iloc[-1]
                elif not df_ytd.empty and not df_ytd['Nasdaq'].eq(0).all():
                    start_ndx = df_ytd.loc[df_ytd['Nasdaq'] > 0, 'Nasdaq'].iloc[0]
                else:
                    start_ndx = 0.0
                curr_nasdaq = df_j['Nasdaq'].iloc[-1]
                if start_ndx > 0:
                    ytd_nasdaq = ((curr_nasdaq / start_ndx) - 1) * 100

        # Live data
        curr_nlv = df_j['End NLV'].iloc[-1]
        daily_dol = df_j['Daily $ Change'].iloc[-1]
        daily_pct_display = df_j['Daily_Pct'].iloc[-1] * 100
        ltd_return = df_j['LTD_Pct'].iloc[-1]

        calc_exposure_pct = 0.0
        num_open_pos = 0
        risk_pct = 0.0

        if not df_s.empty and curr_nlv > 0:
            df_open = df_s[df_s['Status'] == 'OPEN'].copy()
            if not df_open.empty:
                num_open_pos = len(df_open)

                # Get live prices
                tickers = df_open['Ticker'].unique().tolist()
                try:
                    live_batch = yf.download(tickers, period="1d", progress=False)['Close'].iloc[-1]

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
                    df_open['Mkt_Val'] = df_open['Cur_Px'] * df_open['Shares']
                    calc_exposure_pct = (df_open['Mkt_Val'].sum() / curr_nlv) * 100

                    # Calculate risk
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
                    df_open['R_Dol'] = (df_open['Cur_Px'] - df_open['Stop_Loss']) * df_open['Shares']
                    risk_dol = df_open[df_open['R_Dol'] > 0]['R_Dol'].sum()
                    risk_pct = (risk_dol / curr_nlv) * 100
                except:
                    pass

        # === MODERN METRICS CARDS ===
        st.markdown("### 📊 Performance Snapshot")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">Net Liq Value</div>
                <div style="font-size: 32px; font-weight: 700; margin: 8px 0;">${curr_nlv:,.0f}</div>
                <div style="font-size: 16px; color: {'#90EE90' if daily_dol >= 0 else '#ffcccb'};">
                    {'+' if daily_dol >= 0 else ''}{daily_dol:,.0f} ({'+' if daily_pct_display >= 0 else ''}{daily_pct_display:.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">LTD Return</div>
                <div style="font-size: 32px; font-weight: 700; margin: 8px 0;">{ltd_return:.2f}%</div>
                <div style="font-size: 16px;">Life to Date</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 20px; border-radius: 10px; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">YTD Return</div>
                <div style="font-size: 32px; font-weight: 700; margin: 8px 0;">{ytd_val:.2f}%</div>
                <div style="font-size: 16px; color: #f0f0f0;">
                    SPY: {'+' if ytd_spy >= 0 else ''}{ytd_spy:.2f}% | NDX: {'+' if ytd_nasdaq >= 0 else ''}{ytd_nasdaq:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            limit = 12
            mode_color = "#2ca02c" if num_open_pos <= limit else "#ff4b4b"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%); padding: 20px; border-radius: 10px; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">Live Exposure</div>
                <div style="font-size: 32px; font-weight: 700; margin: 8px 0;">{calc_exposure_pct:.1f}%</div>
                <div style="font-size: 16px;">
                    {num_open_pos}/{limit} Pos | Risk: {risk_pct:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # === EQUITY CURVE (INTERACTIVE PLOTLY) ===
        st.markdown("### 📈 Equity Curve")

        # Calculate moving averages
        df_j['EC_10SMA'] = df_j['LTD_Pct'].rolling(window=10).mean()
        df_j['EC_21SMA'] = df_j['LTD_Pct'].rolling(window=21).mean()
        df_j['EC_50SMA'] = df_j['LTD_Pct'].rolling(window=50).mean()
        if 'Nasdaq' in df_j.columns:
            df_j['NDX_21SMA'] = df_j['Nasdaq'].rolling(window=21).mean()

        if PLOTLY_AVAILABLE:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = go.Figure()

            # Fill areas removed for cleaner look
            # (Can be re-enabled if desired)

            # Benchmarks
            if 'SPY_Bench' in df_j.columns:
                lbl_spy = f"SPY ({df_j['SPY_Bench'].iloc[-1]:+.1f}%)"
                fig.add_trace(go.Scatter(
                    x=df_j['Day'], y=df_j['SPY_Bench'],
                    mode='lines',
                    name=lbl_spy,
                    line=dict(color='gray', width=1.5),
                    opacity=0.7
                ))

            if 'NDX_Bench' in df_j.columns:
                lbl_ndx = f"Nasdaq ({df_j['NDX_Bench'].iloc[-1]:+.1f}%)"
                fig.add_trace(go.Scatter(
                    x=df_j['Day'], y=df_j['NDX_Bench'],
                    mode='lines',
                    name=lbl_ndx,
                    line=dict(color='#1f77b4', width=1.5),
                    opacity=0.7
                ))

            # Moving averages
            fig.add_trace(go.Scatter(
                x=df_j['Day'], y=df_j['EC_50SMA'],
                mode='lines',
                name='50 SMA',
                line=dict(color='red', width=1.2)
            ))

            fig.add_trace(go.Scatter(
                x=df_j['Day'], y=df_j['EC_21SMA'],
                mode='lines',
                name='21 SMA',
                line=dict(color='green', width=1.2)
            ))

            fig.add_trace(go.Scatter(
                x=df_j['Day'], y=df_j['EC_10SMA'],
                mode='lines',
                name='10 SMA',
                line=dict(color='purple', width=1.2)
            ))

            # Portfolio (main line - on top)
            lbl_port = f"Portfolio ({df_j['LTD_Pct'].iloc[-1]:+.1f}%)"
            fig.add_trace(go.Scatter(
                x=df_j['Day'], y=df_j['LTD_Pct'],
                mode='lines',
                name=lbl_port,
                line=dict(color='darkblue', width=2.5)
            ))

            # Exposure on secondary y-axis
            fig.add_trace(go.Scatter(
                x=df_j['Day'], y=df_j['% Invested'],
                mode='lines',
                name='Exposure %',
                line=dict(color='#e67e22', width=1),
                fill='tozeroy',
                fillcolor='rgba(230, 126, 34, 0.3)',
                yaxis='y2',
                opacity=0.6
            ))

            # === MARKET REGIME INDICATOR BAR ===
            # Add colored rectangles showing when Nasdaq is above/below 21 EMA
            shapes = []
            annotations = []
            if 'Nasdaq' in df_j.columns and 'NDX_21SMA' in df_j.columns:
                # Filter out NaN values for proper regime detection
                df_regime = df_j[df_j['NDX_21SMA'].notna()].copy()

                if not df_regime.empty:
                    # Create regime indicator (True = green/above, False = red/below)
                    df_regime['Regime'] = df_regime['Nasdaq'] >= df_regime['NDX_21SMA']

                    # Find regime changes to create continuous colored sections
                    df_regime['Regime_Change'] = df_regime['Regime'] != df_regime['Regime'].shift()
                    change_points = df_regime[df_regime['Regime_Change']].index.tolist()

                    # Add start and end points
                    if df_regime.index[0] not in change_points:
                        change_points.insert(0, df_regime.index[0])
                    if df_regime.index[-1] not in change_points:
                        change_points.append(df_regime.index[-1])

                    for i in range(len(change_points) - 1):
                        start_idx = change_points[i]
                        end_idx = change_points[i + 1]

                        regime_val = df_regime.loc[start_idx, 'Regime']
                        start_date = df_regime.loc[start_idx, 'Day']
                        end_date = df_regime.loc[end_idx, 'Day']

                        # Bright green when above 21 EMA, bright red when below
                        color = 'green' if regime_val else 'red'

                        shapes.append(dict(
                            type='rect',
                            xref='x',
                            yref='paper',
                            x0=start_date,
                            x1=end_date,
                            y0=0.97,
                            y1=1.0,
                            fillcolor=color,
                            opacity=0.4,
                            line=dict(width=0),
                            layer='below'
                        ))

                # Add label for market regime bar
                annotations.append(dict(
                    text='MARKET TREND (COMP vs 21s)',
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=0.985,
                    showarrow=False,
                    font=dict(size=9, color='black'),
                    xanchor='center',
                    yanchor='middle'
                ))

            # Layout
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Return %',
                yaxis2=dict(
                    title=dict(text='% Exposure', font=dict(color='#e67e22')),
                    overlaying='y',
                    side='right',
                    range=[0, 1000],
                    tickvals=[0, 100, 200],
                    tickfont=dict(color='#e67e22')
                ),
                hovermode='x unified',
                height=700,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='left',
                    x=0
                ),
                template='plotly_white',
                shapes=shapes,
                annotations=annotations
            )

            # Add 100% exposure reference line
            fig.add_hline(y=100, line_dash='dash', line_color='black',
                         opacity=0.4, line_width=0.8, yref='y2')

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Plotly not available. Install plotly for interactive charts.")

# ==============================================================================
# PAGE 2B: DASHBOARD (LEGACY - ORIGINAL VERSION)
# ==============================================================================
elif page == "Dashboard (Legacy)":
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

    # Load journal data (database-aware via load_data)
    p_clean = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')
    df_j = load_data(p_clean)
    df_d = load_data(DETAILS_FILE)
    df_s = load_data(SUMMARY_FILE)

    if df_j.empty:
        st.error("Journal missing.")
    else:
        
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
            df_j['Adjusted_Beg'] = df_j['Beg NLV'] + df_j['Cash -/+']
            df_j['Daily_Pct'] = 0.0
            
            mask = df_j['Adjusted_Beg'] != 0
            df_j.loc[mask, 'Daily_Pct'] = (df_j.loc[mask, 'End NLV'] - df_j.loc[mask, 'Adjusted_Beg']) / df_j.loc[mask, 'Adjusted_Beg']
            
            df_j['Equity_Curve'] = (1 + df_j['Daily_Pct']).cumprod()
            df_j['LTD_Pct'] = (df_j['Equity_Curve'] - 1) * 100
            
            if 'SPY' in df_j.columns and not df_j['SPY'].eq(0).all():
                 start_spy = df_j.loc[df_j['SPY'] > 0, 'SPY'].iloc[0]
                 df_j['SPY_Bench'] = ((df_j['SPY'] / start_spy) - 1) * 100
            if 'Nasdaq' in df_j.columns and not df_j['Nasdaq'].eq(0).all():
                 start_ndx = df_j.loc[df_j['Nasdaq'] > 0, 'Nasdaq'].iloc[0]
                 df_j['NDX_Bench'] = ((df_j['Nasdaq'] / start_ndx) - 1) * 100

            curr_year = datetime.now().year
            df_ytd = df_j[df_j['Day'].dt.year == curr_year].copy()
            
            ytd_val = 0.0
            ytd_spy = 0.0
            
            if not df_ytd.empty:
                ytd_val = ((1 + df_ytd['Daily_Pct']).prod() - 1) * 100
                if 'SPY' in df_j.columns:
                    prior_year_data = df_j[(df_j['Day'].dt.year < curr_year) & (df_j['SPY'] > 0)]
                    if not prior_year_data.empty:
                        start_s = prior_year_data['SPY'].iloc[-1]
                    elif not df_ytd.empty and not df_ytd['SPY'].eq(0).all():
                        start_s = df_ytd.loc[df_ytd['SPY'] > 0, 'SPY'].iloc[0]
                    else:
                        start_s = 0.0
                    curr_spy = df_j['SPY'].iloc[-1]
                    if start_s > 0:
                        ytd_spy = ((curr_spy / start_s) - 1) * 100
            
            # --- 4. TOP DISPLAY ---
            c1, c2, c3, c4 = st.columns(4)
            daily_dol = df_j['Daily $ Change'].iloc[-1]
            daily_pct_display = df_j['Daily_Pct'].iloc[-1] * 100
            
            c1.metric("Net Liq Value", fmt_money(curr_nlv), f"{daily_dol:+,.2f} ({daily_pct_display:+.2f}%)")
            c2.metric("LTD Return", f"{df_j['LTD_Pct'].iloc[-1]:.2f}%")
            c3.metric("YTD Return", f"{ytd_val:.2f}%", delta=f"{ytd_spy:+.2f}% SPY")
            
            limit = 12
            delta_msg = f"{num_open_pos}/{limit} Pos | Risk: {risk_pct:.2f}%"
            mode = "normal" if num_open_pos <= limit else "inverse"
            c4.metric("Live Exposure", f"{calc_exposure_pct:.1f}%", delta=delta_msg, delta_color=mode)
            
            st.markdown("---")
            
            # --- 5. PLOTS (UPDATED MAs: 10S and 21S) ---
            df_j['EC_10SMA'] = df_j['LTD_Pct'].rolling(window=10).mean()
            df_j['EC_21SMA'] = df_j['LTD_Pct'].rolling(window=21).mean()
            df_j['EC_50SMA'] = df_j['LTD_Pct'].rolling(window=50).mean()
            if 'Nasdaq' in df_j.columns: df_j['NDX_21SMA'] = df_j['Nasdaq'].rolling(window=21).mean()
            
            plt.style.use('bmh')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Top Plot
            if 'Nasdaq' in df_j.columns:
                ax1.fill_between(df_j['Day'], 0.97, 1.0, transform=ax1.transAxes, where=(df_j['Nasdaq']>=df_j['NDX_21SMA']), color='green', alpha=0.4, zorder=0)
                ax1.fill_between(df_j['Day'], 0.97, 1.0, transform=ax1.transAxes, where=(df_j['Nasdaq']<df_j['NDX_21SMA']), color='red', alpha=0.4, zorder=0)
                ax1.text(0.5, 0.985, "MARKET TREND (COMP vs 21s)", transform=ax1.transAxes, ha='center', fontsize=8, fontweight='bold')
            
            lbl_port = f"Portfolio ({df_j['LTD_Pct'].iloc[-1]:+.1f}%)"
            if 'SPY_Bench' in df_j.columns:
                lbl_spy = f"SPY ({df_j['SPY_Bench'].iloc[-1]:+.1f}%)"
                ax1.plot(df_j['Day'], df_j['SPY_Bench'], color='gray', linestyle='-', linewidth=1.5, alpha=0.7, label=lbl_spy)
            if 'NDX_Bench' in df_j.columns:
                lbl_ndx = f"Nasdaq ({df_j['NDX_Bench'].iloc[-1]:+.1f}%)"
                ax1.plot(df_j['Day'], df_j['NDX_Bench'], color='#1f77b4', linestyle='-', linewidth=1.5, alpha=0.7, label=lbl_ndx)
            
            ax1.plot(df_j['Day'], df_j['LTD_Pct'], color='darkblue', linewidth=2.5, label=lbl_port)
            ax1.plot(df_j['Day'], df_j['EC_10SMA'], color='purple', linewidth=1.2, label='10 SMA')
            ax1.plot(df_j['Day'], df_j['EC_21SMA'], color='green', linewidth=1.2, label='21 SMA')
            ax1.plot(df_j['Day'], df_j['EC_50SMA'], color='red', linewidth=1.2, label='50 SMA')
            
            ax1.fill_between(df_j['Day'], df_j['LTD_Pct'], df_j['EC_21SMA'], where=(df_j['LTD_Pct'] >= df_j['EC_21SMA']), interpolate=True, color='green', alpha=0.15)
            ax1.fill_between(df_j['Day'], df_j['LTD_Pct'], df_j['EC_21SMA'], where=(df_j['LTD_Pct'] < df_j['EC_21SMA']), interpolate=True, color='red', alpha=0.15)
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
            ax2.fill_between(df_j['Day'], y_min, y_min + (y_max-y_min)*0.05, where=(df_j['LTD_Pct'] >= df_j['EC_21SMA']), color='green', alpha=0.5)
            ax2.fill_between(df_j['Day'], y_min, y_min + (y_max-y_min)*0.05, where=(df_j['LTD_Pct'] < df_j['EC_21SMA']), color='red', alpha=0.5)
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
# PAGE 2: TRADING OVERVIEW
# ==============================================================================
elif page == "Trading Overview":
    st.title("📊 TRADING OVERVIEW")
    st.caption(f"Portfolio: {CURR_PORT_NAME} • {datetime.now().strftime('%B %d, %Y')}")

    # === DATE FILTER ===
    st.markdown("### 📅 Date Range Filter")

    col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 1])

    today = get_current_date_ct()

    with col_filter1:
        # Preset date ranges
        date_preset = st.selectbox(
            "Quick Select",
            ["All Time", "Today", "This Week", "This Month", "Last 30 Days", "Last Month", "This Quarter", "YTD", "Custom Range"],
            index=0
        )

    # Calculate date range based on preset
    if date_preset == "Today":
        start_date = today
        end_date = today
    elif date_preset == "This Week":
        start_date = today - timedelta(days=today.weekday())
        end_date = today
    elif date_preset == "This Month":
        start_date = today.replace(day=1)
        end_date = today
    elif date_preset == "Last 30 Days":
        start_date = today - timedelta(days=30)
        end_date = today
    elif date_preset == "Last Month":
        first_this_month = today.replace(day=1)
        end_date = first_this_month - timedelta(days=1)
        start_date = end_date.replace(day=1)
    elif date_preset == "This Quarter":
        quarter = (today.month - 1) // 3
        start_date = datetime(today.year, quarter * 3 + 1, 1).date()
        end_date = today
    elif date_preset == "YTD":
        start_date = datetime(today.year, 1, 1).date()
        end_date = today
    elif date_preset == "Custom Range":
        with col_filter2:
            start_date = st.date_input("Start Date", value=today - timedelta(days=90))
        with col_filter3:
            end_date = st.date_input("End Date", value=today)
    else:  # All Time
        start_date = None
        end_date = None

    if date_preset != "Custom Range" and date_preset != "All Time":
        with col_filter2:
            st.info(f"📆 {start_date.strftime('%b %d, %Y')} → {end_date.strftime('%b %d, %Y')}")

    st.markdown("---")

    # === LOAD DATA ===
    df_summary = load_data(SUMMARY_FILE)
    df_journal = load_data(JOURNAL_FILE)
    df_details = load_data(DETAILS_FILE)

    if df_summary.empty and df_journal.empty:
        st.warning("No trading data found. Start logging trades in Trade Manager.")
    else:
        # === HELPER FUNCTION ===
        def clean_num_local(x):
            try:
                if isinstance(x, str):
                    return float(x.replace('$', '').replace(',', '').replace('%', '').strip())
                return float(x)
            except: return 0.0

        # === PREPARE JOURNAL DATA ===
        if not df_journal.empty:
            df_journal['Day'] = pd.to_datetime(df_journal['Day'], errors='coerce')
            df_journal = df_journal.sort_values('Day')

            # Clean numeric columns
            for c in ['Beg NLV', 'End NLV', 'Cash -/+', 'Daily $ Change', 'SPY', 'Nasdaq']:
                if c in df_journal.columns:
                    df_journal[c] = df_journal[c].apply(clean_num_local)

            # Calculate cash-flow-adjusted equity curve on FULL dataset first
            df_journal['Adjusted_Beg'] = df_journal['Beg NLV'] + df_journal['Cash -/+']
            df_journal['Daily_Pct'] = 0.0
            mask = df_journal['Adjusted_Beg'] != 0
            df_journal.loc[mask, 'Daily_Pct'] = (df_journal.loc[mask, 'End NLV'] - df_journal.loc[mask, 'Adjusted_Beg']) / df_journal.loc[mask, 'Adjusted_Beg']
            df_journal['Equity_Curve'] = (1 + df_journal['Daily_Pct']).cumprod()
            df_journal['LTD_Pct'] = (df_journal['Equity_Curve'] - 1) * 100

        # Keep full dataset for YTD calculation AFTER processing
        df_journal_full = df_journal.copy()

        # FILTER BY DATE RANGE (if not "All Time")
        if not df_journal.empty and start_date is not None and end_date is not None:
            df_journal = df_journal[(df_journal['Day'].dt.date >= start_date) & (df_journal['Day'].dt.date <= end_date)]

        # === FILTER SUMMARY DATA BY DATE RANGE ===
        if not df_summary.empty and start_date is not None and end_date is not None:
            if 'Closed_Date' in df_summary.columns:
                df_summary['Closed_Date'] = pd.to_datetime(df_summary['Closed_Date'], errors='coerce')
                # Show closed trades in range OR currently open trades
                df_summary = df_summary[
                    ((df_summary['Closed_Date'].dt.date >= start_date) & (df_summary['Closed_Date'].dt.date <= end_date)) |
                    (df_summary['Status'].str.lower().isin(['active', 'open']))
                ]
        # === 1. CALCULATE ALL METRICS ===
        # Journal metrics
        current_nlv = 0
        daily_change = 0
        ltd_return = 0

        if not df_journal.empty:
            current_nlv = df_journal['End NLV'].iloc[-1]
            daily_change = df_journal.iloc[-1].get('Daily $ Change', 0) if 'Daily $ Change' in df_journal.columns else 0
            ltd_return = df_journal['LTD_Pct'].iloc[-1]

        # Trade statistics
        win_rate = 0
        total_trades = 0
        wins = 0
        losses = 0
        active_trades = 0
        profit_factor = 0
        avg_win = 0
        avg_loss = 0
        max_drawdown = 0
        current_streak = 0
        streak_type = ""

        if not df_summary.empty:
            closed_trades = df_summary[df_summary['Status'].str.lower() == 'closed'].copy()
            closed_trades_count = len(closed_trades)

            if closed_trades_count > 0:
                closed_trades['Realized_PL'] = closed_trades['Realized_PL'].apply(clean_num_local)
                wins = len(closed_trades[closed_trades['Realized_PL'] > 0])
                losses = closed_trades_count - wins
                win_rate = (wins / closed_trades_count) * 100

                # Profit Factor
                total_wins = closed_trades[closed_trades['Realized_PL'] > 0]['Realized_PL'].sum()
                total_losses = abs(closed_trades[closed_trades['Realized_PL'] < 0]['Realized_PL'].sum())
                if total_losses > 0:
                    profit_factor = total_wins / total_losses

                # Avg Win/Loss
                if wins > 0:
                    avg_win = closed_trades[closed_trades['Realized_PL'] > 0]['Realized_PL'].mean()
                if losses > 0:
                    avg_loss = closed_trades[closed_trades['Realized_PL'] < 0]['Realized_PL'].mean()

                # Current Streak
                if 'Closed_Date' in closed_trades.columns:
                    closed_trades['Closed_Date'] = pd.to_datetime(closed_trades['Closed_Date'], errors='coerce')
                    recent_trades = closed_trades.sort_values('Closed_Date', ascending=False)
                    if not recent_trades.empty:
                        last_result = 'W' if recent_trades.iloc[0]['Realized_PL'] > 0 else 'L'
                        current_streak = 1
                        for idx, trade in recent_trades.iloc[1:].iterrows():
                            result = 'W' if trade['Realized_PL'] > 0 else 'L'
                            if result == last_result:
                                current_streak += 1
                            else:
                                break
                        streak_type = "Win" if last_result == 'W' else "Loss"

            # Count active trades
            active_trades = len(df_summary[df_summary['Status'].str.lower().isin(['active', 'open'])])

            # Total trades = closed + active
            total_trades = closed_trades_count + active_trades

        # Max Drawdown from equity curve
        if not df_journal.empty and 'Equity_Curve' in df_journal.columns:
            running_max = df_journal['Equity_Curve'].cummax()
            drawdown = (df_journal['Equity_Curve'] - running_max) / running_max
            max_drawdown = drawdown.min() * 100  # Convert to percentage

        # YTD Return with SPY and Nasdaq comparison
        ytd_return = 0
        ytd_spy_delta = 0
        ytd_nasdaq_delta = 0
        spy_ytd = 0
        nasdaq_ytd = 0

        if not df_journal_full.empty and 'Daily_Pct' in df_journal_full.columns:
            curr_year = datetime.now().year
            df_ytd = df_journal_full[df_journal_full['Day'].dt.year == curr_year].copy()

            if not df_ytd.empty:
                ytd_return = ((1 + df_ytd['Daily_Pct']).prod() - 1) * 100

                # SPY YTD Return (use YTD data only)
                if 'SPY' in df_ytd.columns and not df_ytd['SPY'].eq(0).all():
                    spy_data = df_ytd[df_ytd['SPY'] > 0]
                    if not spy_data.empty:
                        start_spy = spy_data['SPY'].iloc[0]  # First SPY value of the year
                        curr_spy = spy_data['SPY'].iloc[-1]  # Latest SPY value of the year
                        if start_spy > 0:
                            spy_ytd = ((curr_spy / start_spy) - 1) * 100
                            ytd_spy_delta = ytd_return - spy_ytd

                # Nasdaq YTD Return (use YTD data only)
                if 'Nasdaq' in df_ytd.columns and not df_ytd['Nasdaq'].eq(0).all():
                    nasdaq_data = df_ytd[df_ytd['Nasdaq'] > 0]
                    if not nasdaq_data.empty:
                        start_nasdaq = nasdaq_data['Nasdaq'].iloc[0]  # First Nasdaq value of the year
                        curr_nasdaq = nasdaq_data['Nasdaq'].iloc[-1]  # Latest Nasdaq value of the year
                        if start_nasdaq > 0:
                            nasdaq_ytd = ((curr_nasdaq / start_nasdaq) - 1) * 100
                            ytd_nasdaq_delta = ytd_return - nasdaq_ytd

        # Live Exposure (from current positions)
        live_exposure_pct = 0
        num_positions = 0
        risk_pct = 0
        if not df_summary.empty and current_nlv > 0:
            df_open = df_summary[df_summary['Status'].str.lower().isin(['active', 'open'])].copy()
            if not df_open.empty:
                num_positions = len(df_open)

                # Get live prices
                try:
                    tickers = df_open['Ticker'].unique().tolist()
                    live_batch = yf.download(tickers, period="1d", progress=False)['Close'].iloc[-1]

                    def get_live_price(r):
                        try:
                            if len(tickers) == 1:
                                val = float(live_batch) if not pd.isna(live_batch) else float(r.get('Avg_Entry', 0))
                                return val
                            else:
                                val = live_batch.get(r['Ticker'])
                                return float(val) if not pd.isna(val) else float(r.get('Avg_Entry', 0))
                        except:
                            return float(r.get('Avg_Entry', 0))

                    df_open['Cur_Px'] = df_open.apply(get_live_price, axis=1)
                    df_open['Mkt_Val'] = df_open['Cur_Px'] * df_open.get('Shares', 0)
                    live_exposure_pct = (df_open['Mkt_Val'].sum() / current_nlv) * 100

                    # Calculate risk if stop loss data available
                    if not df_details.empty:
                        def get_true_stop(trade_id):
                            txs = df_details[df_details['Trade_ID'] == trade_id]
                            if txs.empty: return 0.0
                            if 'Date' in txs.columns:
                                txs['Date'] = pd.to_datetime(txs['Date'], errors='coerce')
                                txs = txs.sort_values('Date')
                            valid_stops = txs['Stop_Loss'].dropna()
                            valid_stops = valid_stops[valid_stops > 0.01]
                            if not valid_stops.empty: return float(valid_stops.iloc[-1])
                            return 0.0

                        df_open['Stop_Loss'] = df_open['Trade_ID'].apply(get_true_stop)
                        df_open['R_Dol'] = (df_open['Cur_Px'] - df_open['Stop_Loss']) * df_open.get('Shares', 0)
                        risk_dol = df_open[df_open['R_Dol'] > 0]['R_Dol'].sum()
                        risk_pct = (risk_dol / current_nlv) * 100
                except:
                    pass

        # === 2. DISPLAY METRICS CARDS ===
        st.markdown("### 📊 Performance Snapshot")

        # Row 0: Featured metrics (YTD & Live Exposure)
        col_ytd, col_live = st.columns(2)

        with col_ytd:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 25px; border-radius: 12px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 16px; opacity: 0.9; font-weight: 600;">YTD Return</div>
                <div style="font-size: 48px; font-weight: 800; margin: 10px 0;">{ytd_return:.2f}%</div>
                <div style="font-size: 18px; opacity: 0.95;">
                    SPY: {'+' if spy_ytd >= 0 else ''}{spy_ytd:.2f}% | NDX: {'+' if nasdaq_ytd >= 0 else ''}{nasdaq_ytd:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_live:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%); padding: 25px; border-radius: 12px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 16px; opacity: 0.9; font-weight: 600;">Live Exposure</div>
                <div style="font-size: 48px; font-weight: 800; margin: 10px 0;">{live_exposure_pct:.1f}%</div>
                <div style="font-size: 18px; opacity: 0.95;">
                    ↑ {num_positions}/12 Pos | Risk: {risk_pct:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)

        # Row 1: Primary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">Account Balance & P&L</div>
                <div style="font-size: 32px; font-weight: 700; margin: 8px 0;">${current_nlv:,.0f}</div>
                <div style="font-size: 16px; color: {'#90EE90' if daily_change >= 0 else '#ffcccb'};">
                    {'+' if daily_change >= 0 else ''}{daily_change:,.0f} • {ltd_return:+.1f}% LTD
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">Trade Win %</div>
                <div style="font-size: 32px; font-weight: 700; margin: 8px 0;">{win_rate:.1f}%</div>
                <div style="font-size: 16px;">{wins} W / {losses} L</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">Avg Win/Loss Trade</div>
                <div style="font-size: 32px; font-weight: 700; margin: 8px 0;">{avg_win_loss_ratio:.2f}</div>
                <div style="font-size: 16px;">${avg_win:,.0f} / ${avg_loss:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 10px; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">Profit Factor</div>
                <div style="font-size: 32px; font-weight: 700; margin: 8px 0;">{profit_factor:.2f}</div>
                <div style="font-size: 16px;">{'Profitable' if profit_factor > 1 else 'Losing'}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)

        # Row 2: Secondary metrics
        col5, col6, col7 = st.columns(3)

        with col5:
            streak_color = "#2ca02c" if streak_type == "Win" else "#ff4b4b"
            st.markdown(f"""
            <div style="background: {streak_color}; padding: 20px; border-radius: 10px; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">Current Streak</div>
                <div style="font-size: 32px; font-weight: 700; margin: 8px 0;">{current_streak}</div>
                <div style="font-size: 16px;">{streak_type} Streak</div>
            </div>
            """, unsafe_allow_html=True)

        with col6:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%); padding: 20px; border-radius: 10px; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">Total Trades</div>
                <div style="font-size: 32px; font-weight: 700; margin: 8px 0;">{total_trades}</div>
                <div style="font-size: 16px;">{active_trades} Active</div>
            </div>
            """, unsafe_allow_html=True)

        with col7:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); padding: 20px; border-radius: 10px; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">Max Drawdown</div>
                <div style="font-size: 32px; font-weight: 700; margin: 8px 0;">{max_drawdown:.1f}%</div>
                <div style="font-size: 16px;">Peak to Trough</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # === 3. EQUITY CURVE (CASH-FLOW ADJUSTED) ===
        st.markdown("### 📈 Equity Curve (Life-to-Date % Return)")

        if not df_journal.empty and 'LTD_Pct' in df_journal.columns:
            if PLOTLY_AVAILABLE:
                import plotly.graph_objects as go
                fig = go.Figure()

                # Main equity curve
                fig.add_trace(go.Scatter(
                    x=df_journal['Day'],
                    y=df_journal['LTD_Pct'],
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#667eea', width=3),
                    fillcolor='rgba(102, 126, 234, 0.2)',
                    name='Portfolio Return %'
                ))

                # Zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

                fig.update_layout(
                    title='Portfolio Returns (Adjusted for Cash Flows)',
                    xaxis_title='Date',
                    yaxis_title='Return %',
                    hovermode='x unified',
                    height=450,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df_journal['Day'], df_journal['LTD_Pct'], linewidth=2.5, color='#667eea', label='Portfolio Return %')
                ax.fill_between(df_journal['Day'], df_journal['LTD_Pct'], alpha=0.3, color='#667eea')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_title('Portfolio Returns (Adjusted for Cash Flows)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Return %')
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("No equity curve data available")

# ==============================================================================
# PAGE 3: DAILY JOURNAL (CLEAN & FINAL)
# ==============================================================================
elif page == "Daily Journal":
    st.header(f"DAILY TRADING JOURNAL ({CURR_PORT_NAME})")
    
    TARGET_FILE = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')

    # Load journal data (database-aware via load_data)
    df_j = load_data(TARGET_FILE)

    if df_j.empty:
        st.warning("No Journal File Found. Please go to 'Daily Routine' to log your first day.")
    else:
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
                    today = get_current_date_ct()
                    start_week = today - timedelta(days=today.weekday()) 
                    df_view = df_calc[df_calc['Day'].dt.date >= start_week]
                elif view_opt == "By Month":
                    df_calc['Month_Str'] = df_calc['Day'].dt.strftime('%B %Y')
                    months = sorted(df_calc['Month_Str'].unique().tolist(), key=lambda x: datetime.strptime(x, '%B %Y'), reverse=True)
                    sel_month = st.selectbox("Select Month", months) if months else None
                    df_view = df_calc[df_calc['Month_Str'] == sel_month] if sel_month else df_calc
                else:
                    df_view = df_calc

                show_cols = [
                    'Day', 
                    'Score',          # <--- Process first!
                    'Daily_Pct', 
                    'SPY_Pct', 
                    'Nasdaq_Pct', 
                    'Market_Action', 
                    'Mistakes', 
                    'Top_Lesson',
                    'Highlights', 
                    'Lowlights', 
                    'Market_Notes',
                    'End NLV'         # <--- P&L last
                ]
                valid_cols = [c for c in show_cols if c in df_view.columns]
                
                def color_score(val):
                    """Colors the Score column based on the 1-5 scale."""
                    color = ''
                    if val == 5: color = 'background-color: #008000; color: white;' # Dark Green
                    elif val == 4: color = 'background-color: #90EE90; color: black;' # Light Green
                    elif val == 3: color = 'background-color: #FFFFE0; color: black;' # Light Yellow
                    elif val == 2: color = 'background-color: #FFD700; color: black;' # Gold/Orange
                    elif val == 1: color = 'background-color: #FF4B4B; color: white;' # Red
                    return color

                st.dataframe(
                    df_view.sort_values('Day', ascending=False)[valid_cols]
                    .style.format({
                        'Day': '{:%m/%d/%y}', 
                        'End NLV': '${:,.2f}', 
                        'Daily_Pct': '{:+.2f}%', 
                        'SPY_Pct': '{:+.2f}%', 
                        'Nasdaq_Pct': '{:+.2f}%',
                        'Score': '{:.0f}'
                    })
                    .applymap(color_pnl, subset=[c for c in ['Daily_Pct', 'SPY_Pct', 'Nasdaq_Pct'] if c in df_view.columns]) 
                    .applymap(color_score, subset=['Score']), 
                    hide_index=True, 
                    use_container_width=True
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
        # Dual-index confirmation logic
        nasdaq_state = nasdaq['state']
        spy_state = spy['state']

        # Combined state logic
        if nasdaq_state == "POWERTREND" or spy_state == "POWERTREND":
            combined_state = "POWERTREND"  # Either index in power trend
        elif nasdaq_state == "CLOSED" and spy_state == "CLOSED":
            combined_state = "CLOSED"  # Both violated 50 SMA
        elif nasdaq_state in ["NEUTRAL", "CLOSED"] or spy_state in ["NEUTRAL", "CLOSED"]:
            combined_state = "NEUTRAL"  # At least one violated 21 EMA
        else:
            combined_state = "OPEN"  # Both above 21 EMA

        status = combined_state

        if status == "POWERTREND":
            bg = "#8A2BE2"
            exp = "200% (Margin Enabled)"
            explanation = "Either index in super cycle - aggressive"
        elif status == "OPEN":
            bg = "#2ca02c"
            exp = "100% (Full Exposure)"
            explanation = "Both indices above 21 EMA - healthy"
        elif status == "NEUTRAL":
            bg = "#ffcc00"
            exp = "50% Max (Caution)"
            explanation = "1+ index violated 21 EMA - hold winners, avoid new buys"
        else:
            bg = "#ff4b4b"
            exp = "0% (Defensive)"
            explanation = "Both indices violated 50 SMA - protect capital"

        st.markdown(f"""<div class="market-banner" style="background-color: {bg};"><div style="font-size: 14px; opacity: 0.9;">MARKET WINDOW</div><div style="font-size: 48px; font-weight: 800; margin: 5px 0;">{status}</div><div style="font-size: 16px;">RECOMMENDED EXPOSURE: {exp}</div><div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">{explanation}</div></div>""", unsafe_allow_html=True)

        # === IBD EXPOSURE CROSS-REFERENCE ===
        if USE_DATABASE:
            try:
                df_signals = db.load_market_signals(days=5)  # Get recent data

                nasdaq_ibd = df_signals[df_signals['symbol'] == '^IXIC'].iloc[0] if not df_signals[df_signals['symbol'] == '^IXIC'].empty else None
                spy_ibd = df_signals[df_signals['symbol'] == 'SPY'].iloc[0] if not df_signals[df_signals['symbol'] == 'SPY'].empty else None

                if nasdaq_ibd is not None and spy_ibd is not None:
                    st.markdown("---")
                    st.markdown("### 📚 IBD Market School Comparison")

                    col_ibd1, col_ibd2, col_ibd3 = st.columns(3)

                    with col_ibd1:
                        ibd_nasdaq_exp = nasdaq_ibd['market_exposure']
                        ibd_nasdaq_dd = nasdaq_ibd['distribution_count']
                        st.metric("IBD Nasdaq Exposure", f"{ibd_nasdaq_exp}/6",
                                 delta=f"{ibd_nasdaq_dd} distribution days")

                    with col_ibd2:
                        ibd_spy_exp = spy_ibd['market_exposure']
                        ibd_spy_dd = spy_ibd['distribution_count']
                        st.metric("IBD SPY Exposure", f"{ibd_spy_exp}/6",
                                 delta=f"{ibd_spy_dd} distribution days")

                    with col_ibd3:
                        avg_ibd_exp = (ibd_nasdaq_exp + ibd_spy_exp) / 2
                        st.metric("IBD Average", f"{avg_ibd_exp:.1f}/6")

                    st.caption("ℹ️ For reference: IBD uses distribution day count. M Factor uses moving average violations + leadership behavior.")
            except Exception as e:
                st.caption(f"IBD data unavailable: {e}")

        st.markdown("---")
        c1, c2 = st.columns(2)
        
        def make_card_html(title, d, individual_state):
            def arr(v): return "⬆" if v>0 else "⬇"
            def col(v): return "#2ca02c" if v>0 else "#ff4b4b"

            # State badge color
            state_colors = {
                'POWERTREND': '#8A2BE2',
                'OPEN': '#2ca02c',
                'NEUTRAL': '#ffcc00',
                'CLOSED': '#ff4b4b'
            }
            state_bg = state_colors.get(individual_state, '#999')

            # --- FLATTENED STRING TO PREVENT MARKDOWN CODE BLOCK ERROR ---
            html = f"""<div class="ticker-box"><div style="display:flex; justify-content:space-between; border-bottom: 2px solid #f0f0f0; padding-bottom:15px; margin-bottom:15px;"><span style="font-size:20px; font-weight:700;">{title}</span><span style="font-size:20px; color:#555;">${d['price']:,.2f}</span></div><div style="background:{state_bg}; color:white; padding:8px; border-radius:5px; text-align:center; margin-bottom:12px; font-weight:700;">{individual_state}</div><div class="metric-row"><div><span style="font-weight:600;">Short (21e)</span> <span class="sub-text">({d['ema21']:,.2f})</span></div><div style="font-weight:700; color:{col(d['d21'])};">{arr(d['d21'])} {d['s21']} <span style="font-size:13px; opacity:0.8;">({abs(d['d21']):.2f}%)</span></div></div><div class="metric-row"><div><span style="font-weight:600;">Med (50s)</span> <span class="sub-text">({d['sma50']:,.2f})</span></div><div style="font-weight:700; color:{col(d['d50'])};">{arr(d['d50'])} {d['s50']} <span style="font-size:13px; opacity:0.8;">({abs(d['d50']):.2f}%)</span></div></div><div class="metric-row"><div><span style="font-weight:600;">Long (200s)</span> <span class="sub-text">({d['sma200']:,.2f})</span></div><div style="font-weight:700; color:{col(d['d200'])};">{arr(d['d200'])} {d['s200']} <span style="font-size:13px; opacity:0.8;">({abs(d['d200']):.2f}%)</span></div></div></div>"""
            return html

        with c1: st.markdown(make_card_html("NASDAQ", nasdaq, nasdaq_state), unsafe_allow_html=True)
        with c2: st.markdown(make_card_html("S&P 500", spy, spy_state), unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("📖 M Factor Methodology"):
            st.markdown("""
### Market Phases

**🟣 POWER TREND (200% - Margin)**
- Either Nasdaq OR SPY shows super cycle signal
- Super cycle = Low > 21 EMA for 3 consecutive days + up day on 3rd day
- Maximum aggression - ride leaders hard

**🟢 GROW (100% - Full Exposure)**
- Both Nasdaq AND SPY above 21 EMA
- Healthy market - normal buying and holding

**🟡 NEUTRAL (50% Max - Caution)**
- 1 or both indices violated 21 EMA
- Defensive mode - hold existing winners, avoid new buys
- Rely on individual stock signals for position management

**🔴 PROTECT (0% - Defensive)**
- Both Nasdaq AND SPY violated 50 SMA
- No new buys - protect capital
- Only hold positions with strong individual charts

### Philosophy

**21 EMA is the primary filter** - violation signals immediate caution

**Dual-index confirmation** - requires both indices to agree before downgrading

**Stock-first approach** - if your holdings are outperforming, don't panic sell just because indices are weak

**IBD cross-reference** - use distribution day count as a "paranoia check" but don't let it override strong individual stock performance
            """)
    else: st.error("Market Data Unavailable")

# ==============================================================================
# Performance Heat Map
# ==============================================================================
elif page == "Performance Heat Map":
    st.header(f"🔥 PERFORMANCE HEAT MAP ({CURR_PORT_NAME})")
    
    # --- 1. DATA LOADING ---
    df_s = load_data(SUMMARY_FILE)
    df_j = load_data(JOURNAL_FILE)
    
    if PLOTLY_AVAILABLE:
        if not df_s.empty:
            # --- 2. SELECTOR CONTROLS ---
            c_v1, c_v2 = st.columns([2, 1])
            view_mode = c_v1.radio("Portfolio Filter", ["All 2026 Trades", "Open Only", "Closed Only"], horizontal=True)
            metric_mode = c_v2.radio("Heat Metric", ["Return %", "R-Multiple", "Account Impact %"], horizontal=True)
            
            # Base Data Prep
            df_s['Open_DT'] = pd.to_datetime(df_s['Open_Date'], errors='coerce')
            df_s['Close_DT'] = pd.to_datetime(df_s['Closed_Date'], errors='coerce')
            cutoff = pd.Timestamp("2026-01-01")
            
            # Filter Logic
            df_heat = df_s[(df_s['Open_DT'] >= cutoff) | (df_s['Status'] == 'OPEN') | (df_s['Close_DT'] >= cutoff)].copy()
            
            if view_mode == "Open Only": df_heat = df_heat[df_heat['Status'] == 'OPEN']
            elif view_mode == "Closed Only": df_heat = df_heat[df_heat['Status'] == 'CLOSED']

            if not df_heat.empty:
                # --- 3. UPDATED CALCULATION ENGINE ---
                def get_metric_data(row):
                    # Stock Return %
                    ret_val = row.get('Return_Pct', 0.0)
                    
                    # R-Multiple
                    budget = row.get('Risk_Budget', 500.0)
                    total_pl = row['Realized_PL'] if row['Status'] == 'CLOSED' else (row['Realized_PL'] + row.get('Unrealized_PL', 0))
                    r_val = total_pl / budget if budget > 0 else 0.0
                    
                    # NLV Impact % (Fixed Math)
                    impact_val = 0.0
                    if not df_j.empty and 'Day' in df_j.columns:
                        trade_date = row['Open_DT']
                        
                        # 1. Convert everything to datetime to ensure match
                        df_j['Day_DT'] = pd.to_datetime(df_j['Day'], errors='coerce')
                        
                        # 2. Find the LATEST journal entry that is <= the trade open date
                        mask = df_j['Day_DT'] <= trade_date
                        historical_entries = df_j.loc[mask].sort_values('Day_DT')
                        
                        if not historical_entries.empty:
                            # Get the actual NLV from that specific day
                            nlv_at_open = historical_entries['End NLV'].iloc[-1]
                            
                            # 3. Final Calculation Check
                            if nlv_at_open > 0:
                                impact_val = (total_pl / nlv_at_open) * 100
                    
                    return pd.Series([ret_val, r_val, impact_val])

                    # Add this temporary debug line inside 'get_metric_data'
                    if row['Ticker'] == 'AMKR':
                        st.write(f"DEBUG AMKR: P&L=${total_pl:.2f} | Found NLV=${nlv_at_open:.2f} | Impact={impact_val:.2f}%")

                df_heat[['M_Ret', 'M_R', 'M_Impact']] = df_heat.apply(get_metric_data, axis=1)
                
                # Dynamic Setup
                if metric_mode == "Return %":
                    active_col, z_min, z_max, fmt, suffix = 'M_Ret', -7.0, 15.0, ".1f", "%"
                elif metric_mode == "R-Multiple":
                    active_col, z_min, z_max, fmt, suffix = 'M_R', -1.2, 3.0, ".2f", "R"
                else: # Account Impact %
                    active_col, z_min, z_max, fmt, suffix = 'M_Impact', -1.0, 2.0, ".2f", "% Impact"

                df_heat = df_heat.sort_values(active_col, ascending=False)

                # --- 4. GRID DATA WITH TRADE ID ---
                tickers = df_heat['Ticker'].tolist()
                values = df_heat[active_col].astype(float).tolist()
                # Adding Trade ID to labels
                trade_ids = [str(row.get('Trade_ID', 'N/A')) for _, row in df_heat.iterrows()]
                statuses = [('O' if s=='OPEN' else 'C') for s in df_heat['Status']]
                
                labels = [f"{t}<br>({s})" for t, s in zip(tickers, statuses)]
                hover_labels = [f"<b>{t}</b> (ID: {tid})<br>Status: {s}" for t, tid, s in zip(tickers, trade_ids, statuses)]
                
                cols = 8
                rows = math.ceil(len(tickers) / cols)
                z_data = [values[i:i + cols] + [0]*(cols - len(values[i:i + cols])) for i in range(0, len(values), cols)]
                text_data = [labels[i:i + cols] + [""]*(cols - len(labels[i:i + cols])) for i in range(0, len(labels), cols)]
                hover_data = [hover_labels[i:i + cols] + [""]*(cols - len(hover_labels[i:i + cols])) for i in range(0, len(hover_labels), cols)]
                
                # --- 5. CREATE PLOTLY HEATMAP ---
                import plotly.graph_objects as go
                fig = go.Figure(data=go.Heatmap(
                    z=z_data, text=text_data,
                    customdata=hover_data,
                    texttemplate=f"<b>%{{text}}</b><br>%{{z:{fmt}}}{suffix}",
                    hovertemplate=" %{customdata}<br>" + f"{metric_mode}: %{{z:{fmt}}}{suffix}<extra></extra>",
                    colorscale=[[0, '#ff4b4b'], [abs(z_min)/(z_max-z_min), '#ffffff'], [1, '#2ca02c']],
                    zmin=z_min, zmax=z_max, showscale=True, xgap=4, ygap=4
                ))
                
                fig.update_layout(
                    height=200 + (rows * 100), 
                    xaxis=dict(visible=False), yaxis=dict(visible=False, autorange='reversed'),
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # --- 6. AUDIT FOOTER ---
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                fatalities = len(df_heat[df_heat['M_Impact'] < -1.0])
                c1.metric("Fatal Hits (>1% Portfolio)", f"{fatalities} Trades", delta="Target: 0", delta_color="inverse")
                c2.metric("Avg Portfolio Impact", f"{df_heat['M_Impact'].mean():.2f}%")
                
                worst_idx = values.index(min(values))
                c3.error(f"Worst Impact: {tickers[worst_idx]} ({min(values):{fmt}}{suffix})")

            else: st.info("No trades match this view.")
        else: st.warning("Trade Log (df_s) is empty.")

elif page == "Ticker Forensics":
    st.header("🔍 TICKER FORENSICS")
    
    if os.path.exists(SUMMARY_FILE):
        df_s = load_data(SUMMARY_FILE)
        
        # 1. UPDATED TAB NAMES
        tab_pnl, tab_r, tab_total = st.tabs(["💲 Closed P&L (History)", "🎯 Closed R-Multiples (2026 Closed)", "🚀 Total Campaign R (2026 All)"])
        
        # ==============================================================================
        # TAB 1: ORIGINAL P&L VIEW (UNCHANGED)
        # ==============================================================================
        with tab_pnl:
            closed = df_s[df_s['Status']=='CLOSED'].copy()
            if not closed.empty:
                closed['Closed_Date'] = pd.to_datetime(closed['Closed_Date'], errors='coerce')
                available_years = sorted(closed['Closed_Date'].dt.year.dropna().unique().astype(int).tolist(), reverse=True)
                
                st.sidebar.markdown("---")
                year_filter = st.sidebar.radio("Analysis Period (P&L Tab)", ["All Time"] + [str(y) for y in available_years])
                
                if year_filter != "All Time":
                    closed = closed[closed['Closed_Date'].dt.year == int(year_filter)]
                    st.caption(f"Showing results for the year **{year_filter}**")

                st.subheader("1. Ticker Leaderboard")
                ticker_stats = closed.groupby('Ticker').agg(
                    Total_PL=('Realized_PL', 'sum'), 
                    Trade_Count=('Trade_ID', 'count'), 
                    Win_Rate=('Realized_PL', lambda x: (x > 0).mean())
                ).sort_values('Total_PL', ascending=True)
                
                top_movers = pd.concat([ticker_stats.head(10), ticker_stats.tail(10)])
                top_movers = top_movers[~top_movers.index.duplicated()].sort_values('Total_PL')
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#2ca02c' if x >= 0 else '#ff4b4b' for x in top_movers['Total_PL']]
                top_movers['Total_PL'].plot(kind='barh', ax=ax, color=colors)
                ax.axvline(0, color='black', linewidth=1)
                ax.set_title("Total P&L by Ticker")
                st.pyplot(fig)

                st.markdown("---")
                st.subheader("2. Ticker Deep Dive")
                target_ticker = st.selectbox("Select a Ticker to Analyze", sorted(closed['Ticker'].unique().tolist()), key="tf_pnl_select")
                
                if target_ticker:
                    t_df = closed[closed['Ticker'] == target_ticker].sort_values('Closed_Date', ascending=False)
                    t_count = len(t_df)
                    t_pl = t_df['Realized_PL'].sum()
                    t_wins = len(t_df[t_df['Realized_PL'] > 0])
                    t_wr = (t_wins / t_count) * 100 if t_count > 0 else 0
                    
                    gross_profits = t_df[t_df['Realized_PL'] > 0]['Realized_PL'].sum()
                    gross_losses = abs(t_df[t_df['Realized_PL'] <= 0]['Realized_PL'].sum())
                    profit_factor = gross_profits / gross_losses if gross_losses > 0 else (float('inf') if gross_profits > 0 else 0)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Trades", t_count)
                    m2.metric("Total P&L", f"${t_pl:,.2f}")
                    m3.metric("Win Rate", f"{t_wr:.1f}%")
                    m4.metric("Profit Factor", f"{profit_factor:.2f}")

                    st.dataframe(
                        t_df[['Closed_Date', 'Realized_PL', 'Avg_Entry', 'Avg_Exit', 'Sell_Rule', 'Notes']]
                        .style.format({'Realized_PL': '${:,.2f}', 'Avg_Entry': '{:.2f}', 'Avg_Exit': '{:.2f}'})
                        .applymap(color_pnl, subset=['Realized_PL']),
                        use_container_width=True
                    )
            else: st.info("No closed trades found.")

        # ==============================================================================
        # TAB 2: CLOSED R-MULTIPLE VIEW (2026 CLOSED)
        # ==============================================================================
        with tab_r:
            r_df = df_s[df_s['Status'] == 'CLOSED'].copy()
            r_df['Closed_Date'] = pd.to_datetime(r_df['Closed_Date'], errors='coerce')
            r_df = r_df[r_df['Closed_Date'].dt.year >= 2026]
            
            if not r_df.empty:
                # 3. NEW FILTER: Hide Noise
                col_h1, col_h2 = st.columns([3, 1])
                col_h1.caption("Showing R-Multiple performance for trades CLOSED in **2026+**.")
                hide_noise_r = col_h2.checkbox("Hide Noise (-1R to +1R)", value=False, key="hide_noise_r")
                
                def calc_r(row):
                    risk = row.get('Risk_Budget', 0)
                    pl = row.get('Realized_PL', 0)
                    if pd.notnull(risk) and risk > 0: return pl / risk
                    return 0.0

                r_df['R_Multiple'] = r_df.apply(calc_r, axis=1)

                # Apply Noise Filter
                if hide_noise_r:
                    r_df = r_df[(r_df['R_Multiple'] >= 1.0) | (r_df['R_Multiple'] <= -1.0)]

                if not r_df.empty:
                    st.subheader("1. Efficiency Leaderboard (Total R)")
                    r_stats = r_df.groupby('Ticker').agg(
                        Total_R=('R_Multiple', 'sum'),
                        Avg_R=('R_Multiple', 'mean'),
                        Trade_Count=('Trade_ID', 'count')
                    ).sort_values('Total_R', ascending=True)

                    r_movers = pd.concat([r_stats.head(10), r_stats.tail(10)])
                    r_movers = r_movers[~r_movers.index.duplicated()].sort_values('Total_R')

                    fig_r, ax_r = plt.subplots(figsize=(10, 6))
                    colors_r = ['#1f77b4' if x >= 0 else '#ff7f0e' for x in r_movers['Total_R']]
                    r_movers['Total_R'].plot(kind='barh', ax=ax_r, color=colors_r)
                    ax_r.axvline(0, color='black', linewidth=1)
                    ax_r.set_title("Total R-Multiple Generated (Closed Only)")
                    st.pyplot(fig_r)

                    st.markdown("---")
                    st.subheader("2. Ticker R-Analysis")
                    target_r = st.selectbox("Select Ticker", sorted(r_df['Ticker'].unique().tolist()), key="tf_r_select")
                    
                    if target_r:
                        tr_df = r_df[r_df['Ticker'] == target_r].sort_values('Closed_Date', ascending=False)
                        total_r_gen = tr_df['R_Multiple'].sum()
                        avg_r_trade = tr_df['R_Multiple'].mean()
                        
                        rm1, rm2 = st.columns(2)
                        rm1.metric("Total R Generated", f"{total_r_gen:+.2f}R")
                        rm2.metric("Avg R per Trade", f"{avg_r_trade:+.2f}R")
                        
                        st.dataframe(
                            tr_df[['Closed_Date', 'R_Multiple', 'Realized_PL', 'Risk_Budget', 'Notes']]
                            .style.format({'R_Multiple': '{:+.2f}R', 'Realized_PL': '${:,.2f}', 'Risk_Budget': '${:,.2f}'})
                            .applymap(color_pnl, subset=['Realized_PL', 'R_Multiple']),
                            use_container_width=True
                        )
                else: st.warning("No outlier trades found (all within -1R to +1R).")
            else: st.info("No closed trades found for 2026.")

# ==============================================================================
        # TAB 3: TOTAL ACTIVE R (2026 ALL)
        # ==============================================================================
        with tab_total:
            all_df = df_s.copy()
            all_df['Closed_Date'] = pd.to_datetime(all_df['Closed_Date'], errors='coerce')
            
            # Filter: Open Positions OR Closed in 2026+
            mask_open = all_df['Status'] == 'OPEN'
            mask_2026 = (all_df['Status'] == 'CLOSED') & (all_df['Closed_Date'].dt.year >= 2026)
            
            act_df = all_df[mask_open | mask_2026].copy()
            
            if not act_df.empty:
                col_th1, col_th2 = st.columns([3, 1])
                col_th1.caption("Showing **ALL Active Campaigns** (Open Positions + Trades closed in 2026).")
                
                # CHANGE 1: Default value is now TRUE (Hide noise automatically)
                hide_noise_all = col_th2.checkbox("Hide Noise (-1R to +1R)", value=True, key="hide_noise_all")

                # --- CALCULATION LOGIC (With ALAB Fix) ---
                def calc_total_r(row):
                    risk = row.get('Risk_Budget', 0)
                    real = row.get('Realized_PL', 0)
                    unreal = row.get('Unrealized_PL', 0)
                    status = row.get('Status', 'OPEN')
                    
                    # Fix: If CLOSED, ignore Unrealized column (it is historical noise)
                    if status == 'CLOSED':
                        total_pl = real
                    else:
                        total_pl = real + unreal
                    
                    if pd.notnull(risk) and risk > 0:
                        return float(total_pl / risk)
                    return 0.0

                act_df['Total_R'] = act_df.apply(calc_total_r, axis=1).astype(float)
                
                # Visual Fix: Zero out Unrealized for Closed rows in display
                act_df.loc[act_df['Status'] == 'CLOSED', 'Unrealized_PL'] = 0.0
                
                # CHANGE 2: Apply Noise Filter
                if hide_noise_all:
                    # Keep only trades that are Greater than +1R OR Less than -1R
                    act_df = act_df[ (act_df['Total_R'] >= 1.0) | (act_df['Total_R'] <= -1.0) ]

                if not act_df.empty:
                    st.subheader("1. True Performance (Realized + Unrealized)")
                    
                    # Group by Ticker
                    total_stats = act_df.groupby('Ticker').agg(
                        Total_R=('Total_R', 'sum'),
                        Status=('Status', 'first')
                    ).sort_values('Total_R', ascending=True)

                    # Visualization
                    t_movers = pd.concat([total_stats.head(12), total_stats.tail(12)])
                    t_movers = t_movers[~t_movers.index.duplicated()].sort_values('Total_R')

                    fig_t, ax_t = plt.subplots(figsize=(10, 7))
                    colors_t = ['#2ca02c' if x >= 0 else '#d62728' for x in t_movers['Total_R']]
                    
                    t_movers['Total_R'].plot(kind='barh', ax=ax_t, color=colors_t)
                    ax_t.axvline(0, color='black', linewidth=1)
                    ax_t.set_title("Total Campaign R (Open + 2026 Closed)")
                    ax_t.set_xlabel("R Units (Risk Multiples)")
                    
                    for i, v in enumerate(t_movers['Total_R']):
                        ax_t.text(v, i, f" {v:+.1f}R", va='center', fontweight='bold')

                    st.pyplot(fig_t)
                    
                    st.markdown("---")
                    st.subheader("2. Campaign Details")
                    
                    view_cols = ['Ticker', 'Status', 'Open_Date', 'Total_R', 'Realized_PL', 'Unrealized_PL', 'Risk_Budget']
                    valid_cols = [c for c in view_cols if c in act_df.columns]
                    
                    st.dataframe(
                        act_df[valid_cols].sort_values('Total_R', ascending=False)
                        .style.format({
                            'Total_R': '{:+.2f}R',
                            'Realized_PL': '${:,.0f}',
                            'Unrealized_PL': '${:,.0f}',
                            'Risk_Budget': '${:,.0f}'
                        })
                        .applymap(lambda x: 'background-color: #e6fffa; color: #004d40' if x == 'OPEN' else '', subset=['Status'])
                        .applymap(color_pnl, subset=['Total_R']),
                        use_container_width=True,
                        height=500
                    )
                else: 
                    st.info("✅ No outliers found. All active campaigns are currently within normal noise levels (-1R to +1R).")
                    if hide_noise_all:
                        st.caption("Uncheck the box above to see all trades.")
            else:
                st.info("No active or 2026 trades found.")

    else:
        st.error("Summary file missing.")

# ==============================================================================
# PAGE 7: PERIOD REVIEW (MATH FIXED TO MATCH DASHBOARD)
# ==============================================================================
elif page == "Period Review":
    st.header("PERIODIC REVIEW")
    
    # --- DATA ENGINE ---
    def clean_num_local(x):
        try:
            if isinstance(x, str):
                return float(x.replace('$', '').replace(',', '').replace('%', '').strip())
            return float(x)
        except: return 0.0

    def get_df(p_name):
        # Load data using database-aware load_data()
        path = os.path.join(DATA_ROOT, p_name, 'Trading_Journal_Clean.csv')
        summ_path = os.path.join(DATA_ROOT, p_name, 'Trade_Log_Summary.csv')

        d = load_data(path)
        if not d.empty and 'Day' in d.columns:
            d['Day'] = pd.to_datetime(d['Day'], errors='coerce')
            d = d.dropna(subset=['Day']).sort_values('Day')
            for c in ['Beg NLV', 'End NLV', 'Cash -/+', 'Daily $ Change', 'SPY', 'Nasdaq']:
                if c in d.columns: d[c] = d[c].apply(clean_num_local)

        s = load_data(summ_path)

        return d, s

    # LOAD DATA
    df_j, df_s = get_df(PORT_CANSLIM)

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
        PORT_457B:    os.path.join(DATA_ROOT, PORT_457B, 'Trading_Journal_Clean.csv')
    }

    # THE GOLDEN STANDARD (Added 4 Review Columns)
    MASTER_ORDER = [
    'Day', 'Status', 'Market Window', '> 21e', 'Cash -/+', 
    'Beg NLV', 'End NLV', 'Daily $ Change', 'Daily % Change', 
    '% Invested', 'SPY', 'Nasdaq', 'Market_Notes', 'Market_Action',
    'Score', 'Highlights', 'Lowlights', 'Mistakes', 'Top_Lesson' # <--- Added these
    ]

    # --- HELPER: ROBUST LOADER (DATABASE-AWARE) ---
    def load_and_prep_file(path):
        # Database mode: extract portfolio name from path and use load_data
        if USE_DATABASE:
            try:
                portfolio_name = None
                if PORT_CANSLIM in path:
                    portfolio_name = PORT_CANSLIM
                elif PORT_457B in path:
                    portfolio_name = PORT_457B

                if portfolio_name:
                    df = load_data(path)  # This will use db.load_journal() in database mode
                    if df.empty:
                        return pd.DataFrame(columns=MASTER_ORDER)

                    # Ensure Day is string format
                    if 'Day' in df.columns:
                        df['Day'] = pd.to_datetime(df['Day'], errors='coerce').dt.strftime('%Y-%m-%d')
                        df = df.dropna(subset=['Day'])

                    # Ensure all Master Columns exist
                    for c in MASTER_ORDER:
                        if c not in df.columns:
                            df[c] = '' if c in ['Status','Market Window','Keywords','Market_Action','Market_Notes','Daily % Change'] else 0.0

                    return df[MASTER_ORDER]
            except Exception as e:
                st.error(f"Database load error: {e}")
                return pd.DataFrame(columns=MASTER_ORDER)

        # CSV mode (local development)
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
        # Use date() to avoid timezone issues
        entry_date = c1.date_input("Date", get_current_date_ct())
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
        for p_name in [PORT_CANSLIM, PORT_457B]:
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

        st.markdown("---")
        with st.expander("📓 POST-TRADE ANALYSIS & REVIEW", expanded=False):
            # 1. VISUAL REFERENCE BOX
            st.info("""
            **Process Scoring Rubric:**
            - **5 (Elite):** Perfect execution. Followed every rule. Zero impulsive moves.
            - **4 (Good):** Solid discipline. Followed the plan with minor execution lag.
            - **3 (Average):** Followed most rules, but execution was sloppy.
            - **2 (Poor):** Major rule breaks (e.g., chasing, wide stops, oversized).
            - **1 (Fail):** Emotional breakdown. Revenge trading. Zero plan followed.
            """)

        er_c1, er_c2 = st.columns([1, 3])
        
        # 2. THE INPUTS
        daily_score = er_c1.slider("Daily Score", 1, 5, 3, help="Rate your process, not your P&L.")
        
        highlights = st.text_area("Highlights (What went well?)", height=68)
        lowlights = st.text_area("Lowlights (What went wrong?)", height=68)
        mistakes = st.text_area("Daily Mistakes", placeholder="e.g., Chased entry, ignored stop loss...", height=68)
        top_lesson = st.text_input("Top Lesson for Tomorrow")

        force_ovr = st.checkbox("⚠️ Force Overwrite Existing Entry", help="Check this if you get a 'Skipped' error but want to save anyway.")

        if st.form_submit_button("💾 LOG SELECTED ACCOUNTS", type="primary"):
            success_count = 0
            
            for p_name, inputs in input_keys.items():
                p_path = PORTFOLIO_MAP[p_name]
                end_nlv = inputs['nlv']
                sec_val = inputs['sec']
                cash_flow = inputs['cash_flow']
                
                if end_nlv > 0 or cash_flow != 0:  # allow NLV=0 with cash outflow (account close-out)
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
                        
                    # 3. MATH (TWR: denominator includes cash flow to match Dashboard)
                    if prev_nlv > 0:
                        daily_chg = end_nlv - prev_nlv - cash_flow
                        adj_beg = prev_nlv + cash_flow
                        pct_val = (daily_chg / adj_beg) * 100 if adj_beg != 0 else 0.0
                    elif cash_flow > 0:
                        daily_chg = end_nlv - cash_flow
                        pct_val = (daily_chg / cash_flow) * 100
                    else:
                        daily_chg = 0.0
                        pct_val = 0.0
                    
                    daily_pct_str = f"{pct_val:.2f}%"
                    invested_pct = (sec_val / end_nlv) * 100 if end_nlv > 0 else 0.0
                    
                    # 4. CREATE ROW (Updated with Review Data)
                    new_row = {
                        'Day': entry_date_str, 'Status': 'U', 'Market Window': 'Open', '> 21e': 0.0,
                        'Cash -/+': cash_flow, 'Beg NLV': prev_nlv, 'End NLV': end_nlv,
                        'Daily $ Change': daily_chg, 'Daily % Change': daily_pct_str,
                        '% Invested': invested_pct, 'SPY': spy_val, 'Nasdaq': ndx_val,
                        'Market_Notes': market_notes,
                        'Market_Action': inputs['note'],
                        'Score': daily_score,         # <--- Now uses the slider value
                        'Highlights': highlights,     # <--- New
                        'Lowlights': lowlights,       # <--- New
                        'Mistakes': mistakes,         # <--- New
                        'Top_Lesson': top_lesson      # <--- New
                    }
                    
                    # 5. SAVE
                    try:
                        if USE_DATABASE:
                            # Save to database
                            journal_entry = {
                                'portfolio_id': p_name,
                                'day': entry_date_str,
                                'status': 'U',
                                'market_window': 'Open',
                                'above_21ema': 0.0,
                                'cash_flow': cash_flow,
                                'beginning_nlv': prev_nlv,
                                'ending_nlv': end_nlv,
                                'daily_dollar_change': daily_chg,
                                'daily_percent_change': pct_val,
                                'percent_invested': invested_pct,
                                'spy_close': spy_val,
                                'nasdaq_close': ndx_val,
                                'market_notes': market_notes,
                                'market_action': inputs['note'],
                                'score': daily_score,
                                'highlights': highlights,
                                'lowlights': lowlights,
                                'mistakes': mistakes,
                                'top_lesson': top_lesson
                            }
                            db.save_journal_entry(journal_entry)
                            success_count += 1
                        else:
                            # Save to CSV (local mode)
                            new_df_row = pd.DataFrame([new_row])
                            for c in MASTER_ORDER:
                                if c not in new_df_row.columns: new_df_row[c] = ''

                            df_final = pd.concat([new_df_row[MASTER_ORDER], df_curr[MASTER_ORDER]], ignore_index=True)
                            df_final['Sort_Key'] = pd.to_datetime(df_final['Day'], errors='coerce')
                            df_final = df_final.sort_values('Sort_Key', ascending=False).drop(columns=['Sort_Key'])
                            df_final.to_csv(p_path, index=False)
                            success_count += 1
                    except Exception as e:
                        st.error(f"❌ Save Failed {p_name}: {e}")

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
# PAGE 9: POSITION SIZER (CRASH FIX + LIFO LOGIC + VOLATILITY SIZER)
# ==============================================================================
elif page == "Position Sizer":
    st.header(f"POSITION SIZING CALCULATOR ({CURR_PORT_NAME})")
    
    # --- GLOBAL DATA ---
    # Load latest NLV from journal (database-aware)
    equity = 100000.0
    try:
        df = load_data(JOURNAL_FILE)
        if not df.empty and 'End NLV' in df.columns:
            # Sort by date to ensure we get the latest entry
            if 'Day' in df.columns:
                df['Day'] = pd.to_datetime(df['Day'], errors='coerce')
                df = df.dropna(subset=['Day']).sort_values('Day', ascending=False)
            val_str = str(df['End NLV'].iloc[0]).replace('$','').replace(',','')
            equity = float(val_str)
    except Exception as e:
        pass  # Silently fall back to default
    
    df_s = load_data(SUMMARY_FILE)
    df_d = load_data(DETAILS_FILE)
    
    size_map = {"Shotgun (2.5%)": 2.5, "Half (5%)": 5.0, "Full (10%)": 10.0, "Core (15%)": 15.0, "Core+1 (20%)": 20.0, "Max (25%)": 25.0, "30%": 30.0, "35%": 35.0, "40%": 40.0, "45%": 45.0, "50%": 50.0}

    # UPDATED TABS LIST
    tab_new, tab_manage, tab_add, tab_trim, tab_vol = st.tabs([
        "🆕 Plan New Trade", 
        "🔗 Manage Active Campaign", 
        "➕ Add (Pyramid)", 
        "✂️ Trim (Sell Down)",
        "⚖️ Volatility Sizer"
    ])
    
    # --- TAB 1: PLAN NEW TRADE ---
    with tab_new:
        st.caption("Double Constraint Sizing: Capped by Risk Limit OR Target Size.")
        st.markdown("#### 1. Establish Risk Budget")
        c1, c2, c3 = st.columns(3)
        acct_val = c1.number_input("Account Equity ($)", value=equity, step=1000.0, key="np_eq")
        risk_pct = c2.slider("Max Risk %", 0.15, 1.50, 0.50, 0.05, key="np_rp")
        risk_budget_dol = acct_val * (risk_pct / 100)
        c3.metric("Risk Budget ($)", f"${risk_budget_dol:,.2f}")
        
        st.markdown("---")
        st.markdown("#### 2. Price and Stop")
        p1, p2, p3 = st.columns(3)
        ticker = p1.text_input("Ticker", key="np_tk", placeholder="XYZ").upper()
        
        def_entry = 100.00
        if ticker:
            try: 
                live_fetch = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
                if live_fetch > 0: def_entry = float(live_fetch)
            except: pass
            
        entry = p2.number_input("Entry Price ($)", min_value=0.01, value=def_entry, step=0.1, key=f"np_ep_{ticker}")
        stop_mode = st.radio("Stop Mode", ["Price ($)", "Percent (%)"], horizontal=True, key="np_sm_mode", label_visibility="collapsed")
        
        if stop_mode == "Percent (%)":
            stop_pct_in = p3.number_input("Stop Loss %", value=7.0, step=0.5, key=f"np_pct_{ticker}")
            stop_val = entry * (1 - (stop_pct_in/100))
            st.caption(f"Stop Price: ${stop_val:.2f}")
        else:
            def_stop = entry * 0.93 if entry > 0 else 0.0
            stop_val = p3.number_input("Stop Price ($)", min_value=0.0, value=def_stop, step=0.1, key=f"np_sv_{ticker}")
            if entry > 0: 
                stop_pct_in = ((entry - stop_val)/entry)*100
                st.caption(f"Stop Width: {stop_pct_in:.2f}%")
            else: stop_pct_in = 0

        st.markdown("---")
        st.markdown("#### 3. Target Size")
        size_mode = st.select_slider("Desired Position Scale", options=list(size_map.keys()), value="Half (5%)", key="np_sz_slide")
        target_pct = size_map[size_mode]
        st.info(f"Targeting a **{target_pct}%** position (${(acct_val * target_pct/100):,.0f})")

        st.markdown("---")
        if st.button("CALCULATE TRADE VITALS", type="primary", key="np_btn_calc"):
            if entry > 0 and stop_val > 0 and stop_val < entry:
                risk_per_share = entry - stop_val
                max_shares_risk = math.floor(risk_budget_dol / risk_per_share)
                target_equity = acct_val * (target_pct / 100)
                max_shares_size = math.floor(target_equity / entry)
                final_shares = min(max_shares_risk, max_shares_size)
                
                actual_cost = final_shares * entry
                actual_size_pct = (actual_cost / acct_val) * 100
                actual_risk_dol = final_shares * risk_per_share
                actual_risk_pct = (actual_risk_dol / acct_val) * 100
                
                st.markdown("### 🩺 Trade Vitals")
                k1, k2, k3 = st.columns(3)
                k1.metric("RECOMMENDED BUY", f"{final_shares} shares", f"Value: ${actual_cost:,.0f}")
                limit_reason = "Target Size" if max_shares_size < max_shares_risk else "Risk Budget"
                k2.metric("Limiting Factor", limit_reason, f"Capped by {limit_reason}")
                k3.metric("Actual Risk Taken", f"{actual_risk_pct:.2f}%", f"${actual_risk_dol:.2f}", delta_color="off")
                
                st.markdown("---")
                v1, v2 = st.columns(2)
                v1.metric("Actual Position Size", f"{actual_size_pct:.1f}%", f"Target: {target_pct}%")
                v2.metric("Stop Loss Width", f"{stop_pct_in:.2f}%", "Ideal: < 8%")
                
                if stop_pct_in > 8.0: st.error("⚠️ Stop > 8%.")
                if limit_reason == "Risk Budget": st.warning(f"⚠️ **Size Reduced:** Risk Budget capped you below {target_pct}%.")
                else: st.success(f"✅ Full {target_pct}% size approved.")
            else: st.error("Invalid Prices.")

    # --- TAB 2: MANAGE ACTIVE CAMPAIGN (SPLIT STOP LOGIC) ---
    with tab_manage:
        st.subheader("🛡️ Manage Active Campaign (Scale-In)")
        if not df_s.empty:
            open_ops = df_s[df_s['Status'] == 'OPEN']
            if not open_ops.empty:
                opts = [f"{r['Ticker']} (Own {int(r['Shares'])})" for _, r in open_ops.iterrows()]
                sel_camp = st.selectbox("Choose Ticker", opts, key="mo_sel_camp")
                sel_ticker = sel_camp.split(" ")[0]
                
                # Get Campaign Data
                row = open_ops[open_ops['Ticker'] == sel_ticker].iloc[0]
                curr_shares = int(row['Shares'])
                tid = row['Trade_ID']
                
                # Get Avg Cost (Simple from Summary)
                avg_cost = float(row['Total_Cost']) / curr_shares if curr_shares > 0 else 0.0
                
                # Get Live Price
                try: live_curr = yf.Ticker(sel_ticker).history(period='1d')['Close'].iloc[-1]
                except: live_curr = avg_cost
                
                curr_weight = (curr_shares * live_curr / equity) * 100
                db_budget = row.get('Risk_Budget', equity * 0.005)

                realized_pl_camp = 0.0
                if not df_d.empty:
                    camp_tx = df_d[df_d['Trade_ID'] == tid]
                    if 'Realized_PL' in camp_tx.columns:
                        realized_pl_camp = camp_tx['Realized_PL'].sum()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Ticker", sel_ticker)
                c2.metric("Original Budget", f"${db_budget:,.2f}")
                c3.metric("Current Position", f"{curr_shares} shs", f"{curr_weight:.1f}% Weight")
                c4.metric("Avg Cost", f"${avg_cost:,.2f}")
                c5.metric("Realized Bank", f"${realized_pl_camp:,.2f}", help="Total Realized P&L for this campaign so far.")
                
                st.markdown("---")
                
                st.markdown(f"#### 🪜 Scale-In Calculator")
                stop_strategy = st.radio("Stop Strategy", 
                                         ["Universal Stop (Early Trend)", "Split Stop (Late Trend / Add Only)"], 
                                         index=1, horizontal=True)

                sc1, sc2, sc3 = st.columns(3)
                add_price = sc1.number_input("Add Price ($)", value=float(live_curr), step=0.1, key=f"mo_add_px_{sel_ticker}")
                
                stop_label = "New Universal Stop ($)" if "Universal" in stop_strategy else "Stop for NEW Shares ($)"
                new_stop_input = sc2.number_input(stop_label, value=0.0, step=0.1, key=f"mo_new_stop_{sel_ticker}")
                
                atr_input_mo = sc3.number_input("ATR % (21-Day)", value=5.0, step=0.1, key=f"mo_atr_{sel_ticker}")

                st.markdown("---")

                if new_stop_input > 0 and new_stop_input < add_price:
                    
                    # --- A. MAX RISK CALCULATION ---
                    max_risk_shares = 0
                    risk_per_new = add_price - new_stop_input
                    
                    limit_msg = "Risk Budget"
                    if "Universal" in stop_strategy:
                        existing_pl_at_stop = (new_stop_input - avg_cost) * curr_shares
                        total_armor = realized_pl_camp + existing_pl_at_stop
                        risk_capacity = total_armor + db_budget
                        if risk_capacity > 0:
                            max_risk_shares = math.floor(risk_capacity / risk_per_new)
                        limit_msg = "Financed by Core"
                    else:
                        # Split Stop: Use Tier 2 Risk (0.65%) for the fresh tranche
                        tier_risk_dlr = equity * 0.0065 
                        max_risk_shares = math.floor(tier_risk_dlr / risk_per_new)
                        limit_msg = "Risk Budget (0.65% NLV)"

                    # --- B. PYRAMID CHECK ---
                    max_pyramid_shares = math.ceil(curr_shares * 0.50)
                    
                    # --- C. VOLATILITY CHECK (FIXED: Use Current Cushion) ---
                    # We use the CURRENT cushion to determine the tier, not the hypothetical blended one.
                    # You earned the Tier 1 status on the core position.
                    
                    current_cushion_pct = ((add_price - avg_cost) / avg_cost) * 100
                    
                    tol_pct = 0.50 # Tier 3
                    if current_cushion_pct >= 20.0: tol_pct = 1.00 # Tier 1
                    elif current_cushion_pct >= 5.0: tol_pct = 0.65 # Tier 2
                    
                    daily_limit_dlr = equity * (tol_pct / 100)
                    atr_dec = atr_input_mo / 100
                    
                    # Total allowed shares for the whole position
                    total_allowed_shares_vol = int(daily_limit_dlr / (add_price * atr_dec))
                    
                    # Remaining capacity
                    max_vol_add = total_allowed_shares_vol - curr_shares
                    if max_vol_add < 0: max_vol_add = 0
                    
                    # --- FINAL DECISION ---
                    final_shares = min(max_risk_shares, max_pyramid_shares, max_vol_add)
                    
                    limit_reason = "Risk Budget"
                    if final_shares == max_pyramid_shares: limit_reason = "Half-Step Rule (50%)"
                    elif final_shares == max_vol_add: limit_reason = "Volatility (ATR)"
                    elif final_shares == max_risk_shares: limit_reason = limit_msg

                    # --- EXECUTION DISPLAY ---
                    st.markdown("### 🎯 Execution Analysis")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("1. Risk Limit", f"{max_risk_shares} shs", limit_msg)
                    m2.metric("2. Pyramid Limit", f"{max_pyramid_shares} shs", "Max 50% of Core")
                    m3.metric("3. Volatility Limit", f"{max_vol_add} shs", f"Tier {1 if tol_pct==1 else (2 if tol_pct==0.65 else 3)} Capacity")

                    st.markdown("---")

                    k1, k2, k3 = st.columns(3)
                    
                    new_total = curr_shares + final_shares
                    new_avg = ((curr_shares * avg_cost) + (final_shares * add_price)) / new_total
                    new_weight = (new_total * add_price / equity) * 100
                    
                    k1.metric("Rec. Add Size", f"+{final_shares} shares", f"Limit: {limit_reason}")
                    k2.metric("New Total", f"{new_total} shares", f"Avg: ${new_avg:,.2f}")
                    k3.metric("New Weight", f"{new_weight:.1f}%", f"Target: {new_weight:.1f}%")
                    
                    if final_shares <= 0:
                        if max_vol_add <= 0: 
                            st.error(f"⛔ **VOLATILITY BLOCKED:** Tier {1 if tol_pct==1 else 2} Limit is {total_allowed_shares_vol} shares. You own {curr_shares}.")
                        else:
                            st.error("❌ **NO TRADE:** Logic constraints prevent adding.")
                    else: 
                        st.success(f"✅ **APPROVED:** Buy {final_shares} shares.")
                        if "Split" in stop_strategy:
                            st.info(f"📝 **Plan:** Set Stop on NEW {final_shares} shares at ${new_stop_input:,.2f}. Keep Core Stop at 10W.")
                        else:
                            st.info(f"📝 **Plan:** Raise Stop on ALL {new_total} shares to ${new_stop_input:,.2f}.")

                else: st.info("👈 Enter 'Stop Price' to calculate.")
            else: st.info("No active campaigns.")
        else: st.error("Summary file empty.")

    # --- TAB 3: PYRAMIDING (ADD ON) ---
    with tab_add:
        st.caption("Calculate shares to add and new stop to protect total equity.")
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
            
            c3, c4 = st.columns(2)
            target_mode_add = c3.select_slider("Target Total Position Size", options=list(size_map.keys()), value="Full (10%)", key="add_ts_mode")
            target_size_pct = size_map[target_mode_add]
            max_risk_pct = c4.slider("Max Total Risk % (Capital)", 0.25, 1.25, 0.75, 0.05, key="add_mr")
            
            if st.button("Calculate Add-On", key="add_btn"):
                target_value = acct_val_add * (target_size_pct / 100)
                current_value = row['Shares'] * curr_price 
                value_to_add = target_value - current_value
                
                if value_to_add <= 0:
                    st.error(f"You are already over the target weight! (Current: ${current_value:,.0f} vs Target: ${target_value:,.0f})")
                else:
                    shares_to_add = math.floor(value_to_add / curr_price)
                    total_shares = row['Shares'] + shares_to_add
                    new_avg_cost = ((row['Shares'] * row['Avg_Entry']) + (shares_to_add * curr_price)) / total_shares
                    cost_of_add = shares_to_add * curr_price
                    
                    allowed_risk_dollars = acct_val_add * (max_risk_pct / 100)
                    required_cushion_per_share = allowed_risk_dollars / total_shares
                    new_stop_price = curr_price - required_cushion_per_share
                    stop_dist_pct = (required_cushion_per_share / curr_price) * 100
                    
                    st.markdown("### ➕ PYRAMID TICKET")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("ADD SHARES", f"+{shares_to_add}")
                    k2.metric("EST. COST", f"${cost_of_add:,.2f}")
                    k3.metric("NEW TOTAL", f"{int(total_shares)}")
                    k4.metric("AVG COST (Est)", f"${new_avg_cost:.2f}")
                    
                    st.markdown("### 🛡️ RISK MANAGEMENT")
                    r1, r2 = st.columns(2)
                    r1.metric("SUGGESTED NEW STOP", f"${new_stop_price:.2f}", delta=f"-{stop_dist_pct:.2f}% from Current")
                    r2.metric("TOTAL RISK", f"${allowed_risk_dollars:,.0f}", f"{max_risk_pct}% of Equity")
                    
                    if new_stop_price > curr_price: st.error("❌ IMPOSSIBLE RISK.")
                    elif new_stop_price > row['Avg_Entry']: st.success("✅ PROFIT LOCK ACTIVE.")
        else:
            st.info("No Open Positions found in Summary file.")

    # --- TAB 4: TRIM (WITH ACCURATE LIFO P&L) ---
    with tab_trim:
        st.subheader("✂️ Trim Position (Sell Down)")
        st.caption("Calculate shares to sell to reach a desired weight, with LIFO P&L estimation.")
        
        open_positions = df_s[df_s['Status'] == 'OPEN'].sort_values('Ticker')
        
        if not open_positions.empty:
            # 1. Selection & Data Fetching
            sel_trim = st.selectbox("Select Holding to Trim", options=open_positions['Ticker'].unique().tolist(), key="trim_sel")
            row_t = open_positions[open_positions['Ticker'] == sel_trim].iloc[0]
            tid_trim = row_t['Trade_ID']
            
            try: live_price_t = yf.Ticker(row_t['Ticker']).history(period="1d")['Close'].iloc[-1]
            except: live_price_t = row_t.get('Avg_Entry', 100)
            
            t1, t2, t3 = st.columns(3)
            curr_price_t = t1.number_input("Current Price ($)", value=float(live_price_t), step=0.1, key=f"trim_cp_{row_t['Ticker']}")
            curr_val_t = row_t['Shares'] * curr_price_t
            curr_weight_t = (curr_val_t / equity) * 100
            
            t2.metric("Current Weight", f"{curr_weight_t:.1f}%")
            t3.metric("Current Value", f"${curr_val_t:,.0f}")
            
            st.markdown("---")
            
            # 2. Target Setting
            target_mode_trim = st.select_slider("Target Standard Position Size", options=list(size_map.keys()), value="Half (5%)", key="trim_sm")
            target_weight_t = size_map[target_mode_trim]
            
            if st.button("Calculate Trim Impact", type="primary", key="trim_btn"):
                if target_weight_t >= curr_weight_t: 
                    st.warning(f"⚠️ Target ({target_weight_t}%) is higher than Current ({curr_weight_t:.1f}%). No trim needed.")
                else:
                    # A. Basic Trim Math
                    target_val_t = equity * (target_weight_t / 100)
                    value_to_sell = curr_val_t - target_val_t
                    shares_to_sell = math.ceil(value_to_sell / curr_price_t) 
                    remaining_shares = row_t['Shares'] - shares_to_sell
                    actual_new_weight = (remaining_shares * curr_price_t / equity) * 100
                    
                    # B. LIFO ENGINE (Calculate Cost Basis of Specific Shares)
                    lifo_pnl = 0.0
                    cost_basis_trimmed = 0.0
                    
                    if not df_d.empty:
                        # Filter transactions for this specific Trade ID
                        trxs = df_d[df_d['Trade_ID'] == tid_trim].copy()
                        
                        if not trxs.empty:
                            # Sort by Date/Action (Buy First)
                            trxs['Type_Rank'] = trxs['Action'].apply(lambda x: 0 if str(x).upper() == 'BUY' else 1)
                            if 'Date' in trxs.columns: trxs = trxs.sort_values(['Date', 'Type_Rank'])
                            
                            # Build Inventory Stack
                            inventory = []
                            for _, tx in trxs.iterrows():
                                action = str(tx.get('Action', '')).upper()
                                # Clean numbers
                                try: s = abs(float(str(tx.get('Shares', 0)).replace(',','')))
                                except: s = 0.0
                                try: p = float(str(tx.get('Amount', tx.get('Price', 0.0))).replace('$','').replace(',',''))
                                except: p = 0.0
                                
                                if action == 'BUY':
                                    if p == 0: p = float(row_t['Avg_Entry']) # Fallback
                                    inventory.append({'qty': s, 'price': p})
                                    
                                elif action == 'SELL':
                                    sell_q = s
                                    # Remove from end (LIFO)
                                    while sell_q > 0 and inventory:
                                        last = inventory[-1]
                                        take = min(sell_q, last['qty'])
                                        last['qty'] -= take
                                        sell_q -= take
                                        if last['qty'] < 0.00001: inventory.pop()
                            
                            # C. Simulate the Trim (Pop from Inventory)
                            shares_needed = shares_to_sell
                            accumulated_cost = 0.0
                            
                            # Work backwards through inventory (LIFO)
                            while shares_needed > 0 and inventory:
                                last_lot = inventory[-1]
                                take = min(shares_needed, last_lot['qty'])
                                accumulated_cost += (take * last_lot['price'])
                                last_lot['qty'] -= take
                                shares_needed -= take
                                if last_lot['qty'] < 0.00001: inventory.pop()
                                
                            # If history incomplete, use Avg Entry for remainder
                            if shares_needed > 0:
                                accumulated_cost += (shares_needed * row_t['Avg_Entry'])
                                
                            cost_basis_trimmed = accumulated_cost
                            revenue = shares_to_sell * curr_price_t
                            lifo_pnl = revenue - cost_basis_trimmed
                        else:
                            # Fallback if no details
                            cost_basis_trimmed = shares_to_sell * row_t['Avg_Entry']
                            lifo_pnl = (shares_to_sell * curr_price_t) - cost_basis_trimmed
                    else:
                        # Fallback if no details file
                        cost_basis_trimmed = shares_to_sell * row_t['Avg_Entry']
                        lifo_pnl = (shares_to_sell * curr_price_t) - cost_basis_trimmed

                    # D. DISPLAY RESULTS
                    st.markdown("### 🎫 Sell Ticket")
                    c_res1, c_res2, c_res3 = st.columns(3)
                    c_res1.metric("SHARES TO SELL", f"-{int(shares_to_sell)}")
                    c_res2.metric("REMAINING", f"{int(remaining_shares)}")
                    c_res3.metric("NEW WEIGHT", f"{actual_new_weight:.1f}%", f"Target: {target_weight_t}%")
                    
                    st.markdown("### 💰 Financial Impact (LIFO)")
                    
                    f1, f2, f3 = st.columns(3)
                    
                    # Cash Generated
                    cash_gen = shares_to_sell * curr_price_t
                    f1.metric("Cash Generated", f"${cash_gen:,.2f}")
                    
                    # Cost Basis of Sold Shares
                    avg_cost_sold = cost_basis_trimmed / shares_to_sell if shares_to_sell > 0 else 0
                    f2.metric("Cost Basis (Sold)", f"${cost_basis_trimmed:,.2f}", f"Avg: ${avg_cost_sold:.2f}/sh")
                    
                    # Realized P&L
                    pnl_color = "normal" if lifo_pnl >= 0 else "inverse"
                    f3.metric("Realized P&L", f"${lifo_pnl:,.2f}", f"{(lifo_pnl/cost_basis_trimmed)*100:.2f}% Return", delta_color=pnl_color)
                    
                    if lifo_pnl < 0:
                        st.warning(f"⚠️ **Note:** This trim realizes a loss of ${abs(lifo_pnl):,.2f} based on your most recent purchases (LIFO).")
                    else:
                        st.success(f"✅ **Profit Lock:** This trim locks in ${lifo_pnl:,.2f} profit.")
        else: st.info("No active positions.")

   # --- TAB 5: VOLATILITY SIZER (THE GEM STANDARD) ---
    with tab_vol:
        st.subheader("⚖️ Volatility-Adjusted Sizing (The Gem Standard)")
        st.caption("Normalize risk by sizing positions based on Asset Volatility (ATR) AND Technical Stop.")
        
        # 0. TIER CHEAT SHEET
        with st.expander("ℹ️ View Tier System Rules"):
            st.markdown("""
            **The Tolerance Tiers:**
            * **Tier 1 (High Cushion):** Profit > 20% ⮕ **1.00% Risk**
            * **Tier 2 (Moderate):** Profit 5% to 20% ⮕ **0.65% Risk**
            * **Tier 3 (Defense):** Profit < 5% (or New) ⮕ **0.50% Risk**
            """)

        # 1. MODE SELECTION
        vol_mode = st.radio("Sizing Context", ["🆕 New Trade (Defense Mode)", "🔍 Audit Active Position"], horizontal=True, key="vs_mode")
        
        # 2. DATA INPUTS
        vs_ticker = ""
        vs_price = 0.0
        vs_avg_cost = 0.0
        vs_shares = 0.0
        
        c1, c2, c3 = st.columns(3)
        
        if vol_mode.startswith("🆕"):
            # New Trade Mode
            vs_ticker = c1.text_input("Ticker Symbol", placeholder="XYZ", key="vs_tk_new").upper()
            vs_price = c2.number_input("Entry Price ($)", value=0.0, step=0.1, key="vs_px_new")
            vs_avg_cost = vs_price 
            
        else:
            # Audit Mode
            if not df_s.empty:
                open_ops = df_s[df_s['Status'] == 'OPEN']
                if not open_ops.empty:
                    opts = [f"{r['Ticker']} ({int(r['Shares'])} shs)" for _, r in open_ops.iterrows()]
                    sel_audit = c1.selectbox("Select Position", opts, key="vs_sel_audit")
                    vs_ticker = sel_audit.split(" ")[0]
                    
                    row = open_ops[open_ops['Ticker'] == vs_ticker].iloc[0]
                    vs_shares = float(row['Shares'])
                    
                    # --- LIFO ENGINE: Update vs_avg_cost based on Transaction History ---
                    # Default to summary avg in case of empty history
                    lifo_cost = float(row['Avg_Entry']) 
                    
                    if not df_d.empty:
                        tid = row['Trade_ID']
                        trxs = df_d[df_d['Trade_ID'] == tid].copy()
                        if not trxs.empty:
                            # 1. Sort by Date and Action (Buy first)
                            trxs['Type_Rank'] = trxs['Action'].apply(lambda x: 0 if str(x).upper() == 'BUY' else 1)
                            if 'Date' in trxs.columns: trxs = trxs.sort_values(['Date', 'Type_Rank'])
                            
                            inventory = []
                            for _, tx in trxs.iterrows():
                                action = str(tx.get('Action', '')).upper()
                                tx_shares = abs(float(str(tx.get('Shares', 0)).replace(',','')))
                                
                                if action == 'BUY':
                                    price = float(str(tx.get('Amount', tx.get('Price', 0.0))).replace('$','').replace(',',''))
                                    # Fallback if price is 0
                                    if price == 0: price = float(row['Avg_Entry'])
                                    inventory.append({'qty': tx_shares, 'price': price})
                                    
                                elif action == 'SELL':
                                    qty_to_sell = tx_shares
                                    # LIFO Pop
                                    while qty_to_sell > 0 and inventory:
                                        last = inventory[-1]
                                        take = min(qty_to_sell, last['qty'])
                                        last['qty'] -= take
                                        qty_to_sell -= take
                                        if last['qty'] < 0.00001: inventory.pop()
                            
                            # Re-calculate Avg Cost of Remaining Inventory
                            total_rem_shares = sum(i['qty'] for i in inventory)
                            total_rem_cost = sum(i['qty'] * i['price'] for i in inventory)
                            if total_rem_shares > 0:
                                lifo_cost = total_rem_cost / total_rem_shares
                    
                    vs_avg_cost = lifo_cost
                    # -------------------------------------------------------------------
                    
                    try: 
                        fetch_p = yf.Ticker(vs_ticker).history(period="1d")['Close'].iloc[-1]
                        auto_price = float(fetch_p)
                    except: auto_price = float(row.get('Current_Price', 0))
                    
                    vs_price = c2.number_input("Current Price ($)", value=auto_price, step=0.1, key=f"vs_px_{vs_ticker}")
                    c3.metric("Avg Cost", f"${vs_avg_cost:,.2f}") # Layout preserved
                else:
                    st.info("No open positions found to audit.")
            else:
                st.info("Summary file empty.")

        st.markdown("---")
        
        # 3. CRITICAL INPUTS (ATR + MA LEVEL + BUFFER)
        # Split into 4 columns to fit the new Buffer input
        e1, e2, e3, e4 = st.columns(4)
        vs_equity = e1.number_input("Account Equity (NLV)", value=equity, step=1000.0, key="vs_eq")
        vs_atr_pct = e2.number_input("ATR % (21-Day)", value=5.0, step=0.1, help="Enter 21-Day ATR %.", key="vs_atr_manual")
        
        # UPDATED: Split "Stop" into "MA Level" and "Buffer"
        vs_ma_level = 0.0
        vs_buffer_pct = 1.0
        
        if vol_mode.startswith("🆕"):
            vs_ma_level = e3.number_input("Key MA Level ($)", value=0.0, step=0.1, help="Price of the Moving Average (e.g. 21e/50s).", key="vs_ma_level")
            vs_buffer_pct = e4.number_input("Buffer (%)", value=1.0, step=0.1, help="Wiggle room below the MA.", key="vs_buffer")
        
        st.markdown("---")
        
        if st.button("Run Sizing Audit", type="primary", key="vs_btn"):
            if vs_ticker and vs_price > 0 and vs_atr_pct > 0:
                
                # 1. Determine Tier & Budget
                cushion_pct = 0.0
                if vol_mode.startswith("🆕"):
                    cushion_pct = 0.0 
                elif vs_avg_cost > 0:
                    cushion_pct = ((vs_price - vs_avg_cost) / vs_avg_cost) * 100
                
                tier_name = "Tier 3 (Defense)"
                tol_pct = 0.50
                
                if cushion_pct >= 20.0:
                    tier_name = "Tier 1 (High Cushion)"
                    tol_pct = 1.00
                elif cushion_pct >= 5.0:
                    tier_name = "Tier 2 (Moderate)"
                    tol_pct = 0.65
                    
                daily_risk_budget = vs_equity * (tol_pct / 100)
                
                # 2. Calculate Volatility Limit (ATR)
                atr_decimal = vs_atr_pct / 100
                max_shares_vol = int(daily_risk_budget / (vs_price * atr_decimal))
                
                # 3. Calculate Technical Limit (MA - Buffer)
                max_shares_tech = 999999
                effective_stop = 0.0
                tech_dist_pct = 0.0
                
                if vs_ma_level > 0:
                    # Logic: Stop = MA * (1 - Buffer%)
                    effective_stop = vs_ma_level * (1 - (vs_buffer_pct/100))
                    
                    if effective_stop < vs_price:
                        risk_per_share = vs_price - effective_stop
                        tech_dist_pct = (risk_per_share / vs_price) * 100
                        
                        if risk_per_share > 0:
                             max_shares_tech = int(daily_risk_budget / risk_per_share)
                
                # 4. Hard Cap (20% NLV)
                max_shares_cap = int((vs_equity * 0.20) / vs_price)
                
                # 5. Final Decision (Min of all)
                final_max_shares = min(max_shares_vol, max_shares_tech, max_shares_cap)
                final_max_val = final_max_shares * vs_price
                
                # Determine Limiting Factor
                limit_reason = "Volatility (ATR)"
                if final_max_shares == max_shares_cap: limit_reason = "Hard Cap (20%)"
                elif final_max_shares == max_shares_tech: limit_reason = f"MA Support (${vs_ma_level})"
                
                # 6. Display Results
                st.markdown(f"### 📊 Sizing Profile: {vs_ticker}")
                
                k1, k2, k3 = st.columns(3)
                k1.metric("Risk Budget", f"${daily_risk_budget:,.0f}", f"{tol_pct}% Risk ({tier_name})")
                k2.metric("Volatility Risk", f"{vs_atr_pct:.2f}%", f"ATR (Noise)")
                
                if effective_stop > 0:
                    k3.metric("Effective Stop", f"${effective_stop:.2f}", f"{vs_buffer_pct}% below ${vs_ma_level}")
                else:
                    k3.metric("Profit Cushion", f"{cushion_pct:.2f}%", tier_name, delta_color="off")
                
                st.markdown("---")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("ATR Limit", f"{max_shares_vol} shs", "Based on Noise", delta_color="off")
                
                if vol_mode.startswith("🆕") and effective_stop > 0:
                    delta_color = "normal" if max_shares_tech < max_shares_vol else "off"
                    m2.metric("Tech Stop Limit", f"{max_shares_tech} shs", "Based on Support", delta_color=delta_color)
                else:
                    m2.metric("Hard Cap Limit", f"{max_shares_cap} shs", "20% Max Alloc", delta_color="off")
                    
                m3.metric("Limiting Factor", limit_reason, "Determines Final Size", delta_color="off")
                
                st.markdown("### 🏛️ The Verdict")
                
                target_weight = (final_max_val / vs_equity) * 100
                
                if vol_mode.startswith("🆕"):
                    st.success(f"✅ **RECOMMENDED SIZE:** Buy **{final_max_shares}** shares ({target_weight:.1f}% of NLV).")
                    if limit_reason.startswith("MA"):
                        st.info(f"ℹ️ **Note:** Sized for technicals. Your stop (${effective_stop:.2f}) is {tech_dist_pct:.1f}% away (including buffer).")
                else:
                    diff_shares = vs_shares - final_max_shares
                    v1, v2, v3 = st.columns(3)
                    
                    start_weight = (vs_shares * vs_price / vs_equity) * 100
                    v1.metric("Start Position", f"{int(vs_shares)} shs", f"{start_weight:.1f}% Weight")
                    v2.metric("Target Position", f"{final_max_shares} shs", f"{target_weight:.1f}% Weight")
                    
                    if diff_shares > 0:
                        trim_val = diff_shares * vs_price
                        v3.metric("Action Required", f"TRIM {int(diff_shares)}", f"Sell ${trim_val:,.0f}", delta_color="normal")
                        st.warning(f"⚠️ **OVERWEIGHT:** You are holding {int(diff_shares)} shares too many for this volatility/technical profile.")
                    else:
                        v3.metric("Action Required", "NONE", "✅ Safe", delta_color="off")
                        st.success(f"✅ **SAFE:** Your position is within the {final_max_shares} share limit.")
            else:
                st.error("Please ensure Ticker, Price, and ATR are entered correctly.")

# ==============================================================================
# PAGE: LOG BUY (Standalone)
# ==============================================================================
elif page == "Log Buy":
    st.header(f"LOG BUY ({CURR_PORT_NAME})")
    df_d, df_s = load_trade_data()

    st.caption("Live Entry Calculator")

    # Show last upload attempt results (persists through rerun)
    with st.expander("🔍 Last Upload Attempt Results", expanded=True):
        if 'last_upload_attempt' in st.session_state:
            attempt = st.session_state['last_upload_attempt']
            st.json(attempt)
            if st.button("Clear Results"):
                del st.session_state['last_upload_attempt']
                st.rerun()
        else:
            st.info("No upload attempts yet. Upload a file and log a trade to see results here.")

    # Debug: Show system status
    with st.expander("🔧 System Status (Debug)", expanded=False):
        diag_cols = st.columns(3)
        with diag_cols[0]:
            st.metric("R2 Storage", "✅ Available" if R2_AVAILABLE else "❌ Not Available")
        with diag_cols[1]:
            st.metric("Database Mode", "✅ Enabled" if USE_DATABASE else "❌ Disabled")
        with diag_cols[2]:
            img_status = "✅ Enabled" if (R2_AVAILABLE and USE_DATABASE) else "❌ Disabled"
            st.metric("Image Upload", img_status)

        if not R2_AVAILABLE:
            st.error("R2 storage module failed to load. Check if boto3 is installed and R2 secrets are configured.")
        if not USE_DATABASE:
            st.warning("Database mode is disabled. Images require database mode to be enabled.")

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

    b_date = c_top2.date_input("Date", get_current_date_ct(), key="b_date_input")
    b_time = c_top2.time_input("Time", get_current_time_ct(), step=60, key="b_time_input")

    st.markdown("---")
    c1, c2 = st.columns(2)

    # --- 1. TICKER & STRATEGY SELECTION ---
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
        # Scale In Logic
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
        b_rule = st.selectbox("Add Rule", BUY_RULES)

    # --- 2. RISK BUDGET CALCULATOR ---
    risk_budget_dol = 0.0
    def_equity = 100000.0
    try:
        j_df = load_data(JOURNAL_FILE)
        if not j_df.empty and 'End NLV' in j_df.columns:
            if 'Day' in j_df.columns:
                j_df['Day'] = pd.to_datetime(j_df['Day'], errors='coerce')
                j_df = j_df.dropna(subset=['Day']).sort_values('Day', ascending=False)
            val_str = str(j_df['End NLV'].iloc[0]).replace('$','').replace(',','')
            def_equity = float(val_str)
    except Exception as e:
        pass

    if trade_type == "Start New Campaign":
        st.markdown("#### 💰 Risk Budgeting")
        rb1, rb2, rb3 = st.columns(3)
        risk_pct_input = rb1.number_input("Risk % of Equity", value=0.50, step=0.05, format="%.2f")
        risk_budget_dol = def_equity * (risk_pct_input / 100)

        rb2.metric("Account Equity (Prev)", f"${def_equity:,.2f}")
        rb3.metric("Hard Risk Budget ($)", f"${risk_budget_dol:.2f}")
    else:
        if b_id:
            orig_budget = df_s[df_s['Trade_ID'] == b_id]['Risk_Budget'].iloc[0]
            risk_budget_dol = float(orig_budget)
            st.caption(f"Original Risk Budget for {b_id}: ${risk_budget_dol:,.2f}")

    # --- 3. EXECUTION DETAILS ---
    c3, c4 = st.columns(2)
    b_shs = c3.number_input("Shares to Add", min_value=0, step=1, key="b_shs")
    b_px = c4.number_input("Price ($)", min_value=0.0, step=0.1, format="%.2f", key="b_px")

    # --- RBM STOP CALCULATION (THE GUARDRAIL) ---
    rbm_stop = 0.0
    if risk_budget_dol > 0 and b_shs > 0:
        if trade_type == "Start New Campaign":
            risk_per_share_allowable = risk_budget_dol / b_shs
            rbm_stop = b_px - risk_per_share_allowable
        else:
            existing_shares = df_s[df_s['Trade_ID'] == b_id]['Shares'].iloc[0]
            existing_cost = df_s[df_s['Trade_ID'] == b_id]['Total_Cost'].iloc[0]
            new_cost = b_shs * b_px
            total_shares = existing_shares + b_shs
            rbm_stop = (existing_cost + new_cost - risk_budget_dol) / total_shares

        st.info(f"🛑 **RBM Stop (Hard Deck):** ${rbm_stop:.2f} (To maintain total ${risk_budget_dol:.0f} risk)")

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

    # --- VALIDATION MESSAGE ---
    if rbm_stop > 0:
        if b_stop < rbm_stop:
            total_shs_calc = b_shs if trade_type == "Start New Campaign" else (df_s[df_s['Trade_ID'] == b_id]['Shares'].iloc[0] + b_shs)
            excess_risk = (rbm_stop - b_stop) * total_shs_calc
            st.error(f"⚠️ **RISK VIOLATION:** Your stop (${b_stop:.2f}) is too wide! It exceeds budget by ${excess_risk:.2f}.")
        elif b_stop >= b_px and trade_type == "Start New Campaign":
            st.warning("⚠️ Stop Price is above Entry Price.")
        else:
            st.success(f"✅ **WITHIN BUDGET:** Your stop respects the Risk Limit (Above ${rbm_stop:.2f}).")

    st.markdown("---")
    c_note1, c_note2 = st.columns(2)
    b_note = c_note1.text_input("Buy Rationale (Notes)", key="b_note")
    b_trx = c_note2.text_input("Manual Trx ID (Optional)", key="b_trx")

    # --- IMAGE UPLOADS (Optional) ---
    weekly_chart = None
    daily_chart = None
    if R2_AVAILABLE:
        st.markdown("#### 📸 Chart Documentation (Optional)")
        st.caption("Upload weekly and daily charts to document your entry setup")

        status_cols = st.columns([1, 1, 3])
        with status_cols[0]:
            st.caption(f"R2: {'✅' if R2_AVAILABLE else '❌'}")
        with status_cols[1]:
            st.caption(f"DB: {'✅' if USE_DATABASE else '❌'}")
        with status_cols[2]:
            if not (R2_AVAILABLE and USE_DATABASE):
                st.caption("⚠️ Images will NOT be saved")

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            weekly_chart = st.file_uploader(
                "Weekly Chart",
                type=['png', 'jpg', 'jpeg'],
                key='b_weekly_chart',
                help="Upload a screenshot of the weekly chart"
            )
            if weekly_chart:
                st.caption(f"✅ Selected: {weekly_chart.name}")
        with img_col2:
            daily_chart = st.file_uploader(
                "Daily Chart",
                type=['png', 'jpg', 'jpeg'],
                key='b_daily_chart',
                help="Upload a screenshot of the daily chart"
            )
            if daily_chart:
                st.caption(f"✅ Selected: {daily_chart.name}")

    if st.button("LOG BUY ORDER", type="primary", use_container_width=True):
        st.info(f"🔍 DEBUG - weekly_chart: {weekly_chart is not None}, daily_chart: {daily_chart is not None}")
        if weekly_chart:
            st.info(f"📁 Weekly: {weekly_chart.name}, {weekly_chart.size} bytes")
        if daily_chart:
            st.info(f"📁 Daily: {daily_chart.name}, {daily_chart.size} bytes")

        if b_tick and b_id:
            is_valid, errors = validate_trade_entry(
                action='BUY',
                ticker=b_tick,
                shares=b_shs,
                price=b_px,
                stop_loss=b_stop,
                trade_id=b_id if trade_type == "Start New Campaign" else None,
                df_s=df_s
            )

            equity_val = def_equity
            size_valid, size_msg = validate_position_size(b_shs, b_px, equity_val, max_pct=25.0)

            if not is_valid:
                st.error(f"**Validation failed with {len(errors)} error(s):**")
                for error in errors:
                    if "Warning" in error:
                        st.warning(error)
                    else:
                        st.error(error)

            if not size_valid:
                st.error(size_msg)

            critical_errors = [e for e in errors if "❌" in e]
            if critical_errors or not size_valid:
                st.error("❌ Cannot proceed - fix validation errors above")
                st.stop()

            if size_msg:
                st.warning(size_msg)

            # --- PROCEED WITH TRADE ---
            ts = datetime.combine(b_date, b_time).strftime("%Y-%m-%d %H:%M")
            cost = b_shs * b_px
            if not b_trx: b_trx = generate_trx_id(df_d, b_id, 'BUY', ts)

            if trade_type == "Start New Campaign":
                new_s = {
                    'Trade_ID': b_id, 'Ticker': b_tick, 'Status': 'OPEN', 'Open_Date': ts,
                    'Shares': 0, 'Avg_Entry': 0, 'Total_Cost': 0, 'Realized_PL': 0, 'Unrealized_PL': 0,
                    'Rule': b_rule,
                    'Notes': b_note,
                    'Buy_Notes': b_note,
                    'Risk_Budget': risk_budget_dol,
                    'Sell_Rule': '', 'Sell_Notes': ''
                }

                if USE_DATABASE:
                    try:
                        db.save_summary_row(portfolio, new_s)
                    except Exception as e:
                        st.warning(f"⚠️ Database save failed: {e}. CSV saved successfully.")

                df_s = pd.concat([df_s, pd.DataFrame([new_s])], ignore_index=True)

            new_d = {'Trade_ID': b_id, 'Trx_ID': b_trx, 'Ticker': b_tick, 'Action': 'BUY', 'Date': ts, 'Shares': b_shs, 'Amount': b_px, 'Value': cost, 'Rule': b_rule, 'Notes': b_note, 'Realized_PL': 0, 'Stop_Loss': b_stop}

            if USE_DATABASE:
                try:
                    db.save_detail_row(portfolio, new_d)
                except Exception as e:
                    st.warning(f"⚠️ Database save failed: {e}. CSV saved successfully.")

            df_d = pd.concat([df_d, pd.DataFrame([new_d])], ignore_index=True)

            df_d, df_s = update_campaign_summary(b_id, df_d, df_s)

            secure_save(df_d, DETAILS_FILE)
            secure_save(df_s, SUMMARY_FILE)

            log_audit_trail(
                action='BUY',
                trade_id=b_id,
                ticker=b_tick,
                details=f"{b_shs} shares @ ${b_px:.2f} | Cost: ${cost:.2f} | Rule: {b_rule}"
            )

            # --- UPLOAD IMAGES (if provided) ---
            print(f"[UPLOAD] Checking upload conditions: R2={R2_AVAILABLE}, DB={USE_DATABASE}, weekly={weekly_chart is not None}, daily={daily_chart is not None}")

            st.session_state['last_upload_attempt'] = {
                'R2_AVAILABLE': R2_AVAILABLE,
                'USE_DATABASE': USE_DATABASE,
                'weekly_chart': weekly_chart is not None,
                'daily_chart': daily_chart is not None,
                'upload_results': []
            }

            if R2_AVAILABLE and USE_DATABASE:
                print("[UPLOAD] Entering upload block")
                st.session_state['last_upload_attempt']['entered_block'] = True
                images_uploaded = []

                try:
                    if weekly_chart is not None:
                        print(f"[UPLOAD] About to call r2.upload_image for weekly chart")
                        st.session_state['last_upload_attempt']['upload_results'].append('Attempting weekly upload...')
                        st.info(f"Uploading weekly chart: {weekly_chart.name}")
                        weekly_url = r2.upload_image(weekly_chart, portfolio, b_id, b_tick, 'weekly')
                        print(f"[UPLOAD] r2.upload_image returned: {weekly_url}")
                        st.session_state['last_upload_attempt']['upload_results'].append(f'Weekly result: {weekly_url}')
                        if weekly_url:
                            print(f"[UPLOAD] Saving to database: {weekly_url}")
                            db.save_trade_image(portfolio, b_id, b_tick, 'weekly', weekly_url, weekly_chart.name)
                            images_uploaded.append('Weekly')
                            st.session_state['last_upload_attempt']['upload_results'].append('Weekly: SUCCESS')
                            print(f"[UPLOAD] Successfully saved weekly chart")
                        else:
                            print(f"[UPLOAD] weekly_url is None - upload failed")
                            st.session_state['last_upload_attempt']['upload_results'].append('Weekly: FAILED (None returned)')
                            st.error("Failed to upload weekly chart to R2")

                    if daily_chart is not None:
                        st.info(f"Uploading daily chart: {daily_chart.name}")
                        daily_url = r2.upload_image(daily_chart, portfolio, b_id, b_tick, 'daily')
                        if daily_url:
                            db.save_trade_image(portfolio, b_id, b_tick, 'daily', daily_url, daily_chart.name)
                            images_uploaded.append('Daily')
                        else:
                            st.error("Failed to upload daily chart to R2")

                    if images_uploaded:
                        st.success(f"📸 Uploaded charts: {', '.join(images_uploaded)}")
                    elif weekly_chart is not None or daily_chart is not None:
                        st.warning("Charts were selected but upload failed")
                except Exception as e:
                    st.error(f"Image upload error: {str(e)}")
            elif weekly_chart is not None or daily_chart is not None:
                if not R2_AVAILABLE:
                    st.warning("⚠️ R2 storage not available - charts not uploaded")
                if not USE_DATABASE:
                    st.warning("⚠️ Database mode disabled - charts not uploaded")

            st.success(f"✅ EXECUTED: Bought {b_shs} {b_tick} @ ${b_px}")
            for k in ['b_tick','b_id','b_shs','b_px','b_note','b_trx','b_stop_val','b_weekly_chart','b_daily_chart']:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
        else:
            st.error("⚠️ Missing required fields: Ticker and Trade ID are required.")

# ==============================================================================
# PAGE: LOG SELL (Standalone)
# ==============================================================================
elif page == "Log Sell":
    st.header(f"LOG SELL ({CURR_PORT_NAME})")
    df_d, df_s = load_trade_data()

    # Display success message from previous sell if exists
    if 'sell_success' in st.session_state:
        st.success(st.session_state.sell_success)
        del st.session_state.sell_success

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
            s_date = c1.date_input("Date", get_current_date_ct(), key='s_date')
            s_time = c2.time_input("Time", get_current_time_ct(), step=60, key='s_time')

            c3, c4 = st.columns(2)
            s_shs = c3.number_input("Shares", min_value=1, max_value=int(row['Shares']), step=1)
            s_px = c4.number_input("Price", min_value=0.0, step=0.1)

            # --- EXPLICIT SELL RULE & NOTES ---
            c5, c6 = st.columns(2)
            s_rule = c5.selectbox("Sell Rule / Reason", SELL_RULES)
            s_note = c6.text_input("Sell Context / Notes", key='s_note', placeholder="Why did you sell?")
            s_trx = st.text_input("Manual Trx ID (Optional)", key='s_trx')

            # --- EXIT CHART UPLOAD (Optional) ---
            exit_chart = None
            if R2_AVAILABLE:
                st.markdown("#### 📸 Exit Chart (Optional)")
                exit_chart = st.file_uploader(
                    "Upload Exit Chart",
                    type=['png', 'jpg', 'jpeg'],
                    key='s_exit_chart',
                    help="Upload a screenshot showing the exit point"
                )

            if st.button("LOG SELL ORDER", type="primary"):
                # --- VALIDATION CHECKS ---
                is_valid, errors = validate_trade_entry(
                    action='SELL',
                    ticker=s_tick,
                    shares=s_shs,
                    price=s_px,
                    trade_id=s_id,
                    df_s=df_s
                )

                if not is_valid:
                    for error in errors:
                        if "Warning" in error:
                            st.warning(error)
                        else:
                            st.error(error)

                critical_errors = [e for e in errors if "❌" in e]
                if critical_errors:
                    st.stop()

                # --- PROCEED WITH SELL ---
                ts = datetime.combine(s_date, s_time).strftime("%Y-%m-%d %H:%M")
                proc = s_shs * s_px
                if not s_trx: s_trx = generate_trx_id(df_d, s_id, 'SELL', ts)

                new_d = {'Trade_ID':s_id, 'Trx_ID': s_trx, 'Ticker':s_tick, 'Action':'SELL', 'Date':ts, 'Shares':s_shs, 'Amount':s_px, 'Value':proc, 'Rule':s_rule, 'Notes': s_note, 'Realized_PL': 0}

                if USE_DATABASE:
                    try:
                        db.save_detail_row(CURR_PORT_NAME, new_d)

                        df_d_temp = db.load_details(CURR_PORT_NAME, s_id)
                        df_s_temp = db.load_summary(CURR_PORT_NAME)
                        df_d_temp, df_s_temp = update_campaign_summary(s_id, df_d_temp, df_s_temp)

                        summary_matches = df_s_temp[df_s_temp['Trade_ID'].astype(str) == str(s_id)]
                        if not summary_matches.empty:
                            summary_row = summary_matches.iloc[0].to_dict()
                            summary_row['Sell_Rule'] = s_rule
                            summary_row['Sell_Notes'] = s_note
                            db.save_summary_row(CURR_PORT_NAME, summary_row)
                            realized_pl = summary_row.get('Realized_PL', 0)
                        else:
                            realized_pl = 0

                        log_audit_trail(
                            action='SELL',
                            trade_id=s_id,
                            ticker=s_tick,
                            details=f"{s_shs} shares @ ${s_px:.2f} | Proceeds: ${proc:.2f} | Rule: {s_rule} | P&L: ${realized_pl:.2f}"
                        )

                        chart_uploaded = False
                        if R2_AVAILABLE and exit_chart is not None:
                            try:
                                exit_url = r2.upload_image(exit_chart, CURR_PORT_NAME, s_id, s_tick, 'exit')
                                if exit_url:
                                    db.save_trade_image(CURR_PORT_NAME, s_id, s_tick, 'exit', exit_url, exit_chart.name)
                                    chart_uploaded = True
                            except Exception as chart_err:
                                st.warning(f"⚠️ Sell saved but chart upload failed: {chart_err}")

                        if chart_uploaded:
                            st.session_state.sell_success = f"✅ Sold! Transaction ID: {s_trx} | Exit chart uploaded | Saved to database"
                        else:
                            st.session_state.sell_success = f"✅ Sold! Transaction ID: {s_trx} | Saved to database"
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Database save failed: {str(e)}")
                else:
                    # CSV fallback
                    df_d = pd.concat([df_d, pd.DataFrame([new_d])], ignore_index=True)
                    secure_save(df_d, DETAILS_FILE)

                    df_d, df_s = update_campaign_summary(s_id, df_d, df_s)

                    idx = df_s[df_s['Trade_ID'] == s_id].index
                    if not idx.empty:
                        df_s.at[idx[0], 'Sell_Rule'] = s_rule
                        df_s.at[idx[0], 'Sell_Notes'] = s_note

                    secure_save(df_s, SUMMARY_FILE)

                    realized_pl = df_s[df_s['Trade_ID'] == s_id]['Realized_PL'].iloc[0] if not df_s[df_s['Trade_ID'] == s_id].empty else 0
                    log_audit_trail(
                        action='SELL',
                        trade_id=s_id,
                        ticker=s_tick,
                        details=f"{s_shs} shares @ ${s_px:.2f} | Proceeds: ${proc:.2f} | Rule: {s_rule} | P&L: ${realized_pl:.2f}"
                    )

                    chart_uploaded = False
                    if R2_AVAILABLE and USE_DATABASE and exit_chart is not None:
                        try:
                            exit_url = r2.upload_image(exit_chart, CURR_PORT_NAME, s_id, s_tick, 'exit')
                            if exit_url:
                                db.save_trade_image(CURR_PORT_NAME, s_id, s_tick, 'exit', exit_url, exit_chart.name)
                                chart_uploaded = True
                        except Exception as chart_err:
                            st.warning(f"⚠️ Sell saved but chart upload failed: {chart_err}")

                    if chart_uploaded:
                        st.session_state.sell_success = f"✅ Sold! Transaction ID: {s_trx} | Exit chart uploaded"
                    else:
                        st.session_state.sell_success = f"✅ Sold! Transaction ID: {s_trx}"
                    st.rerun()
    else: st.info("No positions to sell.")

# ==============================================================================
# PAGE 10: TRADE MANAGER (FULL CONTEXT: BUY/SELL NOTES & RULES)
# ==============================================================================
elif page == "Trade Manager":
    st.header(f"TRADE MANAGER ({CURR_PORT_NAME})")

    df_d, df_s = load_trade_data()

    valid_sum_cols = ['Trade_ID', 'Ticker', 'Status', 'Open_Date', 'Shares', 'Avg_Entry', 'Total_Cost', 'Unrealized_PL', 'Return_Pct', 'Rule', 'Buy_Notes', 'Sell_Rule']
    valid_sum_cols = [c for c in valid_sum_cols if c in df_s.columns]

    # --------------------------------------------------------------------------
    # TAB LIST (Operational tabs only)
    # --------------------------------------------------------------------------
    tab3, tab4, tab5, tab6, tab8, tab9, tab_cy, tab10 = st.tabs([
        "Update Prices",
        "Edit Transaction",
        "Database Health",
        "Delete Trade",
        "Active Campaign Detailed",
        "Detailed Trade Log",
        "CY Campaigns (2026)",
        "All Campaigns"
    ])

    # Note: The following are now standalone pages:
    # - Log Buy → 💼 Trading Ops section
    # - Log Sell → 💼 Trading Ops section
    # - Active Campaign Summary → 💼 Trading Ops section
    # - Risk Manager → 🛡️ Risk Management section
    # - Portfolio Heat → 🛡️ Risk Management section
    # - Earnings Planner → 🛡️ Risk Management section
    # - Performance Audit → 🔍 Deep Dive section

    # --- TAB 3: UPDATE PRICES ---
    with tab3:
        st.subheader("🛡️ Risk Control Center")
        if st.button("REFRESH MARKET PRICES", type="primary"):
            if USE_DATABASE:
                # Database-first approach - use existing refresh function
                with st.spinner("Fetching current prices..."):
                    try:
                        result = db.refresh_open_position_prices(CURR_PORT_NAME)
                        if 'error' in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            st.success(f"✅ {result['message']} (saved to database)")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Price refresh failed: {str(e)}")
            else:
                # CSV fallback
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

                    if USE_DATABASE:
                        # Database-first approach
                        try:
                            # Get database ID
                            db_id = df_d.at[last_idx, '_DB_ID']
                            if pd.isna(db_id):
                                st.error("❌ Cannot update: Database ID not found")
                                st.stop()

                            # Prepare update dict
                            update_dict = {
                                'Trade_ID': df_d.at[last_idx, 'Trade_ID'],
                                'Ticker': df_d.at[last_idx, 'Ticker'],
                                'Action': df_d.at[last_idx, 'Action'],
                                'Date': df_d.at[last_idx, 'Date'],
                                'Shares': df_d.at[last_idx, 'Shares'],
                                'Amount': df_d.at[last_idx, 'Amount'],
                                'Value': df_d.at[last_idx, 'Value'],
                                'Rule': df_d.at[last_idx, 'Rule'],
                                'Notes': df_d.at[last_idx, 'Notes'],
                                'Stop_Loss': new_stop_price,
                                'Trx_ID': df_d.at[last_idx, 'Trx_ID']
                            }

                            # Update database
                            db.update_detail_row(CURR_PORT_NAME, int(db_id), update_dict)
                            st.success(f"✅ Stop Updated to ${new_stop_price:.2f} (saved to database)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Database update failed: {str(e)}")
                    else:
                        # CSV fallback
                        df_d.at[last_idx, 'Stop_Loss'] = new_stop_price
                        secure_save(df_d, DETAILS_FILE)
                        st.success(f"✅ Stop Updated to ${new_stop_price:.2f}")
                        st.rerun()
                else: st.error("Could not find a BUY transaction.")
        else: st.info("No active positions.")

    # --- TAB 4: EDIT TRANSACTION (CALCULATED FILTER) ---
    with tab4:
        st.header("📝 Transaction Maintenance")
        
        edit_mode = st.radio("Select Mode", ["🚀 Bulk Update Stops (Fast)", "🛠️ Single Transaction Edit (Deep Fix)"], horizontal=True)
        
        # --- MODE A: BULK STOP UPDATER ---
        if edit_mode == "🚀 Bulk Update Stops (Fast)":
            st.caption("Rapidly update Stop Losses for ACTIVE BUY tranches (Calculated Remaining > 0).")
            
            if not df_s.empty and not df_d.empty:
                open_ids = df_s[df_s['Status'] == 'OPEN']['Trade_ID'].unique().tolist()
                
                # --- PRE-CALCULATE REMAINING SHARES FOR ALL OPEN TRADES ---
                # We need to map every BUY row index to its remaining share count
                remaining_map = {}
                
                # Filter Df_d to relevant trades only for speed
                relevant_txs = df_d[df_d['Trade_ID'].isin(open_ids)].copy()
                
                for tid in open_ids:
                    subset = relevant_txs[relevant_txs['Trade_ID'] == tid].copy()
                    # Sort: Date asc, Buys(0) before Sells(1)
                    subset['Type_Rank'] = subset['Action'].apply(lambda x: 0 if x == 'BUY' else 1)
                    subset = subset.sort_values(['Date', 'Type_Rank'])
                    
                    inventory = [] # Stores: {'idx': original_index, 'qty': shares}
                    
                    for idx, row in subset.iterrows():
                        if row['Action'] == 'BUY':
                            inventory.append({'idx': idx, 'qty': row['Shares']})
                            remaining_map[idx] = row['Shares'] # Init with full
                            
                        elif row['Action'] == 'SELL':
                            to_sell = row['Shares']
                            # LIFO Pop
                            while to_sell > 0 and inventory:
                                last = inventory.pop()
                                take = min(to_sell, last['qty'])
                                last['qty'] -= take
                                to_sell -= take
                                remaining_map[last['idx']] = last['qty'] # Update map
                                
                                if last['qty'] > 0.001: inventory.append(last)
                
                # --- APPLY FILTER ---
                # 1. Must be in Open IDs
                # 2. Must be a BUY
                # 3. Calculated Remaining > 0
                
                # Create mask
                mask = (df_d['Trade_ID'].isin(open_ids)) & (df_d['Action'] == 'BUY')
                
                # Filter logic
                valid_indices = []
                for idx in df_d[mask].index:
                    rem = remaining_map.get(idx, 0.0)
                    if rem > 0.01: # Strict filter
                        valid_indices.append(idx)
                
                if valid_indices:
                    # Prepare View
                    cols = ['Date', 'Trx_ID', 'Ticker', 'Amount', 'Stop_Loss', 'Notes']
                    df_bulk = df_d.loc[valid_indices, cols].copy()
                    
                    # Add Calculated Remaining Column for Visual Context
                    df_bulk['Held'] = [remaining_map[i] for i in valid_indices]
                    
                    # Reorder columns
                    df_bulk = df_bulk[['Date', 'Trx_ID', 'Ticker', 'Held', 'Amount', 'Stop_Loss', 'Notes']]
                    
                    # Format Date
                    if 'Date' in df_bulk.columns:
                        df_bulk['Date'] = pd.to_datetime(df_bulk['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    
                    # Sort
                    df_bulk = df_bulk.sort_values(['Ticker', 'Date'], ascending=[True, False])
                    
                    # EDITOR
                    edited_df = st.data_editor(
                        df_bulk,
                        column_config={
                            "Date": st.column_config.TextColumn("Date", disabled=True),
                            "Trx_ID": st.column_config.TextColumn("Trx ID", disabled=True, width="small"),
                            "Ticker": st.column_config.TextColumn("Ticker", disabled=True, width="small"),
                            "Held": st.column_config.NumberColumn("Held Shares", disabled=True, width="small"),
                            "Amount": st.column_config.NumberColumn("Entry Price", format="$%.2f", disabled=True),
                            "Stop_Loss": st.column_config.NumberColumn("Stop Loss ($)", format="$%.2f", required=True, width="medium"),
                            "Notes": st.column_config.TextColumn("Notes", width="large")
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=600
                    )
                    
                    # SAVE LOGIC
                    if st.button("💾 Save All Bulk Changes", type="primary"):
                        changes = False
                        affected = set()

                        if USE_DATABASE:
                            # Database-first approach
                            try:
                                for idx, row in edited_df.iterrows():
                                    # idx matches df_d index
                                    old_stop = df_d.at[idx, 'Stop_Loss']
                                    new_stop = row['Stop_Loss']
                                    old_note = df_d.at[idx, 'Notes']
                                    new_note = row['Notes']

                                    needs_update = False
                                    if (pd.isna(old_stop) and new_stop > 0) or (old_stop != new_stop):
                                        needs_update = True
                                        affected.add(df_d.at[idx, 'Trade_ID'])

                                    if str(old_note) != str(new_note):
                                        needs_update = True

                                    if needs_update:
                                        # Get database ID
                                        db_id = df_d.at[idx, '_DB_ID']
                                        if pd.isna(db_id):
                                            st.warning(f"⚠️ Skipping row {idx}: Database ID not found")
                                            continue

                                        # Prepare update dict
                                        update_dict = {
                                            'Trade_ID': df_d.at[idx, 'Trade_ID'],
                                            'Ticker': df_d.at[idx, 'Ticker'],
                                            'Action': df_d.at[idx, 'Action'],
                                            'Date': df_d.at[idx, 'Date'],
                                            'Shares': df_d.at[idx, 'Shares'],
                                            'Amount': df_d.at[idx, 'Amount'],
                                            'Value': df_d.at[idx, 'Value'],
                                            'Rule': df_d.at[idx, 'Rule'],
                                            'Notes': new_note,
                                            'Stop_Loss': new_stop,
                                            'Trx_ID': df_d.at[idx, 'Trx_ID']
                                        }

                                        # Update database
                                        db.update_detail_row(CURR_PORT_NAME, int(db_id), update_dict)
                                        changes = True

                                if changes and affected:
                                    # Recalculate summaries for affected trades
                                    prog = st.progress(0)
                                    for i, tid in enumerate(affected):
                                        df_d_temp = db.load_details(CURR_PORT_NAME, tid)
                                        df_s_temp = db.load_summary(CURR_PORT_NAME)
                                        df_d_temp, df_s_temp = update_campaign_summary(tid, df_d_temp, df_s_temp)

                                        # Save summary
                                        summary_matches = df_s_temp[df_s_temp['Trade_ID'].astype(str) == str(tid)]
                                        if not summary_matches.empty:
                                            summary_row = summary_matches.iloc[0].to_dict()
                                            db.save_summary_row(CURR_PORT_NAME, summary_row)

                                        prog.progress((i+1)/len(affected))

                                    st.success(f"✅ Saved {len(affected)} trade(s) to database!")
                                    st.rerun()
                                elif not changes:
                                    st.info("No changes detected.")
                            except Exception as e:
                                st.error(f"❌ Bulk update failed: {str(e)}")
                        else:
                            # CSV fallback
                            for idx, row in edited_df.iterrows():
                                # idx matches df_d index
                                old_stop = df_d.at[idx, 'Stop_Loss']
                                new_stop = row['Stop_Loss']
                                old_note = df_d.at[idx, 'Notes']
                                new_note = row['Notes']

                                if (pd.isna(old_stop) and new_stop > 0) or (old_stop != new_stop):
                                    df_d.at[idx, 'Stop_Loss'] = new_stop
                                    changes = True
                                    affected.add(df_d.at[idx, 'Trade_ID'])

                                if str(old_note) != str(new_note):
                                    df_d.at[idx, 'Notes'] = new_note
                                    changes = True

                            if changes:
                                secure_save(df_d, DETAILS_FILE)
                                prog = st.progress(0)
                                for i, tid in enumerate(affected):
                                    df_d, df_s = update_campaign_summary(tid, df_d, df_s)
                                    prog.progress((i+1)/len(affected))
                                secure_save(df_s, SUMMARY_FILE)
                                st.success("✅ Saved!"); st.rerun()
                            else: st.info("No changes.")
                else: st.info("No active buy tranches found.")
            else: st.info("No data.")

        # --- MODE B: SINGLE EDIT (LEGACY) ---
        else:
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

                                        if USE_DATABASE:
                                            # Database-first approach
                                            try:
                                                # Get database ID
                                                db_id = current_row.get('_DB_ID')
                                                if pd.isna(db_id):
                                                    st.error("❌ Cannot update: Database ID not found. Try refreshing the page.")
                                                    st.stop()

                                                # Prepare update data
                                                update_dict = {
                                                    'Trade_ID': edit_id,
                                                    'Ticker': current_row['Ticker'],
                                                    'Action': current_row['Action'],
                                                    'Date': new_ts,
                                                    'Shares': e_shs,
                                                    'Amount': e_amt,
                                                    'Value': e_shs * e_amt,
                                                    'Rule': e_rule,
                                                    'Notes': e_note,
                                                    'Stop_Loss': e_stop,
                                                    'Trx_ID': e_trx
                                                }

                                                # Update database
                                                db.update_detail_row(CURR_PORT_NAME, int(db_id), update_dict)

                                                # Recalculate summary (LIFO engine will run)
                                                df_d_temp = db.load_details(CURR_PORT_NAME, edit_id)
                                                df_s_temp = db.load_summary(CURR_PORT_NAME)
                                                df_d_temp, df_s_temp = update_campaign_summary(edit_id, df_d_temp, df_s_temp)

                                                # Save summary back to database
                                                # Ensure Trade_ID type matches
                                                summary_matches = df_s_temp[df_s_temp['Trade_ID'].astype(str) == str(edit_id)]
                                                if not summary_matches.empty:
                                                    summary_row = summary_matches.iloc[0].to_dict()
                                                    db.save_summary_row(CURR_PORT_NAME, summary_row)
                                                else:
                                                    st.warning(f"⚠️ Summary row for Trade_ID '{edit_id}' not found after update. Transaction updated but summary may need manual refresh.")

                                                st.success("✅ Updated in database!"); st.rerun()
                                            except Exception as e:
                                                st.error(f"❌ Database update failed: {str(e)}")
                                        else:
                                            # CSV fallback
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
                                    if USE_DATABASE:
                                        # Database-first approach
                                        try:
                                            # Get database ID
                                            db_id = current_row.get('_DB_ID')
                                            if pd.isna(db_id):
                                                st.error("❌ Cannot delete: Database ID not found.")
                                                st.stop()

                                            # Delete from database
                                            db.delete_detail_row(CURR_PORT_NAME, int(db_id))

                                            # Recalculate summary
                                            df_d_temp = db.load_details(CURR_PORT_NAME, edit_id)
                                            df_s_temp = db.load_summary(CURR_PORT_NAME)

                                            if not df_d_temp.empty:
                                                df_d_temp, df_s_temp = update_campaign_summary(edit_id, df_d_temp, df_s_temp)
                                                # Ensure Trade_ID type matches
                                                summary_matches = df_s_temp[df_s_temp['Trade_ID'].astype(str) == str(edit_id)]
                                                if not summary_matches.empty:
                                                    summary_row = summary_matches.iloc[0].to_dict()
                                                    db.save_summary_row(CURR_PORT_NAME, summary_row)

                                            st.warning("🗑️ Transaction deleted from database."); st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Delete failed: {str(e)}")
                                    else:
                                        # CSV fallback
                                        df_d = df_d.drop(row_idx)
                                        secure_save(df_d, DETAILS_FILE)
                                        df_d, df_s = update_campaign_summary(edit_id, df_d, df_s)
                                        secure_save(df_s, SUMMARY_FILE)
                                        st.warning("Transaction Deleted."); st.rerun()

    # --- TAB 5: DATABASE HEALTH ---
    with tab5:
        st.subheader("Database Maintenance")
        st.info("ℹ️ This tool recalculates all campaign summaries from transaction details.")

        # Show what will be rebuilt
        if not df_d.empty:
            det_ids = df_d['Trade_ID'].unique()
            sum_ids = df_s['Trade_ID'].unique() if not df_s.empty else []
            missing = [tid for tid in det_ids if tid not in sum_ids]

            st.write(f"**Total Campaigns:** {len(det_ids)}")
            if missing:
                st.warning(f"⚠️ **Missing Summaries:** {len(missing)} campaigns need summary records")

        # Confirmation
        st.markdown("---")
        rebuild_confirm = st.checkbox("I understand this will recalculate all campaigns", key='rebuild_confirm')

        if st.button("FULL REBUILD (Generate Missing Summaries)", type="secondary", disabled=not rebuild_confirm):
            if df_d.empty:
                st.error("Details file is empty.")
            else:
                # Create backup before rebuild
                backup_dir = globals().get('BACKUP_DIR', os.path.join(os.path.dirname(DETAILS_FILE), 'backups'))
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_s = os.path.join(backup_dir, f"Summary_pre_rebuild_{timestamp}.csv")
                backup_d = os.path.join(backup_dir, f"Details_pre_rebuild_{timestamp}.csv")
                df_s.to_csv(backup_s, index=False)
                df_d.to_csv(backup_d, index=False)

                # Generate missing summaries
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

                # Rebuild all campaigns
                all_ids = df_d['Trade_ID'].unique()
                p=st.progress(0)
                for i, tid in enumerate(all_ids):
                    df_d, df_s = update_campaign_summary(tid, df_d, df_s)
                    p.progress((i+1)/len(all_ids))

                secure_save(df_d, DETAILS_FILE)
                secure_save(df_s, SUMMARY_FILE)

                # Log to audit
                log_audit_trail(
                    action='REBUILD',
                    trade_id='ALL',
                    ticker='N/A',
                    details=f"Full database rebuild: {len(all_ids)} campaigns recalculated"
                )

                st.success(f"✅ Rebuilt {len(all_ids)} campaigns. Backup saved.")
                st.rerun()

    # --- TAB 6: DELETE TRADE ---
    with tab6:
        st.warning("⚠️ **Danger Zone**: Deleting a trade will permanently remove ALL transactions for that campaign.")

        del_id = st.selectbox("ID to Delete", df_s['Trade_ID'].tolist() if not df_s.empty else [])

        if del_id:
            # Show what will be deleted
            trade_info = df_s[df_s['Trade_ID'] == del_id]
            if not trade_info.empty:
                row = trade_info.iloc[0]
                st.info(f"**{row['Ticker']}** | Status: {row['Status']} | {int(row['Shares'])} shares")

                # Count transactions
                trx_count = len(df_d[df_d['Trade_ID'] == del_id])
                st.warning(f"This will delete **{trx_count} transaction(s)** for this trade.")

        # Confirmation step
        st.markdown("---")
        confirm = st.text_input("Type **DELETE** to confirm (case-sensitive):", key='delete_confirm')

        if st.button("DELETE PERMANENTLY", type="secondary"):
            if confirm == "DELETE":
                # Create backup before delete
                backup_dir = globals().get('BACKUP_DIR', os.path.join(os.path.dirname(DETAILS_FILE), 'backups'))
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_s = os.path.join(backup_dir, f"Summary_pre_delete_{del_id}_{timestamp}.csv")
                backup_d = os.path.join(backup_dir, f"Details_pre_delete_{del_id}_{timestamp}.csv")
                df_s.to_csv(backup_s, index=False)
                df_d.to_csv(backup_d, index=False)

                # Log to audit trail
                log_audit_trail(
                    action='DELETE',
                    trade_id=del_id,
                    ticker=trade_info.iloc[0]['Ticker'] if not trade_info.empty else 'UNKNOWN',
                    details=f"Deleted entire campaign with {trx_count} transactions"
                )

                # Delete from database first
                if USE_DATABASE:
                    try:
                        db.delete_trade(portfolio, del_id)
                        st.info("🗑️ Deleted from database")
                    except Exception as e:
                        st.warning(f"⚠️ Database delete failed: {e}. CSV deleted successfully.")

                # Perform deletion from DataFrames
                df_s = df_s[df_s['Trade_ID']!=del_id]
                df_d = df_d[df_d['Trade_ID']!=del_id]
                secure_save(df_s, SUMMARY_FILE)
                secure_save(df_d, DETAILS_FILE)

# ==============================================================================
# TAB 8: ACTIVE CAMPAIGN DETAILED (DYNAMIC FLIGHT DECK + ORIG COST)
# ==============================================================================
    with tab8:
        st.subheader("Active Campaign Detailed (Transactions)")
        if not df_d.empty and not df_s.empty:
            open_ids = df_s[df_s['Status'] == 'OPEN']['Trade_ID'].unique().tolist()
            view_df = df_d[df_d['Trade_ID'].isin(open_ids)].copy()
            
            if not view_df.empty:
                unique_open_tickers = sorted(view_df['Ticker'].unique().tolist())
                tick_filter = st.selectbox("Filter Open Ticker", ["All"] + unique_open_tickers, key='act_det')
                
                # --- NEW: FILTER BY STATUS WIDGET ---
                status_filter = st.radio("Filter Status", ["All", "Open", "Closed"], index=1, horizontal=True, key='act_stat_filter')
                
                # --- 1. PREPARE DATA (GLOBAL OR FILTERED) ---
                if tick_filter != "All":
                    target_df = view_df[view_df['Ticker'] == tick_filter].copy()
                else:
                    target_df = view_df.copy()

                # --- 2. RUN LIFO ENGINE FIRST (SOURCE OF TRUTH) ---
                remaining_map = {}
                lifo_pl_map = {} 
                
                fd_realized_pl = 0.0
                fd_remaining_shares = 0.0
                fd_cost_basis_sum = 0.0
                
                curr_prices = {}
                for idx, row in df_s[df_s['Status']=='OPEN'].iterrows():
                    if row['Shares'] > 0: 
                        val = row['Total_Cost'] + row.get('Unrealized_PL', 0)
                        curr_prices[row['Trade_ID']] = val / row['Shares']
                
                for tid in target_df['Trade_ID'].unique():
                    subset = target_df[target_df['Trade_ID'] == tid].copy()
                    subset['Type_Rank'] = subset['Action'].apply(lambda x: 0 if x == 'BUY' else 1)
                    subset = subset.sort_values(['Date', 'Type_Rank'])
                    
                    inventory = [] 
                    
                    for idx, row in subset.iterrows():
                        if row['Action'] == 'BUY':
                            p = float(row.get('Amount', row.get('Price', 0.0)))
                            inventory.append({'idx': idx, 'qty': row['Shares'], 'price': p})
                            remaining_map[idx] = row['Shares']
                            
                        elif row['Action'] == 'SELL':
                            to_sell = row['Shares']
                            sell_price = float(row.get('Amount', row.get('Price', 0.0)))
                            cost_basis_accum = 0.0
                            sold_qty_accum = 0.0
                            
                            while to_sell > 0 and inventory:
                                last = inventory[-1] 
                                take = min(to_sell, last['qty'])
                                
                                cost_basis_accum += (take * last['price'])
                                sold_qty_accum += take
                                
                                last['qty'] -= take
                                to_sell -= take
                                remaining_map[last['idx']] = last['qty']
                                
                                if last['qty'] < 0.00001: inventory.pop()
                            
                            revenue = sold_qty_accum * sell_price
                            true_pl = revenue - cost_basis_accum
                            lifo_pl_map[idx] = true_pl
                            fd_realized_pl += true_pl

                    for item in inventory:
                        fd_remaining_shares += item['qty']
                        fd_cost_basis_sum += (item['qty'] * item['price'])

                # Apply to DataFrame
                display_df = target_df.copy()
                display_df['Remaining_Shares'] = display_df.index.map(remaining_map).fillna(0)
                display_df['Realized_PL'] = display_df.index.map(lifo_pl_map).fillna(0)
                
                # --- CALCULATE STATUS COLUMN ---
                display_df['Status'] = display_df['Remaining_Shares'].apply(lambda x: 'Open' if x > 0 else 'Closed')
                
                # --- 3. THE FLIGHT DECK (DYNAMIC) ---
                if tick_filter != "All":
                    try:
                        live_px = yf.Ticker(tick_filter).history(period="1d")['Close'].iloc[-1]
                    except:
                        live_px = 0.0
                        if fd_remaining_shares > 0:
                            live_px = fd_cost_basis_sum / fd_remaining_shares
                    
                    shares = fd_remaining_shares
                    avg_cost = (fd_cost_basis_sum / shares) if shares > 0 else 0.0
                    mkt_val = shares * live_px
                    unrealized = mkt_val - fd_cost_basis_sum
                    return_pct = (unrealized / fd_cost_basis_sum * 100) if fd_cost_basis_sum > 0 else 0.0
                    
                    # Load equity from journal (database-aware)
                    equity = 100000.0
                    try:
                        j_df = load_data(JOURNAL_FILE)
                        if not j_df.empty and 'End NLV' in j_df.columns:
                            # Sort by date to get the latest entry
                            if 'Day' in j_df.columns:
                                j_df['Day'] = pd.to_datetime(j_df['Day'], errors='coerce')
                                j_df = j_df.dropna(subset=['Day']).sort_values('Day', ascending=False)
                            equity = float(str(j_df['End NLV'].iloc[0]).replace('$','').replace(',',''))
                    except:
                        pass
                    
                    pos_size_pct = (mkt_val / equity) * 100 if equity > 0 else 0.0

                    # --- NEW LOGIC: ORIGINAL AVG COST ---
                    # Filter for Initial Buys (Trx_ID starts with 'B')
                    orig_avg_cost = 0.0
                    if 'Trx_ID' in target_df.columns:
                        # Ensure string and handle case
                        init_buys = target_df[target_df['Trx_ID'].astype(str).str.upper().str.startswith('B')]
                        if not init_buys.empty:
                            init_val = (init_buys['Shares'] * init_buys['Amount']).sum()
                            init_shs = init_buys['Shares'].sum()
                            orig_avg_cost = init_val / init_shs if init_shs > 0 else 0.0

                    st.markdown(f"### 🚁 Flight Deck: {tick_filter}")
                    # Expanded to 7 Columns to fit "Orig Cost"
                    f1, f2, f3, f4, f5, f6, f7 = st.columns(7)
                    f1.metric("Current Price", f"${live_px:,.2f}")
                    f2.metric("Orig Cost", f"${orig_avg_cost:,.2f}", help="Avg Cost of Initial Buys (Trx 'B')")
                    f3.metric("Avg Cost", f"${avg_cost:,.2f}", help="Current Cost Basis of Held Shares")
                    f4.metric("Shares Held", f"{int(shares):,}")
                    f5.metric("Unrealized P&L", f"${unrealized:,.2f}", f"{return_pct:.2f}%")
                    f6.metric("Realized P&L", f"${fd_realized_pl:,.2f}", delta_color="normal")
                    f7.metric("Total Equity", f"${mkt_val:,.2f}", f"{pos_size_pct:.1f}% Size")
                    st.markdown("---")

                # --- 4. FINALIZE TABLE DISPLAY ---
                def calc_unrealized(row): 
                     if row['Action'] == 'BUY' and row['Remaining_Shares'] > 0:
                         price = live_px if tick_filter != "All" else curr_prices.get(row['Trade_ID'], 0)
                         entry = float(row.get('Amount', row.get('Price', 0.0)))
                         return (price - entry) * row['Remaining_Shares']
                     return 0.0
                display_df['Unrealized_PL'] = display_df.apply(calc_unrealized, axis=1)

                def calc_return_pct(row):
                    if row['Action'] == 'BUY' and row['Remaining_Shares'] > 0:
                         price = live_px if tick_filter != "All" else curr_prices.get(row['Trade_ID'], 0)
                         entry = float(row.get('Amount', row.get('Price', 0.0)))
                         if entry > 0: return ((price - entry) / entry) * 100
                    return 0.0
                display_df['Return_Pct'] = display_df.apply(calc_return_pct, axis=1)

                start_map = df_s.set_index('Trade_ID')['Open_Date'].to_dict()
                display_df['Campaign_Start'] = display_df['Trade_ID'].map(start_map)

                display_df['Shares'] = display_df.apply(lambda x: -x['Shares'] if x['Action'] == 'SELL' else x['Shares'], axis=1)
                
                if 'Value' not in display_df.columns and 'Amount' in display_df.columns:
                    display_df['Value'] = display_df['Shares'].abs() * display_df['Amount']
                
                display_df['Value'] = display_df.apply(lambda x: -x['Value'] if x['Action'] == 'SELL' else x['Value'], axis=1)
                
                # --- APPLY STATUS FILTER ---
                if status_filter != "All":
                    display_df = display_df[display_df['Status'] == status_filter]
                
                final_cols = ['Trade_ID', 'Trx_ID', 'Campaign_Start', 'Date', 'Ticker', 'Action', 'Status', 'Shares', 'Remaining_Shares', 'Amount', 'Stop_Loss', 'Value', 'Realized_PL', 'Unrealized_PL', 'Return_Pct', 'Rule', 'Notes']
                show_cols = [c for c in final_cols if c in display_df.columns]
                
                st.dataframe(
                    display_df[show_cols].sort_values(['Trade_ID', 'Date']).style
                    .format({
                        'Date': lambda x: x.strftime('%Y-%m-%d %H:%M') if isinstance(x, (pd.Timestamp, datetime)) else 'None',
                        'Campaign_Start': lambda x: x if isinstance(x, str) else (x.strftime('%Y-%m-%d %H:%M') if isinstance(x, (pd.Timestamp, datetime)) else 'None'), 
                        'Amount':'${:,.2f}', 'Stop_Loss':'${:,.2f}', 'Value':'${:,.2f}', 
                        'Realized_PL':'${:,.2f}', 'Unrealized_PL':'${:,.2f}', 
                        'Return_Pct':'{:.2f}%', 'Remaining_Shares':'{:.0f}'
                    })
                    .applymap(color_pnl, subset=['Value','Realized_PL','Unrealized_PL', 'Return_Pct'])
                    .applymap(color_neg_value, subset=['Shares']), 
                    height=(len(display_df) + 1) * 35 + 3, 
                    use_container_width=True
                )
            else: st.info("No open transactions found.")
        else: st.info("No data available.")

# --- TAB 9: DETAILED TRADE LOG (FINAL: LIFO + TV + TRX_ID) ---
    with tab9:
        st.subheader("🕵️ Campaign Inspector (Post-Mortem)")
        
        # 0. ENSURE JOURNAL IS LOADED
        p_clean = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')
        p_legacy = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal.csv')
        path_j = p_clean if os.path.exists(p_clean) else p_legacy
        df_j_hist = pd.DataFrame()
        if os.path.exists(path_j):
            try:
                df_j_hist = pd.read_csv(path_j)
                df_j_hist['Day'] = pd.to_datetime(df_j_hist['Day'], errors='coerce')
                df_j_hist = df_j_hist.sort_values('Day', ascending=False)
                
                def clean_nlv_val(x):
                    try: return float(str(x).replace('$', '').replace(',', '').strip())
                    except: return 0.0
                if 'End NLV' in df_j_hist.columns:
                    df_j_hist['End NLV'] = df_j_hist['End NLV'].apply(clean_nlv_val)
            except: pass

        # 1. TWO-STAGE FILTER
        all_tickers = sorted(df_d['Ticker'].dropna().unique().tolist())
        
        c_filt1, c_filt2 = st.columns(2)
        sel_tick = c_filt1.selectbox("1. Select Ticker", ["All"] + all_tickers)
        
        view_df = pd.DataFrame()
        sel_id = None
        
        if sel_tick != "All":
            subset_d = df_d[df_d['Ticker'] == sel_tick]
            subset_s = df_s[df_s['Ticker'] == sel_tick]
            
            trade_ids = sorted(subset_d['Trade_ID'].unique().tolist(), reverse=True)
            sel_id = c_filt2.selectbox("2. Select Campaign ID", trade_ids)
            
            if sel_id:
                # Filter specifically for this ID
                camp_txs = subset_d[subset_d['Trade_ID'] == sel_id].sort_values('Date')
                
                # --- A. RUN LIFO ENGINE FIRST (TO GET TRUE P&L) ---
                calc_df = camp_txs.copy().reset_index()
                buy_attribution = {} 
                inventory = [] 
                
                for idx, row in calc_df.iterrows():
                    if row['Action'] == 'BUY':
                        inventory.append({'idx': idx, 'price': row['Amount'], 'qty': row['Shares']})
                        buy_attribution[idx] = {'pl': 0.0, 'sold_cost': 0.0, 'sold_val': 0.0}
                    elif row['Action'] == 'SELL':
                        to_sell = row['Shares']
                        sell_price = row['Amount']
                        while to_sell > 0 and inventory:
                            last = inventory.pop()
                            take = min(to_sell, last['qty'])
                            seg_cost = take * last['price']
                            seg_rev = take * sell_price
                            seg_pl = seg_rev - seg_cost
                            buy_attribution[last['idx']]['pl'] += seg_pl
                            buy_attribution[last['idx']]['sold_cost'] += seg_cost
                            buy_attribution[last['idx']]['sold_val'] += seg_rev
                            last['qty'] -= take
                            to_sell -= take
                            if last['qty'] > 0.0001: inventory.append(last)

                def get_lifo_pl(idx, action, original_pl):
                    if action == 'SELL': return original_pl 
                    if idx in buy_attribution: return buy_attribution[idx]['pl']
                    return 0.0

                def get_lifo_ret(idx, action):
                    if action == 'BUY' and idx in buy_attribution:
                        data = buy_attribution[idx]
                        if data['sold_cost'] > 0:
                            return ((data['sold_val'] - data['sold_cost']) / data['sold_cost']) * 100
                    return 0.0

                calc_df['Lot P&L'] = calc_df.apply(lambda x: get_lifo_pl(x.name, x['Action'], x['Realized_PL']), axis=1)
                calc_df['Return %'] = calc_df.apply(lambda x: get_lifo_ret(x.name, x['Action']), axis=1)
                
                # --- B. CALCULATE METRICS ---
                realized_pl = calc_df[calc_df['Action'] == 'BUY']['Lot P&L'].sum()
                
                camp_sum = subset_s[subset_s['Trade_ID'] == sel_id].iloc[0] if not subset_s.empty else pd.Series()
                start_date = pd.to_datetime(calc_df['Date'].iloc[0])
                last_date = pd.to_datetime(calc_df['Date'].iloc[-1])
                
                is_closed = False
                if not camp_sum.empty and camp_sum['Status'] == 'CLOSED':
                    is_closed = True
                    if pd.notnull(camp_sum['Closed_Date']):
                        last_date = pd.to_datetime(camp_sum['Closed_Date'])
                else: last_date = datetime.now()
                
                days_held = (last_date - start_date).days
                if days_held < 1: days_held = 1

                # Risk Budget
                risk_budget = camp_sum.get('Risk_Budget', 0.0)
                risk_source = "Locked"
                if risk_budget <= 0:
                    risk_source = "Est. (0.5% NLV)"
                    if not df_j_hist.empty:
                        prior_days = df_j_hist[df_j_hist['Day'] < start_date]
                        if not prior_days.empty:
                            risk_budget = prior_days.iloc[0]['End NLV'] * 0.005
                        else: risk_budget = 500.0
                    else: risk_budget = 500.0
                
                r_str = "N/A"; r_color = "off"
                if risk_budget > 0:
                    r_multiple = realized_pl / risk_budget
                    r_str = f"{r_multiple:+.2f}R"
                    r_color = "normal" if r_multiple > 0 else "inverse"

                # Efficiency
                mfe_str = "N/A"
                try:
                    chart_start = start_date - timedelta(days=5)
                    chart_end = last_date + timedelta(days=5)
                    chart_data = yf.Ticker(sel_tick).history(start=chart_start, end=chart_end)
                    
                    if not chart_data.empty:
                        hold_mask = (chart_data.index >= start_date.tz_localize(chart_data.index.tz)) & (chart_data.index <= last_date.tz_localize(chart_data.index.tz))
                        if any(hold_mask):
                             period_high = chart_data.loc[hold_mask]['High'].max()
                             sells = calc_df[calc_df['Action'] == 'SELL']
                             if not sells.empty:
                                avg_exit = (sells['Amount'] * sells['Shares']).sum() / sells['Shares'].sum()
                                efficiency = (avg_exit / period_high) * 100
                                mfe_str = f"{efficiency:.1f}% (High: ${period_high:.2f})"
                             elif is_closed: mfe_str = "0% (Stopped Out?)"
                except: pass

                # --- C. DISPLAY FLIGHT DECK ---
                st.markdown(f"### 🚁 Flight Deck: {sel_tick} ({sel_id})")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Realized P&L", f"${realized_pl:+,.2f}", f"{days_held} Days Held")
                m2.metric("R-Multiple", r_str, f"Risk Base: ${risk_budget:,.0f} ({risk_source})", delta_color=r_color)
                
                buys = calc_df[calc_df['Action'] == 'BUY']
                avg_in = (buys['Amount'] * buys['Shares']).sum() / buys['Shares'].sum() if not buys.empty else 0
                m3.metric("Avg Entry Price", f"${avg_in:.2f}")
                m4.metric("Exit Efficiency", mfe_str, "vs Period High")
                
                # --- D. TRADINGVIEW EMBED (THE BATTLEFIELD) ---
                st.markdown("### 🗺️ The Battlefield (TradingView)")
                tv_widget_code = f"""
                <div class="tradingview-widget-container">
                  <div id="tradingview_chart"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                  <script type="text/javascript">
                  new TradingView.widget(
                  {{
                    "width": "100%",
                    "height": 500,
                    "symbol": "{sel_tick}",
                    "interval": "D",
                    "timezone": "America/New_York",
                    "theme": "light",
                    "style": "1",
                    "locale": "en",
                    "toolbar_bg": "#f1f3f6",
                    "enable_publishing": false,
                    "hide_side_toolbar": false,
                    "allow_symbol_change": true,
                    "container_id": "tradingview_chart"
                  }}
                  );
                  </script>
                </div>
                """
                st.components.v1.html(tv_widget_code, height=500)
                
                # DEEP LINK
                tv_link = f"https://www.tradingview.com/chart/?symbol={sel_tick}"
                st.link_button(f"🚀 Analyze {sel_tick} on TradingView (Premium)", tv_link)
                st.markdown("---")

                # --- E. NARRATIVE ---
                n1, n2 = st.columns(2)
                with n1:
                    st.info(f"**📝 Buy Rationale:**\n{camp_sum.get('Buy_Notes', 'No notes.')}")
                    st.caption(f"Strategy: {camp_sum.get('Rule', 'N/A')}")
                with n2:
                    sell_note = camp_sum.get('Sell_Notes', '')
                    sell_rule = camp_sum.get('Sell_Rule', '')
                    if not sell_note and not sell_rule:
                        st.warning("**No Exit Plan/Notes Logged**")
                    else:
                        st.error(f"**👋 Exit Context:**\n{sell_note}")
                        st.caption(f"Exit Rule: {sell_rule}")

                # --- F. RENDER TABLE ---
                st.markdown("#### 📜 Transaction History (LIFO Attribution)")
                display_df = calc_df.copy()
                
                # Visuals
                display_df['Shares'] = display_df.apply(lambda x: -x['Shares'] if x['Action'] == 'SELL' else x['Shares'], axis=1)
                display_df['Value'] = display_df.apply(lambda x: -x['Value'] if x['Action'] == 'SELL' else x['Value'], axis=1)
                
                # ADDED 'Trx_ID' back to the list
                cols = ['Trade_ID', 'Trx_ID', 'Date', 'Ticker', 'Action', 'Shares', 'Amount', 'Value', 'Lot P&L', 'Return %', 'Rule', 'Notes']
                show_cols = [c for c in cols if c in display_df.columns]
                
                st.dataframe(
                    display_df[show_cols].sort_values(['Trade_ID', 'Date']).style
                    .format({
                        'Date': lambda x: x.strftime('%Y-%m-%d %H:%M') if isinstance(x, (pd.Timestamp, datetime)) else 'None', 
                        'Shares':'{:.0f}', 'Amount':'${:,.2f}', 'Value':'${:,.2f}', 
                        'Lot P&L':'${:,.2f}', 'Return %':'{:.2f}%'
                    })
                    .applymap(color_pnl, subset=['Lot P&L', 'Return %'])
                    .applymap(color_neg_value, subset=['Shares']),
                    use_container_width=True
                )
        else:
            view_df = df_d.copy()
            view_df['Lot P&L'] = view_df['Realized_PL']
            view_df['Return %'] = 0.0
            st.markdown("### 🗄️ Master Transaction Log")
            st.dataframe(view_df.sort_values(['Date'], ascending=False), use_container_width=True)

# --- TAB CY: CURRENT YEAR CAMPAIGNS (2026 + ROLLOVERS) ---
    with tab_cy:
        st.subheader("CY 2026 Campaigns (Risk & Performance)")
        st.caption("Showing 2026 trades + Rollovers. Auditing both Risk Discipline and Financial Performance.")

        if not df_s.empty:
            # --- 1. FILTER LOGIC ---
            df_s['Open_DT'] = pd.to_datetime(df_s['Open_Date'], errors='coerce')
            df_s['Close_DT'] = pd.to_datetime(df_s['Closed_Date'], errors='coerce')
            cutoff_date = pd.Timestamp("2026-01-01")
            
            cy_mask = (
                (df_s['Open_DT'] >= cutoff_date) | 
                (df_s['Status'] == 'OPEN') | 
                (df_s['Close_DT'] >= cutoff_date)
            )
            df_cy = df_s[cy_mask].copy()
            
            if not df_cy.empty:
                # --- 2. CALCULATE METRICS (Combined Engine) ---
                df_cy = df_cy.reset_index().rename(columns={'index': 'Seq_ID'})
                
                # Ensure Risk_Budget exists
                if 'Risk_Budget' not in df_cy.columns: df_cy['Risk_Budget'] = 0.0
                df_cy['Risk_Budget'] = df_cy['Risk_Budget'].fillna(0.0).astype(float)
                
                def calc_row_metrics(row):
                    # P&L Logic
                    pl = row['Realized_PL'] if row['Status'] == 'CLOSED' else row.get('Unrealized_PL', 0.0)
                    
                    # Risk Logic
                    budget = row['Risk_Budget']
                    if budget > 0:
                        r_mult = pl / budget
                    else:
                        r_mult = 0.0
                    
                    # Compliance Logic (Losses Only)
                    compliance = "N/A"
                    if pl >= 0:
                        compliance = "✅ WIN"
                    else:
                        if budget > 0:
                            loss_ratio = abs(pl) / budget
                            if loss_ratio <= 1.1: compliance = "✅ OK"      
                            elif loss_ratio <= 1.5: compliance = "⚠️ SLIP"  
                            else: compliance = "🛑 BREACH"                  
                        else:
                            compliance = "⚪ NO BUDGET"
                    
                    return pd.Series([pl, r_mult, compliance])

                df_cy[['Active_PL', 'R_Multiple', 'Compliance']] = df_cy.apply(calc_row_metrics, axis=1)

                # --- 3. FILTERS ---
                c_f1, c_f2 = st.columns(2)
                unique_tickers_cy = sorted(df_cy['Ticker'].dropna().astype(str).unique().tolist())
                tick_filter_cy = c_f1.selectbox("Filter Ticker (CY)", ["All"] + unique_tickers_cy, key="cy_tick")
                
                # Combined Filter: Status and Compliance
                filter_options = ["OPEN", "CLOSED", "✅ WIN", "✅ OK", "⚠️ SLIP", "🛑 BREACH"]
                active_filters = c_f2.multiselect("Filter", filter_options, key="cy_combined_filter")
                
                view_cy = df_cy.copy()
                if tick_filter_cy != "All": 
                    view_cy = view_cy[view_cy['Ticker'] == tick_filter_cy]
                
                if active_filters:
                    # Logic to handle both Status and Compliance strings in one filter
                    status_filters = [f for f in active_filters if f in ["OPEN", "CLOSED"]]
                    comp_filters = [f for f in active_filters if f not in ["OPEN", "CLOSED"]]
                    
                    if status_filters:
                        view_cy = view_cy[view_cy['Status'].isin(status_filters)]
                    if comp_filters:
                        view_cy = view_cy[view_cy['Compliance'].isin(comp_filters)]
                
                if not view_cy.empty:
                    # --- 4. FLIGHT DECK (RESTORED & EXPANDED) ---
                    closed_cy = view_cy[view_cy['Status'] == 'CLOSED']
                    
                    # Defaults
                    net_pl = view_cy['Active_PL'].sum() # Active P&L (Open + Closed)
                    win_rate = 0.0; expectancy = 0.0
                    gross_profit = 0.0; gross_loss = 0.0
                    avg_win = 0.0; avg_loss = 0.0
                    avg_r_loss = 0.0; discipline_score = 0.0
                    wl_ratio = 0.0; num_wins = 0; num_losses = 0

                    if not closed_cy.empty:
                        # Financials
                        winners = closed_cy[closed_cy['Active_PL'] > 0]
                        losers = closed_cy[closed_cy['Active_PL'] <= 0]
                        
                        gross_profit = winners['Active_PL'].sum()
                        gross_loss = abs(losers['Active_PL'].sum())
                        
                        num_wins = len(winners)
                        num_losses = len(losers)
                        total_closed = len(closed_cy)
                        
                        win_rate = (num_wins / total_closed) * 100 if total_closed > 0 else 0.0
                        avg_win = gross_profit / num_wins if num_wins > 0 else 0.0
                        avg_loss = gross_loss / num_losses if num_losses > 0 else 0.0
                        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
                        
                        # Expectancy
                        win_pct_dec = win_rate / 100
                        loss_pct_dec = 1 - win_pct_dec
                        expectancy = (win_pct_dec * avg_win) - (loss_pct_dec * avg_loss)
                        
                        # Risk Auditing
                        avg_r_loss = losers['R_Multiple'].mean() if not losers.empty else 0.0
                        if not losers.empty:
                            compliant_losers = losers[losers['Compliance'] == '✅ OK']
                            discipline_score = (len(compliant_losers) / len(losers)) * 100
                        else: discipline_score = 100.0

                    # RENDER SCOREBOARD (MATCHING TAB 10 STYLE)
                    st.markdown("### 🚁 Flight Deck: Performance & Risk")
                    
                    # Row 1: The Bottom Line (Added Deltas)
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Net P&L (CY)", f"${net_pl:,.2f}", f"{len(view_cy)} Total Campaigns")
                    m2.metric("Win Rate", f"{win_rate:.1f}%", f"{num_wins}W - {num_losses}L")
                    m3.metric("Expectancy", f"${expectancy:,.2f}", "Avg value per trade")
                    
                    # Risk Deltas (Custom for Risk)
                    disc_delta = "Perfect" if discipline_score == 100 else ("Needs Work" if discipline_score < 80 else "Solid")
                    disc_col = "normal" if discipline_score >= 90 else "inverse"
                    m4.metric("Risk Compliance", f"{discipline_score:.0f}%", disc_delta, delta_color=disc_col)
                    
                    loss_col = "normal" if avg_r_loss > -1.2 else "inverse"
                    m5.metric("Avg R-Loss", f"{avg_r_loss:.2f}R", "Target: > -1.0R", delta_color=loss_col)
                    
                    st.markdown("---")
                    
                    # Row 2: Dollar Stats (Added Deltas)
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Gross Profit", f"${gross_profit:,.2f}", delta_color="normal")
                    k2.metric("Gross Loss", f"-${gross_loss:,.2f}", delta_color="inverse")
                    k3.metric("Avg Win", f"${avg_win:,.2f}", delta_color="normal")
                    k4.metric("Avg Loss", f"-${avg_loss:,.2f}", f"W/L Ratio: {wl_ratio:.2f}")

                    st.markdown("---")

                    # --- 5. DATA TABLE (REORDERED & COLORED) ---
                    def calc_days_open(row):
                        try:
                            start = row['Open_DT']
                            end = row['Close_DT'] if row['Status'] == 'CLOSED' and pd.notna(row['Close_DT']) else datetime.now()
                            return (end - start).days
                        except: return 0
                    view_cy['Days_Open'] = view_cy.apply(calc_days_open, axis=1)

                    # NEW COLUMN ORDER
                    target_cols = [
                        'Seq_ID', 'Trade_ID', 'Ticker', 'Status', 
                        'Open_Date', 'Closed_Date', 'Days_Open', 
                        'Total_Cost', 'Avg_Entry', 'Avg_Exit', 
                        'Risk_Budget', 'Active_PL', 'R_Multiple', 
                        'Compliance', 'Rule', 'Buy_Notes', 'Sell_Notes'
                    ]
                    # Ensure cols exist
                    for c in ['Avg_Exit', 'Buy_Notes', 'Sell_Notes']:
                        if c not in view_cy.columns: view_cy[c] = ""
                            
                    valid_cols = [c for c in target_cols if c in view_cy.columns]
                    
                    view_cy = view_cy.sort_values('Open_DT', ascending=False)
                    
                    # --- STYLING FUNCTIONS ---
                    def style_status(val): 
                        if val == 'CLOSED': return 'color: #ff4b4b; font-weight: bold' # Red
                        return 'color: #2ca02c; font-weight: bold' # Green
                        
                    def style_pl(val):
                        if val > 0: return 'color: #2ca02c'
                        if val < 0: return 'color: #ff4b4b'
                        return ''
                        
                    def style_compliance(val):
                        if 'BREACH' in str(val): return 'color: white; background-color: #ff4b4b; font-weight: bold' 
                        if 'SLIP' in str(val): return 'color: #ff4b4b; font-weight: bold' 
                        if 'WIN' in str(val): return 'color: #2ca02c; font-weight: bold'
                        return ''
                    
                    def style_r(val):
                        if val > 1.0: return 'color: #2ca02c; font-weight: bold'
                        if val < -1.2: return 'color: #ff4b4b; font-weight: bold'
                        return ''

                    st.dataframe(
                        view_cy[valid_cols].style.format({
                            'Open_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else '',
                            'Closed_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else '',
                            'Total_Cost':'${:,.2f}', 'Avg_Entry':'${:,.2f}', 'Avg_Exit':'${:,.2f}',
                            'Risk_Budget':'${:,.0f}', 'Active_PL':'${:+,.2f}', 'R_Multiple':'{:+.2f}R'
                        })
                        .applymap(style_status, subset=['Status'])
                        .applymap(style_pl, subset=['Active_PL'])
                        .applymap(style_compliance, subset=['Compliance'])
                        .applymap(style_r, subset=['R_Multiple']),
                        hide_index=True,
                        use_container_width=True
                    )
                else: st.info("No trades match your filters.")
            else: st.info("No trades found for 2026 or Rollovers.")
        else: st.warning("Summary Database is empty.")

# --- TAB 10: ALL CAMPAIGNS (PRO SCOREBOARD) ---
    with tab10:
        st.subheader("All Campaigns (Summary)")

        # --- IMAGE VIEWER ---
        if R2_AVAILABLE and USE_DATABASE:
            with st.expander("📸 View Trade Charts"):
                st.caption("Select a trade to view uploaded chart images")

                # Get list of trades that have images
                trade_options = df_s['Trade_ID'].unique().tolist()
                selected_trade = st.selectbox("Select Trade ID", ["Select..."] + trade_options, key='img_viewer_trade')

                if selected_trade and selected_trade != "Select...":
                    # Get images for this trade
                    images = db.get_trade_images(CURR_PORT_NAME, selected_trade)

                    if images:
                        # Get ticker for display
                        ticker_row = df_s[df_s['Trade_ID'] == selected_trade]
                        ticker = ticker_row['Ticker'].iloc[0] if not ticker_row.empty else "Unknown"

                        st.markdown(f"### {ticker} - {selected_trade}")

                        # Display images in columns
                        image_types = {img['image_type']: img for img in images}

                        # Create columns for weekly, daily, exit
                        cols = st.columns(3)

                        for idx, (img_type, col) in enumerate(zip(['weekly', 'daily', 'exit'], cols)):
                            with col:
                                if img_type in image_types:
                                    img_data = image_types[img_type]
                                    st.markdown(f"**{img_type.title()} Chart**")

                                    # Download and display image
                                    image_bytes = r2.download_image(img_data['image_url'])
                                    if image_bytes:
                                        st.image(image_bytes, use_container_width=True)
                                        st.caption(f"Uploaded: {img_data['uploaded_at']}")
                                    else:
                                        st.warning(f"Failed to load {img_type} chart")
                                else:
                                    st.info(f"No {img_type} chart")
                    else:
                        st.info("No charts uploaded for this trade")

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
                    'Shares': lambda x: f'{x:.0f}' if pd.notnull(x) else '-',
                    'Avg_Entry': lambda x: f'${x:,.2f}' if pd.notnull(x) else '-',
                    'Avg_Exit': lambda x: f'${x:,.2f}' if pd.notnull(x) else '-',
                    'Total_Cost': lambda x: f'${x:,.2f}' if pd.notnull(x) else '-',
                    'Realized_PL': lambda x: f'${x:,.2f}' if pd.notnull(x) else '-',
                    'Return_Pct': lambda x: f'{x:.2f}%' if pd.notnull(x) else '-'
                }).applymap(highlight_status, subset=['Status']),
                use_container_width=True
            )
        else:
            st.info("No closed trades found for this period.")




# ====================================================================
# ACTIVE CAMPAIGN SUMMARY
# ====================================================================
elif page == "Active Campaign Summary":
    st.subheader("Active Campaign Summary")

    # Load data
    if not os.path.exists(DETAILS_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Amount','Value','Rule','Notes','Realized_PL','Stop_Loss','Trx_ID']).to_csv(DETAILS_FILE, index=False)
    if not os.path.exists(SUMMARY_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Total_Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Rule','Notes','Buy_Notes','Sell_Rule','Sell_Notes']).to_csv(SUMMARY_FILE, index=False)

    df_d = load_data(DETAILS_FILE)
    df_s = load_data(SUMMARY_FILE)

    # --- IMAGE VIEWER FOR ACTIVE TRADES ---

    if R2_AVAILABLE and USE_DATABASE and not df_s.empty:

        df_open = df_s[df_s['Status'] == 'OPEN'].copy()

        if not df_open.empty:

            with st.expander("📸 View Entry Charts (Active Trades)"):

                st.caption("View weekly and daily charts for your active positions")



                # Get list of open trades

                open_trades = df_open[['Trade_ID', 'Ticker']].values.tolist()

                trade_opts = [f"{ticker} | {trade_id}" for trade_id, ticker in open_trades]



                selected = st.selectbox("Select Trade", ["Select..."] + trade_opts, key='active_img_viewer')



                if selected and selected != "Select...":

                    ticker, trade_id = selected.split(" | ")

                    images = db.get_trade_images(CURR_PORT_NAME, trade_id)



                    if images:

                        st.markdown(f"### {ticker} - {trade_id}")



                        image_types = {img['image_type']: img for img in images}



                        # Show weekly and daily (no exit for active trades)

                        col1, col2 = st.columns(2)



                        with col1:

                            if 'weekly' in image_types:

                                img_data = image_types['weekly']

                                st.markdown("**Weekly Chart**")

                                image_bytes = r2.download_image(img_data['image_url'])

                                if image_bytes:

                                    st.image(image_bytes, use_container_width=True)

                                    st.caption(f"Uploaded: {img_data['uploaded_at']}")

                                else:

                                    st.warning("Failed to load weekly chart")

                            else:

                                st.info("No weekly chart")



                        with col2:

                            if 'daily' in image_types:

                                img_data = image_types['daily']

                                st.markdown("**Daily Chart**")

                                image_bytes = r2.download_image(img_data['image_url'])

                                if image_bytes:

                                    st.image(image_bytes, use_container_width=True)

                                    st.caption(f"Uploaded: {img_data['uploaded_at']}")

                                else:

                                    st.warning("Failed to load daily chart")

                            else:

                                st.info("No daily chart")

                    else:

                        st.info("No charts uploaded for this trade")



    # --- INIT SESSION STATE ---

    if 'live_prices' not in st.session_state:

        st.session_state['live_prices'] = {}

        st.session_state['last_update'] = None



    if not df_s.empty:

         df_open = df_s[df_s['Status'] == 'OPEN'].copy()

         

         if not df_open.empty:

             

             # --- 0. ON-DEMAND REFRESH (FAST MODE) ---

             c_btn, c_info = st.columns([1, 3])

             

             if c_btn.button("🔄 Refresh Live Prices"):

                 tickers = df_open['Ticker'].unique().tolist()

                 if tickers:

                     with st.spinner("Fetching current prices..."):

                         try:

                             # 1. Download only 1 day (Much faster, less data)

                             data = yf.download(tickers, period="1d", progress=False)['Close']

                             

                             new_prices = {}

                             

                             if len(tickers) == 1:

                                 val = data.iloc[-1]

                                 if hasattr(val, 'iloc'): val = val.iloc[0]

                                 new_prices[tickers[0]] = float(val)

                             else:

                                 # Check valid columns

                                 valid_cols = [t for t in tickers if t in data.columns]

                                 last_row = data.iloc[-1]

                                 

                                 for t in valid_cols:

                                     try:

                                         val = last_row[t]

                                         new_prices[t] = float(val)

                                     except: pass

                             

                             # Update Session

                             st.session_state['live_prices'] = new_prices

                             st.session_state['last_update'] = datetime.now().strftime("%H:%M:%S")

                             st.success("Prices Updated!")

                             

                         except Exception as e:

                             st.warning(f"Could not fetch prices. Using saved values.")



             # Show status

             if st.session_state['last_update']:

                 c_info.caption(f"Last Update: {st.session_state['last_update']}")

             else:

                 c_info.info("Using saved data. Click Refresh to update.")



             # --- 1. LIFO ENGINE (UNCHANGED) ---

             def run_lifo_engine(row):

                 tid = row['Trade_ID']

                 

                 def force_float(x):

                     try: return float(str(x).replace('$','').replace(',',''))

                     except: return 0.0

                 

                 summary_stop = force_float(row.get('Stop_Loss', 0))

                 summary_entry = force_float(row.get('Avg_Entry', 0))

                 shares = force_float(row.get('Shares', 0))



                 tid_str = str(tid).strip()

                 df_d['Trade_ID_Str'] = df_d['Trade_ID'].astype(str).str.strip()

                 subset = df_d[df_d['Trade_ID_Str'] == tid_str].copy()

                 

                 if subset.empty:

                     eff_stop = summary_stop if summary_stop > 0 else summary_entry

                     risk = max(0.0, (summary_entry - eff_stop) * shares)

                     proj = (eff_stop - summary_entry) * shares

                     return risk, eff_stop, summary_entry, proj, 0.0



                 subset['Type_Rank'] = subset['Action'].apply(lambda x: 0 if str(x).upper() == 'BUY' else 1)

                 if 'Date' in subset.columns:

                     subset['Date'] = pd.to_datetime(subset['Date'], errors='coerce')

                     subset = subset.sort_values(['Date', 'Type_Rank'])

                 

                 inventory = [] 

                 realized_bank = 0.0 

                 

                 for _, tx in subset.iterrows():

                     action = str(tx.get('Action', '')).upper()

                     tx_shares = abs(force_float(tx.get('Shares', 0)))

                     

                     if action == 'BUY':

                         price = force_float(tx.get('Amount', tx.get('Price', 0.0)))

                         if price == 0: price = summary_entry

                         stop = force_float(tx.get('Stop_Loss', tx.get('Stop', 0.0)))

                         if stop == 0: stop = price 

                         inventory.append({'qty': tx_shares, 'price': price, 'stop': stop})

                         

                     elif action == 'SELL':

                         to_sell = tx_shares

                         sell_price = force_float(tx.get('Amount', tx.get('Price', 0.0)))

                         cost_basis_accum = 0.0

                         sold_qty_accum = 0.0

                         

                         while to_sell > 0 and inventory:

                             last = inventory[-1] 

                             take = min(to_sell, last['qty'])

                             cost_basis_accum += (take * last['price'])

                             sold_qty_accum += take

                             last['qty'] -= take

                             to_sell -= take

                             if last['qty'] < 0.00001: inventory.pop()

                         

                         revenue = sold_qty_accum * sell_price

                         realized_bank += (revenue - cost_basis_accum)



                 inventory_proj_pl = 0.0

                 total_open_shares = 0.0

                 weighted_cost = 0.0

                 weighted_stop = 0.0 

                 

                 for item in inventory:

                     qty = item['qty']

                     price = item['price']

                     stop = item['stop']

                     if qty > 0:

                         total_open_shares += qty

                         weighted_cost += (qty * price)

                         weighted_stop += (qty * stop)

                         inventory_proj_pl += (stop - price) * qty

                 

                 avg_cost = (weighted_cost / total_open_shares) if total_open_shares > 0 else summary_entry

                 avg_log_stop = (weighted_stop / total_open_shares) if total_open_shares > 0 else 0.0



                 master_stop = summary_stop if summary_stop > 0 else (avg_log_stop if avg_log_stop > 0 else avg_cost)

                 initial_risk = max(0.0, (avg_cost - master_stop) * total_open_shares)

                 final_projected_floor = inventory_proj_pl + realized_bank

                 

                 return initial_risk, master_stop, avg_cost, final_projected_floor, realized_bank



             metrics = df_open.apply(run_lifo_engine, axis=1, result_type='expand')

             

             df_open['Risk $'] = metrics[0]

             df_open['Avg Stop'] = metrics[1]

             df_open['Avg_Entry'] = metrics[2]

             df_open['Projected P&L'] = metrics[3]

             df_open['Realized Bank'] = metrics[4]

             

             # --- 2. UPDATE FINANCIALS ---

             def get_live_price(row):

                 if row['Ticker'] in st.session_state['live_prices']: 

                     return st.session_state['live_prices'][row['Ticker']]

                 try: return float(row.get('Current_Price', 0)) if float(row.get('Current_Price', 0)) > 0 else float(row['Avg_Entry'])

                 except: return float(row['Avg_Entry'])



             df_open['Current Price'] = df_open.apply(get_live_price, axis=1)

             df_open['Current Value'] = df_open['Shares'] * df_open['Current Price']

             df_open['Unrealized_PL'] = (df_open['Current Price'] - df_open['Avg_Entry']) * df_open['Shares']

             

             df_open['Return_Pct'] = df_open.apply(

                 lambda x: ((x['Current Price'] - x['Avg_Entry']) / x['Avg_Entry'] * 100) if x['Avg_Entry'] != 0 else 0.0, 

                 axis=1

             )



             df_open['Safe_Stop'] = df_open.apply(lambda x: x['Avg Stop'] if x['Avg Stop'] > 0 else x['Avg_Entry'], axis=1)

             df_open['Open Risk Equity'] = (df_open['Current Price'] - df_open['Safe_Stop']) * df_open['Shares']

             

             def get_days_held(val):

                 try:

                     tid_str = str(val).strip()

                     rows = df_d[df_d['Trade_ID_Str'] == tid_str]

                     if not rows.empty and 'Date' in rows:

                         dates = pd.to_datetime(rows['Date'], errors='coerce')

                         return (pd.Timestamp.now() - dates.min()).days

                     return 0

                 except: return 0

             

             df_open['Days Held'] = df_open['Trade_ID'].apply(get_days_held)



             # --- 3. NEW: RISK STATUS (REPLACED TREND) ---

             def get_risk_status(row):

                 # If Risk $ is 0, it means Stop >= Cost (or net financed)

                 if row['Risk $'] <= 0.01:

                     return "🆓 Free Roll"

                 else:

                     return "⚠️ At Risk"

                 

             df_open['Risk Status'] = df_open.apply(get_risk_status, axis=1)



             # Load equity from journal (database-aware)

             equity = 100000.0

             try:

                 j_df = load_data(JOURNAL_FILE)

                 if not j_df.empty and 'End NLV' in j_df.columns:

                     # Sort by date to get the latest entry

                     if 'Day' in j_df.columns:

                         j_df['Day'] = pd.to_datetime(j_df['Day'], errors='coerce')

                         j_df = j_df.dropna(subset=['Day']).sort_values('Day', ascending=False)

                     equity = float(str(j_df['End NLV'].iloc[0]).replace('$','').replace(',',''))

             except:

                 pass

             

             df_open['Risk %'] = (df_open['Risk $'] / equity) * 100

             df_open['Pos Size %'] = (df_open['Current Value'] / equity) * 100



             # --- 4. DISPLAY METRICS ---

             total_mkt = df_open['Current Value'].sum()

             total_unreal = df_open['Unrealized_PL'].sum()

             total_realized_bank = df_open['Realized Bank'].sum()

             total_initial_risk = df_open['Risk $'].sum() 

             total_open_risk_equity = df_open['Open Risk Equity'].sum() 

             

             live_exp = (total_mkt / equity) * 100

             

             m1, m2, m3, m4, m5, m6 = st.columns(6)

             m1.metric("Open Positions", len(df_open))

             m2.metric("Total Market Value", f"${total_mkt:,.2f}")

             m3.metric("Live Exposure", f"{live_exp:.1f}%", f"of ${equity:,.0f}")

             

             m4.metric(

                 "Total Unrealized P&L", 

                 f"${total_unreal:,.2f}", 

                 f"Realized: ${total_realized_bank:,.2f}", 

                 delta_color="off"

             )

             

             ir_pct = (total_initial_risk / equity) * 100

             m5.metric("Initial Risk", f"${total_initial_risk:,.2f}", f"{ir_pct:.2f}% of NLV", delta_color="off")

             

             or_pct = (total_open_risk_equity / equity) * 100

             m6.metric("Open Risk (Heat)", f"${total_open_risk_equity:,.2f}", f"{or_pct:.2f}% of NLV", delta_color="inverse")

             

             # --- 5. DATAFRAME ---

             if 'Return_Pct' in df_open.columns:

                 df_open = df_open.sort_values(by='Return_Pct', ascending=False)

             

             # UPDATED COLUMNS: Removed Trend Status, Added Risk Status

             cols = ['Trade_ID', 'Ticker', 'Days Held', 'Risk Status', 'Return_Pct', 'Pos Size %', 

                     'Shares', 'Avg_Entry', 'Current Price', 'Avg Stop', 'Risk_Budget', 

                     'Risk $', 'Risk %', 'Current Value', 'Unrealized_PL', 'Projected P&L']

             

             final_cols = [c for c in cols if c in df_open.columns]

             

             st.dataframe(

                 df_open[final_cols].style.format({

                     'Shares':'{:.0f}', 'Total_Cost':'${:,.2f}', 'Unrealized_PL':'${:,.2f}', 'Avg_Entry':'${:,.2f}', 

                     'Current Price':'${:,.2f}', 'Return_Pct':'{:.2f}%', 'Current Value': '${:,.2f}', 'Pos Size %': '{:.1f}%', 

                     'Avg Stop': '${:,.2f}', 'Risk $': '${:,.2f}', 'Risk %': '{:.2f}%', 'Risk_Budget': '${:,.2f}', 'Days Held': '{:.0f}',

                     'Projected P&L': '${:,.2f}'

                 }).applymap(color_pnl, subset=['Unrealized_PL', 'Return_Pct', 'Projected P&L']),

                 height=(len(df_open) + 1) * 35 + 3,

                 use_container_width=True

             )

             

             # --- 6. MONITOR ---

             st.markdown("---"); st.subheader("🛡️ Risk Monitor")

             all_clear = True

             for _, r in df_open.iterrows():

                 budget = 0.0

                 try: budget = float(str(r.get('Risk_Budget', 0)).replace('$','').replace(',',''))

                 except: pass

                 

                 if r['Risk $'] > (budget + 5):

                     st.warning(f"⚠️ **{r['Ticker']}**: Initial Risk (${r['Risk $']:.0f}) > Budget (${budget:.0f}).")

                     all_clear = False

                 

                 if r['Return_Pct'] <= -7.0:

                     st.error(f"🔴 **{r['Ticker']}**: Down {r['Return_Pct']:.2f}%. Violates Stop Rule.")

                     all_clear = False

             if all_clear: st.success("✅ System Health Good.")

             

         else: st.info("No open positions.")

    else: st.info("No data available.")



# --- TAB RISK: RISK MANAGER (FULL ENGINE CLONE) ---


# ====================================================================
# RISK MANAGER
# ====================================================================
elif page == "Risk Manager":
    import numpy as np
    import matplotlib.pyplot as plt

    RESET_DATE = pd.Timestamp("2026-02-24")

    st.subheader(f"Risk Manager ({CURR_PORT_NAME})")

    # Load data
    if not os.path.exists(DETAILS_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Amount','Value','Rule','Notes','Realized_PL','Stop_Loss','Trx_ID']).to_csv(DETAILS_FILE, index=False)
    if not os.path.exists(SUMMARY_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Total_Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Rule','Notes','Buy_Notes','Sell_Rule','Sell_Notes']).to_csv(SUMMARY_FILE, index=False)

    df_d = load_data(DETAILS_FILE)
    df_s = load_data(SUMMARY_FILE)


    # 1. LOAD JOURNAL (For Historical Chart)

    # Load journal data (database-aware)

    p_path = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')

    df_j = load_data(p_path)



    if not df_j.empty:

        

        # Clean Data

        if not df_j.empty and 'Day' in df_j.columns:

            df_j['Day'] = pd.to_datetime(df_j['Day'], errors='coerce')

            df_j.sort_values('Day', inplace=True) 

            

            def clean_num_rm(x):

                try:

                    if isinstance(x, str):

                        return float(x.replace('$', '').replace(',', '').replace('%', '').strip())

                    return float(x)

                except: return 0.0



            for c in ['End NLV', 'Beg NLV', 'Cash -/+']: 

                if c in df_j.columns: df_j[c] = df_j[c].apply(clean_num_rm)

            

            # Filter for Reset Date

            df_active = df_j[df_j['Day'] >= RESET_DATE].copy()

            

            if not df_active.empty:

                curr_nlv = df_active['End NLV'].iloc[-1]

                peak_nlv = df_active['End NLV'].max()

                

                # Drawdown Metrics

                dd_dol = peak_nlv - curr_nlv

                dd_pct = (dd_dol / peak_nlv) * 100 if peak_nlv > 0 else 0.0

                

                # Hard Decks (UPDATED TRIGGERS: 7.5%, 12.5%, 15%)

                deck_l1 = peak_nlv * 0.925  # -7.5%

                deck_l2 = peak_nlv * 0.875  # -12.5%

                deck_l3 = peak_nlv * 0.850  # -15.0%

                dist_l1 = curr_nlv - deck_l1



                # ==========================================================

                # 2. CALCULATE LIVE OPEN RISK (EXACT ENGINE CLONE)

                # ==========================================================

                current_open_risk = 0.0

                

                if not df_s.empty:

                    # A. Filter Open Positions

                    if 'Status' in df_s.columns:

                        df_s['Status_Clean'] = df_s['Status'].astype(str).str.strip().str.upper()

                        df_open = df_s[df_s['Status_Clean'] == 'OPEN'].copy()

                    else: df_open = pd.DataFrame()

                    

                    if not df_open.empty:

                        # B. Get Live Prices

                        tickers = df_open['Ticker'].unique().tolist()

                        live_prices = {}

                        if tickers:

                            try:

                                data = yf.download(tickers, period="1d", progress=False)['Close']

                                if len(tickers) == 1:

                                    val = data.iloc[-1]

                                    if isinstance(val, pd.Series): val = val.iloc[0]

                                    live_prices[tickers[0]] = float(val)

                                else:

                                    last_row = data.iloc[-1]

                                    for t in tickers:

                                        if t in last_row.index:

                                            live_prices[t] = float(last_row[t])

                            except: pass



                        # C. DEFINE THE ENGINE

                        def calculate_risk_exact(row):

                            tid = row['Trade_ID']

                            

                            # Helpers

                            def force_float(x):

                                try: return float(str(x).replace('$','').replace(',',''))

                                except: return 0.0

                            

                            summary_stop = force_float(row.get('Stop_Loss', 0))

                            summary_entry = force_float(row.get('Avg_Entry', 0))

                            shares = force_float(row.get('Shares', 0))

                            

                            ticker = row.get('Ticker')

                            current_price = live_prices.get(ticker, force_float(row.get('Current_Price', summary_entry)))

                            

                            tid_str = str(tid).strip()

                            df_d['Trade_ID_Str'] = df_d['Trade_ID'].astype(str).str.strip()

                            subset = df_d[df_d['Trade_ID_Str'] == tid_str].copy()

                            

                            avg_log_stop = 0.0

                            if not subset.empty:

                                subset['Type_Rank'] = subset['Action'].apply(lambda x: 0 if str(x).upper() == 'BUY' else 1)

                                if 'Date' in subset.columns:

                                    subset['Date'] = pd.to_datetime(subset['Date'], errors='coerce')

                                    subset = subset.sort_values(['Date', 'Type_Rank'])

                                

                                inventory = []

                                for _, tx in subset.iterrows():

                                    action = str(tx.get('Action', '')).upper()

                                    s = abs(force_float(tx.get('Shares', 0)))

                                    if action == 'BUY':

                                        pr = force_float(tx.get('Amount', tx.get('Price', 0.0)))

                                        if pr == 0: pr = summary_entry

                                        st = force_float(tx.get('Stop_Loss', tx.get('Stop', 0.0)))

                                        if st == 0: st = pr

                                        inventory.append({'qty': s, 'stop': st})

                                    elif action == 'SELL':

                                        sell_q = s

                                        while sell_q > 0 and inventory:

                                            last = inventory[-1]

                                            take = min(sell_q, last['qty'])

                                            last['qty'] -= take

                                            sell_q -= take

                                            if last['qty'] < 0.00001: inventory.pop()

                                

                                tot_q = 0.0

                                w_st = 0.0

                                for i in inventory:

                                    if i['qty'] > 0:

                                        tot_q += i['qty']

                                        w_st += (i['qty'] * i['stop'])

                                

                                if tot_q > 0:

                                    avg_log_stop = w_st / tot_q



                            master_stop = summary_stop if summary_stop > 0 else (avg_log_stop if avg_log_stop > 0 else summary_entry)

                            heat = max(0.0, (current_price - master_stop) * shares)

                            return heat



                        df_open['Calculated_Heat'] = df_open.apply(calculate_risk_exact, axis=1)

                        current_open_risk = df_open['Calculated_Heat'].sum()



                # --- 3. HEADS UP DISPLAY ---

                st.markdown(f"### Current Status: ${curr_nlv:,.2f}")

                

                col1, col2, col3 = st.columns(3)

                col1.metric("Current Peak (HWM)", f"${peak_nlv:,.2f}")

                col1.caption(f"Since {RESET_DATE.strftime('%m/%d/%y')}")

                

                col2.metric("Current Drawdown", f"-{dd_pct:.2f}%", f"-${dd_dol:,.2f}", delta_color="inverse")

                

                # UPDATED STATUS LOGIC

                status_txt = "🟢 ALL CLEAR"

                if dd_pct >= 15.0: status_txt = "☠️ GO TO CASH"

                elif dd_pct >= 12.5: status_txt = "🟠 MAX 30% INVESTED"

                elif dd_pct >= 7.5: status_txt = "🟡 REMOVE MARGIN"

                

                col3.metric("Required Action", status_txt)

                

                # Stop Out Floor

                stop_out_floor_val = curr_nlv - current_open_risk

                

                if dd_pct < 7.5:

                    col3.caption(f"Buffer: ${dist_l1:,.0f} to Level 1")

                

                col3.caption(f"Total Open Risk: -${current_open_risk:,.2f}")



                st.markdown("---")



                # --- 4. VISUALIZATION (UPDATED: HORIZONTAL LINES) ---

                st.subheader("📉 The Hard Deck")

                

                dates = df_active['Day']

                nlvs = df_active['End NLV']

                hwm_series = df_active['End NLV'].cummax()



                fig, ax = plt.subplots(figsize=(10, 6))

                

                # Main Series (Historical)

                ax.plot(dates, nlvs, color='black', linewidth=2.5, label='Net Liquidity')

                ax.plot(dates, hwm_series, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Peak (HWM)')

                

                # Hard Decks (Horizontal Lines based on CURRENT Peak)

                # using axhline for full width lines

                ax.axhline(y=deck_l1, color='#f1c40f', linewidth=1.5, alpha=0.8, label='L1: Remove Margin (-7.5%)')

                ax.axhline(y=deck_l2, color='#e67e22', linewidth=1.5, alpha=0.8, label='L2: 30% Invested (-12.5%)')

                ax.axhline(y=deck_l3, color='#c0392b', linewidth=2, alpha=0.8, label='L3: Cash (-15%)')

                

                # Stop Floor (Horizontal Line based on CURRENT Status)

                # If all stops hit today, this is where you land.

                ax.axhline(y=stop_out_floor_val, color='red', linestyle='--', linewidth=2, label=f'Stop-Out Floor')



                # Dynamic Scaling

                # We want to see at least the floor and the peak

                vals_to_see = [stop_out_floor_val, deck_l3, curr_nlv, peak_nlv]

                min_view = min(vals_to_see)

                max_view = peak_nlv

                

                if not np.isnan(min_view) and not np.isnan(max_view):

                    ax.set_ylim(bottom=min_view * 0.98, top=max_view * 1.01)



                ax.set_title(f"Risk Levels relative to Peak (Dynamic)")

                ax.set_ylabel("Account Value ($)")

                ax.legend(loc='upper left')

                ax.grid(True, linestyle='--', alpha=0.3)

                

                st.pyplot(fig)

                

                # --- 5. FUSE BOX INSTRUCTIONS (UPDATED) ---

                st.markdown("### 🧨 Fuse Box Protocols")

                f1, f2, f3 = st.columns(3)

                

                # LEVEL 1

                f1.markdown("#### 🟡 LEVEL 1")

                f1.markdown(f"**Trigger:** -7.5% DD (**${deck_l1:,.0f}**)")

                if curr_nlv <= deck_l1: f1.error("❌ FUSE BLOWN")

                else: f1.success("✅ SECURE")

                f1.info("**Action:** Remove Margin.\n\nLockout New Buys until steady.")



                # LEVEL 2

                f2.markdown("#### 🟠 LEVEL 2")

                f2.markdown(f"**Trigger:** -12.5% DD (**${deck_l2:,.0f}**)")

                if curr_nlv <= deck_l2: f2.error("❌ FUSE BLOWN")

                else: f2.success("✅ SECURE")

                f2.warning("**Action:** Max 30% Invested.\n\nManage winners only. Cut loose ends.")



                # LEVEL 3

                f3.markdown("#### ☠️ LEVEL 3")

                f3.markdown(f"**Trigger:** -15% DD (**${deck_l3:,.0f}**)")

                if curr_nlv <= deck_l3: f3.error("❌ FUSE BLOWN")

                else: f3.success("✅ SECURE")

                f3.error("**Action:** GO TO CASH.\n\nProtection Mode. No trading for 48hrs.")



            else:

                st.info(f"No journal data available after reset date.")

    else:

        st.warning("No journal data found. Please log your first trading day.")



# ==============================================================================

# TAB: PORTFOLIO VOLATILITY (HEAT CHECK)

# ==============================================================================

# --- TAB: PORTFOLIO HEAT (TRADINGVIEW ALIGNED) ---


# ====================================================================
# PORTFOLIO HEAT
# ====================================================================
elif page == "Portfolio Heat":
    st.subheader("🔥 Portfolio Volatility (Heat Check)")

    # Load data
    if not os.path.exists(DETAILS_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Amount','Value','Rule','Notes','Realized_PL','Stop_Loss','Trx_ID']).to_csv(DETAILS_FILE, index=False)
    if not os.path.exists(SUMMARY_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Total_Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Rule','Notes','Buy_Notes','Sell_Rule','Sell_Notes']).to_csv(SUMMARY_FILE, index=False)

    df_d = load_data(DETAILS_FILE)
    df_s = load_data(SUMMARY_FILE)



    # 1. Calculation Method Selector

    heat_mode = st.radio("Calculation Method", ["🤖 Automated (TradingView Formula)", "✍️ Manual Override"], horizontal=True)



    # Load current equity from journal

    calc_equity = 100000.0

    try:

        j_df = load_data(JOURNAL_FILE)

        if not j_df.empty and 'End NLV' in j_df.columns:

            # Sort by date to get the latest entry

            if 'Day' in j_df.columns:

                j_df['Day'] = pd.to_datetime(j_df['Day'], errors='coerce')

                j_df = j_df.dropna(subset=['Day']).sort_values('Day', ascending=False)

            calc_equity = float(str(j_df['End NLV'].iloc[0]).replace('$','').replace(',',''))

    except:

        pass



    if not df_s.empty:

        open_ops = df_s[df_s['Status'].astype(str).str.strip().str.upper() == 'OPEN'].copy()

        

        if not open_ops.empty:

            vol_data = []

            tickers_list = open_ops['Ticker'].unique().tolist()



            if heat_mode == "🤖 Automated (TradingView Formula)":

                st.info(f"📡 Syncing with TV 'SMA-Fixed' formula for {len(tickers_list)} positions...")

                my_bar = st.progress(0)

                

                try:

                    # Fetching 40 days of data to calculate a 21-period SMA

                    # group_by='ticker' makes it easy to loop through multiple stocks

                    batch_data = yf.download(

                        tickers_list, 

                        period="40d", 

                        interval="1d", 

                        progress=False, 

                        group_by='ticker'

                    )

                except Exception as e:

                    st.error(f"Download failed: {e}")

                    batch_data = pd.DataFrame()



                for i, ticker in enumerate(tickers_list):

                    my_bar.progress((i + 1) / len(tickers_list))

                    atr_pct = 0.0

                    

                    try:

                        # Handling single vs multiple ticker return formats

                        if len(tickers_list) > 1:

                            df_t = batch_data[ticker].copy().dropna()

                        else:

                            df_t = batch_data.copy().dropna()

                        

                        if len(df_t) >= 21:

                            # TRADINGVIEW "FIXED" FORMULA

                            # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))

                            df_t['H-L'] = df_t['High'] - df_t['Low']

                            df_t['H-PC'] = (df_t['High'] - df_t['Close'].shift(1)).abs()

                            df_t['L-PC'] = (df_t['Low'] - df_t['Close'].shift(1)).abs()

                            df_t['TR'] = df_t[['H-L', 'H-PC', 'L-PC']].max(axis=1)

                            

                            # ATR% = (SMA of TR / SMA of Lows) * 100

                            sma_tr = df_t['TR'].tail(21).mean()

                            sma_low = df_t['Low'].tail(21).mean()

                            

                            if sma_low > 0:

                                atr_pct = (sma_tr / sma_low) * 100

                    except:

                        atr_pct = 0.0

                    

                    # Calculate portfolio weighting

                    row = open_ops[open_ops['Ticker'] == ticker].iloc[0]

                    weight_pct = (row['Total_Cost'] / calc_equity) * 100

                    vol_data.append({

                        "Ticker": ticker, 

                        "Weight (%)": weight_pct, 

                        "ATR (21S) %": atr_pct, 

                        "Heat Contribution": weight_pct * (atr_pct / 100)

                    })

                my_bar.empty()



            else:

                # ✍️ MANUAL OVERRIDE MODE

                st.warning("Enter the ATR% (21S) value directly from your TradingView Table:")

                c_man = st.columns(4)

                for i, ticker in enumerate(tickers_list):

                    col_idx = i % 4

                    row = open_ops[open_ops['Ticker'] == ticker].iloc[0]

                    weight_pct = (row['Total_Cost'] / calc_equity) * 100

                    

                    with c_man[col_idx]:

                        m_atr = st.number_input(f"{ticker} ATR%", value=5.0, step=0.1, key=f"man_atr_{ticker}")

                        vol_data.append({

                            "Ticker": ticker, 

                            "Weight (%)": weight_pct, 

                            "ATR (21S) %": m_atr, 

                            "Heat Contribution": weight_pct * (m_atr / 100)

                        })



            # 3. DISPLAY RESULTS

            df_vol = pd.DataFrame(vol_data)

            total_heat = df_vol['Heat Contribution'].sum()

            

            # Check heat against MO Risk Rules (Target < 2.5%)

            m1, m2, m3 = st.columns(3)

            heat_color = "normal" if total_heat < 2.5 else "inverse"

            m1.metric("Total Portfolio Heat", f"{total_heat:.2f}%", delta="Target < 2.5%", delta_color=heat_color)

            m2.metric("Avg Stock Volatility", f"{df_vol['ATR (21S) %'].mean():.2f}%")

            m3.metric("Equity Basis", f"${calc_equity:,.0f}")



            st.dataframe(df_vol.style.format({

                "Weight (%)": "{:.1f}%", 

                "ATR (21S) %": "{:.2f}%", 

                "Heat Contribution": "{:.2f}%"

            }).background_gradient(subset=["Heat Contribution"], cmap="Oranges"), use_container_width=True)

            

        else:

            st.info("No open positions found to calculate heat.")

    else:

        st.warning("Trade Summary (df_s) is currently empty.")



# ==============================================================================

# TAB 8: ACTIVE CAMPAIGN DETAILED (DYNAMIC FLIGHT DECK + ORIG COST)

# ==============================================================================


# ====================================================================
# EARNINGS PLANNER
# ====================================================================
elif page == "Earnings Planner":
    st.subheader("💣 Earnings Risk Planner")

    st.caption("Binary Event Logic: Principal Protection (House Money Buffer).")

    # Load Data
    if not os.path.exists(DETAILS_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Amount','Value','Rule','Notes','Realized_PL','Stop_Loss','Trx_ID']).to_csv(DETAILS_FILE, index=False)
    if not os.path.exists(SUMMARY_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Total_Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Rule','Notes','Buy_Notes','Sell_Rule','Sell_Notes']).to_csv(SUMMARY_FILE, index=False)

    df_d = load_data(DETAILS_FILE)
    df_s = load_data(SUMMARY_FILE)

    if not df_s.empty:

        open_pos = df_s[df_s['Status'] == 'OPEN'].copy()

    else:

        if os.path.exists(SUMMARY_FILE):

            open_pos = pd.read_csv(SUMMARY_FILE)

            open_pos = open_pos[open_pos['Status'] == 'OPEN'].copy()

        else:

            open_pos = pd.DataFrame()



    # Load equity from journal (database-aware)

    equity = 100000.0

    try:

        j_df = load_data(JOURNAL_FILE)

        if not j_df.empty and 'End NLV' in j_df.columns:

            equity = float(str(j_df['End NLV'].iloc[-1]).replace('$','').replace(',',''))

    except:

        pass



    if not open_pos.empty:

        # 1. SELECT TICKER

        tickers = sorted(open_pos['Ticker'].unique())

        c_sel, c_blank = st.columns([1, 2])

        sel_ticker = c_sel.selectbox("Select Ticker into Earnings", tickers)

        

        # Get Position Data

        row = open_pos[open_pos['Ticker'] == sel_ticker].iloc[0]

        shares = float(row['Shares'])

        avg_cost = float(row['Avg_Entry'])

        

        # FIX 1: Price Default Logic

        # Priority: Session Cache -> Row Current Price -> 0.0 (Manual)

        def_price = 0.0

        if 'live_prices' in st.session_state and sel_ticker in st.session_state['live_prices']:

            def_price = st.session_state['live_prices'][sel_ticker]

        elif 'Current_Price' in row and float(row['Current_Price']) > 0:

            def_price = float(row['Current_Price'])

        

        # Fallback: If price is 0, we leave it 0 so user notices they need to input it

        # We explicitly DO NOT use avg_cost as default to avoid confusion

        

        # INPUTS SECTION

        st.markdown("---")

        st.markdown("#### 1. Setup & Cushion Check")

        

        c1, c2, c3, c4 = st.columns(4)

        curr_price = c1.number_input("Current Price ($)", value=float(def_price), step=0.10, format="%.2f")

        nlv_val = c2.number_input("Account Equity (NLV)", value=float(equity), step=1000.0)

        shares_held = c3.number_input("Shares Held", value=int(shares), step=1)

        cost_basis = c4.number_input("Avg Cost ($)", value=float(avg_cost), disabled=True)



        # CALCULATE CUSHION

        # Guard against zero division or zero price

        if avg_cost > 0 and curr_price > 0:

            unrealized_pct = ((curr_price - avg_cost) / avg_cost) * 100

            unrealized_dlr = (curr_price - avg_cost) * shares_held

        else:

            unrealized_pct = 0.0

            unrealized_dlr = 0.0

        

        # VISUAL CUSHION CHECK

        if unrealized_pct >= 10.0:

            st.success(f"✅ **PASS:** Cushion is {unrealized_pct:.2f}% (${unrealized_dlr:,.0f}). You have earned the right to hold.")

        elif unrealized_pct > 0:

            st.warning(f"⚠️ **THIN ICE:** Cushion is only {unrealized_pct:.2f}%. Any gap will likely eat principal.")

        else:

            st.error(f"❌ **FAIL:** You are underwater (-${abs(unrealized_dlr):,.0f}). Strategy Rule: **SELL ALL** before earnings.")



        st.markdown("---")

        st.markdown("#### 2. Stress Test Parameters")

        

        r1, r2, r3 = st.columns(3)

        risk_tol_pct = r1.slider("Max Capital Risk %", 0.1, 1.0, 0.5, 0.05, help="Max % of PRINCIPAL you are willing to lose.")

        

        # FIX 2: Expected Move in DOLLARS

        exp_move_dlr = r2.number_input("Implied Move (+/- $)", value=5.00, step=0.50, help="Enter the Market Maker Move (Straddle Price) in Dollars.")

        

        stress_mult = r3.radio("Stress Multiplier", [1.5, 2.0], index=1, horizontal=True)



        # CALCULATIONS

        # 1. The Gap

        gap_dlr = exp_move_dlr * stress_mult

        disaster_price = curr_price - gap_dlr

        

        # 2. The Drop (Market Value Loss)

        # This is how much equity vanishes from the screen

        total_drop_equity = gap_dlr * shares_held

        

        # FIX 3: Principal Risk Calculation

        # Capital Risk = (Avg Cost - Disaster Price) * Shares

        # If Disaster Price > Avg Cost, we are still profitable (Risk = 0)

        if disaster_price < avg_cost:

            principal_risk_dlr = (avg_cost - disaster_price) * shares_held

        else:

            principal_risk_dlr = 0.0 # House Money absorbed it all

        

        pct_impact_principal = (principal_risk_dlr / nlv_val) * 100

        max_allowed_loss = nlv_val * (risk_tol_pct / 100)

        

        # OUTPUTS

        st.markdown("---")

        st.markdown("#### 3. The Verdict")

        

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Disaster Price", f"${disaster_price:.2f}", f"-${gap_dlr:.2f} Gap")

        k2.metric("Profit Buffer", f"${unrealized_dlr:,.0f}", "Your Cushion")

        k3.metric("Projected Drawdown", f"-${total_drop_equity:,.0f}", "Equity Drop", delta_color="off")

        

        # The Critical Metric

        k4.metric("Risk to Principal", f"${principal_risk_dlr:,.0f}", f"{pct_impact_principal:.2f}% of NLV", delta_color="inverse")



        st.markdown("---")

        

        # LOGIC ENGINE

        if principal_risk_dlr <= max_allowed_loss:

            if principal_risk_dlr == 0:

                st.success(f"🛡️ **SAFE (HOUSE MONEY):** Even with a ${gap_dlr:.2f} gap, price (${disaster_price:.2f}) stays above your cost (${avg_cost:.2f}). No principal at risk.")

            else:

                st.success(f"✅ **APPROVED:** Principal risk is ${principal_risk_dlr:,.0f} ({pct_impact_principal:.2f}%), which is within your {risk_tol_pct}% budget.")

        else:

            # Calculate Trim needed to protect PRINCIPAL

            # Target Loss = Max Allowed

            # Current Loss = Principal Risk

            # Excess Loss = Principal Risk - Max Allowed

            # Shares to Sell = Excess Loss / (Avg Cost - Disaster Price)

            

            loss_per_share = avg_cost - disaster_price

            excess_loss = principal_risk_dlr - max_allowed_loss

            

            import math

            if loss_per_share > 0:

                shares_to_trim = math.ceil(excess_loss / loss_per_share)

            else:

                shares_to_trim = 0 # Should not happen in else block

            

            safe_shares = shares_held - shares_to_trim

            

            st.error(f"⛔ **RISK EXCEEDED:** You risk losing **{pct_impact_principal:.2f}%** of your starting capital.")

            

            c_act1, c_act2 = st.columns(2)

            c_act1.metric("REQUIRED TRIM", f"-{shares_to_trim} Shares", "Sell Before Close")

            c_act2.metric("Max Safe Hold", f"{safe_shares} Shares", f"Protects {risk_tol_pct}% Principal")



    else:

        st.info("No open positions found to analyze.")



# --- TAB 11: PERFORMANCE AUDIT (WITH PERIOD SELECTOR & FIXED MATH) ---


# ====================================================================
# PERFORMANCE AUDIT
# ====================================================================
elif page == "Performance Audit":
    st.subheader("🏆 Performance Audit: The Best & The Worst")

    st.markdown("Analysis of outlier trades to determine 'R' efficiency and P&L concentration.")

    # Load necessary data
    if not os.path.exists(DETAILS_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Amount','Value','Rule','Notes','Realized_PL','Stop_Loss','Trx_ID']).to_csv(DETAILS_FILE, index=False)
    if not os.path.exists(SUMMARY_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Total_Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Rule','Notes','Buy_Notes','Sell_Rule','Sell_Notes']).to_csv(SUMMARY_FILE, index=False)

    df_d = load_data(DETAILS_FILE)
    df_s = load_data(SUMMARY_FILE)

    if not df_s.empty:

        

        # --- 1. PERIOD SELECTOR ---

        c_scope1, c_scope2 = st.columns(2)

        scope_mode = c_scope1.selectbox("Analysis Period", ["All Time", "Current Year (YTD)", "Previous Year", "Custom Range"])

        

        # Filter Logic

        audit_source = df_s[df_s['Status'] == 'CLOSED'].copy()

        audit_source['Closed_Date'] = pd.to_datetime(audit_source['Closed_Date'], errors='coerce')

        

        start_d, end_d = None, None

        now = datetime.now()

        

        if scope_mode == "Current Year (YTD)":

            start_d = datetime(now.year, 1, 1)

            end_d = now

        elif scope_mode == "Previous Year":

            start_d = datetime(now.year - 1, 1, 1)

            end_d = datetime(now.year - 1, 12, 31)

        elif scope_mode == "Custom Range":

            d_range = c_scope2.date_input("Select Range", [now - timedelta(days=90), now])

            if len(d_range) == 2:

                start_d, end_d = datetime.combine(d_range[0], datetime.min.time()), datetime.combine(d_range[1], datetime.max.time())



        # Apply Filter

        if start_d and end_d:

            # Filter by Closed Date

            audit_df = audit_source[

                (audit_source['Closed_Date'] >= start_d) & 

                (audit_source['Closed_Date'] <= end_d)

            ].copy()

            st.caption(f"Showing trades closed between {start_d.strftime('%Y-%m-%d')} and {end_d.strftime('%Y-%m-%d')}")

        else:

            audit_df = audit_source.copy()

            st.caption("Showing ALL closed trades.")

        

        st.markdown("---")



        if not audit_df.empty:

            if st.button("🚀 RUN AUDIT", type="primary"):

                

                # 2. PREPARE HISTORY FOR NLV LOOKUP

                p_clean = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')

                p_legacy = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal.csv')

                path_j = p_clean if os.path.exists(p_clean) else p_legacy

                

                df_j_hist = pd.DataFrame()

                if os.path.exists(path_j):

                    try:

                        df_j_hist = pd.read_csv(path_j)

                        df_j_hist['Day'] = pd.to_datetime(df_j_hist['Day'], errors='coerce')

                        df_j_hist = df_j_hist.sort_values('Day', ascending=True)

                        

                        def clean_nlv_audit(x):

                            try: return float(str(x).replace('$', '').replace(',', '').strip())

                            except: return 0.0

                        if 'End NLV' in df_j_hist.columns:

                            df_j_hist['End NLV'] = df_j_hist['End NLV'].apply(clean_nlv_audit)

                    except: pass



                # 3. CALCULATION ENGINE

                results = []

                progress_bar = st.progress(0)

                total_rows = len(audit_df)

                

                for i, (idx, row) in enumerate(audit_df.iterrows()):

                    progress_bar.progress((i + 1) / total_rows)

                    

                    # A. Risk Budget & R-Multiple

                    budget = row.get('Risk_Budget', 0.0)

                    

                    if budget <= 0:

                        open_date = pd.to_datetime(row['Open_Date'])

                        if not df_j_hist.empty:

                            prior = df_j_hist[df_j_hist['Day'] < open_date]

                            if not prior.empty:

                                budget = prior.iloc[-1]['End NLV'] * 0.005

                            else: budget = 500.0

                        else: budget = 500.0

                    

                    realized = row['Realized_PL']

                    r_mult = realized / budget if budget > 0 else 0.0

                    

                    # B. Exit Efficiency

                    eff_val = 0.0

                    try:

                        o_date = pd.to_datetime(row['Open_Date']).tz_localize(None)

                        c_date = row['Closed_Date'].tz_localize(None) if pd.notnull(row['Closed_Date']) else datetime.now()

                        

                        h_data = yf.Ticker(row['Ticker']).history(start=o_date, end=c_date + timedelta(days=1))

                        if not h_data.empty:

                            period_high = h_data['High'].max()

                            if row['Shares'] > 0:

                                calc_exit = (row['Realized_PL'] / row['Shares']) + row['Avg_Entry']

                                if period_high > 0:

                                    eff_val = (calc_exit / period_high) * 100

                    except: pass

                    

                    results.append({

                        'Trade_ID': row['Trade_ID'],

                        'Ticker': row['Ticker'],

                        'Open_Date': row['Open_Date'],

                        'Closed_Date': row['Closed_Date'], # <--- ADDED CLOSED DATE

                        'Net P&L': realized,

                        'Return %': row.get('Return_Pct', 0.0),

                        'Risk Budget': budget,

                        'R-Multiple': r_mult,

                        'Exit Eff %': eff_val

                    })

                

                res_df = pd.DataFrame(results)

                progress_bar.empty()

                

                # 4. SORTING

                top_15 = res_df.sort_values('Net P&L', ascending=False).head(15)

                bot_15 = res_df.sort_values('Net P&L', ascending=True).head(15)

                

                # 5. AGGREGATE STATS (CORRECTED MATH)

                # Calculate Total Gross Profit (Sum of all positives) and Total Gross Loss (Sum of all negatives)

                gross_profit = res_df[res_df['Net P&L'] > 0]['Net P&L'].sum()

                gross_loss = res_df[res_df['Net P&L'] < 0]['Net P&L'].sum() # This is a negative number

                net_pl = gross_profit + gross_loss

                

                top_sum = top_15['Net P&L'].sum()

                bot_sum = bot_15['Net P&L'].sum()

                

                # Ratios

                pct_top_of_gross = (top_sum / gross_profit * 100) if gross_profit != 0 else 0

                pct_bot_of_loss = (bot_sum / gross_loss * 100) if gross_loss != 0 else 0 # e.g. -15k / -20k = 75%

                

                # --- DISPLAY METRICS ---

                st.markdown("### 📊 Concentration Analysis (Pareto)")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Net P&L (Period)", f"${net_pl:,.2f}")

                c2.metric("Total Gross Profit", f"${gross_profit:,.2f}")

                c3.metric("Total Gross Loss", f"${gross_loss:,.2f}")

                c4.metric("Profit Factor", f"{abs(gross_profit/gross_loss):.2f}" if gross_loss != 0 else "Inf")

                

                st.markdown("#### Outlier Impact")

                k1, k2 = st.columns(2)

                k1.metric("Top 15 Winners Sum", f"${top_sum:,.2f}", f"{pct_top_of_gross:.1f}% of Gross Profit")

                k2.metric("Bottom 15 Losers Sum", f"${bot_sum:,.2f}", f"{pct_bot_of_loss:.1f}% of Gross Loss", delta_color="inverse")

                

                st.markdown("---")

                

                # --- TOP 15 TABLE ---

                st.subheader("🟢 Top 15 Best Trades")

                st.dataframe(

                    top_15.style

                    .format({

                        'Net P&L': '${:,.2f}', 'Return %': '{:.2f}%', 

                        'Risk Budget': '${:,.0f}', 'R-Multiple': '{:+.2f}R',

                        'Exit Eff %': '{:.1f}%',

                        'Open_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else '',

                        'Closed_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else ''

                    })

                    .applymap(lambda x: 'color: #4CAF50' if x > 0 else 'color: #FF5252', subset=['Net P&L', 'R-Multiple', 'Return %']),

                    use_container_width=True,

                    height=550

                )

                

                st.markdown("---")

                

                # --- BOTTOM 15 TABLE ---

                st.subheader("🔴 Top 15 Worst Trades")

                st.dataframe(

                    bot_15.style

                    .format({

                        'Net P&L': '${:,.2f}', 'Return %': '{:.2f}%', 

                        'Risk Budget': '${:,.0f}', 'R-Multiple': '{:+.2f}R',

                        'Exit Eff %': '{:.1f}%',

                        'Open_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else '',

                        'Closed_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else ''

                    })

                    .applymap(lambda x: 'color: #4CAF50' if x > 0 else 'color: #FF5252', subset=['Net P&L', 'R-Multiple', 'Return %']),

                    use_container_width=True,

                    height=550

                )

                

        else:

            st.info("No closed trades found for this period.")

    else:

        st.warning("Summary file empty.")







# ==============================================================================

# PAGE 11: ANALYTICS (REVERTED TAB 1 + DRILL-DOWN LIVE TAB)

# ==============================================================================


# ==============================================================================
# TRADE JOURNAL - Unified view of all trades with card layout
# ==============================================================================
elif page == "Trade Journal":
    st.header("📔 Trade Journal")
    st.caption("Visual review of all your trades with embedded charts")

    # Quick action buttons
    _ac1, _ac2, _ac3 = st.columns([1, 1, 4])
    with _ac1:
        if st.button("🟢 Log Buy", key="tj_log_buy", use_container_width=True):
            st.session_state.page = "Log Buy"
            st.rerun()
    with _ac2:
        if st.button("🔴 Log Sell", key="tj_log_sell", use_container_width=True):
            st.session_state.page = "Log Sell"
            st.rerun()

    # Clear search state to force fresh search each time page is visited
    if 'last_visited_page' not in st.session_state or st.session_state.last_visited_page != 'Trade Journal':
        if 'journal_searched' in st.session_state:
            del st.session_state.journal_searched
    st.session_state.last_visited_page = 'Trade Journal'

    # Load data
    if not os.path.exists(DETAILS_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Amount','Value','Rule','Notes','Realized_PL','Stop_Loss','Trx_ID']).to_csv(DETAILS_FILE, index=False)
    if not os.path.exists(SUMMARY_FILE):
        pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Total_Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Rule','Notes','Buy_Notes','Sell_Rule','Sell_Notes']).to_csv(SUMMARY_FILE, index=False)

    df_d = load_data(DETAILS_FILE)
    df_s = load_data(SUMMARY_FILE)

    if df_s.empty:
        st.info("No trades found. Start logging trades in Trade Manager!")
    else:
        # === FILTERS ===
        st.markdown("### 🔍 Filters")

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        with col_f1:
            status_filter = st.selectbox(
                "Status",
                ["All", "Open", "Closed"],
                index=0
            )

        with col_f2:
            # Get unique tickers for hint
            all_tickers = sorted(df_s['Ticker'].unique().tolist())
            ticker_input = st.text_input(
                "Ticker(s) - comma separated",
                value="",
                placeholder="e.g., SNDK or SNDK, FIG, AAPL",
                help=f"Enter one or more tickers separated by commas. Available: {', '.join(all_tickers)}"
            ).strip().upper()
            # Parse tickers
            if ticker_input:
                ticker_filter = [t.strip() for t in ticker_input.split(',') if t.strip()]
            else:
                ticker_filter = "All"

        with col_f3:
            sort_by = st.selectbox(
                "Sort By",
                ["Newest First", "Oldest First", "Best P&L", "Worst P&L", "Ticker A-Z"],
                index=0
            )

        with col_f4:
            # Date range filter
            date_range = st.selectbox(
                "Date Range",
                ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", "This Year"],
                index=0
            )

        # Search button
        search_clicked = st.button("🔍 Search Trades", type="primary", use_container_width=True)

        st.markdown("---")

        # === APPLY FILTERS (only if search clicked) ===
        if not search_clicked and 'journal_searched' not in st.session_state:
            st.info("👆 Select your filters and click **Search Trades** to view your journal")
            st.stop()

        # Mark that search was performed
        st.session_state['journal_searched'] = True

        df_filtered = df_s.copy()

        # Status filter
        if status_filter == "Open":
            df_filtered = df_filtered[df_filtered['Status'].str.upper() == 'OPEN']
        elif status_filter == "Closed":
            df_filtered = df_filtered[df_filtered['Status'].str.upper() == 'CLOSED']

        # Ticker filter (supports multiple tickers)
        if ticker_filter != "All":
            df_filtered = df_filtered[df_filtered['Ticker'].isin(ticker_filter)]

        # Date range filter
        if 'Open_Date' in df_filtered.columns:
            df_filtered['Open_Date'] = pd.to_datetime(df_filtered['Open_Date'], errors='coerce')
            today = pd.Timestamp.now()

            if date_range == "Last 7 Days":
                df_filtered = df_filtered[df_filtered['Open_Date'] >= (today - pd.Timedelta(days=7))]
            elif date_range == "Last 30 Days":
                df_filtered = df_filtered[df_filtered['Open_Date'] >= (today - pd.Timedelta(days=30))]
            elif date_range == "Last 90 Days":
                df_filtered = df_filtered[df_filtered['Open_Date'] >= (today - pd.Timedelta(days=90))]
            elif date_range == "This Year":
                df_filtered = df_filtered[df_filtered['Open_Date'].dt.year == today.year]

        # === SORTING ===
        if sort_by == "Newest First":
            if 'Open_Date' in df_filtered.columns:
                df_filtered = df_filtered.sort_values('Open_Date', ascending=False)
        elif sort_by == "Oldest First":
            if 'Open_Date' in df_filtered.columns:
                df_filtered = df_filtered.sort_values('Open_Date', ascending=True)
        elif sort_by == "Best P&L":
            if 'Realized_PL' in df_filtered.columns:
                df_filtered = df_filtered.sort_values('Realized_PL', ascending=False)
        elif sort_by == "Worst P&L":
            if 'Realized_PL' in df_filtered.columns:
                df_filtered = df_filtered.sort_values('Realized_PL', ascending=True)
        elif sort_by == "Ticker A-Z":
            df_filtered = df_filtered.sort_values('Ticker', ascending=True)

        # === DISPLAY TRADES AS CARDS ===
        if df_filtered.empty:
            st.info("No trades match your filters.")
        else:
            st.markdown(f"### {len(df_filtered)} Trades Found")

            # Display each trade as a card
            for idx, trade in df_filtered.iterrows():
                trade_id = trade['Trade_ID']
                ticker = trade['Ticker']
                status = trade['Status']

                # Calculate metrics
                is_open = status.upper() == 'OPEN'

                # Get avg entry/exit first
                avg_entry = trade.get('Avg_Entry', 0)
                avg_exit = trade.get('Avg_Exit', 0)

                try:
                    avg_entry_val = float(str(avg_entry).replace('$', '').replace(',', ''))
                    avg_exit_val = float(str(avg_exit).replace('$', '').replace(',', ''))
                except:
                    avg_entry_val = 0.0
                    avg_exit_val = 0.0

                # === Calculate LIFO-based P&L for OPEN trades (use live price) ===
                if is_open:
                    # Run LIFO engine to get accurate unrealized P&L
                    target_df = df_d[df_d['Trade_ID'] == trade_id].copy()

                    if not target_df.empty:
                        # Sort transactions
                        target_df['Type_Rank'] = target_df['Action'].apply(lambda x: 0 if x == 'BUY' else 1)
                        if 'Date' in target_df.columns:
                            target_df['Date'] = pd.to_datetime(target_df['Date'], errors='coerce')
                            target_df = target_df.sort_values(['Date', 'Type_Rank'])

                        # LIFO engine
                        inventory = []
                        fd_realized_pl = 0.0
                        fd_remaining_shares = 0.0
                        fd_cost_basis_sum = 0.0

                        for tidx, row in target_df.iterrows():
                            if row['Action'] == 'BUY':
                                p = float(row.get('Amount', row.get('Price', 0.0)))
                                inventory.append({'idx': tidx, 'qty': row['Shares'], 'price': p})

                            elif row['Action'] == 'SELL':
                                to_sell = row['Shares']
                                sell_price = float(row.get('Amount', row.get('Price', 0.0)))
                                cost_basis_accum = 0.0
                                sold_qty_accum = 0.0

                                while to_sell > 0 and inventory:
                                    last = inventory[-1]
                                    take = min(to_sell, last['qty'])
                                    cost_basis_accum += (take * last['price'])
                                    sold_qty_accum += take
                                    last['qty'] -= take
                                    to_sell -= take
                                    if last['qty'] < 0.00001:
                                        inventory.pop()

                                revenue = sold_qty_accum * sell_price
                                true_pl = revenue - cost_basis_accum
                                fd_realized_pl += true_pl

                        for item in inventory:
                            fd_remaining_shares += item['qty']
                            fd_cost_basis_sum += (item['qty'] * item['price'])

                        # Get live price
                        try:
                            live_px = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                        except:
                            live_px = avg_entry_val

                        # Calculate unrealized P&L with live price
                        mkt_val = fd_remaining_shares * live_px
                        unrealized_pl = mkt_val - fd_cost_basis_sum

                        # Total P&L = Realized + Unrealized
                        pl_val = fd_realized_pl + unrealized_pl

                        # Return % based on total cost basis (all buys)
                        total_cost_basis = 0.0
                        for tidx, row in target_df.iterrows():
                            if row['Action'] == 'BUY':
                                p = float(row.get('Amount', row.get('Price', 0.0)))
                                total_cost_basis += (row['Shares'] * p)

                        return_pct = (pl_val / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
                        pl_label = "Total P&L"
                    else:
                        # Fallback if no detail data
                        pl_val = 0.0
                        return_pct = 0.0
                        pl_label = "Unrealized P&L"
                else:
                    # For closed trades, use summary data
                    pl = trade.get('Realized_PL', 0)
                    pl_label = "Realized P&L"
                    try:
                        pl_val = float(str(pl).replace('$', '').replace(',', ''))
                    except:
                        pl_val = 0.0

                    if avg_entry_val > 0:
                        return_pct = ((avg_exit_val - avg_entry_val) / avg_entry_val) * 100
                    else:
                        return_pct = 0.0

                # Days held
                open_date = trade.get('Open_Date')
                if pd.notna(open_date):
                    open_dt = pd.to_datetime(open_date, errors='coerce')
                    if pd.notna(open_dt):
                        if is_open:
                            days_held = (pd.Timestamp.now() - open_dt).days
                        else:
                            closed_date = trade.get('Closed_Date')
                            if pd.notna(closed_date):
                                closed_dt = pd.to_datetime(closed_date, errors='coerce')
                                if pd.notna(closed_dt):
                                    days_held = (closed_dt - open_dt).days
                                else:
                                    days_held = 0
                            else:
                                days_held = 0
                    else:
                        days_held = 0
                else:
                    days_held = 0

                # Card color based on P&L
                if pl_val > 0:
                    card_color = "#d4edda"  # Light green
                    border_color = "#28a745"  # Green
                elif pl_val < 0:
                    card_color = "#f8d7da"  # Light red
                    border_color = "#dc3545"  # Red
                else:
                    card_color = "#fff3cd"  # Light yellow
                    border_color = "#ffc107"  # Yellow

                # Status badge color
                status_color = "#28a745" if is_open else "#6c757d"

                # === CARD HTML ===
                st.markdown(f"""
                <div style="
                    background: {card_color};
                    border-left: 5px solid {border_color};
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <div>
                            <span style="font-size: 24px; font-weight: bold; color: #333;">{ticker}</span>
                            <span style="
                                margin-left: 10px;
                                padding: 4px 12px;
                                background: {status_color};
                                color: white;
                                border-radius: 12px;
                                font-size: 12px;
                                font-weight: bold;
                            ">{status.upper()}</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 20px; font-weight: bold; color: {border_color};">
                                {'+' if pl_val >= 0 else ''}${pl_val:,.2f}
                            </div>
                            <div style="font-size: 14px; color: #666;">
                                {'+' if return_pct >= 0 else ''}{return_pct:.2f}%
                            </div>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 15px;">
                        <div>
                            <div style="font-size: 11px; color: #666; text-transform: uppercase;">Entry</div>
                            <div style="font-size: 16px; font-weight: 600;">${avg_entry_val:,.2f}</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #666; text-transform: uppercase;">Exit</div>
                            <div style="font-size: 16px; font-weight: 600;">{'$' + f'{avg_exit_val:,.2f}' if not is_open else 'Active'}</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #666; text-transform: uppercase;">Shares</div>
                            <div style="font-size: 16px; font-weight: 600;">{trade.get('Shares', 0)}</div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #666; text-transform: uppercase;">Days Held</div>
                            <div style="font-size: 16px; font-weight: 600;">{days_held}</div>
                        </div>
                    </div>
                    <div style="font-size: 12px; color: #666;">
                        <strong>Trade ID:</strong> {trade_id} |
                        <strong>Opened:</strong> {open_date if pd.notna(open_date) else 'N/A'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # === CHARTS (In Expander) ===
                with st.expander(f"📊 Charts - {ticker}", expanded=False):
                    if R2_AVAILABLE and USE_DATABASE:
                        images = db.get_trade_images(CURR_PORT_NAME, trade_id)

                        if images:
                            image_types = {img['image_type']: img for img in images}

                            # Display images in columns
                            cols = st.columns(3 if not is_open else 2)

                            with cols[0]:
                                if 'weekly' in image_types:
                                    img_data = image_types['weekly']
                                    st.markdown("**📊 Weekly Chart**")
                                    image_bytes = r2.download_image(img_data['image_url'])
                                    if image_bytes:
                                        st.image(image_bytes, use_container_width=True, output_format="PNG")
                                    else:
                                        st.info("Chart not available")
                                else:
                                    st.info("No weekly chart")

                            with cols[1]:
                                if 'daily' in image_types:
                                    img_data = image_types['daily']
                                    st.markdown("**📈 Daily Chart**")
                                    image_bytes = r2.download_image(img_data['image_url'])
                                    if image_bytes:
                                        st.image(image_bytes, use_container_width=True, output_format="PNG")
                                    else:
                                        st.info("Chart not available")
                                else:
                                    st.info("No daily chart")

                            if not is_open and len(cols) > 2:
                                with cols[2]:
                                    if 'exit' in image_types:
                                        img_data = image_types['exit']
                                        st.markdown("**🎯 Exit Chart**")
                                        image_bytes = r2.download_image(img_data['image_url'])
                                        if image_bytes:
                                            st.image(image_bytes, use_container_width=True, output_format="PNG")
                                        else:
                                            st.info("Chart not available")
                                    else:
                                        st.info("No exit chart")
                        else:
                            st.info("No charts available for this trade")

                        # === UPLOAD/UPDATE CHARTS ===
                        st.markdown("---")
                        st.markdown("### 📤 Upload/Update Charts")

                        # Create columns based on trade status
                        if not is_open:
                            upload_col1, upload_col2, upload_col3 = st.columns(3)
                        else:
                            upload_col1, upload_col2 = st.columns(2)

                        with upload_col1:
                            weekly_upload = st.file_uploader(
                                "📊 Weekly Chart",
                                type=['png', 'jpg', 'jpeg'],
                                key=f'weekly_upload_{trade_id}'
                            )

                        with upload_col2:
                            daily_upload = st.file_uploader(
                                "📈 Daily Chart",
                                type=['png', 'jpg', 'jpeg'],
                                key=f'daily_upload_{trade_id}'
                            )

                        # Exit chart only for closed trades
                        if not is_open:
                            with upload_col3:
                                exit_upload = st.file_uploader(
                                    "🎯 Exit Chart",
                                    type=['png', 'jpg', 'jpeg'],
                                    key=f'exit_upload_{trade_id}'
                                )
                        else:
                            exit_upload = None

                        # Upload button
                        if st.button("💾 Save Charts", key=f'save_charts_{trade_id}', type="primary"):
                            if weekly_upload or daily_upload or exit_upload:
                                try:
                                    upload_count = 0

                                    if weekly_upload:
                                        url = r2.upload_image(weekly_upload, CURR_PORT_NAME, trade_id, ticker, 'weekly')
                                        if url:
                                            db.save_trade_image(CURR_PORT_NAME, trade_id, ticker, 'weekly', url, weekly_upload.name)
                                            upload_count += 1

                                    if daily_upload:
                                        url = r2.upload_image(daily_upload, CURR_PORT_NAME, trade_id, ticker, 'daily')
                                        if url:
                                            db.save_trade_image(CURR_PORT_NAME, trade_id, ticker, 'daily', url, daily_upload.name)
                                            upload_count += 1

                                    if exit_upload:
                                        url = r2.upload_image(exit_upload, CURR_PORT_NAME, trade_id, ticker, 'exit')
                                        if url:
                                            db.save_trade_image(CURR_PORT_NAME, trade_id, ticker, 'exit', url, exit_upload.name)
                                            upload_count += 1

                                    st.success(f"✅ Successfully uploaded {upload_count} chart(s)! Refresh to see changes.")
                                except Exception as e:
                                    st.error(f"❌ Error uploading charts: {e}")
                            else:
                                st.warning("⚠️ Please select at least one chart to upload")

                    else:
                        st.info("Chart display requires R2 storage and database connection")

                # === TRANSACTION DETAILS & NOTES (Using Active Campaign Detailed Logic) ===
                with st.expander(f"📊 Transaction Details & Notes - {ticker}"):
                    # Get all transactions for this trade
                    target_df = df_d[df_d['Trade_ID'] == trade_id].copy()

                    if not target_df.empty:
                        # === RUN LIFO ENGINE (Same as Active Campaign Detailed) ===
                        target_df['Type_Rank'] = target_df['Action'].apply(lambda x: 0 if x == 'BUY' else 1)
                        if 'Date' in target_df.columns:
                            target_df['Date'] = pd.to_datetime(target_df['Date'], errors='coerce')
                            target_df = target_df.sort_values(['Date', 'Type_Rank'])

                        remaining_map = {}
                        lifo_pl_map = {}
                        inventory = []
                        fd_realized_pl = 0.0
                        fd_remaining_shares = 0.0
                        fd_cost_basis_sum = 0.0

                        for idx, row in target_df.iterrows():
                            if row['Action'] == 'BUY':
                                p = float(row.get('Amount', row.get('Price', 0.0)))
                                inventory.append({'idx': idx, 'qty': row['Shares'], 'price': p})
                                remaining_map[idx] = row['Shares']

                            elif row['Action'] == 'SELL':
                                to_sell = row['Shares']
                                sell_price = float(row.get('Amount', row.get('Price', 0.0)))
                                cost_basis_accum = 0.0
                                sold_qty_accum = 0.0

                                while to_sell > 0 and inventory:
                                    last = inventory[-1]
                                    take = min(to_sell, last['qty'])

                                    cost_basis_accum += (take * last['price'])
                                    sold_qty_accum += take

                                    last['qty'] -= take
                                    to_sell -= take
                                    remaining_map[last['idx']] = last['qty']

                                    if last['qty'] < 0.00001:
                                        inventory.pop()

                                revenue = sold_qty_accum * sell_price
                                true_pl = revenue - cost_basis_accum
                                lifo_pl_map[idx] = true_pl
                                fd_realized_pl += true_pl

                        for item in inventory:
                            fd_remaining_shares += item['qty']
                            fd_cost_basis_sum += (item['qty'] * item['price'])

                        # Apply LIFO results to DataFrame
                        display_df = target_df.copy()
                        display_df['Remaining_Shares'] = display_df.index.map(remaining_map).fillna(0)
                        display_df['Realized_PL'] = display_df.index.map(lifo_pl_map).fillna(0)
                        display_df['Status'] = display_df['Remaining_Shares'].apply(lambda x: 'Open' if x > 0 else 'Closed')

                        # === FLIGHT DECK ===
                        try:
                            live_px = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                        except:
                            live_px = avg_entry_val

                        shares = fd_remaining_shares if is_open else 0
                        avg_cost = (fd_cost_basis_sum / shares) if shares > 0 else avg_entry_val
                        mkt_val = shares * live_px
                        unrealized = mkt_val - fd_cost_basis_sum if is_open else 0
                        return_pct_calc = (unrealized / fd_cost_basis_sum * 100) if fd_cost_basis_sum > 0 else 0.0

                        # Original avg cost (initial buys only)
                        orig_avg_cost = 0.0
                        if 'Trx_ID' in target_df.columns:
                            init_buys = target_df[target_df['Trx_ID'].astype(str).str.upper().str.startswith('B')]
                            if not init_buys.empty:
                                init_val = (init_buys['Shares'] * init_buys['Amount']).sum()
                                init_shs = init_buys['Shares'].sum()
                                orig_avg_cost = init_val / init_shs if init_shs > 0 else 0.0

                        # Get equity for position size
                        equity = 100000.0
                        try:
                            j_df = load_data(JOURNAL_FILE)
                            if not j_df.empty and 'End NLV' in j_df.columns:
                                if 'Day' in j_df.columns:
                                    j_df['Day'] = pd.to_datetime(j_df['Day'], errors='coerce')
                                    j_df = j_df.dropna(subset=['Day']).sort_values('Day', ascending=False)
                                equity = float(str(j_df['End NLV'].iloc[0]).replace('$','').replace(',',''))
                        except:
                            pass

                        pos_size_pct = (mkt_val / equity) * 100 if equity > 0 else 0.0

                        st.markdown(f"### 🚁 Flight Deck: {ticker}")
                        f1, f2, f3, f4, f5, f6, f7 = st.columns(7)
                        f1.metric("Current Price", f"${live_px:,.2f}")
                        f2.metric("Orig Cost", f"${orig_avg_cost:,.2f}", help="Avg Cost of Initial Buys")
                        f3.metric("Avg Cost", f"${avg_cost:,.2f}", help="Current Cost Basis")
                        f4.metric("Shares Held", f"{int(shares):,}")
                        f5.metric("Unrealized P&L", f"${unrealized:,.2f}", f"{return_pct_calc:.2f}%")
                        f6.metric("Realized P&L", f"${fd_realized_pl:,.2f}")
                        f7.metric("Total Equity", f"${mkt_val:,.2f}", f"{pos_size_pct:.1f}% Size")

                        st.markdown("---")

                        # === TRANSACTION TABLE (All columns like Active Campaign Detailed) ===
                        st.markdown("### 📋 Transaction History")

                        # Status filter for transaction rows
                        trx_status_filter = st.radio(
                            "Filter Status",
                            ["All", "Open", "Closed"],
                            index=0,
                            horizontal=True,
                            key=f'trx_status_{trade_id}'
                        )

                        # Calculate unrealized and return % per transaction
                        def calc_unrealized(row):
                            if row['Action'] == 'BUY' and row['Remaining_Shares'] > 0:
                                entry = float(row.get('Amount', row.get('Price', 0.0)))
                                return (live_px - entry) * row['Remaining_Shares']
                            return 0.0
                        display_df['Unrealized_PL'] = display_df.apply(calc_unrealized, axis=1)

                        def calc_return_pct(row):
                            if row['Action'] == 'BUY' and row['Remaining_Shares'] > 0:
                                entry = float(row.get('Amount', row.get('Price', 0.0)))
                                if entry > 0:
                                    return ((live_px - entry) / entry) * 100
                            return 0.0
                        display_df['Return_Pct'] = display_df.apply(calc_return_pct, axis=1)

                        # Add campaign start date
                        display_df['Campaign_Start'] = open_date if pd.notna(open_date) else 'N/A'

                        # Negate shares and value for sells
                        display_df['Shares'] = display_df.apply(lambda x: -x['Shares'] if x['Action'] == 'SELL' else x['Shares'], axis=1)

                        if 'Value' not in display_df.columns and 'Amount' in display_df.columns:
                            display_df['Value'] = display_df['Shares'].abs() * display_df['Amount']

                        display_df['Value'] = display_df.apply(lambda x: -x['Value'] if x['Action'] == 'SELL' else x['Value'], axis=1)

                        # Apply status filter to transaction rows
                        if trx_status_filter != "All":
                            display_df = display_df[display_df['Status'] == trx_status_filter]

                        # Define columns (same as Active Campaign Detailed)
                        final_cols = ['Trade_ID', 'Trx_ID', 'Campaign_Start', 'Date', 'Ticker', 'Action', 'Status',
                                    'Shares', 'Remaining_Shares', 'Amount', 'Stop_Loss', 'Value',
                                    'Realized_PL', 'Unrealized_PL', 'Return_Pct', 'Rule', 'Notes']
                        show_cols = [c for c in final_cols if c in display_df.columns]

                        # Color function for P&L
                        def color_pnl(val):
                            try:
                                if isinstance(val, str):
                                    val = float(val.replace('$', '').replace(',', '').replace('%', ''))
                                if val > 0:
                                    return 'color: #2ca02c'
                                elif val < 0:
                                    return 'color: #ff4b4b'
                            except:
                                pass
                            return ''

                        def color_neg_value(val):
                            try:
                                if isinstance(val, str):
                                    val = float(val.replace('$', '').replace(',', ''))
                                if val < 0:
                                    return 'color: #ff4b4b'
                            except:
                                pass
                            return ''

                        st.dataframe(
                            display_df[show_cols].style.format({
                                'Date': lambda x: x.strftime('%Y-%m-%d %H:%M') if isinstance(x, (pd.Timestamp, datetime)) else 'None',
                                'Campaign_Start': lambda x: x if isinstance(x, str) else (x.strftime('%Y-%m-%d %H:%M') if isinstance(x, (pd.Timestamp, datetime)) else 'None'),
                                'Amount':'${:,.2f}', 'Stop_Loss':'${:,.2f}', 'Value':'${:,.2f}',
                                'Realized_PL':'${:,.2f}', 'Unrealized_PL':'${:,.2f}',
                                'Return_Pct':'{:.2f}%', 'Remaining_Shares':'{:.0f}'
                            })
                            .applymap(color_pnl, subset=['Value','Realized_PL','Unrealized_PL', 'Return_Pct'])
                            .applymap(color_neg_value, subset=['Shares']),
                            height=min(len(display_df) * 35 + 38, 500),
                            use_container_width=True
                        )

                        st.markdown("---")

                    # === NOTES ===
                    st.markdown("### 📝 Trade Notes")

                    note_col1, note_col2 = st.columns(2)

                    with note_col1:
                        st.markdown("**Entry Notes**")
                        buy_notes = trade.get('Buy_Notes', trade.get('Notes', ''))
                        st.write(buy_notes if buy_notes else "_No entry notes_")

                        st.markdown("**Setup/Rule**")
                        st.write(trade.get('Rule', '_Not specified_'))

                    with note_col2:
                        if not is_open:
                            st.markdown("**Exit Notes**")
                            sell_notes = trade.get('Sell_Notes', '')
                            st.write(sell_notes if sell_notes else "_No exit notes_")

                            st.markdown("**Exit Rule**")
                            st.write(trade.get('Sell_Rule', '_Not specified_'))

                st.markdown("---")

# ==============================================================================
# PAGE 11: ANALYTICS (REVERTED TAB 1 + DRILL-DOWN LIVE TAB)
# ==============================================================================
elif page == "Analytics":
    st.header(f"ANALYTICS & AUDIT ({CURR_PORT_NAME})")
    
    # 1. LOAD DATA
    if os.path.exists(SUMMARY_FILE):
        df_s_raw = load_data(SUMMARY_FILE) # Load raw data first
        
        df_j = pd.DataFrame()
        if os.path.exists(JOURNAL_FILE):
            df_j = load_data(JOURNAL_FILE)

        # --- DATA PREP ---
        df_s_raw['Closed_Date'] = pd.to_datetime(df_s_raw['Closed_Date'], errors='coerce')
        df_s_raw['Open_Date_DT'] = pd.to_datetime(df_s_raw['Open_Date'], errors='coerce')
        
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
        
        # --- ADD TIME-FRAME TOGGLE ---
        st.sidebar.markdown("---")
        view_scope = st.sidebar.radio("Analysis Scope (Tab 1 Only)", ["Life to Date (LTD)", "Current Year (2026)"], index=0)
        
        if view_scope == "Current Year (2026)":
            df_s = df_s_raw[df_s_raw['Open_Date_DT'].dt.year == 2026].copy()
        else:
            df_s = df_s_raw.copy()

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
        
        # Ensure Strat_Rule exists - prioritize Buy_Rule (initial buy) over Rule (can be overwritten by adds)
        if 'Buy_Rule' in all_sorted.columns: all_sorted['Strat_Rule'] = all_sorted['Buy_Rule'].fillna("Unknown")
        elif 'Rule' in all_sorted.columns: all_sorted['Strat_Rule'] = all_sorted['Rule'].fillna("Unknown")
        else: all_sorted['Strat_Rule'] = "Unknown"

        # Create closed from all_sorted (which has Strat_Rule) instead of df_s
        closed = all_sorted[all_sorted['Status']=='CLOSED'].copy()
        
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
        tab_perf, tab_live, tab_dd, tab_stats = st.tabs(["📊 Performance Audit", "🔥 Live Performance (MtM)", "📉 Drawdown Detective", "📈 Career Stats"])

        # --- TAB 1: PERFORMANCE AUDIT (EXACT ORIGINAL) ---
        with tab_perf:
            st.subheader(f"1. Scoreboard ({view_scope})")
            
            if PLOTLY_AVAILABLE:
                # Enhanced Profit Factor Gauge
                pf_color = "#2ca02c" if pf_val > 1.5 else ("#ffcc00" if pf_val > 1.0 else "#ff4b4b")
                fig_pf = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = pf_val,
                    title = {'text': "Profit Factor", 'font': {'size': 20}},
                    delta = {'reference': 1.5, 'increasing': {'color': '#2ca02c'}, 'decreasing': {'color': '#ff4b4b'}},
                    number = {'font': {'size': 40}},
                    gauge = {
                        'axis': {'range': [0, 5], 'tickwidth': 1},
                        'bar': {'color': pf_color, 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 1], 'color': '#ffe6e6'},
                            {'range': [1, 1.5], 'color': '#fff4e6'},
                            {'range': [1.5, 5], 'color': '#e6ffe6'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 1.5
                        }
                    }
                ))
                fig_pf.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor="white")

                # Enhanced Win Rate Gauge
                wr_color = "#2ca02c" if bat_avg >= 50 else ("#1f77b4" if bat_avg >= 40 else "#ff4b4b")
                fig_wr = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = bat_avg,
                    title = {'text': "Win Rate", 'font': {'size': 20}},
                    delta = {'reference': 50, 'increasing': {'color': '#2ca02c'}, 'decreasing': {'color': '#ff4b4b'}},
                    number = {'suffix': "%", 'font': {'size': 40}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': wr_color, 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': '#ffe6e6'},
                            {'range': [40, 50], 'color': '#fff4e6'},
                            {'range': [50, 100], 'color': '#e6ffe6'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_wr.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor="white")

                # Enhanced Win/Loss Comparison (Butterfly Chart)
                fig_wl = go.Figure()
                fig_wl.add_trace(go.Bar(
                    y=['Average'],
                    x=[avg_win],
                    name='Win',
                    orientation='h',
                    marker=dict(color='#2ca02c', line=dict(color='#1f7a1f', width=2)),
                    text=f"${avg_win:,.0f}",
                    textposition='outside',
                    textfont=dict(size=14, color='#2ca02c', family='Arial Black')
                ))
                fig_wl.add_trace(go.Bar(
                    y=['Average'],
                    x=[avg_loss],
                    name='Loss',
                    orientation='h',
                    marker=dict(color='#ff4b4b', line=dict(color='#cc0000', width=2)),
                    text=f"${avg_loss:,.0f}",
                    textposition='outside',
                    textfont=dict(size=14, color='#ff4b4b', family='Arial Black')
                ))
                fig_wl.update_layout(
                    title={
                        'text': f"<b>Win/Loss Ratio: {wl_ratio:.2f}x</b><br><sub>Target: 2.0x+ for asymmetric edge</sub>",
                        'font': {'size': 18}
                    },
                    barmode='relative',
                    height=280,
                    margin=dict(l=20, r=20, t=80, b=40),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='lightgray',
                        zeroline=True,
                        zerolinecolor='black',
                        zerolinewidth=2,
                        title="Dollar Amount"
                    ),
                    yaxis=dict(showticklabels=True),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

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

            # --- ADVANCED PERFORMANCE METRICS (NEW) ---
            if not closed.empty:
                st.markdown("---")
                st.subheader("📊 Advanced Performance Metrics")

                # Calculate Expectancy
                win_prob = bat_avg / 100
                loss_prob = 1 - win_prob
                expectancy = (win_prob * avg_win) + (loss_prob * avg_loss)

                # Calculate R-Multiple stats (if Risk_Budget exists)
                has_risk_data = 'Risk_Budget' in closed.columns and closed['Risk_Budget'].sum() > 0
                if has_risk_data:
                    closed_with_r = closed[closed['Risk_Budget'] > 0].copy()
                    closed_with_r['R_Multiple'] = closed_with_r['Realized_PL'] / closed_with_r['Risk_Budget']
                    avg_r_multiple = closed_with_r['R_Multiple'].mean()
                    max_r_multiple = closed_with_r['R_Multiple'].max()
                    min_r_multiple = closed_with_r['R_Multiple'].min()
                else:
                    avg_r_multiple = 0
                    max_r_multiple = 0
                    min_r_multiple = 0

                # Calculate Time in Trade
                closed_with_dates = closed.dropna(subset=['Open_Date_DT', 'Closed_Date'])
                if not closed_with_dates.empty:
                    closed_with_dates['Days_Held'] = (closed_with_dates['Closed_Date'] - closed_with_dates['Open_Date_DT']).dt.days
                    avg_hold_time = closed_with_dates['Days_Held'].mean()

                    winners_time = closed_with_dates[closed_with_dates['Realized_PL'] > 0]['Days_Held'].mean() if not wins.empty else 0
                    losers_time = closed_with_dates[closed_with_dates['Realized_PL'] <= 0]['Days_Held'].mean() if not losses.empty else 0
                else:
                    avg_hold_time = 0
                    winners_time = 0
                    losers_time = 0

                # Display Advanced Metrics
                adv1, adv2, adv3, adv4 = st.columns(4)

                # Expectancy
                exp_color = "normal" if expectancy > 0 else "inverse"
                adv1.metric(
                    "Expectancy",
                    f"${expectancy:,.2f}",
                    help="Expected $ per trade = (Win% × Avg Win) + (Loss% × Avg Loss). Positive = profitable system.",
                    delta_color=exp_color
                )

                # R-Multiple
                if has_risk_data:
                    r_color = "normal" if avg_r_multiple > 0 else "inverse"
                    adv2.metric(
                        "Avg R-Multiple",
                        f"{avg_r_multiple:.2f}R",
                        help="Average reward-to-risk ratio. 2R+ means you're making 2x your risk per trade on average.",
                        delta=f"Max: {max_r_multiple:.1f}R",
                        delta_color=r_color
                    )
                else:
                    adv2.metric(
                        "Avg R-Multiple",
                        "N/A",
                        help="Risk data not available. Log trades with Risk_Budget to track R-multiples."
                    )

                # Time in Trade - Winners vs Losers
                if avg_hold_time > 0:
                    time_ratio = winners_time / losers_time if losers_time > 0 else 0
                    adv3.metric(
                        "Avg Hold Time",
                        f"{avg_hold_time:.1f} days",
                        help="Average days held across all closed positions."
                    )

                    time_delta = "✅ Cut losses faster" if winners_time > losers_time else "⚠️ Hold losers too long"
                    adv4.metric(
                        "Win/Loss Hold Ratio",
                        f"{time_ratio:.2f}x",
                        help="Winners held / Losers held. >1.0 = Let winners run, cut losers quick.",
                        delta=time_delta
                    )
                else:
                    adv3.metric("Avg Hold Time", "N/A")
                    adv4.metric("Win/Loss Hold Ratio", "N/A")

                # Optional: Add a visual breakdown if data available
                if has_risk_data and PLOTLY_AVAILABLE:
                    st.markdown("#### R-Multiple Distribution")
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Histogram(
                        x=closed_with_r['R_Multiple'],
                        nbinsx=20,
                        marker=dict(color='#1f77b4', line=dict(color='black', width=1)),
                        name='R-Multiple Distribution'
                    ))
                    fig_r.update_layout(
                        xaxis_title="R-Multiple",
                        yaxis_title="Number of Trades",
                        height=300,
                        margin=dict(l=40, r=40, t=40, b=40),
                        showlegend=False
                    )
                    fig_r.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Breakeven")
                    st.plotly_chart(fig_r, use_container_width=True)

            # --- MARKET ENVIRONMENT (RESTORED) ---
            if len(all_sorted) >= 3:
                st.markdown("---")
                st.subheader("2. Market Environment (Last 10 Initiated)")
                st.caption("Includes both Open and Closed trades to show current real-time form.")
                
                def calc_form(n):
                    slice_df = all_sorted.head(n)
                    w = slice_df[slice_df['Slump_PL'] > 0]
                    l = slice_df[slice_df['Slump_PL'] <= 0]
                    win_rate = len(w) / len(slice_df) if len(slice_df) > 0 else 0.0
                    p_fac = w['Slump_PL'].sum() / abs(l['Slump_PL'].sum()) if abs(l['Slump_PL'].sum()) > 0 else 999
                    net_env = slice_df['Slump_PL'].sum()
                    return win_rate, p_fac, net_env
                
                wr_10, pf_10, net_10 = calc_form(10)
                wr_20, pf_20, net_20 = calc_form(20)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Env Win Rate (10)", f"{wr_10:.1%}", delta="- COLD" if wr_10 < 0.4 else "+ OK", delta_color="normal")
                c2.metric("Env P&L (10)", f"${net_10:,.0f}")
                c3.metric("Env Win Rate (20)", f"{wr_20:.1%}")
                c4.metric("Env P&L (20)", f"${net_20:,.0f}")
            
            if not closed.empty:
                # Filter out "Add:" rules (these are adds to existing positions, not initial entries)
                closed_initial = closed[~closed['Strat_Rule'].astype(str).str.contains('Add:', case=False, na=False)].copy()

                if not closed_initial.empty:
                    strat = closed_initial.groupby('Strat_Rule').agg(
                        Trades=('Trade_ID','count'),
                        PL=('Realized_PL','sum'),
                        WinRate=('Realized_PL', lambda x: (x>0).mean())
                    ).sort_values('PL', ascending=False)

                    st.markdown("---")
                    st.subheader("3. Strategy Breakdown")
                    st.dataframe(
                        strat.style.format({'PL':'${:,.2f}', 'WinRate':'{:.1%}'})
                        .applymap(lambda x: 'color: #2ca02c' if x>0 else 'color: #ff4b4b', subset=['PL'])
                    )

                    st.markdown("---")
                    st.subheader("4. Rule Forensics (Drill Down)")
                    avail_rules = sorted(closed_initial['Strat_Rule'].astype(str).unique().tolist())
                else:
                    st.info("No initial buy rules found (all rules contain 'Add:')")
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
                st.info("No closed trades yet in this view.")

        # --- TAB 2: LIVE PERFORMANCE (CORRECTED MTM LOGIC + DRILL DOWN) ---
        with tab_live:
            st.subheader(f"🔥 Live Year-to-Date Performance (2026)")
            st.caption("Consolidates trades Open in 2026 OR Closed in 2026 (including carry-over from 2025).")

            # Price Refresh Control
            col_refresh1, col_refresh2 = st.columns([1, 3])
            with col_refresh1:
                if st.button("🔄 Refresh Prices", help="Update unrealized P/L with current market prices"):
                    if USE_DATABASE:
                        with st.spinner("Fetching current prices..."):
                            try:
                                result = db.refresh_open_position_prices(CURR_PORT_NAME)
                                st.success(f"✅ {result['message']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Refresh failed: {str(e)}")
                    else:
                        st.warning("Database mode required for price refresh")

            st.markdown("---")

            # --- LOGIC UPDATE: INCLUDE CARRY-OVER TRADES ---
            mask_open_2026 = df_s_raw['Open_Date_DT'].dt.year == 2026
            mask_active = df_s_raw['Status'] == 'OPEN'
            mask_closed_2026 = df_s_raw['Closed_Date'].dt.year == 2026
            
            df_mtm = df_s_raw[mask_open_2026 | mask_active | mask_closed_2026].copy()
            
            if not df_mtm.empty:
                # 2. Calculate "Contribution"
                def get_contribution(row):
                    if row['Status'] == 'CLOSED':
                        return float(row['Realized_PL'])
                    else:
                        return float(row.get('Unrealized_PL', 0.0))
                
                df_mtm['Contribution'] = df_mtm.apply(get_contribution, axis=1)
                df_mtm['Is_Winner'] = df_mtm['Contribution'] > 0
                
                # 3. High Level Metrics
                total_contrib = df_mtm['Contribution'].sum()
                live_batting_avg = df_mtm['Is_Winner'].mean() * 100
                total_trades = len(df_mtm)
                open_cnt = len(df_mtm[df_mtm['Status']=='OPEN'])
                closed_cnt = len(df_mtm[df_mtm['Status']=='CLOSED'])
                
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total 2026 Equity Added", f"${total_contrib:,.2f}", f"{total_trades} Active/Closed")
                k2.metric("Live Batting Avg", f"{live_batting_avg:.1f}%", "Includes Open Pos")
                k3.metric("Closed P&L", f"${df_mtm[df_mtm['Status']=='CLOSED']['Contribution'].sum():,.2f}", f"{closed_cnt} Trades")
                k4.metric("Open Paper P&L", f"${df_mtm[df_mtm['Status']=='OPEN']['Contribution'].sum():,.2f}", f"{open_cnt} Active")
                
                st.markdown("---")
                
                # 4. Strategy Leaderboard
                # Prioritize Buy_Rule (initial buy) over Rule (can be overwritten by adds)
                if 'Buy_Rule' in df_mtm.columns: df_mtm['Strat_Rule'] = df_mtm['Buy_Rule'].fillna("Unknown")
                elif 'Rule' in df_mtm.columns: df_mtm['Strat_Rule'] = df_mtm['Rule'].fillna("Unknown")
                else: df_mtm['Strat_Rule'] = "Unknown"

                # Filter out "Add:" rules (these are adds to existing positions, not initial entries)
                df_mtm_initial = df_mtm[~df_mtm['Strat_Rule'].astype(str).str.contains('Add:', case=False, na=False)].copy()

                if not df_mtm_initial.empty:
                    mtm_strat = df_mtm_initial.groupby('Strat_Rule').agg(
                        Total_Trades=('Trade_ID', 'count'),
                        Active_Count=('Status', lambda x: (x=='OPEN').sum()),
                        Net_Equity=('Contribution', 'sum'),
                        Live_WinRate=('Contribution', lambda x: (x>0).mean())
                    ).sort_values('Net_Equity', ascending=False)

                    st.subheader("🏆 Strategy Leaderboard (Mark-to-Market)")
                    st.dataframe(
                        mtm_strat.style.format({
                            'Net_Equity': '${:,.2f}',
                            'Live_WinRate': '{:.1%}'
                        })
                        .applymap(lambda x: 'color: #2ca02c' if x>0 else 'color: #ff4b4b', subset=['Net_Equity'])
                    )

                    # 5. Rule Forensics (Drill Down)
                    st.markdown("---")
                    st.subheader("4. Rule Forensics (Drill Down)")
                    avail_mtm_rules = sorted(df_mtm_initial['Strat_Rule'].astype(str).unique().tolist())
                else:
                    st.info("No initial buy rules found (all rules contain 'Add:')")
                sel_mtm_rule = st.selectbox("Select Strategy to Inspect", ["None"] + avail_mtm_rules, key="mtm_drill")
                
                if sel_mtm_rule != "None":
                    rule_trades_mtm = df_mtm[df_mtm['Strat_Rule'] == sel_mtm_rule].copy()
                    if not rule_trades_mtm.empty:
                        r_eq = rule_trades_mtm['Contribution'].sum()
                        r_count = len(rule_trades_mtm)
                        r_wr_mtm = rule_trades_mtm['Is_Winner'].mean()
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Net Equity Contribution", f"${r_eq:,.2f}", delta_color="normal")
                        m2.metric("Total Trades", r_count)
                        m3.metric("Live Win Rate", f"{r_wr_mtm:.1%}")
                        
                        cols_show = ['Trade_ID', 'Ticker', 'Status', 'Open_Date', 'Closed_Date', 'Contribution', 'Return_Pct']
                        valid_cols = [c for c in cols_show if c in rule_trades_mtm.columns]
                        
                        st.dataframe(
                            rule_trades_mtm[valid_cols].sort_values('Contribution', ascending=False)
                            .style.format({'Contribution': '${:,.2f}', 'Return_Pct': '{:.2f}%'})
                            .applymap(lambda x: 'color: #2ca02c' if x>0 else 'color: #ff4b4b', subset=['Contribution'])
                        )

            else:
                st.info("No active or closed trades found relevant to 2026.")

        # --- TAB 3: DRAWDOWN DETECTIVE (START DEC 16, 2025) ---
        with tab_dd:
            st.subheader("📉 Drawdown Forensics (Since Dec 16, 2025)")
            
            # 0. "HOW TO READ" GUIDE
            with st.expander("📚 How to read this chart"):
                st.markdown("""
                **1. The Red Chart (Underwater Plot):**
                * This shows how far your equity is below its **All-Time High**.
                * **0% Line:** You are at a new account high.
                * **Red Depth:** The percentage distance from your peak. (e.g. -5% means you are 5% off highs).
                
                **2. The Metrics:**
                * **Total Dollar Drop:** The actual cash value lost from Peak to Trough.
                * **Realized P&L:** Losses locked in by selling during the drop.
                * **The Bleed:** The drop in value of your *Open Positions*. High 'Bleed' means you held through pain rather than stopping out.
                """)

            # 1. FILTER DATA (START DATE)
            start_cutoff = pd.to_datetime("2025-12-16")
            
            if not df_j.empty:
                # Filter Journal for date range
                df_dd = df_j[df_j['Day'] >= start_cutoff].copy()
                
                if len(df_dd) > 2:
                    # 2. TWR CURVE CALCULATION
                    df_dd['Adjusted_Beg'] = df_dd['Beg NLV'] + df_dd['Cash -/+']
                    df_dd['Day_Ret'] = 0.0
                    mask = df_dd['Adjusted_Beg'] != 0
                    df_dd.loc[mask, 'Day_Ret'] = (df_dd.loc[mask, 'End NLV'] - df_dd.loc[mask, 'Adjusted_Beg']) / df_dd.loc[mask, 'Adjusted_Beg']
                    
                    # Cumulative Returns (Time-Weighted)
                    df_dd['TWR_Curve'] = (1 + df_dd['Day_Ret']).cumprod()
                    
                    # 3. DRAWDOWN CALCULATION
                    df_dd['HWM_TWR'] = df_dd['TWR_Curve'].cummax()
                    df_dd['DD_Pct'] = (df_dd['TWR_Curve'] - df_dd['HWM_TWR']) / df_dd['HWM_TWR']
                    
                    # 4. IDENTIFY PERIODS (CLUSTERING)
                    # We consider a drawdown "active" if it's deeper than -0.5% (noise filter)
                    df_dd['Is_DD'] = df_dd['DD_Pct'] < -0.005 
                    df_dd['DD_Group'] = (df_dd['Is_DD'] != df_dd['Is_DD'].shift()).cumsum()
                    
                    dd_periods = []
                    # Group contiguous drawdown days
                    for grp_id, data in df_dd[df_dd['Is_DD']].groupby('DD_Group'):
                        start_d = data['Day'].min()
                        end_d = data['Day'].max()
                        depth_pct = data['DD_Pct'].min()
                        duration = (end_d - start_d).days + 1
                        
                        # Total Pain = Sum of Daily $ Changes during the drop
                        # Note: This is a simplified approx of NLV drop
                        peak_val = df_dd.loc[df_dd['Day'] == start_d, 'Beg NLV'].iloc[0]
                        trough_val = df_dd.loc[df_dd['Day'] == end_d, 'End NLV'].iloc[0]
                        total_pain = trough_val - peak_val
                        
                        dd_periods.append({
                            'Start': start_d, 'End': end_d, 
                            'Depth': depth_pct * 100, 
                            'Total_Pain': total_pain,
                            'Days': duration
                        })
                    
                    # Sort by Depth (Worst First)
                    worst_dds = sorted(dd_periods, key=lambda x: x['Depth'])[:3]
                    
                    # 5. VISUALIZATION
                    if worst_dds:
                        fig_dd, ax_dd = plt.subplots(figsize=(10, 4))
                        
                        # Fill area red
                        ax_dd.fill_between(df_dd['Day'], df_dd['DD_Pct']*100, 0, color='red', alpha=0.3)
                        # Plot line
                        ax_dd.plot(df_dd['Day'], df_dd['DD_Pct']*100, color='red', linewidth=1)
                        
                        # Formatting
                        ax_dd.set_title("Drawdown Depth % (from High Water Mark)")
                        ax_dd.set_ylabel("Percentage Below Peak")
                        ax_dd.grid(True, linestyle='--', alpha=0.3)
                        # Format x-axis dates
                        ax_dd.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
                        
                        st.pyplot(fig_dd)

                        # 6. SEQUENCE ANALYSIS
                        st.markdown("### 🚑 The Top Drawdown Sequences")
                        cols = st.columns(3)
                        
                        for i, dd in enumerate(worst_dds):
                            with cols[i]:
                                st.error(f"Sequence #{i+1}")
                                st.caption(f"{dd['Start'].strftime('%Y-%m-%d')} ⮕ {dd['End'].strftime('%Y-%m-%d')}")
                                
                                st.metric("Max Depth", f"{dd['Depth']:.2f}%", f"{dd['Days']} Days Duration")
                                st.metric("Account Value Drop", f"${dd['Total_Pain']:,.0f}", delta_color="inverse")
                                
                                # Analyze Culprits (Closed Trades during this window)
                                mask_loss = ((closed['Closed_Date'] >= dd['Start']) & (closed['Closed_Date'] <= dd['End']))
                                trades_in_window = closed[mask_loss]
                                
                                realized_pain = trades_in_window['Realized_PL'].sum()
                                # Bleed = Total Drop - Realized. (If you dropped $5k but only realized -$1k, $4k was open bleed)
                                bleed_pain = dd['Total_Pain'] - realized_pain
                                
                                st.markdown("---")
                                st.caption("Pain Breakdown:")
                                st.markdown(f"**Realized Losses:** :red[${realized_pain:,.0f}]")
                                st.markdown(f"**Open Position Bleed:** :red[${bleed_pain:,.0f}]")
                                
                                st.caption("Top Realized Losers:")
                                culprits = trades_in_window[trades_in_window['Realized_PL'] < 0].sort_values('Realized_PL', ascending=True).head(3)
                                if not culprits.empty:
                                    for _, trade in culprits.iterrows():
                                        st.write(f"• **{trade['Ticker']}**: :red[${trade['Realized_PL']:,.0f}]")
                                else:
                                    st.info("No realized losses. All pain was unrealized (Bleed).")
                    else:
                        st.success("✅ No significant drawdowns (>0.5%) detected since Dec 16.")
                else:
                    st.info("Not enough data points since Dec 16 to plot curve.")
            else:
                st.error("No Journal Data found.")

        # --- TAB 4: CAREER STATS (ALL-TIME OVERVIEW) ---
        with tab_stats:
            st.subheader("📈 Career Stats (All-Time Overview)")
            st.caption("Comprehensive statistics across all closed trades in your trading career")

            # Use df_s_raw for all-time data (not filtered by year)
            all_trades = df_s_raw.copy()
            all_closed = all_trades[all_trades['Status'] == 'CLOSED'].copy()

            if not all_closed.empty:
                # Calculate all metrics
                total_pl = all_closed['Realized_PL'].sum()
                total_trades = len(all_closed)

                # Add Hold_Days column BEFORE creating winners/losers slices
                all_closed['Hold_Days'] = (all_closed['Closed_Date'] - all_closed['Open_Date_DT']).dt.total_seconds() / 86400

                winners = all_closed[all_closed['Realized_PL'] > 0]
                losers = all_closed[all_closed['Realized_PL'] < 0]
                break_even = all_closed[all_closed['Realized_PL'] == 0]

                num_winners = len(winners)
                num_losers = len(losers)
                num_break_even = len(break_even)

                win_rate = (num_winners / total_trades * 100) if total_trades > 0 else 0

                avg_win = winners['Realized_PL'].mean() if not winners.empty else 0
                avg_loss = losers['Realized_PL'].mean() if not losers.empty else 0
                avg_trade = all_closed['Realized_PL'].mean() if not all_closed.empty else 0

                wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

                largest_win = winners['Realized_PL'].max() if not winners.empty else 0
                largest_loss = losers['Realized_PL'].min() if not losers.empty else 0

                # Profit Factor
                gross_profit = winners['Realized_PL'].sum() if not winners.empty else 0
                gross_loss = abs(losers['Realized_PL'].sum()) if not losers.empty else 0
                profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

                # Consecutive wins/losses
                def get_max_consecutive(df, condition_col, threshold):
                    """Calculate max consecutive wins or losses"""
                    if df.empty:
                        return 0
                    df_sorted = df.sort_values('Closed_Date')
                    is_match = df_sorted[condition_col] > threshold if threshold >= 0 else df_sorted[condition_col] < threshold
                    groups = (is_match != is_match.shift()).cumsum()
                    consecutive = is_match.groupby(groups).sum()
                    return int(consecutive.max()) if not consecutive.empty else 0

                max_consecutive_wins = get_max_consecutive(all_closed, 'Realized_PL', 0)
                max_consecutive_losses = get_max_consecutive(all_closed, 'Realized_PL', -0.01)

                # Hold times (Hold_Days already added above)
                avg_hold_all = all_closed['Hold_Days'].mean() if not all_closed.empty else 0

                winners_hold = winners['Hold_Days'].mean() if not winners.empty else 0
                losers_hold = losers['Hold_Days'].mean() if not losers.empty else 0
                scratch_hold = break_even['Hold_Days'].mean() if not break_even.empty else 0

                hold_ratio = winners_hold / losers_hold if losers_hold > 0 else 0

                # R-Multiple metrics
                has_risk_data = 'Risk_Budget' in all_closed.columns and all_closed['Risk_Budget'].notna().any()
                if has_risk_data:
                    closed_with_r = all_closed[all_closed['Risk_Budget'] > 0].copy()
                    closed_with_r['R_Multiple'] = closed_with_r['Realized_PL'] / closed_with_r['Risk_Budget']
                    avg_r_multiple = closed_with_r['R_Multiple'].mean()
                    max_r_multiple = closed_with_r['R_Multiple'].max()
                else:
                    avg_r_multiple = 0
                    max_r_multiple = 0

                # Expectancy
                expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

                # Monthly Performance (if we have enough data)
                if 'Closed_Date' in all_closed.columns:
                    all_closed['Month'] = pd.to_datetime(all_closed['Closed_Date']).dt.to_period('M')
                    monthly_pl = all_closed.groupby('Month')['Realized_PL'].sum()

                    if not monthly_pl.empty:
                        best_month = monthly_pl.max()
                        worst_month = monthly_pl.min()
                        avg_month = monthly_pl.mean()
                        best_month_date = monthly_pl.idxmax().strftime('%b %Y')
                        worst_month_date = monthly_pl.idxmin().strftime('%b %Y')
                    else:
                        best_month = worst_month = avg_month = 0
                        best_month_date = worst_month_date = "N/A"
                else:
                    best_month = worst_month = avg_month = 0
                    best_month_date = worst_month_date = "N/A"

                # Open positions
                open_trades = all_trades[all_trades['Status'] == 'OPEN']
                num_open = len(open_trades)

                # ==============================================================================
                # DISPLAY LAYOUT
                # ==============================================================================

                # --- MONTHLY PERFORMANCE ---
                st.markdown("### 📅 Monthly Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Month", f"${best_month:,.2f}",
                             delta=f"in {best_month_date}")
                with col2:
                    st.metric("Worst Month", f"${worst_month:,.2f}",
                             delta=f"in {worst_month_date}",
                             delta_color="inverse")
                with col3:
                    st.metric("Average Month", f"${avg_month:,.2f}")

                st.markdown("---")

                # --- TRADE PERFORMANCE ---
                st.markdown("### 💰 Trade Performance")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    pl_color = "normal" if total_pl > 0 else "inverse"
                    st.metric("Total P&L", f"${total_pl:,.2f}", delta_color=pl_color)
                with col2:
                    st.metric("Total Trades", f"{total_trades:,}")
                with col3:
                    st.metric("Average Trade", f"${avg_trade:,.2f}")
                with col4:
                    st.metric("Profit Factor", f"{profit_factor:.2f}",
                             delta="✅ Good" if profit_factor > 1.5 else "⚠️ Improve")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Winning Trades", f"{num_winners:,}",
                             delta=f"{win_rate:.1f}% win rate")
                with col2:
                    st.metric("Losing Trades", f"{num_losers:,}")
                with col3:
                    st.metric("Break-Even Trades", f"{num_break_even:,}")
                with col4:
                    st.metric("Open Positions", f"{num_open:,}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Winner", f"${avg_win:,.2f}",
                             delta=f"Largest: ${largest_win:,.2f}")
                with col2:
                    st.metric("Average Loser", f"${avg_loss:,.2f}",
                             delta=f"Largest: ${largest_loss:,.2f}",
                             delta_color="inverse")
                with col3:
                    st.metric("Win/Loss Ratio", f"{wl_ratio:.2f}x",
                             delta="✅ Good" if wl_ratio > 2.0 else "⚠️ Improve")

                st.markdown("---")

                # --- CONSISTENCY METRICS ---
                st.markdown("### 🎯 Consistency Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Max Consecutive Wins", f"{max_consecutive_wins:,}")
                with col2:
                    st.metric("Max Consecutive Losses", f"{max_consecutive_losses:,}")
                with col3:
                    st.metric("Win Rate", f"{win_rate:.1f}%",
                             delta="✅ Good" if win_rate >= 40 else "⚠️ Improve")
                with col4:
                    st.metric("Expectancy", f"${expectancy:,.2f}",
                             delta="Per trade average")

                st.markdown("---")

                # --- HOLDING TIME ANALYSIS ---
                st.markdown("### ⏱️ Holding Time Analysis")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Hold Time (All)", f"{avg_hold_all:.1f} days")
                with col2:
                    st.metric("Winners Hold Time", f"{winners_hold:.1f} days")
                with col3:
                    st.metric("Losers Hold Time", f"{losers_hold:.1f} days")
                with col4:
                    hold_status = "✅ Cut losses faster" if hold_ratio > 1.0 else "⚠️ Hold losers too long"
                    st.metric("Win/Loss Hold Ratio", f"{hold_ratio:.2f}x",
                             delta=hold_status)

                if num_break_even > 0:
                    st.caption(f"ℹ️ Scratch trades held on average: {scratch_hold:.1f} days")

                st.markdown("---")

                # --- RISK METRICS ---
                st.markdown("### 📊 Risk Metrics")
                if has_risk_data:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        r_color = "normal" if avg_r_multiple > 0 else "inverse"
                        st.metric("Avg R-Multiple", f"{avg_r_multiple:.2f}R",
                                 delta=f"Max: {max_r_multiple:.1f}R",
                                 delta_color=r_color)
                    with col2:
                        st.metric("Trade Expectancy", f"${expectancy:,.2f}",
                                 help="Average expected P&L per trade")
                    with col3:
                        # Calculate realized R-Multiple (for trades with risk budget)
                        if not closed_with_r.empty:
                            avg_realized_r = closed_with_r['R_Multiple'].mean()
                            st.metric("Avg Realized R", f"{avg_realized_r:.2f}R",
                                     help="Average R achieved on closed trades")
                        else:
                            st.metric("Avg Realized R", "N/A")
                else:
                    st.info("💡 Risk metrics not available. Log trades with Risk_Budget to track R-multiples and risk-adjusted performance.")

                st.markdown("---")

                # --- POSITION MANAGEMENT ---
                st.markdown("### 📈 Position Management")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Largest Profit", f"${largest_win:,.2f}",
                             delta="Single trade")
                with col2:
                    st.metric("Largest Loss", f"${largest_loss:,.2f}",
                             delta="Single trade",
                             delta_color="inverse")
                with col3:
                    # Calculate % of total profit from top winners
                    if not winners.empty and gross_profit > 0:
                        top_3_profit = winners.nlargest(3, 'Realized_PL')['Realized_PL'].sum()
                        top_3_pct = (top_3_profit / gross_profit * 100)
                        st.metric("Top 3 Winners", f"{top_3_pct:.1f}%",
                                 delta="of total profit")
                    else:
                        st.metric("Top 3 Winners", "N/A")

                # Optional: Add a summary insight
                st.markdown("---")
                with st.expander("📖 How to Read These Stats"):
                    st.markdown("""
                    **Monthly Performance:**
                    - Shows your best, worst, and average monthly P&L
                    - Helps identify seasonal patterns and consistency

                    **Trade Performance:**
                    - Total P&L and trade count give the big picture
                    - Profit Factor > 1.5 = healthy (making $1.50+ for every $1 lost)
                    - Win Rate 40%+ = good for trend following systems

                    **Consistency:**
                    - Max consecutive wins/losses show your streak patterns
                    - Expectancy = what you make per trade on average (should be positive!)

                    **Holding Time:**
                    - Win/Loss Hold Ratio > 1.0 = you let winners run and cut losers (good!)
                    - < 1.0 = holding losers too long (emotional trading)

                    **Risk Metrics:**
                    - Avg R-Multiple shows reward-to-risk ratio
                    - 2R+ means making 2x your risk per trade on average (excellent!)

                    **Position Management:**
                    - Top 3 Winners % shows concentration risk
                    - If >50%, you rely heavily on a few big winners
                    """)
            else:
                st.info("No closed trades yet. Start trading to see your career stats!")

# ==============================================================================
# PAGE 12: DAILY REPORT CARD (FIXED MARKET DATA)
# ==============================================================================
elif page == "Daily Report Card":
    st.header(f"📠 DAILY REPORT CARD ({CURR_PORT_NAME})")
    
    # 1. LOAD ALL DATA (database-aware)
    path_j = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')
    path_s = os.path.join(DATA_ROOT, portfolio, 'Trade_Log_Summary.csv')
    path_d = os.path.join(DATA_ROOT, portfolio, 'Trade_Log_Details.csv')

    df_j = load_data(path_j)
    df_s = load_data(path_s)
    df_d = load_data(path_d)

    if not df_j.empty: 
        
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
        available_dates = df_j['Day'].dt.date.unique()
        if len(available_dates) > 0:
            selected_date = st.selectbox("Select Date for Report", available_dates, index=0)
            
            # CHECK: IS THIS TODAY/RECENT?
            is_current_report = (selected_date >= get_current_date_ct() - timedelta(days=1))
            
            # --- GET DAY'S DATA ---
            day_stats = df_j[df_j['Day'].dt.date == selected_date].iloc[0]
            
            # --- 1. STATE OF THE MARKET (ROBUST FETCH) ---
            m_factor_str = "N/A"
            spy_chg_str = "0.00%"
            ndx_chg_str = "0.00%"
            
            # Fetch a wider window to ensure we catch the trading days
            start_fetch = pd.Timestamp(selected_date) - pd.Timedelta(days=7)
            end_fetch = pd.Timestamp(selected_date) + pd.Timedelta(days=1)
            
            try:
                # 1. FETCH DATA
                spy_hist = yf.Ticker("SPY").history(start=start_fetch, end=end_fetch)
                ndx_hist = yf.Ticker("^IXIC").history(start=start_fetch, end=end_fetch)
                
                # 2. NORMALIZE INDEX TO SIMPLE DATES (Fixes Timezone Mismatch)
                spy_hist.index = spy_hist.index.date
                ndx_hist.index = ndx_hist.index.date
                
                # 3. LOCATE SELECTED DATE (OR LATEST AVAILABLE IN WINDOW)
                # We iterate backwards from selected_date to find the first valid trading day
                check_date = selected_date
                
                if check_date in spy_hist.index:
                    # Perfect match
                    curr_loc = spy_hist.index.get_loc(check_date)
                    if curr_loc > 0:
                        # SPY
                        spy_close = spy_hist.iloc[curr_loc]['Close']
                        spy_prev = spy_hist.iloc[curr_loc - 1]['Close']
                        spy_pct = ((spy_close - spy_prev) / spy_prev) * 100
                        spy_chg_str = f"{spy_pct:+.2f}%"
                        
                        # NDX
                        if check_date in ndx_hist.index:
                            n_loc = ndx_hist.index.get_loc(check_date)
                            ndx_close = ndx_hist.iloc[n_loc]['Close']
                            ndx_prev = ndx_hist.iloc[n_loc - 1]['Close']
                            ndx_pct = ((ndx_close - ndx_prev) / ndx_prev) * 100
                            ndx_chg_str = f"{ndx_pct:+.2f}%"
                            
                            # M-FACTOR LOGIC (NDX > 21EMA)
                            # We need a longer lookback for EMA, so we fetch just NDX long history separately
                            long_ndx = yf.Ticker("^IXIC").history(start=pd.Timestamp(selected_date) - pd.Timedelta(days=60), end=end_fetch)
                            long_ndx.index = long_ndx.index.date
                            long_ndx['21EMA'] = long_ndx['Close'].ewm(span=21, adjust=False).mean()
                            
                            if check_date in long_ndx.index:
                                hist_close = long_ndx.loc[check_date]['Close']
                                hist_ema = long_ndx.loc[check_date]['21EMA']
                                trend = "UPTREND (Open)" if hist_close > hist_ema else "PRESSURE (Caution)"
                                m_factor_str = f"{trend} | NDX > 21e: {hist_close > hist_ema}"
                else:
                    spy_chg_str = "Closed/No Data"
                    
            except Exception as e: 
                m_factor_str = "Data Error"
                # st.error(f"Debug: {e}") # Uncomment to see error if needed

            # --- PREP TRADE DATA ---
            # 1. TRADES OPENED TODAY
            bought_today = pd.DataFrame()
            if not df_d.empty:
                df_d['Date_Obj'] = pd.to_datetime(df_d['Date'], errors='coerce')
                bought_today = df_d[
                    (df_d['Action'] == 'BUY') & 
                    (df_d['Date_Obj'].dt.date == selected_date)
                ]

            # 2. TRADES CLOSED TODAY
            sold_today = pd.DataFrame()
            if not df_s.empty:
                df_s['Closed_Date'] = pd.to_datetime(df_s['Closed_Date'], errors='coerce')
                sold_today = df_s[
                    (df_s['Status'] == 'CLOSED') & 
                    (df_s['Closed_Date'].dt.date == selected_date)
                ]

            # 3. ACTIVE CAMPAIGNS (LIVE STATE)
            open_pos_snapshot = []
            if is_current_report and not df_s.empty:
                current_open = df_s[df_s['Status'] == 'OPEN'].copy()
                
                for _, row in current_open.iterrows():
                    tkr = row['Ticker']
                    try:
                        live_p = yf.Ticker(tkr).history(period='1d')['Close'].iloc[-1]
                    except: 
                        if row['Shares'] > 0:
                            live_p = (row['Total_Cost'] + row['Unrealized_PL']) / row['Shares']
                        else: live_p = row['Avg_Entry']
                    
                    mkt_val = row['Shares'] * live_p
                    unreal = mkt_val - row['Total_Cost']
                    ret = (unreal / row['Total_Cost']) * 100 if row['Total_Cost'] else 0
                    
                    open_pos_snapshot.append({
                        'Ticker': tkr,
                        'Price': live_p,
                        'Mkt_Value': mkt_val,
                        'Unreal_PL': unreal,
                        'Return': ret
                    })
                
            snapshot_df = pd.DataFrame(open_pos_snapshot)
            if not snapshot_df.empty:
                snapshot_df = snapshot_df.sort_values('Return', ascending=False)

            # --- 4. RISK PROTOCOL ---
            RESET_DATE = pd.Timestamp("2026-02-24")
            hist_slice = df_j[df_j['Day'] <= pd.Timestamp(selected_date)].sort_values('Day')
            hist_slice_post = hist_slice[hist_slice['Day'] >= RESET_DATE]
            
            risk_msg = "⚪ NO DATA"
            dd_pct = 0.0
            
            if not hist_slice_post.empty:
                curr_nlv = hist_slice_post['End NLV'].iloc[-1]
                peak_nlv = hist_slice_post['End NLV'].max()
                dd_pct = ((curr_nlv - peak_nlv) / peak_nlv) * 100 if peak_nlv > 0 else 0.0
                
                if dd_pct >= -5: risk_msg = "🟢 GREEN LIGHT"
                elif -5 > dd_pct >= -7: risk_msg = "🟡 YELLOW LIGHT"
                elif -7 > dd_pct >= -10: risk_msg = "🟠 ORANGE LIGHT"
                else: risk_msg = "🔴 RED LIGHT"

            # --- GENERATE MARKDOWN ---
            nlv = day_stats['End NLV']
            day_dol = day_stats['Daily $ Change']
            prev_adj = day_stats['Beg NLV'] + day_stats['Cash -/+']
            day_pct = (day_dol / prev_adj * 100) if prev_adj != 0 else 0.0
            
            report = f"""
# 📜 DAILY TRADING RECORD
**Date:** {selected_date.strftime('%A, %B %d, %Y')}
**Account:** {CURR_PORT_NAME}
**Net Liquidity:** ${nlv:,.2f}

---

### 1. 🌍 STATE OF THE MARKET (M-FACTOR)
**Trend Status:** {m_factor_str}
**Daily Action:**
* **SPY:** {spy_chg_str}
* **NASDAQ:** {ndx_chg_str}

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

---

### 4. 📈 TRADES OPENED TODAY
"""
            if not bought_today.empty:
                report += "| Ticker | Shares | Price | Value | Strategy |\n| :--- | :--- | :--- | :--- | :--- |\n"
                for _, row in bought_today.iterrows():
                    report += f"| **{row['Ticker']}** | {int(row['Shares'])} | ${row['Amount']:.2f} | ${row['Value']:,.2f} | {row.get('Rule', '')} |\n"
            else:
                report += "*No new positions opened today.*\n"

            report += "\n---\n\n### 5. 📉 TRADES CLOSED TODAY\n"
            if not sold_today.empty:
                report += "| Ticker | Result P&L | Return % | Reason |\n| :--- | :--- | :--- | :--- |\n"
                for _, row in sold_today.iterrows():
                    report += f"| **{row['Ticker']}** | ${row['Realized_PL']:,.2f} | {row['Return_Pct']:.2f}% | {row.get('Sell_Rule', '')} |\n"
            else:
                report += "*No positions closed today.*\n"

            report += "\n---\n\n### 6. ⚔️ ACTIVE CAMPAIGNS\n"
            
            if is_current_report:
                if not snapshot_df.empty:
                    report += "| Ticker | Close Price | Market Value | Open P&L | Return % |\n| :--- | :--- | :--- | :--- | :--- |\n"
                    total_exp = 0
                    for _, row in snapshot_df.iterrows():
                        exp_pct = (row['Mkt_Value'] / nlv) * 100 if nlv != 0 else 0
                        report += f"| **{row['Ticker']}** | ${row['Price']:.2f} | ${row['Mkt_Value']:,.2f} | ${row['Unreal_PL']:,.2f} | {row['Return']:.2f}% |\n"
                        total_exp += exp_pct
                    report += f"\n**Total Exposure:** {total_exp:.1f}%"
                else:
                    report += "**100% CASH** (or no open trades found)\n"
            else:
                 report += "*(Historical snapshot not available for past dates. View Trade Log for details.)*"

            # --- RENDER ---
            c_view, c_copy = st.columns([2, 1])
            with c_view:
                st.markdown("### 👁️ Report Preview")
                with st.container(border=True):
                    st.markdown(report)
            
            with c_copy:
                st.markdown("### 📋 Copy for Records")
                st.text_area("Raw Text", report, height=400)
            
        else:
            st.info("No journal entries found.")
    else:
        st.info("No journal data available. Please log your first trading day.")

# ==============================================================================
# PAGE 12: WEEKLY RETRO (OPTIMIZED WORKFLOW)
# ==============================================================================
elif page == "Weekly Retro":
    st.header(f"WEEKLY PROCESS REVIEW ({CURR_PORT_NAME})")
    
    # 1. LOAD DETAILS DATA
    if os.path.exists(DETAILS_FILE):
        df_d = load_data(DETAILS_FILE)
        
        # --- ENSURE COLUMNS EXIST ---
        if 'Exec_Grade' not in df_d.columns: df_d['Exec_Grade'] = None
        if 'Behavior_Tag' not in df_d.columns: df_d['Behavior_Tag'] = None
        if 'Retro_Notes' not in df_d.columns: df_d['Retro_Notes'] = ""

        # 2. WEEK SELECTOR
        st.subheader("1. Select Week to Review")
        
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        
        c1, c2 = st.columns(2)
        sel_date = c1.date_input("Select any day in the target week", start_of_week)
        
        # Calculate Range
        monday = sel_date - timedelta(days=sel_date.weekday())
        friday = monday + timedelta(days=4)
        sunday = monday + timedelta(days=6)
        
        c2.info(f"Reviewing Trading Week: **{monday.strftime('%m-%d')}** through **{friday.strftime('%m-%d')}**")

        # 3. FILTER TRANSACTIONS
        df_d['Date_DT'] = pd.to_datetime(df_d['Date'], errors='coerce')
        
        mask = (df_d['Date_DT'] >= pd.Timestamp(monday)) & (df_d['Date_DT'] <= pd.Timestamp(sunday) + pd.Timedelta(days=1))
        week_df = df_d[mask].copy()
        
        if not week_df.empty:
            # Sort chronologically
            week_df = week_df.sort_values('Date_DT', ascending=True)
            
            # --- 4. ENHANCED METRICS ---
            st.markdown("---")
            st.subheader("2. Activity Monitor")
            
            total_tx = len(week_df)
            unique_tickers = week_df['Ticker'].nunique()
            
            # Logic to split Buys vs Adds
            # We look at the MASTER dataframe to see if a buy was the first for that ID
            def classify_buy_type(row):
                if row['Action'] != 'BUY': return 'N/A'
                # Find all txs for this Trade_ID in master DF
                all_txs = df_d[df_d['Trade_ID'] == row['Trade_ID']].sort_values('Date_DT')
                if all_txs.empty: return "New"
                # If this row is the very first one, it's a New Buy
                if row.name == all_txs.iloc[0].name: return "New"
                return "Add"

            week_df['Buy_Type'] = week_df.apply(classify_buy_type, axis=1)
            
            cnt_new = len(week_df[week_df['Buy_Type'] == 'New'])
            cnt_add = len(week_df[week_df['Buy_Type'] == 'Add'])
            cnt_sells = len(week_df[week_df['Action'] == 'SELL'])
            
            ACTIVITY_THRESHOLD = 15 
            is_overactive = total_tx > ACTIVITY_THRESHOLD
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Tickets", total_tx, delta="High" if is_overactive else "Normal", delta_color="inverse" if is_overactive else "off")
            k2.metric("Unique Tickers", unique_tickers)
            k3.metric("New Positions", cnt_new, f"+ {cnt_add} Adds")
            k4.metric("Sells / Trims", cnt_sells)
            
            if is_overactive:
                st.warning(f"⚠️ **Over-Trading Alert:** {total_tx} trades > {ACTIVITY_THRESHOLD}. Check impulse control.")
            
            # --- 5. THE GRADER INTERFACE ---
            st.markdown("---")
            st.subheader("3. Execution Grading")
            
            # TICKER FILTER (For efficient Charting)
            week_tickers = sorted(week_df['Ticker'].unique().tolist())
            view_ticker = st.selectbox("🔍 Filter by Ticker (Focus Mode)", ["All"] + week_tickers)
            
            if view_ticker != "All":
                display_df = week_df[week_df['Ticker'] == view_ticker].copy()
            else:
                display_df = week_df.copy()
            
            # Format Date for Display
            display_df['Display_Date'] = display_df['Date_DT'].dt.strftime('%Y-%m-%d')
            
            # Ensure Rule/Trx exists
            if 'Rule' not in display_df.columns: display_df['Rule'] = "N/A"
            if 'Trx_ID' not in display_df.columns: display_df['Trx_ID'] = ""

            # Define Columns
            cols_to_show = ['Display_Date', 'Trx_ID', 'Ticker', 'Action', 'Rule', 'Shares', 'Amount', 'Exec_Grade', 'Behavior_Tag', 'Retro_Notes']
            
            # EDITOR
            edited_week = st.data_editor(
                display_df[cols_to_show],
                column_config={
                    "Display_Date": st.column_config.TextColumn("Date", disabled=True, width="small"),
                    "Trx_ID": st.column_config.TextColumn("Trx ID", disabled=True, width="small"),
                    "Ticker": st.column_config.TextColumn("Ticker", disabled=True, width="small"),
                    "Action": st.column_config.TextColumn("Side", disabled=True, width="small"),
                    "Rule": st.column_config.TextColumn("Strategy Rule", disabled=True, width="medium"),
                    "Shares": st.column_config.NumberColumn("Qty", disabled=True, width="small"),
                    "Amount": st.column_config.NumberColumn("Price", format="$%.2f", disabled=True, width="small"),
                    
                    # EDITABLE FIELDS
                    "Exec_Grade": st.column_config.SelectboxColumn(
                        "Grade", 
                        options=["A (Perfect)", "B (Good)", "C (Sloppy)", "D (Bad)", "F (Impulse)"],
                        width="medium",
                        required=True
                    ),
                    "Behavior_Tag": st.column_config.SelectboxColumn(
                        "Behavior",
                        options=["✅ Followed Plan", "🚀 FOMO Entry", "🔪 Caught Knife", "🛑 Late Stop", "😴 Hesitated", "🤷‍♂️ Boredom Trade", "🤏 Sized Too Big", "👻 Revenge Trade", "📉 Panic Sell"],
                        width="medium"
                    ),
                    "Retro_Notes": st.column_config.TextColumn("Analysis / Lesson", width="large")
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
            # --- 6. SAVE LOGIC ---
            if st.button("💾 Save Weekly Review", type="primary"):
                changes_count = 0
                for idx, row in edited_week.iterrows():
                    # Check if changed
                    old_grade = df_d.at[idx, 'Exec_Grade']
                    new_grade = row['Exec_Grade']
                    
                    old_note = df_d.at[idx, 'Retro_Notes']
                    new_note = row['Retro_Notes']
                    
                    old_beh = df_d.at[idx, 'Behavior_Tag']
                    new_beh = row['Behavior_Tag']

                    # Update Master DF
                    if old_grade != new_grade or old_note != new_note or old_beh != new_beh:
                        df_d.at[idx, 'Exec_Grade'] = new_grade
                        df_d.at[idx, 'Behavior_Tag'] = new_beh
                        df_d.at[idx, 'Retro_Notes'] = new_note
                        changes_count += 1
                
                if changes_count > 0:
                    secure_save(df_d, DETAILS_FILE)
                    st.success(f"✅ Saved {changes_count} updates!")
                    st.rerun()
                else:
                    st.info("No changes to save.")

            # --- 7. WEEKLY REPORT CARD ---
            # We calculate this on the FULL week (week_df), not just the filtered view
            # But we need the edited values for live updates.
            # Merging edits back into week_df for stat calculation
            
            # Simple approach: If user saves, page reloads and df_d is fresh.
            # If user hasn't saved, stats are based on old data. This is standard streamlit behavior.
            
            valid_grades = week_df['Exec_Grade'].dropna()
            if not valid_grades.empty:
                points_map = {"A (Perfect)":4, "B (Good)":3, "C (Sloppy)":2, "D (Bad)":1, "F (Impulse)":0}
                total_pts = 0
                count = 0
                f_counts = 0
                
                for g in valid_grades:
                    if g in points_map:
                        total_pts += points_map[g]
                        count += 1
                        if "F" in g: f_counts += 1
                
                if count > 0:
                    gpa = total_pts / count
                    st.markdown("---")
                    st.subheader("4. Report Card")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Execution GPA", f"{gpa:.2f} / 4.0")
                    c2.metric("Impulse Trades (F)", f_counts, delta="Danger" if f_counts > 0 else "Clean", delta_color="inverse")
                    
                    if gpa >= 3.5: st.success("🌟 **Elite Discipline:** You traded professionally this week.")
                    elif gpa >= 2.5: st.warning("⚠️ **Mixed Bag:** Tighten up your process.")
                    else: st.error("🛑 **Tilt Warning:** Reduce size immediately.")
        else:
            st.info("No transactions found for this week.")
    else:
        st.error("Details file not found.")

# ==============================================================================
# PAGE 13: IBD MARKET SCHOOL
# ==============================================================================
elif page == "IBD Market School":
    st.title("📚 IBD MARKET SCHOOL - Market Timing Signals")
    st.caption("Track buy/sell signals and recommended exposure for Nasdaq and SPY")

    # Import market_school_rules
    try:
        from market_school_rules import MarketSchoolRules
    except ImportError:
        st.error("market_school_rules.py not found. Please ensure file is in project root.")
        st.stop()

    from datetime import datetime, timedelta
    import time

    # === HELPER FUNCTIONS ===

    @st.cache_data(ttl=3600, show_spinner=False)
    def analyze_symbol(symbol, start_date, end_date):
        """Analyze market signals for a symbol. Returns list of daily summaries."""
        try:
            analyzer = MarketSchoolRules(symbol)
            analyzer.fetch_data(start_date=start_date, end_date=end_date)

            # Debug: Check data was fetched
            if analyzer.data is None or analyzer.data.empty:
                st.error(f"{symbol}: No data fetched from yfinance")
                return []

            st.info(f"{symbol}: Fetched {len(analyzer.data)} days of data")

            analyzer.analyze_market()
            st.info(f"{symbol}: Generated {len(analyzer.signals)} total signals")

            summaries = []
            dates_to_process = analyzer.data.index[260:]
            st.info(f"{symbol}: Processing {len(dates_to_process)} days (after 260-day lookback)")

            for date in dates_to_process:
                date_str = date.strftime('%Y-%m-%d')
                summary = analyzer.get_daily_summary(date_str)

                # Parse signals for this date (normalize to date-only for comparison)
                date_normalized = pd.Timestamp(date).normalize()
                day_signals = [s for s in analyzer.signals if pd.Timestamp(s.date).normalize() == date_normalized]
                buy_sigs = [s.signal_type.name for s in day_signals if s.signal_type.name.startswith('B')]
                sell_sigs = [s.signal_type.name for s in day_signals if s.signal_type.name.startswith('S')]

                summary['buy_signals'] = ','.join(buy_sigs) if buy_sigs else None
                summary['sell_signals'] = ','.join(sell_sigs) if sell_sigs else None
                summary['symbol'] = symbol
                summaries.append(summary)

            return summaries
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {e}")
            import traceback
            st.code(traceback.format_exc())
            return []

    def get_active_distribution_days(symbol):
        """Get currently active distribution days for a symbol."""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            # Use same date range as sync to ensure proper analysis (need full year+ of data)
            fetch_start = "2024-02-24"

            analyzer = MarketSchoolRules(symbol)
            analyzer.fetch_data(start_date=fetch_start, end_date=end_date)

            if analyzer.data is None or analyzer.data.empty:
                return []

            analyzer.analyze_market()

            # Filter to only active distribution days (not removed)
            active_dist_days = [
                dd for dd in analyzer.distribution_days
                if dd.removed_date is None
            ]

            return active_dist_days
        except Exception as e:
            st.error(f"Error loading distribution days for {symbol}: {e}")
            return []

    def calculate_dd_changes(symbol, start_date, end_date):
        """Calculate daily distribution day additions, removals, and notes."""
        try:
            analyzer = MarketSchoolRules(symbol)
            analyzer.fetch_data(start_date="2024-02-24", end_date=end_date)

            if analyzer.data is None or analyzer.data.empty:
                return {}

            analyzer.analyze_market()

            # Build a dict of changes by date
            changes_by_date = {}

            # Track additions (when distribution day was created)
            for dd in analyzer.distribution_days:
                dd_date = pd.Timestamp(dd.date).normalize()
                date_str = dd_date.strftime('%Y-%m-%d')

                if date_str not in changes_by_date:
                    changes_by_date[date_str] = {'added': [], 'removed': []}

                changes_by_date[date_str]['added'].append({
                    'date': dd_date,
                    'type': dd.type,
                    'loss': dd.loss_percent
                })

                # Track removals (when distribution day was removed)
                if dd.removed_date:
                    removed_date = pd.Timestamp(dd.removed_date).normalize()
                    removed_str = removed_date.strftime('%Y-%m-%d')

                    if removed_str not in changes_by_date:
                        changes_by_date[removed_str] = {'added': [], 'removed': []}

                    changes_by_date[removed_str]['removed'].append({
                        'date': dd_date,
                        'type': dd.type,
                        'reason': dd.removal_reason or 'Unknown'
                    })

            return changes_by_date
        except Exception as e:
            return {}

    def sync_signals_to_db(symbol, summaries, filter_from_date=None):
        """Store analysis results in database."""
        if not USE_DATABASE:
            return 0

        saved_count = 0
        filtered_summaries = [s for s in summaries
                              if not filter_from_date or str(s['date']) >= str(filter_from_date.date())]

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, summary in enumerate(filtered_summaries):
            signal_dict = {
                'symbol': symbol,
                'signal_date': summary['date'],
                'close_price': float(summary['close']),  # Convert to Python float
                'daily_change_pct': float(summary['daily_change'].rstrip('%')),
                'market_exposure': int(summary['market_exposure']),  # Convert to Python int
                'position_allocation': float(summary['position_allocation'].rstrip('%')) / 100,
                'buy_switch': summary['buy_switch'] == 'ON',
                'distribution_count': int(summary['distribution_count']),  # Convert to Python int
                'above_21ema': bool(summary['above_21ema']),  # Convert numpy.bool to Python bool
                'above_50ma': bool(summary['above_50ma']),  # Convert numpy.bool to Python bool
                'buy_signals': summary.get('buy_signals'),
                'sell_signals': summary.get('sell_signals')
            }

            try:
                db.save_market_signal(signal_dict)
                saved_count += 1
                # Update progress every record
                progress_bar.progress((idx + 1) / len(filtered_summaries))
                status_text.text(f"💾 Saving {symbol}: {saved_count}/{len(filtered_summaries)} records")
            except Exception as e:
                st.warning(f"Failed to save {symbol} {summary['date']}: {e}")

        progress_bar.empty()
        status_text.empty()
        return saved_count

    # === DATA REFRESH CONTROLS ===

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])

    with col_btn1:
        if st.button("🔄 Refresh Market Data"):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()

    with col_btn2:
        if st.button("💾 Sync to Database") and USE_DATABASE:
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')

                # Check what dates we already have in the database
                nasdaq_latest_date = db.get_latest_signal_date("^IXIC")
                spy_latest_date = db.get_latest_signal_date("SPY")

                # Determine fetch strategy
                if nasdaq_latest_date is None:
                    # Initial sync: fetch from Feb 2024 for 260-day lookback, save from Feb 2025
                    nasdaq_fetch_start = "2024-02-24"
                    nasdaq_save_from = pd.Timestamp("2025-02-24")
                    st.info("📥 Initial Nasdaq sync from Feb 24, 2025")
                else:
                    # Daily update: fetch from 30 days before latest to ensure continuity
                    nasdaq_fetch_start = (nasdaq_latest_date - timedelta(days=30)).strftime('%Y-%m-%d')
                    nasdaq_save_from = pd.Timestamp(nasdaq_latest_date) + timedelta(days=1)
                    st.info(f"🔄 Updating Nasdaq from {nasdaq_save_from.date()}")

                if spy_latest_date is None:
                    spy_fetch_start = "2024-02-24"
                    spy_save_from = pd.Timestamp("2025-02-24")
                    st.info("📥 Initial SPY sync from Feb 24, 2025")
                else:
                    spy_fetch_start = (spy_latest_date - timedelta(days=30)).strftime('%Y-%m-%d')
                    spy_save_from = pd.Timestamp(spy_latest_date) + timedelta(days=1)
                    st.info(f"🔄 Updating SPY from {spy_save_from.date()}")

                # Nasdaq sync
                with st.spinner("📊 Analyzing Nasdaq..."):
                    nasdaq_summaries = analyze_symbol("^IXIC", nasdaq_fetch_start, end_date)

                nasdaq_saved = sync_signals_to_db("^IXIC", nasdaq_summaries, filter_from_date=nasdaq_save_from)
                st.success(f"✅ Saved {nasdaq_saved} Nasdaq records")

                # SPY sync
                with st.spinner("📈 Analyzing SPY..."):
                    spy_summaries = analyze_symbol("SPY", spy_fetch_start, end_date)

                spy_saved = sync_signals_to_db("SPY", spy_summaries, filter_from_date=spy_save_from)
                st.success(f"✅ Saved {spy_saved} SPY records")

                st.success(f"🎉 Sync complete! Total: {nasdaq_saved + spy_saved} new records")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Sync failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")

    # === LOAD DATA ===

    if USE_DATABASE:
        df_signals = db.load_market_signals(days=90)

        if df_signals.empty:
            st.warning("📭 No market signals in database. Click 'Sync to Database' to populate.")
            st.stop()

        nasdaq_latest = df_signals[df_signals['symbol'] == '^IXIC'].iloc[0] if not df_signals[df_signals['symbol'] == '^IXIC'].empty else None
        spy_latest = df_signals[df_signals['symbol'] == 'SPY'].iloc[0] if not df_signals[df_signals['symbol'] == 'SPY'].empty else None
    else:
        # On-the-fly analysis (no database)
        end_date = datetime.now().strftime('%Y-%m-%d')
        fetch_start = "2024-02-24"  # 1 year for 260-day lookback

        nasdaq_summaries = analyze_symbol("^IXIC", fetch_start, end_date)
        spy_summaries = analyze_symbol("SPY", fetch_start, end_date)

        nasdaq_latest = nasdaq_summaries[-1] if nasdaq_summaries else None
        spy_latest = spy_summaries[-1] if spy_summaries else None

    # === CURRENT STATUS DISPLAY ===

    st.subheader("📊 Current Market Status")

    col_nasdaq, col_spy = st.columns(2)

    with col_nasdaq:
        st.markdown("### 🟦 NASDAQ (^IXIC)")

        if nasdaq_latest is not None:
            if USE_DATABASE:
                close = nasdaq_latest['close_price']
                daily_chg = nasdaq_latest['daily_change_pct']
                exposure = nasdaq_latest['market_exposure']
                allocation = nasdaq_latest['position_allocation'] * 100
                dist_count = nasdaq_latest['distribution_count']
                buy_switch = nasdaq_latest['buy_switch']
                buy_sigs = nasdaq_latest['buy_signals']
                sell_sigs = nasdaq_latest['sell_signals']
            else:
                close = nasdaq_latest['close']
                daily_chg = float(nasdaq_latest['daily_change'].rstrip('%'))
                exposure = nasdaq_latest['market_exposure']
                allocation = float(nasdaq_latest['position_allocation'].rstrip('%'))
                dist_count = nasdaq_latest['distribution_count']
                buy_switch = nasdaq_latest['buy_switch'] == 'ON'
                buy_sigs = nasdaq_latest.get('buy_signals')
                sell_sigs = nasdaq_latest.get('sell_signals')

            m1, m2 = st.columns(2)
            m1.metric("Close", f"${close:,.2f}", f"{daily_chg:+.2f}%")
            m2.metric("Buy Switch", "ON ✅" if buy_switch else "OFF ❌")

            m3, m4 = st.columns(2)
            m3.metric("Exposure Level", f"{exposure}/6", f"{allocation:.0f}% allocation")
            m4.metric("Distribution Days", dist_count)

            if buy_sigs or sell_sigs:
                st.markdown("**Signals Today:**")
                if buy_sigs:
                    st.success(f"🟢 BUY: {buy_sigs}")
                if sell_sigs:
                    st.error(f"🔴 SELL: {sell_sigs}")
            else:
                st.info("No new signals today")

            # Distribution Days Detail
            with st.expander(f"📋 Distribution Days Detail ({dist_count} active)"):
                nasdaq_dist_days = get_active_distribution_days("^IXIC")
                if nasdaq_dist_days:
                    st.markdown("**Active Distribution Days:**")
                    for dd in sorted(nasdaq_dist_days, key=lambda x: x.date, reverse=True):
                        days_ago = (get_current_date_ct() - dd.date.date()).days
                        days_until_expire = 25 - days_ago

                        st.markdown(f"""
                        **{dd.date.strftime('%Y-%m-%d')}** ({days_ago} days ago)
                        - Type: {dd.type.upper()}
                        - Loss: {dd.loss_percent:.2f}%
                        - Expires in: {days_until_expire} days (if not removed earlier)
                        """)
                else:
                    st.info("No active distribution days")
        else:
            st.warning("No data available")

    with col_spy:
        st.markdown("### 🟩 S&P 500 (SPY)")

        if spy_latest is not None:
            if USE_DATABASE:
                close = spy_latest['close_price']
                daily_chg = spy_latest['daily_change_pct']
                exposure = spy_latest['market_exposure']
                allocation = spy_latest['position_allocation'] * 100
                dist_count = spy_latest['distribution_count']
                buy_switch = spy_latest['buy_switch']
                buy_sigs = spy_latest['buy_signals']
                sell_sigs = spy_latest['sell_signals']
            else:
                close = spy_latest['close']
                daily_chg = float(spy_latest['daily_change'].rstrip('%'))
                exposure = spy_latest['market_exposure']
                allocation = float(spy_latest['position_allocation'].rstrip('%'))
                dist_count = spy_latest['distribution_count']
                buy_switch = spy_latest['buy_switch'] == 'ON'
                buy_sigs = spy_latest.get('buy_signals')
                sell_sigs = spy_latest.get('sell_signals')

            m1, m2 = st.columns(2)
            m1.metric("Close", f"${close:,.2f}", f"{daily_chg:+.2f}%")
            m2.metric("Buy Switch", "ON ✅" if buy_switch else "OFF ❌")

            m3, m4 = st.columns(2)
            m3.metric("Exposure Level", f"{exposure}/6", f"{allocation:.0f}% allocation")
            m4.metric("Distribution Days", dist_count)

            if buy_sigs or sell_sigs:
                st.markdown("**Signals Today:**")
                if buy_sigs:
                    st.success(f"🟢 BUY: {buy_sigs}")
                if sell_sigs:
                    st.error(f"🔴 SELL: {sell_sigs}")
            else:
                st.info("No new signals today")

            # Distribution Days Detail
            with st.expander(f"📋 Distribution Days Detail ({dist_count} active)"):
                spy_dist_days = get_active_distribution_days("SPY")
                if spy_dist_days:
                    st.markdown("**Active Distribution Days:**")
                    for dd in sorted(spy_dist_days, key=lambda x: x.date, reverse=True):
                        days_ago = (get_current_date_ct() - dd.date.date()).days
                        days_until_expire = 25 - days_ago

                        st.markdown(f"""
                        **{dd.date.strftime('%Y-%m-%d')}** ({days_ago} days ago)
                        - Type: {dd.type.upper()}
                        - Loss: {dd.loss_percent:.2f}%
                        - Expires in: {days_until_expire} days (if not removed earlier)
                        """)
                else:
                    st.info("No active distribution days")
        else:
            st.warning("No data available")

    st.markdown("---")

    # === HISTORICAL VIEW ===

    st.subheader("📈 Historical Signal Tracking (Last 30 Days)")

    if USE_DATABASE and not df_signals.empty:
        cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=30))
        df_30d = df_signals[df_signals['signal_date'] >= cutoff_date].copy()

        if not df_30d.empty:
            tab1, tab2 = st.tabs(["Exposure Levels", "Signal History"])

            with tab1:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 5))

                nasdaq_hist = df_30d[df_30d['symbol'] == '^IXIC'].sort_values('signal_date')
                spy_hist = df_30d[df_30d['symbol'] == 'SPY'].sort_values('signal_date')

                ax.plot(nasdaq_hist['signal_date'], nasdaq_hist['market_exposure'],
                       marker='o', label='NASDAQ', color='blue', linewidth=2)
                ax.plot(spy_hist['signal_date'], spy_hist['market_exposure'],
                       marker='s', label='SPY', color='green', linewidth=2)

                ax.set_xlabel('Date')
                ax.set_ylabel('Exposure Level (0-6)')
                ax.set_title('Market Exposure Trend')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.5, 6.5)

                st.pyplot(fig)

            with tab2:
                # Add filter for symbol
                filter_col1, filter_col2 = st.columns([1, 3])
                with filter_col1:
                    symbol_filter = st.radio(
                        "Filter by Index:",
                        options=["Both", "NASDAQ (^IXIC)", "SPY"],
                        horizontal=False
                    )

                # Apply filter
                if symbol_filter == "NASDAQ (^IXIC)":
                    filtered_df = df_30d[df_30d['symbol'] == '^IXIC'].copy()
                    symbols_to_analyze = ['^IXIC']
                elif symbol_filter == "SPY":
                    filtered_df = df_30d[df_30d['symbol'] == 'SPY'].copy()
                    symbols_to_analyze = ['SPY']
                else:
                    filtered_df = df_30d.copy()
                    symbols_to_analyze = ['^IXIC', 'SPY']

                # Get DD changes for filtered symbols
                all_dd_changes = {}
                for sym in symbols_to_analyze:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                    dd_changes = calculate_dd_changes(sym, start_date, end_date)
                    for date_str, changes in dd_changes.items():
                        if date_str not in all_dd_changes:
                            all_dd_changes[date_str] = {}
                        all_dd_changes[date_str][sym] = changes

                # Build enhanced display dataframe
                display_df = filtered_df[['signal_date', 'symbol', 'close_price', 'daily_change_pct',
                                    'distribution_count', 'market_exposure', 'buy_signals', 'sell_signals']].copy()

                # Add DD+, DD-, and Notes columns
                display_df['dd_added'] = 0
                display_df['dd_removed'] = 0
                display_df['notes'] = ''

                for idx, row in display_df.iterrows():
                    date_str = row['signal_date'].strftime('%Y-%m-%d')
                    symbol = row['symbol']

                    if date_str in all_dd_changes and symbol in all_dd_changes[date_str]:
                        changes = all_dd_changes[date_str][symbol]

                        # Count additions and removals
                        display_df.at[idx, 'dd_added'] = len(changes.get('added', []))
                        display_df.at[idx, 'dd_removed'] = -len(changes.get('removed', []))

                        # Build notes
                        notes = []
                        for removed in changes.get('removed', []):
                            dd_date = removed['date'].strftime('%Y-%m-%d')
                            reason = removed['reason']
                            notes.append(f"DD {dd_date} removed ({reason})")

                        display_df.at[idx, 'notes'] = ' | '.join(notes) if notes else ''

                # Reorder and rename columns
                display_df = display_df[['signal_date', 'symbol', 'close_price', 'daily_change_pct',
                                         'dd_added', 'dd_removed', 'distribution_count', 'market_exposure',
                                         'buy_signals', 'sell_signals', 'notes']]
                display_df.columns = ['Date', 'Symbol', 'Close', 'Daily %', 'DD+', 'DD-', 'Cum DD',
                                     'Exposure', 'Buy Signals', 'Sell Signals', 'Notes']
                display_df = display_df.sort_values('Date', ascending=False)

                with filter_col2:
                    st.dataframe(display_df, use_container_width=True, height=400)

    # === SIGNAL LEGEND ===

    with st.expander("📖 Signal Reference Guide"):
        col_buy, col_sell = st.columns(2)

        with col_buy:
            st.markdown("**🟢 BUY SIGNALS**")
            st.markdown("""
            - **B1**: Follow-Through Day (FTD)
            - **B2**: Additional FTD
            - **B3**: Low Above 21-day MA
            - **B4**: Trending Above 21-day MA
            - **B5**: Living Above 21-day MA
            - **B6**: Low Above 50-day MA
            - **B7**: Accumulation Day
            - **B8**: Higher High
            - **B9**: Downside Reversal Buyback
            - **B10**: Distribution Day Fall Off
            """)

        with col_sell:
            st.markdown("**🔴 SELL SIGNALS**")
            st.markdown("""
            - **S1**: FTD Undercut
            - **S2**: Failed Rally
            - **S3**: Full Distribution -1
            - **S4**: Full Distribution
            - **S5**: Break Below 21-day MA
            - **S9**: Break Below 50-day MA
            - **S10**: Bad Break
            - **S11**: Downside Reversal Day
            - **S12**: Lower Low
            - **S13**: Distribution Cluster
            """)
