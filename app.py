import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math
from datetime import date, datetime, time, timedelta
import os
import shutil

# Helper function to get current date in Central Time
def get_current_date_ct():
    """Get current date in US Central Time (Chicago)."""
    from zoneinfo import ZoneInfo
    ct_now = datetime.now(ZoneInfo("America/Chicago"))
    return ct_now.date()

def get_current_time_ct():
    """Get current time in US Central Time (Chicago)."""
    from zoneinfo import ZoneInfo
    ct_now = datetime.now(ZoneInfo("America/Chicago"))
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

# Vision API (MarketSurge screenshot extraction)
try:
    import vision_extract
    _VISION_IMPORT = True
except (ImportError, Exception) as e:
    _VISION_IMPORT = False
    print(f"⚠️  vision_extract import failed: {type(e).__name__}: {e}")

def check_vision_available():
    """Lazy check — runs after st.secrets is fully loaded."""
    if not _VISION_IMPORT:
        return False
    try:
        api_key = st.secrets.get("anthropic", {}).get("api_key", "")
        return bool(api_key)
    except Exception:
        return False

# Feature flag: Use database instead of CSV
# Auto-enable if running on Streamlit Cloud with database secrets
if DB_AVAILABLE and hasattr(st, 'secrets') and 'database' in st.secrets:
    USE_DATABASE = True  # Running on Streamlit Cloud with database configured
    print("✅ Database mode enabled (Streamlit Cloud)")
else:
    USE_DATABASE = os.getenv('USE_DATABASE', 'false').lower() == 'true' and DB_AVAILABLE
    if USE_DATABASE:
        print("✅ Database mode enabled (environment variable)")

# --- DB MIGRATIONS (safe to run repeatedly) ---
if USE_DATABASE:
    try:
        with db.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("ALTER TABLE trading_journal ADD COLUMN IF NOT EXISTS portfolio_heat NUMERIC(10, 4) DEFAULT 0")
                cur.execute("ALTER TABLE trading_journal ADD COLUMN IF NOT EXISTS spy_atr NUMERIC(10, 4) DEFAULT 0")
                cur.execute("ALTER TABLE trading_journal ADD COLUMN IF NOT EXISTS nasdaq_atr NUMERIC(10, 4) DEFAULT 0")
            conn.commit()
    except Exception as e:
        print(f"⚠️  DB migration note: {e}")

# --- CONFIGURATION ---
st.set_page_config(page_title="CAN SLIM COMMAND CENTER", layout="wide", page_icon="📈")
APP_VERSION = "17.1 (Auth + UI Refresh)"

# =============================================================================
# AUTH — Simple password login gate
# =============================================================================
_AUTH_ACTIVE = False
_app_password = ""
try:
    _app_password = st.secrets.get("app", {}).get("password", "")
except Exception:
    pass

if _app_password:
    # Password is configured — require login
    if not st.session_state.get('authenticated', False):
        # Center the login form
        _spacer1, _login_col, _spacer2 = st.columns([1, 1, 1])
        with _login_col:
            st.markdown("""
            <div style="text-align: center; margin-top: 15vh; margin-bottom: 2rem;">
                <div style="font-size: 2.5rem; font-weight: 800; margin-bottom: 0.25rem;">MO Money</div>
                <div style="font-size: 0.95rem; opacity: 0.5;">Trading Journal & Analytics</div>
            </div>
            """, unsafe_allow_html=True)
            with st.form("login_form"):
                _pwd = st.text_input("Password", type="password", placeholder="Enter password")
                _submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                if _submitted:
                    if _pwd == _app_password:
                        st.session_state['authenticated'] = True
                        st.rerun()
                    else:
                        st.error("Incorrect password")
        st.stop()

    _AUTH_ACTIVE = True

st.session_state['user_name'] = 'MO'

# =============================================================================
# GLOBAL CSS THEME — Injected once, applies to every page
# =============================================================================
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Root Variables (Light Mode) ── */
:root {
    --font-main: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --bg-primary: #f8f9fc;
    --bg-card: #ffffff;
    --bg-sidebar: #1a1d29;
    --text-primary: #1a1d29;
    --text-secondary: #6b7280;
    --text-muted: #9ca3af;
    --border-color: #e5e7eb;
    --border-radius: 12px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.12);
    --accent-blue: #6366f1;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    --transition: all 0.2s ease;
}

/* ── Global Typography ── */
html, body, [class*="css"] {
    font-family: var(--font-main) !important;
}

/* ── Main Content Area ── */
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}

/* ── Page Titles & Headers ── */
h1 {
    font-weight: 800 !important;
    font-size: 1.75rem !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 1.5rem !important;
}

h2 {
    font-weight: 700 !important;
    font-size: 1.35rem !important;
    letter-spacing: -0.01em !important;
}

h3 {
    font-weight: 600 !important;
    font-size: 1.1rem !important;
}

/* ── Streamlit Containers → Card Style ── */
[data-testid="stExpander"] {
    border-radius: var(--border-radius) !important;
    margin-bottom: 0.75rem;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* ── Tabs → Clean Underline Style ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    font-family: var(--font-main) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 1.25rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #6366f1 !important;
    border-bottom-color: #6366f1 !important;
    background: transparent !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #6366f1 !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

/* ── Buttons → Modern Rounded ── */
.stButton > button {
    font-family: var(--font-main) !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s ease !important;
    font-size: 0.875rem !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    color: white !important;
    opacity: 0.9;
}

/* ── Selectbox, Text Input, Number Input → Rounded ── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
.stTextInput > div > div,
.stNumberInput > div > div > div {
    border-radius: 8px !important;
    font-family: var(--font-main) !important;
}

[data-baseweb="select"] > div:focus-within,
.stTextInput > div > div:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* ── DataFrames / Tables → Clean Cards ── */
[data-testid="stDataFrame"],
.stDataFrame {
    border-radius: var(--border-radius) !important;
    overflow: hidden;
}

/* ── Metrics → Subtle Card ── */
[data-testid="stMetric"] {
    border-radius: var(--border-radius);
    padding: 1rem 1.25rem;
}

[data-testid="stMetric"] label {
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-size: 1.5rem !important;
}

/* ── Sidebar Styling (follows Streamlit theme) ── */
[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 0.75rem !important;
}

/* ── Alerts → Rounded & Soft ── */
.stAlert {
    border-radius: var(--border-radius) !important;
    border: none !important;
    font-size: 0.9rem !important;
}

[data-testid="stAlert"] {
    border-radius: var(--border-radius) !important;
}

/* ── Dividers → Subtle ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(128,128,128,0.2) !important;
    margin: 1.5rem 0 !important;
}

/* ── Plotly Charts → Remove Border ── */
.js-plotly-plot {
    border-radius: var(--border-radius) !important;
    overflow: hidden;
}

/* ── Form Containers ── */
[data-testid="stForm"] {
    border-radius: var(--border-radius) !important;
    padding: 1.5rem !important;
}

/* ── Multiselect Tags ── */
[data-baseweb="tag"] {
    border-radius: 6px !important;
    font-family: var(--font-main) !important;
}

/* ── Toggle → Accent Color ── */
[data-testid="stCheckbox"] span[role="checkbox"][aria-checked="true"] {
    background-color: var(--accent-blue) !important;
}

/* ── Smooth Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #a1a1a1;
}

/* ── Custom Card Class (for HTML cards) ── */
.mo-card {
    border-radius: var(--border-radius);
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    transition: var(--transition);
}

.mo-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.mo-card-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.6;
    margin-bottom: 0.5rem;
}

.mo-card-value {
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1.2;
}

.mo-card-sub {
    font-size: 0.85rem;
    opacity: 0.6;
    margin-top: 0.35rem;
}

/* ── Gradient Card (for hero metrics) ── */
.mo-gradient-card {
    padding: 1.25rem;
    border-radius: 12px;
    color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    transition: all 0.2s ease;
}

.mo-gradient-card:hover {
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    transform: translateY(-2px);
}

.mo-gradient-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.9;
}

.mo-gradient-card .value {
    font-size: 2rem;
    font-weight: 800;
    margin: 0.4rem 0;
    line-height: 1.2;
}

.mo-gradient-card .sub {
    font-size: 0.85rem;
    opacity: 0.9;
}

/* ── Status Badges ── */
.mo-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.mo-badge-green { background: #d1fae5; color: #065f46; }
.mo-badge-red { background: #fee2e2; color: #991b1b; }
.mo-badge-amber { background: #fef3c7; color: #92400e; }
.mo-badge-blue { background: #dbeafe; color: #1e40af; }
.mo-badge-purple { background: #ede9fe; color: #5b21b6; }

/* ── Section Accent Borders (color-coded by page category) ── */
.section-trading { border-left: 4px solid var(--accent-blue); padding-left: 1rem; }
.section-risk { border-left: 4px solid var(--accent-red); padding-left: 1rem; }
.section-daily { border-left: 4px solid var(--accent-amber); padding-left: 1rem; }
.section-market { border-left: 4px solid var(--accent-green); padding-left: 1rem; }
.section-analytics { border-left: 4px solid var(--accent-purple); padding-left: 1rem; }

/* ── Dark Mode: no overrides needed — Streamlit theme handles it ── */

/* ── Mobile Breakpoints ── */
@media (max-width: 768px) {
    .main .block-container {
        padding-top: 1rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        padding-bottom: 1rem !important;
    }
    h1 { font-size: 1.4rem !important; margin-bottom: 1rem !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1rem !important; }
    /* Stack metric cards 2-wide instead of overflowing on narrow screens */
    [data-testid="stMetric"] {
        padding: 0.5rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
    }
    /* Make buttons easier to tap */
    .stButton > button {
        min-height: 44px;
    }
    /* Compact dataframe rows on mobile */
    [data-testid="stDataFrame"] {
        font-size: 0.8rem !important;
    }
}

@media (max-width: 480px) {
    [data-testid="stMetricValue"] {
        font-size: 0.95rem !important;
    }
    /* Hide sidebar section labels on very narrow screens — icons still visible */
    h1 { font-size: 1.2rem !important; }
}

</style>
""", unsafe_allow_html=True)

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
    "br1.5 IPO Base", "br1.6 Flat Base", "br1.7 Consolidation Pivot", "br1.8 High Tight Flag",
    # 2. Volume & Volatility
    "br2.1 HVE", "br2.2 HVSI", "br2.3 HV1",
    # 3. Moving Average Reclaims
    "br3.1 Reclaim 21e", "br3.2 Reclaim 50s", "br3.3 Reclaim 200s", "br3.4 Reclaim 10W", "br3.5 Reclaim 8e",
    # 4. Pullbacks
    "br4.1 PB 21e", "br4.2 PB 50s", "br4.3 PB 10w", "br4.4 PB 200s", 
    "br4.5 PB 8e", "br4.6 VWAP",  # <--- UPDATED HERE
    # 5. Reversals
    "br5.1 Undercut & Rally", "br5.2 Upside Reversal", 
    # 6. Gaps
    "br6.1 Gapper", "br6.2 Continuation Gap Up",
    # 7. Strategies
    "br7.1 TQQQ Strategy", "br7.2 New High after Gentle PB", "br7.3 JL Century Mark", 
    # 8. Trendline Breaks
    "br8.1 Daily STL Break", "br8.2 Weekly STL Break", "br8.3 Monthly STL Break", 
    # 9. Moving Average Strategies
    "br9.1 21e Strategy",
    # 10. Hedging
    "br10.1 Hedging with leverage product",
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
    "sr9 Breakout Failure",
    "sr10 Scale-Out T1 (-3%)",
    "sr11 Scale-Out T2 (-5%)",
    "sr12 Scale-Out T3 (-7%)"
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

# --- CACHED R2 IMAGE DOWNLOAD ---
# Images rarely change once uploaded, so cache aggressively (1 hour TTL).
# This prevents re-downloading every time the Trade Journal re-renders.
@st.cache_data(ttl=3600, show_spinner=False, max_entries=200)
def cached_r2_download(url):
    if not url:
        return None
    try:
        return r2.download_image(url)
    except Exception:
        return None

# --- CACHED SINGLE-TICKER LIVE PRICE ---
# yfinance single-ticker fetches are slow (1-2s each). Cache for 60s so
# repeated reruns during form edits hit the cache instead of the network.
@st.cache_data(ttl=60, show_spinner=False, max_entries=500)
def cached_live_price(ticker):
    if not ticker:
        return None
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if hist.empty:
            return None
        return float(hist['Close'].iloc[-1])
    except Exception:
        return None

# --- CACHED BATCH LIVE PRICES ---
# For dashboards that need live prices for all open positions. Cache key is
# a sorted tuple so the order of tickers doesn't matter. 60s TTL.
@st.cache_data(ttl=60, show_spinner=False)
def cached_batch_live_prices(tickers_tuple):
    """Returns dict[ticker -> price]. Pass a tuple (hashable) of tickers."""
    if not tickers_tuple:
        return {}
    tickers = list(tickers_tuple)
    try:
        data = yf.download(tickers, period="1d", progress=False)['Close']
        if len(tickers) == 1:
            val = float(data.iloc[-1]) if not data.empty else None
            return {tickers[0]: val} if val is not None else {}
        last = data.iloc[-1]
        return {t: float(last[t]) for t in tickers if t in last.index and not pd.isna(last[t])}
    except Exception:
        return {}

# --- MARKET STATE HELPERS (used by M Factor + Daily Routine) ---
@st.cache_data(ttl=1800, show_spinner=False)
def get_market_state(ticker):
    """Compute M Factor market state for a given index ticker."""
    try:
        df = yf.Ticker(ticker).history(period="2y")
        if df.empty: return None

        df['21EMA'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['50SMA'] = df['Close'].rolling(window=50).mean()
        df['200SMA'] = df['Close'].rolling(window=200).mean()
        df['Prev_Close'] = df['Close'].shift(1)
        df['Is_Up'] = df['Close'] > df['Prev_Close']

        def pct_diff(a, b): return ((a - b) / b) * 100

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

        subset = df.iloc[-60:].copy()
        state = "OPEN"
        setup_low_21 = None; setup_low_50 = None; pt_streak = 0
        transition_date = subset.index[0]
        prev_state = "OPEN"

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

            if state != prev_state:
                transition_date = date
                prev_state = state

        curr = subset.iloc[-1]
        return {
            'price': curr['Close'], 'state': state,
            'ema21': curr['21EMA'], 'd21': pct_diff(curr['Close'], curr['21EMA']),
            'sma50': curr['50SMA'], 'd50': pct_diff(curr['Close'], curr['50SMA']),
            'sma200': curr['200SMA'], 'd200': pct_diff(curr['Close'], curr['200SMA']),
            's21': streak_21, 's50': streak_50, 's200': streak_200,
            'transition_date': transition_date
        }
    except: return None

def get_combined_market_status():
    """Get combined M Factor market window status from Nasdaq + SPY."""
    nasdaq = get_market_state("^IXIC")
    spy = get_market_state("SPY")
    if not nasdaq or not spy:
        return "Open", None
    ns, ss = nasdaq['state'], spy['state']
    if ns == "POWERTREND" or ss == "POWERTREND":
        status = "Powertrend"
    elif ns == "CLOSED" and ss == "CLOSED":
        status = "Closed"
    elif ns in ["NEUTRAL", "CLOSED"] or ss in ["NEUTRAL", "CLOSED"]:
        status = "Neutral"
    else:
        status = "Open"
    latest_transition = max(nasdaq['transition_date'], spy['transition_date'])
    return status, latest_transition

def compute_portfolio_heat(portfolio_name, equity=None):
    """Compute portfolio heat (ATR-weighted) for current open positions.
    Same formula as Portfolio Heat page: Heat = sum(Weight% * ATR%/100).
    Returns heat as a percentage (e.g., 2.5 means 2.5%).
    """
    try:
        summary_path = os.path.join(DATA_ROOT, portfolio_name, 'Trade_Log_Summary.csv')
        df_s = load_data(summary_path)
        if df_s.empty or 'Status' not in df_s.columns:
            return 0.0

        open_pos = df_s[df_s['Status'].str.strip().str.upper() == 'OPEN'].copy()
        if open_pos.empty:
            return 0.0

        # Use provided equity or try to get from journal
        if equity is None or equity <= 0:
            journal_path = os.path.join(DATA_ROOT, portfolio_name, 'Trading_Journal_Clean.csv')
            df_j = load_data(journal_path)
            if df_j.empty:
                return 0.0
            df_j['End NLV'] = pd.to_numeric(
                df_j['End NLV'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False),
                errors='coerce'
            ).fillna(0)
            equity = df_j['End NLV'].iloc[-1] if not df_j.empty else 0.0
            if equity <= 0:
                return 0.0

        tickers = open_pos['Ticker'].unique().tolist()
        if not tickers:
            return 0.0

        # Batch download price data (40 days for 21-period SMA)
        batch_data = yf.download(tickers, period="40d", interval="1d", progress=False, group_by='ticker')
        if batch_data.empty:
            return 0.0

        total_heat = 0.0
        for ticker in tickers:
            try:
                if len(tickers) > 1:
                    df_t = batch_data[ticker].copy().dropna()
                else:
                    df_t = batch_data.copy().dropna()

                atr_pct = 0.0
                if len(df_t) >= 21:
                    df_t['H-L'] = df_t['High'] - df_t['Low']
                    df_t['H-PC'] = (df_t['High'] - df_t['Close'].shift(1)).abs()
                    df_t['L-PC'] = (df_t['Low'] - df_t['Close'].shift(1)).abs()
                    df_t['TR'] = df_t[['H-L', 'H-PC', 'L-PC']].max(axis=1)
                    sma_tr = df_t['TR'].tail(21).mean()
                    sma_low = df_t['Low'].tail(21).mean()
                    if sma_low > 0:
                        atr_pct = (sma_tr / sma_low) * 100

                row = open_pos[open_pos['Ticker'] == ticker].iloc[0]
                total_cost = float(row.get('Total_Cost', 0))
                weight_pct = (total_cost / equity) * 100 if equity > 0 else 0
                total_heat += weight_pct * (atr_pct / 100)
            except:
                continue

        return float(round(total_heat, 4))
    except:
        return 0.0

def compute_index_atr(ticker, start_date, end_date):
    """Compute daily ATR% (21-period SMA) for an index over a date range.
    Returns dict mapping date -> atr_pct (Python float).
    """
    try:
        # Fetch extra days for the 21-period lookback
        fetch_start = pd.Timestamp(start_date) - pd.Timedelta(days=45)
        fetch_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        df_t = yf.Ticker(ticker).history(start=fetch_start, end=fetch_end)
        if df_t.empty or len(df_t) < 21:
            return {}
        df_t.index = df_t.index.date

        # Compute TR
        df_t['H-L'] = df_t['High'] - df_t['Low']
        df_t['H-PC'] = (df_t['High'] - df_t['Close'].shift(1)).abs()
        df_t['L-PC'] = (df_t['Low'] - df_t['Close'].shift(1)).abs()
        df_t['TR'] = df_t[['H-L', 'H-PC', 'L-PC']].max(axis=1)

        # Rolling 21-period SMA of TR and Low
        df_t['SMA_TR'] = df_t['TR'].rolling(21).mean()
        df_t['SMA_Low'] = df_t['Low'].rolling(21).mean()
        df_t['ATR_Pct'] = (df_t['SMA_TR'] / df_t['SMA_Low']) * 100

        # Filter to requested date range
        start_d = pd.Timestamp(start_date).date()
        end_d = pd.Timestamp(end_date).date()
        result = {}
        for d, row in df_t.iterrows():
            if start_d <= d <= end_d and pd.notna(row['ATR_Pct']):
                result[d] = float(row['ATR_Pct'])
        return result
    except:
        return {}


# --- MARKET CYCLE TRACKER ENGINE (Layer 2 on top of M Factor) ---
# 30-min TTL — NASDAQ daily bars only update at EOD, so intraday recomputation
# is wasted work. The analyzer processes 2y of data which is expensive.
@st.cache_data(ttl=1800, show_spinner=False)
def compute_cycle_state():
    """Compute NASDAQ market cycle state: CORRECTION / RALLY MODE / UPTREND / POWERTREND.
    Uses MarketSchoolRules for correction/reference_high/rally/FTD detection,
    then layers entry/exit ladder on top.
    """
    try:
        from market_school_rules import MarketSchoolRules

        # --- Run IBD Market School analyzer (source of truth) ---
        analyzer = MarketSchoolRules("^IXIC")
        analyzer.fetch_data(start_date="2024-02-24", end_date=datetime.now().strftime('%Y-%m-%d'))
        if analyzer.data is None or analyzer.data.empty:
            return None
        analyzer.analyze_market()

        df = analyzer.data
        reference_high = analyzer.reference_high
        market_in_correction = analyzer.market_in_correction
        rally_start_date = analyzer.rally_start_date
        rally_low = analyzer.rally_low
        rally_low_idx = analyzer.rally_low_idx
        ibd_ftd_date = analyzer.ftd_date
        buy_switch = analyzer.buy_switch

        # --- Reference high: MarketSmith "marked high" logic ---
        # A marked high is the highest high with N bars of lower highs on
        # each side (default period = 9). Scan backwards to find the most
        # recent marked high. This gives a stable, confirmed market peak.
        marked_high_period = 9
        reference_high_date = None
        reference_high = None
        for i in range(len(df) - 1, marked_high_period - 1, -1):
            candidate = df.iloc[i]['High']
            # Need at least marked_high_period bars on each side
            left_start = max(0, i - marked_high_period)
            right_end = min(len(df), i + marked_high_period + 1)
            # Check left side: all bars must have lower highs
            left_ok = all(df.iloc[j]['High'] < candidate for j in range(left_start, i))
            if not left_ok:
                continue
            # Check right side: need enough bars, and all must have lower highs
            if right_end - (i + 1) < marked_high_period:
                continue  # Not enough bars on right side yet (too recent)
            right_ok = all(df.iloc[j]['High'] < candidate for j in range(i + 1, right_end))
            if right_ok:
                reference_high = candidate
                reference_high_date = df.index[i]
                break

        # Fallback to analyzer's reference high if no marked high found
        if reference_high is None:
            reference_high = analyzer.reference_high
            if reference_high is not None:
                for i in range(len(df) - 1, -1, -1):
                    if df.iloc[i]['High'] >= reference_high - 0.01:
                        reference_high_date = df.index[i]
                        break

        # --- Compute MAs for entry ladder (analyzer.data already has some) ---
        df['8EMA'] = df['Close'].ewm(span=8, adjust=False).mean()
        df['21EMA'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['50SMA'] = df['Close'].rolling(window=50).mean()
        df['200SMA'] = df['Close'].rolling(window=200).mean()

        curr = df.iloc[-1]
        last_data_date = df.index[-1]
        price = float(curr['Close'])
        ema8 = float(curr['8EMA']) if pd.notna(curr['8EMA']) else 0
        ema21 = float(curr['21EMA']) if pd.notna(curr['21EMA']) else 0
        sma50 = float(curr['50SMA']) if pd.notna(curr['50SMA']) else 0
        sma200 = float(curr['200SMA']) if pd.notna(curr['200SMA']) else 0

        # --- Drawdown from reference high ---
        drawdown_pct = 0.0
        if reference_high and reference_high > 0:
            drawdown_pct = (price - reference_high) / reference_high * 100

        # --- Streak tracking (over recent data) ---
        low_above_21_streak = 0
        low_above_50_streak = 0
        consecutive_closes_below_21 = 0

        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if pd.notna(row.get('21EMA')):
                if row['Low'] > row['21EMA']:
                    low_above_21_streak += 1
                else:
                    break
            else:
                break
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if pd.notna(row.get('50SMA')):
                if row['Low'] > row['50SMA']:
                    low_above_50_streak += 1
                else:
                    break
            else:
                break
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if pd.notna(row.get('21EMA')):
                if row['Close'] < row['21EMA']:
                    consecutive_closes_below_21 += 1
                else:
                    break
            else:
                break

        # --- Price-only FTD detection (no volume requirement) ---
        price_ftd_date = ibd_ftd_date
        if price_ftd_date is None and rally_low_idx is not None:
            for i in range(rally_low_idx + 3, len(df)):  # Day 4+ (0-indexed: +3)
                row = df.iloc[i]
                if pd.notna(row.get('daily_gain_pct')) and row['daily_gain_pct'] >= 1.0:
                    # Check rally low hasn't been undercut
                    lows = df.iloc[rally_low_idx:i+1]['Low']
                    if lows.min() >= rally_low:
                        price_ftd_date = df.index[i]
                        break

        # --- Determine cycle state ---
        # Rally day count (days since rally start)
        rally_day_idx = rally_low_idx  # Will be updated if rally day is a subsequent day
        days_since_rally = None
        if rally_low_idx is not None:
            days_since_rally = len(df) - 1 - rally_low_idx

        # Correction start: when did the correction begin?
        correction_start = None
        if market_in_correction:
            for i in range(len(df) - 1, -1, -1):
                row = df.iloc[i]
                if reference_high and reference_high > 0:
                    decline = (row['Close'] - reference_high) / reference_high * 100
                    if decline > -7:
                        if i + 1 < len(df):
                            correction_start = df.index[i + 1]
                        break

        # Rally day type classification:
        # - "rally": Close > previous day's close (proper rally day)
        # - "pink": Close < previous day's close BUT closed in upper half of day's range
        # - None: Neither condition met on low day — check next day
        # If the low day doesn't qualify, the next day that closes above
        # the prior day's close becomes the rally day.
        rally_day_type = None
        if rally_start_date is not None and rally_low_idx is not None:
            rd_row = df.iloc[rally_low_idx]
            if rally_low_idx > 0:
                prev_row = df.iloc[rally_low_idx - 1]
                if rd_row['Close'] > prev_row['Close']:
                    rally_day_type = "rally"
                else:
                    # Check if close is in upper half of the day's range
                    day_midpoint = (rd_row['High'] + rd_row['Low']) / 2
                    if rd_row['Close'] >= day_midpoint:
                        rally_day_type = "pink"
                    else:
                        # Low day didn't qualify — check subsequent days
                        for next_i in range(rally_low_idx + 1, len(df)):
                            next_row = df.iloc[next_i]
                            next_prev = df.iloc[next_i - 1]
                            if next_row['Close'] > next_prev['Close']:
                                rally_day_type = "rally"
                                rally_day_idx = next_i
                                rally_start_date = df.index[next_i]
                                break
                            # Also check if subsequent day is pink
                            next_mid = (next_row['High'] + next_row['Low']) / 2
                            if next_row['Close'] >= next_mid:
                                rally_day_type = "pink"
                                rally_day_idx = next_i
                                rally_start_date = df.index[next_i]
                                break

        # Update days_since_rally: Day 1 = rally day itself (IBD numbering)
        if rally_day_idx is not None:
            days_since_rally = len(df) - rally_day_idx

        # --- Entry step calculation ---
        # Steps achieved as a set; each step adds an increment to exposure.
        # Pre-FTD: only steps 0 and 2 can be reached, exposure capped at 40%.
        # FTD unlocks accumulated steps (Step 0 + Step 2 + Step 1 = 60%).
        achieved_steps = set()
        has_rally = not market_in_correction or buy_switch or \
                    (rally_start_date is not None and rally_day_type is not None) or \
                    price_ftd_date is not None

        ftd_achieved = price_ftd_date is not None or (not market_in_correction or buy_switch)

        if has_rally:
            if rally_start_date is not None:
                achieved_steps.add(0)  # Rally day
            if ftd_achieved:
                achieved_steps.add(1)  # FTD
            # Step 2 can be achieved pre-FTD
            if price > ema21:
                achieved_steps.add(2)
            # Steps 3+ require FTD
            if ftd_achieved:
                if curr['Low'] > ema21:
                    achieved_steps.add(3)
                if 3 in achieved_steps and low_above_21_streak >= 3:
                    achieved_steps.add(4)
                if 4 in achieved_steps and low_above_50_streak >= 3:
                    achieved_steps.add(5)
                if 5 in achieved_steps and ema21 > sma50 and ema21 > sma200 and sma50 > sma200:
                    achieved_steps.add(6)
                if 6 in achieved_steps and ema8 > ema21 > sma50 > sma200:
                    achieved_steps.add(7)

        entry_step = max(achieved_steps) if achieved_steps else -1

        # --- Cycle state determination (based on entry step) ---
        if entry_step >= 7:
            cycle_state = "POWERTREND"
        elif entry_step >= 4:
            cycle_state = "UPTREND"
        elif entry_step >= 0:
            cycle_state = "RALLY MODE"
        else:
            cycle_state = "CORRECTION"

        # --- EXIT LADDER ---
        # Scan recent data for violations
        violation_log = []
        ema21_violation_close = None
        ema21_violation_date = None
        sma50_violation_close = None
        sma50_violation_date = None

        lookback = min(120, len(df))
        for i in range(len(df) - lookback, len(df)):
            row = df.iloc[i]
            dt = df.index[i]
            close_val = row['Close']
            low_val = row['Low']
            e21 = row.get('21EMA')
            s50 = row.get('50SMA')

            if pd.notna(e21):
                if close_val < e21 and ema21_violation_close is None:
                    ema21_violation_close = close_val
                    ema21_violation_date = dt
                elif ema21_violation_close is not None and close_val >= e21:
                    ema21_violation_close = None
                    ema21_violation_date = None
                elif ema21_violation_close is not None and low_val < (ema21_violation_close * 0.998):
                    violation_log.append({
                        'date': dt, 'signal': '21 EMA Violation',
                        'price': close_val, 'target': '50%', 'severity': 'WARNING'
                    })
                    ema21_violation_close = None

            if pd.notna(s50):
                if close_val < s50 and sma50_violation_close is None:
                    sma50_violation_close = close_val
                    sma50_violation_date = dt
                elif sma50_violation_close is not None and close_val >= s50:
                    sma50_violation_close = None
                    sma50_violation_date = None
                elif sma50_violation_close is not None and low_val < (sma50_violation_close * 0.998):
                    violation_log.append({
                        'date': dt, 'signal': '50 SMA Violation',
                        'price': close_val, 'target': '0%', 'severity': 'CRITICAL'
                    })
                    sma50_violation_close = None

        # --- BUILD CURRENT EXIT ALERTS ---
        active_exits = []
        if consecutive_closes_below_21 >= 2:
            active_exits.append({
                'signal': '21 EMA Confirmed Break',
                'icon': '📉', 'target': '30%', 'severity': 'SERIOUS',
                'detail': f"Two consecutive closes below 21 EMA. Reduce to 30% exposure."
            })
        elif ema21_violation_close is not None:
            active_exits.append({
                'signal': '21 EMA Violation',
                'icon': '⚠️', 'target': '50%', 'severity': 'WARNING',
                'detail': f"Close below 21 EMA on {ema21_violation_date.strftime('%b %d') if hasattr(ema21_violation_date, 'strftime') else ema21_violation_date}. Monitoring for undercut."
            })
        if sma50_violation_close is not None:
            active_exits.append({
                'signal': '50 SMA Violation',
                'icon': '🔴', 'target': '0%', 'severity': 'CRITICAL',
                'detail': f"Close below 50 SMA. Monitoring for confirmed break."
            })

        # --- ENTRY LADDER STATUS ---
        ftd_date = price_ftd_date
        entry_ladder = [
            {'step': 0, 'label': 'Rally Day', 'exposure': '20%',
             'achieved': 0 in achieved_steps and cycle_state != 'CORRECTION',
             'detail': f"{'Rally Day' if rally_day_type == 'rally' else 'Pink Rally Day'} — {rally_start_date.strftime('%b %d, %Y') if rally_start_date and hasattr(rally_start_date, 'strftime') else 'N/A'}" if rally_start_date else "Waiting for fresh low + reversal after 7%+ correction"},
            {'step': 1, 'label': 'Follow-Through Day', 'exposure': '60%',
             'achieved': 1 in achieved_steps,
             'detail': f"FTD on {ftd_date.strftime('%b %d, %Y') if ftd_date and hasattr(ftd_date, 'strftime') else 'N/A'}" if ftd_date else "Day 4+ of rally, close up 1%+"},
            {'step': 2, 'label': 'Close above 21 EMA', 'exposure': '40%',
             'achieved': 2 in achieved_steps,
             'detail': f"21 EMA at {ema21:,.2f}"},
            {'step': 3, 'label': 'Low above 21 EMA', 'exposure': '80%',
             'achieved': 3 in achieved_steps,
             'detail': "Daily low held above 21 EMA"},
            {'step': 4, 'label': 'Holds 21 EMA — 3 Days', 'exposure': '100%',
             'achieved': 4 in achieved_steps,
             'detail': f"Streak: {low_above_21_streak} days" if low_above_21_streak > 0 else "Need 3 consecutive days"},
            {'step': 5, 'label': 'Holds 50 SMA — 3 Days', 'exposure': '120%',
             'achieved': 5 in achieved_steps,
             'detail': f"Streak: {low_above_50_streak} days" if low_above_50_streak > 0 else "Need 3 consecutive days"},
            {'step': 6, 'label': 'MA Crossovers', 'exposure': '150%',
             'achieved': 6 in achieved_steps,
             'detail': "21 EMA > 50 SMA > 200 SMA"},
            {'step': 7, 'label': 'PowerTrend', 'exposure': '200%',
             'achieved': 7 in achieved_steps,
             'detail': "8 EMA > 21 EMA > 50 SMA > 200 SMA"},
        ]

        # Exposure: sum of increments per achieved step, with pre-FTD cap at 40%
        step_increments = {0: 20, 1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 30, 7: 50}
        entry_exposure = sum(step_increments[s] for s in achieved_steps)
        if 1 not in achieved_steps:
            entry_exposure = min(entry_exposure, 40)  # Pre-FTD cap
        suggested_exposure = entry_exposure

        # Override exposure if exit alert is active (UPTREND/POWERTREND only — not during RALLY MODE)
        exit_override = None
        if active_exits and cycle_state in ("UPTREND", "POWERTREND"):
            worst = min(int(e['target'].replace('%', '')) for e in active_exits)
            if worst < suggested_exposure:
                exit_override = worst
                suggested_exposure = worst

        return {
            'cycle_state': cycle_state,
            'entry_step': entry_step,
            'suggested_exposure': suggested_exposure,
            'entry_exposure': entry_exposure,
            'exit_override': exit_override,
            'entry_ladder': entry_ladder,
            'active_exits': active_exits,
            'violation_log': violation_log[-10:],
            'rally_day_date': rally_start_date,
            'rally_day_type': rally_day_type,
            'ftd_date': ftd_date,
            'days_since_rally': days_since_rally,
            'rally_low': rally_low,
            'rally_low_date': df.index[rally_low_idx] if rally_low_idx is not None else None,
            'days_since_low': (len(df) - 1 - rally_low_idx) if rally_low_idx is not None else None,
            'correction_start': correction_start,
            'drawdown_pct': float(drawdown_pct),
            'reference_high': float(reference_high) if reference_high else 0,
            'reference_high_date': reference_high_date,
            'price': price,
            'ema8': ema8,
            'ema21': ema21,
            'sma50': sma50,
            'sma200': sma200,
            'low_above_21_streak': low_above_21_streak,
            'low_above_50_streak': low_above_50_streak,
            'consecutive_closes_below_21': consecutive_closes_below_21,
            'last_data_date': last_data_date,
        }
    except Exception as e:
        import traceback
        print(f"Cycle tracker error: {traceback.format_exc()}")
        return None


def compute_historical_market_windows(dates):
    """Compute M Factor market window for a list of historical dates.
    Returns dict mapping date_str -> status string.
    """
    if not dates:
        return {}

    # Fetch enough data: 60-day lookback needs data from well before earliest date
    earliest = min(dates)
    fetch_start = (earliest - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
    latest = max(dates)
    fetch_end = (latest + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        ndx_df = yf.Ticker("^IXIC").history(start=fetch_start, end=fetch_end)
        spy_df = yf.Ticker("SPY").history(start=fetch_start, end=fetch_end)
    except:
        return {}

    if ndx_df.empty or spy_df.empty:
        return {}

    # Strip timezone info to avoid tz-naive vs tz-aware comparison errors
    ndx_df.index = ndx_df.index.tz_localize(None)
    spy_df.index = spy_df.index.tz_localize(None)

    def calc_state_on_date(df, target_date):
        """Run M Factor state machine up to target_date using 60-day window."""
        mask = df.index <= target_date
        available = df[mask]
        if len(available) < 30:
            return "Open"

        available = available.copy()
        available['21EMA'] = available['Close'].ewm(span=21, adjust=False).mean()
        available['50SMA'] = available['Close'].rolling(window=50).mean()
        available['Prev_Close'] = available['Close'].shift(1)
        available['Is_Up'] = available['Close'] > available['Prev_Close']

        subset = available.iloc[-60:]
        state = "OPEN"
        setup_low_21 = None; setup_low_50 = None; pt_streak = 0

        for _, row in subset.iterrows():
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

        return state

    results = {}
    for target in dates:
        target_ts = pd.Timestamp(target)
        ndx_state = calc_state_on_date(ndx_df, target_ts)
        spy_state = calc_state_on_date(spy_df, target_ts)

        if ndx_state == "POWERTREND" or spy_state == "POWERTREND":
            combined = "Powertrend"
        elif ndx_state == "CLOSED" and spy_state == "CLOSED":
            combined = "Closed"
        elif ndx_state in ["NEUTRAL", "CLOSED"] or spy_state in ["NEUTRAL", "CLOSED"]:
            combined = "Neutral"
        else:
            combined = "Open"

        date_str = target_ts.strftime('%Y-%m-%d')
        results[date_str] = combined

    return results

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
st.sidebar.markdown("""
<div style="padding: 1.25rem 0.5rem 0.75rem; text-align: left;">
    <span style="font-size: 1.5rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em;">
        MO Money
    </span>
    <span style="font-size: 0.65rem; background: rgba(99,102,241,0.15); color: var(--accent-blue); padding: 2px 8px; border-radius: 10px; margin-left: 8px; font-weight: 600;">
        v17
    </span>
</div>
""", unsafe_allow_html=True)

# A. SINGLE STRATEGY SELECTOR
# This variable 'portfolio' controls the entire app context.
portfolio = st.sidebar.selectbox(
    "Active Strategy",
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

# Initialize session state for page selection
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

# Navigation UI
with st.sidebar:
    # Helper function to create nav button with active state
    def nav_button(label, icon=""):
        is_active = st.session_state.get('page', '') == label
        icon_text = f"{icon} " if icon else ""
        if st.button(
            f"{icon_text}{label}",
            key=f"nav_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.page = label
            # Clear Trade Journal search state when navigating away
            if label != "Trade Journal" and '_tj_prev_page' in st.session_state:
                del st.session_state['_tj_prev_page']
            # Reset cycle tracker auto-refresh flag
            if label != "Market Cycle Tracker":
                st.session_state.pop('_cycle_loaded', None)
            st.rerun()

    # Section header helper
    def nav_section(label):
        st.markdown(f"""
        <div style="font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
                    letter-spacing: 0.1em; color: var(--text-muted);
                    padding: 0.75rem 0 0.25rem 0.25rem; margin-top: 0.25rem;">
            {label}
        </div>
        """, unsafe_allow_html=True)

    nav_section("DASHBOARDS")
    nav_button("Dashboard", "📊")
    nav_button("Trading Overview", "📈")

    nav_section("TRADING OPS")
    nav_button("Active Campaign Summary", "📋")
    nav_button("Log Buy", "🟢")
    nav_button("Log Sell", "🔴")
    nav_button("Position Sizer", "🔢")
    nav_button("Trade Journal", "📔")
    nav_button("Trade Manager", "📝")

    nav_section("RISK MANAGEMENT")
    nav_button("Earnings Planner", "💣")
    nav_button("Portfolio Heat", "🔥")
    nav_button("Risk Manager", "🛡️")

    nav_section("DAILY WORKFLOW")
    nav_button("Daily Journal", "📔")
    nav_button("Daily Report Card", "📊")
    nav_button("Daily Routine", "🌅")
    nav_button("Weekly Retro", "🔄")

    nav_section("MARKET INTEL")
    nav_button("IBD Market School", "🏫")
    nav_button("M Factor", "📊")
    nav_button("Market Cycle Tracker", "🔄")

    nav_section("AI")
    nav_button("AI Coach", "🤖")

    nav_section("DEEP DIVE")
    nav_button("Analytics", "📈")
    nav_button("Performance Audit", "📊")
    nav_button("Performance Heat Map", "🔥")
    nav_button("Period Review", "⏱️")
    nav_button("Ticker Forensics", "🔬")

    nav_section("LEGACY")
    nav_button("Dashboard (Legacy)", "⚙️")

# Get page from session state
page = st.session_state.page

st.sidebar.markdown("---")
# Privacy mode — use two-key pattern so the toggle survives navigation.
# `_privacy_enabled` is a plain session state key that's NEVER tied to a
# widget, so Streamlit won't garbage-collect it between page reruns.
# The widget itself has its own transient key and seeds from the persistent
# one; on change, we write back via the on_change callback.
if '_privacy_enabled' not in st.session_state:
    st.session_state['_privacy_enabled'] = False

def _sync_privacy():
    st.session_state['_privacy_enabled'] = st.session_state.get('privacy_mode_widget', False)

st.sidebar.toggle(
    "🔒 Privacy Mode",
    value=st.session_state['_privacy_enabled'],
    key='privacy_mode_widget',
    on_change=_sync_privacy,
)
st.sidebar.caption(f"📂 **Active:** {CURR_PORT_NAME}")

# Logout button (only if password auth is active)
if _AUTH_ACTIVE:
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", key="logout_btn", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- Privacy mode helper ---
# Read from the persistent key (see two-key pattern above). Fall back to the
# widget key in case the page loads before the toggle has been touched once.
PRIVACY = st.session_state.get('_privacy_enabled', False)

def mask(val, fmt=",.2f", prefix="$", suffix=""):
    """Return formatted string or masked value if privacy mode is on."""
    if PRIVACY:
        return f"{prefix}****{suffix}"
    try:
        return f"{prefix}{format(val, fmt)}{suffix}"
    except:
        return f"{prefix}{val}{suffix}"


# ==============================================================================
# UI HELPER FUNCTIONS — Reusable card components
# ==============================================================================

def metric_card(label, value, sub="", gradient="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"):
    """Render a gradient metric card with consistent styling."""
    return f"""
    <div class="mo-gradient-card" style="background: {gradient};">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """

def flat_card(label, value, sub="", accent_color=None):
    """Render a flat (white) metric card with optional accent border."""
    border_style = f"border-left: 4px solid {accent_color};" if accent_color else ""
    return f"""
    <div class="mo-card" style="{border_style}">
        <div class="mo-card-label">{label}</div>
        <div class="mo-card-value">{value}</div>
        <div class="mo-card-sub">{sub}</div>
    </div>
    """

def page_header(title, subtitle="", icon=""):
    """Render a clean page header with optional subtitle."""
    icon_html = f'<span style="margin-right: 0.5rem;">{icon}</span>' if icon else ""
    sub_html = f'<div style="font-size: 0.85rem; opacity: 0.6; margin-top: 0.25rem;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 2px solid rgba(128,128,128,0.2);">
        <div style="font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;">
            {icon_html}{title}
        </div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

# Gradient presets for consistent color usage
GRADIENTS = {
    'blue':   'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'green':  'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
    'pink':   'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    'orange': 'linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)',
    'cyan':   'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    'sunset': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
    'navy':   'linear-gradient(135deg, #4b6cb7 0%, #182848 100%)',
    'indigo': 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
    'red':    'linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)',
    'teal':   'linear-gradient(135deg, #0d9488 0%, #14b8a6 100%)',
}


# ==============================================================================
# PAGE 2: DASHBOARD (NEW MODERN VERSION)
# ==============================================================================
if page == "Dashboard":
    # Welcome header
    _today_str = datetime.now().strftime('%A, %B %d, %Y')
    _hour = datetime.now().hour
    _greeting = "Good morning" if _hour < 12 else ("Good afternoon" if _hour < 17 else "Good evening")
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <div style="font-size: 1.75rem; font-weight: 800; letter-spacing: -0.02em;">
            {_greeting}, {st.session_state.get('user_name', 'MO').split()[0]}
        </div>
        <div style="font-size: 0.9rem; opacity: 0.6; margin-top: 0.25rem;">
            {_today_str} • {CURR_PORT_NAME}
        </div>
    </div>
    """, unsafe_allow_html=True)

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

                # Get live prices (cached 60s)
                tickers = df_open['Ticker'].unique().tolist()
                try:
                    price_map = cached_batch_live_prices(tuple(sorted(tickers)))

                    def get_live_price(r):
                        return price_map.get(r['Ticker'], float(r['Avg_Entry']))

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

        # === DRAWDOWN CALCULATION (matches Risk Manager) ===
        RESET_DATE = pd.Timestamp("2026-02-24")
        df_active = df_j[df_j['Day'] >= RESET_DATE].copy()
        if not df_active.empty:
            peak_nlv = df_active['End NLV'].max()
            dd_dol = peak_nlv - curr_nlv
            dd_pct_abs = (dd_dol / peak_nlv) * 100 if peak_nlv > 0 else 0.0
            curr_dd = -dd_pct_abs
            # Determine level (matches Risk Manager hard decks)
            if dd_pct_abs >= 15.0:
                dd_level = "L3: Go to Cash"
            elif dd_pct_abs >= 12.5:
                dd_level = "L2: Max 30%"
            elif dd_pct_abs >= 7.5:
                dd_level = "L1: No Margin"
            else:
                dd_level = "Clear"
        else:
            curr_dd = 0.0
            dd_level = "Clear"

        # === MODERN METRICS CARDS ===
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            nlv_display = "$****" if PRIVACY else f"${curr_nlv:,.0f}"
            dol_display = "****" if PRIVACY else f"{'+' if daily_dol >= 0 else ''}{daily_dol:,.0f}"
            change_color = '#90EE90' if daily_dol >= 0 else '#ffcccb'
            st.markdown(metric_card(
                "NET LIQ VALUE", nlv_display,
                f"<span style='color:{change_color}'>{dol_display} ({'+' if daily_pct_display >= 0 else ''}{daily_pct_display:.2f}%)</span>",
                GRADIENTS['blue']
            ), unsafe_allow_html=True)

        with col2:
            ltd_profit = df_j['Daily $ Change'].sum()
            ltd_profit_color = "#2ca02c" if ltd_profit >= 0 else "#ff4b4b"
            if PRIVACY:
                ltd_profit_str = f"<span style='color:{ltd_profit_color}'>$****</span>"
            else:
                ltd_profit_str = f"<span style='color:{ltd_profit_color}'>${ltd_profit:+,.0f}</span>"
            st.markdown(metric_card(
                "LTD RETURN", f"{ltd_return:.2f}%", ltd_profit_str,
                GRADIENTS['pink']
            ), unsafe_allow_html=True)

        with col3:
            st.markdown(metric_card(
                "YTD RETURN", f"{ytd_val:.2f}%",
                f"SPY: {'+' if ytd_spy >= 0 else ''}{ytd_spy:.2f}% | NDX: {'+' if ytd_nasdaq >= 0 else ''}{ytd_nasdaq:.2f}%",
                GRADIENTS['green']
            ), unsafe_allow_html=True)

        with col4:
            limit = 12
            st.markdown(metric_card(
                "LIVE EXPOSURE", f"{calc_exposure_pct:.1f}%",
                f"{num_open_pos}/{limit} Pos | Risk: {risk_pct:.2f}%",
                GRADIENTS['orange']
            ), unsafe_allow_html=True)

        with col5:
            if dd_level == "Clear":
                dd_color = "#90EE90"
            elif dd_level.startswith("L1"):
                dd_color = "#FFD700"
            elif dd_level.startswith("L2"):
                dd_color = "#FFA500"
            else:
                dd_color = "#ff6b6b"
            st.markdown(metric_card(
                "DRAWDOWN", f"<span style='color:{dd_color}'>{curr_dd:.2f}%</span>",
                f"<span style='color:{dd_color}'>{dd_level}</span>",
                GRADIENTS['navy']
            ), unsafe_allow_html=True)

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

    page_header("Market Dashboard", CURR_PORT_NAME, "📊")
    
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

    # --- HELPER: SECTOR CACHE (per-ticker, 7-day TTL — sectors rarely change) ---
    @st.cache_data(ttl=3600*24*7)
    def _get_ticker_sector(t):
        if t == 'CASH':
            return 'Unknown'
        try:
            inf = yf.Ticker(t).info
            return inf.get('sector', 'Unknown')
        except Exception:
            return 'Unknown'

    def get_sector_map(ticker_list):
        return {t: _get_ticker_sector(t) for t in ticker_list}

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
                
                # A. LIVE PRICE (cached 60s)
                tickers = df_open['Ticker'].unique().tolist()
                price_map = cached_batch_live_prices(tuple(sorted(tickers)))
                df_open['Cur_Px'] = df_open['Ticker'].map(price_map).fillna(df_open['Avg_Entry']).astype(float)

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

            nlv_val = "$****" if PRIVACY else fmt_money(curr_nlv)
            nlv_delta = f"$**** ({daily_pct_display:+.2f}%)" if PRIVACY else f"{daily_dol:+,.2f} ({daily_pct_display:+.2f}%)"
            c1.metric("Net Liq Value", nlv_val, nlv_delta)
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
                        .map(lambda x: 'color: #2ca02c' if x > 0 else 'color: #ff4b4b', subset=['Return_Pct'])
                        .map(color_risk, subset=['R_Dol'])
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
    page_header("Trading Overview", f"{CURR_PORT_NAME} • {datetime.now().strftime('%B %d, %Y')}", "📈")

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

                # Get live prices (cached 60s)
                try:
                    tickers = df_open['Ticker'].unique().tolist()
                    price_map = cached_batch_live_prices(tuple(sorted(tickers)))
                    df_open['Cur_Px'] = df_open['Ticker'].map(price_map).fillna(df_open.get('Avg_Entry', 0)).astype(float)
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

        # === WIDGET HEADER HELPER ===
        def widget_header(title, icon=""):
            """Render a widget card header label."""
            st.markdown(f"""
            <div style="font-size: 0.85rem; font-weight: 700; letter-spacing: 0.02em;
                        margin-bottom: 0.5rem; padding-bottom: 0.4rem;
                        border-bottom: 2px solid rgba(99, 102, 241, 0.3);">
                {icon + '  ' if icon else ''}{title}
            </div>
            """, unsafe_allow_html=True)

        # === PERIOD-SPECIFIC METRICS (filter-aware) ===
        # Calculate period return from filtered journal data
        period_return = 0.0
        period_pl = 0.0
        period_days = 0
        if not df_journal.empty:
            period_days = len(df_journal)
            period_return = ((1 + df_journal['Daily_Pct']).prod() - 1) * 100
            period_pl = df_journal['Daily $ Change'].sum() if 'Daily $ Change' in df_journal.columns else 0

        # Period realized P&L from closed trades in range
        period_realized = 0.0
        if not df_summary.empty:
            closed_in_period = df_summary[df_summary['Status'].str.lower() == 'closed']
            if not closed_in_period.empty and 'Realized_PL' in closed_in_period.columns:
                period_realized = closed_in_period['Realized_PL'].apply(clean_num_local).sum()

        # Period label
        _period_label = date_preset if date_preset != "All Time" else "All Time"

        # === 2. WIDGET-BASED LAYOUT ===

        # --- Row 1: Period stats (gradient cards, filter-aware) ---
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            pl_color = '#90EE90' if period_return >= 0 else '#ffcccb'
            st.markdown(metric_card(
                f"RETURN ({_period_label.upper()})", f"{period_return:+.2f}%",
                f"<span style='color:{pl_color}'>{period_days} trading days</span>",
                GRADIENTS['blue']
            ), unsafe_allow_html=True)
        with h2:
            pl_color2 = '#90EE90' if period_pl >= 0 else '#ffcccb'
            if PRIVACY:
                pl_big = "$****"
                pl_sub = f"<span style='color:{pl_color2}'>Realized: $****</span>"
            else:
                pl_big = f"${period_pl:+,.0f}"
                pl_sub = f"<span style='color:{pl_color2}'>Realized: ${period_realized:+,.0f}</span>"
            st.markdown(metric_card(
                f"P&L ({_period_label.upper()})", pl_big,
                pl_sub,
                GRADIENTS['green']
            ), unsafe_allow_html=True)
        with h3:
            avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            st.markdown(metric_card(
                "WIN RATE", f"{win_rate:.1f}%",
                f"{wins}W / {losses}L | PF: {profit_factor:.2f}",
                GRADIENTS['pink']
            ), unsafe_allow_html=True)
        with h4:
            st.markdown(metric_card(
                "MAX DRAWDOWN", f"{max_drawdown:.1f}%",
                f"Avg W/L: {avg_win_loss_ratio:.2f}",
                GRADIENTS['navy']
            ), unsafe_allow_html=True)

        # --- Row 2: Widget grid (2 columns) ---
        w_left, w_right = st.columns([3, 2])

        # LEFT COLUMN: Equity Curve widget
        with w_left:
            widget_header("Equity Curve", "📈")
            if not df_journal.empty and 'Daily_Pct' in df_journal.columns and PLOTLY_AVAILABLE:
                import plotly.graph_objects as go
                # Recalculate return curve within the filtered period
                _filtered_curve = ((1 + df_journal['Daily_Pct']).cumprod() - 1) * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_journal['Day'], y=_filtered_curve,
                    mode='lines', fill='tozeroy',
                    line=dict(color='#6366f1', width=2),
                    fillcolor='rgba(99, 102, 241, 0.15)',
                    name='Return %'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
                fig.update_layout(
                    height=320, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
                    hovermode='x unified', showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No equity curve data")

        # RIGHT COLUMN: Stats widgets stacked
        with w_right:
            # Trade Stats widget
            widget_header("Trade Stats", "📊")
            ts1, ts2 = st.columns(2)
            ts1.metric("Total Trades", total_trades)
            ts2.metric("Active", active_trades)
            ts3, ts4 = st.columns(2)
            ts3.metric("Avg W/L Ratio", f"{avg_win_loss_ratio:.2f}")
            streak_display = f"{current_streak} {'W' if streak_type == 'Win' else 'L'}" if streak_type else "—"
            ts4.metric("Streak", streak_display)

            st.markdown("")  # spacer

            # Profit breakdown widget
            widget_header("Profit Breakdown", "💰")
            pb1, pb2 = st.columns(2)
            pb1.metric("Avg Win", f"${avg_win:,.0f}" if avg_win else "—")
            pb2.metric("Avg Loss", f"${avg_loss:,.0f}" if avg_loss else "—")
            pb3, pb4 = st.columns(2)
            pb3.metric("Profit Factor", f"{profit_factor:.2f}")
            pb4.metric("Closed Trades", f"{wins + losses}")

        # --- Row 3: Recent Trades widget (full width) ---
        widget_header("Recent Trades", "🔄")
        if not df_details.empty:
            _recent = df_details.copy()
            if 'Date' in _recent.columns:
                _recent['Date'] = pd.to_datetime(_recent['Date'], errors='coerce')
                # Filter to date range if set
                if start_date is not None and end_date is not None:
                    _recent = _recent[(_recent['Date'].dt.date >= start_date) & (_recent['Date'].dt.date <= end_date)]
                _recent = _recent.sort_values('Date', ascending=False)
            _recent = _recent.head(10)
            _show_cols = [c for c in ['Date', 'Ticker', 'Action', 'Shares', 'Amount', 'Value', 'Rule'] if c in _recent.columns]
            if 'Date' in _show_cols:
                _recent['Date'] = _recent['Date'].dt.strftime('%m/%d %H:%M')
            st.dataframe(_recent[_show_cols], use_container_width=True, hide_index=True)
        else:
            st.caption("No trades logged yet.")

# ==============================================================================
# PAGE 3: DAILY JOURNAL (CLEAN & FINAL)
# ==============================================================================
elif page == "Daily Journal":
    page_header("Daily Trading Journal", CURR_PORT_NAME, "📔")
    
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
                
                # LTD (Life-to-Date) — same method as Dashboard equity curve
                df_calc['Adjusted_Beg'] = df_calc['Beg NLV'] + df_calc['Cash -/+']
                df_calc['_daily_dec'] = 0.0
                _mask = df_calc['Adjusted_Beg'] != 0
                df_calc.loc[_mask, '_daily_dec'] = (df_calc.loc[_mask, 'End NLV'] - df_calc.loc[_mask, 'Adjusted_Beg']) / df_calc.loc[_mask, 'Adjusted_Beg']
                df_calc['LTD_Pct'] = ((1 + df_calc['_daily_dec']).cumprod() - 1) * 100

                # Ensure ATR/Heat columns are numeric
                for _atr_col in ['Portfolio_Heat', 'SPY_ATR', 'Nasdaq_ATR']:
                    if _atr_col in df_calc.columns:
                        df_calc[_atr_col] = pd.to_numeric(df_calc[_atr_col], errors='coerce').fillna(0.0)
                    else:
                        df_calc[_atr_col] = 0.0

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
                    'Market Window',
                    'End NLV',
                    'Score',
                    'Daily_Pct',
                    'LTD_Pct',
                    'Portfolio_Heat',
                    'SPY_Pct',
                    'SPY_ATR',
                    'Nasdaq_Pct',
                    'Nasdaq_ATR',
                    'Market_Notes',
                    'Market_Action',
                    'Mistakes',
                    'Top_Lesson',
                    'Highlights',
                    'Lowlights',
                ]
                valid_cols = [c for c in show_cols if c in df_view.columns]
                
                def color_score(val):
                    """Colors the Score column based on the 1-5 scale."""
                    color = ''
                    if val == 5: color = 'background-color: #008000; color: white;'
                    elif val == 4: color = 'background-color: #90EE90; color: black;'
                    elif val == 3: color = 'background-color: #FFFFE0; color: black;'
                    elif val == 2: color = 'background-color: #FFD700; color: black;'
                    elif val == 1: color = 'background-color: #FF4B4B; color: white;'
                    return color

                def color_market_window(val):
                    """Colors the Market Window column to match M Factor."""
                    v = str(val).strip().upper()
                    if v == 'POWERTREND': return 'background-color: #8A2BE2; color: white;'
                    elif v == 'OPEN': return 'background-color: #2ca02c; color: white;'
                    elif v == 'NEUTRAL': return 'background-color: #ffcc00; color: black;'
                    elif v == 'CLOSED': return 'background-color: #ff4b4b; color: white;'
                    return ''

                def color_atr(val):
                    """Green < 1, Yellow 1-1.25, Red > 1.25 (matching Market Window colors)."""
                    try:
                        v = float(val)
                    except (ValueError, TypeError):
                        return ''
                    if v < 1.0: return 'background-color: #2ca02c; color: white;'
                    elif v <= 1.25: return 'background-color: #ffcc00; color: black;'
                    else: return 'background-color: #ff4b4b; color: white;'

                st.dataframe(
                    df_view.sort_values('Day', ascending=False)[valid_cols]
                    .style.format({
                        'Day': '{:%m/%d/%y}',
                        'End NLV': '${:,.2f}',
                        'Daily_Pct': '{:+.2f}%',
                        'LTD_Pct': '{:+.2f}%',
                        'Portfolio_Heat': '{:.2f}%',
                        'SPY_Pct': '{:+.2f}%',
                        'SPY_ATR': '{:.2f}%',
                        'Nasdaq_Pct': '{:+.2f}%',
                        'Nasdaq_ATR': '{:.2f}%',
                        'Score': '{:.0f}'
                    })
                    .map(color_pnl, subset=[c for c in ['Daily_Pct', 'LTD_Pct', 'SPY_Pct', 'Nasdaq_Pct'] if c in df_view.columns])
                    .map(color_score, subset=['Score'])
                    .map(color_market_window, subset=[c for c in ['Market Window'] if c in df_view.columns])
                    .map(color_atr, subset=[c for c in ['SPY_ATR', 'Nasdaq_ATR'] if c in df_view.columns]),
                    hide_index=True,
                    use_container_width=True
                )

                # Export button
                export_df = df_view.sort_values('Day', ascending=False)[valid_cols].copy()
                export_df['Day'] = export_df['Day'].dt.strftime('%Y-%m-%d')
                csv_data = export_df.to_csv(index=False)
                filter_label = view_opt.lower().replace(" ", "_")
                st.download_button(
                    "📥 Export to CSV",
                    csv_data,
                    file_name=f"daily_journal_{filter_label}.csv",
                    mime="text/csv",
                )

            else: st.info("No journal entries found.")

        # --- TAB 2: MANAGE LOGS ---
        with tab_manage:
            st.subheader("Edit / Correct Entry")
            if not df_j.empty:
                df_j_edit = df_j.dropna(subset=['Day']).sort_values('Day', ascending=False).copy()
                if not df_j_edit.empty:
                    options = [f"{row['Day'].strftime('%Y-%m-%d')} | End NLV: ${float(row['End NLV']):,.2f}" for i, row in df_j_edit.iterrows()]
                    sel_edit = st.selectbox("Select Entry to Edit", options, key="edit_entry_sel")
                    edit_date = sel_edit.split("|")[0].strip()

                    # Load selected entry
                    sel_row = df_j_edit[df_j_edit['Day'].dt.strftime('%Y-%m-%d') == edit_date].iloc[0]

                    with st.form("edit_journal_entry"):
                        ed_c1, ed_c2 = st.columns([1, 3])
                        edit_new_date = ed_c1.date_input("Entry Date", value=pd.to_datetime(edit_date).date(), key="edit_date")
                        edit_new_date_str = edit_new_date.strftime("%Y-%m-%d")
                        if edit_new_date_str != edit_date:
                            ed_c2.info(f"Date will change: {edit_date} → {edit_new_date_str}")

                        e1, e2, e3, e4 = st.columns(4)
                        edit_end_nlv = e1.number_input("End NLV ($)", value=float(sel_row['End NLV']), step=100.0, format="%.2f", key="edit_nlv")
                        edit_beg_nlv = e2.number_input("Beg NLV ($)", value=float(sel_row['Beg NLV']), step=100.0, format="%.2f", key="edit_beg")
                        edit_cash = e3.number_input("Cash -/+", value=float(sel_row.get('Cash -/+', 0)), step=100.0, format="%.2f", key="edit_cash")

                        # Calculate % invested from holdings
                        curr_invested = float(sel_row.get('% Invested', 0))
                        edit_invested = e4.number_input("% Invested", value=curr_invested, step=1.0, format="%.2f", key="edit_invested")

                        e5, e6, e7, e8 = st.columns(4)
                        edit_spy = e5.number_input("SPY Close", value=float(sel_row.get('SPY', 0)), format="%.2f", key="edit_spy")
                        edit_ndx = e6.number_input("Nasdaq Close", value=float(sel_row.get('Nasdaq', 0)), format="%.2f", key="edit_ndx")
                        edit_score = e7.number_input("Score", value=int(sel_row.get('Score', 3)), min_value=1, max_value=5, key="edit_score")

                        edit_notes = st.text_input("Market Notes", value=str(sel_row.get('Market_Notes', '') or ''), key="edit_notes")
                        edit_action = st.text_input("Market Action", value=str(sel_row.get('Market_Action', '') or ''), key="edit_action")

                        fc1, fc2 = st.columns(2)
                        save_edit = fc1.form_submit_button("💾 SAVE CHANGES", type="primary")
                        delete_edit = fc2.form_submit_button("🗑️ DELETE ENTRY")

                    if save_edit:
                        # Recalculate daily change
                        adj_beg = edit_beg_nlv + edit_cash
                        daily_chg = edit_end_nlv - edit_beg_nlv - edit_cash
                        pct_val = (daily_chg / adj_beg) * 100 if adj_beg != 0 else 0.0
                        date_changed = edit_new_date_str != edit_date

                        if USE_DATABASE:
                            # If date changed, delete old entry first
                            if date_changed:
                                db.delete_journal_entry(CURR_PORT_NAME, edit_date)
                            journal_entry = {
                                'portfolio_id': CURR_PORT_NAME,
                                'day': edit_new_date_str,
                                'status': str(sel_row.get('Status', 'U')),
                                'market_window': str(sel_row.get('Market Window', 'Open')),
                                'above_21ema': float(sel_row.get('> 21e', 0)),
                                'cash_flow': edit_cash,
                                'beginning_nlv': edit_beg_nlv,
                                'ending_nlv': edit_end_nlv,
                                'daily_dollar_change': daily_chg,
                                'daily_percent_change': pct_val,
                                'percent_invested': edit_invested,
                                'spy_close': edit_spy,
                                'nasdaq_close': edit_ndx,
                                'market_notes': edit_notes,
                                'market_action': edit_action,
                                'score': edit_score,
                                'highlights': str(sel_row.get('Highlights', '') or ''),
                                'lowlights': str(sel_row.get('Lowlights', '') or ''),
                                'mistakes': str(sel_row.get('Mistakes', '') or ''),
                                'top_lesson': str(sel_row.get('Top_Lesson', '') or ''),
                            }
                            db.save_journal_entry(journal_entry)
                        else:
                            if date_changed:
                                # Remove old date row and append with new date
                                df_j = df_j[df_j['Day'].dt.strftime('%Y-%m-%d') != edit_date]
                            idx_list = df_j[df_j['Day'].dt.strftime('%Y-%m-%d') == (edit_new_date_str if date_changed else edit_date)].index
                            if date_changed or len(idx_list) == 0:
                                new_row = sel_row.copy()
                                new_row['Day'] = pd.to_datetime(edit_new_date_str)
                                new_row['End NLV'] = edit_end_nlv
                                new_row['Beg NLV'] = edit_beg_nlv
                                new_row['Cash -/+'] = edit_cash
                                new_row['Daily $ Change'] = daily_chg
                                new_row['Daily % Change'] = f"{pct_val:.2f}%"
                                new_row['% Invested'] = edit_invested
                                new_row['SPY'] = edit_spy
                                new_row['Nasdaq'] = edit_ndx
                                new_row['Market_Notes'] = edit_notes
                                new_row['Market_Action'] = edit_action
                                new_row['Score'] = edit_score
                                df_j = pd.concat([df_j, pd.DataFrame([new_row])], ignore_index=True)
                                df_j = df_j.sort_values('Day', ascending=False)
                            else:
                                idx = idx_list[0]
                                df_j.at[idx, 'End NLV'] = edit_end_nlv
                                df_j.at[idx, 'Beg NLV'] = edit_beg_nlv
                                df_j.at[idx, 'Cash -/+'] = edit_cash
                                df_j.at[idx, 'Daily $ Change'] = daily_chg
                                df_j.at[idx, 'Daily % Change'] = f"{pct_val:.2f}%"
                                df_j.at[idx, '% Invested'] = edit_invested
                                df_j.at[idx, 'SPY'] = edit_spy
                                df_j.at[idx, 'Nasdaq'] = edit_ndx
                                df_j.at[idx, 'Market_Notes'] = edit_notes
                                df_j.at[idx, 'Market_Action'] = edit_action
                                df_j.at[idx, 'Score'] = edit_score
                            secure_save(df_j, TARGET_FILE)
                        msg = f"Moved entry from {edit_date} → {edit_new_date_str}" if date_changed else f"Updated entry for {edit_date}"
                        st.success(msg)
                        st.rerun()

                    if delete_edit:
                        if USE_DATABASE:
                            db.delete_journal_entry(CURR_PORT_NAME, edit_date)
                        else:
                            df_j = df_j[df_j['Day'].dt.strftime('%Y-%m-%d') != edit_date]
                            secure_save(df_j, TARGET_FILE)
                        st.success(f"Deleted entry for {edit_date}")
                        st.rerun()

            st.markdown("---")

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

            st.markdown("---")
            st.subheader("Backfill Market Window (2026)")
            st.caption("Retroactively compute M Factor status for all 2026 journal entries.")
            if st.button("🔄 BACKFILL MARKET WINDOW"):
                if not df_j.empty:
                    try:
                        df_2026 = df_j[df_j['Day'].dt.year == 2026].copy()
                        if df_2026.empty:
                            st.warning("No 2026 entries found.")
                        else:
                            dates_to_fill = [d for d in df_2026['Day'].tolist()]
                            with st.spinner(f"Computing M Factor for {len(dates_to_fill)} dates..."):
                                window_map = compute_historical_market_windows(dates_to_fill)

                            updated = 0
                            for idx, row in df_j.iterrows():
                                if row['Day'].year != 2026:
                                    continue
                                date_str = row['Day'].strftime('%Y-%m-%d')
                                if date_str in window_map:
                                    new_window = window_map[date_str]
                                    df_j.at[idx, 'Market Window'] = new_window
                                    # Also update in database if applicable
                                    if USE_DATABASE:
                                        journal_entry = {
                                            'portfolio_id': CURR_PORT_NAME,
                                            'day': date_str,
                                            'status': str(row.get('Status', 'U')),
                                            'market_window': new_window,
                                            'above_21ema': float(row.get('> 21e', 0)),
                                            'cash_flow': float(row.get('Cash -/+', 0)),
                                            'beginning_nlv': float(row.get('Beg NLV', 0)),
                                            'ending_nlv': float(row.get('End NLV', 0)),
                                            'daily_dollar_change': float(row.get('Daily $ Change', 0)),
                                            'daily_percent_change': float(str(row.get('Daily % Change', '0')).replace('%', '') or 0),
                                            'percent_invested': float(row.get('% Invested', 0)),
                                            'spy_close': float(row.get('SPY', 0)),
                                            'nasdaq_close': float(row.get('Nasdaq', 0)),
                                            'market_notes': str(row.get('Market_Notes', '') or ''),
                                            'market_action': str(row.get('Market_Action', '') or ''),
                                            'score': int(row.get('Score', 0) or 0),
                                            'highlights': str(row.get('Highlights', '') or ''),
                                            'lowlights': str(row.get('Lowlights', '') or ''),
                                            'mistakes': str(row.get('Mistakes', '') or ''),
                                            'top_lesson': str(row.get('Top_Lesson', '') or ''),
                                        }
                                        db.save_journal_entry(journal_entry)
                                    updated += 1

                            if not USE_DATABASE:
                                secure_save(df_j, TARGET_FILE)
                            st.success(f"✅ Updated Market Window for {updated} entries!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Backfill Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            st.markdown("---")
            st.subheader("Backfill Portfolio Heat")
            heat_date = st.date_input("Date to backfill", value=date(2026, 3, 13), key="heat_backfill_date")
            if st.button("🔥 COMPUTE & SAVE PORTFOLIO HEAT"):
                heat_date_str = heat_date.strftime('%Y-%m-%d')
                with st.spinner(f"Computing portfolio heat for {heat_date_str}..."):
                    # Get equity on that date
                    day_row = df_j[df_j['Day'].dt.date == heat_date]
                    if day_row.empty:
                        st.error(f"No journal entry for {heat_date_str}")
                    else:
                        equity = float(day_row.iloc[0]['End NLV'])
                        heat = compute_portfolio_heat(CURR_PORT_NAME, equity=equity)
                        st.info(f"Computed Portfolio Heat: **{heat:.4f}%**")

                        if USE_DATABASE:
                            # Direct UPDATE just for portfolio_heat
                            try:
                                with db.get_db_connection() as conn:
                                    with conn.cursor() as cur:
                                        cur.execute("""
                                            UPDATE trading_journal
                                            SET portfolio_heat = %s
                                            WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = %s)
                                            AND day = %s
                                        """, (float(heat), CURR_PORT_NAME, heat_date_str))
                                    conn.commit()

                                    # Verify
                                    with conn.cursor() as cur:
                                        cur.execute("""
                                            SELECT portfolio_heat FROM trading_journal
                                            WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = %s)
                                            AND day = %s
                                        """, (CURR_PORT_NAME, heat_date_str))
                                        result = cur.fetchone()
                                        if result:
                                            st.success(f"✅ Portfolio Heat saved: {heat:.4f}% (DB value: {result[0]})")
                                        else:
                                            st.error("No row found for that date")
                            except Exception as e:
                                st.error(f"DB Error: {e}")
                        else:
                            df_j.loc[day_row.index[0], 'Portfolio_Heat'] = heat
                            secure_save(df_j, TARGET_FILE)
                            st.success(f"✅ Portfolio Heat saved for {heat_date_str}: {heat:.4f}%")

            st.markdown("---")
            st.subheader("Backfill SPY & Nasdaq ATR")
            atr_col1, atr_col2 = st.columns(2)
            with atr_col1:
                atr_start = st.date_input("Start date", value=date(2026, 3, 3), key="atr_bf_start")
            with atr_col2:
                atr_end = st.date_input("End date", value=date(2026, 3, 14), key="atr_bf_end")
            if st.button("📊 COMPUTE & SAVE SPY/NASDAQ ATR"):
                with st.spinner("Computing ATR% for SPY and Nasdaq..."):
                    spy_atr_map = compute_index_atr('SPY', atr_start.strftime('%Y-%m-%d'), atr_end.strftime('%Y-%m-%d'))
                    ndx_atr_map = compute_index_atr('^IXIC', atr_start.strftime('%Y-%m-%d'), atr_end.strftime('%Y-%m-%d'))
                    st.write(f"SPY ATR computed for {len(spy_atr_map)} days, Nasdaq ATR for {len(ndx_atr_map)} days")

                    if USE_DATABASE and (spy_atr_map or ndx_atr_map):
                        saved = 0
                        try:
                            with db.get_db_connection() as conn:
                                with conn.cursor() as cur:
                                    # Get all journal dates in range
                                    all_dates = set(list(spy_atr_map.keys()) + list(ndx_atr_map.keys()))
                                    for d in sorted(all_dates):
                                        s_val = float(spy_atr_map.get(d, 0))
                                        n_val = float(ndx_atr_map.get(d, 0))
                                        cur.execute("""
                                            UPDATE trading_journal
                                            SET spy_atr = %s, nasdaq_atr = %s
                                            WHERE portfolio_id = (SELECT id FROM portfolios WHERE name = %s)
                                            AND day = %s
                                        """, (s_val, n_val, CURR_PORT_NAME, d.strftime('%Y-%m-%d')))
                                        if cur.rowcount > 0:
                                            saved += 1
                                conn.commit()
                            st.success(f"✅ Updated {saved} journal entries with SPY/Nasdaq ATR")
                            db.load_journal.clear()
                        except Exception as e:
                            st.error(f"DB Error: {e}")
                    elif not USE_DATABASE:
                        st.warning("CSV mode — backfill not supported for ATR columns")

# ==============================================================================
# PAGE 4: M FACTOR (MARKET HEALTH) - VISUAL FIX
# ==============================================================================
elif page == "M Factor":
    page_header("Market Health (M Factor)", "", "📊")
    # CSS Styles
    st.markdown("""<style>.market-banner {padding: 20px; border-radius: 12px; text-align: center; color: white; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);} .ticker-box {background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 10px; color: black;} .metric-row {display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 15px; color: #333; border-bottom: 1px dashed #eee; padding-bottom: 8px;} .metric-row:last-child {border-bottom: none; margin-bottom: 0; padding-bottom: 0;} .sub-text {font-size: 12px; color: #999; font-weight: 400; margin-left: 5px;}</style>""", unsafe_allow_html=True)
    
    if st.button("Refresh Market Data"): st.cache_data.clear()

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

        latest_transition = max(nasdaq['transition_date'], spy['transition_date'])
        st.markdown(f"""<div class="market-banner" style="background-color: {bg};"><div style="font-size: 14px; opacity: 0.9;">MARKET WINDOW</div><div style="font-size: 48px; font-weight: 800; margin: 5px 0;">{status}</div><div style="font-size: 16px;">RECOMMENDED EXPOSURE: {exp}</div><div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">{explanation}</div><div style="font-size: 13px; opacity: 0.9; margin-top: 8px;">Window entered {status} on {latest_transition.strftime('%b %d, %Y')}</div></div>""", unsafe_allow_html=True)

        # === IBD EXPOSURE CROSS-REFERENCE ===
        if USE_DATABASE:
            try:
                df_signals = db.load_market_signals(days=5)  # Get recent data

                nasdaq_ibd = df_signals[df_signals['symbol'] == '^IXIC'].iloc[0] if not df_signals[df_signals['symbol'] == '^IXIC'].empty else None

                if nasdaq_ibd is not None:
                    st.markdown("---")
                    st.markdown("### 📚 IBD Market School Comparison")

                    col_ibd1, col_ibd2 = st.columns(2)

                    with col_ibd1:
                        ibd_nasdaq_exp = nasdaq_ibd['market_exposure']
                        ibd_nasdaq_dd = nasdaq_ibd['distribution_count']
                        st.metric("IBD Nasdaq Exposure", f"{ibd_nasdaq_exp}/6",
                                 delta=f"{ibd_nasdaq_dd} distribution days")

                    with col_ibd2:
                        st.metric("IBD Buy Switch", "ON ✅" if nasdaq_ibd['buy_switch'] else "OFF ❌")

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
# MARKET CYCLE TRACKER (Layer 2 on top of M Factor)
# ==============================================================================
elif page == "Market Cycle Tracker":
    page_header("Market Cycle Tracker", "", "🔄")

    # CSS
    st.markdown("""<style>
    .cycle-banner {padding: 24px; border-radius: 12px; text-align: center; color: white; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);}
    .exit-card {padding: 20px; border-radius: 10px; margin-bottom: 12px; border-left: 6px solid; box-shadow: 0 2px 6px rgba(0,0,0,0.1);}
    .exit-warning {background-color: #fff8e1; border-color: #ffcc00; color: #333;}
    .exit-serious {background-color: #fce4ec; border-color: #e53935; color: #333;}
    .exit-critical {background-color: #ffebee; border-color: #b71c1c; color: #333;}
    .exit-clear {background-color: #e8f5e9; border-color: #2ca02c; color: #333; padding: 20px; border-radius: 10px; border-left: 6px solid; margin-bottom: 12px;}
    .ladder-step {padding: 12px 16px; border-radius: 8px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; font-size: 15px;}
    .step-achieved {background-color: #e8f5e9; border: 1px solid #a5d6a7;}
    .step-pending {background-color: #fff8e1; border: 1px solid #ffe082;}
    .step-locked {background-color: #f5f5f5; border: 1px solid #e0e0e0; color: #999;}
    </style>""", unsafe_allow_html=True)

    # Auto-refresh on page visit: clear stale cycle data
    if '_cycle_loaded' not in st.session_state:
        compute_cycle_state.clear()
        st.session_state['_cycle_loaded'] = True

    if st.button("🔄 Refresh Market Data", key="cycle_refresh"):
        compute_cycle_state.clear()
        st.rerun()

    cycle = compute_cycle_state()

    if cycle is None:
        st.error("Unable to compute cycle state. Market data unavailable.")
    else:
        # Show data freshness
        ldd = cycle.get('last_data_date')
        if ldd is not None:
            ldd_str = ldd.strftime('%b %d, %Y') if hasattr(ldd, 'strftime') else str(ldd)
            st.caption(f"Data as of: **{ldd_str}** — Close: {cycle['price']:,.2f}")
        # ==========================================
        # SECTION 1: STATE BANNER
        # ==========================================
        cs = cycle['cycle_state']
        entry_exp = cycle.get('entry_exposure', cycle['suggested_exposure'])
        exit_ovr = cycle.get('exit_override')

        if cs == "POWERTREND":
            bg = "#8A2BE2"
            subtitle = "8 EMA > 21 EMA > 50 SMA > 200 SMA — all systems go"
            text_color = "white"
        elif cs == "UPTREND":
            bg = "#2ca02c"
            subtitle = "Market structure intact — confirmed uptrend"
            text_color = "white"
        elif cs == "RALLY MODE":
            bg = "#ffcc00"
            subtitle = f"Day {cycle['days_since_rally']} of rally attempt" if cycle['days_since_rally'] else "Rally in progress"
            text_color = "black"
        else:  # CORRECTION
            bg = "#ff4b4b"
            if cycle.get('days_since_low') is not None:
                subtitle = f"NASDAQ down {cycle['drawdown_pct']:.1f}% from high ({cycle['reference_high']:,.2f}) — Day 0 (waiting for rally day)"
            else:
                subtitle = f"NASDAQ down {cycle['drawdown_pct']:.1f}% from high ({cycle['reference_high']:,.2f}) — waiting for rally day"
            text_color = "white"

        banner_extra = ""
        if cs == "RALLY MODE" and cycle['rally_day_date']:
            rd = cycle['rally_day_date']
            rd_str = rd.strftime('%b %d, %Y') if hasattr(rd, 'strftime') else str(rd)
            rd_label = "Rally Day" if cycle['rally_day_type'] == 'rally' else "Pink Rally Day" if cycle['rally_day_type'] == 'pink' else "Rally Day"
            banner_extra = f"<div style='font-size: 13px; opacity: 0.8; margin-top: 6px;'>{rd_label}: {rd_str}</div>"
        if cs == "CORRECTION" and cycle['correction_start']:
            cs_dt = cycle['correction_start']
            cs_str = cs_dt.strftime('%b %d, %Y') if hasattr(cs_dt, 'strftime') else str(cs_dt)
            low_info = ""
            if cycle.get('rally_low_date') is not None:
                low_dt = cycle['rally_low_date']
                low_str = low_dt.strftime('%b %d, %Y') if hasattr(low_dt, 'strftime') else str(low_dt)
                low_info = f" · Low: {low_str} (${cycle['rally_low']:,.2f})"
            banner_extra = f"<div style='font-size: 13px; opacity: 0.8; margin-top: 6px;'>Correction began: {cs_str}{low_info}</div>"

        # Build exposure display line
        if exit_ovr is not None:
            exposure_line = f"ENTRY LADDER: {entry_exp}% · EXIT OVERRIDE: {exit_ovr}%"
        else:
            exposure_line = f"SUGGESTED EXPOSURE: {entry_exp}%"

        banner_html = (
            f'<div class="cycle-banner" style="background-color: {bg}; color: {text_color};">'
            f'<div style="font-size: 14px; opacity: 0.85;">NASDAQ MARKET CYCLE</div>'
            f'<div style="font-size: 52px; font-weight: 800; margin: 4px 0;">{cs}</div>'
            f'<div style="font-size: 16px;">{subtitle}</div>'
            f'<div style="font-size: 20px; font-weight: 700; margin-top: 10px;">{exposure_line}</div>'
            f'{banner_extra}'
            f'</div>'
        )
        st.markdown(banner_html, unsafe_allow_html=True)

        # Key metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("NASDAQ", f"{cycle['price']:,.2f}")
        ref_date = cycle['reference_high_date']
        ref_date_str = ref_date.strftime('%m/%d/%y') if hasattr(ref_date, 'strftime') else str(ref_date) if ref_date else ''
        m2.metric("Reference High", f"{cycle['reference_high']:,.2f}",
                  delta=f"{ref_date_str}")
        m3.metric("21 EMA", f"{cycle['ema21']:,.2f}",
                  delta=f"{((cycle['price'] - cycle['ema21']) / cycle['ema21'] * 100):+.2f}%")
        m4.metric("50 SMA", f"{cycle['sma50']:,.2f}",
                  delta=f"{((cycle['price'] - cycle['sma50']) / cycle['sma50'] * 100):+.2f}%")
        m5.metric("200 SMA", f"{cycle['sma200']:,.2f}",
                  delta=f"{((cycle['price'] - cycle['sma200']) / cycle['sma200'] * 100):+.2f}%")

        st.markdown("---")

        # ==========================================
        # SECTION 2: EXIT ALERTS (most prominent)
        # ==========================================
        st.subheader("EXIT ALERTS")
        st.caption("Non-negotiable action rules — when a violation fires, act immediately.")

        if not cycle['active_exits']:
            st.markdown("""<div class="exit-clear">
                <div style="font-size: 18px; font-weight: 700; color: #2ca02c;">No Active Violations</div>
                <div style="font-size: 14px; margin-top: 6px; color: #555;">Market structure intact — all exit signals clear.</div>
            </div>""", unsafe_allow_html=True)

            # Show monitoring status
            mon_cols = st.columns(3)
            with mon_cols[0]:
                if cycle['consecutive_closes_below_21'] == 0:
                    st.success("21 EMA — Holding")
                else:
                    st.warning(f"21 EMA — {cycle['consecutive_closes_below_21']} close(s) below")
            with mon_cols[1]:
                if cycle['price'] > cycle['sma50']:
                    st.success("50 SMA — Above")
                else:
                    st.error("50 SMA — Below")
            with mon_cols[2]:
                if cycle['price'] > cycle['sma200']:
                    st.success("200 SMA — Above")
                else:
                    st.error("200 SMA — Below")
        else:
            for alert in cycle['active_exits']:
                sev_class = {'WARNING': 'exit-warning', 'SERIOUS': 'exit-serious', 'CRITICAL': 'exit-critical'}.get(alert['severity'], 'exit-warning')
                st.markdown(f"""<div class="exit-card {sev_class}">
                    <div style="font-size: 22px; font-weight: 800;">{alert['icon']} {alert['signal']}</div>
                    <div style="font-size: 15px; margin-top: 8px;">{alert['detail']}</div>
                    <div style="font-size: 18px; font-weight: 700; margin-top: 10px;">TARGET EXPOSURE: {alert['target']}</div>
                    <div style="font-size: 12px; margin-top: 6px; opacity: 0.7;">Take profits on extended positions. Tighten all stops immediately.</div>
                </div>""", unsafe_allow_html=True)

        # Exit rules reference
        with st.expander("Exit Ladder Rules"):
            st.markdown("""
| Signal | Condition | Target | Severity |
|---|---|---|---|
| **21 EMA Violation** | Close below 21 EMA + next day low undercuts by >0.2% | **50%** | WARNING |
| **21 EMA Confirmed Break** | Two consecutive closes below 21 EMA | **30%** | SERIOUS |
| **50 SMA Violation** | Close below 50 SMA + next day low undercuts by >0.2% | **0%** | CRITICAL |
""")

        # Historical violations
        if cycle['violation_log']:
            with st.expander(f"Violation History ({len(cycle['violation_log'])} events)"):
                vlog_data = []
                for v in reversed(cycle['violation_log']):
                    vd = v['date']
                    vlog_data.append({
                        'Date': vd.strftime('%Y-%m-%d') if hasattr(vd, 'strftime') else str(vd),
                        'Signal': v['signal'],
                        'Price': f"{v['price']:,.2f}",
                        'Target': v['target'],
                        'Severity': v['severity']
                    })
                st.dataframe(pd.DataFrame(vlog_data), hide_index=True, use_container_width=True)

        st.markdown("---")

        # ==========================================
        # SECTION 3: ENTRY LADDER
        # ==========================================
        st.subheader("ENTRY LADDER")
        st.caption("Suggested maximum exposure — actual sizing driven by individual trade performance.")

        # Progress bar
        max_exp = 200
        entry_exp_val = cycle.get('entry_exposure', cycle['suggested_exposure'])
        progress_val = min(entry_exp_val / max_exp, 1.0)
        st.progress(progress_val)
        ladder_label = f"**Current Step: {cycle['entry_step']}** — Entry Ladder Exposure: **{entry_exp_val}%**"
        if cycle.get('exit_override') is not None:
            ladder_label += f"  ⚠️ Exit override active → **{cycle['exit_override']}%**"
        st.markdown(ladder_label)

        # Ladder steps
        for item in cycle['entry_ladder']:
            step = item['step']
            achieved = item['achieved']
            is_next = not achieved and (step == 0 or cycle['entry_ladder'][step - 1]['achieved'] if step > 0 else True)

            if achieved:
                icon = "✅"
                css_class = "step-achieved"
            elif is_next:
                icon = "⏳"
                css_class = "step-pending"
            else:
                icon = "🔒"
                css_class = "step-locked"

            st.markdown(f"""<div class="ladder-step {css_class}">
                <div>
                    <span style="font-weight: 700;">{icon} Step {step}: {item['label']}</span>
                    <span style="font-size: 12px; margin-left: 10px; opacity: 0.7;">{item['detail']}</span>
                </div>
                <div style="font-weight: 700; font-size: 16px;">{item['exposure']}</div>
            </div>""", unsafe_allow_html=True)

        # Streak info
        st.markdown("---")
        sk1, sk2 = st.columns(2)
        with sk1:
            st.metric("Low > 21 EMA Streak", f"{cycle['low_above_21_streak']} days",
                      delta="UPTREND" if cycle['low_above_21_streak'] >= 3 else f"{3 - cycle['low_above_21_streak']} more needed")
        with sk2:
            st.metric("Low > 50 SMA Streak", f"{cycle['low_above_50_streak']} days",
                      delta="Strong" if cycle['low_above_50_streak'] >= 3 else f"{3 - cycle['low_above_50_streak']} more needed")

        # MA Stack
        st.markdown("---")
        st.subheader("Moving Average Stack")
        ma_data = [
            {'MA': '8 EMA', 'Value': f"{cycle['ema8']:,.2f}", 'vs Price': f"{((cycle['price'] - cycle['ema8']) / cycle['ema8'] * 100):+.2f}%"},
            {'MA': '21 EMA', 'Value': f"{cycle['ema21']:,.2f}", 'vs Price': f"{((cycle['price'] - cycle['ema21']) / cycle['ema21'] * 100):+.2f}%"},
            {'MA': '50 SMA', 'Value': f"{cycle['sma50']:,.2f}", 'vs Price': f"{((cycle['price'] - cycle['sma50']) / cycle['sma50'] * 100):+.2f}%"},
            {'MA': '200 SMA', 'Value': f"{cycle['sma200']:,.2f}", 'vs Price': f"{((cycle['price'] - cycle['sma200']) / cycle['sma200'] * 100):+.2f}%"},
        ]
        st.dataframe(pd.DataFrame(ma_data), hide_index=True, use_container_width=True)

        # Check MA stacking order
        stack_order = []
        if cycle['ema8'] > cycle['ema21']: stack_order.append("8 EMA > 21 EMA ✅")
        else: stack_order.append("8 EMA < 21 EMA ❌")
        if cycle['ema21'] > cycle['sma50']: stack_order.append("21 EMA > 50 SMA ✅")
        else: stack_order.append("21 EMA < 50 SMA ❌")
        if cycle['sma50'] > cycle['sma200']: stack_order.append("50 SMA > 200 SMA ✅")
        else: stack_order.append("50 SMA < 200 SMA ❌")
        st.markdown(" | ".join(stack_order))

        # Methodology
        with st.expander("Cycle Tracker Methodology"):
            st.markdown("""
### Two-Layer Architecture

**Layer 1 — M Factor** (existing, unchanged)
- Drives market window: OPEN / NEUTRAL / CLOSED / POWERTREND
- Used for daily trading decisions and rule triggers

**Layer 2 — Cycle Tracker** (this page)
- Tracks the correction/recovery arc that M Factor doesn't model
- Provides portfolio-level exposure guidance through entry and exit ladders

### Market States

| State | Definition |
|---|---|
| **CORRECTION** | NASDAQ down 7%+ from recent high — Day 0, waiting for rally |
| **RALLY MODE** | Rally day identified, steps 0-3 (rally day → low above 21 EMA) |
| **UPTREND** | Steps 4-6 (holds 21 EMA 3 days → MA crossovers at 100%) |
| **POWERTREND** | Step 7 — 8 EMA > 21 EMA > 50 SMA > 200 SMA (200%) |

### Entry Ladder
Exposure levels are **suggested maximums**. Rally Mode (20-40% pre-FTD, 60% on FTD), Uptrend (80-150%), PowerTrend (200%).

### Exit Ladder
Exit signals are **non-negotiable action rules**:
- **21 EMA Violation** → 50% (close below + 0.2% undercut)
- **21 EMA Confirmed Break** → 30% (two consecutive closes below)
- **50 SMA Violation** → 0% (close below + 0.2% undercut)

### Key Rules
1. Entry ladder = suggestions — actual exposure follows trade performance
2. Exit ladder = rules — must be acted on when triggered
3. All signals are NASDAQ only
4. This page does not replace M Factor — it sits on top of it
""")

# ==============================================================================
# Performance Heat Map
# ==============================================================================
elif page == "Performance Heat Map":
    page_header("Performance Heat Map", CURR_PORT_NAME, "🔥")
    
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
    page_header("Ticker Forensics", "", "🔬")
    
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
                        .map(color_pnl, subset=['Realized_PL']),
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
                            .map(color_pnl, subset=['Realized_PL', 'R_Multiple']),
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
                        .map(lambda x: 'background-color: #e6fffa; color: #004d40' if x == 'OPEN' else '', subset=['Status'])
                        .map(color_pnl, subset=['Total_R']),
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
    page_header("Period Review", "", "⏱️")
    
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
                }).map(color_pnl, subset=['Net P&L ($)', 'Return % (TWR)', 'LTD Return %']),
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
    page_header("Daily Routine", "Master Blotter", "🌅")
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
    'Portfolio_Heat', 'SPY_ATR', 'Nasdaq_ATR',
    'Score', 'Highlights', 'Lowlights', 'Mistakes', 'Top_Lesson'
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

    # 2. AUTO-DETECT TRADE ACTIONS PER PORTFOLIO FOR SELECTED DATE
    def get_trade_actions_for_date(portfolio_name, date_str):
        """Look up trades executed on a given date and return formatted action string."""
        try:
            details_path = os.path.join(DATA_ROOT, portfolio_name, 'Trade_Log_Details.csv')
            df_details = load_data(details_path)
            if df_details.empty or 'Date' not in df_details.columns:
                return ""
            df_details['Date'] = pd.to_datetime(df_details['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
            day_trades = df_details[df_details['Date'] == date_str]
            if day_trades.empty:
                return ""
            grouped = {}
            for _, row in day_trades.iterrows():
                action = str(row.get('Action', '')).upper()
                ticker = str(row.get('Ticker', ''))
                label = "BUY" if action == "BUY" else "SELL" if action == "SELL" else action
                grouped.setdefault(label, []).append(ticker)
            parts = []
            for label in ["SELL", "BUY"]:
                if label in grouped:
                    parts.append(f"{label}: {', '.join(grouped[label])}")
            for label in grouped:
                if label not in ["SELL", "BUY"]:
                    parts.append(f"{label}: {', '.join(grouped[label])}")
            return " | ".join(parts)
        except:
            return ""

    # 3. MASTER FORM
    with st.form("master_routine_form"):
        st.subheader("1. General Market Data")
        c1, c2, c3, c4 = st.columns(4)
        # Use date() to avoid timezone issues
        entry_date = c1.date_input("Date", get_current_date_ct())
        entry_date_str = entry_date.strftime("%Y-%m-%d")

        live_spy = cached_live_price("SPY") or 0.0
        live_ndx = cached_live_price("^IXIC") or 0.0

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

                # AUTO-POPULATE ACTIONS FROM TRADE LOGS
                auto_actions = get_trade_actions_for_date(p_name, entry_date_str)
                note_in = c_d.text_input(f"Actions ({p_name})", value=auto_actions, key=f"note_{p_name}", placeholder="e.g. BUY: NVDA, SELL: GOOG")

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
            live_market_window, _ = get_combined_market_status()

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

                    # Compute portfolio heat (ATR-weighted)
                    heat_val = compute_portfolio_heat(p_name, equity=end_nlv)

                    # Compute SPY and Nasdaq daily ATR%
                    spy_atr_val = 0.0
                    nasdaq_atr_val = 0.0
                    try:
                        spy_atr_dict = compute_index_atr('SPY', entry_date_str, entry_date_str)
                        if spy_atr_dict:
                            spy_atr_val = list(spy_atr_dict.values())[0]
                        nasdaq_atr_dict = compute_index_atr('^IXIC', entry_date_str, entry_date_str)
                        if nasdaq_atr_dict:
                            nasdaq_atr_val = list(nasdaq_atr_dict.values())[0]
                    except:
                        pass

                    # 4. CREATE ROW (Updated with Review Data)
                    new_row = {
                        'Day': entry_date_str, 'Status': 'U', 'Market Window': live_market_window, '> 21e': 0.0,
                        'Cash -/+': cash_flow, 'Beg NLV': prev_nlv, 'End NLV': end_nlv,
                        'Daily $ Change': daily_chg, 'Daily % Change': daily_pct_str,
                        '% Invested': invested_pct, 'SPY': spy_val, 'Nasdaq': ndx_val,
                        'Market_Notes': market_notes,
                        'Market_Action': inputs['note'],
                        'Portfolio_Heat': heat_val,
                        'SPY_ATR': spy_atr_val,
                        'Nasdaq_ATR': nasdaq_atr_val,
                        'Score': daily_score,
                        'Highlights': highlights,
                        'Lowlights': lowlights,
                        'Mistakes': mistakes,
                        'Top_Lesson': top_lesson
                    }
                    
                    # 5. SAVE
                    try:
                        if USE_DATABASE:
                            # Save to database
                            journal_entry = {
                                'portfolio_id': p_name,
                                'day': entry_date_str,
                                'status': 'U',
                                'market_window': live_market_window,
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
                                'portfolio_heat': heat_val,
                                'spy_atr': spy_atr_val,
                                'nasdaq_atr': nasdaq_atr_val,
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
    page_header("Position Sizer", CURR_PORT_NAME, "🔢")
    
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
    
    size_map = {"Shotgun (2.5%)": 2.5, "Half (5%)": 5.0, "7.5%": 7.5, "Full (10%)": 10.0, "12.5%": 12.5, "Core (15%)": 15.0, "Core+1 (20%)": 20.0, "Max (25%)": 25.0, "30%": 30.0, "35%": 35.0, "40%": 40.0, "45%": 45.0, "50%": 50.0}

    @st.cache_data(ttl=300)
    def fetch_price_and_atr(ticker):
        """Fetch current price and 21-day ATR% in one yfinance call."""
        try:
            hist = yf.Ticker(ticker).history(period="3mo")
            if hist.empty: return 0.0, 5.0
            current_price = float(hist['Close'].iloc[-1])
            atr_pct = 5.0
            if len(hist) >= 21:
                hist['TR'] = pd.concat([
                    hist['High'] - hist['Low'],
                    (hist['High'] - hist['Close'].shift(1)).abs(),
                    (hist['Low'] - hist['Close'].shift(1)).abs()
                ], axis=1).max(axis=1)
                sma_tr = hist['TR'].tail(21).mean()
                sma_low = hist['Low'].tail(21).mean()
                if sma_low > 0:
                    atr_pct = (sma_tr / sma_low) * 100
            return current_price, round(atr_pct, 2)
        except:
            return 0.0, 5.0

    tab_normal, tab_vol, tab_add, tab_pyr, tab_trim = st.tabs([
        "📏 Normal Sizer",
        "⚖️ Volatility Sizer",
        "📐 Scale In Sizer",
        "🔺 Pyramid Sizer",
        "✂️ Trim (Sell Down)"
    ])
    
    # --- TAB: SCALE IN SIZER ---
    with tab_add:
        st.caption("Scale up to target weight while respecting your global stop and risk budget.")
        open_positions = df_s[df_s['Status'] == 'OPEN'].sort_values('Ticker')

        if not open_positions.empty:
            sel_pos = st.selectbox("Select Holding", options=open_positions['Ticker'].unique().tolist(), key="add_sel")
            row = open_positions[open_positions['Ticker'] == sel_pos].iloc[0]

            # Fetch Price safely (cached 60s)
            live_price = cached_live_price(row['Ticker']) or row.get('Avg_Entry', 100)

            c1, c2 = st.columns(2)
            curr_price = c1.number_input("Current Price ($)", min_value=0.01, value=float(live_price), step=0.1, key=f"add_cp_{row['Ticker']}")
            acct_val_add = c2.number_input("Account Equity ($)", value=equity, disabled=True, key="add_av")

            curr_weight = (row['Shares'] * curr_price / equity) * 100 if equity > 0 else 0
            st.markdown(f"**Current Status:** {int(row['Shares'])} shares @ ${row['Avg_Entry']:.2f} ({curr_weight:.1f}% Weight)")
            st.markdown("---")

            # Target & Risk sliders
            c3, c4 = st.columns(2)
            target_mode_add = c3.select_slider("Target Total Position Size", options=list(size_map.keys()), value="Full (10%)", key="add_ts_mode")
            target_size_pct = size_map[target_mode_add]
            max_risk_pct = c4.slider("Max Total Risk % (Capital)", 0.25, 1.25, 0.75, 0.05, key="add_mr")

            # Global Stop: MA Level + Buffer
            st.markdown("---")
            s1, s2 = st.columns(2)
            add_ma_level = s1.number_input("Key MA Level ($)", value=0.0, step=0.1, help="Price of the Moving Average (e.g. 21e/50s).", key="add_ma_level")
            add_buffer_pct = s2.number_input("Buffer (%)", value=1.0, step=0.1, help="Wiggle room below the MA.", key="add_buffer")

            if add_ma_level > 0:
                calc_stop = add_ma_level * (1 - add_buffer_pct / 100)
                stop_dist = ((curr_price - calc_stop) / curr_price * 100) if curr_price > 0 else 0
                st.info(f"📍 **Calculated Stop:** ${calc_stop:.2f} (MA ${add_ma_level:.2f} − {add_buffer_pct:.1f}% buffer) — {stop_dist:.1f}% below current price")

            st.markdown("---")

            if st.button("Calculate Add-On", key="add_btn"):
                # Validate stop
                if add_ma_level <= 0:
                    st.error("Enter a Key MA Level to calculate your global stop.")
                else:
                    calc_stop = add_ma_level * (1 - add_buffer_pct / 100)
                    risk_per_share = curr_price - calc_stop

                    if risk_per_share <= 0:
                        st.error(f"❌ **INVALID** — Stop (${calc_stop:.2f}) is at or above current price (${curr_price:.2f}).")
                    else:
                        current_shares = int(row['Shares'])
                        current_value = current_shares * curr_price

                        # What you WANT (target)
                        target_value = acct_val_add * (target_size_pct / 100)
                        target_total_shares = math.floor(target_value / curr_price)
                        target_add = target_total_shares - current_shares

                        # What you can AFFORD (risk-limited)
                        max_risk_dollars = acct_val_add * (max_risk_pct / 100)
                        max_total_shares = math.floor(max_risk_dollars / risk_per_share)
                        affordable_add = max_total_shares - current_shares

                        if target_add <= 0:
                            st.error(f"You are already at or above the target weight! (Current: ${current_value:,.0f} vs Target: ${target_value:,.0f})")
                        elif affordable_add <= 0:
                            risk_at_current = current_shares * risk_per_share
                            st.error(f"🚫 **NO ADD** — Your current {current_shares} shares already risk ${risk_at_current:,.0f} (budget: ${max_risk_dollars:,.0f}). Tighten your stop or reduce position.")
                        else:
                            # Recommendation = min of want and afford
                            recommended_add = min(target_add, affordable_add)
                            new_total = current_shares + recommended_add
                            new_avg_cost = ((current_shares * row['Avg_Entry']) + (recommended_add * curr_price)) / new_total
                            cost_of_add = recommended_add * curr_price
                            total_risk_at_new = new_total * risk_per_share
                            new_weight = (new_total * curr_price / acct_val_add) * 100

                            # --- PYRAMID TICKET ---
                            st.markdown("### ➕ PYRAMID TICKET")
                            k1, k2, k3, k4 = st.columns(4)
                            k1.metric("ADD SHARES", f"+{recommended_add}")
                            k2.metric("EST. COST", f"${cost_of_add:,.2f}")
                            k3.metric("NEW TOTAL", f"{new_total} shs", f"{new_weight:.1f}% Weight")
                            k4.metric("NEW AVG COST", f"${new_avg_cost:.2f}", f"From ${row['Avg_Entry']:.2f}")

                            # --- RISK MANAGEMENT ---
                            st.markdown("### 🛡️ RISK MANAGEMENT")
                            r1, r2, r3 = st.columns(3)
                            r1.metric("Global Stop", f"${calc_stop:.2f}", f"-{(risk_per_share/curr_price)*100:.1f}% from price")
                            r2.metric("Total Risk at New Size", f"${total_risk_at_new:,.0f}", f"{(total_risk_at_new/acct_val_add)*100:.2f}% of NLV")
                            r3.metric("Risk Budget", f"${max_risk_dollars:,.0f}", f"{max_risk_pct}% of Equity")

                            # --- VERDICT ---
                            st.markdown("### 🏛️ The Verdict")
                            if affordable_add >= target_add:
                                st.success(f"✅ **ADD {recommended_add} shares** to reach {new_weight:.1f}% — Total risk ${total_risk_at_new:,.0f} within ${max_risk_dollars:,.0f} budget.")
                            else:
                                target_risk = target_total_shares * risk_per_share
                                st.warning(
                                    f"⚠️ **RISK LIMIT:** Full target ({target_add} shares) would risk ${target_risk:,.0f} (over ${max_risk_dollars:,.0f} budget). "
                                    f"**Safe add: {recommended_add} shares** ({new_weight:.1f}% weight). Scale up on next pullback to MA."
                                )

                            # Store results for Send to Log Buy button
                            st.session_state['_add_result'] = {
                                'ticker': sel_pos, 'trade_id': row['Trade_ID'],
                                'shares': recommended_add, 'price': curr_price, 'stop': calc_stop,
                            }

            # --- SEND TO LOG BUY (outside button block) ---
            if '_add_result' in st.session_state:
                r = st.session_state['_add_result']
                st.markdown("---")
                if st.button(f"📝 Send to Log Buy — {r['ticker']} (+{r['shares']} shs @ ${r['price']:.2f})", key="add_send_logbuy", type="secondary", use_container_width=True):
                    st.session_state['ps_ticker'] = r['ticker']
                    st.session_state['ps_trade_id'] = r['trade_id']
                    st.session_state['ps_shares'] = r['shares']
                    st.session_state['ps_price'] = r['price']
                    st.session_state['ps_stop'] = r['stop']
                    st.session_state['ps_action'] = 'scale_in'
                    del st.session_state['_add_result']
                    st.session_state.page = "Log Buy"
                    st.rerun()
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
            
            live_price_t = cached_live_price(row_t['Ticker']) or row_t.get('Avg_Entry', 100)
            
            t1, t2, t3, t4 = st.columns(4)
            curr_price_t = t1.number_input("Current Price ($)", value=float(live_price_t), step=0.1, key=f"trim_cp_{row_t['Ticker']}")
            trim_equity = t2.number_input("Account NLV ($)", value=equity, step=100.0, help="Override with current intraday NLV for accurate sizing.", key="trim_nlv")
            curr_val_t = row_t['Shares'] * curr_price_t
            curr_weight_t = (curr_val_t / trim_equity) * 100 if trim_equity > 0 else 0

            t3.metric("Current Weight", f"{curr_weight_t:.1f}%")
            t4.metric("Current Value", f"${curr_val_t:,.0f}")
            
            st.markdown("---")
            
            # 2. Target Setting
            target_mode_trim = st.select_slider("Target Standard Position Size", options=list(size_map.keys()), value="Half (5%)", key="trim_sm")
            target_weight_t = size_map[target_mode_trim]
            
            if st.button("Calculate Trim Impact", type="primary", key="trim_btn"):
                if target_weight_t >= curr_weight_t: 
                    st.warning(f"⚠️ Target ({target_weight_t}%) is higher than Current ({curr_weight_t:.1f}%). No trim needed.")
                else:
                    # A. Basic Trim Math
                    target_val_t = trim_equity * (target_weight_t / 100)
                    value_to_sell = curr_val_t - target_val_t
                    shares_to_sell = math.ceil(value_to_sell / curr_price_t)
                    remaining_shares = row_t['Shares'] - shares_to_sell
                    actual_new_weight = (remaining_shares * curr_price_t / trim_equity) * 100
                    
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
    # --- TAB: NORMAL SIZER ---
    with tab_normal:
        st.subheader("📏 Normal Sizer")
        st.caption("Size positions based on a key support level with buffer. No ATR involved.")

        ns_ticker = st.text_input("Ticker Symbol", placeholder="XYZ", key="ns_ticker").upper()
        ns_price = 0.0
        if ns_ticker:
            ns_auto_price, _ = fetch_price_and_atr(ns_ticker)
            ns_price = ns_auto_price if ns_auto_price > 0 else 0.0

        c1, c2 = st.columns(2)
        ns_price = c1.number_input("Entry Price ($)", value=ns_price, step=0.1, key=f"ns_px_{ns_ticker}")
        ns_equity = c2.number_input("Account Equity (NLV)", value=equity, step=1000.0, key="ns_eq")

        st.markdown("---")

        c3, c4 = st.columns(2)
        ns_ma_level = c3.number_input("Key MA Level ($)", value=0.0, step=0.1, help="Price of the Moving Average (e.g. 21e/50s).", key="ns_ma_level")
        ns_buffer_pct = c4.number_input("Buffer (%)", value=1.0, step=0.1, help="Wiggle room below the MA.", key="ns_buffer")

        if ns_ma_level > 0 and ns_price > 0:
            ns_stop = ns_ma_level * (1 - ns_buffer_pct / 100)
            ns_stop_dist = ((ns_price - ns_stop) / ns_price * 100)
            st.info(f"📍 **Calculated Stop:** `{ns_stop:.2f}` (MA {ns_ma_level:.2f} − {ns_buffer_pct:.1f}% buffer) — {ns_stop_dist:.1f}% below entry")

        ns_sizing_mode = st.radio("Sizing Mode",
            ["🛡️ Defense (0.50%)", "⚖️ Normal (0.75%)", "⚔️ Offense (1.00%)"],
            horizontal=True, key="ns_sizing_mode")

        ns_target_mode = st.select_slider("Target Position Size", options=list(size_map.keys()), value="Full (10%)", key="ns_target_slider")
        ns_target_pct = size_map[ns_target_mode]

        st.markdown("---")

        if st.button("Calculate Size", type="primary", key="ns_btn"):
            if ns_ticker and ns_price > 0 and ns_ma_level > 0:
                ns_stop = ns_ma_level * (1 - ns_buffer_pct / 100)
                ns_risk_per_share = ns_price - ns_stop

                if ns_risk_per_share <= 0:
                    st.error(f"Stop (${ns_stop:.2f}) is at or above entry price (${ns_price:.2f}).")
                else:
                    # Risk budget from sizing mode
                    if ns_sizing_mode.startswith("⚔️"):
                        ns_tol_pct, ns_tier = 1.00, "Offense Mode"
                    elif ns_sizing_mode.startswith("⚖️"):
                        ns_tol_pct, ns_tier = 0.75, "Normal Mode"
                    else:
                        ns_tol_pct, ns_tier = 0.50, "Defense Mode"

                    ns_risk_budget = ns_equity * (ns_tol_pct / 100)

                    # Shares from risk budget / risk per share
                    ns_risk_shares = int(ns_risk_budget / ns_risk_per_share)

                    # Cap by target position size
                    ns_target_shares = int((ns_equity * ns_target_pct / 100) / ns_price)
                    ns_final_shares = min(ns_risk_shares, ns_target_shares)
                    ns_final_val = ns_final_shares * ns_price
                    ns_final_pct = (ns_final_val / ns_equity * 100) if ns_equity > 0 else 0

                    # Limiting factor
                    if ns_final_shares == ns_target_shares and ns_target_shares < ns_risk_shares:
                        ns_limit = f"Target Size ({ns_target_pct}%)"
                    else:
                        ns_limit = f"MA Support (${ns_ma_level})"

                    # Display
                    st.markdown(f"### 📊 Sizing Profile: {ns_ticker}")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Risk Budget", f"${ns_risk_budget:,.0f}", f"{ns_tol_pct}% Risk ({ns_tier})")
                    ns_stop_dist = ((ns_price - ns_stop) / ns_price * 100)
                    k2.metric("Stop Distance", f"{ns_stop_dist:.1f}%", f"${ns_risk_per_share:.2f}/share")
                    k3.metric("Target Size", f"{ns_target_pct}%", f"${ns_equity * ns_target_pct / 100:,.0f}")

                    st.markdown("---")

                    m1, m2, m3 = st.columns(3)
                    risk_cost = f"${ns_risk_shares * ns_price:,.0f} ({ns_risk_shares * ns_price / ns_equity * 100:.1f}% NLV)"
                    m1.metric("Risk-Based Limit", f"{ns_risk_shares} shs", risk_cost, delta_color="off")
                    target_cost = f"${ns_target_shares * ns_price:,.0f} ({ns_target_pct:.0f}% NLV)"
                    m2.metric("Target Limit", f"{ns_target_shares} shs", target_cost, delta_color="off")
                    m3.metric("Limiting Factor", ns_limit, "Determines Final Size", delta_color="off")

                    st.markdown("### 🏛️ The Verdict")
                    st.success(f"✅ **RECOMMENDED SIZE:** Buy **{ns_final_shares}** shares ({ns_final_pct:.1f}% of NLV).")

                    # Store for Send to Log Buy
                    st.session_state['_ns_result'] = {
                        'ticker': ns_ticker, 'shares': ns_final_shares, 'price': ns_price,
                        'stop': ns_stop, 'risk_budget': ns_risk_budget,
                    }
            else:
                st.error("Please enter Ticker, Entry Price, and Key MA Level.")

        # Send to Log Buy (outside button block)
        if '_ns_result' in st.session_state:
            r = st.session_state['_ns_result']
            st.markdown("---")
            if st.button(f"📝 Send to Log Buy — {r['ticker']} ({r['shares']} shs @ ${r['price']:.2f})", key="ns_send_logbuy", type="secondary", use_container_width=True):
                st.session_state['ps_ticker'] = r['ticker']
                st.session_state['ps_shares'] = r['shares']
                st.session_state['ps_price'] = r['price']
                st.session_state['ps_stop'] = r['stop']
                st.session_state['ps_risk_budget'] = r['risk_budget']
                st.session_state['ps_action'] = 'new'
                del st.session_state['_ns_result']
                st.session_state.page = "Log Buy"
                st.rerun()

    with tab_vol:
        st.subheader("⚖️ Volatility-Adjusted Sizing (The Gem Standard)")
        st.caption("Normalize risk by sizing positions based on Asset Volatility (ATR) AND Technical Stop.")
        
        # 0. TIER CHEAT SHEET
        with st.expander("ℹ️ View Tier System Rules"):
            st.markdown("""
            **Sizing Mode (New Trades — Equity Curve State):**
            * **Defense:** 0.50% Risk — equity curve flat/down
            * **Normal:** 0.75% Risk — equity curve recovering
            * **Offense:** 1.00% Risk — equity curve strong, confirmed uptrend

            **Stock Volatility Profile (New Trades — Stock Character):**
            * **Tight:** 1.0x ATR — low-volatility, tight setups
            * **Normal:** 1.25x ATR — standard growth stocks
            * **High-Vol:** 1.5x ATR — high-volatility names

            **Tolerance Tiers (Active Positions — Profit Cushion):**
            * **Tier 1 (High Cushion):** Profit > 20% ⮕ 1.00% Risk, 2.0x ATR
            * **Tier 2 (Moderate):** Profit 5% to 20% ⮕ 0.65% Risk, 1.5x ATR
            * **Tier 3 (Defense):** Profit < 5% ⮕ 0.50% Risk, 1.0x ATR
            """)

        # 1. MODE SELECTION
        vol_mode = st.radio("Sizing Context", ["🆕 New Trade", "🔍 Audit Active Position"], horizontal=True, key="vs_mode")

        # 1b. SIZING MODE + VOL PROFILE (New Trade only)
        sizing_mode = None
        vol_profile = None
        if vol_mode.startswith("🆕"):
            sm_col1, sm_col2 = st.columns(2)
            with sm_col1:
                sizing_mode = st.radio("Sizing Mode",
                    ["🛡️ Defense (0.50%)", "⚖️ Normal (0.75%)", "⚔️ Offense (1.00%)"],
                    horizontal=True, key="vs_sizing_mode")
            with sm_col2:
                vol_profile = st.radio("Stock Volatility Profile",
                    ["Tight (1.0x)", "Normal (1.25x)", "High-Vol (1.5x)"],
                    horizontal=True, key="vs_vol_profile")

        # 2. DATA INPUTS
        vs_ticker = ""
        vs_price = 0.0
        vs_avg_cost = 0.0
        vs_shares = 0.0
        auto_atr = 5.0  # default, updated below when ticker is known

        c1, c2, c3 = st.columns(3)

        if vol_mode.startswith("🆕"):
            # New Trade Mode
            vs_ticker = c1.text_input("Ticker Symbol", placeholder="XYZ", key="vs_tk_new").upper()
            if vs_ticker:
                auto_price_new, auto_atr = fetch_price_and_atr(vs_ticker)
                def_price = auto_price_new if auto_price_new > 0 else 0.0
            else:
                def_price = 0.0
            vs_price = c2.number_input("Entry Price ($)", value=def_price, step=0.1, key=f"vs_px_new_{vs_ticker}")
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

                    auto_price, auto_atr = fetch_price_and_atr(vs_ticker)
                    if auto_price == 0: auto_price = float(row.get('Current_Price', 0))

                    vs_price = c2.number_input("Current Price ($)", value=auto_price, step=0.1, key=f"vs_px_{vs_ticker}")
                    c3.metric("Avg Cost", f"${vs_avg_cost:,.2f}")
                else:
                    st.info("No open positions found to audit.")
            else:
                st.info("Summary file empty.")

        st.markdown("---")
        
        # 3. CRITICAL INPUTS (ATR + MA LEVEL + BUFFER)
        # Split into 4 columns to fit the new Buffer input
        e1, e2, e3, e4 = st.columns(4)
        vs_equity = e1.number_input("Account Equity (NLV)", value=equity, step=1000.0, key="vs_eq")
        vs_atr_pct = e2.number_input("ATR % (21-Day)", value=auto_atr, step=0.1, help="Auto-calculated from 21-day price data. Adjust if needed.", key=f"vs_atr_{vs_ticker}")
        
        # UPDATED: Split "Stop" into "MA Level" and "Buffer"
        vs_ma_level = 0.0
        vs_buffer_pct = 1.0
        
        vs_target_pct = 0.0
        if vol_mode.startswith("🆕"):
            vs_ma_level = e3.number_input("Key MA Level ($)", value=0.0, step=0.1, help="Price of the Moving Average (e.g. 21e/50s).", key="vs_ma_level")
            vs_buffer_pct = e4.number_input("Buffer (%)", value=1.0, step=0.1, help="Wiggle room below the MA.", key="vs_buffer")
            if vs_ma_level > 0:
                calc_stop = vs_ma_level * (1 - vs_buffer_pct / 100)
                stop_dist = ((vs_price - calc_stop) / vs_price * 100) if vs_price > 0 else 0
                st.info(f"📍 **Calculated Stop:** ${calc_stop:.2f} (MA ${vs_ma_level:.2f} − {vs_buffer_pct:.1f}% buffer) — {stop_dist:.1f}% below entry")
            vs_target_mode = st.select_slider("Target Position Size", options=list(size_map.keys()), value="Full (10%)", key="vs_target_slider")
            vs_target_pct = size_map[vs_target_mode]

        st.markdown("---")
        
        if st.button("Run Sizing Audit", type="primary", key="vs_btn"):
            if vs_ticker and vs_price > 0 and vs_atr_pct > 0:
                
                # 1. Determine Tier & Budget
                if vol_mode.startswith("🆕"):
                    # New Trade: use Sizing Mode + Stock Vol Profile toggles
                    if sizing_mode and sizing_mode.startswith("⚔️"):
                        tier_name = "Offense Mode"
                        tol_pct = 1.00
                    elif sizing_mode and sizing_mode.startswith("⚖️"):
                        tier_name = "Normal Mode"
                        tol_pct = 0.75
                    else:
                        tier_name = "Defense Mode"
                        tol_pct = 0.50

                    if vol_profile and vol_profile.startswith("High"):
                        atr_multiplier = 1.5
                    elif vol_profile and vol_profile.startswith("Normal"):
                        atr_multiplier = 1.25
                    else:
                        atr_multiplier = 1.0
                else:
                    # Audit Mode: use Tolerance Tiers (cushion-based)
                    cushion_pct = ((vs_price - vs_avg_cost) / vs_avg_cost * 100) if vs_avg_cost > 0 else 0.0

                    tier_name = "Tier 3 (Defense)"
                    tol_pct = 0.50
                    atr_multiplier = 1.0

                    if cushion_pct >= 20.0:
                        tier_name = "Tier 1 (High Cushion)"
                        tol_pct = 1.00
                        atr_multiplier = 2.0
                    elif cushion_pct >= 5.0:
                        tier_name = "Tier 2 (Moderate)"
                        tol_pct = 0.65
                        atr_multiplier = 1.5

                daily_risk_budget = vs_equity * (tol_pct / 100)
                atr_risk_budget = daily_risk_budget * atr_multiplier

                # 2. Calculate Volatility Limit (ATR) - scaled by confidence multiplier
                atr_decimal = vs_atr_pct / 100
                max_shares_vol = int(atr_risk_budget / (vs_price * atr_decimal))
                
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
                             # Cap tech stop limit by target position size
                             if vol_mode.startswith("🆕") and vs_target_pct > 0:
                                 target_cap = int((vs_equity * vs_target_pct / 100) / vs_price)
                                 max_shares_tech = min(max_shares_tech, target_cap)
                
                # 4. Hard Cap (20% NLV)
                max_shares_cap = int((vs_equity * 0.20) / vs_price)

                # 4b. Target Position Size cap (New Trade mode only)
                max_shares_target = 999999
                if vol_mode.startswith("🆕") and vs_target_pct > 0:
                    max_shares_target = int((vs_equity * vs_target_pct / 100) / vs_price)

                # 5. Final Decision (Min of all)
                final_max_shares = min(max_shares_vol, max_shares_tech, max_shares_cap, max_shares_target)
                final_max_val = final_max_shares * vs_price

                # Determine Limiting Factor
                limit_reason = "Volatility (ATR)"
                if final_max_shares == max_shares_target and max_shares_target < min(max_shares_vol, max_shares_tech, max_shares_cap):
                    limit_reason = f"Target Size ({vs_target_pct}%)"
                elif final_max_shares == max_shares_cap: limit_reason = "Hard Cap (20%)"
                elif final_max_shares == max_shares_tech: limit_reason = f"MA Support (${vs_ma_level})"
                
                # 6. Display Results
                st.markdown(f"### 📊 Sizing Profile: {vs_ticker}")
                
                k1, k2, k3 = st.columns(3)
                k1.metric("Risk Budget", f"${daily_risk_budget:,.0f}", f"{tol_pct}% Risk ({tier_name})")
                k2.metric("Volatility Risk", f"{vs_atr_pct:.2f}%", f"ATR (Noise)")
                if vol_mode.startswith("🆕"):
                    vol_label = vol_profile if vol_profile else "Tight (1.0x)"
                    k3.metric("Vol Profile", vol_label, f"{atr_multiplier:.1f}x ATR", delta_color="off")
                else:
                    k3.metric("Profit Cushion", f"{cushion_pct:.2f}%", tier_name, delta_color="off")

                if atr_multiplier > 1.0:
                    if vol_mode.startswith("🆕"):
                        st.info(f"🎯 **ATR Boost:** ATR budget scaled {atr_multiplier:.1f}x (${atr_risk_budget:,.0f}) — stock volatility profile")
                    else:
                        st.info(f"🎯 **Confidence Boost:** ATR budget scaled {atr_multiplier:.1f}x (${atr_risk_budget:,.0f}) — earned by {cushion_pct:.1f}% profit cushion")

                st.markdown("---")

                m1, m2, m3 = st.columns(3)
                # ATR Limit: dollar risk = shares × (price × ATR%)
                atr_risk_at_vol = max_shares_vol * vs_price * atr_decimal
                atr_cost_pct = max_shares_vol * vs_price / vs_equity * 100
                m1.metric("ATR Limit", f"{max_shares_vol} shs", f"Risk ${atr_risk_at_vol:,.0f} · {atr_cost_pct:.1f}% NLV", delta_color="off")

                if vol_mode.startswith("🆕") and effective_stop > 0:
                    delta_color = "normal" if max_shares_tech < max_shares_vol else "off"
                    # Tech Stop Limit: dollar risk = shares × (price - stop)
                    tech_risk_at_max = max_shares_tech * (vs_price - effective_stop)
                    tech_cost_pct = max_shares_tech * vs_price / vs_equity * 100
                    m2.metric("Tech Stop Limit", f"{max_shares_tech} shs",
                              f"Risk ${tech_risk_at_max:,.0f} · {tech_cost_pct:.1f}% NLV", delta_color=delta_color)
                else:
                    m2.metric("Hard Cap Limit", f"{max_shares_cap} shs", "20% Max Alloc", delta_color="off")
                    
                # Actual dollar risk at recommended size, using the calculated stop
                # Falls back to a 1-ATR move if no stop is set
                if effective_stop > 0 and effective_stop < vs_price:
                    risk_per_share = vs_price - effective_stop
                    risk_label = f"Stop ${effective_stop:.2f} ({risk_per_share/vs_price*100:.1f}%)"
                else:
                    risk_per_share = vs_price * atr_decimal
                    risk_label = f"1 ATR ({vs_atr_pct:.1f}%)"
                final_risk_dol = final_max_shares * risk_per_share
                m3.metric("Trade Risk $", f"${final_risk_dol:,.0f}", risk_label, delta_color="off")

                st.markdown("### 🏛️ The Verdict")
                
                target_weight = (final_max_val / vs_equity) * 100
                
                if vol_mode.startswith("🆕"):
                    st.success(f"✅ **RECOMMENDED SIZE:** Buy **{final_max_shares}** shares ({target_weight:.1f}% of NLV).")
                    if limit_reason.startswith("MA"):
                        st.info(f"ℹ️ **Note:** Sized for technicals. Your stop (${effective_stop:.2f}) is {tech_dist_pct:.1f}% away (including buffer).")

                    # Store results for Send to Log Buy button (rendered outside this block)
                    st.session_state['_vs_result'] = {
                        'ticker': vs_ticker, 'shares': final_max_shares, 'price': vs_price,
                        'stop': effective_stop if effective_stop > 0 else vs_price * 0.92,
                        'risk_budget': daily_risk_budget,
                    }
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
                    elif diff_shares < 0:
                        add_room = abs(int(diff_shares))
                        add_val = add_room * vs_price
                        v3.metric("Action Required", f"CAN ADD {add_room}", f"Buy up to ${add_val:,.0f}")
                        st.success(f"✅ **Room to add** up to {add_room} shares ({add_val / vs_equity * 100:.1f}% of NLV) within limits.")
                    else:
                        v3.metric("Action Required", "AT LIMIT", "No room to add", delta_color="off")
                        st.info(f"ℹ️ Position is exactly at the {final_max_shares} share limit.")
            else:
                st.error("Please ensure Ticker, Price, and ATR are entered correctly.")

        # --- SEND TO LOG BUY (outside button block so it persists) ---
        if '_vs_result' in st.session_state:
            r = st.session_state['_vs_result']
            st.markdown("---")
            if st.button(f"📝 Send to Log Buy — {r['ticker']} ({r['shares']} shs @ ${r['price']:.2f})", key="vs_send_logbuy", type="secondary", use_container_width=True):
                st.session_state['ps_ticker'] = r['ticker']
                st.session_state['ps_shares'] = r['shares']
                st.session_state['ps_price'] = r['price']
                st.session_state['ps_stop'] = r['stop']
                st.session_state['ps_risk_budget'] = r['risk_budget']
                st.session_state['ps_action'] = 'new'
                del st.session_state['_vs_result']
                st.session_state.page = "Log Buy"
                st.rerun()

    # ==========================================================================
    # TAB 6: PYRAMID SIZER
    # ==========================================================================
    with tab_pyr:
        st.subheader("🔺 Pyramid Sizer")
        st.caption("Size add-on purchases to winning positions. Enforces pace: max 20% of current shares per add, gated by last buy's profit.")

        with st.expander("ℹ️ Pyramid Rules"):
            st.markdown("""
            **How it works:**
            1. Each add is capped at **20% of your current shares**
            2. Your last buy must be up **at least 5%** for a full-size add
            3. If last buy is up **less than 5%**, the add scales proportionally: `(profit% / 5%) × 20%`
            4. If last buy is **flat or down**, no add is allowed
            5. The add is also capped by your ATR limit and 20% hard cap
            """)

        # --- Position Selection (Audit mode only) ---
        pyr_ticker = ""
        pyr_price = 0.0
        pyr_shares = 0.0
        pyr_avg_cost = 0.0
        pyr_inventory = []
        pyr_trade_id = ""

        if not df_s.empty:
            open_ops = df_s[df_s['Status'] == 'OPEN']
            if not open_ops.empty:
                c1, c2, c3 = st.columns(3)
                opts = [f"{r['Ticker']} ({int(r['Shares'])} shs)" for _, r in open_ops.iterrows()]
                sel_pyr = c1.selectbox("Select Position", opts, key="pyr_sel")
                pyr_ticker = sel_pyr.split(" ")[0]

                row = open_ops[open_ops['Ticker'] == pyr_ticker].iloc[0]
                pyr_shares = float(row['Shares'])
                pyr_trade_id = row['Trade_ID']

                # --- LIFO ENGINE (clone from vol sizer) ---
                lifo_cost = float(row['Avg_Entry'])

                if not df_d.empty:
                    tid = row['Trade_ID']
                    trxs = df_d[df_d['Trade_ID'] == tid].copy()
                    if not trxs.empty:
                        trxs['Type_Rank'] = trxs['Action'].apply(lambda x: 0 if str(x).upper() == 'BUY' else 1)
                        if 'Date' in trxs.columns: trxs = trxs.sort_values(['Date', 'Type_Rank'])

                        pyr_inventory = []
                        for _, tx in trxs.iterrows():
                            action = str(tx.get('Action', '')).upper()
                            tx_shares = abs(float(str(tx.get('Shares', 0)).replace(',','')))

                            if action == 'BUY':
                                price = float(str(tx.get('Amount', tx.get('Price', 0.0))).replace('$','').replace(',',''))
                                if price == 0: price = float(row['Avg_Entry'])
                                pyr_inventory.append({'qty': tx_shares, 'price': price})
                            elif action == 'SELL':
                                qty_to_sell = tx_shares
                                while qty_to_sell > 0 and pyr_inventory:
                                    last = pyr_inventory[-1]
                                    take = min(qty_to_sell, last['qty'])
                                    last['qty'] -= take
                                    qty_to_sell -= take
                                    if last['qty'] < 0.00001: pyr_inventory.pop()

                        total_rem_shares = sum(i['qty'] for i in pyr_inventory)
                        total_rem_cost = sum(i['qty'] * i['price'] for i in pyr_inventory)
                        if total_rem_shares > 0:
                            lifo_cost = total_rem_cost / total_rem_shares

                pyr_avg_cost = lifo_cost

                auto_price, auto_atr_pyr = fetch_price_and_atr(pyr_ticker)
                if auto_price == 0: auto_price = float(row.get('Current_Price', 0))

                pyr_price = c2.number_input("Current Price ($)", value=auto_price, step=0.1, key=f"pyr_px_{pyr_ticker}")
                c3.metric("Avg Cost", f"${pyr_avg_cost:,.2f}")

                st.markdown("---")

                # --- ATR & Equity Inputs ---
                e1, e2 = st.columns(2)
                pyr_equity = e1.number_input("Account Equity (NLV)", value=equity, step=1000.0, key="pyr_eq")
                pyr_atr_pct = e2.number_input("ATR % (21-Day)", value=auto_atr_pyr, step=0.1, help="Auto-calculated from 21-day price data. Adjust if needed.", key=f"pyr_atr_{pyr_ticker}")

                if st.button("Run Pyramid Analysis", type="primary", key="pyr_btn"):
                    if pyr_ticker and pyr_price > 0 and pyr_atr_pct > 0 and pyr_inventory:

                        # === 1. LAST BUY ANALYSIS ===
                        last_buy = pyr_inventory[-1]
                        last_buy_price = last_buy['price']
                        last_buy_profit_pct = ((pyr_price - last_buy_price) / last_buy_price) * 100

                        # Get last buy date from transaction history
                        last_buy_date = "N/A"
                        if not df_d.empty:
                            buy_txs = df_d[(df_d['Trade_ID'] == pyr_trade_id) & (df_d['Action'].str.upper() == 'BUY')].sort_values('Date')
                            if not buy_txs.empty:
                                last_buy_date = str(buy_txs.iloc[-1].get('Date', 'N/A'))[:10]

                        st.markdown(f"### 📊 Pyramid Analysis: {pyr_ticker}")

                        # --- Last Buy Info ---
                        b1, b2, b3 = st.columns(3)
                        b1.metric("Last Buy Price", f"${last_buy_price:,.2f}", f"Date: {last_buy_date}")
                        b2.metric("Last Buy P&L", f"{last_buy_profit_pct:.2f}%",
                                  f"${pyr_price - last_buy_price:,.2f}/share",
                                  delta_color="normal" if last_buy_profit_pct >= 0 else "inverse")
                        cushion_pct = ((pyr_price - pyr_avg_cost) / pyr_avg_cost) * 100 if pyr_avg_cost > 0 else 0
                        b3.metric("Total Cushion", f"{cushion_pct:.2f}%", f"Avg Cost: ${pyr_avg_cost:,.2f}")

                        st.markdown("---")

                        # === 2. PYRAMID PACING RULE ===
                        base_add_pct = 0.20
                        threshold_pct = 5.0

                        if last_buy_profit_pct >= threshold_pct:
                            scale_factor = 1.0
                        elif last_buy_profit_pct > 0:
                            scale_factor = last_buy_profit_pct / threshold_pct
                        else:
                            scale_factor = 0.0

                        pyramid_max_shares = int(pyr_shares * base_add_pct * scale_factor)

                        # === 3. ATR / HARD CAP CEILING (same as vol sizer) ===
                        tier_name = "Tier 3 (Defense)"
                        tol_pct = 0.50
                        atr_multiplier = 1.0

                        if cushion_pct >= 20.0:
                            tier_name = "Tier 1 (High Cushion)"
                            tol_pct = 1.00
                            atr_multiplier = 2.0
                        elif cushion_pct >= 5.0:
                            tier_name = "Tier 2 (Moderate)"
                            tol_pct = 0.65
                            atr_multiplier = 1.5

                        daily_risk_budget = pyr_equity * (tol_pct / 100)
                        atr_risk_budget = daily_risk_budget * atr_multiplier
                        atr_decimal = pyr_atr_pct / 100
                        max_shares_atr = int(atr_risk_budget / (pyr_price * atr_decimal))
                        max_shares_cap = int((pyr_equity * 0.20) / pyr_price)
                        position_ceiling = min(max_shares_atr, max_shares_cap)
                        room_to_add = max(0, position_ceiling - int(pyr_shares))

                        # === 4. FINAL PYRAMID ALLOWED ===
                        pyramid_allowed = min(pyramid_max_shares, room_to_add)
                        pyramid_value = pyramid_allowed * pyr_price

                        # === 5. DISPLAY ===
                        st.markdown("### 🔺 Pyramid Calculation")

                        p1, p2, p3 = st.columns(3)
                        base_add = int(pyr_shares * base_add_pct)
                        p1.metric("Base Add (20%)", f"{base_add} shs", f"20% of {int(pyr_shares)} shares")
                        p2.metric("Scale Factor", f"{scale_factor:.0%}",
                                  f"Last buy up {last_buy_profit_pct:.1f}% (need 5%)")
                        p3.metric("Pyramid Max", f"{pyramid_max_shares} shs",
                                  f"After scaling")

                        st.markdown("---")

                        r1, r2, r3 = st.columns(3)
                        r1.metric("Position Ceiling", f"{position_ceiling} shs",
                                  f"{tier_name} | {atr_multiplier:.1f}x ATR", delta_color="off")
                        r2.metric("Current Position", f"{int(pyr_shares)} shs",
                                  f"{pyr_shares * pyr_price / pyr_equity * 100:.1f}% Weight")
                        r3.metric("Room to Add", f"{room_to_add} shs",
                                  f"Before hitting ceiling")

                        # === 6. VERDICT ===
                        st.markdown("### 🏛️ The Verdict")

                        if scale_factor == 0:
                            st.error(f"🚫 **NO ADD** — Last buy is {'down' if last_buy_profit_pct < 0 else 'flat'} ({last_buy_profit_pct:.2f}%). Wait for it to work.")
                        elif pyramid_allowed == 0 and pyramid_max_shares > 0:
                            st.warning(f"⚠️ **NO ROOM** — Pyramid says {pyramid_max_shares} shares, but position is at ATR/cap ceiling ({position_ceiling} shs).")
                        elif pyramid_allowed > 0:
                            binding = "Pyramid pace" if pyramid_allowed == pyramid_max_shares else "ATR/Cap ceiling"
                            st.success(f"✅ **ADD {pyramid_allowed} shares** (${pyramid_value:,.0f}) — Limited by: {binding}")
                            v1, v2, v3 = st.columns(3)
                            new_total = int(pyr_shares) + pyramid_allowed
                            new_weight = (new_total * pyr_price / pyr_equity) * 100
                            v1.metric("Add Shares", f"{pyramid_allowed} shs", f"${pyramid_value:,.0f}")
                            v2.metric("New Total", f"{new_total} shs", f"{new_weight:.1f}% Weight")
                            new_avg = (pyr_avg_cost * pyr_shares + pyr_price * pyramid_allowed) / new_total
                            v3.metric("New Avg Cost", f"${new_avg:,.2f}", f"From ${pyr_avg_cost:,.2f}")

                            # Store results for Send to Log Buy button
                            st.session_state['_pyr_result'] = {
                                'ticker': pyr_ticker, 'trade_id': pyr_trade_id,
                                'shares': pyramid_allowed, 'price': pyr_price,
                            }
                        else:
                            st.info("ℹ️ Scale factor resulted in 0 shares. Last buy needs more profit before adding.")

                    elif not pyr_inventory:
                        st.error("No buy transactions found for this position.")
                    else:
                        st.error("Please ensure Price and ATR are entered correctly.")

                # --- SEND TO LOG BUY (outside button block) ---
                if '_pyr_result' in st.session_state:
                    r = st.session_state['_pyr_result']
                    st.markdown("---")
                    if st.button(f"📝 Send to Log Buy — {r['ticker']} (+{r['shares']} shs @ ${r['price']:.2f})", key="pyr_send_logbuy", type="secondary", use_container_width=True):
                        st.session_state['ps_ticker'] = r['ticker']
                        st.session_state['ps_trade_id'] = r['trade_id']
                        st.session_state['ps_shares'] = r['shares']
                        st.session_state['ps_price'] = r['price']
                        st.session_state['ps_stop'] = 0.0
                        st.session_state['ps_action'] = 'scale_in'
                        del st.session_state['_pyr_result']
                        st.session_state.page = "Log Buy"
                        st.rerun()
            else:
                st.info("No open positions found. Open a position first to use the Pyramid Sizer.")
        else:
            st.info("No trade data available.")

# ==============================================================================
# PAGE: LOG BUY (Standalone)
# ==============================================================================
elif page == "Log Buy":
    page_header("Log Buy", CURR_PORT_NAME, "🟢")
    df_d, df_s = load_trade_data()

    st.caption("Live Entry Calculator")

    # --- Pre-fill from Position Sizer ---
    ps_prefill = st.session_state.pop('ps_action', None)
    ps_ticker = st.session_state.pop('ps_ticker', '')
    ps_shares = st.session_state.pop('ps_shares', 0)
    ps_price = st.session_state.pop('ps_price', 0.0)
    ps_stop = st.session_state.pop('ps_stop', 0.0)
    ps_trade_id = st.session_state.pop('ps_trade_id', '')
    ps_risk_budget = st.session_state.pop('ps_risk_budget', 0.0)

    if ps_prefill:
        st.session_state['b_tick'] = ps_ticker
        st.session_state['b_shs'] = int(ps_shares)
        st.session_state['b_px'] = float(ps_price)
        st.session_state['b_stop_val'] = float(ps_stop) if ps_stop > 0 else 0.0
        if ps_prefill == 'scale_in' and ps_trade_id:
            st.session_state['b_id'] = ps_trade_id
        st.success(f"Pre-filled from Position Sizer: **{ps_ticker}** — {int(ps_shares)} shares @ ${ps_price:.2f}" +
                   (f" | Stop: ${ps_stop:.2f}" if ps_stop > 0 else ""))

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
    default_action_idx = 1 if ps_prefill == 'scale_in' else 0
    trade_type = c_top1.radio("Action Type", ["Start New Campaign", "Scale In (Add to Existing)"], index=default_action_idx, horizontal=True)

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
        b_rule = st.selectbox("Buy Rule", BUY_RULES, index=None, placeholder="Type to search rules...")
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
        b_rule = st.selectbox("Add Rule", BUY_RULES, index=None, placeholder="Type to search rules...")

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
        sizing_mode_buy = rb1.radio("Sizing Mode",
            ["🛡️ Defense (0.50%)", "⚖️ Normal (0.75%)", "⚔️ Offense (1.00%)"],
            index=1, key="b_sizing_mode")
        if sizing_mode_buy.startswith("⚔️"):
            risk_pct_input = 1.00
        elif sizing_mode_buy.startswith("⚖️"):
            risk_pct_input = 0.75
        else:
            risk_pct_input = 0.50
        risk_budget_dol = def_equity * (risk_pct_input / 100)

        rb2.metric("Account Equity (Prev)", f"${def_equity:,.2f}")
        rb3.metric("Risk Budget", f"${risk_budget_dol:.2f}", f"{risk_pct_input}% of equity")
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
    entry_charts = []
    position_charts = []
    if R2_AVAILABLE:
        st.markdown("#### 📸 Chart Documentation (Optional)")

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            entry_charts = st.file_uploader(
                "📊 Entry Charts (Weekly / Daily)",
                type=['png', 'jpg', 'jpeg'],
                key='b_entry_charts',
                accept_multiple_files=True,
                help="Upload weekly or daily charts documenting your entry setup"
            )
            if entry_charts:
                st.caption(f"✅ {len(entry_charts)} file(s) selected")
        with img_col2:
            position_charts = st.file_uploader(
                "🔄 Position Changes (Add-ons / Trims / Exits)",
                type=['png', 'jpg', 'jpeg'],
                key='b_position_charts',
                accept_multiple_files=True,
                help="Upload charts showing add-on entries, partial sells, or full exits"
            )
            if position_charts:
                st.caption(f"✅ {len(position_charts)} file(s) selected")

    # --- MARKETSURGE SCREENSHOT (Fundamental Extraction) ---
    ms_screenshot = None
    if check_vision_available():
        st.markdown("#### 🔬 MarketSurge Fundamentals (Optional)")
        st.caption("Upload a MarketSurge screenshot to auto-extract ratings and fundamentals via AI.")
        ms_screenshot = st.file_uploader(
            "MarketSurge Screenshot",
            type=['png', 'jpg', 'jpeg'],
            key='b_ms_screenshot',
            help="Upload a MarketSurge stock detail screenshot — AI will extract EPS Rating, Composite Rating, RS Rating, and more."
        )
        if ms_screenshot:
            st.caption("✅ Screenshot selected")
            if st.button("🔬 Extract & Preview Fundamentals", key="btn_extract_preview"):
                with st.spinner("Analyzing screenshot with AI..."):
                    ms_screenshot.seek(0)
                    img_bytes = ms_screenshot.read()
                    extracted = vision_extract.extract_fundamentals(img_bytes, ms_screenshot.name)
                    if extracted:
                        st.session_state['_ms_extracted'] = extracted
                    else:
                        st.error("Could not extract data from this screenshot. Make sure it's a MarketSurge stock detail page.")

            # Show extracted preview (persists through reruns)
            if '_ms_extracted' in st.session_state:
                extracted = st.session_state['_ms_extracted']
                st.success(f"**Extracted: {extracted.get('ticker', 'Unknown')}**")
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Composite", extracted.get('composite_rating', 'N/A'))
                rc2.metric("EPS Rating", extracted.get('eps_rating', 'N/A'))
                rc3.metric("RS Rating", extracted.get('rs_rating', 'N/A'))
                rc4.metric("Acc/Dis", extracted.get('acc_dis_rating', 'N/A'))

                rc5, rc6, rc7, rc8 = st.columns(4)
                rc5.metric("SMR", extracted.get('smr_rating', 'N/A'))
                rc6.metric("Group RS", extracted.get('group_rs_rating', 'N/A'))
                rc7.metric("EPS Growth", f"{extracted['eps_growth_rate']}%" if extracted.get('eps_growth_rate') else "N/A")
                rc8.metric("U/D Vol", extracted.get('ud_vol_ratio', 'N/A'))

                if extracted.get('industry_group'):
                    st.caption(f"Industry: {extracted['industry_group']} (Rank #{extracted.get('industry_group_rank', '?')})")
                if extracted.get('funds_own_pct') is not None:
                    st.caption(f"Ownership — Funds: {extracted['funds_own_pct']}% | Banks: {extracted.get('banks_own_pct', '?')}% | Mgmt: {extracted.get('mgmt_own_pct', '?')}%")

                # Show annual EPS if available
                if extracted.get('annual_eps'):
                    eps_str = " | ".join([f"{e['year']}: ${e['eps']}" for e in extracted['annual_eps'] if e.get('year') and e.get('eps')])
                    if eps_str:
                        st.caption(f"Annual EPS: {eps_str}")

                st.caption("Data will be saved to DB when you click LOG BUY ORDER")

    if st.button("LOG BUY ORDER", type="primary", use_container_width=True):

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
            if R2_AVAILABLE and USE_DATABASE and (entry_charts or position_charts):
                images_uploaded = 0
                try:
                    for f in entry_charts:
                        url = r2.upload_image(f, portfolio, b_id, b_tick, 'entry')
                        if url:
                            db.save_trade_image(portfolio, b_id, b_tick, 'entry', url, f.name)
                            images_uploaded += 1

                    for f in position_charts:
                        url = r2.upload_image(f, portfolio, b_id, b_tick, 'position')
                        if url:
                            db.save_trade_image(portfolio, b_id, b_tick, 'position', url, f.name)
                            images_uploaded += 1

                    if images_uploaded:
                        st.success(f"📸 Uploaded {images_uploaded} chart(s)")
                    else:
                        st.warning("Charts were selected but upload failed")
                except Exception as e:
                    st.error(f"Image upload error: {str(e)}")

            # --- SAVE FUNDAMENTALS FROM MARKETSURGE (if extracted or screenshot provided) ---
            if USE_DATABASE and (ms_screenshot or '_ms_extracted' in st.session_state):
                try:
                    # Use already-extracted data if available, otherwise extract now
                    extracted = st.session_state.pop('_ms_extracted', None)

                    ms_image_id = None
                    if ms_screenshot and R2_AVAILABLE:
                        ms_screenshot.seek(0)
                        ms_url = r2.upload_image(ms_screenshot, portfolio, b_id, b_tick, 'marketsurge')
                        if ms_url:
                            ms_image_id = db.save_trade_image(portfolio, b_id, b_tick, 'marketsurge', ms_url, ms_screenshot.name)
                            # Also save as entry chart so it appears in Entry Charts section
                            db.save_trade_image(portfolio, b_id, b_tick, 'entry', ms_url, ms_screenshot.name)

                    if not extracted and ms_screenshot and check_vision_available():
                        with st.spinner("🔬 Extracting fundamentals..."):
                            ms_screenshot.seek(0)
                            img_bytes = ms_screenshot.read()
                            extracted = vision_extract.extract_fundamentals(img_bytes, ms_screenshot.name)

                    if extracted:
                        db.save_trade_fundamentals(portfolio, b_id, b_tick, extracted, ms_image_id)
                        st.success("🔬 Fundamentals saved to database!")
                    elif ms_screenshot:
                        st.warning("⚠️ Could not extract data. Screenshot saved for manual review.")
                except Exception as e:
                    st.warning(f"⚠️ Fundamental extraction failed: {e}. Trade was logged successfully.")

            st.success(f"✅ EXECUTED: Bought {b_shs} {b_tick} @ ${b_px}")
            for k in ['b_tick','b_id','b_shs','b_px','b_note','b_trx','b_stop_val','b_weekly_chart','b_daily_chart','b_ms_screenshot']:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
        else:
            st.error("⚠️ Missing required fields: Ticker and Trade ID are required.")

# ==============================================================================
# PAGE: LOG SELL (Standalone)
# ==============================================================================
elif page == "Log Sell":
    page_header("Log Sell", CURR_PORT_NAME, "🔴")
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

            # --- CHART UPLOADS (Optional) ---
            sell_position_charts = []
            if R2_AVAILABLE:
                st.markdown("#### 📸 Position Changes (Optional)")
                sell_position_charts = st.file_uploader(
                    "🔄 Upload charts (Partial Sells / Full Exits)",
                    type=['png', 'jpg', 'jpeg'],
                    key='s_position_charts',
                    accept_multiple_files=True,
                    help="Upload charts showing partial sells or full exits"
                )
                if sell_position_charts:
                    st.caption(f"✅ {len(sell_position_charts)} file(s) selected")

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

                        # Upload charts
                        charts_uploaded = 0
                        if R2_AVAILABLE:
                            try:
                                for f in sell_position_charts:
                                    url = r2.upload_image(f, CURR_PORT_NAME, s_id, s_tick, 'position')
                                    if url:
                                        db.save_trade_image(CURR_PORT_NAME, s_id, s_tick, 'position', url, f.name)
                                        charts_uploaded += 1
                            except Exception as chart_err:
                                st.warning(f"⚠️ Sell saved but chart upload failed: {chart_err}")

                        chart_msg = f" | {charts_uploaded} chart(s) uploaded" if charts_uploaded > 0 else ""
                        st.session_state.sell_success = f"✅ Sold! Transaction ID: {s_trx}{chart_msg} | Saved to database"
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
    page_header("Trade Manager", CURR_PORT_NAME, "📝")

    df_d, df_s = load_trade_data()

    valid_sum_cols = ['Trade_ID', 'Ticker', 'Status', 'Open_Date', 'Shares', 'Avg_Entry', 'Total_Cost', 'Unrealized_PL', 'Return_Pct', 'Rule', 'Buy_Notes', 'Sell_Rule']
    valid_sum_cols = [c for c in valid_sum_cols if c in df_s.columns]

    # --------------------------------------------------------------------------
    # TAB LIST (Operational tabs only)
    # --------------------------------------------------------------------------
    tab3, tab4, tab5, tab6, tab8, tab9, tab_cy, tab_cy_detail, tab10 = st.tabs([
        "Update Prices",
        "Edit Transaction",
        "Database Health",
        "Delete Trade",
        "Active Campaign Detailed",
        "Detailed Trade Log",
        "CY Campaigns (2026)",
        "Export",
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

    # --- TAB 3: RISK CONTROL CENTER ---
    with tab3:
        st.markdown("### 🛑 Rapid Stop Adjustment")
        open_pos = df_s[df_s['Status'] == 'OPEN'].sort_values('Ticker')
        if not open_pos.empty:
            def get_current_stop_display(tid):
                try:
                    stops = df_d[df_d['Trade_ID'] == tid]['Stop_Loss'].dropna()
                    val = float(stops.iloc[-1]) if not stops.empty else 0.0
                    if pd.isna(val):
                        return 0.0
                    return val
                except: return 0.0

            opts_dict = {f"{r['Ticker']} (Current: ${get_current_stop_display(r['Trade_ID']):.2f})": r['Trade_ID'] for _, r in open_pos.iterrows()}
            sel_label = st.selectbox("Select Position to Protect", list(opts_dict.keys()), key="rapid_stop_select")
            sel_id = opts_dict[sel_label]
            curr_stop_val = get_current_stop_display(sel_id)

            c_up1, c_up2, c_up3 = st.columns(3)
            # Use dynamic key tied to trade_id so value resets when selection changes
            new_stop_price = c_up1.number_input("New Hard Stop Price ($)", value=float(curr_stop_val), min_value=0.0, step=0.01, format="%.2f", key=f"rapid_stop_val_{sel_id}")
            
            if c_up3.button("UPDATE STOP LOSS"):
                mask = (df_d['Trade_ID'] == sel_id) & (df_d['Action'] == 'BUY')
                buy_rows = df_d[mask]
                if not buy_rows.empty:
                    if USE_DATABASE:
                        try:
                            def to_native(v):
                                if hasattr(v, 'item'): return v.item()
                                return v

                            # Update ALL buy lots for this campaign
                            for row_idx in buy_rows.index:
                                db_id = df_d.at[row_idx, '_DB_ID']
                                if pd.isna(db_id):
                                    continue
                                update_dict = {
                                    'Trade_ID': to_native(df_d.at[row_idx, 'Trade_ID']),
                                    'Ticker': to_native(df_d.at[row_idx, 'Ticker']),
                                    'Action': to_native(df_d.at[row_idx, 'Action']),
                                    'Date': to_native(df_d.at[row_idx, 'Date']),
                                    'Shares': to_native(df_d.at[row_idx, 'Shares']),
                                    'Amount': to_native(df_d.at[row_idx, 'Amount']),
                                    'Value': to_native(df_d.at[row_idx, 'Value']),
                                    'Rule': to_native(df_d.at[row_idx, 'Rule']),
                                    'Notes': to_native(df_d.at[row_idx, 'Notes']),
                                    'Stop_Loss': float(new_stop_price),
                                    'Trx_ID': to_native(df_d.at[row_idx, 'Trx_ID'])
                                }
                                db.update_detail_row(CURR_PORT_NAME, int(db_id), update_dict)

                            # Recalculate summary so Active Campaign Manager reflects the new stop
                            df_d_temp = db.load_details(CURR_PORT_NAME, sel_id)
                            df_s_temp = db.load_summary(CURR_PORT_NAME)
                            df_d_temp, df_s_temp = update_campaign_summary(sel_id, df_d_temp, df_s_temp)
                            summary_matches = df_s_temp[df_s_temp['Trade_ID'].astype(str) == str(sel_id)]
                            if not summary_matches.empty:
                                db.save_summary_row(CURR_PORT_NAME, summary_matches.iloc[0].to_dict())

                            # Invalidate cached loaders so the next render picks up new stops
                            try:
                                db.load_details.clear()
                                db.load_summary.clear()
                            except Exception:
                                pass
                            st.toast(f"✅ {sel_label.split(' (')[0]} stop → ${new_stop_price:.2f} ({len(buy_rows)} lot(s))", icon="🛡️")
                        except Exception as e:
                            st.error(f"❌ Database update failed: {str(e)}")
                    else:
                        # CSV fallback — update all buy lots
                        for row_idx in buy_rows.index:
                            df_d.at[row_idx, 'Stop_Loss'] = new_stop_price
                        secure_save(df_d, DETAILS_FILE)
                        df_d, df_s = update_campaign_summary(sel_id, df_d, df_s)
                        secure_save(df_s, SUMMARY_FILE)
                        st.toast(f"✅ {sel_label.split(' (')[0]} stop → ${new_stop_price:.2f}", icon="🛡️")
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

                                        # Prepare update dict (cast numpy types to native Python)
                                        update_dict = {
                                            'Trade_ID': str(df_d.at[idx, 'Trade_ID']),
                                            'Ticker': str(df_d.at[idx, 'Ticker']),
                                            'Action': str(df_d.at[idx, 'Action']),
                                            'Date': str(df_d.at[idx, 'Date']),
                                            'Shares': float(df_d.at[idx, 'Shares']),
                                            'Amount': float(df_d.at[idx, 'Amount']),
                                            'Value': float(df_d.at[idx, 'Value']),
                                            'Rule': str(df_d.at[idx, 'Rule']),
                                            'Notes': str(new_note),
                                            'Stop_Loss': float(new_stop),
                                            'Trx_ID': str(df_d.at[idx, 'Trx_ID'])
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
                # Build lookup: Trade_ID -> Ticker, Status
                id_ticker_map = {}
                for tid in all_ids:
                    try:
                        ticker = df_d[df_d['Trade_ID'].astype(str) == tid]['Ticker'].iloc[0]
                        status = df_s[df_s['Trade_ID'].astype(str) == tid]['Status'].iloc[0] if not df_s.empty else ''
                        id_ticker_map[tid] = (ticker, status)
                    except:
                        id_ticker_map[tid] = ('???', '')

                # Search filters
                sf1, sf2, sf3 = st.columns([2, 1, 1])
                ticker_search = sf1.text_input("Search by Ticker", placeholder="e.g. GOOGL", key="edit_ticker_search").strip().upper()
                all_tickers = sorted(set(t for t, s in id_ticker_map.values() if t != '???'))
                status_filter = sf2.selectbox("Status", ["All", "OPEN", "CLOSED"], key="edit_status_filter")
                sort_order = sf3.selectbox("Sort", ["Newest First", "Oldest First", "Ticker A-Z"], key="edit_sort_order")

                # Apply filters
                filtered_ids = all_ids
                if ticker_search:
                    filtered_ids = [tid for tid in filtered_ids if ticker_search in id_ticker_map[tid][0].upper()]
                if status_filter != "All":
                    filtered_ids = [tid for tid in filtered_ids if id_ticker_map[tid][1] == status_filter]

                # Apply sort
                if sort_order == "Oldest First":
                    filtered_ids = list(reversed(filtered_ids))
                elif sort_order == "Ticker A-Z":
                    filtered_ids = sorted(filtered_ids, key=lambda x: id_ticker_map[x][0])

                if not filtered_ids:
                    st.warning("No trades match your search.")

                def fmt_func(x):
                    ticker, status = id_ticker_map.get(x, ('???', ''))
                    flag = "🟢" if status == "OPEN" else "⚪"
                    return f"{flag} {ticker} | {x}"

                edit_id = st.selectbox("Select Trade ID to Edit", filtered_ids, format_func=fmt_func) if filtered_ids else None
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

        # --- ORPHAN CLEANUP ---
        st.markdown("---")
        st.subheader("🧹 Orphan Cleanup")
        if not df_d.empty and not df_s.empty:
            det_ids = set(df_d['Trade_ID'].unique())
            sum_ids = set(df_s['Trade_ID'].unique())
            orphan_details = det_ids - sum_ids  # Details with no summary
            if orphan_details:
                st.warning(f"⚠️ Found **{len(orphan_details)} orphaned campaign(s)** in details with no matching summary:")
                for oid in sorted(orphan_details):
                    orphan_rows = df_d[df_d['Trade_ID'] == oid]
                    ticker = orphan_rows['Ticker'].iloc[0] if not orphan_rows.empty else '?'
                    st.write(f"  - **{ticker}** ({oid}) — {len(orphan_rows)} transaction(s)")
                if st.button("DELETE ORPHANED DETAIL ROWS", type="secondary"):
                    if USE_DATABASE:
                        try:
                            for oid in orphan_details:
                                db.delete_trade(CURR_PORT_NAME, oid)
                            db.load_details.clear()
                        except Exception as e:
                            st.error(f"❌ DB cleanup failed: {e}")
                    df_d = df_d[~df_d['Trade_ID'].isin(orphan_details)]
                    secure_save(df_d, DETAILS_FILE)
                    st.success(f"✅ Removed {len(orphan_details)} orphaned campaign(s)")
                    st.rerun()
            else:
                st.success("✅ No orphaned records found.")

    # --- TAB 6: DELETE TRADE ---
    with tab6:
        st.warning("⚠️ **Danger Zone**: Deleting a trade will permanently remove ALL transactions for that campaign.")

        del_opts = df_s.apply(lambda r: f"{r['Trade_ID']} — {r['Ticker']} ({r['Status']})", axis=1).tolist() if not df_s.empty else []
        del_sel = st.selectbox("ID to Delete", del_opts, index=None, placeholder="Type to search trades...")
        del_id = del_sel.split(" — ")[0] if del_sel else None

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
                        db.delete_trade(CURR_PORT_NAME, del_id)
                        # Force clear all caches to ensure fresh data on reload
                        db.load_summary.clear()
                        db.load_details.clear()
                        st.success(f"🗑️ Deleted {del_id} from database")
                    except Exception as e:
                        st.warning(f"⚠️ Database delete failed: {e}. Trying CSV cleanup.")

                # Perform deletion from DataFrames
                df_s = df_s[df_s['Trade_ID']!=del_id]
                df_d = df_d[df_d['Trade_ID']!=del_id]
                secure_save(df_s, SUMMARY_FILE)
                secure_save(df_d, DETAILS_FILE)
                st.rerun()

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
                buy_realized_pl = {}   # Back-attribute realized P&L to each BUY lot
                buy_exit_price = {}    # Weighted avg exit price per BUY lot
                buy_shares_sold = {}   # Shares sold from each BUY lot

                fd_realized_pl = 0.0
                fd_remaining_shares = 0.0
                fd_cost_basis_sum = 0.0

                # Fetch live prices for accurate unrealized P&L (cached per-ticker 60s)
                curr_prices = {}
                open_tickers = df_s[df_s['Status']=='OPEN'][['Trade_ID','Ticker']].drop_duplicates()
                for _, r in open_tickers.iterrows():
                    px = cached_live_price(r['Ticker'])
                    if px is not None:
                        curr_prices[r['Trade_ID']] = px
                    else:
                        # Fallback to summary-derived price
                        s_row = df_s[df_s['Trade_ID'] == r['Trade_ID']]
                        if not s_row.empty and s_row.iloc[0]['Shares'] > 0:
                            val = s_row.iloc[0]['Total_Cost'] + s_row.iloc[0].get('Unrealized_PL', 0)
                            curr_prices[r['Trade_ID']] = val / s_row.iloc[0]['Shares']

                for tid in target_df['Trade_ID'].unique():
                    subset = target_df[target_df['Trade_ID'] == tid].copy()
                    subset['Date'] = pd.to_datetime(subset['Date'], errors='coerce')
                    subset['Sort_Date'] = subset['Date'].dt.normalize()
                    subset['Type_Rank'] = subset['Action'].apply(lambda x: 0 if x == 'BUY' else 1)
                    subset = subset.sort_values(['Sort_Date', 'Type_Rank', 'Date'])
                    
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

                                # Back-attribute P&L to the BUY lot
                                lot_pl = take * (sell_price - last['price'])
                                buy_realized_pl[last['idx']] = buy_realized_pl.get(last['idx'], 0) + lot_pl
                                prev_sold = buy_shares_sold.get(last['idx'], 0)
                                buy_shares_sold[last['idx']] = prev_sold + take
                                total_sold = buy_shares_sold[last['idx']]
                                prev_avg = buy_exit_price.get(last['idx'], 0)
                                buy_exit_price[last['idx']] = (prev_avg * prev_sold + sell_price * take) / total_sold

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
                # Realized_PL: SELL rows get lifo_pl_map, BUY rows get buy_realized_pl
                combined_pl = {**buy_realized_pl, **lifo_pl_map}
                display_df['Realized_PL'] = display_df.index.map(combined_pl).fillna(0)
                display_df['Exit_Price'] = display_df.index.map(buy_exit_price)
                
                # --- CALCULATE STATUS COLUMN ---
                display_df['Status'] = display_df['Remaining_Shares'].apply(lambda x: 'Open' if x > 0 else 'Closed')
                
                # --- 3. THE FLIGHT DECK (DYNAMIC) ---
                if tick_filter != "All":
                    live_px = cached_live_price(tick_filter)
                    if live_px is None:
                        live_px = (fd_cost_basis_sum / fd_remaining_shares) if fd_remaining_shares > 0 else 0.0
                    
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
                    if row['Action'] == 'BUY':
                        entry = float(row.get('Amount', row.get('Price', 0.0)))
                        if row['Remaining_Shares'] > 0:
                            # Open: unrealized return from current price
                            price = live_px if tick_filter != "All" else curr_prices.get(row['Trade_ID'], 0)
                            if entry > 0: return ((price - entry) / entry) * 100
                        elif row.name in buy_realized_pl:
                            # Closed: realized return from LIFO back-attribution
                            original_shares = abs(float(row['Shares']))
                            cost_basis = entry * original_shares
                            if cost_basis > 0: return (buy_realized_pl[row.name] / cost_basis) * 100
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
                
                final_cols = ['Trade_ID', 'Trx_ID', 'Campaign_Start', 'Date', 'Ticker', 'Action', 'Status', 'Shares', 'Remaining_Shares', 'Amount', 'Exit_Price', 'Stop_Loss', 'Value', 'Realized_PL', 'Unrealized_PL', 'Return_Pct', 'Rule', 'Notes']
                show_cols = [c for c in final_cols if c in display_df.columns]
                
                st.dataframe(
                    display_df[show_cols].sort_values(['Trade_ID', 'Date']).style
                    .format({
                        'Date': lambda x: x.strftime('%Y-%m-%d %H:%M') if isinstance(x, (pd.Timestamp, datetime)) else 'None',
                        'Campaign_Start': lambda x: x if isinstance(x, str) else (x.strftime('%Y-%m-%d %H:%M') if isinstance(x, (pd.Timestamp, datetime)) else 'None'), 
                        'Amount':'${:,.2f}', 'Exit_Price':'${:,.2f}', 'Stop_Loss':'${:,.2f}', 'Value':'${:,.2f}',
                        'Realized_PL':'${:,.2f}', 'Unrealized_PL':'${:,.2f}', 
                        'Return_Pct':'{:.2f}%', 'Remaining_Shares':'{:.0f}'
                    })
                    .map(color_pnl, subset=['Value','Realized_PL','Unrealized_PL', 'Return_Pct'])
                    .map(color_neg_value, subset=['Shares']), 
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
                        buy_attribution[idx] = {'pl': 0.0, 'sold_cost': 0.0, 'sold_val': 0.0, 'sold_qty': 0.0}
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
                            buy_attribution[last['idx']]['sold_qty'] += take
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

                def get_exit_price(idx, action):
                    if action == 'BUY' and idx in buy_attribution:
                        data = buy_attribution[idx]
                        qty = data.get('sold_qty', 0)
                        if qty > 0:
                            return data['sold_val'] / qty
                    return None

                calc_df['Lot P&L'] = calc_df.apply(lambda x: get_lifo_pl(x.name, x['Action'], x['Realized_PL']), axis=1)
                calc_df['Return %'] = calc_df.apply(lambda x: get_lifo_ret(x.name, x['Action']), axis=1)
                calc_df['Exit_Price'] = calc_df.apply(lambda x: get_exit_price(x.name, x['Action']), axis=1)
                
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
                cols = ['Trade_ID', 'Trx_ID', 'Date', 'Ticker', 'Action', 'Shares', 'Amount', 'Exit_Price', 'Value', 'Lot P&L', 'Return %', 'Rule', 'Notes']
                show_cols = [c for c in cols if c in display_df.columns]
                
                st.dataframe(
                    display_df[show_cols].sort_values(['Trade_ID', 'Date']).style
                    .format({
                        'Date': lambda x: x.strftime('%Y-%m-%d %H:%M') if isinstance(x, (pd.Timestamp, datetime)) else 'None', 
                        'Shares':'{:.0f}', 'Amount':'${:,.2f}', 'Exit_Price':'${:,.2f}', 'Value':'${:,.2f}',
                        'Lot P&L':'${:,.2f}', 'Return %':'{:.2f}%'
                    })
                    .map(color_pnl, subset=['Lot P&L', 'Return %'])
                    .map(color_neg_value, subset=['Shares']),
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
                        .map(style_status, subset=['Status'])
                        .map(style_pl, subset=['Active_PL'])
                        .map(style_compliance, subset=['Compliance'])
                        .map(style_r, subset=['R_Multiple']),
                        hide_index=True,
                        use_container_width=True
                    )
                else: st.info("No trades match your filters.")
            else: st.info("No trades found for 2026 or Rollovers.")
        else: st.warning("Summary Database is empty.")

# --- TAB CY DETAIL: 2026 DETAILED TRADE LOG ---
    with tab_cy_detail:
        st.subheader("📥 Export 2026 Trades")
        st.caption("Export 2026 campaigns (opened in 2026 + rollovers from 2025). Per-lot LIFO attribution with Core/Add classification.")

        if not df_s.empty:
            # --- 1. CY 2026 FILTER ---
            df_s['Open_DT'] = pd.to_datetime(df_s['Open_Date'], errors='coerce')
            df_s['Close_DT'] = pd.to_datetime(df_s['Closed_Date'], errors='coerce')
            cutoff_2026 = pd.Timestamp("2026-01-01")

            cy_mask_detail = (
                (df_s['Open_DT'] >= cutoff_2026) |
                (df_s['Status'] == 'OPEN') |
                (df_s['Close_DT'] >= cutoff_2026)
            )
            _cy_all = df_s[cy_mask_detail].copy()

            # --- STATUS FILTER ---
            _open_count = len(_cy_all[_cy_all['Status'].str.upper() == 'OPEN'])
            _closed_count = len(_cy_all[_cy_all['Status'].str.upper() == 'CLOSED'])
            st.caption(f"**{len(_cy_all)}** total campaigns  •  **{_open_count}** open  •  **{_closed_count}** closed")

            _export_filter = st.radio("Filter:", ["All 2026 Trades", "Open Only", "Closed Only"], horizontal=True, key="export_filter_2026")

            if _export_filter == "Open Only":
                _cy_filtered = _cy_all[_cy_all['Status'].str.upper() == 'OPEN']
            elif _export_filter == "Closed Only":
                _cy_filtered = _cy_all[_cy_all['Status'].str.upper() == 'CLOSED']
            else:
                _cy_filtered = _cy_all

            cy_ids = _cy_filtered['Trade_ID'].unique().tolist()
            cy_detail_df = df_d[df_d['Trade_ID'].isin(cy_ids)].copy()

            if not cy_detail_df.empty:
                # --- SUMMARY DOWNLOAD (quick) ---
                _summary_cols = [c for c in ['Trade_ID', 'Ticker', 'Status', 'Open_Date', 'Closed_Date',
                                 'Total_Shares', 'Avg_Entry', 'Avg_Exit', 'Total_Cost', 'Realized_PL',
                                 'Return_Pct', 'Rule', 'Sell_Rule', 'Stop_Loss'] if c in _cy_filtered.columns]
                _summary_csv = _cy_filtered[_summary_cols].to_csv(index=False)

                _s1, _s2 = st.columns(2)
                with _s1:
                    st.download_button(
                        f"📥 Download Trade Summary ({len(_cy_filtered)} campaigns)",
                        data=_summary_csv,
                        file_name=f"2026_trade_summary_{_export_filter.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="export_summary_btn"
                    )
                with _s2:
                    _txn_csv = cy_detail_df.to_csv(index=False)
                    st.download_button(
                        f"📥 Download Transaction Details ({len(cy_detail_df)} rows)",
                        data=_txn_csv,
                        file_name=f"2026_transactions_{_export_filter.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="export_txn_btn"
                    )

                st.markdown("---")

                # --- DETAILED EXPORT: LIFO + Core/Add for filtered trades ---
                if st.button(f"📥 Export Detailed LIFO Log ({len(cy_ids)} campaigns)", key="cy_export_btn"):
                    all_rows = []
                    for tid in cy_ids:
                        t_df = cy_detail_df[cy_detail_df['Trade_ID'] == tid].copy()
                        if t_df.empty: continue
                        t_df['Date'] = pd.to_datetime(t_df['Date'], errors='coerce')
                        t_df['Type_Rank'] = t_df['Action'].apply(lambda x: 0 if x == 'BUY' else 1)
                        t_df = t_df.sort_values(['Date', 'Type_Rank'])
                        t_calc = t_df.reset_index(drop=True)

                        # LIFO engine
                        inv = []
                        ba = {}
                        for ix, r in t_calc.iterrows():
                            if r['Action'] == 'BUY':
                                inv.append({'idx': ix, 'price': r['Amount'], 'qty': r['Shares']})
                                ba[ix] = {'pl': 0.0, 'sold_cost': 0.0, 'sold_val': 0.0, 'sold_qty': 0.0}
                            elif r['Action'] == 'SELL':
                                ts = r['Shares']
                                sp = r['Amount']
                                while ts > 0 and inv:
                                    lt = inv.pop()
                                    tk = min(ts, lt['qty'])
                                    sc = tk * lt['price']
                                    sr = tk * sp
                                    ba[lt['idx']]['pl'] += sr - sc
                                    ba[lt['idx']]['sold_cost'] += sc
                                    ba[lt['idx']]['sold_val'] += sr
                                    ba[lt['idx']]['sold_qty'] += tk
                                    lt['qty'] -= tk
                                    ts -= tk
                                    if lt['qty'] > 0.0001: inv.append(lt)

                        # Compute columns
                        for ix, r in t_calc.iterrows():
                            if r['Action'] == 'SELL':
                                t_calc.at[ix, 'Lot P&L'] = r.get('Realized_PL', 0.0)
                                t_calc.at[ix, 'Return %'] = 0.0
                                t_calc.at[ix, 'Exit_Price'] = None
                            elif ix in ba:
                                d = ba[ix]
                                t_calc.at[ix, 'Lot P&L'] = d['pl']
                                t_calc.at[ix, 'Return %'] = ((d['sold_val'] - d['sold_cost']) / d['sold_cost'] * 100) if d['sold_cost'] > 0 else 0.0
                                t_calc.at[ix, 'Exit_Price'] = (d['sold_val'] / d['sold_qty']) if d['sold_qty'] > 0 else None
                            else:
                                t_calc.at[ix, 'Lot P&L'] = 0.0
                                t_calc.at[ix, 'Return %'] = 0.0
                                t_calc.at[ix, 'Exit_Price'] = None

                        # Core/Add classification
                        t_calc['Category'] = ''
                        b1 = t_calc[(t_calc['Action'] == 'BUY') & (t_calc['Trx_ID'].astype(str).str.upper().str.startswith('B'))]
                        if not b1.empty:
                            bp = float(b1.iloc[0]['Amount'])
                            bl = bp * 0.975
                            bh = bp * 1.025
                            t_calc['B1_Price'] = bp
                            t_calc['Core_Band_Low'] = bl
                            t_calc['Core_Band_High'] = bh
                            for ix, r in t_calc.iterrows():
                                if r['Action'] == 'BUY':
                                    t_calc.at[ix, 'Category'] = 'Core' if bl <= r['Amount'] <= bh else 'Add'

                        # Trade status from summary
                        s_row = df_s[df_s['Trade_ID'] == tid]
                        if not s_row.empty:
                            t_calc['Trade_Status'] = s_row.iloc[0]['Status']
                            t_calc['Open_Date'] = s_row.iloc[0].get('Open_Date', '')
                            t_calc['Closed_Date'] = s_row.iloc[0].get('Closed_Date', '')

                        # Negate shares/value for sells
                        t_calc['Shares'] = t_calc.apply(lambda x: -x['Shares'] if x['Action'] == 'SELL' else x['Shares'], axis=1)
                        if 'Value' in t_calc.columns:
                            t_calc['Value'] = t_calc.apply(lambda x: -x['Value'] if x['Action'] == 'SELL' else x['Value'], axis=1)

                        all_rows.append(t_calc)

                    if all_rows:
                        export_df = pd.concat(all_rows, ignore_index=True)
                        export_cols = ['Trade_ID', 'Trx_ID', 'Date', 'Ticker', 'Action', 'Category', 'Trade_Status',
                                       'Shares', 'Amount', 'Exit_Price', 'Value', 'Lot P&L', 'Return %',
                                       'B1_Price', 'Core_Band_Low', 'Core_Band_High',
                                       'Open_Date', 'Closed_Date', 'Stop_Loss', 'Rule', 'Notes']
                        export_cols = [c for c in export_cols if c in export_df.columns]
                        export_df = export_df[export_cols].sort_values(['Ticker', 'Trade_ID', 'Date'])

                        csv_data = export_df.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=csv_data,
                            file_name="2026_detailed_trade_log.csv",
                            mime="text/csv",
                            key="cy_download_csv"
                        )
                        st.success(f"✅ Ready! {len(export_df)} rows across {len(cy_ids)} campaigns.")
                    else:
                        st.warning("No data to export.")

                st.markdown("---")

                # --- 2. TWO-STAGE FILTER ---
                cy_tickers = sorted(cy_detail_df['Ticker'].dropna().unique().tolist())
                st.markdown(f"**{len(cy_tickers)} tickers** across **{len(cy_ids)} campaigns**")
                cf1, cf2 = st.columns(2)
                cy_sel_tick = cf1.selectbox("1. Select Ticker", cy_tickers, index=None, placeholder="Type to search...", key="cy_det_tick")

                cy_sel_id = None
                if cy_sel_tick:
                    cy_subset = cy_detail_df[cy_detail_df['Ticker'] == cy_sel_tick]
                    cy_trade_ids = sorted(cy_subset['Trade_ID'].unique().tolist(), reverse=True)
                    cy_sel_id = cf2.selectbox("2. Select Campaign ID", cy_trade_ids, key="cy_det_id")

                    if cy_sel_id:
                        camp_txs = cy_subset[cy_subset['Trade_ID'] == cy_sel_id].sort_values('Date')

                        # --- 3. RUN LIFO ENGINE ---
                        calc_df = camp_txs.copy().reset_index()
                        buy_attribution = {}
                        inventory = []

                        for idx, row in calc_df.iterrows():
                            if row['Action'] == 'BUY':
                                inventory.append({'idx': idx, 'price': row['Amount'], 'qty': row['Shares']})
                                buy_attribution[idx] = {'pl': 0.0, 'sold_cost': 0.0, 'sold_val': 0.0, 'sold_qty': 0.0}
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
                                    buy_attribution[last['idx']]['sold_qty'] += take
                                    last['qty'] -= take
                                    to_sell -= take
                                    if last['qty'] > 0.0001: inventory.append(last)

                        def get_lifo_pl_cy(idx, action, original_pl):
                            if action == 'SELL': return original_pl
                            if idx in buy_attribution: return buy_attribution[idx]['pl']
                            return 0.0

                        def get_lifo_ret_cy(idx, action):
                            if action == 'BUY' and idx in buy_attribution:
                                data = buy_attribution[idx]
                                if data['sold_cost'] > 0:
                                    return ((data['sold_val'] - data['sold_cost']) / data['sold_cost']) * 100
                            return 0.0

                        def get_exit_price_cy(idx, action):
                            if action == 'BUY' and idx in buy_attribution:
                                data = buy_attribution[idx]
                                qty = data.get('sold_qty', 0)
                                if qty > 0:
                                    return data['sold_val'] / qty
                            return None

                        calc_df['Lot P&L'] = calc_df.apply(lambda x: get_lifo_pl_cy(x.name, x['Action'], x['Realized_PL']), axis=1)
                        calc_df['Return %'] = calc_df.apply(lambda x: get_lifo_ret_cy(x.name, x['Action']), axis=1)
                        calc_df['Exit_Price'] = calc_df.apply(lambda x: get_exit_price_cy(x.name, x['Action']), axis=1)

                        # --- 4. CORE/ADD CLASSIFICATION ---
                        calc_df['Category'] = ''
                        b1_rows = calc_df[(calc_df['Action'] == 'BUY') & (calc_df['Trx_ID'].astype(str).str.upper().str.startswith('B'))]
                        b1_price_cy = 0.0
                        band_low_cy = 0.0
                        band_high_cy = 0.0

                        if not b1_rows.empty:
                            b1_price_cy = float(b1_rows.iloc[0]['Amount'])
                            band_low_cy = b1_price_cy * 0.975
                            band_high_cy = b1_price_cy * 1.025

                            def classify_buy(row):
                                if row['Action'] != 'BUY': return ''
                                if band_low_cy <= row['Amount'] <= band_high_cy: return 'Core'
                                return 'Add'
                            calc_df['Category'] = calc_df.apply(classify_buy, axis=1)

                        # --- 5. FLIGHT DECK ---
                        realized_pl_cy = calc_df[calc_df['Action'] == 'BUY']['Lot P&L'].sum()
                        camp_sum_cy = df_s[df_s['Trade_ID'] == cy_sel_id]
                        is_closed_cy = False
                        if not camp_sum_cy.empty and camp_sum_cy.iloc[0]['Status'] == 'CLOSED':
                            is_closed_cy = True

                        buys_cy = calc_df[calc_df['Action'] == 'BUY']
                        avg_in_cy = (buys_cy['Amount'] * buys_cy['Shares']).sum() / buys_cy['Shares'].sum() if not buys_cy.empty else 0

                        # Core/Add aggregation
                        core_buys = calc_df[(calc_df['Action'] == 'BUY') & (calc_df['Category'] == 'Core')]
                        add_buys = calc_df[(calc_df['Action'] == 'BUY') & (calc_df['Category'] == 'Add')]
                        core_pl_cy = core_buys['Lot P&L'].sum() if not core_buys.empty else 0.0
                        add_pl_cy = add_buys['Lot P&L'].sum() if not add_buys.empty else 0.0
                        core_cost_cy = (core_buys['Amount'] * core_buys['Shares']).sum() if not core_buys.empty else 0.0
                        add_cost_cy = (add_buys['Amount'] * add_buys['Shares']).sum() if not add_buys.empty else 0.0
                        core_ret_cy = (core_pl_cy / core_cost_cy * 100) if core_cost_cy > 0 else 0.0
                        add_ret_cy = (add_pl_cy / add_cost_cy * 100) if add_cost_cy > 0 else 0.0
                        total_cost_cy = core_cost_cy + add_cost_cy
                        total_ret_cy = (realized_pl_cy / total_cost_cy * 100) if total_cost_cy > 0 else 0.0

                        st.markdown(f"### 🚁 Flight Deck: {cy_sel_tick} ({cy_sel_id})")
                        fd1, fd2, fd3, fd4 = st.columns(4)
                        fd1.metric("Total P&L", f"${realized_pl_cy:+,.2f}", f"{total_ret_cy:+.2f}% Return")
                        fd2.metric("Avg Entry", f"${avg_in_cy:.2f}", f"{'CLOSED' if is_closed_cy else 'OPEN'}")

                        core_delta = f"{core_ret_cy:+.2f}% | ${core_cost_cy:,.0f} deployed"
                        fd3.metric("Core P&L", f"${core_pl_cy:+,.2f}", core_delta, delta_color="off")

                        if add_cost_cy > 0:
                            add_delta = f"{add_ret_cy:+.2f}% | ${add_cost_cy:,.0f} deployed"
                            fd4.metric("Add P&L", f"${add_pl_cy:+,.2f}", add_delta, delta_color="off")
                        else:
                            fd4.metric("Add P&L", "$0.00", "No adds")

                        if b1_price_cy > 0:
                            st.info(f"**Core Band:** ${band_low_cy:,.2f} – ${band_high_cy:,.2f} (B1 = ${b1_price_cy:,.2f} ± 2.5%)")

                        st.markdown("---")

                        # --- 6. TRANSACTION TABLE ---
                        st.markdown("#### 📜 Transaction History (LIFO + Core/Add)")
                        display_df_cy = calc_df.copy()
                        display_df_cy['Shares'] = display_df_cy.apply(lambda x: -x['Shares'] if x['Action'] == 'SELL' else x['Shares'], axis=1)
                        display_df_cy['Value'] = display_df_cy.apply(lambda x: -x['Value'] if x['Action'] == 'SELL' else x['Value'], axis=1)

                        cols_cy = ['Trade_ID', 'Trx_ID', 'Date', 'Ticker', 'Action', 'Category', 'Shares', 'Amount', 'Exit_Price', 'Value', 'Lot P&L', 'Return %', 'Rule', 'Notes']
                        show_cols_cy = [c for c in cols_cy if c in display_df_cy.columns]

                        st.dataframe(
                            display_df_cy[show_cols_cy].sort_values(['Trade_ID', 'Date']).style
                            .format({
                                'Date': lambda x: x.strftime('%Y-%m-%d %H:%M') if isinstance(x, (pd.Timestamp, datetime)) else 'None',
                                'Shares':'{:.0f}', 'Amount':'${:,.2f}',
                                'Exit_Price': lambda x: f'${x:,.2f}' if pd.notna(x) else '',
                                'Value':'${:,.2f}',
                                'Lot P&L':'${:,.2f}', 'Return %':'{:.2f}%'
                            })
                            .map(color_pnl, subset=['Lot P&L', 'Return %'])
                            .map(color_neg_value, subset=['Shares']),
                            use_container_width=True
                        )
                else:
                    st.info("Select a ticker to view campaign details.")
            else:
                st.info("No 2026 campaigns found.")
        else:
            st.warning("Summary Database is empty.")

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

                        # Only show types that actually have images, in canonical order
                        _canonical_order = ['entry', 'position', 'weekly', 'daily', 'exit', 'marketsurge']
                        _present = set(img['image_type'] for img in images)
                        _img_types = [t for t in _canonical_order if t in _present]
                        # Append any exotic types not in canonical order
                        _img_types += sorted(_present - set(_canonical_order))
                        cols = st.columns(len(_img_types))
                        _type_labels = {'entry': 'Entry Charts', 'position': 'Position Changes', 'weekly': 'Weekly Chart', 'daily': 'Daily Chart', 'exit': 'Exit Chart', 'marketsurge': 'MarketSurge'}
                        for img_type, col in zip(_img_types, cols):
                            with col:
                                type_imgs = [img for img in images if img['image_type'] == img_type]
                                st.markdown(f"**{_type_labels.get(img_type, img_type.title())}**")
                                for img_data in type_imgs:
                                    image_bytes = cached_r2_download(img_data['image_url'])
                                    if image_bytes:
                                        st.image(image_bytes, use_container_width=True, output_format="PNG")
                                        st.caption(f"{img_data.get('file_name', '')} — {img_data['uploaded_at']}")
                    else:
                        st.info("No charts uploaded for this trade")

                    # --- FUNDAMENTALS VIEWER ---
                    if USE_DATABASE and selected_trade != "Select...":
                        fundas = db.get_trade_fundamentals(CURR_PORT_NAME, selected_trade)
                        if fundas:
                            st.markdown("---")
                            st.markdown("#### 🔬 Extracted Fundamentals")
                            f = fundas[0]  # Most recent extraction
                            fc1, fc2, fc3, fc4 = st.columns(4)
                            fc1.metric("Composite", f.get('composite_rating', 'N/A'))
                            fc2.metric("EPS Rating", f.get('eps_rating', 'N/A'))
                            fc3.metric("RS Rating", f.get('rs_rating', 'N/A'))
                            fc4.metric("Acc/Dis", f.get('acc_dis_rating', 'N/A'))

                            fc5, fc6, fc7, fc8 = st.columns(4)
                            fc5.metric("SMR", f.get('smr_rating', 'N/A'))
                            fc6.metric("Group RS", f.get('group_rs_rating', 'N/A'))
                            fc7.metric("EPS Growth", f"{f['eps_growth_rate']}%" if f.get('eps_growth_rate') else "N/A")
                            fc8.metric("U/D Vol", f.get('ud_vol_ratio', 'N/A'))

                            if f.get('industry_group'):
                                st.caption(f"Industry: {f['industry_group']} (Rank #{f.get('industry_group_rank', '?')})")
                            if f.get('funds_own_pct') is not None:
                                st.caption(f"Ownership — Funds: {f['funds_own_pct']}% | Banks: {f.get('banks_own_pct', '?')}% | Mgmt: {f.get('mgmt_own_pct', '?')}%")
                            st.caption(f"Extracted: {f['extracted_at']}")

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
                }).map(highlight_status, subset=['Status']),
                use_container_width=True
            )
        else:
            st.info("No closed trades found for this period.")




# ====================================================================
# ACTIVE CAMPAIGN SUMMARY
# ====================================================================
elif page == "Active Campaign Summary":
    page_header("Active Campaign Summary", CURR_PORT_NAME, "📋")

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

                        _canonical_order2 = ['entry', 'position', 'weekly', 'daily', 'exit', 'marketsurge']
                        _present2 = set(img['image_type'] for img in images)
                        _img_types2 = [t for t in _canonical_order2 if t in _present2]
                        _img_types2 += sorted(_present2 - set(_canonical_order2))
                        _labels2 = {'entry': 'Entry Charts', 'position': 'Position Changes', 'weekly': 'Weekly Charts', 'daily': 'Daily Charts', 'exit': 'Exit Chart', 'marketsurge': 'MarketSurge'}
                        cols = st.columns(len(_img_types2))

                        for img_type, col in zip(_img_types2, cols):
                            with col:
                                type_imgs = [img for img in images if img['image_type'] == img_type]
                                st.markdown(f"**{_labels2.get(img_type, img_type.title())}**")
                                for img_data in type_imgs:
                                    image_bytes = cached_r2_download(img_data['image_url'])
                                    if image_bytes:
                                        st.image(image_bytes, use_container_width=True, output_format="PNG")
                                        st.caption(f"{img_data.get('file_name', '')} — {img_data['uploaded_at']}")

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



                 master_stop = avg_log_stop if avg_log_stop > 0 else (summary_stop if summary_stop > 0 else avg_cost)

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

             df_open['Overall_PL'] = df_open['Unrealized_PL'] + df_open['Realized Bank']

             

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

             # --- PYRAMID READY INDICATOR ---
             def get_pyramid_flag(row):
                 tid = str(row['Trade_ID']).strip()
                 curr_price = row['Current Price']
                 if curr_price <= 0:
                     return float('nan')
                 df_d['Trade_ID_Str'] = df_d['Trade_ID'].astype(str).str.strip()
                 subset = df_d[df_d['Trade_ID_Str'] == tid].copy()
                 if subset.empty:
                     return float('nan')
                 # Build LIFO inventory to find last remaining lot
                 def force_float(x):
                     try: return float(str(x).replace('$','').replace(',',''))
                     except: return 0.0
                 subset['Type_Rank'] = subset['Action'].apply(lambda x: 0 if str(x).upper() == 'BUY' else 1)
                 if 'Date' in subset.columns:
                     subset['Date'] = pd.to_datetime(subset['Date'], errors='coerce')
                     subset = subset.sort_values(['Date', 'Type_Rank'])
                 inventory = []
                 for _, tx in subset.iterrows():
                     action = str(tx.get('Action', '')).upper()
                     tx_shares = abs(force_float(tx.get('Shares', 0)))
                     if action == 'BUY':
                         price = force_float(tx.get('Amount', tx.get('Price', 0.0)))
                         if price == 0: price = force_float(row.get('Avg_Entry', 0))
                         inventory.append({'qty': tx_shares, 'price': price})
                     elif action == 'SELL':
                         to_sell = tx_shares
                         while to_sell > 0 and inventory:
                             last = inventory[-1]
                             take = min(to_sell, last['qty'])
                             last['qty'] -= take
                             to_sell -= take
                             if last['qty'] < 0.00001: inventory.pop()
                 if not inventory:
                     return float('nan')
                 last_lot_price = inventory[-1]['price']
                 if last_lot_price <= 0:
                     return float('nan')
                 last_lot_return = ((curr_price - last_lot_price) / last_lot_price) * 100
                 return last_lot_return

             df_open['Pyramid'] = df_open.apply(get_pyramid_flag, axis=1)



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

             total_overall = df_open['Overall_PL'].sum()

             total_initial_risk = df_open['Risk $'].sum() 

             total_open_risk_equity = df_open['Open Risk Equity'].sum() 

             

             live_exp = (total_mkt / equity) * 100

             

             m1, m2, m3, m4, m5, m6 = st.columns(6)

             # Privacy-aware formatting for the Flight Deck summary row.
             # Dollar numbers are masked; percentages and counts stay visible.
             def _mk(val):
                 return "$****" if PRIVACY else f"${val:,.2f}"

             m1.metric("Open Positions", len(df_open))

             m2.metric("Total Market Value", _mk(total_mkt))

             m3.metric("Live Exposure", f"{live_exp:.1f}%", f"of {_mk(equity)}")



             total_projected = df_open['Projected P&L'].sum()
             m4.metric(
                 "Overall P&L",
                 _mk(total_overall),
                 f"Projected: {_mk(total_projected)}",
                 delta_color="normal"
             )



             ir_pct = (total_initial_risk / equity) * 100

             m5.metric("Initial Risk", _mk(total_initial_risk), f"{ir_pct:.2f}% of NLV", delta_color="off")



             or_pct = (total_open_risk_equity / equity) * 100

             m6.metric("Open Risk (Heat)", _mk(total_open_risk_equity), f"{or_pct:.2f}% of NLV", delta_color="inverse")

             

             # --- 5. DATAFRAME ---

             if 'Return_Pct' in df_open.columns:

                 df_open = df_open.sort_values(by='Return_Pct', ascending=False)

             

             # UPDATED COLUMNS: Removed Trend Status, Added Risk Status

             cols = ['Trade_ID', 'Ticker', 'Days Held', 'Risk Status', 'Pyramid', 'Return_Pct', 'Pos Size %',

                     'Shares', 'Avg_Entry', 'Current Price', 'Avg Stop', 'Risk_Budget',

                     'Risk $', 'Risk %', 'Current Value', 'Overall_PL', 'Projected P&L']

             

             final_cols = [c for c in cols if c in df_open.columns]

             

             def color_pyramid(val):
                 try:
                     v = float(val)
                 except (ValueError, TypeError):
                     return ''
                 if v >= 5.0:
                     return 'background-color: #2ca02c; color: white; font-weight: bold;'
                 elif v > 0:
                     return 'background-color: #ffcc00; color: black;'
                 else:
                     return 'background-color: #ff4b4b; color: white;'

             def format_pyramid(val):
                 try:
                     v = float(val)
                 except (ValueError, TypeError):
                     return ''
                 if v >= 5.0:
                     return f'🔺 +{v:.1f}%'
                 elif v > 0:
                     return f'△ +{v:.1f}%'
                 else:
                     return f'▽ {v:.1f}%'

             pyramid_cols = [c for c in ['Pyramid'] if c in df_open.columns]

             st.dataframe(

                 df_open[final_cols].style.format({

                     'Shares':'{:.0f}', 'Total_Cost':'${:,.2f}', 'Overall_PL':'${:,.2f}', 'Avg_Entry':'${:,.2f}',

                     'Current Price':'${:,.2f}', 'Return_Pct':'{:.2f}%', 'Current Value': '${:,.2f}', 'Pos Size %': '{:.1f}%',

                     'Avg Stop': '${:,.2f}', 'Risk $': '${:,.2f}', 'Risk %': '{:.2f}%', 'Risk_Budget': '${:,.2f}', 'Days Held': '{:.0f}',

                     'Projected P&L': '${:,.2f}', 'Pyramid': format_pyramid

                 }).map(color_pnl, subset=['Overall_PL', 'Return_Pct', 'Projected P&L']
                 ).map(color_pyramid, subset=pyramid_cols),

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
                     total_cost = float(r.get('Total_Cost', 0))
                     total_shares = float(r.get('Shares', 0))
                     rbm_stop = (total_cost - budget) / total_shares if total_shares > 0 else 0.0
                     st.warning(f"⚠️ **{r['Ticker']}**: Initial Risk (${r['Risk $']:.0f}) > Budget (${budget:.0f}). Raise stop to **${rbm_stop:.2f}** to stay within budget.")

                     all_clear = False

                 

                 if r['Return_Pct'] <= -7.0:

                     st.error(f"🔴 **{r['Ticker']}**: Down {r['Return_Pct']:.2f}%. Violates Stop Rule.")

                     all_clear = False

                 if r['Return_Pct'] >= 10.0:
                     avg_entry = float(r.get('Avg_Entry', 0))
                     avg_stop = float(r.get('Avg Stop', 0))
                     if avg_entry > 0 and avg_stop > 0 and avg_stop < (avg_entry - 0.01):
                         st.info(f"📈 **{r['Ticker']}**: Up {r['Return_Pct']:.2f}%. Consider moving stop to BE (${avg_entry:.2f}). Current stop: ${avg_stop:.2f}.")
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

    page_header("Risk Manager", CURR_PORT_NAME, "🛡️")

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

                        # B. Get Live Prices (cached 60s)
                        tickers = df_open['Ticker'].unique().tolist()
                        live_prices = cached_batch_live_prices(tuple(sorted(tickers))) if tickers else {}



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



                            master_stop = avg_log_stop if avg_log_stop > 0 else (summary_stop if summary_stop > 0 else summary_entry)

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
    page_header("Portfolio Heat", f"Volatility Check · {CURR_PORT_NAME}", "🔥")

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
    page_header("Earnings Planner", "Binary Event Logic · Principal Protection", "💣")

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



    # Load equity from journal (database-aware) — same pattern as Position Sizer
    ep_equity = 100000.0
    try:
        j_df = load_data(JOURNAL_FILE)
        if not j_df.empty and 'End NLV' in j_df.columns:
            if 'Day' in j_df.columns:
                j_df['Day'] = pd.to_datetime(j_df['Day'], errors='coerce')
                j_df = j_df.dropna(subset=['Day']).sort_values('Day', ascending=False)
            val_str = str(j_df['End NLV'].iloc[0]).replace('$','').replace(',','')
            ep_equity = float(val_str)
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

        

        # Price Default Logic: yfinance (cached) -> Session Cache -> Row Current Price -> 0.0
        def_price = 0.0
        live_fetch = cached_live_price(sel_ticker)
        if live_fetch and live_fetch > 0:
            def_price = float(live_fetch)
        if def_price == 0 and 'live_prices' in st.session_state and sel_ticker in st.session_state['live_prices']:
            def_price = st.session_state['live_prices'][sel_ticker]
        if def_price == 0 and 'Current_Price' in row and float(row['Current_Price']) > 0:
            def_price = float(row['Current_Price'])

        

        # INPUTS SECTION

        st.markdown("---")

        st.markdown("#### 1. Setup & Cushion Check")

        

        c1, c2, c3, c4 = st.columns(4)

        curr_price = c1.number_input("Current Price ($)", value=float(def_price), step=0.10, format="%.2f")

        nlv_val = c2.number_input("Account Equity (NLV)", value=float(ep_equity), step=1000.0)

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
    page_header("Performance Audit", "Outlier trades · R efficiency · P&L concentration", "🏆")

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

                    .map(lambda x: 'color: #4CAF50' if x > 0 else 'color: #FF5252', subset=['Net P&L', 'R-Multiple', 'Return %']),

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

                    .map(lambda x: 'color: #4CAF50' if x > 0 else 'color: #FF5252', subset=['Net P&L', 'R-Multiple', 'Return %']),

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
    page_header("Trade Journal", "", "📔")
    st.caption("Visual review of all your trades with embedded charts")

    # Quick action buttons
    _ac1, _ac2, _ac3, _ac4 = st.columns([1, 1, 1, 3])
    with _ac1:
        if st.button("🟢 Log Buy", key="tj_log_buy", use_container_width=True):
            st.session_state.page = "Log Buy"
            if '_tj_prev_page' in st.session_state: del st.session_state['_tj_prev_page']
            st.rerun()
    with _ac2:
        if st.button("🔴 Log Sell", key="tj_log_sell", use_container_width=True):
            st.session_state.page = "Log Sell"
            if '_tj_prev_page' in st.session_state: del st.session_state['_tj_prev_page']
            st.rerun()
    with _ac3:
        if st.button("📝 Edit Trades", key="tj_edit_trades", use_container_width=True):
            st.session_state.page = "Trade Manager"
            if '_tj_prev_page' in st.session_state: del st.session_state['_tj_prev_page']
            st.rerun()

    # Track page entry to clear stale search results
    # Skip clearing if navigating here with a pre-set ticker (e.g., from Daily Journal)
    if st.session_state.get('_tj_prev_page') != 'Trade Journal':
        if 'tj_ticker_search' not in st.session_state or not st.session_state['tj_ticker_search']:
            if 'journal_searched' in st.session_state:
                del st.session_state['journal_searched']
    st.session_state['_tj_prev_page'] = 'Trade Journal'

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
            all_tickers = sorted(df_s['Ticker'].unique().tolist())
            ticker_filter = st.multiselect(
                "Ticker(s)",
                options=all_tickers,
                default=[],
                placeholder="Search tickers...",
                key="tj_ticker_search"
            )
            if not ticker_filter:
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
        if search_clicked:
            st.session_state['journal_searched'] = True

        if 'journal_searched' not in st.session_state:
            st.info("👆 Select your filters and click **Search Trades** to view your journal")
            st.stop()

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

                # === Calculate LIFO-based P&L (both open and closed) ===
                target_df = df_d[df_d['Trade_ID'] == trade_id].copy()
                buy_realized_pl = {}
                remaining_map = {}
                live_px = avg_entry_val

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
                    buy_shares_sold = {}
                    buy_exit_price = {}
                    lifo_pl_map = {}

                    for tidx, row in target_df.iterrows():
                        if row['Action'] == 'BUY':
                            p = float(row.get('Amount', row.get('Price', 0.0)))
                            inventory.append({'idx': tidx, 'qty': row['Shares'], 'price': p})
                            remaining_map[tidx] = row['Shares']

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

                                # Back-attribute P&L to the BUY lot
                                lot_pl = take * (sell_price - last['price'])
                                buy_realized_pl[last['idx']] = buy_realized_pl.get(last['idx'], 0) + lot_pl
                                prev_sold = buy_shares_sold.get(last['idx'], 0)
                                buy_shares_sold[last['idx']] = prev_sold + take
                                total_sold = buy_shares_sold[last['idx']]
                                prev_avg = buy_exit_price.get(last['idx'], 0)
                                buy_exit_price[last['idx']] = (prev_avg * prev_sold + sell_price * take) / total_sold

                                last['qty'] -= take
                                to_sell -= take
                                remaining_map[last['idx']] = last['qty']
                                if last['qty'] < 0.00001:
                                    inventory.pop()

                            revenue = sold_qty_accum * sell_price
                            true_pl = revenue - cost_basis_accum
                            lifo_pl_map[tidx] = true_pl
                            fd_realized_pl += true_pl

                    for item in inventory:
                        fd_remaining_shares += item['qty']
                        fd_cost_basis_sum += (item['qty'] * item['price'])

                    if is_open:
                        # Get live price for open trades (cached 60s)
                        live_px = cached_live_price(ticker) or avg_entry_val

                        mkt_val = fd_remaining_shares * live_px
                        unrealized_pl = mkt_val - fd_cost_basis_sum
                        pl_val = fd_realized_pl + unrealized_pl

                        total_cost_basis = 0.0
                        for tidx, row in target_df.iterrows():
                            if row['Action'] == 'BUY':
                                p = float(row.get('Amount', row.get('Price', 0.0)))
                                total_cost_basis += (row['Shares'] * p)

                        return_pct = (pl_val / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
                        pl_label = "Total P&L"
                    else:
                        # Closed trade
                        pl_val = fd_realized_pl
                        total_cost_basis = 0.0
                        for tidx, row in target_df.iterrows():
                            if row['Action'] == 'BUY':
                                p = float(row.get('Amount', row.get('Price', 0.0)))
                                total_cost_basis += (row['Shares'] * p)
                        return_pct = (pl_val / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
                        pl_label = "Realized P&L"
                else:
                    pl_val = 0.0
                    return_pct = 0.0
                    pl_label = "P&L"

                # === Core/Add P&L Classification ===
                core_pl = 0.0
                add_pl = 0.0
                core_cost = 0.0
                add_cost = 0.0
                b1_price = 0.0
                band_low = 0.0
                band_high = 0.0
                has_core_add = False

                if not target_df.empty and 'Trx_ID' in target_df.columns:
                    buys = target_df[target_df['Action'] == 'BUY']
                    b1_rows = buys[buys['Trx_ID'].astype(str).str.upper().str.startswith('B')]
                    if not b1_rows.empty:
                        b1_price = float(b1_rows.iloc[0].get('Amount', b1_rows.iloc[0].get('Price', 0.0)))
                        band_low = b1_price * 0.975
                        band_high = b1_price * 1.025
                        has_core_add = True

                        for buy_idx, buy_row in buys.iterrows():
                            buy_price = float(buy_row.get('Amount', buy_row.get('Price', 0.0)))
                            lot_cost = buy_price * buy_row['Shares']
                            lot_pl = buy_realized_pl.get(buy_idx, 0.0)
                            # For open buys, include unrealized P&L
                            if is_open and remaining_map.get(buy_idx, 0) > 0:
                                lot_pl += (live_px - buy_price) * remaining_map[buy_idx]
                            if band_low <= buy_price <= band_high:
                                core_pl += lot_pl
                                core_cost += lot_cost
                            else:
                                add_pl += lot_pl
                                add_cost += lot_cost

                core_return_pct = (core_pl / core_cost * 100) if core_cost > 0 else 0.0
                add_return_pct = (add_pl / add_cost * 100) if add_cost > 0 else 0.0

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

                # === SCALE-OUT PLAN (open trades only) ===
                scale_out_html = ""
                try:
                    _so_shs = int(float(trade.get('Shares', 0) or 0))
                    _so_entry = float(avg_entry_val or 0)
                except (ValueError, TypeError):
                    _so_shs, _so_entry = 0, 0.0
                if is_open and _so_shs > 0 and _so_entry > 0:
                    _t1 = round(_so_shs * 0.25)
                    _t2 = round(_so_shs * 0.25)
                    _t3 = _so_shs - _t1 - _t2

                    # Worst-case scale-out loss (all 3 targets fire)
                    _scale_loss_dol = (
                        _t1 * _so_entry * 0.03 +
                        _t2 * _so_entry * 0.05 +
                        _t3 * _so_entry * 0.07
                    )
                    _pos_cost = _so_shs * _so_entry
                    _scale_loss_pct = (_scale_loss_dol / _pos_cost * 100) if _pos_cost else 0

                    # Hard-stop loss (single stop on full position)
                    _stop_px = 0.0
                    try:
                        if not target_df.empty and 'Stop_Loss' in target_df.columns:
                            _stops = pd.to_numeric(target_df['Stop_Loss'], errors='coerce').dropna()
                            _stops = _stops[_stops > 0]
                            if not _stops.empty:
                                _stop_px = float(_stops.iloc[-1])
                    except Exception:
                        _stop_px = 0.0

                    # Hard stop loss
                    _stop_loss_dol = None
                    if _stop_px > 0 and _stop_px < _so_entry:
                        _stop_loss_dol = (_so_entry - _stop_px) * _so_shs

                    # Risk budget on trade
                    _risk_budget = 0.0
                    try:
                        _risk_budget = float(trade.get('Risk_Budget', 0) or 0)
                    except (ValueError, TypeError):
                        _risk_budget = 0.0

                    # Build comparison rows
                    _rows = [f'<div><strong>Scale-Out worst case:</strong> -${_scale_loss_dol:,.0f} ({_scale_loss_pct:.2f}%)</div>']
                    if _stop_loss_dol is not None:
                        _stop_loss_pct = (_stop_loss_dol / _pos_cost * 100)
                        _rows.append(f'<div><strong>Hard Stop @ ${_stop_px:,.2f}:</strong> -${_stop_loss_dol:,.0f} ({_stop_loss_pct:.2f}%)</div>')

                    # Verdict: scale-out vs risk budget
                    if _risk_budget > 0:
                        _diff = _risk_budget - _scale_loss_dol
                        if _diff >= 0:
                            _verdict_color = "#16a34a"
                            _verdict = f"✅ Scale-out within risk budget (${_risk_budget:,.0f}) — buffer ${_diff:,.0f}"
                        else:
                            _verdict_color = "#dc2626"
                            _verdict = f"⚠️ Scale-out EXCEEDS risk budget (${_risk_budget:,.0f}) by ${-_diff:,.0f}"
                    else:
                        _verdict_color = "#6b7280"
                        _verdict = "No risk budget on file"

                    _compare_html = (
                        '<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #fde68a; '
                        'font-size: 12px; color: #374151;">'
                        f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">'
                        + ''.join(_rows) +
                        '</div>'
                        f'<div style="margin-top: 4px; font-weight: 600; color: {_verdict_color};">{_verdict}</div>'
                        '</div>'
                    )

                    scale_out_html = (
                        '<div style="background: #fff8e1; border-left: 3px solid #f59e0b; '
                        'padding: 10px 12px; margin-bottom: 12px; border-radius: 4px;">'
                        '<div style="font-size: 11px; color: #92400e; text-transform: uppercase; '
                        'font-weight: 600; margin-bottom: 6px;">Scale-Out Plan</div>'
                        '<div style="display: grid; grid-template-columns: repeat(3, 1fr); '
                        'gap: 8px; font-size: 13px; color: #333;">'
                        f'<div><strong>T1 (-3%):</strong> {_t1} shs @ ${_so_entry * 0.97:,.2f}</div>'
                        f'<div><strong>T2 (-5%):</strong> {_t2} shs @ ${_so_entry * 0.95:,.2f}</div>'
                        f'<div><strong>T3 (-7%):</strong> {_t3} shs @ ${_so_entry * 0.93:,.2f}</div>'
                        '</div>'
                        f'{_compare_html}'
                        '</div>'
                    )

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
                    {('<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 15px;">' +
                    '<div>' +
                    '<div style="font-size: 11px; color: #666; text-transform: uppercase;">Core P&L</div>' +
                    f'<div style="font-size: 16px; font-weight: 600; color: {"#28a745" if core_pl >= 0 else "#dc3545"};">' +
                    f'{"+" if core_pl >= 0 else ""}${core_pl:,.2f}</div>' +
                    f'<div style="font-size: 12px; color: {"#28a745" if core_return_pct >= 0 else "#dc3545"};">' +
                    f'{"+" if core_return_pct >= 0 else ""}{core_return_pct:.2f}%</div>' +
                    '</div>' +
                    '<div>' +
                    '<div style="font-size: 11px; color: #666; text-transform: uppercase;">Add P&L</div>' +
                    f'<div style="font-size: 16px; font-weight: 600; color: {"#28a745" if add_pl >= 0 else "#dc3545"};">' +
                    f'{"+" if add_pl >= 0 else ""}${add_pl:,.2f}</div>' +
                    (f'<div style="font-size: 12px; color: {"#28a745" if add_return_pct >= 0 else "#dc3545"};">' +
                    f'{"+" if add_return_pct >= 0 else ""}{add_return_pct:.2f}%</div>' if add_cost > 0 else '') +
                    '</div>' +
                    '<div>' +
                    '<div style="font-size: 11px; color: #666; text-transform: uppercase;">Core Band</div>' +
                    f'<div style="font-size: 14px;">${band_low:,.2f} – ${band_high:,.2f}</div>' +
                    '</div>' +
                    '<div>' +
                    '<div style="font-size: 11px; color: #666; text-transform: uppercase;">B1 Price</div>' +
                    f'<div style="font-size: 16px; font-weight: 600;">${b1_price:,.2f}</div>' +
                    '</div>' +
                    '</div>') if has_core_add else '<!-- -->'}
                    {scale_out_html or '<!-- -->'}
                    <div style="font-size: 12px; color: #666;">
                        <strong>Trade ID:</strong> {trade_id} |
                        <strong>Opened:</strong> {open_date if pd.notna(open_date) else 'N/A'}{f" | <strong>Closed:</strong> {trade.get('Closed_Date')}" if not is_open and pd.notna(trade.get('Closed_Date')) else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # === CHARTS (In Expander) ===
                with st.expander(f"📊 Charts - {ticker}", expanded=False):
                    if R2_AVAILABLE and USE_DATABASE:
                        images = db.get_trade_images(CURR_PORT_NAME, trade_id)

                        if images:
                            # Only show types that actually have images, in canonical order
                            _canonical_order = ['entry', 'position', 'weekly', 'daily', 'exit', 'marketsurge']
                            _present = set(img['image_type'] for img in images)
                            chart_types = [t for t in _canonical_order if t in _present]
                            chart_types += sorted(_present - set(_canonical_order))
                            cols = st.columns(len(chart_types))

                            _type_labels = {'entry': 'Entry Charts', 'position': 'Position Changes', 'weekly': 'Weekly Chart', 'daily': 'Daily Chart', 'exit': 'Exit Chart', 'marketsurge': 'MarketSurge'}
                            icons = {'entry': '📊', 'position': '🔄', 'weekly': '📊', 'daily': '📈', 'exit': '🎯', 'marketsurge': '🔬'}
                            for img_type, col in zip(chart_types, cols):
                                with col:
                                    type_imgs = [img for img in images if img['image_type'] == img_type]
                                    _label = _type_labels.get(img_type, img_type.title())
                                    st.markdown(f"**{icons.get(img_type, '')} {_label}**")
                                    for img_data in type_imgs:
                                        image_bytes = cached_r2_download(img_data['image_url'])
                                        if image_bytes:
                                            st.image(image_bytes, use_container_width=True, output_format="PNG")
                                            cap_col, del_col = st.columns([4, 1])
                                            cap_col.caption(f"{img_data.get('file_name', '')} — {img_data['uploaded_at']}")
                                            if del_col.button("🗑️", key=f"del_img_{img_data['id']}"):
                                                url = db.delete_trade_image_by_id(img_data['id'])
                                                if url:
                                                    try: r2.delete_image(url)
                                                    except: pass
                                                st.success("Image deleted")
                                                st.rerun()
                        else:
                            st.info("No charts available for this trade")

                        # === FUNDAMENTALS (if extracted) ===
                        fundas = db.get_trade_fundamentals(CURR_PORT_NAME, trade_id)
                        if fundas:
                            st.markdown("---")
                            st.markdown("**🔬 MarketSurge Fundamentals**")
                            f = fundas[0]
                            fc1, fc2, fc3, fc4 = st.columns(4)
                            fc1.metric("Composite", f.get('composite_rating', 'N/A'))
                            fc2.metric("EPS Rating", f.get('eps_rating', 'N/A'))
                            fc3.metric("RS Rating", f.get('rs_rating', 'N/A'))
                            fc4.metric("Acc/Dis", f.get('acc_dis_rating', 'N/A'))

                            fc5, fc6, fc7, fc8 = st.columns(4)
                            fc5.metric("SMR", f.get('smr_rating', 'N/A'))
                            fc6.metric("Group RS", f.get('group_rs_rating', 'N/A'))
                            fc7.metric("EPS Growth", f"{f['eps_growth_rate']}%" if f.get('eps_growth_rate') else "N/A")
                            fc8.metric("U/D Vol", f.get('ud_vol_ratio', 'N/A'))

                            if f.get('industry_group'):
                                st.caption(f"Industry: {f['industry_group']} (Rank #{f.get('industry_group_rank', '?')})")
                            if f.get('funds_own_pct') is not None:
                                st.caption(f"Ownership — Funds: {f['funds_own_pct']}% | Banks: {f.get('banks_own_pct', '?')}% | Mgmt: {f.get('mgmt_own_pct', '?')}%")
                            st.caption(f"Extracted: {f['extracted_at']}")

                        # === UPLOAD/UPDATE CHARTS ===
                        st.markdown("---")
                        st.markdown("### 📤 Upload Charts")

                        upload_col1, upload_col2 = st.columns(2)
                        with upload_col1:
                            tj_entry_uploads = st.file_uploader(
                                "📊 Entry Charts (Weekly / Daily)",
                                type=['png', 'jpg', 'jpeg'],
                                key=f'entry_upload_{trade_id}',
                                accept_multiple_files=True
                            )
                        with upload_col2:
                            tj_position_uploads = st.file_uploader(
                                "🔄 Position Changes (Add-ons / Trims / Exits)",
                                type=['png', 'jpg', 'jpeg'],
                                key=f'position_upload_{trade_id}',
                                accept_multiple_files=True
                            )

                        # MarketSurge screenshot extraction
                        ms_upload = None
                        if check_vision_available():
                            st.markdown("---")
                            st.markdown("**🔬 MarketSurge Fundamentals**")
                            ms_upload = st.file_uploader(
                                "MarketSurge Screenshot",
                                type=['png', 'jpg', 'jpeg'],
                                key=f'ms_upload_{trade_id}',
                                help="Upload a MarketSurge screenshot to auto-extract ratings"
                            )

                        # Upload button
                        if st.button("💾 Save Charts", key=f'save_charts_{trade_id}', type="primary"):
                            all_uploads = tj_entry_uploads + tj_position_uploads
                            has_ms = ms_upload is not None
                            if all_uploads or has_ms:
                                try:
                                    upload_count = 0

                                    for f in tj_entry_uploads:
                                        url = r2.upload_image(f, CURR_PORT_NAME, trade_id, ticker, 'entry')
                                        if url:
                                            db.save_trade_image(CURR_PORT_NAME, trade_id, ticker, 'entry', url, f.name)
                                            upload_count += 1

                                    for f in tj_position_uploads:
                                        url = r2.upload_image(f, CURR_PORT_NAME, trade_id, ticker, 'position')
                                        if url:
                                            db.save_trade_image(CURR_PORT_NAME, trade_id, ticker, 'position', url, f.name)
                                            upload_count += 1

                                    # Extract MarketSurge fundamentals
                                    if ms_upload and check_vision_available():
                                        with st.spinner("🔬 Extracting fundamentals..."):
                                            ms_image_id = None
                                            if R2_AVAILABLE:
                                                ms_upload.seek(0)
                                                ms_url = r2.upload_image(ms_upload, CURR_PORT_NAME, trade_id, ticker, 'marketsurge')
                                                if ms_url:
                                                    ms_image_id = db.save_trade_image(CURR_PORT_NAME, trade_id, ticker, 'marketsurge', ms_url, ms_upload.name)

                                            ms_upload.seek(0)
                                            img_bytes = ms_upload.read()
                                            extracted = vision_extract.extract_fundamentals(img_bytes, ms_upload.name)
                                            if extracted:
                                                db.save_trade_fundamentals(CURR_PORT_NAME, trade_id, ticker, extracted, ms_image_id)
                                                st.success("🔬 Fundamentals extracted and saved!")
                                            else:
                                                st.warning("⚠️ Could not extract data from screenshot")

                                    if upload_count > 0:
                                        st.success(f"✅ Successfully uploaded {upload_count} chart(s)! Refresh to see changes.")
                                except Exception as e:
                                    st.error(f"❌ Error uploading: {e}")
                            else:
                                st.warning("⚠️ Please select at least one file to upload")

                    else:
                        st.info("Chart display requires R2 storage and database connection")

                # === TRANSACTION DETAILS & NOTES (Using Active Campaign Detailed Logic) ===
                with st.expander(f"📊 Transaction Details & Notes - {ticker}"):
                    if not target_df.empty:
                        # LIFO already ran before card — reuse results
                        display_df = target_df.copy()
                        display_df['Remaining_Shares'] = display_df.index.map(remaining_map).fillna(0)
                        combined_pl = {**buy_realized_pl, **lifo_pl_map}
                        display_df['Realized_PL'] = display_df.index.map(combined_pl).fillna(0)
                        display_df['Exit_Price'] = display_df.index.map(buy_exit_price)
                        display_df['Status'] = display_df['Remaining_Shares'].apply(lambda x: 'Open' if x > 0 else 'Closed')

                        # === FLIGHT DECK (cached 60s) ===
                        live_px = cached_live_price(ticker) or avg_entry_val

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
                            if row['Action'] == 'BUY':
                                entry = float(row.get('Amount', row.get('Price', 0.0)))
                                if row['Remaining_Shares'] > 0:
                                    if entry > 0:
                                        return ((live_px - entry) / entry) * 100
                                elif row.name in buy_realized_pl:
                                    original_shares = abs(float(row['Shares']))
                                    cost_basis = entry * original_shares
                                    if cost_basis > 0: return (buy_realized_pl[row.name] / cost_basis) * 100
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
                                    'Shares', 'Remaining_Shares', 'Amount', 'Exit_Price', 'Stop_Loss', 'Value',
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
                                'Amount':'${:,.2f}', 'Exit_Price':'${:,.2f}', 'Stop_Loss':'${:,.2f}', 'Value':'${:,.2f}',
                                'Realized_PL':'${:,.2f}', 'Unrealized_PL':'${:,.2f}',
                                'Return_Pct':'{:.2f}%', 'Remaining_Shares':'{:.0f}'
                            })
                            .map(color_pnl, subset=['Value','Realized_PL','Unrealized_PL', 'Return_Pct'])
                            .map(color_neg_value, subset=['Shares']),
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
    page_header("Analytics & Audit", CURR_PORT_NAME, "📈")
    
    # 1. LOAD DATA
    if os.path.exists(SUMMARY_FILE):
        df_s_raw = load_data(SUMMARY_FILE) # Load raw data first
        
        # Load journal via load_data() so DB mode works on Streamlit Cloud
        # (local CSV check would fail in cloud where data lives only in the DB).
        df_j = load_data(JOURNAL_FILE)
        if df_j is None:
            df_j = pd.DataFrame()

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
        # Shared across tabs: all closed trades (all-time) from df_s_raw
        all_closed = df_s_raw[df_s_raw['Status'] == 'CLOSED'].copy()
        if not all_closed.empty and 'Closed_Date' in all_closed.columns:
            all_closed['Closed_Date'] = pd.to_datetime(all_closed['Closed_Date'], errors='coerce')

        tab_stats, tab_buy_rules, tab_sell_rules, tab_dd, tab_review = st.tabs(["🎯 Overview", "🟢 Buy Rules", "🔴 Sell Rules", "🛡️ Drawdown Discipline", "🔬 Trade Review"])

        # --- TAB 2: BUY RULES (Rule Studio — 2026 only) ---
        with tab_buy_rules:
            st.subheader("🟢 Buy Rules — What's Working in 2026")
            st.caption("Study your entry rules. Sort by the metric you care about, click any rule to drill into individual trades.")

            br_source = df_s_raw.copy()
            br_source['Closed_Date'] = pd.to_datetime(br_source['Closed_Date'], errors='coerce')
            br_closed_2026 = br_source[
                (br_source['Status'] == 'CLOSED') &
                (br_source['Closed_Date'].dt.year == 2026)
            ].copy()

            # Determine the buy rule column — prefer 'Buy_Rule' if populated, else 'Rule'
            rule_col = None
            if 'Buy_Rule' in br_closed_2026.columns and br_closed_2026['Buy_Rule'].astype(str).str.strip().replace('', pd.NA).notna().any():
                rule_col = 'Buy_Rule'
            elif 'Rule' in br_closed_2026.columns:
                rule_col = 'Rule'

            if rule_col is None or br_closed_2026.empty:
                st.info("💡 No 2026 closed trades with buy rule data yet.")
            else:
                br_closed_2026['_Rule'] = br_closed_2026[rule_col].astype(str).str.strip()
                br_closed_2026 = br_closed_2026[
                    (br_closed_2026['_Rule'] != '') &
                    (br_closed_2026['_Rule'].str.lower() != 'nan')
                ]

                if br_closed_2026.empty:
                    st.info("💡 No 2026 closed trades with buy rule data yet.")
                else:
                    # Load saved buy rule notes for this portfolio
                    br_notes_map = {}
                    if USE_DATABASE and hasattr(db, 'get_rule_notes'):
                        try:
                            br_notes_map = db.get_rule_notes(CURR_PORT_NAME, 'buy')
                        except Exception:
                            br_notes_map = {}

                    # Compute R-multiple where possible
                    br_closed_2026['_R'] = br_closed_2026.apply(
                        lambda r: (float(r['Realized_PL']) / float(r['Risk_Budget']))
                        if r.get('Risk_Budget') and float(r['Risk_Budget']) > 0 else None,
                        axis=1
                    )

                    # Aggregate per rule
                    rule_stats = []
                    for rule_name, group in br_closed_2026.groupby('_Rule'):
                        trades_count = len(group)
                        total_pl = float(group['Realized_PL'].sum())
                        avg_pl = float(group['Realized_PL'].mean())
                        wins = group[group['Realized_PL'] > 0]
                        losses = group[group['Realized_PL'] < 0]
                        win_rate = (len(wins) / trades_count * 100) if trades_count > 0 else 0.0
                        r_vals = group['_R'].dropna()
                        avg_r = float(r_vals.mean()) if not r_vals.empty else None
                        rule_stats.append({
                            'Rule': rule_name,
                            'Trades': trades_count,
                            'Win Rate %': win_rate,
                            'Avg P&L': avg_pl,
                            'Total P&L': total_pl,
                            'Avg R': avg_r,
                        })

                    rule_df = pd.DataFrame(rule_stats)

                    # --- HEADER INSIGHTS ---
                    profitable = rule_df[rule_df['Total P&L'] > 0].sort_values('Total P&L', ascending=False)
                    unprofitable = rule_df[rule_df['Total P&L'] < 0].sort_values('Total P&L', ascending=True)

                    ih1, ih2, ih3 = st.columns(3)
                    with ih1:
                        if not profitable.empty:
                            best = profitable.iloc[0]
                            st.markdown(
                                f'<div style="background:#dcfce7;border-radius:12px;padding:16px 18px;">'
                                f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#15803d;">💰 Best Rule</div>'
                                f'<div style="font-size:15px;font-weight:700;color:#111;margin-top:6px;">{best["Rule"]}</div>'
                                f'<div style="font-size:20px;font-weight:800;color:#15803d;">${best["Total P&L"]:,.0f}</div>'
                                f'<div style="font-size:11px;color:#64748b;">{int(best["Trades"])} trades · {best["Win Rate %"]:.0f}% win rate</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.info("No profitable rules yet")
                    with ih2:
                        if not unprofitable.empty:
                            worst = unprofitable.iloc[0]
                            st.markdown(
                                f'<div style="background:#fee2e2;border-radius:12px;padding:16px 18px;">'
                                f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#b91c1c;">🚨 Worst Rule</div>'
                                f'<div style="font-size:15px;font-weight:700;color:#111;margin-top:6px;">{worst["Rule"]}</div>'
                                f'<div style="font-size:20px;font-weight:800;color:#b91c1c;">${worst["Total P&L"]:,.0f}</div>'
                                f'<div style="font-size:11px;color:#64748b;">{int(worst["Trades"])} trades · {worst["Win Rate %"]:.0f}% win rate</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div style="background:#f0fdf4;border-radius:12px;padding:16px 18px;">'
                                f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#15803d;">✅ Clean Sheet</div>'
                                f'<div style="font-size:15px;font-weight:700;color:#111;margin-top:6px;">No losing rules</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    with ih3:
                        total_rules = len(rule_df)
                        profitable_count = len(profitable)
                        st.markdown(
                            f'<div style="background:#eff6ff;border-radius:12px;padding:16px 18px;">'
                            f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#1d4ed8;">📊 Rules Used</div>'
                            f'<div style="font-size:28px;font-weight:800;color:#111;margin-top:6px;">{total_rules}</div>'
                            f'<div style="font-size:11px;color:#64748b;">{profitable_count} profitable · {len(unprofitable)} losing</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                    # --- SORT CONTROL ---
                    sort_by = st.selectbox(
                        "Sort by",
                        ["Total P&L", "Win Rate %", "Avg P&L", "Trades"],
                        index=0,
                        key="br_sort"
                    )
                    ascending = False
                    rule_df_sorted = rule_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

                    # --- LEADERBOARD ---
                    st.markdown("**Leaderboard**")
                    # Header row
                    hcols = st.columns([3, 1, 1, 1.2, 1.2, 1, 1.5])
                    hcols[0].markdown("**Rule**")
                    hcols[1].markdown("**Trades**")
                    hcols[2].markdown("**Win %**")
                    hcols[3].markdown("**Avg P&L**")
                    hcols[4].markdown("**Total P&L**")
                    hcols[5].markdown("**Avg R**")
                    hcols[6].markdown("**Status**")

                    for _, row in rule_df_sorted.iterrows():
                        # Status badge logic
                        small_sample = row['Trades'] < 5
                        if row['Total P&L'] >= 5000 and row['Trades'] >= 5 and row['Win Rate %'] >= 50:
                            badge = '<span style="background:#dcfce7;color:#15803d;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;">⭐ Winner</span>'
                        elif row['Total P&L'] > 0:
                            badge = '<span style="background:#f0fdf4;color:#15803d;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;">✅ Working</span>'
                        elif row['Total P&L'] < 0:
                            badge = '<span style="background:#fee2e2;color:#b91c1c;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;">🚨 Losing</span>'
                        else:
                            badge = '<span style="background:#f1f5f9;color:#64748b;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;">— Flat</span>'

                        if small_sample:
                            badge += ' <span style="background:#fef3c7;color:#b45309;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;" title="Small sample (<5 trades)">🧪</span>'

                        pl_color = "#15803d" if row['Total P&L'] > 0 else ("#b91c1c" if row['Total P&L'] < 0 else "#64748b")
                        avg_r_txt = f"{row['Avg R']:.2f}R" if pd.notna(row['Avg R']) else "—"

                        rcols = st.columns([3, 1, 1, 1.2, 1.2, 1, 1.5])
                        # Show 📝 icon if a saved note exists for this rule
                        has_note = row['Rule'] in br_notes_map and (br_notes_map[row['Rule']][0] or br_notes_map[row['Rule']][1])
                        rule_label = f"**{row['Rule']}**" + (" 📝" if has_note else "")
                        rcols[0].markdown(rule_label)
                        rcols[1].markdown(f"{int(row['Trades'])}")
                        rcols[2].markdown(f"{row['Win Rate %']:.0f}%")
                        rcols[3].markdown(f"<span style='color:{pl_color};font-weight:600;'>${row['Avg P&L']:,.0f}</span>", unsafe_allow_html=True)
                        rcols[4].markdown(f"<span style='color:{pl_color};font-weight:700;'>${row['Total P&L']:,.0f}</span>", unsafe_allow_html=True)
                        rcols[5].markdown(avg_r_txt)
                        rcols[6].markdown(badge, unsafe_allow_html=True)

                    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                    # Rule status options — shared by Buy Rules and Sell Rules tabs
                    RULE_STATUS_OPTIONS = [
                        "— no status —",
                        "✅ Validated",
                        "✏️ Modify",
                        "⚠️ Review",
                        "🛑 Avoid",
                    ]

                    # --- DRILL-DOWN ---
                    st.markdown("**Drill-down: study an individual rule**")
                    all_rules = rule_df_sorted['Rule'].tolist()
                    sel_rule = st.selectbox("Select a rule to see its trades", all_rules, key="br_drill")
                    if sel_rule:
                        rule_trades = br_closed_2026[br_closed_2026['_Rule'] == sel_rule].copy()
                        rule_trades['Open_Date_str'] = pd.to_datetime(rule_trades['Open_Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                        rule_trades['Closed_Date_str'] = pd.to_datetime(rule_trades['Closed_Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                        # Keep P&L and R numeric so column-header sorting works correctly
                        display_df = rule_trades[['Trade_ID', 'Ticker', 'Open_Date_str', 'Closed_Date_str', 'Realized_PL', '_R']].copy()
                        display_df.columns = ['Trade ID', 'Ticker', 'Opened', 'Closed', 'P&L', 'R']
                        display_df = display_df.sort_values('P&L', ascending=False).reset_index(drop=True)
                        st.dataframe(
                            display_df,
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                'Trade ID': st.column_config.TextColumn('Trade ID', width='small'),
                                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                                'Opened': st.column_config.TextColumn('Opened', width='small'),
                                'Closed': st.column_config.TextColumn('Closed', width='small'),
                                'P&L': st.column_config.NumberColumn('P&L', format='$%.2f'),
                                'R': st.column_config.NumberColumn('R', format='%.2fR'),
                            },
                        )

                        # --- RULE OBSERVATIONS ---
                        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                        st.markdown(f"### 📝 Rule Observations — {sel_rule}")
                        existing_note, existing_status = br_notes_map.get(sel_rule, ('', ''))
                        status_index = 0
                        if existing_status and existing_status in RULE_STATUS_OPTIONS:
                            status_index = RULE_STATUS_OPTIONS.index(existing_status)
                        note_status = st.selectbox(
                            "Status",
                            RULE_STATUS_OPTIONS,
                            index=status_index,
                            key=f"br_note_status_{sel_rule}",
                        )
                        note_text = st.text_area(
                            "What did you observe? What's working or not working with this rule?",
                            value=existing_note,
                            key=f"br_note_text_{sel_rule}",
                            height=120,
                            placeholder="e.g. br3.1 Reclaim 21e — I'm using this on late-stage bases where the 21EMA has already been tested multiple times. Only take this setup on first pullbacks to a fresh 21EMA.",
                        )
                        if st.button("Save observation", key=f"br_note_save_{sel_rule}"):
                            if not USE_DATABASE:
                                st.warning("Database unavailable — notes require DB mode.")
                            elif not hasattr(db, 'save_rule_note'):
                                st.error("Rule notes feature not yet loaded — please reboot the app from Streamlit Cloud (Manage app → Reboot) to pick up the latest db_layer.py.")
                            else:
                                save_status = '' if note_status == RULE_STATUS_OPTIONS[0] else note_status
                                ok = db.save_rule_note(CURR_PORT_NAME, 'buy', sel_rule, note_text, save_status)
                                if ok:
                                    st.toast(f"Note saved for {sel_rule} ✅")
                                else:
                                    st.error("Save failed")

        # --- TAB 3: SELL RULES (Rule Studio — 2026 only) ---
        with tab_sell_rules:
            st.subheader("🔴 Sell Rules — Exit Quality in 2026")
            st.caption("Study your exit rules. Which are protecting capital, which are capturing profits, which are hurting you.")

            sr_source = df_s_raw.copy()
            sr_source['Closed_Date'] = pd.to_datetime(sr_source['Closed_Date'], errors='coerce')
            sr_closed_2026 = sr_source[
                (sr_source['Status'] == 'CLOSED') &
                (sr_source['Closed_Date'].dt.year == 2026)
            ].copy()

            if 'Sell_Rule' not in sr_closed_2026.columns or sr_closed_2026.empty:
                st.info("💡 No 2026 closed trades with sell rule data yet.")
            else:
                sr_closed_2026['_SellRule'] = sr_closed_2026['Sell_Rule'].astype(str).str.strip()
                sr_closed_2026 = sr_closed_2026[
                    (sr_closed_2026['_SellRule'] != '') &
                    (sr_closed_2026['_SellRule'].str.lower() != 'nan')
                ]

                if sr_closed_2026.empty:
                    st.info("💡 No 2026 closed trades with sell rule data yet.")
                else:
                    # Load saved sell rule notes for this portfolio
                    sr_notes_map = {}
                    if USE_DATABASE and hasattr(db, 'get_rule_notes'):
                        try:
                            sr_notes_map = db.get_rule_notes(CURR_PORT_NAME, 'sell')
                        except Exception:
                            sr_notes_map = {}

                    # Compute R-multiple and hold days
                    sr_closed_2026['_R'] = sr_closed_2026.apply(
                        lambda r: (float(r['Realized_PL']) / float(r['Risk_Budget']))
                        if r.get('Risk_Budget') and float(r['Risk_Budget']) > 0 else None,
                        axis=1
                    )
                    sr_closed_2026['Open_Date_DT'] = pd.to_datetime(sr_closed_2026['Open_Date'], errors='coerce')
                    sr_closed_2026['_Hold'] = (sr_closed_2026['Closed_Date'] - sr_closed_2026['Open_Date_DT']).dt.total_seconds() / 86400

                    # Aggregate per sell rule
                    sell_stats = []
                    for rule_name, group in sr_closed_2026.groupby('_SellRule'):
                        uses = len(group)
                        total_pl = float(group['Realized_PL'].sum())
                        avg_pl = float(group['Realized_PL'].mean())
                        r_vals = group['_R'].dropna()
                        avg_r = float(r_vals.mean()) if not r_vals.empty else None
                        avg_hold = float(group['_Hold'].mean()) if group['_Hold'].notna().any() else None
                        winners_pct = (len(group[group['Realized_PL'] > 0]) / uses * 100) if uses > 0 else 0.0
                        sell_stats.append({
                            'Sell Rule': rule_name,
                            'Uses': uses,
                            'Avg P&L': avg_pl,
                            'Total P&L': total_pl,
                            'Avg R': avg_r,
                            'Avg Hold': avg_hold,
                            'Winners %': winners_pct,
                        })

                    sell_df = pd.DataFrame(sell_stats)

                    # --- HEADER INSIGHTS ---
                    if not sell_df.empty:
                        # Best capital protector = smallest negative avg loss among negative rules
                        negatives = sell_df[sell_df['Avg P&L'] < 0]
                        positives = sell_df[sell_df['Avg P&L'] > 0]
                        # Most used
                        most_used = sell_df.sort_values('Uses', ascending=False).iloc[0]

                        ih1, ih2, ih3 = st.columns(3)
                        with ih1:
                            if not negatives.empty:
                                best_protect = negatives.sort_values('Avg P&L', ascending=False).iloc[0]
                                st.markdown(
                                    f'<div style="background:#dcfce7;border-radius:12px;padding:16px 18px;">'
                                    f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#15803d;">🛡️ Best Protector</div>'
                                    f'<div style="font-size:15px;font-weight:700;color:#111;margin-top:6px;">{best_protect["Sell Rule"]}</div>'
                                    f'<div style="font-size:20px;font-weight:800;color:#15803d;">${best_protect["Avg P&L"]:,.0f}</div>'
                                    f'<div style="font-size:11px;color:#64748b;">smallest avg loss · {int(best_protect["Uses"])} uses</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f'<div style="background:#f0fdf4;border-radius:12px;padding:16px 18px;">'
                                    f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#15803d;">🛡️ Best Protector</div>'
                                    f'<div style="font-size:15px;font-weight:700;color:#111;margin-top:6px;">No losing exits yet</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                        with ih2:
                            if not positives.empty:
                                best_capture = positives.sort_values('Total P&L', ascending=False).iloc[0]
                                st.markdown(
                                    f'<div style="background:#dbeafe;border-radius:12px;padding:16px 18px;">'
                                    f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#1d4ed8;">💰 Top Profit Capture</div>'
                                    f'<div style="font-size:15px;font-weight:700;color:#111;margin-top:6px;">{best_capture["Sell Rule"]}</div>'
                                    f'<div style="font-size:20px;font-weight:800;color:#1d4ed8;">${best_capture["Total P&L"]:,.0f}</div>'
                                    f'<div style="font-size:11px;color:#64748b;">avg ${best_capture["Avg P&L"]:,.0f} · {int(best_capture["Uses"])} uses</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f'<div style="background:#fef3c7;border-radius:12px;padding:16px 18px;">'
                                    f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#b45309;">💰 Top Profit Capture</div>'
                                    f'<div style="font-size:15px;font-weight:700;color:#111;margin-top:6px;">No winning exits yet</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                        with ih3:
                            st.markdown(
                                f'<div style="background:#f1f5f9;border-radius:12px;padding:16px 18px;">'
                                f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#475569;">📊 Most Used Exit</div>'
                                f'<div style="font-size:15px;font-weight:700;color:#111;margin-top:6px;">{most_used["Sell Rule"]}</div>'
                                f'<div style="font-size:20px;font-weight:800;color:#111;">{int(most_used["Uses"])} uses</div>'
                                f'<div style="font-size:11px;color:#64748b;">avg ${most_used["Avg P&L"]:,.0f}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                    # --- SORT CONTROL ---
                    sr_sort = st.selectbox(
                        "Sort by",
                        ["Total P&L", "Uses", "Avg P&L", "Winners %"],
                        index=0,
                        key="sr_sort"
                    )
                    sell_df_sorted = sell_df.sort_values(sr_sort, ascending=False).reset_index(drop=True)

                    # --- LEADERBOARD ---
                    st.markdown("**Leaderboard**")
                    shcols = st.columns([3, 1, 1.2, 1.2, 1, 1, 1, 1.5])
                    shcols[0].markdown("**Sell Rule**")
                    shcols[1].markdown("**Uses**")
                    shcols[2].markdown("**Avg P&L**")
                    shcols[3].markdown("**Total P&L**")
                    shcols[4].markdown("**Avg R**")
                    shcols[5].markdown("**Hold**")
                    shcols[6].markdown("**Win %**")
                    shcols[7].markdown("**Status**")

                    for _, row in sell_df_sorted.iterrows():
                        small_sample = row['Uses'] < 5
                        avg_r_val = row['Avg R']
                        avg_pl_val = row['Avg P&L']

                        # Sell-side status framing
                        if avg_pl_val > 0:
                            badge = '<span style="background:#dcfce7;color:#15803d;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;">💰 Capturing</span>'
                        elif pd.notna(avg_r_val) and avg_r_val < -1.0:
                            badge = '<span style="background:#fee2e2;color:#b91c1c;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;">🚨 Hurting</span>'
                        elif avg_pl_val < 0:
                            badge = '<span style="background:#f0fdf4;color:#15803d;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;">🛡️ Protecting</span>'
                        else:
                            badge = '<span style="background:#f1f5f9;color:#64748b;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;">— Flat</span>'

                        if small_sample:
                            badge += ' <span style="background:#fef3c7;color:#b45309;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;">🧪</span>'

                        pl_color = "#15803d" if row['Total P&L'] > 0 else ("#b91c1c" if row['Total P&L'] < 0 else "#64748b")
                        avg_r_txt = f"{avg_r_val:.2f}R" if pd.notna(avg_r_val) else "—"
                        hold_txt = f"{row['Avg Hold']:.0f}d" if pd.notna(row['Avg Hold']) else "—"

                        srcols = st.columns([3, 1, 1.2, 1.2, 1, 1, 1, 1.5])
                        # Show 📝 icon if a saved note exists for this rule
                        sr_has_note = row['Sell Rule'] in sr_notes_map and (
                            sr_notes_map[row['Sell Rule']][0] or sr_notes_map[row['Sell Rule']][1]
                        )
                        sr_label = f"**{row['Sell Rule']}**" + (" 📝" if sr_has_note else "")
                        srcols[0].markdown(sr_label)
                        srcols[1].markdown(f"{int(row['Uses'])}")
                        srcols[2].markdown(f"<span style='color:{pl_color};font-weight:600;'>${row['Avg P&L']:,.0f}</span>", unsafe_allow_html=True)
                        srcols[3].markdown(f"<span style='color:{pl_color};font-weight:700;'>${row['Total P&L']:,.0f}</span>", unsafe_allow_html=True)
                        srcols[4].markdown(avg_r_txt)
                        srcols[5].markdown(hold_txt)
                        srcols[6].markdown(f"{row['Winners %']:.0f}%")
                        srcols[7].markdown(badge, unsafe_allow_html=True)

                    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                    # --- DRILL-DOWN ---
                    st.markdown("**Drill-down: study an individual sell rule**")
                    all_sell_rules = sell_df_sorted['Sell Rule'].tolist()
                    sel_sr = st.selectbox("Select a sell rule to see its exits", all_sell_rules, key="sr_drill")
                    if sel_sr:
                        sr_trades = sr_closed_2026[sr_closed_2026['_SellRule'] == sel_sr].copy()
                        sr_trades['Open_Date_str'] = pd.to_datetime(sr_trades['Open_Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                        sr_trades['Closed_Date_str'] = pd.to_datetime(sr_trades['Closed_Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                        # Keep numeric columns so column-header sorting works correctly
                        sr_display = sr_trades[['Trade_ID', 'Ticker', 'Open_Date_str', 'Closed_Date_str', 'Realized_PL', '_R', '_Hold']].copy()
                        sr_display.columns = ['Trade ID', 'Ticker', 'Opened', 'Closed', 'P&L', 'R', 'Hold']
                        sr_display = sr_display.sort_values('P&L', ascending=False).reset_index(drop=True)
                        st.dataframe(
                            sr_display,
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                'Trade ID': st.column_config.TextColumn('Trade ID', width='small'),
                                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                                'Opened': st.column_config.TextColumn('Opened', width='small'),
                                'Closed': st.column_config.TextColumn('Closed', width='small'),
                                'P&L': st.column_config.NumberColumn('P&L', format='$%.2f'),
                                'R': st.column_config.NumberColumn('R', format='%.2fR'),
                                'Hold': st.column_config.NumberColumn('Hold', format='%.0fd'),
                            },
                        )

                        # --- RULE OBSERVATIONS ---
                        # Status list mirrored from Buy Rules tab
                        SR_STATUS_OPTIONS = [
                            "— no status —",
                            "✅ Validated",
                            "✏️ Modify",
                            "⚠️ Review",
                            "🛑 Avoid",
                        ]
                        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                        st.markdown(f"### 📝 Rule Observations — {sel_sr}")
                        sr_existing_note, sr_existing_status = sr_notes_map.get(sel_sr, ('', ''))
                        sr_status_index = 0
                        if sr_existing_status and sr_existing_status in SR_STATUS_OPTIONS:
                            sr_status_index = SR_STATUS_OPTIONS.index(sr_existing_status)
                        sr_note_status = st.selectbox(
                            "Status",
                            SR_STATUS_OPTIONS,
                            index=sr_status_index,
                            key=f"sr_note_status_{sel_sr}",
                        )
                        sr_note_text = st.text_area(
                            "What did you observe? Is this exit protecting capital or cutting winners short?",
                            value=sr_existing_note,
                            key=f"sr_note_text_{sel_sr}",
                            height=120,
                            placeholder="e.g. sr3 Portfolio Management — I'm cutting too early on winners to rebalance. Next time, only trim trades that are failing, not ones near targets.",
                        )
                        if st.button("Save observation", key=f"sr_note_save_{sel_sr}"):
                            if not USE_DATABASE:
                                st.warning("Database unavailable — notes require DB mode.")
                            elif not hasattr(db, 'save_rule_note'):
                                st.error("Rule notes feature not yet loaded — please reboot the app from Streamlit Cloud (Manage app → Reboot) to pick up the latest db_layer.py.")
                            else:
                                sr_save_status = '' if sr_note_status == SR_STATUS_OPTIONS[0] else sr_note_status
                                ok = db.save_rule_note(CURR_PORT_NAME, 'sell', sel_sr, sr_note_text, sr_save_status)
                                if ok:
                                    st.toast(f"Note saved for {sel_sr} ✅")
                                else:
                                    st.error("Save failed")

        # --- TAB 3: DRAWDOWN DETECTIVE (START DEC 16, 2025) ---
        # --- TAB 4: DRAWDOWN DISCIPLINE (deck compliance tracker) ---
        with tab_dd:
            st.subheader("🛡️ Drawdown Discipline")
            st.caption("Did you follow your own deck rules? Each historical crossing of L1 (−7.5%), L2 (−12.5%), or L3 (−15%) is logged with a pass/fail verdict.")

            RESET_DATE = pd.Timestamp("2026-02-24")

            if df_j.empty:
                st.info("💡 No journal data found. Drawdown Discipline requires End NLV history.")
            else:
                dd_j = df_j[df_j['Day'] >= RESET_DATE].copy()
                if dd_j.empty or 'End NLV' not in dd_j.columns:
                    st.info("💡 No journal entries since the 2026-02-24 reset date. This tab populates as new entries are logged.")
                else:
                    dd_j = dd_j.sort_values('Day').reset_index(drop=True)
                    dd_j['End NLV'] = pd.to_numeric(dd_j['End NLV'], errors='coerce')
                    dd_j = dd_j.dropna(subset=['End NLV'])

                    # Rolling peak NLV from reset date
                    dd_j['Peak_NLV'] = dd_j['End NLV'].cummax()
                    dd_j['DD_Pct'] = (dd_j['End NLV'] - dd_j['Peak_NLV']) / dd_j['Peak_NLV'] * 100
                    dd_j['Pct_Invested'] = pd.to_numeric(dd_j.get('% Invested', 0), errors='coerce').fillna(0)

                    # Current state
                    current_dd = dd_j['DD_Pct'].iloc[-1]
                    current_nlv = dd_j['End NLV'].iloc[-1]
                    current_peak = dd_j['Peak_NLV'].iloc[-1]
                    current_exposure = dd_j['Pct_Invested'].iloc[-1]

                    deck_levels = {
                        'L1': (-7.5, "Remove Margin", "#eab308"),
                        'L2': (-12.5, "Max 30% Invested", "#f97316"),
                        'L3': (-15.0, "Go to Cash", "#dc2626"),
                    }

                    # ============================================================
                    # SECTION 1: LIVE STATUS
                    # ============================================================
                    st.markdown("**Live Status** — where you are right now vs each deck")
                    ls_cols = st.columns(3)
                    for (lvl, (thresh, rule_txt, color)), lc in zip(deck_levels.items(), ls_cols):
                        distance = current_dd - thresh  # positive = safe, negative = breached
                        if current_dd <= thresh:
                            status_bg = "#fee2e2"
                            status_txt = "#b91c1c"
                            status_icon = "🚨 BREACHED"
                            sub = f"{abs(distance):.2f}% into breach"
                        elif distance < 2.0:
                            status_bg = "#fef3c7"
                            status_txt = "#b45309"
                            status_icon = "⚠️ Close"
                            sub = f"{distance:.2f}% from deck"
                        else:
                            status_bg = "#f0fdf4"
                            status_txt = "#15803d"
                            status_icon = "✅ Safe"
                            sub = f"{distance:.2f}% from deck"
                        lc.markdown(
                            f'<div style="background:{status_bg};border-left:4px solid {color};'
                            f'border-radius:10px;padding:14px 16px;height:100%;">'
                            f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
                            f'letter-spacing:0.06em;color:#64748b;">{lvl} · {thresh:.1f}%</div>'
                            f'<div style="font-size:13px;font-weight:600;color:#111;margin-top:2px;">{rule_txt}</div>'
                            f'<div style="font-size:18px;font-weight:800;color:{status_txt};margin-top:6px;">{status_icon}</div>'
                            f'<div style="font-size:11px;color:#64748b;margin-top:2px;">{sub}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # Current snapshot line
                    st.markdown(
                        f'<div style="margin-top:12px;padding:10px 14px;background:#f8fafc;border-radius:8px;'
                        f'font-size:13px;color:#334155;">'
                        f'Current DD: <b>{current_dd:.2f}%</b> · '
                        f'NLV: <b>${current_nlv:,.0f}</b> · '
                        f'Peak: <b>${current_peak:,.0f}</b> · '
                        f'Exposure: <b>{current_exposure:.1f}%</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                    # ============================================================
                    # SECTION 2: DECK CROSSINGS LOG
                    # ============================================================
                    # Detect EOD crossings: a new crossing starts when End NLV first drops
                    # below a deck threshold, and ends when it rises above it again.
                    crossings = []
                    for lvl, (thresh, rule_txt, color) in deck_levels.items():
                        dd_j[f'_below_{lvl}'] = dd_j['DD_Pct'] <= thresh
                        # Group contiguous periods of below
                        dd_j[f'_grp_{lvl}'] = (dd_j[f'_below_{lvl}'] != dd_j[f'_below_{lvl}'].shift()).cumsum()
                        for grp_id, grp in dd_j[dd_j[f'_below_{lvl}']].groupby(f'_grp_{lvl}'):
                            start_idx = grp.index.min()
                            end_idx = grp.index.max()
                            max_depth_idx = grp['DD_Pct'].idxmin()
                            # Exposure at the bar BEFORE the start (when you could still act)
                            prior_idx = max(0, start_idx - 1)
                            exposure_at_start = dd_j.at[prior_idx, 'Pct_Invested']
                            exposure_at_trough = dd_j.at[max_depth_idx, 'Pct_Invested']
                            max_depth = dd_j.at[max_depth_idx, 'DD_Pct']
                            start_date = dd_j.at[start_idx, 'Day']
                            end_date = dd_j.at[end_idx, 'Day']
                            duration = (end_date - start_date).days + 1

                            # Recovery: find next bar where End NLV >= peak at start
                            peak_at_start = dd_j.at[start_idx, 'Peak_NLV']
                            recovery_days = None
                            recovered = False
                            for k in range(end_idx + 1, len(dd_j)):
                                if dd_j.at[k, 'End NLV'] >= peak_at_start:
                                    recovery_days = (dd_j.at[k, 'Day'] - start_date).days
                                    recovered = True
                                    break
                            if not recovered:
                                # Check if currently recovered
                                if dd_j['End NLV'].iloc[-1] >= peak_at_start:
                                    recovery_days = (dd_j['Day'].iloc[-1] - start_date).days
                                    recovered = True

                            # Per-deck verdict framing — different semantics per level
                            # L1 = informational (guide to slow down), natural portfolio heat
                            # L2 = confirmation to start backing off, should be reducing
                            # L3 = mandatory exit, must be out
                            exposure_drop = exposure_at_start - exposure_at_trough
                            if lvl == 'L1':
                                # L1: Aware (no adding) / Drifted (small add) / Leveraged Up (meaningful add)
                                if exposure_at_trough <= exposure_at_start:
                                    verdict = 'L1_Aware'
                                elif (exposure_at_trough - exposure_at_start) < 10:
                                    verdict = 'L1_Drifted'
                                else:
                                    verdict = 'L1_Leveraged'
                            elif lvl == 'L2':
                                # L2: Reducing (meaningful cut) / Partial / Not Reducing
                                if exposure_drop >= 20 or exposure_at_trough <= 50:
                                    verdict = 'L2_Reducing'
                                elif exposure_drop >= 5:
                                    verdict = 'L2_Partial'
                                else:
                                    verdict = 'L2_NotReducing'
                            else:  # L3
                                # L3: Exited (≤20%) / Partial Exit (20-50%) / Still In (>50%)
                                if exposure_at_trough <= 20:
                                    verdict = 'L3_Exited'
                                elif exposure_at_trough <= 50:
                                    verdict = 'L3_PartialExit'
                                else:
                                    verdict = 'L3_StillIn'

                            # Damage in window
                            closed_in_window = all_closed[
                                (pd.to_datetime(all_closed['Closed_Date'], errors='coerce') >= start_date) &
                                (pd.to_datetime(all_closed['Closed_Date'], errors='coerce') <= end_date)
                            ]
                            losses_in_window = closed_in_window[closed_in_window['Realized_PL'] < 0]['Realized_PL'].sum()

                            crossings.append({
                                'Deck': lvl,
                                'Deck_Thresh': thresh,
                                'Start': start_date,
                                'End': end_date,
                                'Duration': duration,
                                'Max_Depth': max_depth,
                                'Exposure_Start': exposure_at_start,
                                'Exposure_Trough': exposure_at_trough,
                                'Exposure_Drop': exposure_drop,
                                'Verdict': verdict,
                                'Recovery_Days': recovery_days,
                                'Recovered': recovered,
                                'Losses_In_Window': float(losses_in_window),
                            })

                    # Sort crossings newest first — ALL crossings kept for later calcs
                    crossings_sorted_all = sorted(crossings, key=lambda c: c['Start'], reverse=True)
                    # The log itself only shows L2 and L3 (L1 is for Live Status awareness only)
                    crossings_for_log = [c for c in crossings_sorted_all if c['Deck'] in ('L2', 'L3')]

                    st.markdown("**Deck Crossings Log** <span style='font-size:12px;color:#64748b;font-weight:400;'>— L2 & L3 only (L1 is informational, see Live Status above)</span>",
                                unsafe_allow_html=True)
                    if not crossings_for_log:
                        st.success("✅ No L2 or L3 crossings since the 2026-02-24 reset. Keep it up.")
                    else:
                        # Load notes for this portfolio
                        notes_map = {}
                        if USE_DATABASE:
                            try:
                                notes_map = db.get_drawdown_notes(CURR_PORT_NAME)
                            except Exception:
                                notes_map = {}

                        verdict_style = {
                            'L1_Aware':       ('#dcfce7', '#15803d', '✅ Aware'),
                            'L1_Drifted':     ('#fef3c7', '#b45309', '⚠️ Drifted'),
                            'L1_Leveraged':   ('#fee2e2', '#b91c1c', '🚨 Leveraged Up'),
                            'L2_Reducing':    ('#dcfce7', '#15803d', '✅ Reducing'),
                            'L2_Partial':     ('#fef3c7', '#b45309', '⚠️ Partial'),
                            'L2_NotReducing': ('#fee2e2', '#b91c1c', '🚨 Not Reducing'),
                            'L3_Exited':      ('#dcfce7', '#15803d', '✅ Exited'),
                            'L3_PartialExit': ('#fef3c7', '#b45309', '⚠️ Partial Exit'),
                            'L3_StillIn':     ('#fee2e2', '#b91c1c', '🚨 Still In'),
                        }

                        for c in crossings_for_log:
                            v_bg, v_tx, v_lbl = verdict_style[c['Verdict']]
                            start_str = c['Start'].strftime('%b %d, %Y') if hasattr(c['Start'], 'strftime') else str(c['Start'])
                            end_str = c['End'].strftime('%b %d, %Y') if hasattr(c['End'], 'strftime') else str(c['End'])
                            rec_str = f"{c['Recovery_Days']}d" if c['Recovery_Days'] is not None else "ongoing"
                            exposure_delta = c['Exposure_Start'] - c['Exposure_Trough']

                            st.markdown(
                                f'<div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;'
                                f'padding:14px 16px;margin-bottom:10px;">'
                                f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
                                f'<div style="font-size:15px;font-weight:700;color:#111;">'
                                f'{c["Deck"]} · {c["Deck_Thresh"]:.1f}% '
                                f'<span style="color:#64748b;font-weight:500;font-size:12px;">({start_str})</span>'
                                f'</div>'
                                f'<div style="background:{v_bg};color:{v_tx};padding:4px 10px;border-radius:6px;'
                                f'font-size:12px;font-weight:700;">{v_lbl}</div>'
                                f'</div>'
                                f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px;">'
                                f'<div><div style="font-size:10px;color:#64748b;text-transform:uppercase;font-weight:600;">Max Depth</div>'
                                f'<div style="font-size:15px;font-weight:700;color:#b91c1c;">{c["Max_Depth"]:.2f}%</div></div>'
                                f'<div><div style="font-size:10px;color:#64748b;text-transform:uppercase;font-weight:600;">Exposure</div>'
                                f'<div style="font-size:15px;font-weight:700;color:#111;">{c["Exposure_Start"]:.0f}% → {c["Exposure_Trough"]:.0f}%</div>'
                                f'<div style="font-size:10px;color:#64748b;">Δ {exposure_delta:+.0f}pp</div></div>'
                                f'<div><div style="font-size:10px;color:#64748b;text-transform:uppercase;font-weight:600;">Recovery</div>'
                                f'<div style="font-size:15px;font-weight:700;color:#111;">{rec_str}</div></div>'
                                f'<div><div style="font-size:10px;color:#64748b;text-transform:uppercase;font-weight:600;">Realized in Window</div>'
                                f'<div style="font-size:15px;font-weight:700;color:#b91c1c;">${c["Losses_In_Window"]:,.0f}</div></div>'
                                f'</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            # Editable note for this crossing
                            note_key = f"{c['Deck']}_{c['Start'].strftime('%Y-%m-%d') if hasattr(c['Start'], 'strftime') else str(c['Start'])[:10]}"
                            existing_note = notes_map.get(note_key, '')
                            with st.expander(f"📝 Lessons & notes — {c['Deck']} {start_str}", expanded=False):
                                note_val = st.text_area(
                                    "What happened? What would you do differently?",
                                    value=existing_note,
                                    key=f"note_{note_key}",
                                    height=100,
                                    placeholder="e.g. I held through L1 because I thought the market would bounce — instead it kept falling. Next time, take at least partial off at L1.",
                                )
                                save_col, _ = st.columns([1, 4])
                                if save_col.button("Save note", key=f"savenote_{note_key}"):
                                    if USE_DATABASE:
                                        ok = db.save_drawdown_note(
                                            CURR_PORT_NAME,
                                            c['Deck'],
                                            c['Start'],
                                            note_val,
                                        )
                                        if ok:
                                            st.toast("Note saved ✅")
                                        else:
                                            st.error("Save failed")
                                    else:
                                        st.warning("Database unavailable — notes require DB mode.")

                        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                        # ============================================================
                        # SECTION 3: COST OF NON-COMPLIANCE (L2 + L3 only)
                        # ============================================================
                        # Group verdicts into tiers — only L2/L3 count as "compliance"
                        l2_l3 = [c for c in crossings_sorted_all if c['Deck'] in ('L2', 'L3')]
                        best_set = [c for c in l2_l3 if c['Verdict'] in ('L2_Reducing', 'L3_Exited')]
                        middle_set = [c for c in l2_l3 if c['Verdict'] in ('L2_Partial', 'L3_PartialExit')]
                        worst_set = [c for c in l2_l3 if c['Verdict'] in ('L2_NotReducing', 'L3_StillIn')]

                        worst_cost = sum(abs(c['Losses_In_Window']) for c in worst_set)
                        middle_cost = sum(abs(c['Losses_In_Window']) for c in middle_set)
                        best_cost = sum(abs(c['Losses_In_Window']) for c in best_set)

                        st.markdown("**Cost of Non-Compliance**")
                        cost_cols = st.columns(3)
                        cost_cols[0].markdown(
                            f'<div style="background:#fee2e2;border-radius:10px;padding:14px 16px;">'
                            f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#b91c1c;">🚨 Non-Compliance Losses</div>'
                            f'<div style="font-size:24px;font-weight:800;color:#b91c1c;margin-top:4px;">${worst_cost:,.0f}</div>'
                            f'<div style="font-size:11px;color:#64748b;">{len(worst_set)} L2/L3 crossing(s) ignored</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        cost_cols[1].markdown(
                            f'<div style="background:#fef3c7;border-radius:10px;padding:14px 16px;">'
                            f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#b45309;">⚠️ Partial-Compliance Losses</div>'
                            f'<div style="font-size:24px;font-weight:800;color:#b45309;margin-top:4px;">${middle_cost:,.0f}</div>'
                            f'<div style="font-size:11px;color:#64748b;">{len(middle_set)} L2/L3 crossing(s) partially cut</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        cost_cols[2].markdown(
                            f'<div style="background:#dcfce7;border-radius:10px;padding:14px 16px;">'
                            f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#15803d;">✅ Rule-Respected Losses</div>'
                            f'<div style="font-size:24px;font-weight:800;color:#15803d;margin-top:4px;">${best_cost:,.0f}</div>'
                            f'<div style="font-size:11px;color:#64748b;">{len(best_set)} L2/L3 crossing(s) honored</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption("Only L2 (start backing off) and L3 (get out) crossings count toward compliance — L1 is informational only.")

                        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                        # ============================================================
                        # SECTION 4: DISCIPLINE REPORT CARD (weighted: L1=0, L2=1, L3=3)
                        # ============================================================
                        # Weight each crossing by deck level. L1 doesn't count.
                        deck_weight = {'L1': 0, 'L2': 1, 'L3': 3}
                        verdict_score_map = {
                            'L1_Aware': 1.0, 'L1_Drifted': 0.5, 'L1_Leveraged': 0.0,
                            'L2_Reducing': 1.0, 'L2_Partial': 0.5, 'L2_NotReducing': 0.0,
                            'L3_Exited': 1.0, 'L3_PartialExit': 0.5, 'L3_StillIn': 0.0,
                        }
                        total_weight = sum(deck_weight[c['Deck']] for c in crossings_sorted_all)
                        weighted_score = sum(
                            deck_weight[c['Deck']] * verdict_score_map[c['Verdict']]
                            for c in crossings_sorted_all
                        )
                        if total_weight > 0:
                            compliance_score = weighted_score / total_weight
                        else:
                            # No L2/L3 crossings — full marks by default
                            compliance_score = 1.0
                        compliance_pct = compliance_score * 100

                        if compliance_pct >= 90:
                            grade, g_color, g_bg = 'A', '#15803d', '#dcfce7'
                        elif compliance_pct >= 75:
                            grade, g_color, g_bg = 'B', '#16a34a', '#f0fdf4'
                        elif compliance_pct >= 60:
                            grade, g_color, g_bg = 'C', '#b45309', '#fef3c7'
                        elif compliance_pct >= 40:
                            grade, g_color, g_bg = 'D', '#d97706', '#fffbeb'
                        else:
                            grade, g_color, g_bg = 'F', '#b91c1c', '#fee2e2'

                        # Report card stats — scoped to L2/L3 for discipline metrics
                        total_crossings = len(crossings_sorted_all)
                        graded_crossings = [c for c in crossings_sorted_all if c['Deck'] in ('L2', 'L3')]
                        recovered_crossings = [c for c in graded_crossings if c['Recovered']]
                        avg_recovery = (
                            sum(c['Recovery_Days'] for c in recovered_crossings) / len(recovered_crossings)
                            if recovered_crossings else None
                        )
                        avg_depth = (
                            sum(c['Max_Depth'] for c in graded_crossings) / len(graded_crossings)
                            if graded_crossings else 0
                        )

                        # Recompute counts by group for the breakdown line
                        respected = best_set
                        partial = middle_set
                        pushed = worst_set

                        st.markdown("**Discipline Report Card**")
                        rc_col1, rc_col2 = st.columns([1, 2])
                        with rc_col1:
                            st.markdown(
                                f'<div style="background:{g_bg};border-radius:14px;padding:20px 24px;text-align:center;'
                                f'box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
                                f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
                                f'letter-spacing:0.08em;color:#64748b;">Behavior Grade</div>'
                                f'<div style="font-size:72px;font-weight:900;color:{g_color};line-height:1;margin:6px 0;">{grade}</div>'
                                f'<div style="font-size:13px;font-weight:600;color:{g_color};">{compliance_pct:.0f}% compliance</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        with rc_col2:
                            recovery_txt = f"{avg_recovery:.0f}d" if avg_recovery is not None else "—"
                            recovery_color = "#111" if avg_recovery is not None else "#64748b"
                            breakdown_html = (
                                f'<span style="color:#15803d;">{len(respected)} ✅</span> · '
                                f'<span style="color:#b45309;">{len(partial)} ⚠️</span> · '
                                f'<span style="color:#b91c1c;">{len(pushed)} 🚨</span>'
                            )
                            st.markdown(
                                f'<div style="background:#f8fafc;border-radius:10px;padding:18px 20px;height:100%;">'
                                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">'
                                f'<div><div style="font-size:11px;color:#64748b;text-transform:uppercase;font-weight:600;">Total Crossings</div>'
                                f'<div style="font-size:22px;font-weight:800;color:#111;">{total_crossings}</div></div>'
                                f'<div><div style="font-size:11px;color:#64748b;text-transform:uppercase;font-weight:600;">Avg Depth</div>'
                                f'<div style="font-size:22px;font-weight:800;color:#b91c1c;">{avg_depth:.2f}%</div></div>'
                                f'<div><div style="font-size:11px;color:#64748b;text-transform:uppercase;font-weight:600;">Avg Recovery</div>'
                                f'<div style="font-size:22px;font-weight:800;color:{recovery_color};">{recovery_txt}</div></div>'
                                f'<div><div style="font-size:11px;color:#64748b;text-transform:uppercase;font-weight:600;">Breakdown</div>'
                                f'<div style="font-size:13px;font-weight:700;color:#111;">{breakdown_html}</div></div>'
                                f'</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

        # --- TAB 4: CAREER STATS (ALL-TIME OVERVIEW) ---
        with tab_stats:
            st.subheader("🎯 All-Time Overview")
            st.caption("The headline numbers across every closed trade — start here for a quick health check.")

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

                # Top-3-winner concentration (for later)
                if not winners.empty and gross_profit > 0:
                    top_3_profit = winners.nlargest(3, 'Realized_PL')['Realized_PL'].sum()
                    top_3_pct = (top_3_profit / gross_profit * 100)
                else:
                    top_3_pct = 0.0

                # Status flags for the quality tiles
                pf_ok = profit_factor >= 1.5
                wl_ok = wl_ratio >= 2.0
                win_rate_ok = win_rate >= 40
                hold_ok = hold_ratio >= 1.0

                # ==============================================================================
                # HERO ROW — THE BIG 4 (the health check a glance)
                # ==============================================================================
                _hero_bg_pl = "#dcfce7" if total_pl >= 0 else "#fee2e2"
                _hero_txt_pl = "#15803d" if total_pl >= 0 else "#b91c1c"
                _hero_bg_pf = "#dcfce7" if pf_ok else "#fef3c7"
                _hero_txt_pf = "#15803d" if pf_ok else "#b45309"
                _hero_bg_wr = "#dcfce7" if win_rate_ok else "#fef3c7"
                _hero_txt_wr = "#15803d" if win_rate_ok else "#b45309"
                _hero_bg_ex = "#dcfce7" if expectancy >= 0 else "#fee2e2"
                _hero_txt_ex = "#15803d" if expectancy >= 0 else "#b91c1c"

                def _hero_card(label, value, sub, bg, txt):
                    return (
                        f'<div style="background:{bg};border-radius:14px;padding:20px 22px;'
                        f'box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
                        f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
                        f'letter-spacing:0.08em;color:#64748b;">{label}</div>'
                        f'<div style="font-size:32px;font-weight:800;color:{txt};'
                        f'margin-top:6px;line-height:1.1;">{value}</div>'
                        f'<div style="font-size:12px;color:#64748b;margin-top:4px;">{sub}</div>'
                        f'</div>'
                    )

                h1, h2, h3, h4 = st.columns(4)
                h1.markdown(_hero_card("Total P&L", f"${total_pl:,.0f}",
                    f"{total_trades:,} closed trades", _hero_bg_pl, _hero_txt_pl),
                    unsafe_allow_html=True)
                h2.markdown(_hero_card("Win Rate", f"{win_rate:.1f}%",
                    f"{num_winners}W · {num_losers}L",
                    _hero_bg_wr, _hero_txt_wr), unsafe_allow_html=True)
                h3.markdown(_hero_card("Profit Factor", f"{profit_factor:.2f}",
                    "≥1.5 healthy" if pf_ok else "target ≥1.5",
                    _hero_bg_pf, _hero_txt_pf), unsafe_allow_html=True)
                h4.markdown(_hero_card("Expectancy / Trade", f"${expectancy:,.0f}",
                    "avg $ per trade",
                    _hero_bg_ex, _hero_txt_ex), unsafe_allow_html=True)

                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

                # ==============================================================================
                # WINNERS vs LOSERS — side-by-side breakdown
                # ==============================================================================
                def _side_block(title, color, bg, shares, avg, largest, hold):
                    return (
                        f'<div style="background:{bg};border-left:4px solid {color};'
                        f'border-radius:10px;padding:16px 18px;height:100%;">'
                        f'<div style="font-size:13px;font-weight:700;color:{color};'
                        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">{title}</div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">'
                        f'<div><div style="font-size:11px;color:#64748b;">Count</div>'
                        f'<div style="font-size:18px;font-weight:700;color:#111;">{shares}</div></div>'
                        f'<div><div style="font-size:11px;color:#64748b;">Avg</div>'
                        f'<div style="font-size:18px;font-weight:700;color:#111;">{avg}</div></div>'
                        f'<div><div style="font-size:11px;color:#64748b;">Largest</div>'
                        f'<div style="font-size:18px;font-weight:700;color:#111;">{largest}</div></div>'
                        f'<div><div style="font-size:11px;color:#64748b;">Avg Hold</div>'
                        f'<div style="font-size:18px;font-weight:700;color:#111;">{hold}</div></div>'
                        f'</div></div>'
                    )

                w_col, l_col = st.columns(2)
                w_col.markdown(_side_block(
                    "✅ Winners", "#16a34a", "#f0fdf4",
                    f"{num_winners:,}",
                    f"${avg_win:,.0f}",
                    f"${largest_win:,.0f}",
                    f"{winners_hold:.0f}d",
                ), unsafe_allow_html=True)
                l_col.markdown(_side_block(
                    "❌ Losers", "#dc2626", "#fef2f2",
                    f"{num_losers:,}",
                    f"${avg_loss:,.0f}",
                    f"${largest_loss:,.0f}",
                    f"{losers_hold:.0f}d",
                ), unsafe_allow_html=True)

                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

                # ==============================================================================
                # QUALITY INDICATORS — 4 compact tiles with status
                # ==============================================================================
                def _quality_tile(label, value, status_text, ok):
                    color = "#16a34a" if ok else "#d97706"
                    bg = "#f0fdf4" if ok else "#fffbeb"
                    icon = "✅" if ok else "⚠️"
                    return (
                        f'<div style="background:{bg};border-radius:10px;padding:14px 16px;'
                        f'border-left:3px solid {color};">'
                        f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;'
                        f'letter-spacing:0.06em;color:#64748b;">{label}</div>'
                        f'<div style="font-size:22px;font-weight:800;color:#111;margin-top:4px;">{value}</div>'
                        f'<div style="font-size:11px;color:{color};margin-top:2px;font-weight:600;">'
                        f'{icon} {status_text}</div>'
                        f'</div>'
                    )

                st.markdown("**Quality Indicators**")
                q1, q2, q3, q4 = st.columns(4)
                q1.markdown(_quality_tile("Win/Loss Ratio", f"{wl_ratio:.2f}x",
                    "≥2.0 target", wl_ok), unsafe_allow_html=True)
                q2.markdown(_quality_tile("Hold Ratio (W/L)", f"{hold_ratio:.2f}x",
                    "letting winners run" if hold_ok else "holding losers too long",
                    hold_ok), unsafe_allow_html=True)
                q3.markdown(_quality_tile("Avg Trade", f"${avg_trade:,.0f}",
                    "positive" if avg_trade >= 0 else "negative", avg_trade >= 0),
                    unsafe_allow_html=True)
                if has_risk_data:
                    r_ok = avg_r_multiple >= 1.0
                    q4.markdown(_quality_tile("Avg R-Multiple", f"{avg_r_multiple:.2f}R",
                        f"max {max_r_multiple:.1f}R", r_ok), unsafe_allow_html=True)
                else:
                    q4.markdown(_quality_tile("Top 3 Winners", f"{top_3_pct:.1f}%",
                        "of gross profit", top_3_pct < 50), unsafe_allow_html=True)

                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                # ==============================================================================
                # LOSS DISCIPLINE — enforce the 1% rule (2026 only)
                # Risk budget tracking began in 2026, so earlier trades lack the
                # reference data needed for meaningful impact calculations.
                # Impact % = Realized P&L / NLV at trade open date (from journal)
                # ==============================================================================
                st.markdown("### 🛡️ Loss Discipline <span style='font-size:13px;font-weight:500;color:#64748b;'>— 2026 only</span>",
                            unsafe_allow_html=True)

                # Scope to trades CLOSED in 2026 — that's when risk budget tracking began
                _closed_2026 = all_closed[all_closed['Closed_Date'].dt.year == 2026]
                loss_trades = _closed_2026[_closed_2026['Realized_PL'] < 0].copy()
                impact_df = pd.DataFrame()

                if not loss_trades.empty and not df_j.empty and 'End NLV' in df_j.columns:
                    _jdf = df_j[['Day', 'End NLV']].copy()
                    _jdf['Day_DT'] = pd.to_datetime(_jdf['Day'], errors='coerce')
                    _jdf = _jdf.dropna(subset=['Day_DT']).sort_values('Day_DT')

                    def _nlv_at_open(open_dt):
                        if pd.isna(open_dt) or _jdf.empty:
                            return None
                        mask = _jdf['Day_DT'] <= open_dt
                        hist = _jdf.loc[mask]
                        if hist.empty:
                            return None
                        v = hist['End NLV'].iloc[-1]
                        return float(v) if v and v > 0 else None

                    loss_trades['NLV_at_Open'] = loss_trades['Open_Date_DT'].apply(_nlv_at_open)
                    loss_trades['Impact_Pct'] = loss_trades.apply(
                        lambda r: (r['Realized_PL'] / r['NLV_at_Open'] * 100)
                        if r['NLV_at_Open'] and r['NLV_at_Open'] > 0 else None,
                        axis=1
                    )
                    impact_df = loss_trades.dropna(subset=['Impact_Pct']).copy()

                if impact_df.empty:
                    st.info("💡 No 2026 closed losses with journal NLV data yet. Risk budget tracking began in 2026 — this view will populate as losses accumulate.")
                else:
                    total_losses = len(impact_df)
                    within_rule = (impact_df['Impact_Pct'] >= -1.0).sum()
                    breaches = total_losses - within_rule
                    pass_rate = (within_rule / total_losses * 100) if total_losses > 0 else 100.0

                    # Score card — big number
                    _score_bg = "#dcfce7" if pass_rate >= 95 else ("#fef3c7" if pass_rate >= 85 else "#fee2e2")
                    _score_txt = "#15803d" if pass_rate >= 95 else ("#b45309" if pass_rate >= 85 else "#b91c1c")
                    _score_icon = "✅" if pass_rate >= 95 else ("⚠️" if pass_rate >= 85 else "🚨")

                    st.markdown(
                        f'<div style="background:{_score_bg};border-radius:14px;'
                        f'padding:18px 22px;margin-bottom:12px;'
                        f'box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
                        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:16px;">'
                        f'<div>'
                        f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
                        f'letter-spacing:0.08em;color:#64748b;">1% Rule Compliance</div>'
                        f'<div style="font-size:30px;font-weight:800;color:{_score_txt};'
                        f'margin-top:4px;line-height:1.1;">'
                        f'{_score_icon} {pass_rate:.1f}% within rule</div>'
                        f'<div style="font-size:12px;color:#64748b;margin-top:3px;">'
                        f'{within_rule} of {total_losses} closed losses held under −1% account impact'
                        f'</div>'
                        f'</div>'
                        f'<div style="text-align:right;">'
                        f'<div style="font-size:11px;color:#64748b;text-transform:uppercase;'
                        f'font-weight:700;">Breaches</div>'
                        f'<div style="font-size:36px;font-weight:800;color:{("#b91c1c" if breaches > 0 else "#16a34a")};'
                        f'line-height:1;">{breaches}</div>'
                        f'</div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # Bucket definitions — (label, lower_bound_exclusive, upper_bound_inclusive, color, bg, subtitle)
                    buckets = [
                        ("0 to −0.25%", -0.25, 0.0, "#16a34a", "#f0fdf4", "Minor nicks"),
                        ("−0.25 to −0.50%", -0.50, -0.25, "#65a30d", "#f7fee7", "Small"),
                        ("−0.50 to −1.00%", -1.00, -0.50, "#d97706", "#fffbeb", "Borderline"),
                        ("Over −1.00%", -9999.0, -1.00, "#dc2626", "#fef2f2", "🚨 BREACH"),
                    ]

                    b_cols = st.columns(4)
                    for (label, lo, hi, color, bg, sub), col in zip(buckets, b_cols):
                        # (lo, hi] — bucket contains trades where lo < impact <= hi
                        mask = (impact_df['Impact_Pct'] > lo) & (impact_df['Impact_Pct'] <= hi)
                        bucket_trades = impact_df[mask]
                        count = len(bucket_trades)
                        dollar_sum = bucket_trades['Realized_PL'].sum() if count > 0 else 0.0
                        col.markdown(
                            f'<div style="background:{bg};border-left:4px solid {color};'
                            f'border-radius:10px;padding:14px 16px;height:100%;">'
                            f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;'
                            f'letter-spacing:0.06em;color:#64748b;">{label}</div>'
                            f'<div style="font-size:26px;font-weight:800;color:#111;'
                            f'margin-top:4px;">{count}</div>'
                            f'<div style="font-size:12px;color:{color};font-weight:600;margin-top:2px;">{sub}</div>'
                            f'<div style="font-size:11px;color:#64748b;margin-top:4px;">'
                            f'${dollar_sum:,.0f} total</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    # Worst offenders — expandable table
                    worst = impact_df.nsmallest(5, 'Impact_Pct')[
                        ['Trade_ID', 'Ticker', 'Closed_Date', 'Realized_PL', 'Impact_Pct']
                    ].copy()
                    worst['Closed_Date'] = pd.to_datetime(worst['Closed_Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    worst['Realized_PL'] = worst['Realized_PL'].apply(lambda x: f"${x:,.2f}")
                    worst['Impact_Pct'] = worst['Impact_Pct'].apply(lambda x: f"{x:.2f}%")
                    worst.columns = ['Trade ID', 'Ticker', 'Closed', 'P&L', 'Impact %']

                    with st.expander(f"⚠️ Worst {len(worst)} Offenders" + (f" — {breaches} breach(es)" if breaches > 0 else "")):
                        st.dataframe(worst, hide_index=True, use_container_width=True)

                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                # ==============================================================================
                # STREAKS & ACTIVITY — compact row
                # ==============================================================================
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Max Win Streak", f"{max_consecutive_wins}")
                s2.metric("Max Loss Streak", f"{max_consecutive_losses}")
                s3.metric("Avg Hold (all)", f"{avg_hold_all:.0f}d")
                s4.metric("Open Positions", f"{num_open}")
                s5.metric("Break-Even", f"{num_break_even}")

                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                # ==============================================================================
                # MONTHLY PERFORMANCE — bottom strip
                # ==============================================================================
                st.markdown("**📅 Monthly Performance**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Best Month", f"${best_month:,.0f}",
                         delta=best_month_date, delta_color="off")
                m2.metric("Worst Month", f"${worst_month:,.0f}",
                         delta=worst_month_date, delta_color="off")
                m3.metric("Average Month", f"${avg_month:,.0f}")

                if not has_risk_data:
                    st.caption("💡 Log trades with Risk_Budget to unlock R-multiple metrics.")

                # ==============================================================================
                # HOW TO READ — moved to the bottom
                # ==============================================================================
                with st.expander("📖 How to read these stats"):
                    st.markdown("""
                    **Hero row** — your 4 most important numbers. Green = healthy.

                    - **Total P&L**: closed-trade realized profit
                    - **Win Rate**: ≥40% is good for a trend-following system
                    - **Profit Factor**: gross profit ÷ gross loss. ≥1.5 = healthy, ≥2.0 = excellent
                    - **Expectancy**: average P&L per trade. Must be positive long-term

                    **Winners vs Losers** — symmetric breakdown. Look for:
                    - Avg win **bigger** than avg loss (confirms W/L ratio)
                    - Avg hold on winners **longer** than on losers (confirms you cut losses fast)

                    **Quality Indicators**
                    - **W/L Ratio ≥2.0x**: you make $2+ for every $1 lost per trade on average
                    - **Hold Ratio ≥1.0x**: you hold winners longer than losers (discipline)
                    - **Avg R-Multiple**: actual reward for each unit of risk taken

                    **Streaks**: max consecutive wins/losses show variance — use to gut-check position sizing.

                    **Monthly Performance**: gives context on seasonality and consistency.
                    """)
            else:
                st.info("No closed trades yet. Start trading to see your career stats!")

        # --- TAB 5: TRADE REVIEW ---
        # --- TAB 5: TRADE REVIEW (Top/Bottom N with lesson notes) ---
        with tab_review:
            st.subheader("🔬 Trade Review — Top Winners & Worst Losers")
            st.caption("Study your best and worst trades over a window. Tag each one with what you learned so the lessons compound over time.")

            # Master list of lesson categories — edit this list to tweak options
            LESSON_CATEGORIES = [
                "Entry timing",
                "Stop placement",
                "Undersized",
                "Oversized",
                "Scaled in too fast",
                "Exit too early",
                "Exit too late",
                "Market conditions",
                "Rule deviation",
                "Other",
            ]
            # Pipe delimiter for storing multiple categories in one column
            _CAT_SEP = "|"
            # Per-category (background, foreground) colors for the review pill
            LESSON_CAT_COLORS = {
                "Entry timing":        ("#fef3c7", "#b45309"),
                "Stop placement":      ("#fed7aa", "#c2410c"),
                "Undersized":          ("#dbeafe", "#1e40af"),
                "Oversized":           ("#ede9fe", "#6d28d9"),
                "Scaled in too fast":  ("#fecaca", "#b91c1c"),
                "Exit too early":      ("#ccfbf1", "#0f766e"),
                "Exit too late":       ("#e0e7ff", "#4338ca"),
                "Market conditions":   ("#e5e7eb", "#374151"),
                "Rule deviation":      ("#ffe4e6", "#be123c"),
                "Other":               ("#f1f5f9", "#475569"),
            }

            # Load notes for this portfolio
            lessons_map = {}
            if USE_DATABASE:
                try:
                    lessons_map = db.get_trade_lessons(CURR_PORT_NAME)
                except Exception:
                    lessons_map = {}

            # Load trade details once so every trade card can render its
            # transaction trail (buys + sells) without extra DB calls.
            tr_details = load_data(DETAILS_FILE)
            if not tr_details.empty:
                if 'Date' in tr_details.columns:
                    tr_details['Date'] = pd.to_datetime(tr_details['Date'], errors='coerce')
                if 'Trade_ID' in tr_details.columns:
                    tr_details['Trade_ID'] = tr_details['Trade_ID'].astype(str).str.strip()

            # --- Filter bar ---
            filt_col1, filt_col2, filt_col3 = st.columns([1.2, 1, 1])
            time_range = filt_col1.selectbox(
                "Time range",
                ["2026 YTD", "Last 30 days", "Last 90 days", "All time", "Custom"],
                index=0,
                key="tr_time_range",
            )
            top_n = filt_col2.selectbox("Show top/bottom", [5, 10, 15, 20], index=1, key="tr_top_n")

            # Apply time range filter
            tr_base = all_closed.copy()
            tr_base['Closed_Date'] = pd.to_datetime(tr_base['Closed_Date'], errors='coerce')
            tr_base['Open_Date_DT'] = pd.to_datetime(tr_base['Open_Date'], errors='coerce')
            now_ts = pd.Timestamp.now().normalize()

            if time_range == "2026 YTD":
                tr_filtered = tr_base[tr_base['Closed_Date'].dt.year == 2026]
            elif time_range == "Last 30 days":
                tr_filtered = tr_base[tr_base['Closed_Date'] >= now_ts - pd.Timedelta(days=30)]
            elif time_range == "Last 90 days":
                tr_filtered = tr_base[tr_base['Closed_Date'] >= now_ts - pd.Timedelta(days=90)]
            elif time_range == "All time":
                tr_filtered = tr_base
            else:  # Custom
                with filt_col3:
                    custom_start = st.date_input("From", value=pd.Timestamp("2026-01-01"), key="tr_custom_start")
                    custom_end = st.date_input("To", value=now_ts, key="tr_custom_end")
                tr_filtered = tr_base[
                    (tr_base['Closed_Date'] >= pd.Timestamp(custom_start)) &
                    (tr_base['Closed_Date'] <= pd.Timestamp(custom_end))
                ]

            if time_range != "Custom":
                with filt_col3:
                    st.metric("Closed trades in window", f"{len(tr_filtered):,}")

            if tr_filtered.empty:
                st.info("💡 No closed trades in the selected time range.")
            else:
                # Compute R-multiple and hold days
                tr_filtered['_R'] = tr_filtered.apply(
                    lambda r: (float(r['Realized_PL']) / float(r['Risk_Budget']))
                    if r.get('Risk_Budget') and float(r['Risk_Budget']) > 0 else None,
                    axis=1,
                )
                tr_filtered['_Hold'] = (
                    tr_filtered['Closed_Date'] - tr_filtered['Open_Date_DT']
                ).dt.total_seconds() / 86400

                # Helper to render a single trade card with an editable lesson note
                def _render_trade_card(rank, row, is_winner):
                    trade_id = str(row.get('Trade_ID', ''))
                    ticker = row.get('Ticker', 'N/A')
                    pl = float(row.get('Realized_PL', 0))
                    ret_pct = float(row.get('Return_Pct', 0)) if pd.notna(row.get('Return_Pct')) else 0.0
                    r_val = row.get('_R')
                    r_txt = f"{r_val:.2f}R" if pd.notna(r_val) else "—"
                    hold = row.get('_Hold', 0)
                    hold_txt = f"{hold:.0f}d" if pd.notna(hold) else "—"
                    open_str = row['Open_Date_DT'].strftime('%b %d') if pd.notna(row['Open_Date_DT']) else '—'
                    close_str = row['Closed_Date'].strftime('%b %d') if pd.notna(row['Closed_Date']) else '—'
                    buy_rule = row.get('Buy_Rule') or row.get('Rule') or '—'
                    sell_rule = row.get('Sell_Rule') or '—'

                    border_color = "#16a34a" if is_winner else "#dc2626"
                    bg_tint = "#f0fdf4" if is_winner else "#fef2f2"
                    pl_color = "#15803d" if is_winner else "#b91c1c"
                    pl_sign = "+" if pl >= 0 else ""

                    # Review pills: show one colored badge per saved category
                    # so reviewed trades are obvious at a glance. Supports
                    # multi-select (pipe-separated storage).
                    card_existing_note, card_existing_cat = lessons_map.get(trade_id, ('', ''))
                    card_cat_list = [
                        c.strip() for c in (card_existing_cat or '').split(_CAT_SEP) if c.strip()
                    ]
                    cat_pill_html = ""
                    for _cat in card_cat_list:
                        if _cat in LESSON_CAT_COLORS:
                            _pbg, _pfg = LESSON_CAT_COLORS[_cat]
                            cat_pill_html += (
                                f'<span style="display:inline-block;margin-left:6px;'
                                f'padding:3px 10px;background:{_pbg};color:{_pfg};'
                                f'font-size:11px;font-weight:700;border-radius:12px;'
                                f'vertical-align:middle;letter-spacing:0.02em;">'
                                f'✓ {_cat}</span>'
                            )

                    st.markdown(
                        f'<div style="background:#fff;border-left:4px solid {border_color};'
                        f'border:1px solid #e5e7eb;border-radius:10px;padding:14px 18px;margin-bottom:10px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
                        f'<div style="font-size:16px;font-weight:800;color:#111;">'
                        f'#{rank} · {ticker} '
                        f'<span style="color:#64748b;font-weight:500;font-size:12px;">({trade_id})</span>'
                        f'{cat_pill_html}'
                        f'</div>'
                        f'<div style="font-size:20px;font-weight:800;color:{pl_color};">{pl_sign}${pl:,.0f}</div>'
                        f'</div>'
                        f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:10px;">'
                        f'<div><div style="font-size:10px;color:#64748b;text-transform:uppercase;font-weight:600;">Return</div>'
                        f'<div style="font-size:14px;font-weight:700;color:{pl_color};">{pl_sign}{ret_pct:.1f}%</div></div>'
                        f'<div><div style="font-size:10px;color:#64748b;text-transform:uppercase;font-weight:600;">R-Multiple</div>'
                        f'<div style="font-size:14px;font-weight:700;color:#111;">{r_txt}</div></div>'
                        f'<div><div style="font-size:10px;color:#64748b;text-transform:uppercase;font-weight:600;">Held</div>'
                        f'<div style="font-size:14px;font-weight:700;color:#111;">{hold_txt}</div></div>'
                        f'<div><div style="font-size:10px;color:#64748b;text-transform:uppercase;font-weight:600;">Opened → Closed</div>'
                        f'<div style="font-size:14px;font-weight:700;color:#111;">{open_str} → {close_str}</div></div>'
                        f'<div><div style="font-size:10px;color:#64748b;text-transform:uppercase;font-weight:600;">Rules</div>'
                        f'<div style="font-size:11px;font-weight:600;color:#111;">B: {buy_rule}</div>'
                        f'<div style="font-size:11px;font-weight:600;color:#111;">S: {sell_rule}</div></div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # --- Transaction Trail (buys + sells inline) ---
                    trade_txs = pd.DataFrame()
                    if not tr_details.empty and 'Trade_ID' in tr_details.columns:
                        trade_txs = tr_details[tr_details['Trade_ID'] == trade_id].copy()

                    if not trade_txs.empty:
                        # Sort chronologically; preserve buy-before-sell on same day
                        trade_txs['_rank'] = trade_txs['Action'].map({'BUY': 0, 'SELL': 1}).fillna(2)
                        trade_txs = trade_txs.sort_values(['Date', '_rank'])

                        # Quick summary line above the table
                        _buys = trade_txs[trade_txs['Action'] == 'BUY']
                        _sells = trade_txs[trade_txs['Action'] == 'SELL']
                        total_bought_shs = float(_buys['Shares'].sum()) if not _buys.empty else 0.0
                        total_sold_shs = float(_sells['Shares'].sum()) if not _sells.empty else 0.0
                        avg_entry = (
                            float((_buys['Shares'] * _buys['Amount']).sum() / total_bought_shs)
                            if total_bought_shs > 0 else 0.0
                        )
                        avg_exit = (
                            float((_sells['Shares'] * _sells['Amount']).sum() / total_sold_shs)
                            if total_sold_shs > 0 else 0.0
                        )
                        n_buys = len(_buys)
                        n_sells = len(_sells)

                        with st.expander(f"📋 Transaction Trail — {n_buys} buy(s) · {n_sells} sell(s)", expanded=False):
                            st.markdown(
                                f'<div style="font-size:12px;color:#64748b;margin-bottom:8px;">'
                                f'Bought <b style="color:#15803d;">{total_bought_shs:,.0f} shs</b> '
                                f'@ avg <b>${avg_entry:,.2f}</b> · '
                                f'Sold <b style="color:#b91c1c;">{total_sold_shs:,.0f} shs</b> '
                                f'@ avg <b>${avg_exit:,.2f}</b>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            # LIFO back-attribution — same logic as Trade Journal.
                            # Each BUY row ends up with the realized P&L of the shares
                            # sold against it; the Return % is the LIFO-attributed
                            # return on that specific lot. SELL rows show 0% since
                            # the return is credited to the BUY lot it consumed.
                            inventory = []  # list of [row_idx, remaining_shares, entry_price, original_shares]
                            buy_realized = {}  # row_idx -> realized $
                            for idx, trow in trade_txs.iterrows():
                                action = trow['Action']
                                shs = float(trow['Shares'])
                                px = float(trow['Amount'])
                                if action == 'BUY':
                                    inventory.append([idx, shs, px, shs])
                                    buy_realized.setdefault(idx, 0.0)
                                elif action == 'SELL':
                                    to_sell = shs
                                    while to_sell > 0 and inventory:
                                        lot = inventory[-1]
                                        take = min(to_sell, lot[1])
                                        pl = take * (px - lot[2])
                                        buy_realized[lot[0]] = buy_realized.get(lot[0], 0.0) + pl
                                        to_sell -= take
                                        lot[1] -= take
                                        if lot[1] < 0.0001:
                                            inventory.pop()

                            # Per-row return %
                            def _row_return_pct(idx, trow):
                                if trow['Action'] != 'BUY':
                                    return 0.0
                                entry = float(trow['Amount'])
                                orig_shs = float(trow['Shares'])
                                cost_basis = entry * orig_shs
                                if cost_basis <= 0:
                                    return 0.0
                                return (buy_realized.get(idx, 0.0) / cost_basis) * 100

                            ret_series = pd.Series(
                                [_row_return_pct(i, r) for i, r in trade_txs.iterrows()],
                                index=trade_txs.index,
                            )

                            # Sells display as negative shares to match the Trade Journal style
                            sign_shares = trade_txs.apply(
                                lambda r: -float(r['Shares']) if r['Action'] == 'SELL' else float(r['Shares']),
                                axis=1,
                            )

                            disp = pd.DataFrame({
                                'Date': trade_txs['Date'].dt.strftime('%Y-%m-%d %H:%M')
                                        if 'Date' in trade_txs.columns else '',
                                'Trx': trade_txs.get('Trx_ID', ''),
                                'Action': trade_txs['Action'],
                                'Shares': sign_shares.astype(float),
                                'Price': trade_txs['Amount'].astype(float),
                                'Return %': ret_series.astype(float),
                                'Value': (trade_txs['Shares'].astype(float) * trade_txs['Amount'].astype(float)),
                                'Rule': trade_txs.get('Rule', ''),
                            })
                            st.dataframe(
                                disp,
                                hide_index=True,
                                use_container_width=True,
                                column_config={
                                    'Date': st.column_config.TextColumn('Date', width='small'),
                                    'Trx': st.column_config.TextColumn('Trx', width='small'),
                                    'Action': st.column_config.TextColumn('Action', width='small'),
                                    'Shares': st.column_config.NumberColumn('Shares', format='%.0f'),
                                    'Price': st.column_config.NumberColumn('Price', format='$%.4f'),
                                    'Return %': st.column_config.NumberColumn('Return %', format='%+.2f%%', help='LIFO-attributed return for each BUY lot (SELL rows show 0%)'),
                                    'Value': st.column_config.NumberColumn('Value', format='$%.2f'),
                                    'Rule': st.column_config.TextColumn('Rule'),
                                },
                            )

                    # Editable lesson note
                    existing_note, existing_cat = lessons_map.get(trade_id, ('', ''))
                    existing_cat_list = [
                        c.strip() for c in (existing_cat or '').split(_CAT_SEP) if c.strip()
                    ]
                    # Filter to valid categories (drops any stale tags)
                    default_cats = [c for c in existing_cat_list if c in LESSON_CATEGORIES]
                    with st.expander(f"📝 Lesson — {ticker} {trade_id}", expanded=False):
                        cat_vals = st.multiselect(
                            "Category (pick one or more)",
                            LESSON_CATEGORIES,
                            default=default_cats,
                            key=f"cat_{trade_id}",
                        )
                        note_val = st.text_area(
                            "What did you learn from this trade? What would you do differently?",
                            value=existing_note,
                            key=f"lesson_{trade_id}",
                            height=90,
                            placeholder="e.g. Scaled in too fast on the third add, ended up averaging up into a failed breakout. Next time, wait for the first add to work before pyramiding.",
                        )
                        if st.button("Save lesson", key=f"savelsn_{trade_id}"):
                            if USE_DATABASE:
                                save_cat = _CAT_SEP.join(cat_vals) if cat_vals else ''
                                ok = db.save_trade_lesson(CURR_PORT_NAME, trade_id, note_val, save_cat)
                                if ok:
                                    st.toast(f"Lesson saved for {ticker} ✅")
                                else:
                                    st.error("Save failed")
                            else:
                                st.warning("Database unavailable — lessons require DB mode.")

                # --- Top Winners ---
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                st.markdown(f"### 🏆 Top {top_n} Winners")
                top_winners = tr_filtered[tr_filtered['Realized_PL'] > 0].nlargest(top_n, 'Realized_PL')
                if top_winners.empty:
                    st.info("No profitable closed trades in this window.")
                else:
                    for rank, (_, row) in enumerate(top_winners.iterrows(), start=1):
                        _render_trade_card(rank, row, is_winner=True)

                # --- Worst Losers ---
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                st.markdown(f"### ⚠️ Worst {top_n} Losers")
                worst_losers = tr_filtered[tr_filtered['Realized_PL'] < 0].nsmallest(top_n, 'Realized_PL')
                if worst_losers.empty:
                    st.success("No losing closed trades in this window. Nice.")
                else:
                    for rank, (_, row) in enumerate(worst_losers.iterrows(), start=1):
                        _render_trade_card(rank, row, is_winner=False)

                # --- Pattern snapshot ---
                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
                st.markdown("### 📊 Pattern Snapshot")
                pat_col1, pat_col2 = st.columns(2)

                def _pattern_block(title, group_df, color_bg, color_accent):
                    if group_df.empty:
                        return f'<div style="background:{color_bg};padding:14px 16px;border-radius:10px;"><em>No trades</em></div>'
                    avg_pl = group_df['Realized_PL'].mean()
                    avg_hold = group_df['_Hold'].mean() if '_Hold' in group_df else 0
                    r_vals = group_df['_R'].dropna() if '_R' in group_df else pd.Series(dtype=float)
                    avg_r = r_vals.mean() if not r_vals.empty else None
                    # Most common buy rule
                    buy_col = 'Buy_Rule' if 'Buy_Rule' in group_df.columns and group_df['Buy_Rule'].notna().any() else 'Rule'
                    top_buy = 'n/a'
                    if buy_col in group_df.columns:
                        vc = group_df[buy_col].dropna().astype(str).value_counts()
                        if not vc.empty:
                            top_buy = vc.index[0]
                    top_sell = 'n/a'
                    if 'Sell_Rule' in group_df.columns:
                        vc = group_df['Sell_Rule'].dropna().astype(str).value_counts()
                        if not vc.empty:
                            top_sell = vc.index[0]
                    r_txt = f"{avg_r:.2f}R" if avg_r is not None else "—"
                    return (
                        f'<div style="background:{color_bg};padding:16px 18px;border-radius:10px;'
                        f'border-left:3px solid {color_accent};">'
                        f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
                        f'color:{color_accent};margin-bottom:8px;">{title}</div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;">'
                        f'<div><span style="color:#64748b;">Avg P&L:</span> <b>${avg_pl:,.0f}</b></div>'
                        f'<div><span style="color:#64748b;">Avg Hold:</span> <b>{avg_hold:.0f}d</b></div>'
                        f'<div><span style="color:#64748b;">Avg R:</span> <b>{r_txt}</b></div>'
                        f'<div><span style="color:#64748b;">Top Buy Rule:</span> <b>{top_buy}</b></div>'
                        f'<div style="grid-column:1/-1;"><span style="color:#64748b;">Top Sell Rule:</span> <b>{top_sell}</b></div>'
                        f'</div>'
                        f'</div>'
                    )

                with pat_col1:
                    st.markdown(
                        _pattern_block(f"🏆 Top {top_n} Winners Pattern", top_winners, "#f0fdf4", "#16a34a"),
                        unsafe_allow_html=True,
                    )
                with pat_col2:
                    st.markdown(
                        _pattern_block(f"⚠️ Worst {top_n} Losers Pattern", worst_losers, "#fef2f2", "#dc2626"),
                        unsafe_allow_html=True,
                    )

# ==============================================================================
# PAGE 12: DAILY REPORT CARD (FIXED MARKET DATA)
# ==============================================================================
elif page == "Daily Report Card":
    page_header("Daily Report Card", CURR_PORT_NAME, "📊")

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

            # --- COMPUTE METRICS ---
            nlv = day_stats['End NLV']
            day_dol = day_stats['Daily $ Change']
            prev_adj = day_stats['Beg NLV'] + day_stats['Cash -/+']
            day_pct = (day_dol / prev_adj * 100) if prev_adj != 0 else 0.0

            # --- SPY / NASDAQ: DAILY + YTD (from journal data, same as Dashboard) ---
            spy_chg_str = "N/A"
            ndx_chg_str = "N/A"
            spy_ytd_str = "N/A"
            ndx_ytd_str = "N/A"

            sel_year = selected_date.year

            # Daily change: use journal's SPY/Nasdaq columns (already cleaned as numbers)
            # These store closing prices; compute daily % from previous row
            df_j_sorted = df_j.sort_values('Day')
            sel_idx = df_j_sorted[df_j_sorted['Day'].dt.date == selected_date].index
            if len(sel_idx) > 0 and 'SPY' in df_j_sorted.columns:
                row_pos = df_j_sorted.index.get_loc(sel_idx[0])
                if row_pos > 0:
                    spy_today = df_j_sorted.iloc[row_pos]['SPY']
                    spy_prev = df_j_sorted.iloc[row_pos - 1]['SPY']
                    ndx_today = df_j_sorted.iloc[row_pos]['Nasdaq']
                    ndx_prev = df_j_sorted.iloc[row_pos - 1]['Nasdaq']
                    if spy_prev > 0:
                        spy_chg_str = f"{((spy_today - spy_prev) / spy_prev * 100):+.2f}%"
                    if ndx_prev > 0:
                        ndx_chg_str = f"{((ndx_today - ndx_prev) / ndx_prev * 100):+.2f}%"

            # YTD: same method as Dashboard — prior year's last close vs selected date's close
            if 'SPY' in df_j.columns:
                prior_year = df_j_sorted[(df_j_sorted['Day'].dt.year < sel_year) & (df_j_sorted['SPY'] > 0)]
                sel_row_spy = df_j_sorted[(df_j_sorted['Day'].dt.date <= selected_date) & (df_j_sorted['SPY'] > 0)]
                if not prior_year.empty and not sel_row_spy.empty:
                    spy_base = prior_year.iloc[-1]['SPY']
                    spy_curr = sel_row_spy.iloc[-1]['SPY']
                    if spy_base > 0:
                        spy_ytd_pct = ((spy_curr - spy_base) / spy_base) * 100
                        spy_ytd_str = f"{spy_ytd_pct:+.2f}%"

            if 'Nasdaq' in df_j.columns:
                prior_year_ndx = df_j_sorted[(df_j_sorted['Day'].dt.year < sel_year) & (df_j_sorted['Nasdaq'] > 0)]
                sel_row_ndx = df_j_sorted[(df_j_sorted['Day'].dt.date <= selected_date) & (df_j_sorted['Nasdaq'] > 0)]
                if not prior_year_ndx.empty and not sel_row_ndx.empty:
                    ndx_base = prior_year_ndx.iloc[-1]['Nasdaq']
                    ndx_curr = sel_row_ndx.iloc[-1]['Nasdaq']
                    if ndx_base > 0:
                        ndx_ytd_pct = ((ndx_curr - ndx_base) / ndx_base) * 100
                        ndx_ytd_str = f"{ndx_ytd_pct:+.2f}%"

            # --- PORTFOLIO YTD (TWR - matches Dashboard) ---
            port_ytd_pct = 0.0
            port_ytd_str = "N/A"
            jan1_ts = pd.Timestamp(f"{sel_year}-01-01")
            ytd_journal = df_j[(df_j['Day'] >= jan1_ts) & (df_j['Day'].dt.date <= selected_date)].sort_values('Day')
            if not ytd_journal.empty:
                ytd_j = ytd_journal.copy()
                ytd_j['Adjusted_Beg'] = ytd_j['Beg NLV'] + ytd_j['Cash -/+']
                mask = ytd_j['Adjusted_Beg'] != 0
                ytd_j['Daily_Pct'] = 0.0
                ytd_j.loc[mask, 'Daily_Pct'] = (ytd_j.loc[mask, 'End NLV'] - ytd_j.loc[mask, 'Adjusted_Beg']) / ytd_j.loc[mask, 'Adjusted_Beg']
                port_ytd_pct = ((1 + ytd_j['Daily_Pct']).prod() - 1) * 100
                port_ytd_str = f"{port_ytd_pct:+.2f}%"

            # Market Window from journal
            market_window = str(day_stats.get('Market Window', '')).strip()
            if not market_window or market_window == 'nan':
                market_window = "N/A"

            # Risk / Drawdown
            RESET_DATE = pd.Timestamp("2026-02-24")
            hist_slice = df_j[df_j['Day'] <= pd.Timestamp(selected_date)].sort_values('Day')
            hist_slice_post = hist_slice[hist_slice['Day'] >= RESET_DATE]

            risk_msg = "NO DATA"
            risk_color = "gray"
            dd_pct = 0.0

            if not hist_slice_post.empty:
                curr_nlv = hist_slice_post['End NLV'].iloc[-1]
                peak_nlv = hist_slice_post['End NLV'].max()
                dd_pct = ((curr_nlv - peak_nlv) / peak_nlv) * 100 if peak_nlv > 0 else 0.0

                # Hard decks aligned with Risk Manager: 7.5%, 12.5%, 15%
                if dd_pct >= -7.5:
                    risk_msg = "GREEN LIGHT"
                    risk_color = "#2ca02c"
                elif -7.5 > dd_pct >= -12.5:
                    risk_msg = "CAUTION - Remove Margin"
                    risk_color = "#ff8c00"
                elif -12.5 > dd_pct >= -15:
                    risk_msg = "MAX 30% INVESTED"
                    risk_color = "#ff4b4b"
                else:
                    risk_msg = "GO TO CASH"
                    risk_color = "#8B0000"

            # --- PREP TRADE DATA ---
            bought_today = pd.DataFrame()
            sold_today_details = pd.DataFrame()
            if not df_d.empty:
                df_d['Date_Obj'] = pd.to_datetime(df_d['Date'], errors='coerce')
                bought_today = df_d[
                    (df_d['Action'] == 'BUY') &
                    (df_d['Date_Obj'].dt.date == selected_date)
                ]
                sold_today_details = df_d[
                    (df_d['Action'] == 'SELL') &
                    (df_d['Date_Obj'].dt.date == selected_date)
                ]

            sold_today = pd.DataFrame()
            if not df_s.empty:
                df_s['Closed_Date'] = pd.to_datetime(df_s['Closed_Date'], errors='coerce')
                sold_today = df_s[
                    (df_s['Status'] == 'CLOSED') &
                    (df_s['Closed_Date'].dt.date == selected_date)
                ]

            # =====================================================
            # SECTION 1: HEADER METRICS ROW
            # =====================================================
            st.markdown(f"#### {selected_date.strftime('%A, %B %d, %Y')}")

            # Market Window color badge
            mw_colors = {
                'POWERTREND': ('#8A2BE2', 'white'),
                'OPEN': ('#2ca02c', 'white'),
                'NEUTRAL': ('#ffcc00', 'black'),
                'CLOSED': ('#ff4b4b', 'white'),
            }
            mw_bg, mw_fg = mw_colors.get(market_window.upper(), ('#888', 'white'))

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Net Liquidity", f"${nlv:,.2f}")
            pnl_delta = f"${day_dol:+,.2f} ({day_pct:+.2f}%)"
            mc2.metric("Daily P&L", f"${day_dol:+,.2f}", delta=f"{day_pct:+.2f}%")
            mc3.markdown(
                f"**Market Window**<br>"
                f"<span style='background-color:{mw_bg}; color:{mw_fg}; padding:4px 16px; border-radius:6px; font-weight:bold; font-size:1.1em;'>"
                f"{market_window}</span>",
                unsafe_allow_html=True
            )
            mc4.markdown(
                f"**Risk Status**<br>"
                f"<span style='background-color:{risk_color}; color:white; padding:4px 16px; border-radius:6px; font-weight:bold; font-size:1.1em;'>"
                f"{risk_msg}</span>",
                unsafe_allow_html=True
            )

            st.divider()

            # =====================================================
            # SECTION 2: MARKET & PERFORMANCE
            # =====================================================
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("##### Performance Comparison")
                perf_data = pd.DataFrame({
                    '': ['Portfolio', 'SPY', 'NASDAQ'],
                    'Daily': [f"{day_pct:+.2f}%", spy_chg_str, ndx_chg_str],
                    'YTD': [port_ytd_str, spy_ytd_str, ndx_ytd_str],
                })
                st.dataframe(perf_data, hide_index=True, use_container_width=True)

                # Drawdown info
                st.markdown(f"**Drawdown:** {dd_pct:.2f}% from peak")
                pct_invested = day_stats.get('% Invested', 0)
                if pct_invested:
                    st.markdown(f"**Invested:** {clean_num_local(pct_invested):.0f}%")

            with col_right:
                st.markdown("##### Market Notes")
                market_notes = str(day_stats.get('Market_Notes', '') or '')
                if market_notes and market_notes != 'nan':
                    st.info(market_notes)
                else:
                    st.caption("No market notes logged.")

                market_action = str(day_stats.get('Market_Action', '') or '')
                if market_action and market_action != 'nan':
                    st.markdown(f"**Actions:** {market_action}")

            st.divider()

            # =====================================================
            # SECTION 3: TRADE ACTIVITY (INTERACTIVE)
            # =====================================================
            st.markdown("##### Trade Activity")

            tc1, tc2 = st.columns(2)

            with tc1:
                st.markdown("**Positions Opened**")
                if not bought_today.empty:
                    for ridx, (_, row) in enumerate(bought_today.iterrows()):
                        ticker = str(row['Ticker'])
                        shares = int(row['Shares'])
                        price = row['Amount']
                        value = row['Value']
                        rule = str(row.get('Rule', ''))

                        bcol1, bcol2 = st.columns([1, 3])
                        with bcol1:
                            if st.button(f"🔍 {ticker}", key=f"drc_buy_{ridx}_{ticker}"):
                                st.session_state['tj_ticker_search'] = [ticker]
                                st.session_state['journal_searched'] = True
                                st.session_state.page = "Trade Journal"
                                st.rerun()
                        with bcol2:
                            st.caption(f"{shares} shares @ ${price:.2f} = ${value:,.2f} | {rule}")
                else:
                    st.caption("No new positions opened.")

            with tc2:
                st.markdown("**Positions Closed**")
                if not sold_today.empty:
                    for ridx, (_, row) in enumerate(sold_today.iterrows()):
                        ticker = str(row['Ticker'])
                        realized = row['Realized_PL']
                        ret_pct = row['Return_Pct']
                        sell_rule = str(row.get('Sell_Rule', ''))
                        pl_color = "green" if realized >= 0 else "red"

                        scol1, scol2 = st.columns([1, 3])
                        with scol1:
                            if st.button(f"🔍 {ticker}", key=f"drc_sell_{ridx}_{ticker}"):
                                st.session_state['tj_ticker_search'] = [ticker]
                                st.session_state['journal_searched'] = True
                                st.session_state.page = "Trade Journal"
                                st.rerun()
                        with scol2:
                            st.caption(f"P&L: ${realized:+,.2f} ({ret_pct:+.2f}%) | {sell_rule}")
                elif not sold_today_details.empty:
                    # Show sell transactions even if campaign not fully closed
                    for ridx, (_, row) in enumerate(sold_today_details.iterrows()):
                        ticker = str(row['Ticker'])
                        shares = int(row['Shares'])
                        price = row['Amount']

                        scol1, scol2 = st.columns([1, 3])
                        with scol1:
                            if st.button(f"🔍 {ticker}", key=f"drc_sell_d_{ridx}_{ticker}"):
                                st.session_state['tj_ticker_search'] = [ticker]
                                st.session_state['journal_searched'] = True
                                st.session_state.page = "Trade Journal"
                                st.rerun()
                        with scol2:
                            st.caption(f"Sold {shares} shares @ ${price:.2f}")
                else:
                    st.caption("No positions closed.")

            st.divider()

            # =====================================================
            # SECTION 4: JOURNAL REVIEW
            # =====================================================
            score = int(day_stats.get('Score', 0) or 0)
            highlights = str(day_stats.get('Highlights', '') or '')
            lowlights = str(day_stats.get('Lowlights', '') or '')
            mistakes = str(day_stats.get('Mistakes', '') or '')
            top_lesson = str(day_stats.get('Top_Lesson', '') or '')

            has_review = score > 0 or any(v and v != 'nan' for v in [highlights, lowlights, mistakes, top_lesson])

            if has_review:
                st.markdown("##### Daily Review")

                # Score badge
                if score > 0:
                    score_colors = {5: '#008000', 4: '#90EE90', 3: '#FFFFE0', 2: '#FFD700', 1: '#FF4B4B'}
                    score_fg = {5: 'white', 4: 'black', 3: 'black', 2: 'black', 1: 'white'}
                    s_bg = score_colors.get(score, '#888')
                    s_fg = score_fg.get(score, 'white')
                    st.markdown(
                        f"**Process Score:** "
                        f"<span style='background-color:{s_bg}; color:{s_fg}; padding:3px 12px; border-radius:4px; font-weight:bold;'>"
                        f"{score}/5</span>",
                        unsafe_allow_html=True
                    )

                rv1, rv2 = st.columns(2)
                with rv1:
                    if highlights and highlights != 'nan':
                        st.markdown(f"**Highlights:** {highlights}")
                    if top_lesson and top_lesson != 'nan':
                        st.markdown(f"**Top Lesson:** {top_lesson}")
                with rv2:
                    if lowlights and lowlights != 'nan':
                        st.markdown(f"**Lowlights:** {lowlights}")
                    if mistakes and mistakes != 'nan':
                        st.markdown(f"**Mistakes:** {mistakes}")

                st.divider()

            # =====================================================
            # SECTION 5: MARKDOWN EXPORT (collapsed)
            # =====================================================
            with st.expander("📋 Export Report (Markdown)"):
                # Generate markdown for copy
                report = f"""# DAILY TRADING RECORD
**Date:** {selected_date.strftime('%A, %B %d, %Y')}
**Account:** {CURR_PORT_NAME}
**Net Liquidity:** ${nlv:,.2f}
**Market Window:** {market_window}

---

### Performance
| | Daily | YTD |
| :--- | :--- | :--- |
| **Portfolio** | {day_pct:+.2f}% | {port_ytd_str} |
| **SPY** | {spy_chg_str} | {spy_ytd_str} |
| **NASDAQ** | {ndx_chg_str} | {ndx_ytd_str} |

| Metric | Value |
| :--- | :--- |
| **Daily P&L** | ${day_dol:+,.2f} |
| **Drawdown** | {dd_pct:.2f}% |
| **Risk** | {risk_msg} |
| **Notes** | {day_stats.get('Market_Notes', '')} |

### Trades Opened
"""
                if not bought_today.empty:
                    report += "| Ticker | Shares | Price | Value | Strategy |\n| :--- | :--- | :--- | :--- | :--- |\n"
                    for _, row in bought_today.iterrows():
                        report += f"| {row['Ticker']} | {int(row['Shares'])} | ${row['Amount']:.2f} | ${row['Value']:,.2f} | {row.get('Rule', '')} |\n"
                else:
                    report += "*None*\n"

                report += "\n### Trades Closed\n"
                if not sold_today.empty:
                    report += "| Ticker | P&L | Return | Reason |\n| :--- | :--- | :--- | :--- |\n"
                    for _, row in sold_today.iterrows():
                        report += f"| {row['Ticker']} | ${row['Realized_PL']:,.2f} | {row['Return_Pct']:.2f}% | {row.get('Sell_Rule', '')} |\n"
                else:
                    report += "*None*\n"

                if has_review:
                    report += f"\n### Review (Score: {score}/5)\n"
                    if highlights and highlights != 'nan': report += f"* **Highlights:** {highlights}\n"
                    if lowlights and lowlights != 'nan': report += f"* **Lowlights:** {lowlights}\n"
                    if mistakes and mistakes != 'nan': report += f"* **Mistakes:** {mistakes}\n"
                    if top_lesson and top_lesson != 'nan': report += f"* **Top Lesson:** {top_lesson}\n"

                st.text_area("Raw Text", report, height=300)

        else:
            st.info("No journal entries found.")
    else:
        st.info("No journal data available. Please log your first trading day.")

# ==============================================================================
# PAGE 12: WEEKLY RETRO (OPTIMIZED WORKFLOW)
# ==============================================================================
elif page == "Weekly Retro":
    page_header("Weekly Retro", CURR_PORT_NAME, "🔄")
    
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
    page_header("IBD Market School", "Market timing signals · NASDAQ buy/sell · recommended exposure", "🏫")

    # Import market_school_rules
    try:
        from market_school_rules import MarketSchoolRules
    except ImportError:
        st.error("market_school_rules.py not found. Please ensure file is in project root.")
        st.stop()

    from datetime import datetime, timedelta
    import time

    # === HELPER FUNCTIONS ===

    def _fetch_with_qqq_volume(analyzer, start_date, end_date):
        """
        Fetch price data for analyzer.symbol, but replace Volume with QQQ volume
        (for ^IXIC / ^NDX only). yfinance misreports NASDAQ composite volume —
        QQQ ETF volume is the accurate proxy. Falls back to normal fetch if
        symbol is not a NASDAQ index or if the QQQ fetch fails.
        """
        analyzer.fetch_data(start_date=start_date, end_date=end_date)
        if analyzer.data is None or analyzer.data.empty:
            return
        if analyzer.symbol not in ('^IXIC', '^NDX'):
            return
        try:
            qqq = yf.Ticker('QQQ').history(start=start_date, end=end_date)
            if qqq is None or qqq.empty:
                return
            # Align QQQ volume to analyzer.data by date index
            analyzer.data['Volume'] = qqq['Volume'].reindex(analyzer.data.index).ffill()
            # Recompute volume-derived columns
            analyzer.data['volume_vs_prev'] = analyzer.data['Volume'] / analyzer.data['Volume'].shift(1)
            analyzer.data['volume_up'] = analyzer.data['Volume'] > analyzer.data['Volume'].shift(1)
        except Exception as e:
            print(f"QQQ volume proxy fetch failed: {e}")

    @st.cache_data(ttl=3600, show_spinner=False)
    def analyze_symbol(symbol, start_date, end_date):
        """Analyze market signals for a symbol. Returns list of daily summaries."""
        try:
            analyzer = MarketSchoolRules(symbol)
            _fetch_with_qqq_volume(analyzer, start_date, end_date)

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
            _fetch_with_qqq_volume(analyzer, fetch_start, end_date)

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

    def get_correction_state(symbol):
        """Get current correction/rally/FTD state from a fresh analysis."""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            fetch_start = "2024-02-24"

            analyzer = MarketSchoolRules(symbol)
            _fetch_with_qqq_volume(analyzer, fetch_start, end_date)

            if analyzer.data is None or analyzer.data.empty:
                return None

            analyzer.analyze_market()

            last_idx = len(analyzer.data) - 1
            last_date = analyzer.data.index[-1]

            # Rally day type classification (same logic as Market Cycle Tracker)
            # - "rally": Close > previous day's close
            # - "pink": Close < previous day's close but in upper half of range
            # - None on low day: check subsequent days for close > prior close
            rally_day_type = None
            rally_day_idx = analyzer.rally_low_idx
            actual_rally_date = analyzer.rally_start_date
            df = analyzer.data
            if analyzer.rally_start_date is not None and analyzer.rally_low_idx is not None:
                rd_row = df.iloc[analyzer.rally_low_idx]
                if analyzer.rally_low_idx > 0:
                    prev_row = df.iloc[analyzer.rally_low_idx - 1]
                    if rd_row['Close'] > prev_row['Close']:
                        rally_day_type = "rally"
                    else:
                        day_midpoint = (rd_row['High'] + rd_row['Low']) / 2
                        if rd_row['Close'] >= day_midpoint:
                            rally_day_type = "pink"
                        else:
                            # Check subsequent days
                            for next_i in range(analyzer.rally_low_idx + 1, len(df)):
                                next_row = df.iloc[next_i]
                                next_prev = df.iloc[next_i - 1]
                                if next_row['Close'] > next_prev['Close']:
                                    rally_day_type = "rally"
                                    rally_day_idx = next_i
                                    actual_rally_date = df.index[next_i]
                                    break
                                next_mid = (next_row['High'] + next_row['Low']) / 2
                                if next_row['Close'] >= next_mid:
                                    rally_day_type = "pink"
                                    rally_day_idx = next_i
                                    actual_rally_date = df.index[next_i]
                                    break

            # Rally day number: Day 1 = rally day itself (IBD numbering)
            rally_day = None
            if rally_day_idx is not None:
                rally_day = last_idx - rally_day_idx + 1

            return {
                'market_in_correction': analyzer.market_in_correction,
                'buy_switch': analyzer.buy_switch,
                'rally_start_date': actual_rally_date.strftime('%Y-%m-%d') if actual_rally_date else None,
                'rally_low': analyzer.rally_low,
                'rally_day': rally_day,
                'rally_day_type': rally_day_type,
                'ftd_date': analyzer.ftd_date.strftime('%Y-%m-%d') if analyzer.ftd_date else None,
                'reference_high': analyzer.reference_high,
                'as_of': last_date.strftime('%Y-%m-%d'),
            }
        except Exception:
            return None

    def calculate_dd_changes(symbol, start_date, end_date):
        """Calculate daily distribution day additions, removals, and notes."""
        try:
            analyzer = MarketSchoolRules(symbol)
            _fetch_with_qqq_volume(analyzer, "2024-02-24", end_date)

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

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 1, 2])

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

                # Always fetch full history (need 260-day lookback for indicators)
                # Only save new dates after what's already in DB
                nasdaq_fetch_start = "2024-02-24"

                if nasdaq_latest_date is None:
                    nasdaq_save_from = pd.Timestamp("2025-02-24")
                    st.info("📥 Initial Nasdaq sync from Feb 24, 2025")
                else:
                    nasdaq_save_from = pd.Timestamp(nasdaq_latest_date) + timedelta(days=1)
                    st.info(f"🔄 Updating Nasdaq from {nasdaq_save_from.date()}")

                # Nasdaq sync
                with st.spinner("📊 Analyzing Nasdaq..."):
                    nasdaq_summaries = analyze_symbol("^IXIC", nasdaq_fetch_start, end_date)

                nasdaq_saved = sync_signals_to_db("^IXIC", nasdaq_summaries, filter_from_date=nasdaq_save_from)

                st.success(f"🎉 Sync complete! Saved {nasdaq_saved} Nasdaq records")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Sync failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    with col_btn3:
        if st.button("♻️ Force Resync (14d)", help="Re-analyze and overwrite the last 14 days in the DB — use after logic changes") and USE_DATABASE:
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                nasdaq_fetch_start = "2024-02-24"
                force_from = pd.Timestamp.now().normalize() - timedelta(days=14)
                st.info(f"♻️ Force resyncing Nasdaq from {force_from.date()}")
                st.cache_data.clear()
                with st.spinner("📊 Re-analyzing Nasdaq..."):
                    nasdaq_summaries = analyze_symbol("^IXIC", nasdaq_fetch_start, end_date)
                nasdaq_saved = sync_signals_to_db("^IXIC", nasdaq_summaries, filter_from_date=force_from)
                st.success(f"🎉 Force resync complete! Overwrote {nasdaq_saved} Nasdaq records")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Force resync failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")

    # === AUTO-SYNC: Check if data is stale and sync if needed ===
    if USE_DATABASE:
        nasdaq_latest_date = db.get_latest_signal_date("^IXIC")
        today_date = pd.Timestamp.now().normalize().date()

        # Check if Nasdaq is missing recent data
        # Only auto-sync on weekdays and if data is at least 1 business day stale
        needs_sync = False
        if nasdaq_latest_date is None:
            needs_sync = True
        else:
            nasdaq_dt = nasdaq_latest_date.date() if hasattr(nasdaq_latest_date, 'date') else nasdaq_latest_date
            # Calculate the most recent expected trading day
            check_date = today_date
            # If weekend, step back to Friday
            while check_date.weekday() >= 5:  # 5=Sat, 6=Sun
                check_date -= timedelta(days=1)
            if nasdaq_dt < check_date:
                needs_sync = True

        if needs_sync:
            # Clear cache first so analyze_symbol fetches fresh data
            st.cache_data.clear()
            with st.spinner("📡 Auto-syncing Nasdaq data..."):
                end_date = datetime.now().strftime('%Y-%m-%d')
                # Always fetch full history (need 260-day lookback for indicators)
                fetch_start = "2024-02-24"
                if nasdaq_latest_date is None:
                    save_from = pd.Timestamp("2025-02-24")
                else:
                    save_from = pd.Timestamp(nasdaq_latest_date) + timedelta(days=1)

                summaries = analyze_symbol("^IXIC", fetch_start, end_date)
                sync_signals_to_db("^IXIC", summaries, filter_from_date=save_from)

            st.cache_data.clear()

    # === LOAD DATA ===

    if USE_DATABASE:
        df_signals = db.load_market_signals(days=90)

        if df_signals.empty:
            st.warning("📭 No market signals in database. Click 'Sync to Database' to populate.")
            st.stop()

        nasdaq_latest = df_signals[df_signals['symbol'] == '^IXIC'].iloc[0] if not df_signals[df_signals['symbol'] == '^IXIC'].empty else None
    else:
        # On-the-fly analysis (no database)
        end_date = datetime.now().strftime('%Y-%m-%d')
        fetch_start = "2024-02-24"  # 1 year for 260-day lookback

        nasdaq_summaries = analyze_symbol("^IXIC", fetch_start, end_date)
        nasdaq_latest = nasdaq_summaries[-1] if nasdaq_summaries else None

    # === CURRENT STATUS DISPLAY ===

    st.subheader("📊 Current Market Status")

    st.markdown("### 🟦 NASDAQ (^IXIC)")

    if nasdaq_latest is not None:
        def _clean_sigs(v):
            """Normalize signal column — treat NaN/None/empty as no signal."""
            if v is None:
                return None
            try:
                if pd.isna(v):
                    return None
            except (TypeError, ValueError):
                pass
            s = str(v).strip()
            return s if s and s.lower() != 'nan' else None

        if USE_DATABASE:
            close = nasdaq_latest['close_price']
            daily_chg = nasdaq_latest['daily_change_pct']
            exposure = nasdaq_latest['market_exposure']
            allocation = nasdaq_latest['position_allocation'] * 100
            dist_count = nasdaq_latest['distribution_count']
            buy_switch = nasdaq_latest['buy_switch']
            buy_sigs = _clean_sigs(nasdaq_latest['buy_signals'])
            sell_sigs = _clean_sigs(nasdaq_latest['sell_signals'])
        else:
            close = nasdaq_latest['close']
            daily_chg = float(nasdaq_latest['daily_change'].rstrip('%'))
            exposure = nasdaq_latest['market_exposure']
            allocation = float(nasdaq_latest['position_allocation'].rstrip('%'))
            dist_count = nasdaq_latest['distribution_count']
            buy_switch = nasdaq_latest['buy_switch'] == 'ON'
            buy_sigs = _clean_sigs(nasdaq_latest.get('buy_signals'))
            sell_sigs = _clean_sigs(nasdaq_latest.get('sell_signals'))

        m1, m2 = st.columns(2)
        m1.metric("Close", f"${close:,.2f}", f"{daily_chg:+.2f}%")
        m2.metric("Buy Switch", "ON ✅" if buy_switch else "OFF ❌")

        # Get live correction state and distribution count (not stale DB values)
        corr_state = get_correction_state("^IXIC")
        live_dist_days = get_active_distribution_days("^IXIC")
        live_dist_count = len(live_dist_days)

        m3, m4 = st.columns(2)
        m3.metric("Exposure Level", f"{exposure}/6", f"{allocation:.0f}% allocation")
        m4.metric("Distribution Days", live_dist_count)

        if buy_sigs or sell_sigs:
            st.markdown("**Signals Today:**")
            if buy_sigs:
                st.success(f"🟢 BUY: {buy_sigs}")
            if sell_sigs:
                st.error(f"🔴 SELL: {sell_sigs}")
        else:
            st.info("No new signals today")
        if corr_state:
            if corr_state['market_in_correction']:
                decline_pct = ((close - corr_state['reference_high']) / corr_state['reference_high']) * 100 if corr_state['reference_high'] else 0
                rd_type = corr_state.get('rally_day_type')
                if corr_state['rally_start_date'] and corr_state['rally_day'] is not None and rd_type is not None:
                    day_num = corr_state['rally_day']  # Already 1-indexed: Day 1 = rally day
                    rd_label = "Rally Day" if rd_type == "rally" else "Pink Rally Day"
                    ftd_window = "— in FTD window (days 4-25)" if 4 <= day_num <= 25 else ""
                    if day_num < 4:
                        ftd_window = f"— FTD eligible from Day 4 ({4 - day_num} more days)"
                    st.warning(
                        f"**MARKET IN CORRECTION** ({decline_pct:.1f}% from ref high ${corr_state['reference_high']:,.2f})  \n"
                        f"{rd_label} — Day {day_num} from {corr_state['rally_start_date']} (low ${corr_state['rally_low']:,.2f}) "
                        f"{ftd_window}  \n"
                        f"**Status: Looking for Follow-Through Day**"
                    )
                else:
                    st.error(
                        f"**MARKET IN CORRECTION** ({decline_pct:.1f}% from ref high ${corr_state['reference_high']:,.2f})  \n"
                        f"**Status: Waiting for rally attempt**"
                    )
            elif corr_state['ftd_date']:
                st.success(
                    f"**CONFIRMED UPTREND** — FTD on {corr_state['ftd_date']}  \n"
                    f"Reference high: ${corr_state['reference_high']:,.2f}"
                )

        # Distribution Days Detail
        with st.expander(f"📋 Distribution Days Detail ({live_dist_count} active)"):
            if live_dist_days:
                st.markdown("**Active Distribution Days:**")
                for dd in sorted(live_dist_days, key=lambda x: x.date, reverse=True):
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
        st.warning("No Nasdaq data available")

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

                ax.plot(nasdaq_hist['signal_date'], nasdaq_hist['market_exposure'],
                       marker='o', label='NASDAQ', color='blue', linewidth=2)

                ax.set_xlabel('Date')
                ax.set_ylabel('Exposure Level (0-6)')
                ax.set_title('Market Exposure Trend')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.5, 6.5)

                st.pyplot(fig)

            with tab2:
                filtered_df = df_30d[df_30d['symbol'] == '^IXIC'].copy()
                symbols_to_analyze = ['^IXIC']

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
                display_df = display_df[['signal_date', 'close_price', 'daily_change_pct',
                                         'dd_added', 'dd_removed', 'distribution_count', 'market_exposure',
                                         'buy_signals', 'sell_signals', 'notes']]
                display_df.columns = ['Date', 'Close', 'Daily %', 'DD+', 'DD-', 'Cum DD',
                                     'Exposure', 'Buy Signals', 'Sell Signals', 'Notes']
                display_df = display_df.sort_values('Date', ascending=False)

                st.dataframe(display_df, use_container_width=True, height=400)

    # === SIGNAL LEGEND ===

    with st.expander("📖 Signal Reference Guide"):
        col_buy, col_sell = st.columns(2)

        with col_buy:
            st.markdown("**🟢 BUY SIGNALS**")
            st.markdown("""
**B1 — Follow-Through Day (FTD)**
Initial confirmation of a new rally. Day 4+ of a rally attempt that closes up ≥1% on volume higher than the prior day. Turns the Buy Switch ON and resets distribution count.

**B2 — Additional Follow-Through Day**
Any day within 25 days of the initial FTD that closes up ≥1% on higher volume **and** closes above the low of the initial FTD. Does *not* reset distribution.

**B3 — Low Above 21-day MA**
Buy on an up or flat day when the intraday low is at or above the 21-day EMA. Fires once per cycle; reset by S5.

**B4 — Trending Above 21-day MA**
After B3, fires on the 3rd consecutive day with low ≥ 21EMA, provided the index closes up or flat. If the 3rd day closes down, waits for the next up day.

**B5 — Living Above 21-day MA**
After B3, fires on the 10th consecutive day with low ≥ 21EMA and every 5th day after that (day 15, 20, 25…). Close must be up or flat.

**B6 — Low Above 50-day MA**
Reclaim signal: intraday low ≥ 50SMA on an up/flat day. Only fires if S9 fired previously in this cycle (reset by S9).

**B7 — Accumulation Day**
Strength signal: close up ≥1% on heavier volume, close in upper 25% of day's range, close > 21EMA. Cannot coincide with a B1 or B2.

**B8 — Higher High**
Close above the last marked 13-week high during an uptrend.

**B9 — Downside Reversal Buyback**
After selling on a downside reversal day (S11), buy back if the index closes above the intraday high of that reversal within 2 trading days.

**B10 — Distribution Day Fall Off**
Distribution count falls back to 4 (from 5 or 6) while the close is above the 21EMA and Buy Switch is ON.
            """)

        with col_sell:
            st.markdown("**🔴 SELL SIGNALS**")
            st.markdown("""
**S1 — Follow-Through Day Undercut**
Close below the low of the **initial** FTD. Voids the rally: Buy Switch OFF, full exit, back to correction. Does not apply to Additional FTDs.

**S2 — Failed Rally Attempt**
*Major low*: intraday low breaks the pre-FTD rally low → full exit, Buy Switch OFF, back to correction.
*Minor low*: intraday low breaks a post-FTD 5-bar swing low → exposure −2, Buy Switch stays ON.

**S3 — Full Distribution minus one**
Active distribution day count hits 5 (full distribution is 6).

**S4 — Full Distribution**
Active distribution day count hits 6 (applies up to 8). Turns Buy Switch OFF.

**S5 — Break Below 21-day MA**
Close ≥0.2% below the 21-day EMA after being above it. Once per cycle; reset by B3.

**S6 — Overdue Break Below 21-day MA**
Same ≥0.2% break, but fires after 30+ trading days since the last B3 without any intervening S5. Prevents the lockout from suppressing legitimate sells.

**S7 — Trending Below 21-day MA**
After S5, fires on the 5th consecutive day where the intraday high is below the 21EMA, provided the close is down. If close is up, waits for the next down day.

**S8 — Living Below 21-day MA**
After S5, fires on the 10th consecutive day of high below 21EMA and every 5th day after that. Close must be down.

**S9 — Break Below 50-day MA**
Close below the 50-day SMA. Shakeout exception: closes in upper half of range **and** within 1% of the 50SMA. Reset by B6.

**S10 — Bad Break**
Close down ≥2.25% **and** in bottom 25% of day's range **and** (below 50SMA or intraday high below 21EMA).

**S11 — Downside Reversal Day**
New 13-week intraday high, close in bottom quartile, close down, spread ≥1.75% (lowered to 1% in low-vol regimes).

**S12 — Lower Low**
Close below the last marked 13-week low.

**S13 — Distribution Cluster**
4+ distribution/stall days in a rolling 8-day trading window. Fires again on each successive increase (5, 6, 7, 8). Resets when the count drops to ≤3.

**S14 — Break Below Higher High**
Close below a marked 13-week high that previously triggered a B8. Each marked high can only be broken once.
            """)

# ==============================================================================
# PAGE: AI COACH
# ==============================================================================
elif page == "AI Coach":
    page_header("AI Trading Coach", CURR_PORT_NAME, "🤖")

    # --- Check API key ---
    _ai_api_key = ""
    try:
        _ai_api_key = st.secrets.get("anthropic", {}).get("api_key", "")
    except Exception:
        pass

    if not _ai_api_key:
        st.error("Anthropic API key not configured. Add `[anthropic] api_key = 'sk-...'` to your Streamlit secrets.")
        st.stop()

    import anthropic

    @st.cache_resource
    def get_anthropic_client():
        return anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])

    ai_client = get_anthropic_client()

    # --- Load all data ---
    df_d, df_s = load_trade_data()
    JOURNAL_FILE = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')
    try:
        df_j = load_data(JOURNAL_FILE)
    except Exception:
        df_j = pd.DataFrame()

    # Sanitize journal
    if not df_j.empty:
        df_j['Day'] = pd.to_datetime(df_j['Day'], errors='coerce')
        for _c in ['End NLV', 'Beg NLV', 'Cash -/+', 'Daily $ Change']:
            if _c in df_j.columns:
                df_j[_c] = pd.to_numeric(df_j[_c].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0.0)

    # --- System prompt ---
    SYSTEM_PROMPT = f"""You are MO's personal AI trading coach. You analyze his CANSLIM trading journal data and provide actionable insights.

TRADING SYSTEM CONTEXT:
- MO trades the CANSLIM / IBD growth stock strategy
- Buy rules: Base breakouts, volume events (HVE/HVSI), moving average reclaims, pullbacks, gap-ups, trendline breaks
- Sell rules: Capital protection (stop loss), selling into strength, portfolio management, change of character, breakout failure
- Position sizing: Risk-based with stop losses and risk budgets
- Transaction IDs: B1/B2 = initial buys, A1/A2 = add-on buys, S1/S2 = sells
- Trade IDs format: YYYYMM-NNN (e.g., 202602-001)
- Journal columns: Beg NLV = starting equity for the day, End NLV = ending equity. A week's P&L = End NLV of last day minus Beg NLV of first day (which equals prior week's End NLV)

ANALYSIS GUIDELINES:
- Be specific — reference actual tickers, dates, and numbers from the data
- Focus on patterns, not individual trades (unless asked)
- Highlight both strengths and areas for improvement
- Frame feedback constructively — you're a coach, not a critic
- When discussing sell discipline, compare actual exit vs. stop loss and optimal exit
- Keep responses concise and actionable — use bullet points
- Use dollar amounts and percentages to quantify insights
- IMPORTANT: Never use $ for dollar amounts — Streamlit renders $ as LaTeX math. Instead write "USD" or just the number with commas (e.g., "346,619" not "$346,619"). Same for inline math — avoid single $ delimiters entirely.
- If the data is insufficient to answer, say so honestly

Current portfolio: {CURR_PORT_NAME}
Today's date: {get_current_date_ct().strftime('%Y-%m-%d')}
"""

    # --- Helper: Build data context for prompts ---
    def build_trade_context(scope="recent", n=10):
        """Build a data summary string to include in prompts."""
        parts = []

        if not df_s.empty:
            closed = df_s[df_s['Status'].str.upper() == 'CLOSED'].copy()
            if 'Closed_Date' in closed.columns:
                closed['Closed_Date'] = pd.to_datetime(closed['Closed_Date'], errors='coerce')
                closed = closed.sort_values('Closed_Date', ascending=False)
            open_trades = df_s[df_s['Status'].str.upper() == 'OPEN'].copy()

            if scope == "recent":
                show_closed = closed.head(n)
            else:
                show_closed = closed

            # Use compact CSV format to fit more trades in context
            if not show_closed.empty:
                cols = ['Trade_ID', 'Ticker', 'Open_Date', 'Closed_Date', 'Total_Shares',
                        'Avg_Entry', 'Avg_Exit', 'Realized_PL', 'Return_Pct',
                        'Rule', 'Sell_Rule', 'Stop_Loss']
                use_cols = [c for c in cols if c in show_closed.columns]
                parts.append(f"=== CLOSED TRADES ({len(show_closed)}) ===")
                parts.append(show_closed[use_cols].to_csv(index=False))

            if not open_trades.empty:
                cols = ['Trade_ID', 'Ticker', 'Open_Date', 'Total_Shares',
                        'Avg_Entry', 'Total_Cost', 'Unrealized_PL', 'Rule', 'Stop_Loss']
                use_cols = [c for c in cols if c in open_trades.columns]
                parts.append(f"\n=== OPEN TRADES ({len(open_trades)}) ===")
                parts.append(open_trades[use_cols].to_csv(index=False))

        # Only include transaction details for recent/small scopes (not "all")
        if scope == "recent" and not df_d.empty and not df_s.empty:
            recent_ids = df_s.sort_values('Open_Date', ascending=False).head(n)['Trade_ID'].tolist() if 'Open_Date' in df_s.columns else df_s.head(n)['Trade_ID'].tolist()
            txns = df_d[df_d['Trade_ID'].isin(recent_ids)]
            if not txns.empty:
                cols = ['Trade_ID', 'Trx_ID', 'Ticker', 'Action', 'Date', 'Shares', 'Amount', 'Value', 'Rule']
                use_cols = [c for c in cols if c in txns.columns]
                parts.append(f"\n=== TRANSACTIONS ({len(txns)}) ===")
                parts.append(txns[use_cols].to_csv(index=False))

        return "\n".join(parts)

    def build_journal_context(n=14):
        """Build journal data summary for prompts."""
        if df_j.empty:
            return "No journal data available."
        recent = df_j.sort_values('Day', ascending=False).head(n)
        parts = [f"=== JOURNAL ENTRIES (last {len(recent)} days) ==="]
        for _, row in recent.iterrows():
            day = row['Day'].strftime('%Y-%m-%d') if pd.notna(row.get('Day')) else '?'
            entry = f"\n--- {day} ---"
            for field in ['Status', 'Market_Window', 'Market_Action', 'Market_Notes',
                          'Daily $ Change', 'Daily % Change', 'End NLV', 'Pct_Invested',
                          'Portfolio_Heat', 'Score', 'Highlights', 'Lowlights', 'Mistakes', 'Top_Lesson']:
                if field in row.index and pd.notna(row[field]) and str(row[field]).strip():
                    entry += f"\n  {field}: {row[field]}"
            parts.append(entry)
        return "\n".join(parts)

    def build_journal_context_by_date(start_date):
        """Build journal data summary from a specific start date onwards."""
        if df_j.empty:
            return "No journal data available."
        filtered = df_j[df_j['Day'].dt.date >= start_date].sort_values('Day', ascending=True)
        if filtered.empty:
            return f"No journal entries found from {start_date.strftime('%Y-%m-%d')} onwards."
        parts = [f"=== JOURNAL ENTRIES ({start_date.strftime('%b %d')} onwards, {len(filtered)} days) ==="]
        for _, row in filtered.iterrows():
            day = row['Day'].strftime('%Y-%m-%d') if pd.notna(row.get('Day')) else '?'
            entry = f"\n--- {day} ---"
            for field in ['Status', 'Market_Window', 'Market_Action', 'Market_Notes',
                          'Daily $ Change', 'Daily % Change', 'End NLV', 'Pct_Invested',
                          'Portfolio_Heat', 'Score', 'Highlights', 'Lowlights', 'Mistakes', 'Top_Lesson']:
                if field in row.index and pd.notna(row[field]) and str(row[field]).strip():
                    entry += f"\n  {field}: {row[field]}"
            parts.append(entry)
        return "\n".join(parts)

    def build_stats_context():
        """Build aggregate stats summary."""
        parts = ["=== PORTFOLIO STATS ==="]
        if not df_s.empty:
            closed = df_s[df_s['Status'].str.upper() == 'CLOSED']
            open_t = df_s[df_s['Status'].str.upper() == 'OPEN']
            if not closed.empty and 'Realized_PL' in closed.columns:
                pl = closed['Realized_PL'].apply(clean_num)
                wins = pl[pl > 0]
                losses = pl[pl < 0]
                parts.append(f"Total closed trades: {len(closed)}")
                parts.append(f"Open positions: {len(open_t)}")
                parts.append(f"Win rate: {len(wins)}/{len(closed)} ({len(wins)/len(closed)*100:.1f}%)")
                parts.append(f"Total realized P&L: ${pl.sum():,.2f}")
                parts.append(f"Avg win: ${wins.mean():,.2f}" if len(wins) > 0 else "Avg win: N/A")
                parts.append(f"Avg loss: ${losses.mean():,.2f}" if len(losses) > 0 else "Avg loss: N/A")
                parts.append(f"Best trade: ${pl.max():,.2f}")
                parts.append(f"Worst trade: ${pl.min():,.2f}")
                if len(wins) > 0 and len(losses) > 0 and losses.mean() != 0:
                    parts.append(f"Avg W/L ratio: {abs(wins.mean()/losses.mean()):.2f}")
                if 'Rule' in closed.columns:
                    buy_rules = closed['Rule'].dropna()
                    if not buy_rules.empty:
                        parts.append(f"\nBuy rules used: {buy_rules.value_counts().head(5).to_string()}")
                if 'Sell_Rule' in closed.columns:
                    sell_rules = closed['Sell_Rule'].dropna()
                    if not sell_rules.empty:
                        parts.append(f"\nSell rules used: {sell_rules.value_counts().head(5).to_string()}")
        if not df_j.empty and 'End NLV' in df_j.columns:
            latest = df_j.sort_values('Day', ascending=False).iloc[0]
            parts.append(f"\nLatest NLV: ${clean_num(latest.get('End NLV', 0)):,.0f}")
        return "\n".join(parts)

    # --- AI Call helper ---
    def call_coach(user_prompt, data_context, placeholder):
        """Send prompt to Claude and stream response."""
        messages = [{"role": "user", "content": f"{data_context}\n\n{user_prompt}"}]
        full_response = ""
        def _escape_dollars(text):
            """Escape $ signs so Streamlit doesn't render them as LaTeX."""
            return text.replace("$", "\\$")

        try:
            with ai_client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(_escape_dollars(full_response) + "▌")
            placeholder.markdown(_escape_dollars(full_response))
        except anthropic.RateLimitError:
            full_response = "⚠️ API rate limit reached. Please wait a moment and try again."
            placeholder.warning(full_response)
        except anthropic.APIStatusError as e:
            full_response = f"⚠️ API error: {e.status_code}. Please try again."
            placeholder.warning(full_response)
        return full_response

    # --- Initialize chat history ---
    if 'coach_history' not in st.session_state:
        st.session_state.coach_history = []

    # --- Pre-built Analysis Buttons ---
    st.markdown("#### Quick Analysis")
    b1, b2, b3, b4 = st.columns(4)

    with b1:
        btn_recent = st.button("📋 Review Recent Trades", use_container_width=True)
    with b2:
        btn_mistakes = st.button("⚠️ Mistake Patterns", use_container_width=True)
    with b3:
        btn_weekly = st.button("📅 Weekly Summary", use_container_width=True)
    with b4:
        btn_behavior = st.button("🧠 Behavioral Trends", use_container_width=True)

    st.markdown("---")

    # Handle pre-built analysis buttons
    _preset_prompt = None
    _preset_context = None

    if btn_recent:
        _preset_prompt = """Review my recent closed trades. For each trade, analyze:
1. Entry quality — was the buy rule appropriate? Was timing good?
2. Exit execution — did I follow my sell rules? Did I sell too early or too late?
3. Position sizing — was the risk budget reasonable?

Then give me an overall grade (A-F) and your top 3 actionable recommendations."""
        _preset_context = build_trade_context("all")

    elif btn_mistakes:
        _preset_prompt = """Analyze my journal entries looking for recurring mistake patterns.
Group the mistakes into categories and rank them by frequency and cost.
For each pattern:
1. What the mistake is
2. How often it occurs
3. What it's costing me (dollars or opportunity)
4. A specific, actionable fix

Also look at my 'Lowlights' entries for additional patterns."""
        _preset_context = build_journal_context(30)

    elif btn_weekly:
        # Calculate current trading week (Monday to Friday)
        _today = get_current_date_ct()
        _weekday = _today.weekday()  # 0=Mon, 4=Fri
        _week_start = _today - timedelta(days=_weekday)  # Monday of this week
        _week_end = _week_start + timedelta(days=4)  # Friday of this week

        # Include prior week's last trading day so AI has the starting baseline
        _context_start = _week_start - timedelta(days=3)  # Include from prior Thursday/Friday

        _preset_prompt = f"""Give me a coaching summary for the trading week of {_week_start.strftime('%b %d')} – {_week_end.strftime('%b %d, %Y')}. Cover:
1. P&L performance and equity trend
2. What I did well (from Highlights)
3. What needs work (from Lowlights/Mistakes)
4. Market conditions and how I adapted
5. One key lesson or focus for next week

Keep it concise and actionable — like a halftime talk from a coach."""
        _preset_context = build_journal_context_by_date(_context_start)

    elif btn_behavior:
        _preset_prompt = """Analyze my trading behavior patterns across all my data. Look at:
1. Do I add to winners or average down on losers?
2. Am I cutting losses quickly or holding too long? (compare exit price vs stop loss)
3. Am I selling winners too early? (compare exit price vs highs after entry)
4. Position sizing patterns — am I sizing up on high-conviction trades?
5. Time patterns — am I more profitable on certain setups or market conditions?
6. How is my execution grading trending?

Give me a behavioral profile and 3 specific things to work on."""
        _preset_context = build_trade_context("all") + "\n\n" + build_journal_context(30) + "\n\n" + build_stats_context()

    # --- Display existing chat history FIRST ---
    def _escape_dollars_display(text):
        """Escape $ signs so Streamlit doesn't render them as LaTeX."""
        return text.replace("$", "\\$")

    for msg in st.session_state.coach_history:
        _avatar = "🤖" if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=_avatar):
            _content = _escape_dollars_display(msg["content"]) if msg["role"] == "assistant" else msg["content"]
            st.markdown(_content)

    # --- Handle pre-built analysis (new messages only) ---
    if _preset_prompt:
        st.session_state.coach_history.append({"role": "user", "content": _preset_prompt})
        with st.chat_message("user"):
            st.markdown(_preset_prompt)
        with st.chat_message("assistant", avatar="🤖"):
            _ph = st.empty()
            response = call_coach(_preset_prompt, _preset_context, _ph)
        st.session_state.coach_history.append({"role": "assistant", "content": response})

    # --- Free-form chat input ---
    user_input = st.chat_input("Ask your AI coach anything about your trading...")

    if user_input:
        st.session_state.coach_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Always send all trade data + stats — let Claude decide what's relevant
        context_parts = [build_stats_context(), build_trade_context("all")]

        # Add journal context — more days for broader queries
        _lower = user_input.lower()
        if any(w in _lower for w in ['all', 'history', 'overall', 'pattern', 'behavior', 'trend', 'year', '2026', '2025', '2024']):
            context_parts.append(build_journal_context(90))
        elif any(w in _lower for w in ['week', 'recent', 'last', 'today', 'yesterday']):
            context_parts.append(build_journal_context(14))
        else:
            context_parts.append(build_journal_context(30))

        full_context = "\n\n".join(context_parts)

        with st.chat_message("assistant", avatar="🤖"):
            _ph = st.empty()
            response = call_coach(user_input, full_context, _ph)
        st.session_state.coach_history.append({"role": "assistant", "content": response})

    # --- Clear chat button ---
    if st.session_state.coach_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.coach_history = []
            st.rerun()

