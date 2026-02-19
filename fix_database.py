import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
SOURCE_FILE = 'CLOSED TRADE LOGS.csv'
DETAILS_FILE = 'Trade_Log_Details.csv'
SUMMARY_FILE = 'Trade_Log_Summary.csv'

print("--- CAN SLIM DATABASE REPAIR (FINAL) ---")

# 1. WIPE OLD DATA
if os.path.exists(DETAILS_FILE): os.remove(DETAILS_FILE)
if os.path.exists(SUMMARY_FILE): os.remove(SUMMARY_FILE)
print(">> Old database wiped.")

# 2. INITIALIZE NEW FILES
pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Amount','Value','Rule','Notes','Realized_PL','Stop_Loss']).to_csv(DETAILS_FILE, index=False)
pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Closed_Date','Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Return_Pct','Buy_Rule','Sell_Rule','Notes']).to_csv(SUMMARY_FILE, index=False)

df_d = pd.read_csv(DETAILS_FILE)
df_s = pd.read_csv(SUMMARY_FILE)

# 3. LOAD SOURCE
if not os.path.exists(SOURCE_FILE):
    print(f"CRITICAL ERROR: {SOURCE_FILE} not found.")
    exit()

try:
    # Read with header=0 (First row is headers)
    df_hist = pd.read_csv(SOURCE_FILE, header=0)
    # Clean column names (strip spaces)
    df_hist.columns = [c.strip() for c in df_hist.columns]
    print(f">> Loaded {len(df_hist)} rows.")
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# 4. HELPER: CLEAN NUMBERS
def clean_num(x):
    try:
        if pd.isna(x): return 0.0
        s = str(x).strip().replace('$','').replace(',','').replace('%','').replace('"','')
        if '(' in s: s = s.replace('(','-').replace(')','')
        return float(s)
    except: return 0.0

# 5. HELPER: CLEAN DATES
def parse_date(d_str):
    if pd.isna(d_str): return datetime.now().strftime('%Y-%m-%d')
    try:
        # Try MM/DD/YY (e.g. 11/16/23)
        return pd.to_datetime(d_str, dayfirst=False).strftime('%Y-%m-%d')
    except:
        return datetime.now().strftime('%Y-%m-%d')

# 6. PROCESS
new_s = []
new_d = []
count = 0

print(">> Processing trades...")

for idx, row in df_hist.iterrows():
    try:
        # Map Columns
        tid = str(row.get('Trade_ID') or row.get('Trade #')).strip()
        ticker = str(row.get('Ticker')).strip().upper()
        
        if not tid or tid.lower() == 'nan': continue

        shares = clean_num(row.get('Shares'))
        avg_ent = clean_num(row.get('Avg_Entry'))
        avg_ext = clean_num(row.get('Avg_Exit'))
        
        if shares == 0: continue

        # Dates (Handle 'Close_Date' vs 'Exit Date')
        d_in_raw = row.get('Open_Date') or row.get('Entry Date')
        d_out_raw = row.get('Close_Date') or row.get('Exit Date') or row.get('Closed_Date')
        
        d_in = parse_date(d_in_raw)
        d_out = parse_date(d_out_raw)

        # Math
        cost = shares * avg_ent
        proceeds = shares * avg_ext
        pnl = proceeds - cost
        ret = (pnl / cost) * 100 if cost else 0

        # 1. Summary Entry
        new_s.append({
            'Trade_ID': tid, 'Ticker': ticker, 'Status': 'CLOSED',
            'Open_Date': f"{d_in} 09:30", 'Closed_Date': f"{d_out} 16:00",
            'Shares': shares, 'Avg_Entry': avg_ent, 'Avg_Exit': avg_ext,
            'Total_Cost': cost, 'Realized_PL': pnl, 'Unrealized_PL': 0.0,
            'Return_Pct': ret, 'Buy_Rule': 'History', 'Sell_Rule': 'History', 'Notes': 'Imported'
        })

        # 2. Detail Entry (Buy)
        new_d.append({
            'Trade_ID': tid, 'Ticker': ticker, 'Action': 'BUY',
            'Date': f"{d_in} 09:30", 'Shares': shares, 'Amount': avg_ent,
            'Value': cost, 'Rule': 'History', 'Notes': 'Imported Buy',
            'Realized_PL': 0.0, 'Stop_Loss': 0.0
        })

        # 3. Detail Entry (Sell)
        new_d.append({
            'Trade_ID': tid, 'Ticker': ticker, 'Action': 'SELL',
            'Date': f"{d_out} 16:00", 'Shares': shares, 'Amount': avg_ext,
            'Value': proceeds, 'Rule': 'History', 'Notes': 'Imported Sell',
            'Realized_PL': pnl, 'Stop_Loss': 0.0
        })
        
        count += 1

    except Exception as e:
        print(f"   Skipped Row {idx}: {e}")

# 7. SAVE
if new_s:
    pd.DataFrame(new_s).to_csv(SUMMARY_FILE, index=False)
    pd.DataFrame(new_d).to_csv(DETAILS_FILE, index=False)
    print(f"\n>> SUCCESS! Database rebuilt with {count} historical trades.")
else:
    print("\n>> No trades found. Check headers.")