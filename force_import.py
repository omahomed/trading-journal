import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
SOURCE_FILE = 'CLOSED TRADE LOGS.csv'
DETAILS_FILE = 'Trade_Log_Details.csv'
SUMMARY_FILE = 'Trade_Log_Summary.csv'

print("--- CAN SLIM FACTORY RESET & IMPORT ---")

# 1. WIPE DATABASE
if os.path.exists(DETAILS_FILE): os.remove(DETAILS_FILE)
if os.path.exists(SUMMARY_FILE): os.remove(SUMMARY_FILE)
print(">> Database Wiped (Clean Slate).")

# 2. INITIALIZE FILES
pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Cost','Value','Rule','Notes','Realized_PL','Stop_Loss']).to_csv(DETAILS_FILE, index=False)
pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Closed_Date','Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Return_Pct','Buy_Rule','Sell_Rule','Notes']).to_csv(SUMMARY_FILE, index=False)

df_d = pd.read_csv(DETAILS_FILE)
df_s = pd.read_csv(SUMMARY_FILE)

# 3. HELPERS
def clean_num(x):
    try:
        if pd.isna(x) or str(x).strip() == '': return 0.0
        s = str(x).strip().replace('$','').replace(',','').replace('%','')
        if '(' in s: s = s.replace('(','-').replace(')','')
        return float(s)
    except: return 0.0

# 4. LOAD SOURCE
if not os.path.exists(SOURCE_FILE):
    print(f"CRITICAL ERROR: {SOURCE_FILE} not found.")
    exit()

try:
    # Try reading with header=0 (Standard)
    df_hist = pd.read_csv(SOURCE_FILE, header=0)
    # Clean headers aggressively
    df_hist.columns = [c.strip().replace('"','').replace("'",'') for c in df_hist.columns]
    print(f">> Loaded {len(df_hist)} rows.")
    print(f">> Columns found: {df_hist.columns.tolist()}")
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# 5. MAP COLUMNS
# Find the actual column names even if they have slight variations
col_map = {}
required = {'Trade': 'Trade_ID', 'Ticker': 'Ticker', 'Shares': 'Shares', 'Entry': 'Avg_Entry', 'Exit': 'Avg_Exit'}

print("\n>> Mapping Columns...")
for key, target in required.items():
    match = None
    for c in df_hist.columns:
        if key.lower() in c.lower():
            match = c
            break
    if match:
        col_map[target] = match
        print(f"   - Found '{key}' in column: '{match}'")
    else:
        print(f"   - CRITICAL: Could not find a column for '{key}'")
        exit()

# 6. PROCESS
new_s = []
new_d = []
count = 0

print("\n>> Processing Rows...")
for idx, row in df_hist.iterrows():
    try:
        tid = str(row[col_map['Trade_ID']]).strip()
        ticker = str(row[col_map['Ticker']]).strip().upper()
        
        if not tid or tid.lower() == 'nan': continue
        
        shares = clean_num(row[col_map['Shares']])
        avg_ent = clean_num(row[col_map['Avg_Entry']])
        avg_ext = clean_num(row[col_map['Avg_Exit']])
        
        # DEBUG PRINT FOR FIRST 3 ROWS
        if idx < 3:
            print(f"   Row {idx}: ID={tid}, Tick={ticker}, Shs={shares}, In=${avg_ent}, Out=${avg_ext}")

        if shares == 0:
            if idx < 3: print("     -> SKIPPING: Shares is 0")
            continue

        # Dates (Try to find them, default to today if missing)
        d_in_col = next((c for c in df_hist.columns if 'entry' in c.lower() and 'date' in c.lower()), None)
        d_out_col = next((c for c in df_hist.columns if 'exit' in c.lower() and 'date' in c.lower()), None)
        
        try: d_in = pd.to_datetime(row[d_in_col]).strftime('%Y-%m-%d') if d_in_col else datetime.now().strftime('%Y-%m-%d')
        except: d_in = datetime.now().strftime('%Y-%m-%d')
        
        try: d_out = pd.to_datetime(row[d_out_col]).strftime('%Y-%m-%d') if d_out_col else datetime.now().strftime('%Y-%m-%d')
        except: d_out = datetime.now().strftime('%Y-%m-%d')

        # Math
        cost = shares * avg_ent
        proceeds = shares * avg_ext
        pnl = proceeds - cost
        ret = (pnl / cost) * 100 if cost else 0

        # Add to lists
        new_s.append({'Trade_ID': tid, 'Ticker': ticker, 'Status': 'CLOSED', 'Open_Date': f"{d_in} 09:30", 'Closed_Date': f"{d_out} 16:00", 'Shares': shares, 'Avg_Entry': avg_ent, 'Avg_Exit': avg_ext, 'Total_Cost': cost, 'Realized_PL': pnl, 'Unrealized_PL': 0.0, 'Return_Pct': ret, 'Buy_Rule': 'History', 'Sell_Rule': 'History', 'Notes': 'Imported'})
        
        new_d.append({'Trade_ID': tid, 'Ticker': ticker, 'Action': 'BUY', 'Date': f"{d_in} 09:30", 'Shares': shares, 'Cost': avg_ent, 'Value': cost, 'Rule': 'History', 'Notes': 'Imported Buy', 'Realized_PL': 0.0, 'Stop_Loss': 0.0})
        new_d.append({'Trade_ID': tid, 'Ticker': ticker, 'Action': 'SELL', 'Date': f"{d_out} 16:00", 'Shares': shares, 'Cost': avg_ext, 'Value': proceeds, 'Rule': 'History', 'Notes': 'Imported Sell', 'Realized_PL': pnl, 'Stop_Loss': 0.0})
        
        count += 1

    except Exception as e:
        print(f"   Row {idx} Error: {e}")

# 7. WRITE
if new_s:
    df_s = pd.concat([df_s, pd.DataFrame(new_s)], ignore_index=True)
    df_d = pd.concat([df_d, pd.DataFrame(new_d)], ignore_index=True)
    df_s.to_csv(SUMMARY_FILE, index=False)
    df_d.to_csv(DETAILS_FILE, index=False)
    print(f"\n>> SUCCESS! Imported {count} trades into the database.")
else:
    print("\n>> FAILURE. No trades were valid.")