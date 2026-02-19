import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
# Make sure your file is named EXACTLY this in the folder
SOURCE_FILE = 'CLOSED TRADE LOGS.csv' 

DETAILS_FILE = 'Trade_Log_Details.csv'
SUMMARY_FILE = 'Trade_Log_Summary.csv'

print("--- CAN SLIM HISTORY REPAIR ---")

# 1. RESET DATABASE (Wipe the bad data)
if os.path.exists(DETAILS_FILE): os.remove(DETAILS_FILE)
if os.path.exists(SUMMARY_FILE): os.remove(SUMMARY_FILE)
print(">> Old database wiped.")

# 2. INITIALIZE NEW FILES
pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Amount','Value','Rule','Notes','Realized_PL','Stop_Loss']).to_csv(DETAILS_FILE, index=False)
pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Closed_Date','Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Return_Pct','Buy_Rule','Sell_Rule','Notes']).to_csv(SUMMARY_FILE, index=False)

df_d = pd.read_csv(DETAILS_FILE)
df_s = pd.read_csv(SUMMARY_FILE)

# 3. SMART LOAD SOURCE
if not os.path.exists(SOURCE_FILE):
    # Try the long name just in case
    ALT_FILE = 'CLOSED TRADE LOGS.xlsx - 2025 Trades Log.csv'
    if os.path.exists(ALT_FILE):
        SOURCE_FILE = ALT_FILE
    else:
        print(f"CRITICAL: Could not find {SOURCE_FILE}")
        exit()

print(f">> Reading {SOURCE_FILE}...")

# Detect Header Row
try:
    # Try Row 0 first
    df_temp = pd.read_csv(SOURCE_FILE, header=0)
    # Check if "Ticker" is in columns (handling whitespace)
    cols = [c.strip().lower() for c in df_temp.columns]
    
    if 'ticker' in cols:
        df_hist = df_temp
        print(">> Detected Headers on Row 1")
    else:
        # Try Row 1
        df_hist = pd.read_csv(SOURCE_FILE, header=1)
        print(">> Detected Headers on Row 2 (Skipped empty top row)")

    # Clean Columns
    df_hist.columns = [c.strip() for c in df_hist.columns]
    
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# 4. MAP & PROCESS
def clean_num(x):
    try:
        if pd.isna(x): return 0.0
        s = str(x).replace('$','').replace(',','').replace('%','').strip()
        return float(s)
    except: return 0.0

new_s = []
new_d = []
count = 0

print(">> Processing trades...")

for idx, row in df_hist.iterrows():
    try:
        # Find Columns safely
        tid = str(row.get('Trade #') or row.get('Trade_ID')).strip()
        ticker = str(row.get('Ticker')).strip().upper()
        
        if not tid or tid.lower() == 'nan': continue
        
        shares = clean_num(row.get('Shares'))
        avg_ent = clean_num(row.get('Avg_Entry'))
        avg_ext = clean_num(row.get('Avg_Exit'))
        
        if shares == 0: continue

        # Dates (The Fix)
        raw_entry = row.get('Entry Date')
        raw_exit = row.get('Exit Date')
        
        # Flexible Date Parser
        try: d_in = pd.to_datetime(raw_entry).strftime('%Y-%m-%d')
        except: d_in = datetime.now().strftime('%Y-%m-%d')
            
        try: d_out = pd.to_datetime(raw_exit).strftime('%Y-%m-%d')
        except: d_out = datetime.now().strftime('%Y-%m-%d')

        # Math
        cost = shares * avg_ent
        proceeds = shares * avg_ext
        pnl = proceeds - cost
        ret = (pnl/cost)*100 if cost else 0
        
        # Add Summary
        new_s.append({
            'Trade_ID': tid, 'Ticker': ticker, 'Status': 'CLOSED',
            'Open_Date': f"{d_in} 09:30", 'Closed_Date': f"{d_out} 16:00",
            'Shares': shares, 'Avg_Entry': avg_ent, 'Avg_Exit': avg_ext,
            'Total_Cost': cost, 'Realized_PL': pnl, 'Unrealized_PL': 0.0,
            'Return_Pct': ret, 'Buy_Rule': 'History', 'Sell_Rule': 'History', 'Notes': 'Imported'
        })
        
        # Add Buy Detail
        new_d.append({
            'Trade_ID': tid, 'Ticker': ticker, 'Action': 'BUY',
            'Date': f"{d_in} 09:30", 'Shares': shares, 'Amount': avg_ent,
            'Value': cost, 'Rule': 'History', 'Notes': 'Imported Buy',
            'Realized_PL': 0.0, 'Stop_Loss': 0.0
        })
        
        # Add Sell Detail
        new_d.append({
            'Trade_ID': tid, 'Ticker': ticker, 'Action': 'SELL',
            'Date': f"{d_out} 16:00", 'Shares': shares, 'Amount': avg_ext,
            'Value': proceeds, 'Rule': 'History', 'Notes': 'Imported Sell',
            'Realized_PL': pnl, 'Stop_Loss': 0.0
        })
        
        count += 1
        
    except Exception as e:
        print(f"Skipped row {idx}: {e}")

# 5. WRITE
if new_s:
    pd.concat([df_s, pd.DataFrame(new_s)]).to_csv(SUMMARY_FILE, index=False)
    pd.concat([df_d, pd.DataFrame(new_d)]).to_csv(DETAILS_FILE, index=False)
    print(f"\n>> SUCCESS! Imported {count} trades with CORRECT dates.")
else:
    print("No trades found. Check CSV headers.")