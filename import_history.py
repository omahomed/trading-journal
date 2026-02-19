import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
SOURCE_FILE = 'CLOSED TRADE LOGS.csv'
DETAILS_FILE = 'Trade_Log_Details.csv'
SUMMARY_FILE = 'Trade_Log_Summary.csv'

print("--- HISTORICAL TRADE IMPORTER ---")

def clean_num(x):
    try:
        if pd.isna(x) or x == '': return 0.0
        s = str(x).strip()
        if s == '' or s == '-' or s == '.': return 0.0
        s = s.replace('$', '').replace(',', '').replace('%', '').replace(' ', '')
        if '(' in s: s = s.replace('(', '-').replace(')', '')
        return float(s)
    except: return 0.0

if not os.path.exists(SOURCE_FILE):
    print(f"Error: {SOURCE_FILE} not found.")
    exit()

try:
    # Load with header=0 (First row is header)
    df_hist = pd.read_csv(SOURCE_FILE, header=0)
    df_hist.columns = [c.strip() for c in df_hist.columns]
    print(f"Loaded {len(df_hist)} rows from source.")
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# Create fresh files
if not os.path.exists(DETAILS_FILE):
    pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Cost','Value','Rule','Notes','Realized_PL','Stop_Loss']).to_csv(DETAILS_FILE, index=False)
if not os.path.exists(SUMMARY_FILE):
    pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Closed_Date','Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Return_Pct','Buy_Rule','Sell_Rule','Notes']).to_csv(SUMMARY_FILE, index=False)

df_d = pd.read_csv(DETAILS_FILE)
df_s = pd.read_csv(SUMMARY_FILE)

new_s = []
new_d = []

print("Processing...")

for idx, row in df_hist.iterrows():
    try:
        # Map Columns (Robust)
        tid = str(row.get('Trade #', '')).strip()
        if not tid or tid.lower() == 'nan': 
            # Try alternative mapping if "Trade #" isn't found
            tid = str(row.get('Trade_ID', '')).strip()
        
        ticker = str(row.get('Ticker', '')).strip().upper()
        
        if not tid or not ticker or tid.lower()=='nan': continue

        shares = clean_num(row.get('Shares', 0))
        avg_ent = clean_num(row.get('Avg_Entry', 0))
        avg_ext = clean_num(row.get('Avg_Exit', 0))
        
        if shares == 0: continue

        # Dates
        try: d_in = pd.to_datetime(row.get('Entry Date', '')).strftime('%Y-%m-%d')
        except: d_in = datetime.now().strftime('%Y-%m-%d')
        try: d_out = pd.to_datetime(row.get('Exit Date', '')).strftime('%Y-%m-%d')
        except: d_out = datetime.now().strftime('%Y-%m-%d')

        # Math
        cost = shares * avg_ent
        proceeds = shares * avg_ext
        pnl = proceeds - cost
        ret = (pnl / cost) * 100 if cost else 0

        new_s.append({
            'Trade_ID': tid, 'Ticker': ticker, 'Status': 'CLOSED',
            'Open_Date': f"{d_in} 09:30", 'Closed_Date': f"{d_out} 16:00",
            'Shares': shares, 'Avg_Entry': avg_ent, 'Avg_Exit': avg_ext,
            'Total_Cost': cost, 'Realized_PL': pnl, 'Unrealized_PL': 0.0,
            'Return_Pct': ret, 'Buy_Rule': 'History', 'Sell_Rule': 'History', 'Notes': 'Imported'
        })

        # Buy Detail
        new_d.append({
            'Trade_ID': tid, 'Ticker': ticker, 'Action': 'BUY',
            'Date': f"{d_in} 09:30", 'Shares': shares, 'Cost': avg_ent,
            'Value': cost, 'Rule': 'History', 'Notes': 'Imported',
            'Realized_PL': 0.0, 'Stop_Loss': 0.0
        })
        # Sell Detail
        new_d.append({
            'Trade_ID': tid, 'Ticker': ticker, 'Action': 'SELL',
            'Date': f"{d_out} 16:00", 'Shares': shares, 'Cost': avg_ext,
            'Value': proceeds, 'Rule': 'History', 'Notes': 'Imported',
            'Realized_PL': pnl, 'Stop_Loss': 0.0
        })

    except Exception as e:
        print(f"Skipped Row {idx}: {e}")

if new_s:
    df_s = pd.concat([df_s, pd.DataFrame(new_s)], ignore_index=True)
    df_d = pd.concat([df_d, pd.DataFrame(new_d)], ignore_index=True)
    df_s.to_csv(SUMMARY_FILE, index=False)
    df_d.to_csv(DETAILS_FILE, index=False)
    print(f"SUCCESS: Imported {len(new_s)} trades.")
else:
    print("No trades imported. Check column names in CSV.")