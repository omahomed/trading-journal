import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
IBKR_FILE = 'Transaction_History_IBKR.csv'
DETAILS_FILE = 'Trade_Log_Details.csv'
SUMMARY_FILE = 'Trade_Log_Summary.csv'

print("--- IBKR AUTO-IMPORTER ---")

# 1. LOAD FILES
if not os.path.exists(IBKR_FILE):
    print(f"Error: {IBKR_FILE} not found.")
    exit()

if not os.path.exists(DETAILS_FILE) or not os.path.exists(SUMMARY_FILE):
    print("Error: Database files not found. Run the App first to generate them.")
    exit()

df_ibkr = pd.read_csv(IBKR_FILE)
df_d = pd.read_csv(DETAILS_FILE)
df_s = pd.read_csv(SUMMARY_FILE)

# Ensure columns are clean
df_ibkr.columns = [c.strip() for c in df_ibkr.columns]

# 2. HELPER FUNCTIONS
def get_next_id(df_s):
    if df_s.empty: return "2025-001"
    try:
        # Find max ID logic
        last_id = df_s['Trade_ID'].iloc[-1]
        prefix, num = str(last_id).split('-')
        new_num = int(num) + 1
        return f"{prefix}-{new_num:03d}"
    except:
        return "2025-999" # Fallback

def generate_trx_id(df_d, trade_id, action):
    # Simple counter for auto-import
    existing = df_d[df_d['Trade_ID'] == trade_id]
    count = len(existing) + 1
    prefix = "B" if action == "BUY" else "S"
    return f"{prefix}{count}-Auto" # e.g., B2-Auto

# 3. PROCESS TRANSACTIONS
new_details = []
updated_summaries = {} # Store updates to apply at end
processed_count = 0
skipped_count = 0

print(f"Scanning {len(df_ibkr)} rows from IBKR...")

for idx, row in df_ibkr.iterrows():
    try:
        # Parse IBKR Data
        raw_date = row['Date']
        date_obj = pd.to_datetime(raw_date).strftime('%Y-%m-%d')
        ticker = str(row['Symbol']).upper()
        action = str(row['Transaction Type']).upper() # "BUY" or "SELL"
        shares = float(row['Quantity'])
        price = float(row['Price'])
        
        # Fix Sell Quantity (IBKR might show sells as negative or positive, we need positive for logic)
        shares = abs(shares)
        
        # DUPLICATE CHECK
        # Look for existing transaction in Details with same Date, Ticker, Action, Shares, Price
        is_dup = not df_d[
            (df_d['Ticker'] == ticker) & 
            (df_d['Action'] == action) & 
            (df_d['Shares'] == shares) & 
            (df_d['Amount'] == price) &
            (df_d['Date'].str.contains(date_obj))
        ].empty
        
        if is_dup:
            skipped_count += 1
            continue

        # --- TRADE MATCHING LOGIC ---
        
        # Check if we have an OPEN position for this ticker
        # We prioritize the in-memory updated dataframe or the file
        open_trades = df_s[ (df_s['Ticker'] == ticker) & (df_s['Status'] == 'OPEN') ]
        
        target_id = None
        
        if not open_trades.empty:
            # Found existing campaign
            target_id = open_trades.iloc[0]['Trade_ID']
            idx_s = open_trades.index[0] # Index in df_s
            print(f" -> Match: {ticker} belongs to {target_id}")
        else:
            # New Campaign
            if action == 'BUY':
                target_id = get_next_id(df_s)
                print(f" -> New: Created {target_id} for {ticker}")
                
                # Create New Summary Row
                new_summary = {
                    'Trade_ID': target_id, 'Ticker': ticker, 'Status': 'OPEN',
                    'Open_Date': f"{date_obj} 09:30", 'Closed_Date': None,
                    'Shares': 0, 'Avg_Entry': 0.0, 'Avg_Exit': 0.0, 'Total_Cost': 0.0,
                    'Realized_PL': 0.0, 'Unrealized_PL': 0.0, 'Return_Pct': 0.0,
                    'Buy_Rule': 'IBKR Import', 'Notes': 'Auto-Import'
                }
                # Append immediately so next loop can find it if multiple buys exist
                df_s = pd.concat([df_s, pd.DataFrame([new_summary])], ignore_index=True)
                idx_s = df_s.index[-1]
            else:
                print(f" [!] Warning: SELL record for {ticker} but no Open Position found. Skipping.")
                continue

        # --- UPDATE MATH ---
        timestamp = f"{date_obj} 16:00"
        trx_id = generate_trx_id(df_d, target_id, action)
        total_val = shares * price
        
        # Get current summary values
        curr_shares = df_s.at[idx_s, 'Shares']
        curr_cost = df_s.at[idx_s, 'Total_Cost']
        
        if action == 'BUY':
            new_shares = curr_shares + shares
            new_cost = curr_cost + total_val
            new_avg = new_cost / new_shares if new_shares > 0 else 0
            
            df_s.at[idx_s, 'Shares'] = new_shares
            df_s.at[idx_s, 'Total_Cost'] = new_cost
            df_s.at[idx_s, 'Avg_Entry'] = new_avg
            
            # Detail
            new_details.append({
                'Trade_ID': target_id, 'Trx_ID': trx_id, 'Ticker': ticker, 
                'Action': 'BUY', 'Date': timestamp, 'Shares': shares, 
                'Amount': price, 'Value': total_val, 'Rule': 'IBKR', 
                'Notes': 'Import', 'Realized_PL': 0, 'Stop_Loss': 0
            })
            
        elif action == 'SELL':
            if shares > curr_shares:
                print(f" [!] Warning: Selling {shares} {ticker} but only own {curr_shares}. Cap at {curr_shares}.")
                shares = curr_shares
                total_val = shares * price # Adjust proceeds
            
            avg_cost = df_s.at[idx_s, 'Avg_Entry']
            cost_basis = shares * avg_cost
            realized = total_val - cost_basis
            
            df_s.at[idx_s, 'Shares'] -= shares
            df_s.at[idx_s, 'Total_Cost'] -= cost_basis
            df_s.at[idx_s, 'Avg_Exit'] = price # Update last exit price
            df_s.at[idx_s, 'Realized_PL'] += realized
            
            # Close if 0
            if df_s.at[idx_s, 'Shares'] < 1: # Tolerance for floats
                df_s.at[idx_s, 'Status'] = 'CLOSED'
                df_s.at[idx_s, 'Closed_Date'] = timestamp
                print(f"    -> Closed {target_id}")
                
            # Detail
            new_details.append({
                'Trade_ID': target_id, 'Trx_ID': trx_id, 'Ticker': ticker, 
                'Action': 'SELL', 'Date': timestamp, 'Shares': shares, 
                'Amount': price, 'Value': total_val, 'Rule': 'IBKR', 
                'Notes': 'Import', 'Realized_PL': realized, 'Stop_Loss': 0
            })

        processed_count += 1

    except Exception as e:
        print(f"Error on row {idx}: {e}")

# 4. SAVE CHANGES
if new_details:
    df_d = pd.concat([df_d, pd.DataFrame(new_details)], ignore_index=True)
    
    df_d.to_csv(DETAILS_FILE, index=False)
    df_s.to_csv(SUMMARY_FILE, index=False)
    
    print(f"\n>> SUCCESS: Imported {processed_count} transactions.")
    print(f">> Skipped {skipped_count} duplicates.")
else:
    print("\n>> No new transactions found.")