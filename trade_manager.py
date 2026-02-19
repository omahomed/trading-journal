import pandas as pd
import os
from datetime import datetime
import yfinance as yf

# --- CONFIGURATION ---
DETAILS_FILE = 'Trade_Log_Details.csv'
SUMMARY_FILE = 'Trade_Log_Summary.csv'

print("\n" + "="*50)
print("      CAN SLIM TRADE MANAGER")
print("="*50)

# --- 1. INITIALIZE/LOAD FILES ---
def load_file(filename, columns):
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=columns)
        df.to_csv(filename, index=False)
        return df
    return pd.read_csv(filename)

df_details = load_file(DETAILS_FILE, ['Trade_ID', 'Ticker', 'Action', 'Date', 'Shares', 'Price', 'Comm', 'Net_Amount', 'Notes'])
df_summary = load_file(SUMMARY_FILE, ['Trade_ID', 'Ticker', 'Status', 'Open_Date', 'Close_Date', 'Total_Shares', 'Avg_Entry', 'Avg_Exit', 'Total_Cost', 'Net_Proceeds', 'Realized_PL', 'Unrealized_PL', 'Return_Pct', 'Notes'])

# --- 2. HELPER: GENERATE NEXT ID ---
def get_next_id():
    if df_summary.empty:
        return "2025-001"
    last_id = df_summary['Trade_ID'].iloc[-1]
    try:
        prefix, num = last_id.split('-')
        new_num = int(num) + 1
        return f"{prefix}-{new_num:03d}"
    except:
        return "2025-001"

# --- 3. ACTION MENU ---
print("1. LOG NEW BUY (Start New Campaign)")
print("2. ADD TO EXISTING POSITION (Scale In)")
print("3. LOG SELL (Scale Out/Close)")
print("4. UPDATE MARKET PRICES (Mark-to-Market)")
action = input("\nSelect Action (1-4): ")

today_str = datetime.now().strftime("%Y-%m-%d")

# --- LOGIC: NEW BUY ---
if action == '1':
    ticker = input("Ticker: ").upper()
    shares = int(input("Shares: "))
    price = float(input("Price: "))
    notes = input("Notes (e.g. 'Gap Up buy'): ")
    
    new_id = get_next_id()
    cost = shares * price
    
    # Add to Details
    new_detail = {
        'Trade_ID': new_id, 'Ticker': ticker, 'Action': 'BUY', 'Date': today_str,
        'Shares': shares, 'Price': price, 'Comm': 0.0, 'Net_Amount': cost, 'Notes': notes
    }
    df_details = pd.concat([df_details, pd.DataFrame([new_detail])], ignore_index=True)
    
    # Add to Summary
    new_summary = {
        'Trade_ID': new_id, 'Ticker': ticker, 'Status': 'OPEN', 'Open_Date': today_str,
        'Total_Shares': shares, 'Avg_Entry': price, 'Total_Cost': cost, 
        'Realized_PL': 0.0, 'Unrealized_PL': 0.0, 'Return_Pct': 0.0, 'Notes': notes
    }
    df_summary = pd.concat([df_summary, pd.DataFrame([new_summary])], ignore_index=True)
    
    print(f">> LOGGED: Bought {shares} {ticker} @ {price} (ID: {new_id})")

# --- LOGIC: ADD TO EXISTING ---
elif action == '2':
    # Show Open Positions
    open_trades = df_summary[df_summary['Status'] == 'OPEN']
    if open_trades.empty:
        print("No open positions.")
        exit()
        
    print("\nOPEN POSITIONS:")
    for idx, row in open_trades.iterrows():
        print(f"{row['Trade_ID']} | {row['Ticker']} | {row['Total_Shares']} shs")
        
    trade_id = input("Enter Trade ID to Add to: ")
    shares = int(input("Shares Added: "))
    price = float(input("Price: "))
    
    # Details
    new_detail = {
        'Trade_ID': trade_id, 'Ticker': open_trades[open_trades['Trade_ID']==trade_id]['Ticker'].values[0],
        'Action': 'BUY', 'Date': today_str, 'Shares': shares, 'Price': price, 'Comm': 0.0, 'Net_Amount': shares*price, 'Notes': 'Add-on'
    }
    df_details = pd.concat([df_details, pd.DataFrame([new_detail])], ignore_index=True)
    
    # Update Summary
    idx = df_summary[df_summary['Trade_ID'] == trade_id].index[0]
    old_shares = df_summary.at[idx, 'Total_Shares']
    old_cost = df_summary.at[idx, 'Total_Cost']
    
    new_shares = old_shares + shares
    new_cost = old_cost + (shares * price)
    
    df_summary.at[idx, 'Total_Shares'] = new_shares
    df_summary.at[idx, 'Total_Cost'] = new_cost
    df_summary.at[idx, 'Avg_Entry'] = new_cost / new_shares
    
    print(f">> UPDATED {trade_id}: Now {new_shares} shares @ Avg {new_cost/new_shares:.2f}")

# --- LOGIC: SELL (LIFO/FIFO handling simplified for now to WAvg) ---
elif action == '3':
    open_trades = df_summary[df_summary['Status'] == 'OPEN']
    print("\nOPEN POSITIONS:")
    for idx, row in open_trades.iterrows():
        print(f"{row['Trade_ID']} | {row['Ticker']} | {row['Total_Shares']} shs")
        
    trade_id = input("Enter Trade ID to Sell: ")
    shares_sold = int(input("Shares Sold: "))
    price = float(input("Price: "))
    
    # Validate
    idx = df_summary[df_summary['Trade_ID'] == trade_id].index[0]
    current_shares = df_summary.at[idx, 'Total_Shares']
    
    if shares_sold > current_shares:
        print("Error: You don't have that many shares.")
        exit()
        
    proceeds = shares_sold * price
    
    # Details
    new_detail = {
        'Trade_ID': trade_id, 'Ticker': df_summary.at[idx, 'Ticker'],
        'Action': 'SELL', 'Date': today_str, 'Shares': shares_sold, 'Price': price, 
        'Comm': 0.0, 'Net_Amount': proceeds, 'Notes': 'Partial/Full Sell'
    }
    df_details = pd.concat([df_details, pd.DataFrame([new_detail])], ignore_index=True)
    
    # Update Summary
    # Calculate Realized Gain based on Average Cost
    avg_cost = df_summary.at[idx, 'Avg_Entry']
    cost_basis_sold = shares_sold * avg_cost
    realized_gain = proceeds - cost_basis_sold
    
    df_summary.at[idx, 'Total_Shares'] = current_shares - shares_sold
    df_summary.at[idx, 'Total_Cost'] = df_summary.at[idx, 'Total_Cost'] - cost_basis_sold
    # Accumulate Realized PL
    current_realized = df_summary.at[idx, 'Realized_PL']
    if pd.isna(current_realized): current_realized = 0.0
    df_summary.at[idx, 'Realized_PL'] = current_realized + realized_gain
    
    # Check if Closed
    if (current_shares - shares_sold) == 0:
        df_summary.at[idx, 'Status'] = 'CLOSED'
        df_summary.at[idx, 'Close_Date'] = today_str
        print(f">> CLOSED {trade_id}. Realized P&L: ${df_summary.at[idx, 'Realized_PL']:,.2f}")
    else:
        print(f">> PARTIAL SELL {trade_id}. Realized on this lot: ${realized_gain:,.2f}")

# --- LOGIC: MARK TO MARKET (Daily Update) ---
elif action == '4':
    print("Updating Open Positions...")
    open_trades = df_summary[df_summary['Status'] == 'OPEN']
    
    for idx, row in open_trades.iterrows():
        ticker = row['Ticker']
        try:
            # Fetch Price
            if ticker == 'COMP': ticker = '^IXIC' # handle index if needed
            data = yf.Ticker(ticker).history(period='1d')
            curr_price = data['Close'].iloc[-1]
            
            # Calc Unrealized
            mkt_value = row['Total_Shares'] * curr_price
            unrealized = mkt_value - row['Total_Cost']
            
            df_summary.at[idx, 'Unrealized_PL'] = unrealized
            # Return % (Unrealized only on remaining)
            if row['Total_Cost'] > 0:
                df_summary.at[idx, 'Return_Pct'] = (unrealized / row['Total_Cost']) * 100
                
            print(f" -> {ticker}: ${curr_price:.2f} | Open P&L: ${unrealized:,.2f}")
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

# --- SAVE ---
df_details.to_csv(DETAILS_FILE, index=False)
df_summary.to_csv(SUMMARY_FILE, index=False)
print("\n>> Database Updated.")