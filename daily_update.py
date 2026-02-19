import pandas as pd
import yfinance as yf
from datetime import datetime
import os

# --- CONFIGURATION ---
FILE_NAME = 'Trading_Journal_Clean.csv' 

print("--- CAN SLIM DAILY TRACKER ---")

# --- 1. USER INPUTS ---
try:
    current_nlv = float(input("1. Enter Closing Net Liquidation Value (NLV): "))
    
    # Input Holdings directly (Stock Value)
    holdings_value = float(input("2. Enter Total Market Value of Holdings: "))
    
    cash_flow_in = input("3. Enter Cash Added/Removed (Press Enter for 0): ")
    cash_flow = float(cash_flow_in) if cash_flow_in else 0.0
except ValueError:
    print("Error: Please enter valid numbers.")
    exit()

# --- 2. MARKET ANALYSIS ---
print("\nReading the Tape...")

# FETCH METHOD: INDIVIDUAL (Robust)
try:
    # Fetch SPY
    spy_ticker = yf.Ticker("SPY")
    spy_hist = spy_ticker.history(period="5d")
    spy_close = spy_hist['Close'].iloc[-1]

    # Fetch Nasdaq Composite (^IXIC)
    nasdaq_ticker = yf.Ticker("^IXIC")
    nasdaq_hist = nasdaq_ticker.history(period="3mo") 
    nasdaq_close = nasdaq_hist['Close'].iloc[-1]
    
    # Calculate Moving Averages (on IXIC)
    ema_21 = nasdaq_hist['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
    sma_50 = nasdaq_hist['Close'].rolling(window=50).mean().iloc[-1]

except Exception as e:
    print(f"Data Error: {e}")
    print("Could not fetch market data. Check internet connection.")
    exit()

# Determine Status
if nasdaq_close > ema_21:
    sys_status = 'U'
elif nasdaq_close < sma_50:
    sys_status = 'C'
else:
    sys_status = 'P'

print(f"COMP: {nasdaq_close:.2f} | 21e: {ema_21:.2f} | 50s: {sma_50:.2f}")
print(f"System says: {sys_status} ({'Uptrend' if sys_status=='U' else 'Pressure/Correction'})")

# Override Option
user_status = input(f"Press Enter to accept '{sys_status}' or type U/P/C to override: ").upper()
final_status = user_status if user_status in ['U', 'P', 'C'] else sys_status

# Logic for dependent columns
if final_status == 'U':
    final_window = 'Open'
    final_indicator = 20
else:
    final_window = 'Closed'
    final_indicator = 10

# --- 3. CALCULATIONS ---
try:
    df = pd.read_csv(FILE_NAME)
    
    # Force Clean Headers
    df.columns = [c.strip() for c in df.columns]
    if 'Nsadaq' in df.columns:
        df.rename(columns={'Nsadaq': 'Nasdaq'}, inplace=True)
        
except FileNotFoundError:
    print(f"File {FILE_NAME} not found in current folder!")
    exit()

# Get Previous Day's Closing Value
if not df.empty:
    try:
        last_val = str(df.iloc[-1]['End NLV']).replace('$','').replace(',','').replace(' ','').strip()
        if '(' in last_val: last_val = last_val.replace('(','-').replace(')','')
        beg_nlv = float(last_val)
    except:
        beg_nlv = current_nlv
else:
    beg_nlv = current_nlv

# P&L Math
daily_dollar = current_nlv - beg_nlv - cash_flow
daily_pct = (daily_dollar / beg_nlv) if beg_nlv != 0 else 0.0

# Invested % Math (Using Holdings Value)
invested_pct = (holdings_value / current_nlv) if current_nlv != 0 else 0.0

today_str = datetime.now().strftime('%m/%d/%y')

# --- 4. SAVE ---
new_row = {
    'Day': today_str,
    'Status': final_status,
    'Market Window': final_window,
    '> 21e': final_indicator,
    'Cash -/+': cash_flow if cash_flow != 0 else '',
    'Beg NLV': beg_nlv,
    'End NLV': current_nlv,
    'Daily $ Change': daily_dollar,
    'Daily % Change': f"{daily_pct:.2%}",
    '% Invested': f"{invested_pct:.2%}",
    'SPY': round(spy_close, 2),
    'Nasdaq': round(nasdaq_close, 2) # Now logging ^IXIC
}

# Create DataFrame for new row
new_df = pd.DataFrame([new_row])
# Align columns to match file exactly
final_df = pd.concat([df, new_df], ignore_index=True)
final_df.to_csv(FILE_NAME, index=False)

print("\nSUCCESS! Journal Updated.")
print(f"Daily Change: ${daily_dollar:.2f} ({new_row['Daily % Change']})")
print(f"Exposure:     {new_row['% Invested']}")