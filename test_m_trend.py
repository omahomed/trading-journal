import pandas as pd
import yfinance as yf
from datetime import datetime

print("--- MARKET TREND DIAGNOSTIC ---")

# 1. Load Summary Data
try:
    df_s = pd.read_csv('Trade_Log_Summary.csv')
    print(f">> Loaded {len(df_s)} trades from Summary.")
    
    # Check Date Format
    sample_date = df_s['Open_Date'].iloc[0]
    print(f">> Sample Trade Date (Raw): {sample_date} (Type: {type(sample_date)})")
    
    # Convert
    df_s['Open_Date_DT'] = pd.to_datetime(df_s['Open_Date'], errors='coerce')
    print(f">> Sample Trade Date (Converted): {df_s['Open_Date_DT'].iloc[0]}")

except Exception as e:
    print(f"Error reading summary: {e}")
    exit()

# 2. Fetch Market Data
print("\n>> Fetching Nasdaq Data...")
try:
    # Fetch
    mkt = yf.Ticker("^IXIC").history(period="5y") # Increased to 5y to cover older trades
    if mkt.empty:
        print("!! Nasdaq fetch failed. Trying SPY...")
        mkt = yf.Ticker("SPY").history(period="5y")
    
    # Calculate 21EMA
    mkt['21EMA'] = mkt['Close'].ewm(span=21, adjust=False).mean()
    
    # Inspect Index
    print(f">> Market Data Loaded: {len(mkt)} rows.")
    sample_mkt_date = mkt.index[0]
    print(f">> Sample Market Date (Raw): {sample_mkt_date} (Type: {type(sample_mkt_date)})")
    print(f"   Timezone Info: {sample_mkt_date.tzinfo}")

    # REMOVE TIMEZONE (The likely fix)
    mkt.index = mkt.index.tz_localize(None)
    print(f">> Market Date after tz_localize(None): {mkt.index[0]}")

except Exception as e:
    print(f"Error fetching market data: {e}")
    exit()

# 3. Test Match Logic
print("\n>> Testing Match Logic on First 5 Trades...")

def test_match(trade_date):
    if pd.isna(trade_date): return "No Date"
    
    # Normalize to midnight
    d = trade_date.normalize()
    
    # Search
    try:
        # Find nearest index
        idx = mkt.index.get_indexer([d], method='nearest')[0]
        
        # Check bounds
        if idx == -1: return "Index Error"
        
        found_date = mkt.index[idx]
        diff = (found_date - d).days
        
        price = mkt.iloc[idx]['Close']
        ema = mkt.iloc[idx]['21EMA']
        trend = "UP" if price > ema else "DOWN"
        
        return f"Trade: {d.date()} | Mkt: {found_date.date()} (Diff: {diff}d) | {trend} ({price:.2f} vs {ema:.2f})"
    except Exception as e:
        return f"Error: {e}"

# Loop through first 5 valid dates
for i in range(min(5, len(df_s))):
    t_date = df_s['Open_Date_DT'].iloc[i]
    result = test_match(t_date)
    print(f"   {result}")

print("\n--- END DIAGNOSTIC ---")