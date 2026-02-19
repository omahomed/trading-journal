import pandas as pd
import numpy as np
from datetime import datetime

# Load GEV data
print("Loading GEV data...")
df = pd.read_csv("output/GEV_price_data.csv")  # Fixed path

# Convert date and numeric columns
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

# Convert price columns to numeric
for col in ["Open", "High", "Low", "Close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove any rows with NaN values
df = df.dropna()

# Filter to our date range
START_DATE = "2024-09-04"
END_DATE = "2025-09-05"
df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]

print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Total days: {len(df)}\n")

# Calculate EMA21
df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

# Add helper columns for debugging
df["Low_Above_EMA"] = df["Low"] > df["EMA21"]
df["Is_Green"] = df["Close"] > df["Open"]
df["Close_Below_EMA"] = df["Close"] < df["EMA21"]

# Generate signals using Pine Script logic
print("="*60)
print("SIGNAL GENERATION LOGIC CHECK")
print("="*60)

df["signal"] = 0
position_open = False
first_break_low = None
buy_date = None

for i in range(2, len(df)):
    current_date = df.index[i].date()
    
    # BUY SIGNAL CHECK
    if not position_open and i >= 2:
        # Check 3 consecutive days with Low > EMA21
        today_above = df.iloc[i]["Low"] > df.iloc[i]["EMA21"]
        yesterday_above = df.iloc[i-1]["Low"] > df.iloc[i-1]["EMA21"]
        two_days_ago_above = df.iloc[i-2]["Low"] > df.iloc[i-2]["EMA21"]
        three_days_above = today_above and yesterday_above and two_days_ago_above
        
        # Check if current day is green
        is_green = df.iloc[i]["Close"] > df.iloc[i]["Open"]
        
        if three_days_above and is_green:
            df.at[df.index[i], "signal"] = 1
            position_open = True
            first_break_low = None
            buy_date = current_date
            
            print(f"\n📈 BUY SIGNAL on {current_date}")
            print(f"   Low today ({df.iloc[i]['Low']:.2f}) > EMA21 ({df.iloc[i]['EMA21']:.2f}): {today_above}")
            print(f"   Low yesterday ({df.iloc[i-1]['Low']:.2f}) > EMA21 ({df.iloc[i-1]['EMA21']:.2f}): {yesterday_above}")
            print(f"   Low 2 days ago ({df.iloc[i-2]['Low']:.2f}) > EMA21 ({df.iloc[i-2]['EMA21']:.2f}): {two_days_ago_above}")
            print(f"   Is green candle (Close {df.iloc[i]['Close']:.2f} > Open {df.iloc[i]['Open']:.2f}): {is_green}")
            print(f"   Buy at Close: ${df.iloc[i]['Close']:.2f}")
    
    # SELL SIGNAL CHECK
    elif position_open:
        # Step 1: Check if we closed below EMA for first time
        if first_break_low is None:
            closed_below = df.iloc[i]["Close"] < df.iloc[i]["EMA21"]
            prev_closed_above = df.iloc[i-1]["Close"] >= df.iloc[i-1]["EMA21"] if i > 0 else True
            
            if closed_below and prev_closed_above:
                first_break_low = df.iloc[i]["Low"]
                print(f"\n⚠️  FIRST CLOSE BELOW EMA on {current_date}")
                print(f"   Close ({df.iloc[i]['Close']:.2f}) < EMA21 ({df.iloc[i]['EMA21']:.2f})")
                print(f"   Storing Low of {first_break_low:.2f} as trigger")
        
        # Step 2: Sell if low drops below the stored low
        if first_break_low is not None:
            if df.iloc[i]["Low"] < first_break_low:
                df.at[df.index[i], "signal"] = -1
                position_open = False
                
                days_held = (current_date - buy_date).days if buy_date else 0
                print(f"\n📉 SELL SIGNAL on {current_date}")
                print(f"   Low ({df.iloc[i]['Low']:.2f}) < Stored Low ({first_break_low:.2f})")
                print(f"   Sell at Close: ${df.iloc[i]['Close']:.2f}")
                print(f"   Days held: {days_held}")
                
                first_break_low = None
                buy_date = None

# Summary
print("\n" + "="*60)
print("SUMMARY OF SIGNALS")
print("="*60)

buy_signals = df[df["signal"] == 1]
sell_signals = df[df["signal"] == -1]

print(f"Total BUY signals: {len(buy_signals)}")
print(f"Total SELL signals: {len(sell_signals)}")

if len(buy_signals) > 0:
    print("\nBuy dates:")
    for date, row in buy_signals.iterrows():
        print(f"  {date.date()}: Close = ${row['Close']:.2f}")

if len(sell_signals) > 0:
    print("\nSell dates:")
    for date, row in sell_signals.iterrows():
        print(f"  {date.date()}: Close = ${row['Close']:.2f}")

# Show first 30 days of data to verify EMA calculation
print("\n" + "="*60)
print("FIRST 30 DAYS OF DATA (for verification)")
print("="*60)
print("\nDate        | Open   | High   | Low    | Close  | EMA21  | Low>EMA | Green")
print("-"*80)

for i in range(min(30, len(df))):
    row = df.iloc[i]
    date = df.index[i]
    print(f"{date.date()} | {row['Open']:6.2f} | {row['High']:6.2f} | {row['Low']:6.2f} | {row['Close']:6.2f} | "
          f"{row['EMA21']:6.2f} | {str(row['Low_Above_EMA']):7} | {str(row['Is_Green']):5}")

# Check for potential buy setups in the data
print("\n" + "="*60)
print("CHECKING FOR 3-DAY SETUPS")
print("="*60)

for i in range(2, min(50, len(df))):
    if i >= 2:
        three_days = (df.iloc[i]["Low_Above_EMA"] and 
                     df.iloc[i-1]["Low_Above_EMA"] and 
                     df.iloc[i-2]["Low_Above_EMA"])
        
        if three_days:
            print(f"\n✓ 3-day setup found ending {df.index[i].date()}")
            print(f"  Is green? {df.iloc[i]['Is_Green']} (Close: {df.iloc[i]['Close']:.2f}, Open: {df.iloc[i]['Open']:.2f})")
            if df.iloc[i]['Is_Green']:
                print(f"  → Should be a BUY signal!")
            else:
                print(f"  → Not green, no buy")