"""
strategy_executor.py

Purpose:
- Load historical stock data (Open, High, Low, Close) from CSV
- Calculate a 21-day EMA and 14-day ATR
- Identify buy signals when:
    • 3+ consecutive days where Low > EMA21
    • Followed by an up day (Close > prior Close)
- Identify sell signals when:
    • Close < EMA21 * 0.998
- Track:
    • max_high_since_buy: highest price between buy and sell
    • max_high_pct_above_sell_thresh: % difference from sell_threshold to that max
    • close_plus_2atr and close_plus_3atr: upper bands for plotting
- Export to Excel with all strategy-related indicators
"""

import pandas as pd
import warnings
import os

# Suppress warnings for clean output
warnings.simplefilter(action="ignore", category=FutureWarning)

# === Input File ===
ticker = "TSLA"
input_file = f"output/{ticker}_price_data.csv"
output_file = f"output/{ticker}_strategy_results.xlsx"

# === Step 1: Load CSV ===
df = pd.read_csv(input_file)

# === Step 2: Preprocessing ===
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

for col in ["Open", "High", "Low", "Close"]:
    if col not in df.columns:
        raise ValueError(f"❌ Missing required column: {col}")
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

# === Step 3: Compute Indicators ===
df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
df["sell_threshold"] = df["EMA21"] * 0.998
df["low_above_ema"] = df["Low"] > df["EMA21"]
df["up_day"] = df["Close"] > df["Close"].shift(1)
df["signal"] = 0

# === Step 4: Apply Buy/Sell Strategy ===
streak = 0
in_wait_mode = False
position_open = False

for i in range(2, len(df)):
    if position_open:
        if df.iloc[i]["Close"] < df.iloc[i]["sell_threshold"]:
            df.at[df.index[i], "signal"] = -1
            position_open = False
            in_wait_mode = False
            streak = 0
            continue

    if df.iloc[i]["low_above_ema"]:
        streak += 1
    elif df.iloc[i]["Close"] < df.iloc[i]["EMA21"]:
        streak = 0
        in_wait_mode = False

    if streak >= 3 and not position_open:
        if df.iloc[i]["up_day"]:
            df.at[df.index[i], "signal"] = 1
            position_open = True
            streak = 0
            in_wait_mode = False
        else:
            in_wait_mode = True
    elif in_wait_mode and df.iloc[i]["up_day"] and df.iloc[i]["low_above_ema"] and not position_open:
        df.at[df.index[i], "signal"] = 1
        position_open = True
        streak = 0
        in_wait_mode = False
    elif not df.iloc[i]["low_above_ema"] and df.iloc[i]["Close"] < df.iloc[i]["EMA21"]:
        streak = 0
        in_wait_mode = False

# === Step 5: Track Max High Between Buy and Sell ===
df["max_high_since_buy"] = pd.NA
df["max_high_pct_above_sell_thresh"] = pd.NA

last_buy_index = None
for i in range(len(df)):
    if df.iloc[i]["signal"] == 1:
        last_buy_index = i
    elif df.iloc[i]["signal"] == -1 and last_buy_index is not None:
        max_high = df.iloc[last_buy_index:i+1]["High"].max()
        df.at[df.index[i], "max_high_since_buy"] = round(max_high, 2)

        sell_thresh = df.iloc[i]["sell_threshold"]
        if pd.notna(sell_thresh) and sell_thresh != 0:
            pct_gain = ((max_high - sell_thresh) / sell_thresh) * 100
            df.at[df.index[i], "max_high_pct_above_sell_thresh"] = round(pct_gain, 2)

        last_buy_index = None

# === Step 6: Calculate ATR and Close + 2x/3x ATR for charting ===
tr = pd.concat([
    df["High"] - df["Low"],
    (df["High"] - df["Close"].shift(1)).abs(),
    (df["Low"] - df["Close"].shift(1)).abs()
], axis=1).max(axis=1)

df["ATR"] = tr.rolling(window=14).mean()
df["close_plus_2atr"] = df["Close"] + df["ATR"] * 2
df["close_plus_3atr"] = df["Close"] + df["ATR"] * 3

# === Step 7: Export All Columns ===
df_output = df[[
    "Open", "High", "Low", "Close", "EMA21", "sell_threshold",
    "low_above_ema", "up_day", "signal", "max_high_since_buy",
    "max_high_pct_above_sell_thresh", "close_plus_2atr", "close_plus_3atr"
]].round(2)

os.makedirs("output", exist_ok=True)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_output.to_excel(writer, sheet_name="Strategy", index=True)

print(f"✅ Strategy complete for {ticker}.")
print(f"📁 Saved to: {output_file}")