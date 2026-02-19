"""
simulate_strategy_trades.py

Purpose:
- Load historical stock data (Open, High, Low, Close) from CSV
- Apply a trend-following strategy based on EMA21 and signal rules
- Simulate trades using $1,000 starting on `start_date`, with compounding
- Generate a trade log with Buy/Sell data and account balance after each trade
- Export both signal sheet and trades to Excel
"""

import pandas as pd
import warnings
import os

# === User Configuration ===
ticker = "TSLA"
starting_balance = 67000              # 💰 Initial investment
start_date = "2020-04-01"            # 🗓️ Start trading from this date

# === File Paths ===
input_file = f"output/{ticker}_price_data.csv"
output_file = f"output/{ticker}_strategy_results.xlsx"

# === Suppress Warnings ===
warnings.simplefilter(action="ignore", category=FutureWarning)

# === Load CSV ===
df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df[df.index >= pd.to_datetime(start_date)]  # Filter data from start date

# === Ensure Required Columns Exist and Clean ===
for col in ["Open", "High", "Low", "Close"]:
    if col not in df.columns:
        raise ValueError(f"❌ Missing required column: {col}")
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

# === Compute Indicators ===
df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
df["sell_threshold"] = df["EMA21"] * 0.998
df["low_above_ema"] = df["Low"] > df["EMA21"]
df["up_day"] = df["Close"] > df["Close"].shift(1)
df["signal"] = 0

# === Apply Strategy Logic ===
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

# === Max High and % Above Sell Threshold ===
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

# === ATR and Close + 2x/3x ATR Bands ===
tr = pd.concat([
    df["High"] - df["Low"],
    (df["High"] - df["Close"].shift(1)).abs(),
    (df["Low"] - df["Close"].shift(1)).abs()
], axis=1).max(axis=1)
df["ATR"] = tr.rolling(window=14).mean()
df["close_plus_2atr"] = df["Close"] + df["ATR"] * 2
df["close_plus_3atr"] = df["Close"] + df["ATR"] * 3

# === Trade Simulation with $1,000 Start + Compounding ===
balance = starting_balance
trades = []
buy_price = None
buy_date = None
shares = None
starting_balance_snapshot = None

for i in range(len(df)):
    row = df.iloc[i]
    date = df.index[i]
    signal = row["signal"]

    if signal == 1:
        buy_price = row["Close"]
        buy_date = date
        starting_balance_snapshot = balance
        shares = int(balance // buy_price)
        invested = shares * buy_price
        balance = round(balance - invested, 2)

    elif signal == -1 and buy_price is not None:
        sell_price = row["Close"]
        sell_date = date
        proceeds = round(shares * sell_price, 2)
        balance = round(balance + proceeds, 2)
        trades.append({
            "Buy Date": buy_date.date(),
            "Buy Price": round(buy_price, 2),
            "Shares": shares,
            "Sell Date": sell_date.date(),
            "Sell Price": round(sell_price, 2),
            "Starting Balance": round(starting_balance_snapshot, 2),
            "Cash After Trade": balance
        })
        buy_price = None
        buy_date = None
        shares = None

df_trades = pd.DataFrame(trades)

# === Export to Excel ===
df_output = df[[
    "Open", "High", "Low", "Close", "EMA21", "sell_threshold",
    "low_above_ema", "up_day", "signal", "max_high_since_buy",
    "max_high_pct_above_sell_thresh", "close_plus_2atr", "close_plus_3atr"
]].round(2)

os.makedirs("output", exist_ok=True)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_output.to_excel(writer, sheet_name="Strategy", index=True)
    if not df_trades.empty:
        df_trades.to_excel(writer, sheet_name="Trades", index=False)

print(f"✅ Strategy simulation complete for {ticker} (starting from {start_date})")
print(f"📁 Results saved to: {output_file}")