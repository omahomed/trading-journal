import pandas as pd
import warnings
import os

# === Setup ===
warnings.simplefilter(action="ignore", category=FutureWarning)
os.makedirs("output", exist_ok=True)

# === User Configuration ===
starting_balance = 15000
start_date = "2024-09-04"
end_date = "2025-09-05"

# === Load tickers from tickers.txt ===
with open("select_ticker.txt", "r") as f:
    tickers = [line.strip().upper() for line in f if line.strip()]

# === Summary Collector ===
summaries = []

# === Loop through tickers ===
for ticker in tickers:
    input_file = f"output/{ticker}_price_data.csv"

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"⚠️ File not found: {input_file}. Skipping {ticker}.")
        continue

    # === Filter date range ===
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    if df.empty:
        print(f"⚠️ No data in date range for {ticker}. Skipping.")
        continue

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    # === Indicators ===
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["sell_threshold"] = df["EMA21"] * 0.998
    df["low_above_ema"] = df["Low"] > df["EMA21"]
    df["up_day"] = df["Close"] > df["Close"].shift(1)

    tr = df[["High", "Low", "Close"]].copy()
    tr["prev_close"] = tr["Close"].shift(1)
    tr["tr"] = tr[["High", "prev_close"]].max(axis=1) - tr[["Low", "prev_close"]].min(axis=1)
    df["RangeMA"] = tr["tr"].rolling(window=10).mean()
    df["Upper_2ATR"] = df["EMA21"] + 2 * df["RangeMA"]
    df["Upper_3ATR"] = df["EMA21"] + 3 * df["RangeMA"]
    df["Upper_4ATR"] = df["EMA21"] + 4 * df["RangeMA"]

    df["pct_high_vs_3ATR"] = ((df["High"] - df["Upper_3ATR"]) / df["Upper_3ATR"]) * 100
    df["pct_high_vs_4ATR"] = ((df["High"] - df["Upper_4ATR"]) / df["Upper_4ATR"]) * 100

    df["signal"] = 0
    df["max_high_since_buy"] = None
    df["pct_above_sell_threshold"] = None

    # === Signal Logic ===
    streak = 0
    in_wait_mode = False
    position_open = False
    last_buy_index = None

    for i in range(2, len(df)):
        if position_open:
            if df.iloc[i]["Close"] < df.iloc[i]["sell_threshold"]:
                df.at[df.index[i], "signal"] = -1
                position_open = False
                in_wait_mode = False

                if last_buy_index is not None:
                    max_high = df.loc[df.index[last_buy_index]:df.index[i], "High"].max()
                    df.at[df.index[i], "max_high_since_buy"] = round(max_high, 2)
                    pct = ((max_high - df.iloc[i]["sell_threshold"]) / df.iloc[i]["sell_threshold"]) * 100
                    df.at[df.index[i], "pct_above_sell_threshold"] = round(pct, 2)

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
                last_buy_index = i
                streak = 0
                in_wait_mode = False
            else:
                in_wait_mode = True

        elif in_wait_mode and df.iloc[i]["up_day"] and df.iloc[i]["low_above_ema"] and not position_open:
            df.at[df.index[i], "signal"] = 1
            position_open = True
            last_buy_index = i
            streak = 0
            in_wait_mode = False

        elif not df.iloc[i]["low_above_ema"] and df.iloc[i]["Close"] < df.iloc[i]["EMA21"]:
            streak = 0
            in_wait_mode = False

    # === Simulate Trades ===
    balance = starting_balance
    buy_price = None
    shares = None
    buy_date = None
    trade_count = 0
    final_status = "Fully Closed"

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        signal = row["signal"]

        if signal == 1:
            buy_price = row["Close"]
            buy_date = date
            shares = int(balance // buy_price)
            invested = shares * buy_price
            balance = round(balance - invested, 2)
            trade_count += 1

        elif signal == -1 and buy_price is not None:
            sell_price = row["Close"]
            proceeds = round(shares * sell_price, 2)
            balance = round(balance + proceeds, 2)

            buy_price = None
            shares = None
            buy_date = None
            trade_count += 1

    if buy_price is not None:
        final_status = "Still in Position"
        last_close = df.iloc[-1]["Close"]
        proceeds = round(shares * last_close, 2)
        balance = round(balance + proceeds, 2)
        trade_count += 1

    percent_gain = round(((balance - starting_balance) / starting_balance) * 100, 2)

    summary_row = {
        "Ticker": ticker,
        "Start Date": pd.to_datetime(start_date).date(),
        "End Date": pd.to_datetime(end_date).date(),
        "Starting Balance": starting_balance,
        "Final Balance": round(balance, 2),
        "Percent Gain": percent_gain,
        "Total Trades": trade_count,
        "Status": final_status
    }

    summaries.append(summary_row)
    print(f"✅ {ticker} done | Final Balance: ${balance} | {final_status}")

# === Save Summary ===
summary_df = pd.DataFrame(summaries)
summary_df.to_excel("output/simulation_summary.xlsx", index=False)
print("\n📊 Master summary saved to: output/simulation_summary.xlsx")