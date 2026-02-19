import pandas as pd
import warnings
import os

# === Setup ===
warnings.simplefilter(action="ignore", category=FutureWarning)
os.makedirs("output", exist_ok=True)

# === User Configuration ===
tickers = ["TSLA", "HIMS", "HOOD", "PLTR","NVDA"]
starting_balance = 67000
start_date = "2020-04-01"

# === Summary Collector ===
summaries = []

# === Loop through tickers ===
for ticker in tickers:
    input_file = f"output/{ticker}_price_data.csv"
    output_file = f"output/{ticker}_strategy_results.xlsx"

    # Step 1: Load and clean data
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"⚠️ File not found: {input_file}. Skipping {ticker}.")
        continue

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df[df.index >= start_date]

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    # Step 2: Compute indicators
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["sell_threshold"] = df["EMA21"] * 0.998
    df["low_above_ema"] = df["Low"] > df["EMA21"]
    df["up_day"] = df["Close"] > df["Close"].shift(1)

    # Keltner Channel Bands
    tr = df[["High", "Low", "Close"]].copy()
    tr["prev_close"] = tr["Close"].shift(1)
    tr["tr"] = tr[["High", "prev_close"]].max(axis=1) - tr[["Low", "prev_close"]].min(axis=1)
    df["RangeMA"] = tr["tr"].rolling(window=10).mean()
    df["Upper_2ATR"] = df["EMA21"] + 2 * df["RangeMA"]
    df["Upper_3ATR"] = df["EMA21"] + 3 * df["RangeMA"]
    df["Upper_4ATR"] = df["EMA21"] + 4 * df["RangeMA"]
    df["pct_high_vs_3ATR"] = ((df["High"] - df["Upper_3ATR"]) / df["Upper_3ATR"]) * 100
    df["pct_high_vs_4ATR"] = ((df["High"] - df["Upper_4ATR"]) / df["Upper_4ATR"]) * 100

    # Initialize tracking columns
    df["signal"] = 0
    df["max_high_since_buy"] = None
    df["pct_above_sell_threshold"] = None

    # Step 3: Buy/Sell Signal Logic with staged entry
    position_state = "none"  # can be "none", "partial", or "full"
    streak = 0
    in_wait_mode = False
    last_sell_index = -1
    first_crossed_above_ema21 = False
    partial_buy_index = None
    full_buy_index = None

    for i in range(2, len(df)):
        close = df.iloc[i]["Close"]
        ema21 = df.iloc[i]["EMA21"]

        # === Sell condition applies to any position state
        if position_state in ("partial", "full") and close < df.iloc[i]["sell_threshold"]:
            df.at[df.index[i], "signal"] = -1
            position_state = "none"
            streak = 0
            in_wait_mode = False
            last_sell_index = i
            first_crossed_above_ema21 = False
            partial_buy_index = None
            full_buy_index = None
            continue

        # === Track first close > EMA21 after a sell
        if position_state == "none" and not first_crossed_above_ema21 and close > ema21:
            df.at[df.index[i], "signal"] = 0.25  # flag partial buy
            first_crossed_above_ema21 = True
            position_state = "partial"
            partial_buy_index = i
            continue

        # === Track streak for full buy
        if df.iloc[i]["low_above_ema"]:
            streak += 1
        elif close < ema21:
            streak = 0
            in_wait_mode = False

        # === Full buy condition after partial buy
        if streak >= 3 and position_state == "partial":
            if df.iloc[i]["up_day"]:
                df.at[df.index[i], "signal"] = 0.75  # flag full add
                position_state = "full"
                full_buy_index = i
                streak = 0
                in_wait_mode = False
                continue
            else:
                in_wait_mode = True

        elif in_wait_mode and df.iloc[i]["up_day"] and df.iloc[i]["low_above_ema"] and position_state == "partial":
            df.at[df.index[i], "signal"] = 0.75
            position_state = "full"
            full_buy_index = i
            streak = 0
            in_wait_mode = False

        elif not df.iloc[i]["low_above_ema"] and close < ema21:
            streak = 0
            in_wait_mode = False

    # Step 4: Simulate Trades
    balance = starting_balance
    trade_count = 0
    position_shares = 0
    buy_price = 0
    total_shares = 0
    position_entry_index = None
    trades = []
    final_status = "Fully Closed"

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        signal = row["signal"]
        price = row["Close"]

        # === Partial Buy
        if signal == 0.25:
            buy_price = price
            shares = int((0.25 * balance) // buy_price)
            invested = shares * buy_price
            balance -= invested
            position_shares = shares
            total_shares = shares
            position_entry_index = i
            start_balance_snapshot = balance + invested
            trade_count += 1

        # === Full Add
        elif signal == 0.75:
            buy_price = price
            shares = int((0.75 * balance) // buy_price)
            invested = shares * buy_price
            balance -= invested
            position_shares += shares
            total_shares += shares
            trade_count += 1

        # === Sell
        elif signal == -1 and position_shares > 0:
            sell_price = price
            proceeds = sell_price * position_shares
            balance += proceeds
            max_high = df.loc[df.index[position_entry_index]:date, "High"].max()
            pct_above = ((max_high - row["sell_threshold"]) / row["sell_threshold"]) * 100

            trades.append({
                "Buy Date": df.index[position_entry_index].date(),
                "Buy Price": round(df.iloc[position_entry_index]["Close"], 2),
                "Shares": total_shares,
                "Sell Date": date.date(),
                "Sell Price": round(sell_price, 2),
                "Starting Balance": round(start_balance_snapshot, 2),
                "Cash After Trade": round(balance, 2),
                "Max High": round(max_high, 2),
                "Pct Above Sell Threshold": round(pct_above, 2)
            })

            # Reset position
            position_shares = 0
            total_shares = 0
            position_entry_index = None
            buy_price = 0
            trade_count += 1

    # === Handle open position at end of data
    if position_shares > 0:
        final_status = "Still in Position"
        last_close = df.iloc[-1]["Close"]
        proceeds = last_close * position_shares
        balance += proceeds
        trades.append({
            "Buy Date": df.index[position_entry_index].date(),
            "Buy Price": round(df.iloc[position_entry_index]["Close"], 2),
            "Shares": total_shares,
            "Sell Date": df.index[-1].date(),
            "Sell Price": round(last_close, 2),
            "Starting Balance": round(start_balance_snapshot, 2),
            "Cash After Trade": round(balance, 2),
            "Note": "Still in position"
        })

    # Step 5: Prepare output
    df_output = df[[
        "Open", "High", "Low", "Close", "EMA21", "sell_threshold",
        "low_above_ema", "up_day", "signal", "max_high_since_buy",
        "pct_above_sell_threshold", "Upper_2ATR", "Upper_3ATR", "Upper_4ATR",
        "pct_high_vs_3ATR", "pct_high_vs_4ATR"
    ]].round(2)

    percent_gain = round(((balance - starting_balance) / starting_balance) * 100, 2)

    summary_row = {
        "Ticker": ticker,
        "Start Date": pd.to_datetime(start_date).date(),
        "End Date": df.index[-1].date(),
        "Starting Balance": starting_balance,
        "Final Balance": round(balance, 2),
        "Percent Gain": percent_gain,
        "Total Trades": trade_count,
        "Status": final_status
    }

    summaries.append(summary_row)

    # Step 6: Save Excel
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_output.to_excel(writer, sheet_name="Strategy", index=True)
        pd.DataFrame(trades).to_excel(writer, sheet_name="Trades", index=False)
        pd.DataFrame([summary_row]).to_excel(writer, sheet_name="Summary", index=False)

    print(f"✅ {ticker} strategy complete. Saved to: {output_file}")

# Step 7: Master summary
summary_df = pd.DataFrame(summaries)
summary_df.to_excel("output/simulation_summary.xlsx", index=False)
print("\n📊 Master summary saved to: output/simulation_summary.xlsx")