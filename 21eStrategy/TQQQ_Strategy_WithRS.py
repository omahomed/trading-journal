import pandas as pd
import numpy as np
import warnings
import os

# === Setup ===
warnings.simplefilter(action="ignore", category=FutureWarning)
os.makedirs("output", exist_ok=True)

# === User Configuration ===
starting_balance = 83000
start_date = "2009-06-25" 
end_date = "2026-12-31" 
signal_ticker = "^IXIC" # Restored the ^ to match your standard
trade_ticker = "TQQQ"
benchmark_ticker = "SPY"

output_file = "output/TQQQ_Strategy_with_RS_Filter_v2.xlsx"

# === Data Loading ===
def load_data(ticker):
    # Adjusted to look inside the 'output' folder
    file_name = f"{ticker.replace('^', '')}_price_data.csv"
    file_path = os.path.join("output", file_name)
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.set_index("Date", inplace=True)
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["Close"])
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None

ixic_raw = load_data(signal_ticker)
tqqq_raw = load_data(trade_ticker)
spy_raw = load_data(benchmark_ticker)

if ixic_raw is None or tqqq_raw is None or spy_raw is None:
    print("Critical error: Data files missing in the 'output' folder.")
else:
    # Sync Logic
    all_dates = ixic_raw.index.union(spy_raw.index).union(tqqq_raw.index)
    master_df = pd.DataFrame(index=all_dates).sort_index()
    master_df['IXIC_Close'] = ixic_raw['Close']
    master_df['IXIC_Low'] = ixic_raw['Low']
    master_df['IXIC_High'] = ixic_raw['High']
    master_df['SPY_Close'] = spy_raw['Close']
    master_df['TQQQ_Close'] = tqqq_raw['Close']
    
    master_df = master_df.ffill().dropna(subset=['TQQQ_Close', 'IXIC_Close']) 
    ixic = master_df[(master_df.index >= start_date) & (master_df.index <= end_date)].copy()

    # === Calculations ===
    ixic["EMA21"] = ixic["IXIC_Close"].ewm(span=21, adjust=False).mean()
    ixic["low_above_ema"] = ixic["IXIC_Low"] > ixic["EMA21"]
    ixic["up_day"] = ixic["IXIC_Close"] > ixic["IXIC_Close"].shift(1)
    
    ixic["RS_Line"] = ixic['IXIC_Close'] / ixic['SPY_Close']
    ixic["RS_SMA21"] = ixic["RS_Line"].rolling(window=21).mean()
    
    # LIGHT RS FILTER: Only requires RS Line > RS_SMA21
    ixic["RS_Bullish"] = (ixic["RS_Line"] > ixic["RS_SMA21"])

    # RS Rating
    ixic["RS_Mom"] = ixic["RS_Line"] / ixic["RS_Line"].shift(252)
    ixic["RS_Rating"] = ixic["RS_Mom"].rolling(window=252).apply(
        lambda x: (x.rank(pct=True).iloc[-1] * 98) + 1 if not x.isna().all() else np.nan
    )

    # === Simulation ===
    balance = starting_balance
    position_open = False
    streak, in_wait_mode = 0, False
    sell_triggered, sell_trigger_level = False, None
    trades = []
    
    buy_date, buy_price, shares, start_bal_snapshot = None, 0.0, 0, 0.0
    row_at_buy = None

    for i in range(len(ixic)):
        date = ixic.index[i]
        row = ixic.iloc[i]

        if position_open:
            if not sell_triggered and row["IXIC_Close"] < row["EMA21"]:
                sell_triggered = True
                sell_trigger_level = row["IXIC_Low"] * 0.998
            elif sell_triggered and row["IXIC_Close"] > row["EMA21"]:
                sell_triggered = False
            elif sell_triggered and row["IXIC_Low"] < sell_trigger_level:
                sell_price = row["TQQQ_Close"]
                proceeds = shares * sell_price
                balance += proceeds

                trades.append({
                    "Buy Date": buy_date.date(), "Sell Date": date.date(),
                    "Holding Days": (date - buy_date).days, "Buy Price": round(buy_price, 2),
                    "Sell Price": round(sell_price, 2), "Initial Shares": int(shares),
                    "Final Shares": int(shares), 
                    "Return %": (sell_price - buy_price) / buy_price,
                    "Cumulative Return %": (balance - starting_balance) / starting_balance,
                    "Profit/Loss": round(proceeds - (shares * buy_price), 2),
                    "Beginning Balance": round(start_bal_snapshot, 2), "Balance After": round(balance, 2),
                    "RS Rating @ Entry": round(row_at_buy["RS_Rating"], 0) if (row_at_buy is not None and not pd.isna(row_at_buy["RS_Rating"])) else "N/A",
                    "Status": "Closed"
                })
                position_open, sell_triggered, streak, in_wait_mode = False, False, 0, False
        else:
            if row["low_above_ema"]: streak += 1
            else: streak, in_wait_mode = 0, False

            can_buy = (streak >= 3 and row["up_day"]) or (in_wait_mode and row["up_day"] and row["low_above_ema"])
            if streak >= 3 and not row["up_day"]: in_wait_mode = True

            if can_buy and row["RS_Bullish"] and not pd.isna(row["TQQQ_Close"]):
                buy_date, buy_price = date, row["TQQQ_Close"]
                start_bal_snapshot = balance
                shares = int(balance // buy_price)
                balance -= (shares * buy_price)
                row_at_buy = row
                position_open = True

    # Handle Open Position
    if position_open:
        last_row = ixic.iloc[-1]
        current_val = shares * last_row["TQQQ_Close"]
        bal_with_open = balance + current_val
        trades.append({
            "Buy Date": buy_date.date(), "Sell Date": ixic.index[-1].date(),
            "Holding Days": (ixic.index[-1] - buy_date).days, "Buy Price": round(buy_price, 2),
            "Sell Price": round(last_row["TQQQ_Close"], 2), "Initial Shares": int(shares),
            "Final Shares": int(shares), 
            "Return %": (last_row["TQQQ_Close"] - buy_price) / buy_price,
            "Cumulative Return %": (bal_with_open - starting_balance) / starting_balance,
            "Profit/Loss": round(current_val - (shares * buy_price), 2),
            "Beginning Balance": round(start_bal_snapshot, 2), "Balance After": round(bal_with_open, 2),
            "RS Rating @ Entry": round(row_at_buy["RS_Rating"], 0) if not pd.isna(row_at_buy["RS_Rating"]) else "N/A",
            "Status": "OPEN"
        })

    with pd.ExcelWriter(output_file) as writer:
        pd.DataFrame(trades).to_excel(writer, sheet_name="Detailed Trades", index=False)
        ixic.round(4).to_excel(writer, sheet_name="Strategy Data")

    print(f"\nFinal Account Value: ${balance + (shares * ixic.iloc[-1]['TQQQ_Close'] if position_open else 0):,.2f}")
    print(f"Cumulative Return: {((balance + (shares * ixic.iloc[-1]['TQQQ_Close'] if position_open else 0) - starting_balance) / starting_balance):.2%}")