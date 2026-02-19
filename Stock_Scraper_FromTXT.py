"""
stock_scraper.py

Purpose:
- Download historical stock price data (Open, High, Low, Close, Volume) for multiple tickers
- Ticker symbols are read from 'tickers.txt'
- Save each ticker's cleaned data as a CSV in the 'output/' directory
"""

import yfinance as yf
import pandas as pd
import os

# === Configuration ===
tickers_file = "select_ticker.txt"
output_dir = "output"
start_date = "1975-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read tickers from file
try:
    with open(tickers_file, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
except FileNotFoundError:
    print(f"❌ Error: '{tickers_file}' not found.")
    exit(1)

# === Download and save each ticker ===
for symbol in tickers:
    print(f"📥 Fetching data for {symbol}...")

    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)

        if df.empty:
            print(f"⚠️ No data for {symbol}. Skipping.")
            continue

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

        safe_symbol = symbol.replace("^", "")  # Sanitize filename
        output_path = os.path.join(output_dir, f"{safe_symbol}_price_data.csv")
        df.to_csv(output_path, index=False)

        print(f"✅ Saved: {output_path}")

    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")