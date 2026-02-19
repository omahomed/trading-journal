"""
stock_daily_quotes_specific_date.py

Purpose:
- Read stock tickers from select_ticker.txt
- Fetch stock data for a specific date only
- Calculate technical indicators: 21 EMA, 50 SMA, 200 SMA
- Output all stocks in one table for the specified date
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# === USER CONFIGURATION ===
TARGET_DATE = "07.30.25"  # Format: MM.DD.YY

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def read_tickers(filename='select_ticker.txt'):
    """Read tickers from text file, one per line"""
    try:
        with open(filename, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        print(f"Error: {filename} not found. Creating sample file...")
        # Create sample file
        with open(filename, 'w') as f:
            f.write("AAPL\nMSFT\nGOOGL\nAMZN\nTSLA")
        print(f"Sample {filename} created. Please edit it with your tickers and run again.")
        return []

def get_data_for_date(ticker, target_date):
    """Fetch stock data for a specific date with technical indicators"""
    try:
        # Convert target_date string to datetime
        target_dt = pd.to_datetime(target_date)
        
        # We need historical data to calculate moving averages
        # Fetch 250 days before target date to ensure we have enough data for 200 SMA
        start_date = target_dt - timedelta(days=350)
        end_date = target_dt + timedelta(days=5)  # Few days after in case of weekends
        
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"No data found for {ticker}")
            return None
        
        # Calculate indicators
        df['EMA21'] = calculate_ema(df['Close'], 21)
        df['SMA50'] = calculate_sma(df['Close'], 50)
        df['SMA200'] = calculate_sma(df['Close'], 200)
        
        # Convert target_dt to be timezone-aware to match df.index
        if df.index.tz is not None:
            target_dt = target_dt.tz_localize(df.index.tz)
        
        # Find the exact date or closest trading date
        if target_dt in df.index:
            data = df.loc[target_dt]
        else:
            # Find closest date before target
            valid_dates = df.index[df.index <= target_dt]
            if len(valid_dates) == 0:
                print(f"No data available for {ticker} on or before {target_date}")
                return None
            closest_date = valid_dates[-1]
            data = df.loc[closest_date]
            print(f"Note: {ticker} - Using closest trading date: {closest_date.strftime('%Y-%m-%d')}")
        
        # Create result dictionary
        result = {
            'Ticker': ticker,
            'Date': data.name.strftime('%Y-%m-%d'),
            'Close': round(data['Close'], 2),
            'High': round(data['High'], 2),
            'Low': round(data['Low'], 2),
            'EMA21': round(data['EMA21'], 2) if not pd.isna(data['EMA21']) else 'N/A',
            'SMA50': round(data['SMA50'], 2) if not pd.isna(data['SMA50']) else 'N/A',
            'SMA200': round(data['SMA200'], 2) if not pd.isna(data['SMA200']) else 'N/A',
            'Volume': int(data.get('Volume', 0))
        }
        
        # Calculate percentage differences
        if result['EMA21'] != 'N/A':
            result['% from EMA21'] = round(((result['Close'] - result['EMA21']) / result['EMA21']) * 100, 2)
        else:
            result['% from EMA21'] = 'N/A'
            
        if result['SMA50'] != 'N/A':
            result['% from SMA50'] = round(((result['Close'] - result['SMA50']) / result['SMA50']) * 100, 2)
        else:
            result['% from SMA50'] = 'N/A'
            
        if result['SMA200'] != 'N/A':
            result['% from SMA200'] = round(((result['Close'] - result['SMA200']) / result['SMA200']) * 100, 2)
        else:
            result['% from SMA200'] = 'N/A'
        
        return result
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def main():
    """Main function to fetch quotes for specific date"""
    print("Stock Quotes for Specific Date")
    print("=" * 50)
    
    # Convert date format from MM.DD.YY to YYYY-MM-DD
    try:
        month, day, year = TARGET_DATE.split('.')
        # Assuming 20XX for the year
        year = '20' + year
        target_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # Validate date
        datetime.strptime(target_date, '%Y-%m-%d')
    except:
        print(f"Invalid date format in TARGET_DATE: {TARGET_DATE}")
        print("Please use MM.DD.YY format (e.g., 07.30.25)")
        return
    
    # Read tickers
    tickers = read_tickers()
    if not tickers:
        return
    
    print(f"\nFound {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"\nFetching data for: {TARGET_DATE} ({target_date})")
    print("-" * 50)
    
    # Collect data for all tickers
    all_data = []
    for ticker in tickers:
        print(f"Processing {ticker}...", end=' ')
        data = get_data_for_date(ticker, target_date)
        if data:
            all_data.append(data)
            print("✓")
        else:
            print("✗")
    
    if all_data:
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Display results
        print(f"\n\nStock Data for {TARGET_DATE}")
        print("=" * 120)
        
        # Display main data
        main_cols = ['Ticker', 'Date', 'Close', 'High', 'Low', 'EMA21', 'SMA50', 'SMA200', 'Volume']
        print("\nPrice and Moving Averages:")
        print(df[main_cols].to_string(index=False))
        
        # Display percentage analysis
        pct_cols = ['Ticker', '% from EMA21', '% from SMA50', '% from SMA200']
        print("\n\nPercentage Distance from Moving Averages:")
        print(df[pct_cols].to_string(index=False))
        
        # Save to CSV
        filename = f"stock_quotes_{target_date.replace('-', '')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n\nData saved to: {filename}")
        
        # Quick analysis
        print("\n\nQuick Analysis:")
        print("-" * 50)
        
        # Count stocks above/below key levels
        above_ema21 = df[(df['% from EMA21'] != 'N/A') & (df['% from EMA21'] > 0)]['Ticker'].tolist()
        below_ema21 = df[(df['% from EMA21'] != 'N/A') & (df['% from EMA21'] < 0)]['Ticker'].tolist()
        
        if above_ema21:
            print(f"Above EMA21 ({len(above_ema21)}): {', '.join(above_ema21)}")
        if below_ema21:
            print(f"Below EMA21 ({len(below_ema21)}): {', '.join(below_ema21)}")
        
        # Find strongest/weakest relative to SMA200
        sma200_data = df[df['% from SMA200'] != 'N/A'].copy()
        if not sma200_data.empty:
            strongest = sma200_data.nlargest(3, '% from SMA200')[['Ticker', '% from SMA200']]
            weakest = sma200_data.nsmallest(3, '% from SMA200')[['Ticker', '% from SMA200']]
            
            print(f"\nStrongest vs SMA200:")
            for _, row in strongest.iterrows():
                print(f"  {row['Ticker']}: +{row['% from SMA200']}%")
                
            print(f"\nWeakest vs SMA200:")
            for _, row in weakest.iterrows():
                print(f"  {row['Ticker']}: {row['% from SMA200']}%")
    
    else:
        print("\nNo data retrieved. Please check your date and ticker symbols.")

if __name__ == "__main__":
    main()