# download_nasdaq_data.py
"""
Script to download NASDAQ Composite (^IXIC) price data
This will create the IXIC_price_data.csv file needed for Market School Rules
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_nasdaq_data():
    """Download NASDAQ data and save to CSV"""
    
    # Set date range (2 years of data to ensure we have enough history)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    print(f"Downloading NASDAQ data from {start_date.date()} to {end_date.date()}...")
    
    # Download data using yfinance
    # ^IXIC is the symbol for NASDAQ Composite
    nasdaq = yf.Ticker("^IXIC")
    data = nasdaq.history(start=start_date, end=end_date)
    
    # Check if data was downloaded
    if data.empty:
        print("Error: No data downloaded. Check your internet connection.")
        return False
    
    print(f"Downloaded {len(data)} days of data")
    
    # Prepare data in the format expected by Market School Rules
    # Reset index to make Date a column
    data.reset_index(inplace=True)
    
    # Ensure column names match what the script expects
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    
    # Keep only the columns we need
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Format date as string in YYYY-MM-DD format
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    filename = 'IXIC_price_data.csv'
    data.to_csv(filename, index=False)
    print(f"\nData saved to {filename}")
    
    # Show sample of data
    print("\nFirst few rows:")
    print(data.head())
    print("\nLast few rows:")
    print(data.tail())
    
    return True

def verify_data_file():
    """Verify the downloaded data file"""
    try:
        # Try to read the file
        df = pd.read_csv('IXIC_price_data.csv')
        print(f"\nVerification: File contains {len(df)} rows")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print("Columns:", list(df.columns))
        return True
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

if __name__ == "__main__":
    # Install yfinance if not already installed
    try:
        import yfinance
    except ImportError:
        print("Installing yfinance...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'yfinance'])
        import yfinance
    
    # Download the data
    if download_nasdaq_data():
        print("\n✓ Success! NASDAQ data downloaded.")
        
        # Verify the file
        if verify_data_file():
            print("\n✓ File verified and ready to use with Market School Rules")
            print("\nYou can now run: python test_market_rules.py")
        else:
            print("\n✗ File verification failed")
    else:
        print("\n✗ Download failed")