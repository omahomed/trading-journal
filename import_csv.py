import pandas as pd
import json
from datetime import datetime

# Read the CSV file (skip first 12 rows which are headers/summary)
df = pd.read_csv('MO EC.csv', skiprows=12)

# Clean up the data - remove empty rows
df = df.dropna(subset=['Day'])

# Function to clean currency values
def clean_currency(val):
    if pd.isna(val) or val == '':
        return 0.0
    val_str = str(val).replace('$', '').replace(',', '').replace(' ', '')
    val_str = val_str.replace('(', '-').replace(')', '')
    if val_str == '-' or val_str == '':
        return 0.0
    try:
        return float(val_str)
    except:
        return 0.0

# Function to clean percentage values
def clean_percent(val):
    if pd.isna(val) or val == '':
        return 0.0
    val_str = str(val).replace('%', '').replace(' ', '')
    if val_str == '-' or val_str == '':
        return 0.0
    try:
        return float(val_str)
    except:
        return 0.0

# Function to parse dates
def parse_date(date_str):
    try:
        # Handle format like "9/2/24"
        dt = pd.to_datetime(date_str, format='%m/%d/%y')
        return dt.strftime('%Y-%m-%d')
    except:
        return None

print("Processing CSV data...")
print(f"Total rows in CSV: {len(df)}")

# Prepare equity data
equity_data = []
market_data = []

for idx, row in df.iterrows():
    date_parsed = parse_date(row['Day'])
    if not date_parsed:
        continue
    
    # Extract and clean equity values
    beg_nlv = clean_currency(row.get(' Beg NLV ', 0))
    end_nlv = clean_currency(row.get(' End NLV ', 0))
    cash_flow = clean_currency(row.get(' Cash -/+ ', 0))
    
    # Skip rows with no actual data
    if end_nlv == 0:
        continue
    
    # Get percentages (they're in separate columns due to CSV structure)
    daily_pct_col = [c for c in df.columns if 'Daily' in c and '% Change' in c]
    ltd_pct_col = [c for c in df.columns if 'LTD' in c and '% Change' in c]
    ytd_pct_col = [c for c in df.columns if 'YTD' in c and '% Change' in c]
    
    daily_pct = clean_percent(row[daily_pct_col[0]]) if daily_pct_col else 0
    ltd_pct = clean_percent(row[ltd_pct_col[0]]) if ltd_pct_col else 0
    ytd_pct = clean_percent(row[ytd_pct_col[0]]) if ytd_pct_col else 0
    
    # Calculate daily dollar change
    if cash_flow != 0:
        daily_dollar = end_nlv - (beg_nlv + cash_flow)
    else:
        daily_dollar = end_nlv - beg_nlv
    
    # LTD dollar
    ltd_dollar = end_nlv - 100000  # Starting capital
    
    # % Invested
    pct_invested = clean_percent(row.get('% Invested', 0))
    
    equity_entry = {
        "date": date_parsed,
        "beg_nlv": beg_nlv,
        "end_nlv": end_nlv,
        "cash_flow": cash_flow,
        "pct_invested": pct_invested,
        "daily_dollar": daily_dollar,
        "daily_pct": daily_pct,
        "ltd_dollar": ltd_dollar,
        "ltd_pct": ltd_pct,
        "ytd_pct": ytd_pct
    }
    equity_data.append(equity_entry)
    
    # Extract market data
    spy_close = clean_currency(row.get(' SPY ', 0))
    ndx_close = clean_currency(row.get(' Nsadaq ', 0))
    spy_ytd = clean_percent(row.get('SPY YTD %', 0))
    ndx_ytd = clean_percent(row.get(' NDX YTD % ', 0))
    
    if spy_close > 0 or ndx_close > 0:
        market_entry = {
            "date": date_parsed,
            "spy_close": spy_close,
            "ndx_close": ndx_close,
            "spy_ytd": spy_ytd,
            "ndx_ytd": ndx_ytd
        }
        market_data.append(market_entry)

# Save to JSON files
print(f"\nProcessed {len(equity_data)} equity entries")
print(f"Processed {len(market_data)} market entries")

with open('equity_data.json', 'w') as f:
    json.dump(equity_data, f, indent=2)
print("\n✅ Saved equity_data.json")

with open('market_data.json', 'w') as f:
    json.dump(market_data, f, indent=2)
print("✅ Saved market_data.json")

# Display sample of imported data
print("\n📊 Sample of first 3 equity entries:")
for entry in equity_data[:3]:
    print(f"\nDate: {entry['date']}")
    print(f"  End NLV: ${entry['end_nlv']:,.2f}")
    print(f"  Daily %: {entry['daily_pct']:.2f}%")
    print(f"  YTD %: {entry['ytd_pct']:.2f}%")

print("\n📊 Sample of last 3 equity entries:")
for entry in equity_data[-3:]:
    print(f"\nDate: {entry['date']}")
    print(f"  End NLV: ${entry['end_nlv']:,.2f}")
    print(f"  Daily %: {entry['daily_pct']:.2f}%")
    print(f"  YTD %: {entry['ytd_pct']:.2f}%")

print("\n✅ Import complete! You can now run your Streamlit app.")
print("Run: python3 -m streamlit run trading_scorecard.py")
