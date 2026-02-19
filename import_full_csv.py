import pandas as pd
import json
from datetime import datetime

# Read CSV from row 22 (0-indexed, so skiprows=22)
df = pd.read_csv('MO EC.csv', skiprows=22)

# Print first few column names to debug
print(f"📋 First 10 columns: {df.columns.tolist()[:10]}")

# The date column might be unnamed or have a different name
# Find the column that contains dates
date_col = None
for col in df.columns[:10]:  # Check first 10 columns
    if df[col].dtype == 'object':  # String column
        # Check if first value looks like a date
        first_val = str(df[col].iloc[0]) if len(df) > 0 else ''
        if '/' in first_val or 'Mon' in first_val or 'Tue' in first_val:
            date_col = col
            break

if date_col is None:
    # Try common names
    for name in ['Day', 'Date', 'day', 'date', '']:
        if name in df.columns:
            date_col = name
            break

if date_col is None:
    print("❌ Could not find date column!")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

print(f"✅ Using date column: '{date_col}'")

# Clean up - remove empty rows
df = df.dropna(subset=[date_col])

# Helper function to clean currency values
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

# Helper function to clean percentage values  
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

# Parse dates
def parse_date(date_str):
    try:
        dt = pd.to_datetime(date_str, format='%m/%d/%y')
        return dt.strftime('%Y-%m-%d')
    except:
        return None

# Constants
STARTING_CAPITAL = 100000

print(f"📊 Loaded {len(df)} rows from CSV")

# Helper to find column by partial name
def find_column(df, keywords):
    """Find column that contains any of the keywords"""
    for col in df.columns:
        col_lower = str(col).lower()
        for keyword in keywords:
            if keyword.lower() in col_lower:
                return col
    return None

# Find key columns
beg_nlv_col = find_column(df, ['beg nlv', 'begin nlv', 'beginning nlv'])
end_nlv_col = find_column(df, ['end nlv', 'ending nlv'])
cash_col = find_column(df, ['cash', '+/-'])
invested_col = find_column(df, ['% invested', 'invested'])
ltd_pct_col = find_column(df, ['ltd', '% change'])
daily_pct_col = find_column(df, ['daily', '% change'])
ytd_pct_col = find_column(df, ['ytd', '% change'])

print(f"✅ Found columns:")
print(f"   Beg NLV: '{beg_nlv_col}'")
print(f"   End NLV: '{end_nlv_col}'")
print(f"   Cash: '{cash_col}'")

# Process equity data
equity_data = []

for idx, row in df.iterrows():
    date_parsed = parse_date(row[date_col])
    if not date_parsed:
        continue
    
    # Get values from CSV using found columns
    beg_nlv = clean_currency(row[beg_nlv_col]) if beg_nlv_col else 0
    end_nlv = clean_currency(row[end_nlv_col]) if end_nlv_col else 0
    cash_flow = clean_currency(row[cash_col]) if cash_col else 0
    pct_invested = clean_percent(row[invested_col]) if invested_col else 100
    
    # Skip if no data
    if end_nlv == 0:
        continue
    
    # Get LTD % from CSV - find column with "LTD" and "% Change"
    ltd_pct = 0
    for col in df.columns:
        if 'LTD' in str(col) and '% Change' in str(col):
            ltd_pct = clean_percent(row[col])
            break
    
    # Get daily % from CSV - find column with "Daily" and "% Change"
    daily_pct = 0
    for col in df.columns:
        if 'Daily' in str(col) and '% Change' in str(col):
            daily_pct = clean_percent(row[col])
            break
    
    # Get YTD % from CSV - find column with "YTD" and "% Change"
    ytd_pct = 0
    for col in df.columns:
        if 'YTD' in str(col) and '% Change' in str(col):
            ytd_pct = clean_percent(row[col])
            break
    
    # Calculate daily dollar
    if cash_flow != 0:
        daily_dollar = end_nlv - (beg_nlv + cash_flow)
    else:
        daily_dollar = end_nlv - beg_nlv
    
    # LTD dollar
    ltd_dollar = end_nlv - STARTING_CAPITAL
    
    # Get moving averages from CSV (already calculated)
    ma_10 = None
    ma_21 = None
    ma_50 = None
    
    for col in df.columns:
        if '10SMA%' in str(col) or '10 SMA' in str(col):
            ma_10 = clean_percent(row[col])
        elif '21SMA%' in str(col) or '21 SMA' in str(col):
            ma_21 = clean_percent(row[col])
        elif '50SMA%' in str(col) or '50 SMA' in str(col):
            ma_50 = clean_percent(row[col])
    
    # Convert 0 to None for JSON
    ma_10 = None if pd.isna(ma_10) or ma_10 == 0 else ma_10
    ma_21 = None if pd.isna(ma_21) or ma_21 == 0 else ma_21
    ma_50 = None if pd.isna(ma_50) or ma_50 == 0 else ma_50
    
    entry = {
        "date": date_parsed,
        "beg_nlv": beg_nlv,
        "end_nlv": end_nlv,
        "cash_flow": cash_flow,
        "pct_invested": pct_invested,
        "daily_dollar": daily_dollar,
        "daily_pct": daily_pct,
        "ltd_dollar": ltd_dollar,
        "ltd_pct": ltd_pct,
        "ytd_pct": ytd_pct,
        "ma_10": ma_10,
        "ma_21": ma_21,
        "ma_50": ma_50
    }
    
    equity_data.append(entry)

# Process market data
market_data = []

for idx, row in df.iterrows():
    date_parsed = parse_date(row[date_col])
    if not date_parsed:
        continue
    
    # Get SPY and NASDAQ data - search for columns flexibly
    spy_close = 0
    ndx_close = 0
    spy_ltd = 0
    ndx_ltd = 0
    
    for col in df.columns:
        col_str = str(col).upper()
        if 'SPY' in col_str and 'LTD' not in col_str and 'YTD' not in col_str:
            spy_close = clean_currency(row[col])
        elif 'NSADAQ' in col_str or 'NASDAQ' in col_str:
            if 'LTD' not in col_str and 'YTD' not in col_str:
                ndx_close = clean_currency(row[col])
        elif 'SPY' in col_str and 'LTD' in col_str:
            spy_ltd = clean_percent(row[col])
        elif ('NSADAQ' in col_str or 'NDX' in col_str) and 'LTD' in col_str:
            ndx_ltd = clean_percent(row[col])
    
    if spy_close > 0 or ndx_close > 0:
        entry = {
            "date": date_parsed,
            "spy_close": spy_close,
            "ndx_close": ndx_close,
            "spy_ltd_pct": spy_ltd,
            "ndx_ltd_pct": ndx_ltd
        }
        market_data.append(entry)

print(f"\n✅ Processed {len(equity_data)} equity entries")
print(f"✅ Processed {len(market_data)} market entries")

# Save to JSON
with open('equity_data.json', 'w') as f:
    json.dump(equity_data, f, indent=2)
print("\n✅ Saved equity_data.json")

with open('market_data.json', 'w') as f:
    json.dump(market_data, f, indent=2)
print("✅ Saved market_data.json")

# Show samples
print("\n📊 First 3 entries:")
for entry in equity_data[:3]:
    print(f"  {entry['date']}: NLV=${entry['end_nlv']:,.0f}, LTD%={entry['ltd_pct']:.2f}%, 10SMA={entry['ma_10']}")

print("\n📊 Last 3 entries:")
for entry in equity_data[-3:]:
    print(f"  {entry['date']}: NLV=${entry['end_nlv']:,.0f}, LTD%={entry['ltd_pct']:.2f}%, 10SMA={entry['ma_10']}")

print("\n📊 Market data sample:")
for entry in market_data[-3:]:
    print(f"  {entry['date']}: SPY LTD%={entry['spy_ltd_pct']:.2f}%, NDX LTD%={entry['ndx_ltd_pct']:.2f}%")

print("\n✅ Import complete!")
print("🔄 Now restart your Streamlit app to see the data!")
