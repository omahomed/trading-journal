import pandas as pd
import json
from datetime import datetime
import re

# Constants
STARTING_CAPITAL = 100000

# Cash flow events (date: amount)
CASH_FLOWS = {
    '2025-07-01': 45000,
    '2025-09-12': 4404,
    '2025-09-19': 6896
}

# Read CSV from row 22 (where data starts)
df = pd.read_csv('MO EC.csv', skiprows=22)

print(f"📊 Loaded {len(df)} rows from CSV")
print(f"📋 Total columns: {len(df.columns)}")

# Define column indices based on the debug output
# From your CSV structure:
COL_DATE = 4        # '9/13/24'
COL_BEG_NLV = 7     # ' $118,467.35 '
COL_END_NLV = 8     # ' $118,592.33 '
COL_DAILY_DOLLAR = 9  # ' $124.98 '
COL_DAILY_PCT = 10    # '0.11%'
COL_LTD_DOLLAR = 11   # ' $18,592.33 '
COL_LTD_PCT = 12      # '18.59%'
# Moving averages appear to be around columns 13-21
COL_10SMA_PCT = 18    # 10SMA %
COL_21SMA_PCT = 19    # 21SMA %
COL_50SMA_PCT = 20    # 50SMA %
# Market data
COL_PCT_INVESTED = 27  # % Invested
COL_SPY = 28          # SPY
COL_SPY_LTD = 29      # SPY LTD %
COL_SPY_YTD = 30      # SPY YTD %
COL_NASDAQ = 31       # Nasdaq
COL_NDX_LTD = 32      # NDX LTD %
COL_NDX_YTD = 33      # NDX YTD %

def clean_currency(val):
    """Convert '$118,467.35' or '$(981.49)' to float"""
    if pd.isna(val) or val == '':
        return 0.0
    
    val_str = str(val).strip()
    # Remove currency symbols and commas
    val_str = val_str.replace('$', '').replace(',', '').strip()
    
    # Handle parentheses (negative numbers)
    if '(' in val_str and ')' in val_str:
        val_str = val_str.replace('(', '').replace(')', '')
        return -float(val_str)
    
    try:
        return float(val_str)
    except:
        return 0.0

def clean_percent(val):
    """Convert '18.59%' to 18.59"""
    if pd.isna(val) or val == '':
        return None
    
    val_str = str(val).strip().replace('%', '')
    try:
        return float(val_str)
    except:
        return None

# Filter out rows with invalid dates (keep only actual data rows)
df_clean = df[df.iloc[:, COL_DATE].notna()].copy()

print(f"✅ Found {len(df_clean)} valid data rows")

# Initialize data structures
equity_data = []
market_data = []

# Manually add missing Sept 2-13, 2024 data that's not in the CSV
missing_data = [
    {'date': '2024-09-02', 'beg': 100000.00, 'end': 100000.00},
    {'date': '2024-09-03', 'beg': 100000.00, 'end': 100000.00},
    {'date': '2024-09-04', 'beg': 100000.00, 'end': 113420.72},
    {'date': '2024-09-05', 'beg': 113420.72, 'end': 113969.28},
    {'date': '2024-09-06', 'beg': 113969.28, 'end': 111857.56},
    {'date': '2024-09-09', 'beg': 111857.56, 'end': 114232.75},
    {'date': '2024-09-10', 'beg': 114232.75, 'end': 114708.92},
    {'date': '2024-09-11', 'beg': 114708.92, 'end': 117106.59},
    {'date': '2024-09-12', 'beg': 117106.59, 'end': 118467.35},
    {'date': '2024-09-13', 'beg': 118467.35, 'end': 118592.33},
]

print(f"📊 Adding {len(missing_data)} missing entries (Sept 2-13, 2024)")

# Process missing data first
previous_ytd = 0.0
previous_ltd = 0.0
current_year = 2024

for entry in missing_data:
    beg_nlv = entry['beg']
    end_nlv = entry['end']
    cash_flow = 0.0
    
    # Calculate daily change
    daily_dollar = end_nlv - beg_nlv
    daily_pct = (daily_dollar / beg_nlv) * 100 if beg_nlv != 0 else 0
    
    # Calculate LTD % using compound formula
    if len(equity_data) == 0:
        ltd_pct = daily_pct
    else:
        ltd_pct = ((1 + previous_ltd/100) * (1 + daily_pct/100) - 1) * 100
    
    # Calculate YTD % (same as LTD for 2024)
    ytd_pct = ((1 + previous_ytd/100) * (1 + daily_pct/100) - 1) * 100
    
    # Update for next iteration
    previous_ytd = ytd_pct
    previous_ltd = ltd_pct
    
    # Calculate LTD dollar
    ltd_dollar = end_nlv - STARTING_CAPITAL
    
    equity_entry = {
        "date": entry['date'],
        "beg_nlv": beg_nlv,
        "end_nlv": end_nlv,
        "cash_flow": cash_flow,
        "pct_invested": 0.0,
        "daily_dollar": daily_dollar,
        "daily_pct": daily_pct,
        "ltd_dollar": ltd_dollar,
        "ltd_pct": ltd_pct,
        "ytd_pct": ytd_pct,
        "ma_10": None,
        "ma_21": None,
        "ma_50": None
    }
    
    equity_data.append(equity_entry)

print(f"✅ Missing data added. LTD on Sept 13: {equity_data[-1]['ltd_pct']:.2f}%")
print()

# Process each row and calculate incrementally (continuing from missing data)

for idx, row in df_clean.iterrows():
    try:
        # Get date
        date_val = row.iloc[COL_DATE]
        if pd.isna(date_val):
            continue
            
        # Parse date - handle various formats
        date_str = str(date_val).strip()
        try:
            date_obj = pd.to_datetime(date_str)
            date_formatted = date_obj.strftime('%Y-%m-%d')
        except:
            print(f"⚠️  Skipping row with invalid date: {date_str}")
            continue
        
        # Check if new year - reset YTD
        if current_year is None:
            current_year = date_obj.year
        elif date_obj.year != current_year:
            print(f"📅 New year detected: {date_obj.year} - Resetting YTD")
            previous_ytd = 0.0
            current_year = date_obj.year
        
        # Extract equity data
        beg_nlv = clean_currency(row.iloc[COL_BEG_NLV])
        end_nlv = clean_currency(row.iloc[COL_END_NLV])
        
        # Skip if no NLV data
        if end_nlv == 0:
            continue
        
        # Check if this date has a cash flow
        cash_flow = CASH_FLOWS.get(date_formatted, 0.0)
        
        if cash_flow != 0:
            print(f"💰 Cash flow detected on {date_formatted}: ${cash_flow:,.2f}")
            print(f"   Beg NLV: ${beg_nlv:,.2f}, End NLV: ${end_nlv:,.2f}")
            print(f"   Jump without cash: ${end_nlv - beg_nlv:,.2f}")
            print(f"   Actual gain: ${end_nlv - beg_nlv - cash_flow:,.2f}")
        
        # Calculate daily $ and % change (accounting for cash flow)
        if cash_flow != 0:
            daily_dollar = end_nlv - (beg_nlv + cash_flow)
            daily_pct = (daily_dollar / (beg_nlv + cash_flow)) * 100 if (beg_nlv + cash_flow) != 0 else 0
        else:
            daily_dollar = end_nlv - beg_nlv
            daily_pct = (daily_dollar / beg_nlv) * 100 if beg_nlv != 0 else 0
        
        # Calculate LTD $ (total gain from start)
        ltd_dollar = end_nlv - STARTING_CAPITAL
        
        # Calculate LTD % using compound formula: (1 + prev) * (1 + daily) - 1
        if len(equity_data) == 0:
            # First day
            ltd_pct = daily_pct
        else:
            ltd_pct = ((1 + previous_ltd/100) * (1 + daily_pct/100) - 1) * 100
        
        # Calculate YTD % using compound formula: (1 + prev) * (1 + daily) - 1
        ytd_pct = ((1 + previous_ytd/100) * (1 + daily_pct/100) - 1) * 100
        
        # Update previous YTD for next iteration
        previous_ytd = ytd_pct
        
        # Update previous values for next iteration
        previous_ytd = ytd_pct
        previous_ltd = ltd_pct
        
        # Moving averages (will be calculated later if needed)
        ma_10 = None
        ma_21 = None
        ma_50 = None
        
        # % Invested
        pct_invested = clean_percent(row.iloc[COL_PCT_INVESTED]) if COL_PCT_INVESTED < len(row) else 0.0
        if pct_invested is None:
            pct_invested = 0.0
        
        equity_entry = {
            "date": date_formatted,
            "beg_nlv": beg_nlv,
            "end_nlv": end_nlv,
            "cash_flow": cash_flow,
            "pct_invested": pct_invested,
            "daily_dollar": daily_dollar,
            "daily_pct": daily_pct if daily_pct is not None else 0.0,
            "ltd_dollar": ltd_dollar,
            "ltd_pct": ltd_pct if ltd_pct is not None else 0.0,
            "ytd_pct": ytd_pct,
            "ma_10": ma_10,
            "ma_21": ma_21,
            "ma_50": ma_50
        }
        
        equity_data.append(equity_entry)
        
        # Extract market data if available
        spy_close = clean_currency(row.iloc[COL_SPY]) if COL_SPY < len(row) else 0.0
        spy_ltd_pct = clean_percent(row.iloc[COL_SPY_LTD]) if COL_SPY_LTD < len(row) else None
        spy_ytd_pct = clean_percent(row.iloc[COL_SPY_YTD]) if COL_SPY_YTD < len(row) else None
        ndx_close = clean_currency(row.iloc[COL_NASDAQ]) if COL_NASDAQ < len(row) else 0.0
        ndx_ltd_pct = clean_percent(row.iloc[COL_NDX_LTD]) if COL_NDX_LTD < len(row) else None
        ndx_ytd_pct = clean_percent(row.iloc[COL_NDX_YTD]) if COL_NDX_YTD < len(row) else None
        
        if spy_close > 0 or ndx_close > 0:
            market_entry = {
                "date": date_formatted,
                "spy_close": spy_close,
                "spy_ltd_pct": spy_ltd_pct if spy_ltd_pct is not None else 0.0,
                "spy_ytd_pct": spy_ytd_pct if spy_ytd_pct is not None else 0.0,
                "ndx_close": ndx_close,
                "ndx_ltd_pct": ndx_ltd_pct if ndx_ltd_pct is not None else 0.0,
                "ndx_ytd_pct": ndx_ytd_pct if ndx_ytd_pct is not None else 0.0
            }
            market_data.append(market_entry)
        
    except Exception as e:
        print(f"⚠️  Error processing row {idx}: {e}")
        continue

# Save to JSON files
with open('equity_data.json', 'w') as f:
    json.dump(equity_data, f, indent=2)

with open('market_data.json', 'w') as f:
    json.dump(market_data, f, indent=2)

# Print summary
print("\n" + "="*60)
print("✅ IMPORT COMPLETE!")
print("="*60)
print(f"📊 Processed {len(equity_data)} equity entries")
print(f"📈 Processed {len(market_data)} market entries")

if equity_data:
    latest = equity_data[-1]
    print(f"\n📅 Latest Date: {latest['date']}")
    print(f"💰 Latest NLV: ${latest['end_nlv']:,.2f}")
    print(f"📊 LTD %: {latest['ltd_pct']:.2f}%")
    print(f"📈 YTD %: {latest['ytd_pct']:.2f}%")
    if latest['ma_21'] is not None:
        print(f"📉 21-Day SMA: {latest['ma_21']:.2f}%")

print("\n✅ Files saved:")
print("   - equity_data.json")
print("   - market_data.json")
print("\n🚀 You can now restart your Streamlit app!")
