import pandas as pd
import json

# Read CSV, skipping first 2 header rows
df = pd.read_csv('MO EC Simplified.csv', skiprows=2, header=None)

# Name the columns
df.columns = ['Date', 'LTD %', '10SMA%', '21SMA%', '50SMA%', 'SPY LTD %', 'NDX LTD %']

# Remove empty rows
df = df.dropna(subset=['Date'])

# Clean percentage values
def clean_pct(val):
    if pd.isna(val) or val == '':
        return None
    return float(str(val).replace('%', '').strip())

# Apply cleaning
df['LTD %'] = df['LTD %'].apply(clean_pct)
df['10SMA%'] = df['10SMA%'].apply(clean_pct)
df['21SMA%'] = df['21SMA%'].apply(clean_pct)
df['50SMA%'] = df['50SMA%'].apply(clean_pct)
df['SPY LTD %'] = df['SPY LTD %'].apply(clean_pct)
df['NDX LTD %'] = df['NDX LTD %'].apply(clean_pct)

# Convert dates
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y').dt.strftime('%Y-%m-%d')

# Create equity data (need to calculate NLV from LTD %)
STARTING_CAPITAL = 100000
TOTAL_CAPITAL_INVESTED = 156300

equity_data = []
prev_nlv = STARTING_CAPITAL

for idx, row in df.iterrows():
    ltd_pct = row['LTD %'] if row['LTD %'] is not None else 0
    
    # Calculate NLV from LTD %
    # LTD % = (NLV - Total Capital) / Total Capital * 100
    # So: NLV = (LTD% / 100) * Total Capital + Total Capital
    end_nlv = (ltd_pct / 100) * TOTAL_CAPITAL_INVESTED + TOTAL_CAPITAL_INVESTED
    
    # Calculate daily change
    daily_dollar = end_nlv - prev_nlv
    daily_pct = (daily_dollar / prev_nlv * 100) if prev_nlv > 0 else 0
    
    entry = {
        "date": row['Date'],
        "beg_nlv": prev_nlv,
        "end_nlv": end_nlv,
        "cash_flow": 0,  # Historical data doesn't show individual cash flows
        "pct_invested": 100,  # Default - you can update manually
        "daily_dollar": daily_dollar,
        "daily_pct": daily_pct,
        "ltd_dollar": end_nlv - STARTING_CAPITAL,
        "ltd_pct": ltd_pct,
        "ytd_pct": 0,  # Will be calculated by app
        "ma_10": row['10SMA%'],
        "ma_21": row['21SMA%'],
        "ma_50": row['50SMA%']
    }
    
    equity_data.append(entry)
    prev_nlv = end_nlv

# Create market data with LTD %
market_data = []
for _, row in df.iterrows():
    # Always add market data (even if None - we'll handle it)
    entry = {
        "date": row['Date'],
        "spy_ltd_pct": row['SPY LTD %'] if pd.notna(row['SPY LTD %']) else 0,
        "ndx_ltd_pct": row['NDX LTD %'] if pd.notna(row['NDX LTD %']) else 0
    }
    market_data.append(entry)

print(f"✅ Processed {len(equity_data)} equity entries")
print(f"✅ Processed {len(market_data)} market entries")

# Save to JSON
with open('equity_data.json', 'w') as f:
    json.dump(equity_data, f, indent=2)
print("✅ Saved equity_data.json")

with open('market_data.json', 'w') as f:
    json.dump(market_data, f, indent=2)
print("✅ Saved market_data.json")

print("\n📊 Sample - First 3 entries:")
for entry in equity_data[:3]:
    print(f"  {entry['date']}: NLV=${entry['end_nlv']:,.0f}, LTD%={entry['ltd_pct']:.2f}%, 10SMA={entry['ma_10']}")

print("\n📊 Sample - Last 3 entries:")
for entry in equity_data[-3:]:
    print(f"  {entry['date']}: NLV=${entry['end_nlv']:,.0f}, LTD%={entry['ltd_pct']:.2f}%, 10SMA={entry['ma_10']}")

print("\n✅ Import complete!")
print("\n🔄 Going forward: When you add new entries, the app will auto-calculate moving averages!")
import pandas as pd
import json

# Read CSV, skipping first 2 header rows
df = pd.read_csv('MO EC  Simplified.csv', skiprows=2, header=None)

# Name the columns
df.columns = ['Date', 'LTD %', '10SMA%', '21SMA%', '50SMA%', 'SPY LTD %', 'NDX LTD %']

# Remove empty rows
df = df.dropna(subset=['Date'])

# Clean percentage values
def clean_pct(val):
    if pd.isna(val) or val == '':
        return None
    return float(str(val).replace('%', '').strip())

# Apply cleaning
df['LTD %'] = df['LTD %'].apply(clean_pct)
df['10SMA%'] = df['10SMA%'].apply(clean_pct)
df['21SMA%'] = df['21SMA%'].apply(clean_pct)
df['50SMA%'] = df['50SMA%'].apply(clean_pct)
df['SPY LTD %'] = df['SPY LTD %'].apply(clean_pct)
df['NDX LTD %'] = df['NDX LTD %'].apply(clean_pct)

# Convert dates
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y').dt.strftime('%Y-%m-%d')

# Create equity data (need to calculate NLV from LTD %)
STARTING_CAPITAL = 100000
TOTAL_CAPITAL_INVESTED = 156300

equity_data = []
for _, row in df.iterrows():
    ltd_pct = row['LTD %'] if row['LTD %'] is not None else 0
    
    # Calculate NLV from LTD %
    # LTD % = (NLV - Total Capital) / Total Capital * 100
    # So: NLV = (LTD% / 100) * Total Capital + Total Capital
    end_nlv = (ltd_pct / 100) * TOTAL_CAPITAL_INVESTED + TOTAL_CAPITAL_INVESTED
    
    entry = {
        "date": row['Date'],
        "beg_nlv": 0,  # Will be filled properly in next iteration
        "end_nlv": end_nlv,
        "cash_flow": 0,
        "pct_invested": 0,  # You'll need to add this manually
        "daily_dollar": 0,
        "daily_pct": 0,
        "ltd_dollar": end_nlv - STARTING_CAPITAL,
        "ltd_pct": ltd_pct,
        "ytd_pct": 0,  # Will calculate later
        "ma_10": row['10SMA%'],
        "ma_21": row['21SMA%'],
        "ma_50": row['50SMA%']
    }
    
    equity_data.append(entry)

# Create market data
market_data = []
for _, row in df.iterrows():
    if row['SPY LTD %'] is not None or row['NDX LTD %'] is not None:
        entry = {
            "date": row['Date'],
            "spy_close": 0,  # Don't have actual prices
            "ndx_close": 0,  # Don't have actual prices  
            "spy_ltd_pct": row['SPY LTD %'] if row['SPY LTD %'] is not None else 0,
            "ndx_ltd_pct": row['NDX LTD %'] if row['NDX LTD %'] is not None else 0
        }
        market_data.append(entry)

print(f"✅ Processed {len(equity_data)} equity entries")
print(f"✅ Processed {len(market_data)} market entries")

# Save to JSON
with open('equity_data.json', 'w') as f:
    json.dump(equity_data, f, indent=2)
print("✅ Saved equity_data.json")

with open('market_data.json', 'w') as f:
    json.dump(market_data, f, indent=2)
print("✅ Saved market_data.json")

print("\n📊 Sample - First 3 entries:")
for entry in equity_data[:3]:
    print(f"  {entry['date']}: NLV=${entry['end_nlv']:,.0f}, LTD%={entry['ltd_pct']:.2f}%")

print("\n📊 Sample - Last 3 entries:")
for entry in equity_data[-3:]:
    print(f"  {entry['date']}: NLV=${entry['end_nlv']:,.0f}, LTD%={entry['ltd_pct']:.2f}%")

print("\n✅ Import complete! Now restart your Streamlit app.")
