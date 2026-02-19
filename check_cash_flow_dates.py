import pandas as pd

# Read CSV
df = pd.read_csv('MO EC.csv', skiprows=22)
df = df.dropna(subset=[df.columns[4]])  # Date column

# Convert dates
df['date'] = pd.to_datetime(df.iloc[:, 4])
df['date_formatted'] = df['date'].dt.strftime('%Y-%m-%d')

# Find dates around the cash flow events
target_dates = ['2025-07-01', '2025-09-12', '2025-09-19']

print("Looking for cash flow dates:")
print("=" * 60)

for target in target_dates:
    target_dt = pd.to_datetime(target)
    
    # Find exact match
    exact = df[df['date_formatted'] == target]
    
    if not exact.empty:
        print(f"\n✅ Found {target}:")
        idx = exact.index[0]
        beg_nlv = df.iloc[idx, 7]
        end_nlv = df.iloc[idx, 8]
        print(f"   Beg NLV: {beg_nlv}")
        print(f"   End NLV: {end_nlv}")
    else:
        # Find closest dates
        print(f"\n❌ {target} not found. Closest dates:")
        df['diff'] = abs((df['date'] - target_dt).dt.days)
        closest = df.nsmallest(3, 'diff')[['date_formatted', 'diff']]
        print(closest.to_string(index=False))

print("\n" + "=" * 60)
print("\nAll dates in July 2025:")
july = df[df['date'].dt.month == 7]
print(july[['date_formatted']].head(10).to_string(index=False))

print("\n" + "=" * 60)
print("\nAll dates in September 2025:")
sept = df[df['date'].dt.month == 9]
print(sept[['date_formatted']].head(20).to_string(index=False))
