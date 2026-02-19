import pandas as pd
import json

# Read CSV
df = pd.read_csv('MO EC.csv', skiprows=22)
df['date'] = pd.to_datetime(df.iloc[:, 4]).dt.strftime('%Y-%m-%d')

# Find Sept 4, 2024
sept4 = df[df['date'] == '2024-09-04']

if not sept4.empty:
    idx = sept4.index[0]
    print("Sept 4, 2024 from Excel CSV:")
    print("="*60)
    print(f"Date: {df.loc[idx, 'date']}")
    print(f"Beg NLV: {df.iloc[idx, 7]}")
    print(f"End NLV: {df.iloc[idx, 8]}")
    print(f"Daily $ (col 9): {df.iloc[idx, 9]}")
    print(f"Daily % (col 10): {df.iloc[idx, 10]}")
    print(f"LTD $ (col 11): {df.iloc[idx, 11]}")
    print(f"LTD % (col 12): {df.iloc[idx, 12]}")
    print()
    print("Expected:")
    print(f"Daily %: ($113,420.72 - $100,000) / $100,000 = 13.42%")
    print(f"LTD %: Should be 13.42% (first gain day)")

# Read our calculated data
with open('equity_data.json', 'r') as f:
    equity_data = json.load(f)

sept4_calc = [e for e in equity_data if e['date'] == '2024-09-04']
if sept4_calc:
    print("\n" + "="*60)
    print("Sept 4, 2024 from our calculation:")
    print("="*60)
    print(f"Daily %: {sept4_calc[0]['daily_pct']:.2f}%")
    print(f"LTD %: {sept4_calc[0]['ltd_pct']:.2f}%")
