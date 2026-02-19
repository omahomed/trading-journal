import pandas as pd
import json

# Read the equity data that was just created
with open('equity_data.json', 'r') as f:
    equity_data = json.load(f)

# Get last 5 entries
print("Last 5 entries from equity_data.json:")
print("="*80)
for entry in equity_data[-5:]:
    print(f"Date: {entry['date']}")
    print(f"  Daily %: {entry['daily_pct']:.2f}%")
    print(f"  YTD %: {entry['ytd_pct']:.2f}%")
    print(f"  LTD %: {entry['ltd_pct']:.2f}%")
    print(f"  Cash Flow: ${entry['cash_flow']:,.2f}")
    print()

# Read original CSV to compare
df = pd.read_csv('MO EC.csv', skiprows=22)
df = df.dropna(subset=[df.columns[4]])
df['date'] = pd.to_datetime(df.iloc[:, 4]).dt.strftime('%Y-%m-%d')

# Get the LTD % column from Excel (column 12)
print("\n" + "="*80)
print("Last 5 entries from Excel CSV (Column 12 - LTD %):")
print("="*80)
for idx in df.tail(5).index:
    date = df.loc[idx, 'date']
    ltd_from_excel = df.iloc[idx, 12]  # Column 12 is LTD %
    print(f"Date: {date}, Excel LTD %: {ltd_from_excel}")

print("\n" + "="*80)
print("COMPARISON - Latest entry:")
print("="*80)
latest_calc = equity_data[-1]
latest_excel_idx = df.index[-1]
excel_ltd = df.iloc[latest_excel_idx, 12]

print(f"Calculated LTD %: {latest_calc['ltd_pct']:.2f}%")
print(f"Excel LTD %: {excel_ltd}")
print(f"Difference: {latest_calc['ltd_pct'] - float(str(excel_ltd).replace('%','')):.2f}%")
