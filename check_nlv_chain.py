import pandas as pd

# Read CSV
df = pd.read_csv('MO EC.csv', skiprows=22)
df = df.dropna(subset=[df.columns[4]])
df['date'] = pd.to_datetime(df.iloc[:, 4]).dt.strftime('%Y-%m-%d')

# Check if Beg NLV matches previous End NLV
breaks = []

for i in range(1, len(df)):
    prev_end = float(str(df.iloc[i-1, 8]).replace('$', '').replace(',', '').replace(' ', ''))
    curr_beg = float(str(df.iloc[i, 7]).replace('$', '').replace(',', '').replace(' ', ''))
    
    # They should match (within $0.01 for rounding)
    if abs(prev_end - curr_beg) > 0.01:
        breaks.append({
            'date': df.iloc[i]['date'],
            'prev_date': df.iloc[i-1]['date'],
            'prev_end': prev_end,
            'curr_beg': curr_beg,
            'diff': curr_beg - prev_end
        })

print(f"Found {len(breaks)} breaks in the NLV chain:")
print("="*80)

for b in breaks[:10]:  # Show first 10
    print(f"\n{b['prev_date']} → {b['date']}")
    print(f"  Previous End NLV: ${b['prev_end']:,.2f}")
    print(f"  Current Beg NLV:  ${b['curr_beg']:,.2f}")
    print(f"  Difference:       ${b['diff']:,.2f}")

if len(breaks) > 0:
    print("\n" + "="*80)
    print("❌ The Beginning NLV values don't match the previous day's Ending NLV!")
    print("This breaks the compound calculation chain.")
else:
    print("✅ All NLV values chain correctly!")
