import pandas as pd

# Read CSV starting from row 22 (where the data begins based on your structure)
df = pd.read_csv('MO EC.csv', skiprows=22)

print("=" * 80)
print("ALL COLUMNS IN THE CSV:")
print("=" * 80)
for i, col in enumerate(df.columns):
    print(f"{i:3d}: '{col}'")

print(f"\n{'=' * 80}")
print(f"Total columns: {len(df.columns)}")
print("=" * 80)

# Show first 3 rows of data to see structure
print("\nFirst 3 rows (first 10 columns only):")
print(df.iloc[:3, :10])

# Look for key columns
print("\n" + "=" * 80)
print("SEARCHING FOR KEY COLUMNS:")
print("=" * 80)

keywords_to_find = {
    'Beg NLV': ['beg', 'beginning', 'start'],
    'End NLV': ['end', 'ending', 'nlv'],
    'Cash Flow': ['cash', 'flow', 'in/out'],
    'LTD %': ['ltd', 'lifetime'],
    'YTD %': ['ytd', 'year'],
    'SPY LTD %': ['spy', 'spyltd'],
    'NDX LTD %': ['ndx', 'nasdaq']
}

for key, keywords in keywords_to_find.items():
    found = []
    for col in df.columns:
        col_lower = str(col).lower().replace(' ', '')
        for keyword in keywords:
            if keyword.replace(' ', '') in col_lower:
                found.append(col)
                break
    print(f"{key:15s}: {found if found else 'NOT FOUND'}")
