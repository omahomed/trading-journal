import pandas as pd

# Read CSV from row 22
df = pd.read_csv('MO EC.csv', skiprows=22)

print("All columns in the CSV:")
for i, col in enumerate(df.columns):
    print(f"{i:3d}: '{col}'")

print(f"\nTotal columns: {len(df.columns)}")

# Show first row of data
print("\nFirst row values (first 10 columns):")
if len(df) > 0:
    for i, col in enumerate(df.columns[:10]):
        print(f"{col}: {df[col].iloc[0]}")
