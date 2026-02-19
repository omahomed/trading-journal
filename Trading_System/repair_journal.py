import pandas as pd
import os

# This script cleans your journal ONE TIME to prep it for the daily system.
# PASTE YOUR PATH INSIDE THE QUOTES
FILE_NAME = r"/Users/momacbookair/my_code/Trading_System/Trading_Journal_Clean.csv"

print("--- REPAIRING DATABASE ---")

if not os.path.exists(FILE_NAME):
    print(f"Error: Could not find {FILE_NAME}")
    print("Make sure this script is in the same folder as your CSV file.")
    exit()

# 1. Load the file
df = pd.read_csv(FILE_NAME)
original_count = len(df)

# 2. Drop empty rows (The Abyss)
df = df.dropna(subset=['Day'])

# 3. Fix Typo in Header (Nsadaq -> Nasdaq)
# We strip whitespace and rename
new_columns = []
for col in df.columns:
    clean_name = col.strip()
    if clean_name == 'Nsadaq':
        new_columns.append('Nasdaq')
    else:
        new_columns.append(clean_name)
df.columns = new_columns

# 4. Save it back
df.to_csv(FILE_NAME, index=False)

print("SUCCESS!")
print(f"Removed {original_count - len(df)} empty rows.")
print(f"Fixed column headers.")
print(f"New file saved as: {FILE_NAME}")
print("You can now delete this repair script.")