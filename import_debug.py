import pandas as pd
import os

SOURCE_FILE = 'CLOSED TRADE LOGS.csv'
DETAILS_FILE = 'Trade_Log_Details.csv'
SUMMARY_FILE = 'Trade_Log_Summary.csv'

print("--- DIAGNOSTIC IMPORT ---")

if not os.path.exists(SOURCE_FILE):
    print(f"CRITICAL: {SOURCE_FILE} not found.")
    exit()

# Try loading with header=1
print(f"Reading {SOURCE_FILE}...")
df = pd.read_csv(SOURCE_FILE, header=1)

# 1. CHECK COLUMNS
print("\n[1] COLUMNS DETECTED:")
clean_cols = [c.strip() for c in df.columns]
print(clean_cols)

expected = ['Trade #', 'Ticker', 'Shares', 'Avg_Entry', 'Avg_Exit']
missing = [c for c in expected if c not in clean_cols]

if missing:
    print(f"\n[!] CRITICAL ERROR: Missing Columns: {missing}")
    print("The script cannot read the data because headers don't match.")
    print(f"Found: {clean_cols}")
    exit()
else:
    print("\n[OK] All required columns found.")

# 2. CHECK ROW 1 DATA
print("\n[2] SAMPLE ROW (Row 0):")
print(df.iloc[0])

# 3. CHECK LOGIC
print("\n[3] TESTING IMPORT LOGIC:")
df.columns = [c.strip() for c in df.columns]

# Load existing IDs to check overlaps
existing_ids = []
if os.path.exists(SUMMARY_FILE):
    df_s = pd.read_csv(SUMMARY_FILE)
    existing_ids = df_s['Trade_ID'].astype(str).tolist()
    print(f"Existing Database has {len(existing_ids)} trades.")

skipped_dup = 0
skipped_invalid = 0
valid = 0

for idx, row in df.iterrows():
    tid = str(row.get('Trade #', '')).strip()
    shares = row.get('Shares', 0)
    
    if tid in existing_ids:
        skipped_dup += 1
        if idx < 3: print(f" - Row {idx} ({tid}): SKIPPED (Duplicate)")
    elif not tid or tid.lower() == 'nan':
        skipped_invalid += 1
    else:
        valid += 1
        if idx < 3: print(f" - Row {idx} ({tid}): WOULD IMPORT (Shares: {shares})")

print("-" * 30)
print(f"RESULTS PREDICTION:")
print(f"Duplicates (Already in DB): {skipped_dup}")
print(f"Invalid (Empty/Bad Data): {skipped_invalid}")
print(f"New Valid Trades:         {valid}")
print("-" * 30)

if valid == 0 and skipped_dup > 0:
    print("CONCLUSION: Your data is already imported!")
elif valid == 0:
    print("CONCLUSION: The script can't read the data values properly.")