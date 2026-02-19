import pandas as pd
import os

# --- CONFIGURATION ---
# ADJUSTED PATHS: Look into the CanSlim folder specifically
SUMMARY_FILE = 'portfolios/CanSlim/Trade_Log_Summary.csv'
DETAILS_FILE = 'portfolios/CanSlim/Trade_Log_Details.csv'

def clean_ledger():
    print(f"--- O'NEIL LEDGER REPAIR TOOL ---")
    
    # 1. LOAD SUMMARY
    if not os.path.exists(SUMMARY_FILE):
        print(f"Error: {SUMMARY_FILE} not found. Check your path!")
        return

    print(f"Loading {SUMMARY_FILE}...")
    df = pd.read_csv(SUMMARY_FILE)
    
    # Convert Date and Sort
    df['Open_Date_DT'] = pd.to_datetime(df['Open_Date'])
    df = df.sort_values('Open_Date_DT')
    
    # 2. GENERATE NEW IDs (YYYYMM-###)
    print("Renumbering trades...")
    new_ids = []
    id_mapping = {} # Old -> New
    counters = {}   # YYYYMM -> Count

    for index, row in df.iterrows():
        if pd.isna(row['Open_Date_DT']):
            yyyymm = "000000"
        else:
            yyyymm = row['Open_Date_DT'].strftime('%Y%m')

        if yyyymm not in counters:
            counters[yyyymm] = 1
        else:
            counters[yyyymm] += 1

        seq = counters[yyyymm]
        new_id = f"{yyyymm}-{seq:03d}"
        
        new_ids.append(new_id)
        id_mapping[str(row['Trade_ID'])] = new_id

    df['New_Trade_ID'] = new_ids
    
    # Create Mapping DataFrame
    mapping_path = 'portfolios/CanSlim/Trade_ID_Mapping.csv'
    mapping_df = pd.DataFrame(list(id_mapping.items()), columns=['Old_Trade_ID', 'New_Trade_ID'])
    mapping_df.to_csv(mapping_path, index=False)
    print(f"-> Generated '{mapping_path}'")

    # Apply New IDs to Summary
    df['Trade_ID'] = df['New_Trade_ID']
    df_final = df.drop(columns=['New_Trade_ID', 'Open_Date_DT'])
    
    # Save New Summary
    sum_out_path = 'portfolios/CanSlim/Trade_Log_Summary_Renumbered.csv'
    df_final.to_csv(sum_out_path, index=False)
    print(f"-> Generated '{sum_out_path}'")

    # 3. UPDATE DETAILS FILE (If it exists)
    if os.path.exists(DETAILS_FILE):
        print(f"Processing {DETAILS_FILE}...")
        df_d = pd.read_csv(DETAILS_FILE)
        df_d['Trade_ID'] = df_d['Trade_ID'].astype(str)
        
        # Map old IDs to new IDs
        df_d['Trade_ID'] = df_d['Trade_ID'].map(id_mapping).fillna(df_d['Trade_ID'])
        
        det_out_path = 'portfolios/CanSlim/Trade_Log_Details_Clean.csv'
        df_d.to_csv(det_out_path, index=False)
        print(f"-> Generated '{det_out_path}'")
    else:
        print(f"Warning: {DETAILS_FILE} not found. Only Summary was updated.")

    print("\nSUCCESS. Now go into 'portfolios/CanSlim/' and rename your files.")

if __name__ == "__main__":
    clean_ledger()
