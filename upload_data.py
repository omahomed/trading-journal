import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os

st.set_page_config(page_title="Data Migration Tool", page_icon="🚚")

st.title("🚚 Local -> Cloud Migration Tool")
st.write("This tool will read your local CSV files and push them to your Google Sheet.")

# --- CONFIGURATION ---
SPREADSHEET_NAME = "Master_Trading_Journal"
TAB_MAP = {
    'Trading_Journal_Clean.csv': 'Journal',
    'Trade_Log_Details.csv': 'Details',
    'Trade_Log_Summary.csv': 'Summary'
}

# --- AUTHENTICATION (Reusing your secrets.toml) ---
try:
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(credentials)
    st.success("✅ Google Cloud Authentication Successful")
except Exception as e:
    st.error(f"❌ Authentication Failed: {e}")
    st.stop()

# --- UPLOAD LOGIC ---
if st.button("🚀 START MIGRATION", type="primary"):
    
    progress_bar = st.progress(0)
    status_area = st.empty()
    
    try:
        # Open Spreadsheet
        sh = client.open(SPREADSHEET_NAME)
        
        steps = len(TAB_MAP)
        current_step = 0
        
        for csv_file, tab_name in TAB_MAP.items():
            status_area.write(f"Processing {csv_file} -> {tab_name}...")
            
            if os.path.exists(csv_file):
                # 1. Read CSV
                df = pd.read_csv(csv_file)
                
                # 2. Clean Data for JSON Serialization (Crucial)
                # Convert NaNs to empty strings (JSON doesn't accept NaN)
                df = df.fillna("")
                
                # Convert all date columns to strings to avoid Timestamp errors
                for col in df.columns:
                    if 'date' in col.lower() or 'day' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
                            df[col] = df[col].replace('NaT', '')
                        except: pass
                
                # 3. Select Worksheet
                try:
                    worksheet = sh.worksheet(tab_name)
                except:
                    # Create if missing
                    worksheet = sh.add_worksheet(title=tab_name, rows="1000", cols="20")
                
                # 4. Clear and Update
                worksheet.clear()
                # Prepare payload: Headers + Data
                payload = [df.columns.values.tolist()] + df.values.tolist()
                worksheet.update(payload)
                
                st.write(f"✅ Uploaded {len(df)} rows to '{tab_name}'")
            else:
                st.warning(f"⚠️ File {csv_file} not found locally. Skipping.")
            
            current_step += 1
            progress_bar.progress(current_step / steps)
            
        status_area.success("🎉 MIGRATION COMPLETE! You can now run 'cloud_app.py'.")
        
    except Exception as e:
        st.error(f"Migration Failed: {e}")