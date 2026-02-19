"""
Diagnostic script to check if image upload feature is enabled
"""

import os
import streamlit as st

print("=" * 60)
print("IMAGE UPLOAD FEATURE DIAGNOSTIC")
print("=" * 60)

# Check 1: r2_storage module
print("\n1. Checking r2_storage module...")
try:
    import r2_storage as r2
    print("   ✅ r2_storage module imported successfully")
    R2_AVAILABLE = True
except Exception as e:
    print(f"   ❌ r2_storage import failed: {e}")
    R2_AVAILABLE = False

# Check 2: Database layer
print("\n2. Checking database layer...")
try:
    import db_layer as db
    print("   ✅ db_layer module imported successfully")
    DB_AVAILABLE = True
except Exception as e:
    print(f"   ❌ db_layer import failed: {e}")
    DB_AVAILABLE = False

# Check 3: Database mode
print("\n3. Checking database mode...")
USE_DATABASE = False
if DB_AVAILABLE:
    if hasattr(st, 'secrets') and 'database' in st.secrets:
        USE_DATABASE = True
        print("   ✅ USE_DATABASE = True (Streamlit Cloud mode)")
    else:
        env_var = os.getenv('USE_DATABASE', 'false').lower()
        USE_DATABASE = env_var == 'true'
        if USE_DATABASE:
            print(f"   ✅ USE_DATABASE = True (environment variable)")
        else:
            print(f"   ❌ USE_DATABASE = False")
            print(f"      Environment variable: {env_var}")
else:
    print("   ❌ USE_DATABASE = False (db_layer not available)")

# Check 4: R2 secrets
print("\n4. Checking R2 secrets...")
try:
    r2_config = st.secrets.get("r2", {})
    if r2_config:
        print("   ✅ R2 secrets found")
        print(f"      - endpoint_url: {'✅' if 'endpoint_url' in r2_config else '❌'}")
        print(f"      - access_key_id: {'✅' if 'access_key_id' in r2_config else '❌'}")
        print(f"      - secret_access_key: {'✅' if 'secret_access_key' in r2_config else '❌'}")
        print(f"      - bucket_name: {'✅' if 'bucket_name' in r2_config else '❌'}")
    else:
        print("   ⚠️  R2 secrets not found in Streamlit secrets")
        print("      This is OK if running locally - R2 will check secrets at runtime")
except Exception as e:
    print(f"   ⚠️  Could not check secrets: {e}")

# Check 5: Feature enabled?
print("\n5. Feature Status:")
print(f"   R2_AVAILABLE = {R2_AVAILABLE}")
print(f"   USE_DATABASE = {USE_DATABASE}")
print(f"   DB_AVAILABLE = {DB_AVAILABLE}")

if R2_AVAILABLE and USE_DATABASE:
    print("\n   ✅ IMAGE UPLOAD FEATURE IS ENABLED")
    print("      Upload fields should appear in:")
    print("      - Trade Manager → Log Buy (Weekly & Daily charts)")
    print("      - Trade Manager → Log Sell (Exit chart)")
elif R2_AVAILABLE and not USE_DATABASE:
    print("\n   ⚠️  IMAGE UPLOAD FEATURE IS PARTIALLY ENABLED")
    print("      r2_storage is available, but USE_DATABASE = False")
    print("      Upload fields will appear in Log Buy, but NOT in Log Sell")
else:
    print("\n   ❌ IMAGE UPLOAD FEATURE IS DISABLED")
    if not R2_AVAILABLE:
        print("      Reason: r2_storage module not available")
    if not USE_DATABASE:
        print("      Reason: USE_DATABASE = False")

print("\n" + "=" * 60)
