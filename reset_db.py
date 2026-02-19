import os

print("--- CAN SLIM DATABASE RESET ---")
files = ['Trade_Log_Details.csv', 'Trade_Log_Summary.csv']

for f in files:
    if os.path.exists(f):
        os.remove(f)
        print(f"Deleted: {f}")
    else:
        print(f"Not found: {f}")

print("\nDatabase wiped. Ready for fresh import.")