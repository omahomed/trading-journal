import db_layer as db
import pandas as pd

print("Testing journal loading...")
print("="*60)

portfolios = ['CanSlim', 'TQQQ Strategy', '457B Plan']

for portfolio in portfolios:
    print(f"\nTesting {portfolio}:")
    try:
        df = db.load_journal(portfolio)
        print(f"  Rows returned: {len(df)}")
        if not df.empty:
            print(f"  Columns: {list(df.columns)}")
            print(f"  Date range: {df['Day'].min()} to {df['Day'].max()}")
            print(f"  Sample data:")
            print(df.head(2))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Done!")
