"""
Debug script to check trade_images table
"""

import db_layer as db

def check_images():
    """Check what's in the trade_images table"""

    try:
        with db.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'trade_images'
                    )
                """)
                exists = cur.fetchone()[0]

                if not exists:
                    print("❌ trade_images table does not exist!")
                    return

                print("✅ trade_images table exists")

                # Get all records
                cur.execute("""
                    SELECT id, portfolio_id, trade_id, ticker, image_type,
                           image_url, file_name, uploaded_at
                    FROM trade_images
                    ORDER BY uploaded_at DESC
                """)

                rows = cur.fetchall()

                if not rows:
                    print("\n⚠️  No images found in database")
                    print("\nPossible reasons:")
                    print("1. Image upload to R2 failed")
                    print("2. Database save failed after R2 upload")
                    print("3. Check browser console for errors during upload")
                else:
                    print(f"\n✅ Found {len(rows)} image(s):\n")
                    for row in rows:
                        print(f"ID: {row[0]}")
                        print(f"  Portfolio ID: {row[1]}")
                        print(f"  Trade ID: {row[2]}")
                        print(f"  Ticker: {row[3]}")
                        print(f"  Type: {row[4]}")
                        print(f"  URL: {row[5]}")
                        print(f"  File: {row[6]}")
                        print(f"  Uploaded: {row[7]}")
                        print()

                # Also show recent trades to compare
                print("\n📋 Recent trades for comparison:")
                cur.execute("""
                    SELECT ts.trade_id, ts.ticker, p.name, ts.status
                    FROM trades_summary ts
                    JOIN portfolios p ON ts.portfolio_id = p.id
                    ORDER BY ts.id DESC
                    LIMIT 5
                """)

                trades = cur.fetchall()
                for trade in trades:
                    print(f"  {trade[0]} - {trade[1]} ({trade[2]}) - {trade[3]}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_images()
