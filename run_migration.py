"""
Database Migration Runner
Adds the trade_images table to your database
"""

import db_layer as db

def run_migration():
    """Run the trade_images table migration"""

    migration_sql = """
    -- Create trade_images table
    CREATE TABLE IF NOT EXISTS trade_images (
        id SERIAL PRIMARY KEY,
        portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
        trade_id VARCHAR(50) NOT NULL,
        ticker VARCHAR(20) NOT NULL,
        image_type VARCHAR(20) NOT NULL,
        image_url TEXT NOT NULL,
        file_name VARCHAR(255),
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        CONSTRAINT unique_trade_image UNIQUE (portfolio_id, trade_id, image_type)
    );

    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_trade_images_trade ON trade_images (portfolio_id, trade_id);
    CREATE INDEX IF NOT EXISTS idx_trade_images_type ON trade_images (image_type);
    """

    try:
        print("🔄 Running migration to add trade_images table...")

        with db.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Execute the migration
                cur.execute(migration_sql)
                conn.commit()

                # Verify table was created
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name = 'trade_images'
                """)

                result = cur.fetchone()

                if result:
                    print("✅ SUCCESS! trade_images table created")

                    # Show table structure
                    cur.execute("""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_name = 'trade_images'
                        ORDER BY ordinal_position
                    """)

                    columns = cur.fetchall()
                    print("\n📋 Table structure:")
                    for col in columns:
                        print(f"   - {col[0]}: {col[1]} (nullable: {col[2]})")

                    print("\n🎉 Migration complete! You can now upload trade images.")
                else:
                    print("❌ Table was not created")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

    return True

if __name__ == "__main__":
    # Test database connection first
    print("Testing database connection...")
    if db.test_connection():
        print("✅ Database connected\n")
        run_migration()
    else:
        print("❌ Cannot connect to database. Check your database configuration.")
