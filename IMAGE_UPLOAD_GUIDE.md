# Trade Image Upload Feature - Setup & Usage Guide

## Overview

You can now attach chart images to your trades to document your entry and exit setups. The system supports:
- **Weekly Chart** - Uploaded when logging a BUY
- **Daily Chart** - Uploaded when logging a BUY
- **Exit Chart** - Uploaded when logging a SELL

Images are stored in Cloudflare R2 (S3-compatible object storage) and linked to trades in your PostgreSQL database.

---

## Setup Steps

### 1. Database Migration

First, add the `trade_images` table to your database:

```bash
# Connect to your PostgreSQL database and run:
psql <your-database-connection-string> -f add_trade_images_table.sql
```

Or manually execute the SQL in [add_trade_images_table.sql](add_trade_images_table.sql)

### 2. Verify R2 Credentials

Ensure your `.streamlit/secrets.toml` file contains:

```toml
[r2]
endpoint_url = "https://<account-id>.r2.cloudflarestorage.com"
access_key_id = "your-access-key-id"
secret_access_key = "your-secret-access-key"
bucket_name = "your-bucket-name"
```

### 3. Test the Integration

Run your Streamlit app:

```bash
streamlit run app.py
```

Check the console output for:
- ✅ `Database mode enabled`
- ✅ R2 storage module loaded (no import errors)

---

## How to Use

### Uploading Charts When Logging a BUY

1. Navigate to **Trade Manager → Log Buy**
2. Fill out the trade details as usual
3. Scroll down to the **"📸 Chart Documentation (Optional)"** section
4. Upload your chart images:
   - **Weekly Chart**: Screenshot of weekly timeframe
   - **Daily Chart**: Screenshot of daily timeframe
5. Click **LOG BUY ORDER**

The system will:
- Save the trade to the database
- Upload images to Cloudflare R2
- Link images to the trade in the `trade_images` table

### Uploading Exit Chart When Logging a SELL

1. Navigate to **Trade Manager → Log Sell**
2. Select the trade to sell
3. Fill out sell details (shares, price, rule, notes)
4. Upload an **Exit Chart** showing your sell point
5. Click **LOG SELL ORDER**

The exit chart will be saved alongside the weekly/daily charts.

### Viewing Uploaded Charts

#### Active Campaign Summary Tab

1. Go to **Trade Manager → Active Campaign Summary**
2. Click the **"📸 View Entry Charts (Active Trades)"** expander
3. Select a trade from the dropdown
4. View the weekly and daily charts side-by-side

#### All Campaigns Tab

1. Go to **Trade Manager → All Campaigns**
2. Click the **"📸 View Trade Charts"** expander
3. Select any trade (open or closed)
4. View all three charts: Weekly, Daily, and Exit (if available)

---

## File Structure

```
my_code/
├── r2_storage.py              # Cloudflare R2 upload/download functions
├── db_layer.py                # Database functions for trade_images table
├── app.py                     # UI with file uploaders and image viewers
├── schema.sql                 # Full schema (includes trade_images table)
├── add_trade_images_table.sql # Migration script (one-time setup)
└── IMAGE_UPLOAD_GUIDE.md      # This file
```

---

## Technical Details

### Image Storage Format

Images are stored in R2 with the following path structure:

```
<portfolio-name>/<trade-id>/<image-type>_<timestamp>.<extension>

Example:
CanSlim/202601-001/weekly_20260219_143052.png
CanSlim/202601-001/daily_20260219_143052.png
CanSlim/202601-001/exit_20260219_150022.png
```

### Database Schema

```sql
CREATE TABLE trade_images (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id),
    trade_id VARCHAR(50) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    image_type VARCHAR(20) NOT NULL,  -- 'weekly', 'daily', 'exit'
    image_url TEXT NOT NULL,          -- R2 object key
    file_name VARCHAR(255),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (portfolio_id, trade_id, image_type)
);
```

The unique constraint ensures only one image per type per trade. Re-uploading replaces the previous image.

### Supported File Types

- PNG (.png)
- JPEG (.jpg, .jpeg)

Maximum file size is determined by Streamlit's default uploader limits (~200MB).

---

## Troubleshooting

### "R2 credentials not found in Streamlit secrets"

**Solution**: Add R2 credentials to `.streamlit/secrets.toml` (see Setup Step 2)

### "Failed to upload image to R2"

**Possible causes**:
- Incorrect R2 credentials
- Invalid bucket name
- Network connectivity issues

**Debug**: Check console output for specific error messages

### Images not displaying

**Possible causes**:
- Image was not successfully uploaded (check success message after logging trade)
- R2 download permissions issue
- Corrupted image file

**Debug**:
1. Check the `trade_images` table to verify the record exists
2. Try re-uploading the image
3. Verify R2 bucket permissions

### Database table not found

**Solution**: Run the migration script `add_trade_images_table.sql`

---

## Feature Flags

The image upload feature is automatically enabled when:

1. `USE_DATABASE = True` (database mode is active)
2. `R2_AVAILABLE = True` (r2_storage module loaded successfully)

If either is `False`, the file upload fields will not appear in the UI.

---

## Future Enhancements

Potential improvements:
- Bulk image upload for historical trades
- Image compression/optimization
- Annotation tools (draw on charts)
- Image galleries per ticker
- Export trades with embedded images (PDF reports)

---

## Support

If you encounter issues:

1. Check console output for error messages
2. Verify R2 credentials and database connection
3. Review the migration script output
4. Test with a small image file first (<1MB)

For questions about Cloudflare R2, see: https://developers.cloudflare.com/r2/
