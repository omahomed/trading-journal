# Testing Plan for Bug Fixes

**Branch:** feature/critical-fixes
**Date:** February 16, 2026
**Changes:** 3 critical bug fixes

---

## ✅ What Was Fixed

1. **Duplicate Return Bug** - Removed duplicate line 277
2. **Save Verification** - Files now verified after saving
3. **Column Standardization** - Buy_Rule → Rule (no more crashes)

---

## 🧪 How to Test (Before Trading Tomorrow)

### Test 1: Log a Buy Trade
1. Run app: `streamlit run app.py`
2. Go to "Trade Manager" → "Log Buy"
3. Log a small test trade
4. **Expected:** Success message, no errors
5. **Check:** Trade appears in Active Campaigns

### Test 2: Log a Sell Trade
1. Go to "Log Sell"
2. Sell some shares from test trade
3. **Expected:** Success message with P&L calculation
4. **Check:** Trade updates correctly

### Test 3: Update Prices
1. Go to "Update Prices"
2. Click "REFRESH MARKET PRICES"
3. **Expected:** Progress bar, success message
4. **Check:** Prices update, no save errors

### Test 4: Check Data Files
1. Look in `portfolios/CanSlim/` folder
2. **Expected:** See `.tmp` files briefly during saves
3. **Check:** No orphaned `.tmp` files left behind
4. **Check:** Backup files created in `backups/` folder

---

## 🚨 If Something Breaks

**Quick Rollback:**
```bash
# Switch back to main branch (original code)
git checkout main

# Your app works exactly as before
streamlit run app.py
```

**Then message me what went wrong!**

---

## ✅ If Everything Works

**Merge the fixes:**
```bash
# Merge fixes into main branch
git checkout main
git merge feature/critical-fixes

# Now your main branch has the fixes
```

**Continue using normally!**

---

## 📝 What to Watch For

### Good Signs:
- ✅ "Save verified" or similar messages
- ✅ No error popups
- ✅ Trades calculate correctly
- ✅ Can switch between tabs without crashes

### Bad Signs (Report These):
- ❌ "Save verification failed" errors
- ❌ Missing columns errors
- ❌ Crashes when logging trades
- ❌ P&L calculations wrong

---

## 🔄 Next Session: Data Validation

Once these fixes are stable, we'll add:
- Input validation (no negative shares, etc.)
- Position size warnings
- Better error messages
- Automated tests

---

**Your data is safe!** All backups are in the `backups/` folder.
