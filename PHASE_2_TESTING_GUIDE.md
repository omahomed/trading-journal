# Phase 2 Testing Guide - Data Validation

**Branch:** feature/data-validation
**Time Required:** 15-30 minutes
**When:** Before live trading tomorrow

---

## 🎯 Quick Start

```bash
# 1. Make sure you're on the right branch
cd "/Users/momacbookair/Library/Mobile Documents/com~apple~CloudDocs/my_code"
git status  # Should show: "On branch feature/data-validation"

# 2. Run the app
streamlit run app.py
```

---

## ✅ Test 1: Buy Validation (5 mins)

**Goal:** Verify buy validation catches bad inputs

### Test 1A: Zero Shares
1. Go to **Trade Manager** → **Log Buy**
2. Fill in: Ticker=TEST, Trade_ID=TEST001, Shares=**0**, Price=100
3. Click **LOG BUY ORDER**
4. ✅ **EXPECT:** Red error "❌ Shares must be greater than 0"

### Test 1B: Zero Price
1. Change Shares to 100, Price to **0**
2. Click **LOG BUY ORDER**
3. ✅ **EXPECT:** Red error "❌ Price must be greater than 0"

### Test 1C: Stop Above Entry
1. Fill: Shares=100, Price=100, Stop Loss=**105**
2. Click **LOG BUY ORDER**
3. ✅ **EXPECT:** Red error "❌ Stop loss ($105.00) must be below entry price ($100.00)"

### Test 1D: Wide Stop Warning
1. Change Stop Loss to **88** (12% wide)
2. Click **LOG BUY ORDER**
3. ✅ **EXPECT:** Yellow warning "⚠️ Warning: Stop is 12.0% wide (recommend < 8%)"
4. Trade should still go through

### Test 1E: Position Size Limit
1. Note your current equity (from dashboard)
2. Calculate: shares × price > 25% of equity
   - Example: If equity = $100K, try shares=300, price=$100 = $30K = 30%
3. Click **LOG BUY ORDER**
4. ✅ **EXPECT:** Red error "⛔ Position size 30.0% exceeds 25% limit"

### Test 1F: Valid Buy
1. Use normal values: shares=50, price=100, stop=93 (7% wide)
2. Click **LOG BUY ORDER**
3. ✅ **EXPECT:** Green "✅ EXECUTED: Bought 50 TEST @ $100"
4. Check `portfolios/CanSlim/Audit_Trail.csv` exists
5. ✅ **EXPECT:** New row with BUY action

---

## ✅ Test 2: Sell Validation (5 mins)

**Goal:** Verify you can't oversell

### Test 2A: Overselling
1. From Test 1F, you should own 50 shares of TEST
2. Go to **Trade Manager** → **Log Sell**
3. Select TEST | TEST001
4. Enter Shares=**100** (more than you own)
5. Enter Price=105
6. Click **LOG SELL ORDER**
7. ✅ **EXPECT:** Red error "❌ Cannot sell 100 shares - you only own 50"

### Test 2B: Valid Sell
1. Change Shares to **30**
2. Click **LOG SELL ORDER**
3. ✅ **EXPECT:** Green "Sold. Transaction ID: ..."
4. Check Audit_Trail.csv
5. ✅ **EXPECT:** New row with SELL action

### Test 2C: Remaining Shares
1. Try to sell another **30** shares
2. ✅ **EXPECT:** Red error "❌ Cannot sell 30 shares - you only own 20"
3. Sell **20** shares (exact amount)
4. ✅ **EXPECT:** Success

---

## ✅ Test 3: Delete Protection (5 mins)

**Goal:** Verify delete requires confirmation

### Test 3A: No Confirmation
1. Go to **Trade Manager** → **Delete Trade**
2. Select TEST001 from dropdown
3. ✅ **EXPECT:** Shows ticker, status, share count
4. ✅ **EXPECT:** Shows transaction count (e.g., "3 transaction(s)")
5. Click **DELETE PERMANENTLY** without typing anything
6. ✅ **EXPECT:** Red error "❌ You must type DELETE to confirm"

### Test 3B: Wrong Confirmation
1. Type "delete" (lowercase) in the text box
2. Click **DELETE PERMANENTLY**
3. ✅ **EXPECT:** Red error (case-sensitive)

### Test 3C: Correct Confirmation
1. Type "DELETE" (uppercase)
2. Click **DELETE PERMANENTLY**
3. ✅ **EXPECT:** Green "✅ Trade TEST001 deleted. Backup saved to: ..."
4. Check `portfolios/CanSlim/backups/` folder
5. ✅ **EXPECT:** Files like:
   - `Summary_pre_delete_TEST001_YYYYMMDD_HHMMSS.csv`
   - `Details_pre_delete_TEST001_YYYYMMDD_HHMMSS.csv`
6. Check Audit_Trail.csv
7. ✅ **EXPECT:** New row with DELETE action

---

## ✅ Test 4: Rebuild Protection (5 mins)

**Goal:** Verify rebuild creates backups

### Test 4A: Checkbox Required
1. Go to **Trade Manager** → **Database Health**
2. ✅ **EXPECT:** Button is grayed out (disabled)
3. Try clicking **FULL REBUILD** → Should do nothing

### Test 4B: Rebuild with Backup
1. Check the checkbox "I understand this will recalculate all campaigns"
2. ✅ **EXPECT:** Button becomes blue (enabled)
3. Click **FULL REBUILD**
4. ✅ **EXPECT:** Progress bar, then green "✅ Rebuilt X campaigns. Backup saved."
5. Check `portfolios/CanSlim/backups/` folder
6. ✅ **EXPECT:** Files like:
   - `Summary_pre_rebuild_YYYYMMDD_HHMMSS.csv`
   - `Details_pre_rebuild_YYYYMMDD_HHMMSS.csv`
7. Check Audit_Trail.csv
8. ✅ **EXPECT:** New row with REBUILD action

---

## ✅ Test 5: Audit Trail (2 mins)

**Goal:** Verify all actions are logged

### Final Check
1. Open `portfolios/CanSlim/Audit_Trail.csv`
2. ✅ **EXPECT:** Columns: Timestamp, User, Action, Trade_ID, Ticker, Details
3. ✅ **EXPECT:** Rows for:
   - BUY (from Test 1F)
   - SELL (from Test 2B and 2C)
   - DELETE (from Test 3C)
   - REBUILD (from Test 4B)
4. Each row should have timestamp and details

---

## 🎯 Success Criteria

**ALL tests should pass. If ANY test fails:**
1. Note exactly which test failed
2. Copy the error message
3. Take a screenshot if possible
4. Message me with details

**If all tests pass:**
1. Your validation is working perfectly!
2. Safe to merge to main branch:
   ```bash
   git checkout main
   git merge feature/data-validation
   ```
3. Ready for live trading tomorrow!

---

## 🚨 If Something Breaks

**Rollback to pre-validation code:**
```bash
# Stop Streamlit (Ctrl+C)
git checkout main
streamlit run app.py
# Your app works exactly as before
```

**Then message me with:**
- Which test failed
- Error message
- Screenshot (if applicable)

---

## 📊 Expected Files After Testing

```
portfolios/CanSlim/
├── Summary.csv (updated)
├── Details.csv (updated)
├── Audit_Trail.csv (NEW - all actions logged)
└── backups/
    ├── Summary_pre_delete_TEST001_*.csv (from Test 3)
    ├── Details_pre_delete_TEST001_*.csv (from Test 3)
    ├── Summary_pre_rebuild_*.csv (from Test 4)
    └── Details_pre_rebuild_*.csv (from Test 4)
```

---

## ⏱️ Time Estimate

- Test 1 (Buy): 5 mins
- Test 2 (Sell): 5 mins
- Test 3 (Delete): 5 mins
- Test 4 (Rebuild): 5 mins
- Test 5 (Audit): 2 mins

**Total: ~22 minutes**

---

## 💡 Tips

1. **Use test data:** Don't test with real positions
2. **Check audit trail** after each major action
3. **Backups are automatic** - you don't need to do anything
4. **Case-sensitive:** DELETE must be uppercase
5. **One test at a time:** Don't skip ahead

---

## ✅ After All Tests Pass

1. Celebrate! 🎉
2. Merge to main:
   ```bash
   git checkout main
   git merge feature/data-validation
   git log --oneline -3  # See your commits
   ```
3. Delete test trades if you want to clean up
4. Ready for live trading!

---

**Happy Testing!** 🚀

If you need help, see `SESSION_2_SUMMARY.md` for detailed documentation.
