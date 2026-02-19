# Quick Testing Checklist ✅

**App URL:** http://localhost:8507
**Time:** ~15-20 minutes

---

## Test 1: Buy Validation

### 1A: Zero Shares ❌
- [ ] Go to **Trade Manager** → **Log Buy**
- [ ] Enter: Ticker=TEST, Trade_ID=TEST001, Shares=**0**, Price=100
- [ ] Click **LOG BUY ORDER**
- [ ] ✅ Should see: "❌ Shares must be greater than 0"

### 1B: Zero Price ❌
- [ ] Change: Shares=100, Price=**0**
- [ ] Click **LOG BUY ORDER**
- [ ] ✅ Should see: "❌ Price must be greater than 0"

### 1C: Stop Above Entry ❌
- [ ] Change: Price=100, Stop Loss=**105**
- [ ] Click **LOG BUY ORDER**
- [ ] ✅ Should see: "❌ Stop loss ($105.00) must be below entry price"

### 1D: Wide Stop Warning ⚠️
- [ ] Change: Stop Loss=**88** (12% wide)
- [ ] Click **LOG BUY ORDER**
- [ ] ✅ Should see: "⚠️ Warning: Stop is 12.0% wide"
- [ ] ✅ Trade should still complete

### 1E: Position Size Limit ❌
- [ ] Note your equity from dashboard: $________
- [ ] Calculate 26% of equity = $________
- [ ] Enter shares × price > 26% of equity
- [ ] Click **LOG BUY ORDER**
- [ ] ✅ Should see: "⛔ Position size X% exceeds 25% limit"

### 1F: Valid Buy ✅
- [ ] Enter: Shares=50, Price=100, Stop=93
- [ ] Click **LOG BUY ORDER**
- [ ] ✅ Should see: "✅ EXECUTED: Bought 50 TEST @ $100"

---

## Test 2: Sell Validation

### 2A: Overselling ❌
- [ ] Go to **Trade Manager** → **Log Sell**
- [ ] Select: TEST | TEST001
- [ ] Enter: Shares=**100** (more than you own), Price=105
- [ ] Click **LOG SELL ORDER**
- [ ] ✅ Should see: "❌ Cannot sell 100 shares - you only own 50"

### 2B: Valid Sell ✅
- [ ] Change: Shares=**30**
- [ ] Click **LOG SELL ORDER**
- [ ] ✅ Should see: "Sold. Transaction ID: ..."

### 2C: Remaining Shares
- [ ] Try to sell **30** more shares
- [ ] ✅ Should see: "❌ Cannot sell 30 shares - you only own 20"
- [ ] Change to **20** shares
- [ ] ✅ Should succeed

---

## Test 3: Delete Protection

### 3A: Wrong Confirmation ❌
- [ ] Go to **Trade Manager** → **Delete Trade**
- [ ] Select: TEST001
- [ ] ✅ Should show ticker, status, transaction count
- [ ] Click **DELETE PERMANENTLY** (without typing anything)
- [ ] ✅ Should see: "❌ You must type DELETE to confirm"

### 3B: Lowercase ❌
- [ ] Type: "delete" (lowercase)
- [ ] Click **DELETE PERMANENTLY**
- [ ] ✅ Should still block (case-sensitive)

### 3C: Correct Confirmation ✅
- [ ] Type: "DELETE" (uppercase)
- [ ] Click **DELETE PERMANENTLY**
- [ ] ✅ Should see: "✅ Trade TEST001 deleted. Backup saved to: ..."

---

## Test 4: Rebuild Protection

### 4A: Checkbox Required
- [ ] Go to **Trade Manager** → **Database Health**
- [ ] ✅ Button should be grayed out (disabled)

### 4B: Rebuild Works ✅
- [ ] Check: "I understand this will recalculate all campaigns"
- [ ] ✅ Button should become blue (enabled)
- [ ] Click **FULL REBUILD**
- [ ] ✅ Should see progress bar, then success message

---

## Test 5: Audit Trail

### 5A: Check File Exists ✅
- [ ] Check if file exists: `portfolios/CanSlim/Audit_Trail.csv`
- [ ] ✅ Should have rows for BUY, SELL, DELETE, REBUILD

---

## ✅ ALL TESTS COMPLETE!

If all tests passed:
```bash
git checkout main
git merge feature/data-validation
```

If any failed, message Claude with details!
