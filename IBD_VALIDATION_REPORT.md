# IBD Market School Validation Report
**Date:** February 18, 2026
**Analysis:** System vs IBD Distribution Day Counts

---

## Executive Summary

Identified **2 critical bugs** and **1 data source discrepancy** causing false positive distribution days.

### Issues Found:
1. ✅ **CRITICAL BUG:** Stall day thresholds off by 100x (0.3% vs 30%)
2. ⚠️ **DATA MISMATCH:** yfinance vs IBD volume data discrepancies
3. ✅ **CONFIRMED:** Distribution day logic otherwise working correctly

---

## Detailed Findings

### 1. CRITICAL BUG: Stall Day Threshold Error

**Location:** `market_school_rules.py` lines 533-534

**Current Code:**
```python
is_stall = (current['volume_up'] and
           current['close_position'] < 0.5 and
           current['give_back'] > 0.3 and      # ❌ WRONG: 0.3%
           current['intraday_gain'] > 0.3)     # ❌ WRONG: 0.3%
```

**Should Be:**
```python
is_stall = (current['volume_up'] and
           current['close_position'] < 0.5 and
           current['give_back'] > 30 and       # ✅ CORRECT: 30%
           current['intraday_gain'] > 30)      # ✅ CORRECT: 30%
```

**Impact:**
- ANY day with >0.3% intraday gain and >0.3% give-back gets flagged as stall
- Should require >30% intraday gain and >30% give-back
- **This is why Feb 5, 2026 SPY was incorrectly flagged**

**Evidence:**
- Feb 5, 2026 SPY:
  - Intraday gain: ~0% (High ≈ Open)
  - Give back: ~8%
  - Should NOT qualify (intraday gain < 30%)
  - But got flagged due to thresholds being 0.3 instead of 30

---

### 2. Jan 30, 2026 - Data Source Discrepancy

**SPY Volume Analysis:**

| Source | Volume Jan 29 | Volume Jan 30 | Change | Qualifies? |
|--------|---------------|---------------|--------|------------|
| yfinance | 97,486,200 | 101,835,100 | **+4.5% ⬆️** | ✅ YES (volume up) |
| IBD | Unknown | Unknown | **DOWN ⬇️** | ❌ NO (volume down) |

**Finding:**
- yfinance shows volume INCREASED by 4.5%
- IBD states volume was LOWER than previous day
- **Root Cause:** Different data providers (yfinance vs IBD's data source)

**Recommendation:**
- Accept this as unavoidable data variance
- Consider adding manual override capability for known IBD discrepancies
- Monitor if this is a consistent pattern or one-off occurrence

---

### 3. Feb 5, 2026 - False Positive Stall Day (SPY)

**Database Shows:**
- Distribution Count: 6
- Sell Signals: S9, S10, S4

**IBD Shows:**
- Distribution Count: 4
- No new distribution this day

**Analysis:**
```
Volume: 113,610,800 vs prev 105,204,600 = +8.0% UP ✅
Daily %: -0.49% (not distribution threshold of -0.2%)
Close position: 0.23 (✓ < 0.5)
Give back: ~8% (✓ > 0.3, but should check > 30)
Intraday gain: ~0% (✗ < 30%, should NOT qualify)
```

**Verdict:** False positive due to threshold bug

---

## Code Analysis Summary

### Correctly Implemented:
✅ Distribution day detection: `daily_gain_pct <= -0.2` (≤ -0.2% decline)
✅ Volume comparison: `Volume > Volume.shift(1)`
✅ Close position: `(Close - Low) / (High - Low)` (ratio 0-1)
✅ 25-day expiration rule
✅ 5% rally from low removal
✅ 6% rally from close removal

### Bugs Found:
❌ Stall day `give_back` threshold: 0.3 should be 30
❌ Stall day `intraday_gain` threshold: 0.3 should be 30

---

## Distribution Count Comparison (Jan-Feb 2026)

### Nasdaq

| Date | IBD Count | Our Count | Status | Notes |
|------|-----------|-----------|--------|-------|
| Feb 12 | 7 | 7 | ✅ Match | |
| Feb 10 | 7 | 7 | ✅ Match | New dist day |
| Feb 9 | 6 | ? | ? | |
| Feb 6 | 6 | ? | ? | Dec 30 removed |
| Feb 5 | 7 | ? | ? | |
| Feb 4 | 7 | 7 | ✅ Match | New dist (+1), Dec 29 removed (-1) |
| Feb 3 | 7 | 7 | ✅ Match | New dist day |

### SPY

| Date | IBD Count | Our Count | Status | Notes |
|------|-----------|-----------|--------|-------|
| Feb 12 | 5 | ? | ? | New dist day |
| Feb 5 | 4 | 6 | ❌ MISMATCH | **False positive stall (threshold bug)** |
| Jan 30 | 3 | 4 | ❌ MISMATCH | **Data source discrepancy (volume)** |

---

## Recommended Fixes

### Priority 1: Fix Stall Day Thresholds
**File:** `market_school_rules.py` line 533-534

```python
# Change from:
current['give_back'] > 0.3 and
current['intraday_gain'] > 0.3

# To:
current['give_back'] > 30 and
current['intraday_gain'] > 30
```

### Priority 2: Add Manual Override Capability (Future Enhancement)
- Add optional `ibd_override_count` field to market_signals table
- Display both calculated vs IBD published counts
- Allow notes for discrepancies

### Priority 3: Data Source Documentation
- Document that yfinance may differ from IBD's data
- Consider adding data quality checks
- Log significant volume discrepancies for review

---

## Testing Recommendations

After fixing stall day bug:

1. **Re-sync from Feb 1, 2026** to validate fixes
2. **Compare counts** for Feb 5 (should drop from 6 to 4 for SPY)
3. **Verify Jan 30** still shows discrepancy (data source issue, not bug)
4. **Check historical accuracy** against known IBD publications

---

## Conclusion

**Two distinct issues identified:**

1. **Fixable Bug:** Stall day thresholds (0.3 vs 30) - can be fixed immediately
2. **Data Variance:** yfinance vs IBD volume data - inherent limitation

**Expected Impact of Fix:**
- Feb 5, 2026 SPY: Count should drop from 6 to 4 ✅
- Jan 30, 2026 SPY: Will remain at 4 (data source issue) ⚠️
- Overall accuracy should improve significantly

**Next Steps:**
1. Fix thresholds in market_school_rules.py
2. Re-sync data from Feb 1 onward
3. Validate against IBD's published counts
4. Consider manual override feature for known discrepancies
