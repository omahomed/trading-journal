"""
MO Trading API — Endpoint Test Suite
Runs against the live Railway API. Tests the full trade lifecycle:
  1. Health check
  2. Read endpoints (trades, journal, market data)
  3. Log Buy (new campaign)
  4. Log Buy (scale in)
  5. Log Sell (partial)
  6. Log Sell (close out)
  7. Delete trade (cleanup)

Usage:
  python api/test_endpoints.py              # test against Railway
  python api/test_endpoints.py local        # test against localhost:8000
"""

import requests
import sys
import time
from datetime import datetime

# ── Config ──
BASE = "http://localhost:8000" if "local" in sys.argv else "https://web-production-cdf47.up.railway.app"
PORTFOLIO = "CanSlim"
TEST_TRADE_ID = f"TEST-{int(time.time())}"  # unique per run
TEST_TICKER = "ZZTEST"

passed = 0
failed = 0
errors = []


def test(name, fn):
    """Run a test, track pass/fail."""
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  ✓ {name}")
    except AssertionError as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ✗ {name} — {e}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ✗ {name} — EXCEPTION: {e}")


# ════════════════════════════════════════
# 1. HEALTH & READ ENDPOINTS
# ════════════════════════════════════════
print(f"\n{'='*60}")
print(f"MO Trading API Tests — {BASE}")
print(f"Test Trade ID: {TEST_TRADE_ID}")
print(f"{'='*60}\n")

print("── Health & Read ──")


def test_health():
    r = requests.get(f"{BASE}/api/health")
    assert r.status_code == 200, f"status {r.status_code}"
    data = r.json()
    assert data.get("status") == "ok", f"unexpected: {data}"


def test_trades_open():
    r = requests.get(f"{BASE}/api/trades/open", params={"portfolio": PORTFOLIO})
    assert r.status_code == 200, f"status {r.status_code}"
    data = r.json()
    assert isinstance(data, list), f"expected list, got {type(data)}"


def test_trades_closed():
    r = requests.get(f"{BASE}/api/trades/closed", params={"portfolio": PORTFOLIO, "limit": 5})
    assert r.status_code == 200, f"status {r.status_code}"
    data = r.json()
    assert isinstance(data, list), f"expected list, got {type(data)}"


def test_trades_recent():
    r = requests.get(f"{BASE}/api/trades/recent", params={"portfolio": PORTFOLIO, "limit": 5})
    assert r.status_code == 200, f"status {r.status_code}"
    data = r.json()
    assert isinstance(data, list), f"expected list, got {type(data)}"


def test_journal_latest():
    r = requests.get(f"{BASE}/api/journal/latest", params={"portfolio": PORTFOLIO})
    assert r.status_code == 200, f"status {r.status_code}"


def test_next_trade_id():
    r = requests.get(f"{BASE}/api/trades/next-id", params={"portfolio": PORTFOLIO})
    assert r.status_code == 200, f"status {r.status_code}"
    data = r.json()
    assert "trade_id" in data, f"missing trade_id: {data}"


def test_market_mfactor():
    r = requests.get(f"{BASE}/api/market/mfactor")
    assert r.status_code == 200, f"status {r.status_code}"


def test_r2_status():
    r = requests.get(f"{BASE}/api/r2/status")
    assert r.status_code == 200, f"status {r.status_code}"


test("Health check", test_health)
test("Trades open", test_trades_open)
test("Trades closed", test_trades_closed)
test("Trades recent", test_trades_recent)
test("Journal latest", test_journal_latest)
test("Next trade ID", test_next_trade_id)
test("Market M-Factor", test_market_mfactor)
test("R2 status", test_r2_status)


# ════════════════════════════════════════
# 2. WRITE ENDPOINTS — FULL LIFECYCLE
# ════════════════════════════════════════
print("\n── Trade Lifecycle (Buy → Scale In → Partial Sell → Close → Delete) ──")


def test_log_buy_new():
    """Log a new campaign buy."""
    r = requests.post(f"{BASE}/api/trades/buy", json={
        "portfolio": PORTFOLIO,
        "action_type": "new",
        "ticker": TEST_TICKER,
        "trade_id": TEST_TRADE_ID,
        "shares": 100,
        "price": 50.00,
        "stop_loss": 46.00,
        "rule": "br1.4 Double Bottom",
        "notes": "API test — new campaign",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
    })
    assert r.status_code == 200, f"status {r.status_code}"
    data = r.json()
    assert data.get("status") == "ok", f"buy failed: {data}"
    assert data.get("trx_id"), f"missing trx_id: {data}"
    print(f"       trx_id={data['trx_id']}, detail_id={data.get('detail_id')}")


def test_verify_buy():
    """Verify the new campaign shows up in open trades."""
    r = requests.get(f"{BASE}/api/trades/open", params={"portfolio": PORTFOLIO})
    data = r.json()
    match = [t for t in data if t.get("trade_id") == TEST_TRADE_ID]
    assert len(match) == 1, f"expected 1 match, got {len(match)}"
    t = match[0]
    assert t["ticker"] == TEST_TICKER, f"ticker mismatch: {t['ticker']}"
    assert float(t["shares"]) == 100, f"shares mismatch: {t['shares']}"
    assert t["status"].upper() == "OPEN", f"status mismatch: {t['status']}"


def test_log_buy_scalein():
    """Scale into the existing campaign."""
    r = requests.post(f"{BASE}/api/trades/buy", json={
        "portfolio": PORTFOLIO,
        "action_type": "scalein",
        "ticker": TEST_TICKER,
        "trade_id": TEST_TRADE_ID,
        "shares": 50,
        "price": 52.00,
        "rule": "br1.4 Double Bottom",
        "notes": "API test — scale in",
    })
    data = r.json()
    assert data.get("status") == "ok", f"scale-in failed: {data}"
    print(f"       trx_id={data['trx_id']}")


def test_verify_scalein():
    """Verify shares increased after scale-in."""
    r = requests.get(f"{BASE}/api/trades/open", params={"portfolio": PORTFOLIO})
    data = r.json()
    match = [t for t in data if t.get("trade_id") == TEST_TRADE_ID]
    assert len(match) == 1, f"expected 1 match, got {len(match)}"
    assert float(match[0]["shares"]) == 150, f"shares should be 150, got {match[0]['shares']}"


def test_log_sell_partial():
    """Partial sell — sell 50 of 150 shares."""
    r = requests.post(f"{BASE}/api/trades/sell", json={
        "portfolio": PORTFOLIO,
        "trade_id": TEST_TRADE_ID,
        "shares": 50,
        "price": 55.00,
        "rule": "sr10 Scale-Out T1 (-3%)",
        "notes": "API test — partial sell",
    })
    data = r.json()
    assert data.get("status") == "ok", f"partial sell failed: {data}"
    assert data.get("is_closed") == False, f"should still be open: {data}"
    assert float(data.get("remaining_shares", 0)) == 100, f"remaining should be 100: {data}"
    print(f"       trx_id={data['trx_id']}, realized_pl=${data.get('realized_pl', 0):.2f}, remaining={data.get('remaining_shares')}")


def test_log_sell_close():
    """Close the position — sell remaining 100 shares."""
    r = requests.post(f"{BASE}/api/trades/sell", json={
        "portfolio": PORTFOLIO,
        "trade_id": TEST_TRADE_ID,
        "shares": 100,
        "price": 48.00,
        "rule": "sr1 Capital Protection",
        "notes": "API test — close out",
    })
    data = r.json()
    assert data.get("status") == "ok", f"close sell failed: {data}"
    assert data.get("is_closed") == True, f"should be closed: {data}"
    print(f"       trx_id={data['trx_id']}, realized_pl=${data.get('realized_pl', 0):.2f}, closed={data.get('is_closed')}")


def test_verify_closed():
    """Verify trade is now CLOSED."""
    r = requests.get(f"{BASE}/api/trades/closed", params={"portfolio": PORTFOLIO, "limit": 200})
    data = r.json()
    match = [t for t in data if t.get("trade_id") == TEST_TRADE_ID]
    assert len(match) == 1, f"expected 1 closed match, got {len(match)}"
    assert match[0]["status"].upper() == "CLOSED", f"status should be CLOSED: {match[0]['status']}"


def test_trade_details():
    """Verify all 4 transactions exist (2 buys + 2 sells)."""
    r = requests.get(f"{BASE}/api/trades/details/{TEST_TRADE_ID}", params={"portfolio": PORTFOLIO})
    data = r.json()
    assert isinstance(data, list), f"expected list: {data}"
    assert len(data) == 4, f"expected 4 transactions, got {len(data)}"
    actions = [d.get("action", "").upper() for d in data]
    assert actions.count("BUY") == 2, f"expected 2 buys: {actions}"
    assert actions.count("SELL") == 2, f"expected 2 sells: {actions}"


def test_fundamentals_empty():
    """Fundamentals should be empty for test trade (no screenshot uploaded)."""
    r = requests.get(f"{BASE}/api/fundamentals/{TEST_TRADE_ID}", params={"portfolio": PORTFOLIO})
    assert r.status_code == 200, f"status {r.status_code}"
    data = r.json()
    assert isinstance(data, list) and len(data) == 0, f"expected empty list: {data}"


def test_delete_trade():
    """Delete the test trade."""
    r = requests.delete(f"{BASE}/api/trades/delete", params={
        "trade_id": TEST_TRADE_ID,
        "portfolio": PORTFOLIO,
    })
    data = r.json()
    assert data.get("status") == "ok", f"delete failed: {data}"


def test_verify_deleted():
    """Verify trade is gone from both open and closed."""
    r1 = requests.get(f"{BASE}/api/trades/open", params={"portfolio": PORTFOLIO})
    r2 = requests.get(f"{BASE}/api/trades/closed", params={"portfolio": PORTFOLIO, "limit": 500})
    all_trades = r1.json() + r2.json()
    match = [t for t in all_trades if t.get("trade_id") == TEST_TRADE_ID]
    assert len(match) == 0, f"trade still exists after delete: {match}"


test("Log Buy — new campaign", test_log_buy_new)
test("Verify — trade exists & OPEN", test_verify_buy)
test("Log Buy — scale in (+50 shs)", test_log_buy_scalein)
test("Verify — shares = 150", test_verify_scalein)
test("Log Sell — partial (50 shs)", test_log_sell_partial)
test("Log Sell — close out (100 shs)", test_log_sell_close)
test("Verify — trade CLOSED", test_verify_closed)
test("Verify — 4 transactions (2B + 2S)", test_trade_details)
test("Fundamentals — empty for test", test_fundamentals_empty)
test("Delete trade", test_delete_trade)
test("Verify — trade deleted", test_verify_deleted)


# ════════════════════════════════════════
# 3. EDGE CASES
# ════════════════════════════════════════
print("\n── Edge Cases ──")


def test_sell_nonexistent():
    """Selling a nonexistent trade should return error."""
    r = requests.post(f"{BASE}/api/trades/sell", json={
        "portfolio": PORTFOLIO, "trade_id": "DOESNOTEXIST-999",
        "shares": 10, "price": 100, "rule": "test",
    })
    data = r.json()
    assert "error" in data, f"expected error: {data}"


def test_buy_missing_fields():
    """Buy with missing required fields should return error."""
    r = requests.post(f"{BASE}/api/trades/buy", json={
        "portfolio": PORTFOLIO, "ticker": "", "trade_id": "", "shares": 0, "price": 0,
    })
    data = r.json()
    assert "error" in data, f"expected error: {data}"


def test_delete_nonexistent():
    """Deleting a nonexistent trade should not crash."""
    r = requests.delete(f"{BASE}/api/trades/delete", params={
        "trade_id": "DOESNOTEXIST-999", "portfolio": PORTFOLIO,
    })
    assert r.status_code == 200, f"status {r.status_code}"


test("Sell nonexistent trade → error", test_sell_nonexistent)
test("Buy with missing fields → error", test_buy_missing_fields)
test("Delete nonexistent trade → no crash", test_delete_nonexistent)


# ════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print(f"\nFailures:")
    for name, msg in errors:
        print(f"  ✗ {name}: {msg}")
print(f"{'='*60}\n")

sys.exit(1 if failed > 0 else 0)
