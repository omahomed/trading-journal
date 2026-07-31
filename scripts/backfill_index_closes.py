"""One-shot historical backfill for spy / nasdaq columns on trading_journal.

Root cause: nlv-entry.tsx only passes an explicit date to batchPrices when
the entry date is in the past. When the operator entered NLV for the
current trading day mid-session, batch_prices returned intraday levels
and those got persisted as the day's "close." Audit across 872 rows
found 188+ SPY rows and 4 NASDAQ rows corrupted this way, mostly on
CanSlim (the actively-managed account).

This script walks EVERY distinct day in trading_journal and calls the
same helper the /api/journal/refresh-index-closes endpoint uses, which
enforces yfinance's raw (auto_adjust=False) close across all portfolios.
Idempotent — re-running produces zero updates on already-clean data.

Guardrails:
  * SPY tolerance $0.10; NASDAQ tolerance $5. Below tolerance = leave alone
    (avoids gratuitous writes for dividend-adjustment noise).
  * Only touches spy + nasdaq columns; nothing else.
  * Every update logged; final summary shows total rows corrected per
    portfolio + per symbol.

Run: `python scripts/backfill_index_closes.py`  (dry run — no writes)
     `python scripts/backfill_index_closes.py --commit`  (writes)
"""
from __future__ import annotations
import os, sys
from pathlib import Path

REPO = Path("/Users/momacbookair/Developer/my_code")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
for line in (REPO / ".env").read_text().splitlines():
    if line.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = line.split("=", 1)[1].strip()
        break

import pandas as pd
import db_layer as db
from api.main import _ensure_official_index_closes

FOUNDER = "d7e8f9a0-1b2c-4d3e-8f4a-5b6c7d8e9f0a"
db.current_user_id.set(FOUNDER)


def all_journal_days() -> list:
    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT day FROM trading_journal
            WHERE (spy > 0 OR nasdaq > 0)
            ORDER BY day
        """)
        days = [r[0] for r in cur.fetchall()]
        cur.close()
    return days


def main():
    commit = "--commit" in sys.argv
    days = all_journal_days()
    print(f"Distinct journal days to check: {len(days)}")
    print(f"Mode: {'COMMIT' if commit else 'DRY-RUN'}")
    print()

    total_updates = 0
    per_port = {}
    per_col = {"spy": 0, "nasdaq": 0}
    failures = []
    day_count_shown = 0

    for i, day in enumerate(days, 1):
        if commit:
            res = _ensure_official_index_closes(day)
        else:
            # Dry-run: replicate the check without writing. Call the helper
            # inside a rollback transaction so no data mutates.
            with db.get_db_connection() as conn:
                # SAVEPOINT + ROLLBACK trick — helper commits internally, so
                # dry-run reads only. Simpler: fetch what WOULD change.
                cur = conn.cursor()
                # Use _fetch_historical_closes directly + compare to stored
                from api.main import _fetch_historical_closes
                prices = _fetch_historical_closes(["SPY", "^IXIC"], day)
                spy_off = prices.get("SPY"); ixic_off = prices.get("^IXIC")
                cur.execute("""
                    SELECT tj.id, p.name, tj.spy, tj.nasdaq
                    FROM trading_journal tj JOIN portfolios p ON p.id = tj.portfolio_id
                    WHERE tj.day = %s
                """, (day.isoformat() if hasattr(day, "isoformat") else str(day)[:10],))
                rows = cur.fetchall()
                cur.close()
                res = {"day": str(day), "spy_official": spy_off, "ixic_official": ixic_off,
                        "updates": []}
                for rid, port, sp, ndx in rows:
                    if spy_off is not None and sp is not None and abs(float(sp) - spy_off) > 0.10:
                        res["updates"].append({"portfolio": port, "column": "spy",
                                                "old": round(float(sp), 2), "new": spy_off})
                    if ixic_off is not None and ndx is not None and abs(float(ndx) - ixic_off) > 5.0:
                        res["updates"].append({"portfolio": port, "column": "nasdaq",
                                                "old": round(float(ndx), 2), "new": ixic_off})
        if res.get("error"):
            failures.append((day, res["error"]))
            continue
        for u in res["updates"]:
            total_updates += 1
            per_port[u["portfolio"]] = per_port.get(u["portfolio"], 0) + 1
            per_col[u["column"]] += 1
            if day_count_shown < 20:
                print(f"  {res['day']}  {u['portfolio']:<12}  {u['column']:<7}  "
                      f"{u['old']:>10.2f} → {u['new']:>10.2f}")
                day_count_shown += 1
                if day_count_shown == 20:
                    print("  ...")
        if i % 50 == 0:
            print(f"  ({i}/{len(days)} days scanned, {total_updates} updates so far)")

    print("\n" + "=" * 60)
    print(f"Total updates: {total_updates}")
    print(f"  by column:  {per_col}")
    print(f"  by portfolio:  {per_port}")
    if failures:
        print(f"\n{len(failures)} days had errors:")
        for d, err in failures[:10]:
            print(f"  {d}: {err}")
    if not commit:
        print("\nDry run only. Re-run with --commit to write.")


if __name__ == "__main__":
    main()
