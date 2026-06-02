"""
sweep.py
========
Run a grid of cascade variants across a basket of historical winners and
print a side-by-side comparison.

Cascade variants (GREEN/QUICK/QUICKSAND/GD targets, as fractions of 1.0 unit):
  A "25/25/exit"  (current default) — 1.00 / 0.75 / 0.50 / 0.00
  B "15/15/exit"  gentler           — 1.00 / 0.85 / 0.70 / 0.00
  C "10/15/exit"  very gentle       — 1.00 / 0.90 / 0.75 / 0.00
  D "40/30/exit"  aggressive        — 1.00 / 0.60 / 0.30 / 0.00

Test basket (the four tickers we've already analyzed by hand):
  PLTR  2023-05-01
  VIAV  1998-12-02
  AXON  2003-07-22
  AAPL  2004-02-27
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mors_backtest import run

NAV = 500_000.0
SPY = "data/SPY_daily.csv"
DATA = "data"
MODE = "terminate"
OUT = None  # don't write per-variant CSVs from the sweep

CASCADES = [
    ("A 25/25/exit",  (1.00, 0.75, 0.50, 0.00)),
    ("B 15/15/exit",  (1.00, 0.85, 0.70, 0.00)),
    ("C 10/15/exit",  (1.00, 0.90, 0.75, 0.00)),
    ("D 40/30/exit",  (1.00, 0.60, 0.30, 0.00)),
]

TESTS = [
    ("PLTR",  "2023-05-01"),
    ("VIAV",  "1998-12-02"),
    ("AXON",  "2003-07-22"),
    ("AAPL",  "2004-02-27"),
    ("EBAY",  "1998-10-15"),
    ("GOOGL", "2004-09-15"),
    ("GOOGL", "2005-04-20"),
    ("SNDK",  "2025-08-27"),
]


def main():
    rows = []
    for ticker, start in TESTS:
        tkr_path = f"{DATA}/{ticker}_price_data.csv"
        for label, cascade in CASCADES:
            r = run(SPY, tkr_path, ticker, start, end=None, nav=NAV,
                    out_dir=OUT, mode=MODE, cascade=cascade, quiet=True)
            rows.append({
                "Ticker": ticker,
                "BuyReq": start,
                "Entry": str(r["entry_date"]),
                "Exit": str(r["exit_date"]),
                "Cascade": label,
                "Signals": r["signals"],
                "Realized $": int(round(r["realized"])),
                "Realized %": round(r["total_pct"], 1),
                "Peak %": round(r["peak_gain_pct"], 1),
                "Exit-Entry %": round(r["exit_vs_entry_pct"], 1),
                "% of Peak Captured": round(r["total_pct"] / r["peak_gain_pct"] * 100, 1)
                                       if r["peak_gain_pct"] > 0 else float("nan"),
            })

    df = pd.DataFrame(rows)

    print("=" * 110)
    print("CASCADE SWEEP — MO RS shadow, TERMINATE mode, NAV $500,000, unit = $100,000")
    print("=" * 110)

    # Per-test grid: group by (ticker, requested buy date) so multi-date tickers
    # don't collide.
    for ticker, start in TESTS:
        sub = df[(df.Ticker == ticker) & (df.BuyReq == start)].copy()
        entry = sub.iloc[0]["Entry"]
        exitd = sub.iloc[0]["Exit"]
        peak = sub.iloc[0]["Peak %"]
        exit_move = sub.iloc[0]["Exit-Entry %"]
        print(f"\n{ticker}   buy req {start} (entry {entry}) -> exit {exitd}   "
              f"|   peak gain {peak:+.1f}%   |   entry->exit {exit_move:+.1f}%")
        view = sub[["Cascade", "Signals", "Realized $", "Realized %", "% of Peak Captured"]]
        print(view.to_string(index=False))

    # Cross-ticker means per cascade
    print("\n" + "=" * 110)
    print("AGGREGATE — averaged across the basket")
    print("=" * 110)
    agg = df.groupby("Cascade").agg({
        "Realized %": "mean",
        "% of Peak Captured": "mean",
        "Signals": "mean",
    }).round(1)
    print(agg.to_string())

    # Save full grid to results/
    os.makedirs("results", exist_ok=True)
    out_path = "results/sweep_cascades.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[wrote] {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
