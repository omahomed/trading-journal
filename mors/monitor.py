"""
monitor.py
==========
Daily monitor for live SR8 positions managed under the MO RS framework.

Reads positions.json, takes today's NLV from --nlv, replays each cascade from
its B1 date through today, and reports current tier + any action needed.

Anchoring invariant (2026-07-18 fix):
  SR8 trim TARGETS (Quick/Quicksand destination share count) are anchored to
  the campaign's activation-day NLV, NOT live NLV. Formula:

    quick_target_dollars = 0.10 × sr8_activation_nlv
    qs_target_dollars    = 0.05 × sr8_activation_nlv
    gd_target_dollars    = 0

  Live NLV is used ONLY for:
    - Display metrics (`current_pct_nlv` — "what % of live NLV am I now?")
    - Cap-restore (rebuild ceiling; cap-restore is a downside defense).

  Why: core_shares is fixed at activation. When live-NAV grew (say 2×), old
  "0.10 × live_NLV / price" would compute a target share count LARGER than
  the fixed core. The trim then reads as no-op ("already at target"), leaving
  cores undefended on valid signals. See regression case in
  tests/test_sr8_monitor.py (BE campaign 2026-06-26 fire — old formula
  gave target 319 shs vs 224 held; new formula gives 149 shs, a valid 75-shs
  trim).

Add-ons: allowed on SR8-tagged positions but belong to the "trim-first
cohort" — any shares above core_shares are trimmed before the cascade dips
into core toward the anchored target.

Cascade tier percentages (unchanged, applied to activation_nlv):
  GREEN     15%   (rebuild target only — GREEN never sells)
  QUICK     10%   trim floor
  QUICKSAND  5%   trim floor
  GD         0%   full exit

Usage:
  cd mors && python3 monitor.py --nlv 826486
"""

import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mors_backtest import run

# Single % NLV floor schedule, keyed by live cascade tier label. The 20-cas /
# 15-cas variant selection (audit: monitor.py:31-34 / 80-87) is gone — SR8
# positions get one schedule. Quick / QS / GD are TRIM floors (sell down to
# this %); GREEN's 15% is a REBUILD target, not a trim floor — Green never
# sells (a Green position above 15% is held by design; SR7 owns the excess).
TIER_NLV_FLOORS = {
    "GREEN":      15.00,
    "QUICK":      10.00,
    "QUICKSAND":   5.00,
    "GD":          0.00,
    # Aliases the engine may emit. Sub-entry / TERMINATED both terminal-ish;
    # mapping them keeps lookups defensive.
    "GREEN(sub-entry)": 15.00,
    "TERMINATED":  0.00,
}

CASCADE_SIGNALS = {"ENTRY", "GREEN", "QUICK", "QUICKSAND", "GD",
                   "TERMINATED", "GREEN(sub-entry)"}

# Anchor the data directory to THIS file's location so importers that
# don't `cd mors` first (e.g. the FastAPI server's SR8 Monitor endpoint)
# still resolve the SPY + per-ticker CSVs correctly. The CLI workflow
# (`cd mors && python3 monitor.py …`) still works because the absolute
# paths resolve the same way regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
SPY_PATH = os.path.join(_HERE, "data", "SPY_daily.csv")
DATA_DIR = os.path.join(_HERE, "data")
AT_TARGET_TOL = 500.0  # $ tolerance for "at target"


def load_positions(path):
    with open(path) as f:
        return json.load(f)


def analyze(pos, nlv, refresh=False, activation_nlv=None):
    """Replay + score one SR8 position for the day.

    `nlv` — live NLV (used ONLY for display metrics + cap-restore reference).
    `activation_nlv` — the position's SR8-activation-day NLV. When present,
      Quick/Quicksand trim targets anchor to it (see module docstring for
      why). Falls back to live `nlv` when NULL, for legacy positions that
      pre-date the backfill or activated before migration 048 shipped —
      when this fallback fires the audit line at the report level flags
      the divergence.

    Returns dict shape unchanged EXCEPT for two new fields:
      `activation_nlv` — echoed back on the payload for display
      `anchor_source` — 'activation' | 'live_fallback' — surfaced so
      the daily report can show which anchor drove the target.
    """
    ticker = pos["ticker"]
    tkr_path = f"{DATA_DIR}/{ticker}_price_data.csv"
    # SR8 positions always run the weekly 8/13/21 cascade (force_weekly=True)
    # — Phase 1 daily cascade is skipped entirely. The SR8 tag is the
    # activation gate; we don't re-derive a +50% cushion phase.
    res = run(SPY_PATH, tkr_path, ticker,
              start=str(pos["b1_date"]), end=None,
              nav=500_000.0, out_dir=None, mode="terminate",
              refresh=refresh, quiet=True,
              entry_px_override=pos["b1_price"],
              force_weekly=True)
    log = res["log"]

    cas = log[log["Signal"].isin(CASCADE_SIGNALS)]
    last = cas.iloc[-1]
    last_signal = last["Signal"]
    last_signal_date = last["Date"]
    current_price = float(res["exit_px"])
    last_bar_date = res["exit_date"]

    shares_held = pos["shares_held"]
    avg_price = pos["avg_price"]
    current_dollars = shares_held * current_price
    # Display metric: what % of LIVE NLV is this position now. Correctly
    # uses live nlv — this is a "where is my portfolio right now" number,
    # not a trim target.
    current_pct_nlv = current_dollars / nlv * 100.0 if nlv > 0 else 0.0

    # Live tier from the cascade ratchet (NOT the last emission in the log).
    # Falls back to GREEN for the corner case where the weekly cascade hasn't
    # seeded yet (no weekend bar between B1 and today — vanishingly rare for
    # SR8 positions since they require peak ≥ 50%).
    current_tier = str(res.get("current_tier_label") or "GREEN")
    tier_pct_nlv = TIER_NLV_FLOORS.get(current_tier, 0.0)

    # ── ANCHORED TARGET (this is the bug fix) ───────────────────────────
    # target_dollars uses the ACTIVATION-day NLV, not live NLV. When a
    # position's activation_nlv hasn't been backfilled yet, fall back to
    # live nlv (matches pre-fix behavior for those rows) and flag it so
    # the report shows the divergence.
    anchor_source: str
    if activation_nlv is not None and activation_nlv > 0:
        target_dollars = activation_nlv * tier_pct_nlv / 100.0
        anchor_source = "activation"
    else:
        target_dollars = nlv * tier_pct_nlv / 100.0
        anchor_source = "live_fallback"

    # GREEN never sells. The 15% NAV figure is a REBUILD target, not a trim
    # floor — a Green position currently above 15% is held (SR7 owns the
    # excess). Rebuild (Green BUY back up to 15%) is intentionally NOT
    # computed here — deferred to a later commit.
    if current_tier in ("GREEN", "GREEN(sub-entry)"):
        delta_dollars = 0.0
        delta_shares = 0
    else:
        delta_dollars = max(0.0, current_dollars - target_dollars)
        delta_shares = int(round(delta_dollars / current_price)) if current_price > 0 else 0

    unreal_dollars = (current_price - avg_price) * shares_held
    unreal_pct = (current_price / avg_price - 1.0) * 100.0 if avg_price > 0 else 0.0

    signal_today = (last_signal_date == last_bar_date) and (last_signal != "ENTRY")
    terminated = (last_signal == "TERMINATED") or current_tier == "GD"

    # Phase 1 vs 2 (pre-cushion vs latched-after-1.5×B1). Read off the
    # last bar of the full log so callers don't have to re-derive from
    # price ratios. The log explicitly stamps each bar's phase, including
    # the PHASE2 LATCH transition (mors_backtest.py:589).
    try:
        phase = int(log["Phase"].iloc[-1])
    except Exception:
        phase = 1

    return {
        "ticker": ticker,
        "shares_held": shares_held,
        "avg_price": avg_price,
        "current_price": current_price,
        "current_dollars": current_dollars,
        "current_pct_nlv": current_pct_nlv,
        "current_tier": current_tier,
        "tier_pct_nlv": tier_pct_nlv,
        "target_dollars": target_dollars,
        "delta_dollars": delta_dollars,
        "delta_shares": delta_shares,
        "unreal_dollars": unreal_dollars,
        "unreal_pct": unreal_pct,
        "last_signal": last_signal,
        "last_signal_date": last_signal_date,
        "last_bar_date": last_bar_date,
        "signal_today": signal_today,
        "terminated": terminated,
        "phase": phase,
        "activation_nlv": activation_nlv,
        "anchor_source": anchor_source,
    }


def action_text(r):
    if r["terminated"]:
        return f"EXIT FULLY — SELL all {r['shares_held']:,} sh  (weekly GD, campaign ends)"
    if r["signal_today"] and r["delta_dollars"] > AT_TARGET_TOL:
        return (f"TRIM {r['delta_shares']:,} sh  (~${r['delta_dollars']:,.0f})  "
                f"-> {r['tier_pct_nlv']:.2f}% NLV target")
    return "HOLD"


def is_actionable(r):
    if r["terminated"]:
        return True
    if r["signal_today"] and r["delta_dollars"] > AT_TARGET_TOL:
        return True
    return False


def fmt_hold_row(r):
    pl_sign = "+" if r["unreal_dollars"] >= 0 else ""
    # Anchor annotation surfaces the ladder's teeth at a glance. When
    # activation_nlv is present, show it + the derived Quick/QS targets
    # in shares. Fallback rows get a "[live]" tag so it's obvious the
    # anchor is missing and the trim numbers are subject to the NAV-
    # inflation bug this rewrite fixes.
    px = r["current_price"] or 0
    if r.get("activation_nlv") and px > 0:
        q_shs = int(round(0.10 * r["activation_nlv"] / px))
        qs_shs = int(round(0.05 * r["activation_nlv"] / px))
        anchor_tag = (
            f" | anchor ${r['activation_nlv']:>10,.0f} "
            f"(Q→{q_shs}sh  QS→{qs_shs}sh)"
        )
    else:
        anchor_tag = " | anchor [live-NAV fallback — backfill needed]"
    return (
        f"  {r['ticker']:<5} "
        f"{r['current_tier']:<10} "
        f"{r['current_pct_nlv']:>5.1f}% NLV  "
        f"(floor {r['tier_pct_nlv']:>5.2f}%)  | "
        f"{r['shares_held']:>6,} sh  "
        f"avg ${r['avg_price']:>8,.2f}  "
        f"now ${r['current_price']:>8,.2f}  | "
        f"P&L {pl_sign}${r['unreal_dollars']:>10,.0f}  ({pl_sign}{r['unreal_pct']:>5.1f}%)"
        f"{anchor_tag}"
    )


def fmt_action_row(r):
    base = fmt_hold_row(r)
    sig = f"Last signal: {r['last_signal']} on {r['last_signal_date']}"
    if r["signal_today"]:
        sig += "   ** TODAY **"
    return f"{base}\n        {sig}\n        ACTION: {action_text(r)}\n"


def format_report(results, nlv, today_str):
    lines = []
    lines.append("=" * 140)
    lines.append(f"MO RS Daily Monitor — {today_str}")
    lines.append(f"NLV: ${nlv:,.0f}")
    lines.append("=" * 140)

    actionable = [r for r in results if is_actionable(r)]
    hold = [r for r in results if not is_actionable(r)]

    if actionable:
        lines.append("")
        lines.append(">>> ACTION NEEDED <<<")
        lines.append("-" * 140)
        for r in actionable:
            lines.append(fmt_action_row(r))
    else:
        lines.append("")
        lines.append("ACTION NEEDED: none today.")

    if hold:
        lines.append("")
        lines.append("HOLD — no cascade action today")
        lines.append("-" * 140)
        for r in hold:
            lines.append(fmt_hold_row(r))

    lines.append("=" * 140)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlv", type=float, required=True, help="Today's NLV in dollars")
    ap.add_argument("--positions", default="positions.json")
    ap.add_argument("--refresh", action="store_true",
                    help="force yfinance re-fetch for each ticker")
    args = ap.parse_args()

    positions = load_positions(args.positions)
    if not positions:
        print(f"No positions in {args.positions}.")
        return

    # Pass through activation_nlv from positions.json when present. The
    # DB-driven API caller (_sr8_load_db_positions in api/main.py) fills
    # this from trades_summary.sr8_activation_nlv; the CLI's positions.json
    # is optional — legacy files without the field cleanly fall back.
    results = [
        analyze(
            p, args.nlv, refresh=args.refresh,
            activation_nlv=p.get("activation_nlv"),
        )
        for p in positions
    ]
    today_str = date.today().isoformat()
    report = format_report(results, args.nlv, today_str)
    print(report)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"monitor_{today_str}.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
