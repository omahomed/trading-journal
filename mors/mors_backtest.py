"""
mors_backtest.py
=================
MO RS 2.0 shadow backtest engine.

Tests the pure MO RS exit framework on a single ticker against a stored SPY
benchmark. SPY is the "spine" and lives in data/spy_daily.csv — you never
re-feed it; the engine reads it, checks coverage against your requested end
date, and tells you to append rows when your test window runs past it.

INDICATOR (MO RS 2.0, per MO_RS.pine)
  RS line = ticker_close / SPY_close on common trading days.
  Three EMAs per timeframe:
    daily   21 / 34 / 50
    weekly   8 / 13 / 21   (Friday close)
    monthly  5 /  8 / 13   (informational only — never trades)
  4-tier cascade, fires once per cycle, re-arms only on GREEN:
    GREEN     RS above all three MAs
    QUICK     RS crosses under fast MA
    QUICKSAND RS crosses under mid MA
    GD        RS crosses under slow MA

SHADOW RULES
  Entry      User-specified buy date. First trading day at/after that date is B1.
  Both phases use the same NLV-anchored cascade:
    GREEN -> 20% NLV   QUICK -> 15% NLV   QUICKSAND -> 10% NLV   GD -> exit
  NLV evolves as: NLV = initial_NAV + realized P&L  (no MTM; clean accounting)
  At each cascade signal: target_$ = NLV * 0.20 * cascade_frac, then
                          target_shares = target_$ / current_price.
  This means trims/refills scale with current NLV — a winner that ballooned past
  20% gets trimmed BACK to 15% on QUICK (not just trimmed 25% of B1 shares).
  Phase 1  cushion < +50% from B1  -> evaluate DAILY MO RS, act on close
  Phase 2  cushion >= +50% (latched intra-bar at 1.50 x B1) -> WEEKLY MO RS, Fri close
  LIFO accounting on trims; realized P&L compounds the unit on future signals.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

DEFAULT_NAV = 500_000.0     # portfolio NAV; entry deploys POSITION_PCT of it. Override with --nav.
POSITION_PCT = 0.20         # shadow enters at 20% of NAV (1.0 unit)
WARMUP_DAYS = 50            # daily RS bars required before an entry is allowed
PHASE2_MULT = 1.50          # cushion threshold to latch weekly governance
DEFAULT_CASCADE = (1.00, 0.75, 0.50, 0.00)   # GREEN, QUICK, QUICKSAND, GD targets (frac of unit)
SR7_TRIGGER_DISCOUNT = 0.99                  # SR7 trigger = arming bar's low * 0.99
DEFAULT_WEEKLY_EMAS = (8, 13, 21)            # weekly RS fast/mid/slow EMAs (Fibonacci)
DEFAULT_DAILY_EMAS = (21, 34, 50)            # daily RS fast/mid/slow EMAs

# Weekly cascade variants — each = (EMAs, target_fractions_by_tier, tier_labels)
# tier_labels: [GREEN, ..., GD]. target fractions: 1.0 (GREEN) down to 0.0 (GD).
WEEKLY_VARIANTS = {
    "default": {
        "emas": (8, 13, 21),
        "targets": (1.00, 0.75, 0.50, 0.00),
        "labels": ("GREEN", "QUICK", "QUICKSAND", "GD"),
    },
    "v1": {  # 5/8/13/21 — add EARLY tier at 18% NLV
        "emas": (5, 8, 13, 21),
        "targets": (1.00, 0.90, 0.75, 0.50, 0.00),
        "labels": ("GREEN", "EARLY", "QUICK", "QUICKSAND", "GD"),
    },
    "v2": {  # 3/5/8/13/21 — two new tiers (19%, 17% NLV)
        "emas": (3, 5, 8, 13, 21),
        "targets": (1.00, 0.95, 0.85, 0.75, 0.50, 0.00),
        "labels": ("GREEN", "EARLY1", "EARLY2", "QUICK", "QUICKSAND", "GD"),
    },
    "v3": {  # 3/8/13/21 — add very-early tier (skip 5)
        "emas": (3, 8, 13, 21),
        "targets": (1.00, 0.90, 0.75, 0.50, 0.00),
        "labels": ("GREEN", "EARLY", "QUICK", "QUICKSAND", "GD"),
    },
    # Aggressive variants: keep 8/13/21 MAs but cut deeper at each trim tier
    "va": {  # aggressive: keep 50% on QUICK (10% NLV), 30% on QUICKSAND (6% NLV)
        "emas": (8, 13, 21),
        "targets": (1.00, 0.50, 0.30, 0.00),
        "labels": ("GREEN", "QUICK", "QUICKSAND", "GD"),
    },
    "va2": {  # 10%/5%/exit: down to 10% NLV on QUICK, 5% NLV on QUICKSAND, exit on GD
        "emas": (8, 13, 21),
        "targets": (1.00, 0.50, 0.25, 0.00),
        "labels": ("GREEN", "QUICK", "QUICKSAND", "GD"),
    },
    "vaa": {  # very aggressive: keep 50% on QUICK, FULL EXIT on QUICKSAND
        "emas": (8, 13, 21),
        "targets": (1.00, 0.50, 0.00, 0.00),
        "labels": ("GREEN", "QUICK", "QUICKSAND", "GD"),
    },
    "vk": {  # kill switch: full exit on the FIRST weekly QUICK
        "emas": (8, 13, 21),
        "targets": (1.00, 0.00, 0.00, 0.00),
        "labels": ("GREEN", "QUICK", "QUICKSAND", "GD"),
    },
}


# ----------------------------------------------------------------------------- loaders
def fetch_yf(symbol, dest_path, start="1975-01-01"):
    """Download OHLCV from yfinance and save to dest_path (yfinance-style CSV).
    Mirrors the stock_scraper2 approach: clean Date column, OHLCV only.
    """
    import yfinance as yf
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"[fetch] {symbol} from yfinance ({start} -> {end}) -> {dest_path}")
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    df.to_csv(dest_path, index=False)
    print(f"[fetch] saved {len(df)} rows to {dest_path}")


def ensure_data(path, symbol, refresh=False):
    """Fetch via yfinance if the CSV is missing or --refresh is set."""
    if refresh or not os.path.exists(path):
        fetch_yf(symbol, path)


def load_price_csv(path):
    """Robustly load a yfinance-style CSV, dropping the junk ticker header row."""
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])                 # kills the ",SPY,SPY,..." row
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    return df[["Date", "Open", "High", "Low", "Close"]]


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


# ----------------------------------------------------------------------------- cascade
class Cascade:
    """Once-per-cycle ratchet. Advances only; resets to GREEN on full reclaim.

    Generalized for N MAs (fast -> slow). With N MAs there are N+1 tiers,
    indexed 0..N. Tier 0 = "above all MAs" (GREEN); tier i = "RS below the
    first i MAs". tier_labels maps tier index -> human-readable signal name.
    """
    def __init__(self, tier_labels=("GREEN", "QUICK", "QUICKSAND", "GD")):
        self.tier = 0
        self.greened = False
        self.tier_labels = tuple(tier_labels)

    def step(self, rs, mas):
        if pd.isna(rs) or any(pd.isna(m) for m in mas):
            return None
        if all(rs > m for m in mas):
            if self.tier != 0 or not self.greened:
                self.tier, self.greened = 0, True
                return self.tier_labels[0]
            return None
        if not self.greened:
            return None
        target = 0
        for i, m in enumerate(mas, start=1):
            if rs < m:
                target = max(target, i)
        if target > self.tier:
            self.tier = target
            return self.tier_labels[target]
        return None

    def seed(self, rs, mas):
        """Set baseline state WITHOUT emitting a signal (used at phase switch)."""
        if pd.isna(rs) or any(pd.isna(m) for m in mas):
            return
        if all(rs > m for m in mas):
            self.tier, self.greened = 0, True
        else:
            t = 0
            for i, m in enumerate(mas, start=1):
                if rs < m:
                    t = max(t, i)
            self.tier = t
            self.greened = True   # treat as already in a cycle so we only act on NEW crosses


# ----------------------------------------------------------------------------- engine
def build_frame(spy, tkr, weekly_emas=DEFAULT_WEEKLY_EMAS, daily_emas=DEFAULT_DAILY_EMAS):
    m = pd.merge(tkr, spy[["Date", "Close"]].rename(columns={"Close": "SPY"}),
                 on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    m["RS"] = m["Close"] / m["SPY"]
    # daily EMAs on RS (fast / mid / slow)
    m["d_f"] = ema(m.RS, daily_emas[0])
    m["d_m"] = ema(m.RS, daily_emas[1])
    m["d_s"] = ema(m.RS, daily_emas[2])
    # daily 8 EMA on PRICE — used by SR6 "8e Momentum Trim"
    m["px8"] = ema(m.Close, 8)
    # daily 21 EMA on PRICE (Close) — used by SR7 "Holding Winners" exit
    m["px21"] = ema(m.Close, 21)
    # daily 50 SMA on PRICE and ATR(14) — used by ATR-extension exit
    m["px50"] = m.Close.rolling(50, min_periods=1).mean()
    prev_close = m.Close.shift(1)
    tr = pd.concat([(m.High - m.Low).abs(),
                    (m.High - prev_close).abs(),
                    (m.Low - prev_close).abs()], axis=1).max(axis=1)
    m["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    # week-ending flag (last trading day of each ISO week)
    iso = m["Date"].dt.isocalendar()
    m["wk"] = iso.year.astype(str) + "-" + iso.week.astype(str)
    m["is_weekend"] = m["Date"] == m.groupby("wk")["Date"].transform("max")
    # weekly RS series + N EMAs (configurable Fibonacci ladder), mapped back to week-ending date.
    # Columns are w_0, w_1, ..., w_{N-1} (fast -> slow).
    wk = m[m.is_weekend].copy()
    wk_cols = []
    for i, period in enumerate(weekly_emas):
        col = f"w_{i}"
        wk[col] = ema(wk.RS, period)
        wk_cols.append(col)
    m = m.merge(wk[["Date"] + wk_cols], on="Date", how="left")
    # monthly (informational)
    me = m.groupby(m["Date"].dt.to_period("M"))["Date"].transform("max")
    m["is_monthend"] = m["Date"] == me
    mo = m[m.is_monthend].copy()
    mo["m_f"], mo["m_m"], mo["m_s"] = ema(mo.RS, 5), ema(mo.RS, 8), ema(mo.RS, 13)
    m = m.merge(mo[["Date", "m_f", "m_m", "m_s"]], on="Date", how="left")
    return m


def run(spy_path, tkr_path, ticker, start, end, nav=DEFAULT_NAV, out_dir=None,
        mode="terminate", refresh=False, cascade=DEFAULT_CASCADE, quiet=False,
        entry_px_override=None, sr7=True, sr7_arm_bars=2, atr_trim=None,
        weekly_variant="default", phase2_daily=False,
        sr6=False, sr6_activate_bars=10):
    unit = POSITION_PCT * nav          # dollars deployed at entry (1.0 unit)
    # Daily cascade always uses the legacy 4-tier (--cascade flag still applies here)
    daily_target_frac = {"GREEN": cascade[0], "QUICK": cascade[1],
                         "QUICKSAND": cascade[2], "GD": cascade[3]}
    # Weekly cascade: variant determines EMAs, targets, and tier labels
    if weekly_variant not in WEEKLY_VARIANTS:
        raise ValueError(f"unknown weekly_variant {weekly_variant}; choose from "
                         f"{list(WEEKLY_VARIANTS)}")
    wv = WEEKLY_VARIANTS[weekly_variant]
    weekly_emas = wv["emas"]
    weekly_tier_labels = wv["labels"]
    weekly_target_frac = dict(zip(wv["labels"], wv["targets"]))
    n_weekly_emas = len(weekly_emas)
    weekly_cols = [f"w_{i}" for i in range(n_weekly_emas)]

    ensure_data(spy_path, "SPY", refresh=refresh)
    ensure_data(tkr_path, ticker, refresh=refresh)
    spy = load_price_csv(spy_path)
    tkr = load_price_csv(tkr_path)

    # ---- SPY coverage check (the "ask for new data" behavior) ----
    spy_last = spy["Date"].max()
    end_ts = pd.to_datetime(end) if end else min(spy_last, tkr["Date"].max())
    if end_ts > spy_last:
        print(f"\n[STOP] SPY data only covers through {spy_last.date()}.")
        print(f"       Requested end date is {end_ts.date()}.")
        print(f"       Append SPY rows through {end_ts.date()} to data/spy_daily.csv, then re-run.\n")
        sys.exit(2)

    m = build_frame(spy, tkr, weekly_emas=weekly_emas)
    m = m[(m.Date <= end_ts)].reset_index(drop=True)
    start_ts = pd.to_datetime(start)

    # ---- find entry: first trading day AT/AFTER the buy date ----
    # The user's buy date is the entry, signal state notwithstanding. MO RS only
    # governs the hold/exit/re-entry behavior from this bar forward.
    cand = m.index[m.Date >= start_ts]
    if len(cand) == 0:
        print(f"No trading day at/after {start_ts.date()} within data. No trade.")
        return
    entry_idx = int(cand[0])
    if entry_idx < WARMUP_DAYS:
        print(f"[warn] entry is at bar {entry_idx} of common data; EMAs may be unreliable "
              f"(want >= {WARMUP_DAYS} prior bars).")

    entry = m.loc[entry_idx]
    entry_px = float(entry_px_override) if entry_px_override is not None else float(entry.Close)
    b1_entry_px = entry_px             # B1, preserved for final summary
    initial_shares = unit / entry_px   # B1 deploy: 20% of NAV worth at entry price

    # ---- position state ----
    lots = [[initial_shares, entry_px]]   # LIFO stack of [shares, cost]
    cash = 0.0
    realized = 0.0
    peak = float(entry.High)
    peak_date = entry.Date
    phase = 1
    awaiting_new_entry = False         # set True after weekly GD; cleared on next daily GREEN
    current_cascade_frac = 1.0         # tracks last cascade-signal frac (1.0/0.75/0.50/0.0)
    # SR7 "Holding Winners" state: WATCHING / ARMED. FIRE on intraday low < trigger.
    # `sr7_arm_bars` = how many consecutive closes below 21 EMA needed to arm.
    sr7_state = "WATCHING"
    sr7_arming_low = None
    sr7_trigger = None
    sr7_consec_below = 0  # consecutive closes below px21 (for sr7_arm_bars > 1)
    # ATR-extension: trim to core when Close > px50 + N*ATR (Phase 2 only).
    # `atr_armed` starts True; fires once when threshold crossed; re-arms when
    # extension drops back below threshold.
    atr_armed = True
    # SR6 "8e Momentum Trim" state machine:
    # INACTIVE: counting consecutive closes above 8 EMA, waiting for sr6_activate_bars
    # WATCHING: activated, waiting for first close < 8 EMA -> ARMED
    # ARMED: trigger set; fire on intraday low < trigger; disarm on close > 8 EMA
    sr6_state = "INACTIVE"
    sr6_consec_above = 0
    sr6_arming_low = None
    sr6_trigger = None
    log = []

    def cur_shares():
        return sum(l[0] for l in lots)

    def target_to(frac, px, date, sig, tf, rsrow):
        nonlocal cash, realized, current_cascade_frac
        # NLV-anchored sizing: target dollars = NLV * 20% * cascade_frac
        nlv_pre = nav + realized
        tgt_dollars = max(0.0, nlv_pre * POSITION_PCT * frac)
        tgt = tgt_dollars / px if px > 0 else 0.0
        held = cur_shares()
        action = ""
        shares_delta = 0.0
        trade_val = 0.0
        realized_bar = 0.0
        if tgt < held - 1e-9:                       # SELL down (LIFO)
            to_sell = held - tgt
            sold_val = 0.0; cost = 0.0; rem = to_sell
            while rem > 1e-9 and lots:
                s, c = lots[-1]
                take = min(s, rem)
                sold_val += take * px; cost += take * c
                s -= take; rem -= take
                if s <= 1e-9: lots.pop()
                else: lots[-1][0] = s
            cash += sold_val
            realized_bar = sold_val - cost
            realized += realized_bar
            action = f"SELL -> {frac*100:.0f}% unit"
            shares_delta = -to_sell
            trade_val = -sold_val
        elif tgt > held + 1e-9:                     # BUY back up
            to_buy = tgt - held
            cost = to_buy * px
            cash -= cost
            lots.append([to_buy, px])
            action = f"BUY -> {frac*100:.0f}% unit"
            shares_delta = to_buy
            trade_val = cost
        else:
            action = "no change"
        current_cascade_frac = frac
        held_now = cur_shares()
        nlv_post = nav + realized
        pct_nlv = (held_now * px / nlv_post * 100.0) if nlv_post > 0 else 0.0
        log.append({
            "Date": date.date(), "Phase": phase, "TF": tf, "Signal": sig, "Action": action,
            "SharesDelta": round(shares_delta, 0),
            "Price": round(px, 4),
            "TradeVal$": round(trade_val, 0),
            "SharesHeld": round(held_now, 0),
            "Position$": round(held_now * px, 0),
            "Unit%": round(frac * 100, 1),
            "%NLV": round(pct_nlv, 2),
            "RS": round(rsrow["RS"], 5),
            "RealBar$": round(realized_bar, 0),
            "CumReal$": round(realized, 0),
            "CumReal%": round(realized / unit * 100, 2),
        })

    def trim_to_core(label, px, date, rsrow):
        """Generic 'trim to core (20% NLV)' helper used by SR7 and ATR-extension.
        LIFO. No-op log if position already at/below core."""
        nonlocal cash, realized
        nlv_pre = nav + realized
        core_dollars = nlv_pre * POSITION_PCT
        held = cur_shares()
        held_value = held * px
        if held_value <= core_dollars + 1e-6:
            nlv_post = nav + realized
            pct_nlv = (held * px / nlv_post * 100.0) if nlv_post > 0 else 0.0
            log.append({
                "Date": date.date(), "Phase": phase, "TF": "daily",
                "Signal": label, "Action": "no trim (already <= core)",
                "SharesDelta": 0.0,
                "Price": round(px, 4),
                "TradeVal$": 0.0,
                "SharesHeld": round(held, 0),
                "Position$": round(held * px, 0),
                "Unit%": round(current_cascade_frac * 100, 1),
                "%NLV": round(pct_nlv, 2),
                "RS": round(rsrow["RS"], 5),
                "RealBar$": 0.0,
                "CumReal$": round(realized, 0),
                "CumReal%": round(realized / unit * 100, 2),
            })
            return
        tgt_shares = core_dollars / px
        to_sell = held - tgt_shares
        sold_val = 0.0; cost = 0.0; rem = to_sell
        while rem > 1e-9 and lots:
            s, c = lots[-1]
            take = min(s, rem)
            sold_val += take * px; cost += take * c
            s -= take; rem -= take
            if s <= 1e-9: lots.pop()
            else: lots[-1][0] = s
        cash += sold_val
        realized_bar = sold_val - cost
        realized += realized_bar
        held_now = cur_shares()
        nlv_post = nav + realized
        pct_nlv = (held_now * px / nlv_post * 100.0) if nlv_post > 0 else 0.0
        log.append({
            "Date": date.date(), "Phase": phase, "TF": "daily",
            "Signal": label, "Action": "SELL -> core (20% NLV)",
            "SharesDelta": round(-to_sell, 0),
            "Price": round(px, 4),
            "TradeVal$": round(-sold_val, 0),
            "SharesHeld": round(held_now, 0),
            "Position$": round(held_now * px, 0),
            "Unit%": round(current_cascade_frac * 100, 1),
            "%NLV": round(pct_nlv, 2),
            "RS": round(rsrow["RS"], 5),
            "RealBar$": round(realized_bar, 0),
            "CumReal$": round(realized, 0),
            "CumReal%": round(realized / unit * 100, 2),
        })

    def sr7_trim(px, date, rsrow):
        trim_to_core("SR7", px, date, rsrow)

    def trim_by_frac(label, frac, px, date, rsrow):
        """Sell `frac` of CURRENT shares (e.g., frac=0.25 sells 25%). LIFO."""
        nonlocal cash, realized
        held = cur_shares()
        if held <= 1e-9 or frac <= 0:
            nlv_post = nav + realized
            log.append({
                "Date": date.date(), "Phase": phase, "TF": "daily",
                "Signal": label, "Action": "no trim (no position)",
                "SharesDelta": 0.0,
                "Price": round(px, 4),
                "TradeVal$": 0.0,
                "SharesHeld": 0.0,
                "Position$": 0.0,
                "Unit%": round(current_cascade_frac * 100, 1),
                "%NLV": 0.0,
                "RS": round(rsrow["RS"], 5),
                "RealBar$": 0.0,
                "CumReal$": round(realized, 0),
                "CumReal%": round(realized / unit * 100, 2),
            })
            return
        to_sell = held * frac
        sold_val = 0.0; cost = 0.0; rem = to_sell
        while rem > 1e-9 and lots:
            s, c = lots[-1]
            take = min(s, rem)
            sold_val += take * px; cost += take * c
            s -= take; rem -= take
            if s <= 1e-9: lots.pop()
            else: lots[-1][0] = s
        cash += sold_val
        realized_bar = sold_val - cost
        realized += realized_bar
        held_now = cur_shares()
        nlv_post = nav + realized
        pct_nlv = (held_now * px / nlv_post * 100.0) if nlv_post > 0 else 0.0
        log.append({
            "Date": date.date(), "Phase": phase, "TF": "daily",
            "Signal": label, "Action": f"SELL {frac*100:.0f}% of position",
            "SharesDelta": round(-to_sell, 0),
            "Price": round(px, 4),
            "TradeVal$": round(-sold_val, 0),
            "SharesHeld": round(held_now, 0),
            "Position$": round(held_now * px, 0),
            "Unit%": round(current_cascade_frac * 100, 1),
            "%NLV": round(pct_nlv, 2),
            "RS": round(rsrow["RS"], 5),
            "RealBar$": round(realized_bar, 0),
            "CumReal$": round(realized, 0),
            "CumReal%": round(realized / unit * 100, 2),
        })

    # log entry — initial deploy is exactly 20% NAV at B1
    log.append({
        "Date": entry.Date.date(), "Phase": 1, "TF": "daily", "Signal": "ENTRY",
        "Action": "BUY -> 100% unit",
        "SharesDelta": round(initial_shares, 0),
        "Price": round(entry_px, 4),
        "TradeVal$": round(initial_shares * entry_px, 0),
        "SharesHeld": round(initial_shares, 0),
        "Position$": round(initial_shares * entry_px, 0),
        "Unit%": 100.0,
        "%NLV": round(POSITION_PCT * 100.0, 2),
        "RS": round(entry.RS, 5),
        "RealBar$": 0.0,
        "CumReal$": 0.0,
        "CumReal%": 0.0,
    })

    d_cas = Cascade(); d_cas.seed(entry.RS, [entry.d_f, entry.d_m, entry.d_s])
    w_cas = Cascade(tier_labels=weekly_tier_labels)
    w_seeded = False
    last_idx = len(m) - 1  # effective end-of-test bar; updated to the GD bar on terminate

    for i in range(entry_idx + 1, len(m)):
        r = m.loc[i]
        if float(r.High) > peak:
            peak = float(r.High)
            peak_date = r.Date

        # ATR-extension trim — Phase 2 only. Fires when Close > 50 SMA + N * ATR(14).
        # Re-arms when extension drops back below threshold (so it doesn't fire
        # every bar while price stays extended).
        if (atr_trim is not None and phase == 2 and cur_shares() > 1e-9
                and not pd.isna(r.px50) and not pd.isna(r.atr14) and r.atr14 > 0):
            extension = float(r.Close) - float(r.px50)
            threshold = atr_trim * float(r.atr14)
            if extension > threshold and atr_armed:
                trim_to_core("ATR", float(r.Close), r.Date, r)
                atr_armed = False
            elif extension <= threshold and not atr_armed:
                atr_armed = True

        # SR6 "8e Momentum Trim" — activation gate (N consec closes above 8 EMA),
        # then arm on first close below 8 EMA, fire on intraday low < trigger.
        # Action: trim 25% (or sr6_trim_frac) of current shares.
        if sr6 and not pd.isna(r.px8):
            close_above_8 = float(r.Close) > float(r.px8)
            close_below_8 = float(r.Close) < float(r.px8)
            if sr6_state == "INACTIVE":
                if close_above_8:
                    sr6_consec_above += 1
                    if sr6_consec_above >= sr6_activate_bars:
                        sr6_state = "WATCHING"
                else:
                    sr6_consec_above = 0
            elif sr6_state == "ARMED" and float(r.Low) < sr6_trigger:
                # Action on fire: trim back to 20% NLV core (same as SR7).
                trim_to_core("SR6", float(r.Close), r.Date, r)
                sr6_state = "WATCHING"
                sr6_arming_low = None
                sr6_trigger = None
            elif sr6_state == "ARMED" and close_above_8:
                # Single close back above 8 EMA -> disarm
                sr6_state = "WATCHING"
                sr6_arming_low = None
                sr6_trigger = None
            elif sr6_state == "WATCHING" and close_below_8:
                sr6_state = "ARMED"
                sr6_arming_low = float(r.Low)
                sr6_trigger = sr6_arming_low * SR7_TRIGGER_DISCOUNT  # same 0.99 discount

        # SR7 "Holding Winners" — daily 21 EMA violation, proactive trim to core.
        # Order: check FIRE first (intraday low breaks trigger), else update state.
        # Requires `sr7_arm_bars` consecutive closes below px21 to arm.
        if sr7 and cur_shares() > 1e-9 and not pd.isna(r.px21):
            close_below = float(r.Close) < float(r.px21)
            if sr7_state == "ARMED" and float(r.Low) < sr7_trigger:
                sr7_trim(float(r.Close), r.Date, r)
                sr7_state = "WATCHING"
                sr7_arming_low = None
                sr7_trigger = None
                sr7_consec_below = 0
            elif sr7_state == "ARMED" and not close_below:
                # Single close back above 21 EMA -> disarm
                sr7_state = "WATCHING"
                sr7_arming_low = None
                sr7_trigger = None
                sr7_consec_below = 0
            elif sr7_state == "WATCHING":
                if close_below:
                    sr7_consec_below += 1
                    if sr7_consec_below >= sr7_arm_bars:
                        sr7_state = "ARMED"
                        sr7_arming_low = float(r.Low)
                        sr7_trigger = sr7_arming_low * SR7_TRIGGER_DISCOUNT
                else:
                    sr7_consec_below = 0  # streak broken

        # phase latch (intra-bar high >= 1.5 x current sub-campaign entry_px)
        # blocked while awaiting a new sub-campaign GREEN (stale entry_px must not latch)
        if phase == 1 and not awaiting_new_entry and float(r.High) >= PHASE2_MULT * entry_px:
            phase = 2
            held_now = cur_shares()
            close_px = float(r.Close)
            nlv_now = nav + realized
            pct_nlv = (held_now * close_px / nlv_now * 100.0) if nlv_now > 0 else 0.0
            log.append({
                "Date": r.Date.date(), "Phase": 2, "TF": "—",
                "Signal": "PHASE2 LATCH", "Action": "(weekly governance starts)",
                "SharesDelta": 0.0,
                "Price": round(close_px, 4),
                "TradeVal$": 0.0,
                "SharesHeld": round(held_now, 0),
                "Position$": round(held_now * close_px, 0),
                "Unit%": round(current_cascade_frac * 100, 1),
                "%NLV": round(pct_nlv, 2),
                "RS": round(r.RS, 5),
                "RealBar$": 0.0,
                "CumReal$": round(realized, 0),
                "CumReal%": round(realized / unit * 100, 2),
            })

        if phase == 1:
            sig = d_cas.step(r.RS, [r.d_f, r.d_m, r.d_s])
            if sig:
                # Sub-campaign restart: first daily GREEN after a weekly-GD revert.
                # Under NLV-anchored sizing, target_to() naturally deploys 20% of
                # (NAV + realized) at this bar. We only need to reset entry_px so
                # the Phase 2 cushion threshold re-anchors to the new entry.
                if awaiting_new_entry and sig == "GREEN":
                    entry_px = float(r.Close)
                    awaiting_new_entry = False
                    peak = float(r.High)
                    target_to(daily_target_frac[sig], float(r.Close), r.Date,
                              "GREEN(sub-entry)", "daily", r)
                else:
                    target_to(daily_target_frac[sig], float(r.Close), r.Date, sig, "daily", r)
        else:
            # Phase 2 — optionally evaluate the daily cascade ALONGSIDE weekly
            # (Scenario A: "scale in faster" on the downside).
            if phase2_daily:
                sig_d = d_cas.step(r.RS, [r.d_f, r.d_m, r.d_s])
                if sig_d:
                    target_to(daily_target_frac[sig_d], float(r.Close), r.Date,
                              sig_d, "daily", r)
            # seed weekly cascade once, on first weekly bar in phase 2, no action
            wk_mas = [r[c] for c in weekly_cols]
            if not w_seeded and r.is_weekend and not any(pd.isna(m) for m in wk_mas):
                w_cas.seed(r.RS, wk_mas)
                w_seeded = True
                continue
            if r.is_weekend and w_seeded:
                sig = w_cas.step(r.RS, wk_mas)
                if sig:
                    target_to(weekly_target_frac[sig], float(r.Close), r.Date, sig, "weekly", r)
                    if sig == "GD":
                        if mode == "terminate":
                            # Campaign over: position is flat, no re-entry will happen.
                            log.append({
                                "Date": r.Date.date(), "Phase": 2, "TF": "—",
                                "Signal": "TERMINATED",
                                "Action": "(weekly GD -> campaign ends, no re-entry)",
                                "SharesDelta": 0.0,
                                "Price": round(float(r.Close), 4),
                                "TradeVal$": 0.0,
                                "SharesHeld": 0.0,
                                "Position$": 0.0,
                                "Unit%": 0.0,
                                "%NLV": 0.0,
                                "RS": round(r.RS, 5),
                                "RealBar$": 0.0,
                                "CumReal$": round(realized, 0),
                                "CumReal%": round(realized / unit * 100, 2),
                            })
                            last_idx = i
                            break
                        elif mode == "revert":
                            phase = 1
                            awaiting_new_entry = True
                            d_cas = Cascade()
                            w_cas = Cascade(tier_labels=weekly_tier_labels)
                            w_seeded = False
                            log.append({
                                "Date": r.Date.date(), "Phase": 1, "TF": "—",
                                "Signal": "PHASE1 REVERT",
                                "Action": "(weekly GD -> revert to daily, await GREEN)",
                                "SharesDelta": 0.0,
                                "Price": round(float(r.Close), 4),
                                "TradeVal$": 0.0,
                                "SharesHeld": 0.0,
                                "Position$": 0.0,
                                "Unit%": 0.0,
                                "%NLV": 0.0,
                                "RS": round(r.RS, 5),
                                "RealBar$": 0.0,
                                "CumReal$": round(realized, 0),
                                "CumReal%": round(realized / unit * 100, 2),
                            })
                        # mode == "legacy": no special action; cascade re-arms on next GREEN

        if cur_shares() <= 1e-9 and not lots:
            pass  # flat; stay in loop, GREEN can re-enter via rebuild to 100%

    # ---- final marks ----
    # mark to the effective end bar — the weekly-GD bar if terminated, else
    # the last bar of available data.
    last = m.iloc[last_idx]
    open_sh = cur_shares()
    open_cost = sum(s * c for s, c in lots)
    unreal = open_sh * float(last.Close) - open_cost
    total = realized + unreal
    peak_gain_pct = (peak / b1_entry_px - 1.0) * 100.0
    exit_vs_entry_pct = (float(last.Close) / b1_entry_px - 1.0) * 100.0

    mode_labels = {
        "terminate": "TERMINATE (weekly GD -> end, no re-entry)",
        "revert":    "REVERT (weekly GD -> daily, compounded sub-entry)",
        "legacy":    "LEGACY (Phase 2 stays latched, re-enter on weekly GREEN)",
    }
    sr7_label = "  +SR7" if sr7 else ""
    sr6_label = "  +SR6" if sr6 else ""
    atr_label = f"  +ATR{atr_trim:g}" if atr_trim is not None else ""
    wv_label = f"  weekly={weekly_variant}({'/'.join(str(x) for x in weekly_emas)})" if weekly_variant != "default" else ""
    p2d_label = "  +P2DAILY" if phase2_daily else ""
    ldf = pd.DataFrame(log)
    if not quiet:
        print("=" * 78)
        print(f"MO RS 2.0 SHADOW BACKTEST — {ticker}  [{mode_labels[mode]}{sr7_label}{sr6_label}{atr_label}{wv_label}{p2d_label}]")
        print("=" * 78)
        print(f"Start req: {start_ts.date()}   Entry: {entry.Date.date()} @ {b1_entry_px:.4f}"
              f"   End: {last.Date.date()} @ {float(last.Close):.4f}")
        print(f"SPY coverage through {spy_last.date()}")
        print(f"Portfolio NAV ${nav:,.0f}   |   Entry {POSITION_PCT*100:.0f}% = ${unit:,.0f}  "
              f"(cascade targets: GREEN {cascade[0]:.2f} / QUICK {cascade[1]:.2f} / "
              f"QUICKSAND {cascade[2]:.2f} / GD {cascade[3]:.2f})")
        print("-" * 78)
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(ldf.to_string(index=False))
        print("-" * 78)
        print(f"Signals fired:        {len(ldf)-1}")
        print(f"Open shares at end:   {open_sh:,.0f}  @ ${float(last.Close):.2f} = ${open_sh*float(last.Close):,.0f}  (cascade tier {current_cascade_frac*100:.0f}%)")
        print(f"Realized P&L:         ${realized:,.0f}")
        print(f"Unrealized P&L:       ${unreal:,.0f}")
        print(f"TOTAL P&L:            ${total:,.0f}   ({total/unit*100:+.1f}%)")
        print(f"Price journey:        ${b1_entry_px:.4f} entry -> ${peak:.4f} peak on {peak_date.date()} "
              f"({peak_gain_pct:+.1f}%) -> ${float(last.Close):.4f} exit ({exit_vs_entry_pct:+.1f}%)")
        print("=" * 78)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        suffix = (f"_{mode}"
                  + ("_sr7" if sr7 else "")
                  + ("_sr6" if sr6 else "")
                  + (f"_atr{atr_trim:g}" if atr_trim is not None else "")
                  + (f"_{weekly_variant}" if weekly_variant != "default" else "")
                  + ("_p2d" if phase2_daily else ""))
        fname = f"{ticker}_{start_ts.date()}_{last.Date.date()}{suffix}_signals.csv"
        out_path = os.path.join(out_dir, fname)
        ldf.to_csv(out_path, index=False)
        if not quiet:
            print(f"[wrote] {out_path}  ({len(ldf)} rows)")

    return {
        "ticker": ticker,
        "entry_date": entry.Date.date(),
        "entry_px": b1_entry_px,
        "exit_date": last.Date.date(),
        "exit_px": float(last.Close),
        "peak_px": peak,
        "peak_date": peak_date.date(),
        "peak_gain_pct": peak_gain_pct,
        "exit_vs_entry_pct": exit_vs_entry_pct,
        "realized": realized,
        "unrealized": unreal,
        "total": total,
        "total_pct": total / unit * 100.0,
        "signals": len(ldf) - 1,
        "log": ldf,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD — buy date; entry is the first trading day at/after this")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD — test end (default: last common date)")
    ap.add_argument("--nav", type=float, default=DEFAULT_NAV, help="portfolio NAV; entry deploys 20%% of it")
    ap.add_argument("--spy", default="data/spy_daily.csv")
    ap.add_argument("--data", default="data", help="dir holding {TICKER}_price_data.csv")
    ap.add_argument("--out", default="results", help="dir to write the signals CSV (set empty to skip)")
    ap.add_argument("--mode", default="terminate", choices=["terminate", "revert", "legacy"],
                    help="weekly-GD behavior: terminate (default, end campaign), "
                         "revert (PHASE1 + compounded sub-entry), "
                         "legacy (Phase 2 latched forever, re-enter on weekly GREEN)")
    ap.add_argument("--cascade", default="1.0,0.75,0.5,0.0",
                    help="comma-separated targets for GREEN,QUICK,QUICKSAND,GD as fractions of unit "
                         "(default 1.0,0.75,0.5,0.0)")
    ap.add_argument("--refresh", action="store_true",
                    help="force re-download of ticker + SPY CSVs from yfinance")
    ap.add_argument("--sr7", dest="sr7", action="store_true", default=True,
                    help="enable SR7 'Holding Winners' rule (default ON): trim to 20%% NLV "
                         "core on daily 21 EMA violation (intraday low < arming-bar-low * 0.99)")
    ap.add_argument("--no-sr7", dest="sr7", action="store_false",
                    help="disable SR7 (use for comparison vs MO RS cascade alone)")
    ap.add_argument("--sr7-bars", type=int, default=2,
                    help="number of consecutive closes below 21 EMA needed to arm SR7 (default 2)")
    ap.add_argument("--atr-trim", type=float, default=None,
                    help="trim to core (20%% NLV) when Close > 50 SMA + N * ATR(14); "
                         "Phase 2 only. e.g. --atr-trim 6")
    ap.add_argument("--weekly-variant", default="default",
                    choices=list(WEEKLY_VARIANTS),
                    help="weekly cascade variant: default (8/13/21), v1 (5/8/13/21), "
                         "v2 (3/5/8/13/21), v3 (3/8/13/21), "
                         "va (aggressive 1/.5/.3/0), vaa (1/.5/0/0), vk (1/0/0/0)")
    ap.add_argument("--phase2-daily", action="store_true",
                    help="evaluate the daily MO RS cascade in Phase 2 alongside weekly "
                         "(scales in/out faster on downside reactions)")
    ap.add_argument("--sr6", action="store_true",
                    help="enable SR6 '8e Momentum Trim': trim to 20%% NLV core on "
                         "8 EMA violation (after N consec closes above 8 EMA activate)")
    ap.add_argument("--sr6-activate-bars", type=int, default=10,
                    help="consecutive closes above 8 EMA needed to activate SR6 (default 10)")
    a = ap.parse_args()
    tkr_path = f"{a.data}/{a.ticker}_price_data.csv"
    cascade = tuple(float(x) for x in a.cascade.split(","))
    if len(cascade) != 4:
        ap.error("--cascade must be 4 floats: GREEN,QUICK,QUICKSAND,GD")
    run(a.spy, tkr_path, a.ticker, a.start, a.end, a.nav,
        out_dir=(a.out or None), mode=a.mode, refresh=a.refresh, cascade=cascade,
        sr7=a.sr7, sr7_arm_bars=a.sr7_bars, atr_trim=a.atr_trim,
        weekly_variant=a.weekly_variant, phase2_daily=a.phase2_daily,
        sr6=a.sr6, sr6_activate_bars=a.sr6_activate_bars)
