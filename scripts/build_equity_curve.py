#!/usr/bin/env python3
"""Build the 21e Strategy equity curve PNG and (optionally) upload to R2.

Reads daily journal data from a JSON file, fetches QQQ/SPY closes via yfinance,
and plots cumulative returns vs benchmarks. Uploads to R2 at
21ema/equity_curve_YYYYMMDD.png so the Notion parent page can embed it.

Usage:
    python scripts/build_equity_curve.py --journal /tmp/21e_journal.json
    python scripts/build_equity_curve.py --journal /tmp/21e_journal.json --upload

Journal JSON shape (one of):
    Source of truth = Notion Equity Tracker. Each entry is one trading day.

    Pass `portfolio_ltd_pct` (preferred — matches recorded TWR, handles cash
    flow on reset day cleanly):
        {
            "start_date": "2026-04-03",
            "entries": [
                {"date": "2026-04-03", "portfolio_ltd_pct": 0.0,
                 "market_window": "Closed"},
                {"date": "2026-04-30", "portfolio_ltd_pct": 0.2402,
                 "market_window": "Open"},
                ...
            ]
        }

    OR pass beg/end/cash and let the script compute TWR (use only if recorded
    LTD% is unavailable):
        {"date": "...", "beg_nlv": ..., "end_nlv": ..., "cash_flow": ...,
         "market_window": "Open"}

Benchmarks are simple price-based from the closing price on start_date.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from matplotlib.patches import Patch

R2_BUCKET_DEFAULT = "trading-journal-images"
R2_KEY_PREFIX = "21ema"
R2_PUBLIC_URL_DEFAULT = "https://pub-a55e7ca9f1ed4305a3de0d614ea0ea79.r2.dev"

STATUS_COLOR = {
    "Powertrend": "#0a3a0a",
    "Open": "#0a2510",
    "Neutral": "#332e00",
    "Closed": "#330000",
}
STRATEGY_COLOR = "#00d4ff"
QQQ_COLOR = "#ff9933"
SPY_COLOR = "#33cc66"


def build_strategy_series(entries):
    """Build the strategy cumulative-return series.

    Prefers `portfolio_ltd_pct` per entry (matches the Notion Equity Tracker's
    recorded TWR/LTD field, which is the canonical record). Falls back to
    computing TWR from beg/end/cash if not provided.
    """
    twr = 1.0
    rows = []
    for e in entries:
        if "portfolio_ltd_pct" in e and e["portfolio_ltd_pct"] is not None:
            cum_pct = float(e["portfolio_ltd_pct"]) * 100
        else:
            beg = e["beg_nlv"] + e.get("cash_flow", 0.0)
            ret = (e["end_nlv"] - beg) / beg if beg > 0 else 0.0
            twr *= 1.0 + ret
            cum_pct = (twr - 1.0) * 100
        rows.append({
            "date": pd.to_datetime(e["date"]),
            "cum_return_pct": cum_pct,
            "market_window": e.get("market_window")
                or e.get("market_window", "Neutral"),
        })
    return pd.DataFrame(rows)


def fetch_benchmark(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date + pd.Timedelta(days=1),
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    # yfinance returns a multi-index column frame even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    closes = df["Close"].astype(float)
    base = float(closes.iloc[0])
    out = pd.DataFrame({
        "date": closes.index,
        "close": closes.values,
        "cum_return_pct": (closes.values / base - 1.0) * 100,
    }).reset_index(drop=True)
    return out


def plot_equity_curve(strategy_df, qqq_df, spy_df, out_path):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#0a0a0a")

    prev_status = None
    span_start = None
    for _, row in strategy_df.iterrows():
        if row["market_window"] != prev_status:
            if prev_status is not None and span_start is not None:
                ax.axvspan(
                    span_start, row["date"],
                    alpha=0.4, color=STATUS_COLOR.get(prev_status, "#1a1a1a"),
                )
            prev_status = row["market_window"]
            span_start = row["date"]
    if prev_status is not None and span_start is not None:
        ax.axvspan(
            span_start, strategy_df["date"].iloc[-1],
            alpha=0.4, color=STATUS_COLOR.get(prev_status, "#1a1a1a"),
        )

    ax.plot(strategy_df["date"], strategy_df["cum_return_pct"], "o-",
            color=STRATEGY_COLOR, label="21 EMA Strategy",
            linewidth=2, markersize=6)
    ax.plot(qqq_df["date"], qqq_df["cum_return_pct"], "s-",
            color=QQQ_COLOR, label="QQQ", linewidth=1.5, markersize=5)
    ax.plot(spy_df["date"], spy_df["cum_return_pct"], "^-",
            color=SPY_COLOR, label="SPY", linewidth=1.5, markersize=5)

    for df, color in [(strategy_df, STRATEGY_COLOR), (qqq_df, QQQ_COLOR), (spy_df, SPY_COLOR)]:
        last = df.iloc[-1]
        sign = "+" if last["cum_return_pct"] >= 0 else ""
        ax.annotate(
            f"{sign}{last['cum_return_pct']:.2f}%",
            xy=(last["date"], last["cum_return_pct"]),
            xytext=(8, 0), textcoords="offset points",
            color=color,
            fontweight="bold" if color == STRATEGY_COLOR else "normal",
            fontsize=12 if color == STRATEGY_COLOR else 11,
        )

    ax.axhline(0, color="#666", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_title(
        "21 EMA Strategy - Equity Curve vs Benchmarks",
        color="white", fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Cumulative Return (%)", color="white")
    ax.grid(True, color="#333", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    line_handles, line_labels = ax.get_legend_handles_labels()
    statuses_present = list(strategy_df["market_window"].unique())
    status_order = ["Powertrend", "Open", "Neutral", "Closed"]
    for status in status_order:
        if status in statuses_present:
            line_handles.append(Patch(facecolor=STATUS_COLOR[status], alpha=0.4))
            line_labels.append(status)
    ax.legend(
        line_handles, line_labels,
        loc="upper left", framealpha=0.85,
        facecolor="#1a1a1a", edgecolor="#444",
        fontsize=9, ncol=2,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def _load_r2_config():
    cfg = {
        "endpoint_url": os.environ.get("R2_ENDPOINT_URL"),
        "access_key_id": os.environ.get("R2_ACCESS_KEY_ID"),
        "secret_access_key": os.environ.get("R2_SECRET_ACCESS_KEY"),
        "bucket_name": os.environ.get("R2_BUCKET_NAME", R2_BUCKET_DEFAULT),
        "public_url": os.environ.get("R2_PUBLIC_URL", R2_PUBLIC_URL_DEFAULT),
    }
    if all([cfg["endpoint_url"], cfg["access_key_id"], cfg["secret_access_key"]]):
        return cfg

    secrets_path = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # py<3.11
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        r2 = secrets.get("r2", {})
        for key in ("endpoint_url", "access_key_id", "secret_access_key", "bucket_name", "public_url"):
            if not cfg.get(key) and r2.get(key):
                cfg[key] = r2[key]

    return cfg


def upload_to_r2(local_path, object_key):
    import boto3
    from botocore.config import Config

    cfg = _load_r2_config()
    if not all([cfg["endpoint_url"], cfg["access_key_id"], cfg["secret_access_key"]]):
        raise RuntimeError(
            "R2 credentials not found. Set R2_ENDPOINT_URL/R2_ACCESS_KEY_ID/"
            "R2_SECRET_ACCESS_KEY env vars or configure .streamlit/secrets.toml [r2]"
        )

    client = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint_url"],
        aws_access_key_id=cfg["access_key_id"],
        aws_secret_access_key=cfg["secret_access_key"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=cfg["bucket_name"],
            Key=object_key,
            Body=f.read(),
            ContentType="image/png",
        )
    return f"{cfg['public_url']}/{object_key}"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--journal", required=True,
                   help="Path to JSON file with start_date + entries[]")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: scripts/equity_curve_<last_date>.png)")
    p.add_argument("--upload", action="store_true",
                   help="Upload to R2 at 21ema/equity_curve_YYYYMMDD.png")
    args = p.parse_args()

    with open(args.journal) as f:
        data = json.load(f)

    entries = data.get("entries") or []
    if not entries:
        print("No journal entries in input; nothing to plot.", file=sys.stderr)
        sys.exit(1)

    strategy_df = build_strategy_series(entries)
    start_date = pd.to_datetime(data["start_date"])
    end_date = strategy_df["date"].max()

    qqq_df = fetch_benchmark("QQQ", start_date, end_date)
    spy_df = fetch_benchmark("SPY", start_date, end_date)

    last_date_str = end_date.strftime("%Y%m%d")
    out_path = args.out or f"scripts/equity_curve_{last_date_str}.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    plot_equity_curve(strategy_df, qqq_df, spy_df, out_path)
    print(f"Wrote {out_path}")

    if args.upload:
        object_key = f"{R2_KEY_PREFIX}/equity_curve_{last_date_str}.png"
        public_url = upload_to_r2(out_path, object_key)
        print(f"Uploaded → {public_url}")


if __name__ == "__main__":
    main()
