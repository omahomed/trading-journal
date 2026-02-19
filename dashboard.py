import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# --- CONFIGURATION ---
FILE_NAME = 'Trading_Journal_Clean.csv'

print("--- GENERATING FINAL DASHBOARD ---")

if not os.path.exists(FILE_NAME):
    print(f"Error: {FILE_NAME} not found.")
    exit()

# 1. LOAD & CLEAN DATA
try:
    df = pd.read_csv(FILE_NAME)
    df.columns = [c.strip() for c in df.columns]
    if 'Nsadaq' in df.columns:
        df.rename(columns={'Nsadaq': 'Nasdaq'}, inplace=True)
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# Convert Dates
df['Day'] = pd.to_datetime(df['Day'], errors='coerce')
df = df.dropna(subset=['Day']).sort_values('Day')

# Cleaning Helpers
def clean_num(x):
    if isinstance(x, str):
        x = x.replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
        if '(' in x: x = x.replace('(', '-').replace(')', '')
        if x in ['-', '']: return 0.0
        try:
            return float(x)
        except:
            return 0.0
    return x

cols_to_fix = ['Beg NLV', 'End NLV', 'Cash -/+', 'Daily $ Change', 'SPY', 'Nasdaq', '% Invested']
for col in cols_to_fix:
    if col in df.columns:
        df[col] = df[col].apply(clean_num)
    else:
        df[col] = 0.0

# 2. RECALCULATE PERFORMANCE
beg = df['Beg NLV']
end = df['End NLV']
cash = df['Cash -/+'].fillna(0.0)
denom = beg + cash

valid_rows = denom != 0
df.loc[valid_rows, 'Calculated_Daily_Pct'] = (end - denom) / denom
df.loc[~valid_rows, 'Calculated_Daily_Pct'] = 0.0

# 3. BUILD INDICATORS
# Equity Curves
df['Equity_Curve'] = (1 + df['Calculated_Daily_Pct']).cumprod()
df['LTD_Pct'] = (df['Equity_Curve'] - 1) * 100

# YTD
current_year = df['Day'].iloc[-1].year
df_ytd = df[df['Day'].dt.year == current_year].copy()
if not df_ytd.empty:
    df_ytd['YTD_Curve'] = (1 + df_ytd['Calculated_Daily_Pct']).cumprod()
    ytd_val = (df_ytd['YTD_Curve'].iloc[-1] - 1) * 100
else:
    ytd_val = 0.0
ltd_val = df['LTD_Pct'].iloc[-1]

# Benchmarks
initial_spy = df['SPY'].iloc[0]
df['SPY_Pct'] = (df['SPY'] / initial_spy - 1) * 100 if initial_spy != 0 else 0.0

initial_ndx = df['Nasdaq'].iloc[0]
df['NDX_Pct'] = (df['Nasdaq'] / initial_ndx - 1) * 100 if initial_ndx != 0 else 0.0

# Portfolio Moving Averages
df['EC_8EMA'] = df['LTD_Pct'].ewm(span=8, adjust=False).mean()
df['EC_21EMA'] = df['LTD_Pct'].ewm(span=21, adjust=False).mean()
df['EC_50SMA'] = df['LTD_Pct'].rolling(window=50).mean()

# Market Analysis (Nasdaq vs 21EMA)
df['NDX_21EMA'] = df['Nasdaq'].ewm(span=21, adjust=False).mean()

# 4. PLOTTING
plt.style.use('bmh') 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={'height_ratios': [3, 1]})

# --- TOP CHART ---

# A) MARKET STATUS STRIP (Top Edge)
# 0.97 to 1.0 is top 3% of chart area
ax1.fill_between(df['Day'], 0.97, 1.0, transform=ax1.transAxes, 
                 where=(df['Nasdaq'] >= df['NDX_21EMA']), color='green', alpha=0.4, zorder=0)
ax1.fill_between(df['Day'], 0.97, 1.0, transform=ax1.transAxes, 
                 where=(df['Nasdaq'] < df['NDX_21EMA']), color='red', alpha=0.4, zorder=0)
ax1.text(0.5, 0.985, "MARKET TREND (NDX vs 21e)", transform=ax1.transAxes, 
         ha='center', va='center', fontsize=8, color='black', fontweight='bold')

# B) RIGHT AXIS: % Invested (CRUSHED)
ax1_right = ax1.twinx()
ax1_right.fill_between(df['Day'], df['% Invested'], color='orange', alpha=0.2, label='% Invested')
# KEY CHANGE: Limit set to 600. This pushes "100%" down to the bottom ~15% of chart.
ax1_right.set_ylim(0, 600) 
ax1_right.set_yticks([]) 

# C) LEFT AXIS: Performance
# Benchmarks
ax1.plot(df['Day'], df['SPY_Pct'], label='S&P 500', color='gray', linewidth=1.5, alpha=0.6)
ax1.plot(df['Day'], df['NDX_Pct'], label='Nasdaq', color='black', linewidth=1.5, alpha=0.7)

# Portfolio (Dark Blue)
ax1.plot(df['Day'], df['LTD_Pct'], label='Portfolio', color='darkblue', linewidth=2.5)

# Moving Averages
ax1.plot(df['Day'], df['EC_8EMA'], label='8 EMA', color='purple', linewidth=1.2)
ax1.plot(df['Day'], df['EC_21EMA'], label='21 EMA', color='green', linewidth=1.2)
ax1.plot(df['Day'], df['EC_50SMA'], label='50 SMA', color='red', linewidth=1.2)

# Formatting
ax1.set_title(f"EQUITY CURVE & MARKET CONTEXT\nLTD: {ltd_val:+.2f}% | YTD: {ytd_val:+.2f}%", fontsize=12, fontweight='bold')
ax1.set_ylabel("Return (%)")
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Legend
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='upper left', framealpha=0.9, fontsize=9)

# Stats Box
invested_curr = df['% Invested'].iloc[-1]
stats = (f"NAV: ${df['End NLV'].iloc[-1]:,.2f}\n"
         f"Exp: {invested_curr:.0f}%")
ax1.text(0.02, 0.50, stats, transform=ax1.transAxes, 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round'))

# --- BOTTOM CHART: Daily P&L ---
colors = ['green' if x >= 0 else 'red' for x in df['Daily $ Change']]
ax2.bar(df['Day'], df['Daily $ Change'], color=colors, alpha=0.8, width=1.0)

# ADDED: PORTFOLIO TREND STRIP (Bottom of P&L Chart)
# Shows Red/Green bar at the very bottom depending on if Portfolio > 21EMA
y_min_pl, y_max_pl = df['Daily $ Change'].min(), df['Daily $ Change'].max()
ax2.fill_between(df['Day'], y_min_pl, y_min_pl + (y_max_pl-y_min_pl)*0.05, 
                 where=(df['LTD_Pct'] >= df['EC_21EMA']), color='green', alpha=0.5, label='Port > 21e')
ax2.fill_between(df['Day'], y_min_pl, y_min_pl + (y_max_pl-y_min_pl)*0.05, 
                 where=(df['LTD_Pct'] < df['EC_21EMA']), color='red', alpha=0.5, label='Port < 21e')

ax2.set_title("Daily Profit/Loss ($) | Bottom Strip: Portfolio Trend", fontsize=10)
ax2.set_ylabel("$ Amount")
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# 5. SAVE & SHOW
plt.tight_layout()
output_file = "dashboard_final.png"
plt.savefig(output_file, dpi=120)
print(f"Chart saved to {output_file}")
plt.show()