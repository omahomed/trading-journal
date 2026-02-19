import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math
from datetime import datetime

# --- CONFIGURATION ---
FILE_NAME = 'Trading_Journal_Clean.csv'
LOG_FILE = 'Trade_Plan_Log.txt'
TICKET_IMG = 'Trade_Ticket.png'

print("\n" + "="*50)
print("      CAN SLIM POSITION SIZING CALCULATOR")
print("="*50)

# --- 1. GET CAPITAL ---
if not os.path.exists(FILE_NAME):
    print(f"Error: {FILE_NAME} not found.")
    exit()

try:
    df = pd.read_csv(FILE_NAME)
    df.columns = [c.strip() for c in df.columns]
    
    def clean_num(x):
        if pd.isna(x) or x == '': return 0.0
        if isinstance(x, str):
            x = x.replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            if '(' in x: x = x.replace('(', '-').replace(')', '')
            if x in ['-', '']: return 0.0
            try:
                return float(x)
            except:
                return 0.0
        return float(x)
    
    if 'End NLV' in df.columns:
        df['End NLV'] = df['End NLV'].apply(clean_num)
        df_valid = df[df['End NLV'] > 0].copy()
        if not df_valid.empty:
            current_equity = df_valid['End NLV'].iloc[-1]
            date_str = df_valid['Day'].iloc[-1]
        else:
            current_equity = float(input("Enter Manual Equity: "))
            date_str = "Manual"
    else:
        current_equity = float(input("Enter Manual Equity: "))
        date_str = "Manual"

except Exception as e:
    print(f"Error reading file: {e}")
    current_equity = float(input("Enter Manual Equity: "))
    date_str = "Manual"

print(f"Acct Value ({date_str}): ${current_equity:,.2f}")

# --- 2. INPUTS ---
print("-" * 50)
ticker = input("Ticker Symbol: ").upper()
try:
    entry_price = float(input("Entry Price:   $"))
except ValueError:
    print("Invalid number.")
    exit()

print("\nSTOP LOSS METHOD:")
print("1. TECHNICAL LEVEL (Specific Price)")
print("2. ATR % TRAILING")
stop_method = input("Select (1-2): ")
stop_note = ""

if stop_method == '1':
    try:
        stop_price = float(input("Stop Loss Price: $"))
        stop_note = "Technical Level"
    except:
        print("Invalid price."); exit()
elif stop_method == '2':
    atr_input = input("Enter ATR % Value (e.g. 3.28): ").replace('%', '')
    try:
        atr_pct = float(atr_input)
        mult = float(input("Enter Multiplier (e.g. 1.5, 2.0): "))
    except:
        print("Invalid number."); exit()
    atr_dollars = entry_price * (atr_pct / 100)
    stop_distance = atr_dollars * mult
    stop_price = entry_price - stop_distance
    # This is the note that will appear on the ticket
    stop_note = f"ATR ({mult}x of {atr_pct}%)"
    print(f"   -> Calc Stop: ${stop_price:.2f}")
else:
    print("Invalid selection."); exit()

risk_per_share = entry_price - stop_price
if risk_per_share <= 0:
    print("\nERROR: Stop must be lower than Entry."); exit()
stop_pct = (risk_per_share / entry_price) * 100

# --- 3. CUSTOM SIZING ---
print("-" * 50)
print("STEP 1: RISK APPETITE")
try:
    risk_limit_pct = float(input("Risk % to Capital (0.5 - 1.25): "))
except:
    risk_limit_pct = 0.75

print("\nSTEP 2: POSITION SCALING")
print("1. SHOTGUN (2.5%) | 2. HALF (5%) | 3. FULL (10%)")
print("4. FULL+1 (15%)   | 5. FULL+2 (20%) | 6. MAX (25%)")
size_choice = input("Select (1-6): ")

size_map = {'1': (2.5, "SHOTGUN"), '2': (5.0, "HALF POS"), '3': (10.0, "FULL POS"),
            '4': (15.0, "FULL+1"), '5': (20.0, "FULL+2"), '6': (25.0, "MAX POS")}
if size_choice in size_map:
    max_pos_pct, label = size_map[size_choice]
else:
    max_pos_pct, label = 5.0, "HALF (Default)"

# --- 4. MATH ---
if math.isnan(current_equity) or current_equity <= 0: print("Error: Invalid Equity."); exit()

risk_budget_dollars = current_equity * (risk_limit_pct / 100)
shares_by_risk = math.floor(risk_budget_dollars / risk_per_share)

max_position_dollars = current_equity * (max_pos_pct / 100)
shares_by_cap = math.floor(max_position_dollars / entry_price)

recommended_shares = int(min(shares_by_risk, shares_by_cap))
actual_cost = recommended_shares * entry_price
actual_weight = (actual_cost / current_equity) * 100
actual_risk_dollars = recommended_shares * risk_per_share
actual_risk_pct = (actual_risk_dollars / current_equity) * 100

# Warnings
warnings = []
if stop_pct > 8.0: warnings.append(f"[!] STOP {stop_pct:.2f}% > 8% (Rule #1)")
elif stop_pct > 7.0: warnings.append(f"[!] STOP {stop_pct:.2f}% is wide.")
if recommended_shares == shares_by_cap and actual_risk_pct < risk_limit_pct:
    warnings.append(f"[*] Capped by SIZE ({max_pos_pct}%).")
elif recommended_shares == shares_by_risk and actual_weight < max_pos_pct:
    warnings.append(f"[*] Capped by RISK ({risk_limit_pct}%).")

# --- 5. GENERATE IMAGE TICKET ---
def create_ticket_image():
    # Setup Canvas
    fig, ax = plt.subplots(figsize=(6, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Colors
    bg_color = '#fcfcfc'
    header_color = '#003366' # Navy
    accent_color = '#e0e0e0'
    
    # Border
    rect = patches.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=2, edgecolor='#333', facecolor=bg_color)
    ax.add_patch(rect)
    
    # Header
    header = patches.Rectangle((0.02, 0.85), 0.96, 0.13, linewidth=0, facecolor=header_color)
    ax.add_patch(header)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.text(0.5, 0.93, "TRADE TICKET", ha='center', va='center', fontsize=22, color='white', fontweight='bold', fontname='sans-serif')
    ax.text(0.5, 0.88, timestamp, ha='center', va='center', fontsize=10, color='white', fontname='sans-serif')

    # Main Info
    ax.text(0.5, 0.78, f"{ticker}", ha='center', fontsize=24, fontweight='bold', color='black')
    ax.text(0.5, 0.74, f"STRATEGY: {label}", ha='center', fontsize=12, color='#555', fontweight='bold')
    
    # Divider
    ax.plot([0.1, 0.9], [0.71, 0.71], color=accent_color, lw=2)
    
    # Data Grid
    # Left Column
    x_L = 0.15
    y = 0.65
    ax.text(x_L, y, "ENTRY PRICE", fontsize=9, color='#777'); 
    ax.text(x_L, y-0.04, f"${entry_price:,.2f}", fontsize=14, fontweight='bold')
    
    y -= 0.12
    ax.text(x_L, y, "STOP LOSS", fontsize=9, color='#777'); 
    ax.text(x_L, y-0.04, f"${stop_price:,.2f}", fontsize=14, fontweight='bold', color='#cc0000')
    ax.text(x_L, y-0.08, f"(-{stop_pct:.2f}%)", fontsize=10, color='#cc0000')
    
    # ADDED: Stop Note (ATR Info)
    ax.text(x_L, y-0.12, stop_note, fontsize=8, color='#444', style='italic')
    
    # Right Column
    x_R = 0.60
    y = 0.65
    ax.text(x_R, y, "EQUITY BASIS", fontsize=9, color='#777'); 
    ax.text(x_R, y-0.04, f"${current_equity:,.0f}", fontsize=14, fontweight='bold')
    
    y -= 0.12
    ax.text(x_R, y, "RISK BUDGET", fontsize=9, color='#777'); 
    ax.text(x_R, y-0.04, f"{risk_limit_pct}% (${risk_budget_dollars:,.0f})", fontsize=14, fontweight='bold')
    
    # Action Box
    action_y = 0.28
    box = patches.FancyBboxPatch((0.1, action_y), 0.8, 0.18, boxstyle="round,pad=0.1", 
                                 linewidth=2, edgecolor='#006400', facecolor='#e8f5e9')
    ax.add_patch(box)
    
    ax.text(0.5, action_y+0.13, "EXECUTION PLAN", ha='center', fontsize=10, color='#006400', fontweight='bold')
    ax.text(0.5, action_y+0.07, f"BUY {recommended_shares} SHARES", ha='center', fontsize=20, fontweight='bold', color='black')
    ax.text(0.5, action_y+0.03, f"Cost: ${actual_cost:,.2f} ({actual_weight:.1f}%)", ha='center', fontsize=10, color='#333')
    
    # Warnings Area
    wy = 0.20
    if warnings:
        for w in warnings:
            ax.text(0.5, wy, w, ha='center', fontsize=9, color='red', fontweight='bold')
            wy -= 0.03
    else:
        ax.text(0.5, wy, "clean trade setup", ha='center', fontsize=8, color='green', style='italic')

    # Save
    plt.savefig(TICKET_IMG, dpi=150, bbox_inches='tight')
    print(f"\n>> Generated Ticket Image: {TICKET_IMG}")
    plt.close()

# --- 6. OUTPUT TEXT & IMAGE ---
create_ticket_image()

# Also print text summary for terminal
print(f"\n{'='*30}")
print(f"BUY {recommended_shares} {ticker} @ ${entry_price}")
print(f"Risk: ${actual_risk_dollars:.0f} ({actual_risk_pct:.2f}%)")
print(f"Cost: ${actual_cost:,.0f} ({actual_weight:.1f}%)")
if warnings:
    print("WARNINGS:")
    for w in warnings: print(f" - {w}")
print(f"{'='*30}")

# Append to Log
log_entry = f"{datetime.now()} | {ticker} ({label}) | Buy {recommended_shares} @ {entry_price} | Stop {stop_price} | Risk {actual_risk_pct:.2f}% | {stop_note}"
with open(LOG_FILE, 'a') as f:
    f.write(log_entry + "\n")
print(f">> Logged to {LOG_FILE}")