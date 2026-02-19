import pandas as pd
import numpy as np
import os
from datetime import datetime

print("="*60)
print("📊 PORTFOLIO DATA DIAGNOSTIC TOOL")
print("="*60)

# Configuration (match your main script)
DATA_START_DATE = "2023-09-01"
PORTFOLIO_START_DATE = "2024-09-04"
END_DATE = "2025-09-05"

# Load tickers
with open("select_ticker.txt", "r") as f:
    tickers = [line.strip().upper() for line in f if line.strip()]

print(f"\n📁 Found {len(tickers)} tickers in select_ticker.txt")

# Diagnostic results
issues = []
suspicious_tickers = []
data_summary = []

print("\n🔍 Analyzing each ticker's data file...")
print("-" * 60)

for ticker in tickers:
    input_file = f"output/{ticker}_price_data.csv"
    
    if not os.path.exists(input_file):
        issues.append(f"{ticker}: File not found")
        continue
    
    try:
        # Load data
        df = pd.read_csv(input_file)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        
        # Filter to date range
        df = df[(df.index >= DATA_START_DATE) & (df.index <= END_DATE)]
        
        if df.empty:
            issues.append(f"{ticker}: No data in date range")
            continue
        
        # Convert to numeric
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Calculate statistics
        max_price = df[["Open", "High", "Low", "Close"]].max().max()
        min_price = df[["Open", "High", "Low", "Close"]].min().min()
        avg_price = df["Close"].mean()
        
        # Check for data issues
        ticker_issues = []
        
        # 1. Check for negative or zero prices
        if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
            ticker_issues.append("Has non-positive prices")
            suspicious_tickers.append(ticker)
        
        # 2. Check for NaN values
        nan_count = df[["Open", "High", "Low", "Close"]].isna().sum().sum()
        if nan_count > 0:
            ticker_issues.append(f"Has {nan_count} NaN values")
        
        # 3. Check for extreme prices
        if max_price > 10000:
            ticker_issues.append(f"Very high price: ${max_price:.2f}")
            suspicious_tickers.append(ticker)
        
        if min_price < 0.01 and min_price > 0:
            ticker_issues.append(f"Very low price: ${min_price:.4f}")
            suspicious_tickers.append(ticker)
        
        # 4. Check for extreme daily moves
        daily_returns = df["Close"].pct_change()
        extreme_moves = daily_returns[daily_returns.abs() > 0.5]
        if len(extreme_moves) > 0:
            ticker_issues.append(f"{len(extreme_moves)} days with >50% moves")
            if len(extreme_moves) > 5:
                suspicious_tickers.append(ticker)
        
        # 5. Check for price scale issues (prices might be in cents)
        if avg_price < 1:
            ticker_issues.append(f"Avg price < $1 (${avg_price:.4f}) - might be in cents?")
            suspicious_tickers.append(ticker)
        
        # 6. Check for sudden price jumps that might indicate data errors
        if len(df) > 1:
            # Look for 100x or 0.01x changes (common with unit errors)
            ratios = df["Close"] / df["Close"].shift(1)
            if (ratios > 50).any() or (ratios < 0.02).any():
                ticker_issues.append("Has 50x+ or 0.02x price jumps (possible unit error)")
                suspicious_tickers.append(ticker)
        
        # Store summary
        data_summary.append({
            'ticker': ticker,
            'min_price': min_price,
            'max_price': max_price,
            'avg_price': avg_price,
            'data_points': len(df),
            'issues': len(ticker_issues)
        })
        
        # Report issues for this ticker
        if ticker_issues:
            print(f"\n⚠️ {ticker}:")
            for issue in ticker_issues:
                print(f"   - {issue}")
                issues.append(f"{ticker}: {issue}")
        
    except Exception as e:
        issues.append(f"{ticker}: Error reading file - {str(e)}")
        print(f"\n❌ {ticker}: Error reading file - {str(e)}")

# Remove duplicates from suspicious tickers
suspicious_tickers = list(set(suspicious_tickers))

# Summary Report
print("\n" + "="*60)
print("📊 DIAGNOSTIC SUMMARY")
print("="*60)

print(f"\n✅ Successfully analyzed: {len(data_summary)} tickers")
print(f"⚠️ Total issues found: {len(issues)}")
print(f"🚨 Highly suspicious tickers: {len(suspicious_tickers)}")

if suspicious_tickers:
    print("\n🚨 TICKERS REQUIRING IMMEDIATE ATTENTION:")
    print("These tickers have data that could cause the simulation to fail:")
    for ticker in sorted(suspicious_tickers)[:20]:  # Show top 20
        # Find the specific issues for this ticker
        ticker_issues = [issue for issue in issues if issue.startswith(f"{ticker}:")]
        print(f"\n{ticker}:")
        for issue in ticker_issues[:3]:  # Show first 3 issues
            print(f"  {issue}")

# Create detailed report
print("\n📝 Creating detailed diagnostic report...")

# Save summary to CSV
summary_df = pd.DataFrame(data_summary)
if not summary_df.empty:
    summary_df = summary_df.sort_values('issues', ascending=False)
    summary_df.to_csv("output/data_diagnostic_summary.csv", index=False)
    print(f"💾 Summary saved to: output/data_diagnostic_summary.csv")
    
    # Show top problematic tickers
    print("\n🔝 Top 10 tickers with most issues:")
    print(summary_df.head(10)[['ticker', 'min_price', 'max_price', 'avg_price', 'issues']].to_string())

# Save detailed issues list
with open("output/data_diagnostic_issues.txt", "w") as f:
    f.write("PORTFOLIO DATA DIAGNOSTIC REPORT\n")
    f.write(f"Generated: {datetime.now()}\n")
    f.write("="*60 + "\n\n")
    
    if suspicious_tickers:
        f.write("HIGHLY SUSPICIOUS TICKERS:\n")
        for ticker in sorted(suspicious_tickers):
            f.write(f"- {ticker}\n")
        f.write("\n")
    
    f.write("ALL ISSUES FOUND:\n")
    for issue in sorted(issues):
        f.write(f"- {issue}\n")

print(f"💾 Detailed issues saved to: output/data_diagnostic_issues.txt")

# Specific check for DOCS since it was problematic
if "DOCS" in tickers:
    print("\n🔍 Special analysis for DOCS (previously problematic):")
    try:
        docs_df = pd.read_csv("output/DOCS_price_data.csv")
        docs_df["Date"] = pd.to_datetime(docs_df["Date"])
        docs_df = docs_df[(docs_df["Date"] >= PORTFOLIO_START_DATE)]
        
        if not docs_df.empty:
            docs_df["Close"] = pd.to_numeric(docs_df["Close"], errors="coerce")
            print(f"  First 5 closing prices: {docs_df['Close'].head().tolist()}")
            print(f"  Last 5 closing prices: {docs_df['Close'].tail().tolist()}")
            print(f"  Price range: ${docs_df['Close'].min():.2f} - ${docs_df['Close'].max():.2f}")
            
            # Check if prices might be in cents
            if docs_df['Close'].mean() < 1:
                print(f"  ⚠️ WARNING: Average price is ${docs_df['Close'].mean():.4f}")
                print(f"  💡 SUGGESTION: Prices might be in cents. Multiply by 100?")
    except:
        print("  Could not analyze DOCS data")

print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)
print("""
1. Check the suspicious tickers listed above
2. Look for these common issues:
   - Prices in cents instead of dollars (multiply by 100)
   - Extra zeros or missing decimal points
   - Date parsing issues causing wrong data alignment
   - Corporate actions (splits) not properly adjusted

3. To fix data issues:
   - Re-download data from your source
   - Check if the data provider's format changed
   - Verify stock split adjustments are correct
   
4. Test with a small subset first:
   - Create a test file with only 5-10 clean tickers
   - Run the simulation on this subset
   - If it works, gradually add more tickers
""")

print("\n✅ Diagnostic complete! Check the output files for details.")