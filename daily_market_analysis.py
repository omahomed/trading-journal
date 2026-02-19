"""
daily_market_analysis.py
Combined script to fetch latest data and run market analysis
Starting from April 2025 FTD period
"""
import subprocess
import sys
import os
import pandas as pd
from datetime import datetime

def needs_data_update():
    """Check if we need to run the scraper"""
    data_file = "output/IXIC_price_data.csv"
    
    # If file doesn't exist, definitely need to run scraper
    if not os.path.exists(data_file):
        print("📊 No data file found. Need to download data from April 2025.")
        return True
    
    # Check if today's data is already in the file
    try:
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        last_date = df['Date'].max().date()
        today = datetime.now().date()
        
        if last_date < today:
            print(f"📅 Last data is from {last_date}. Need to update.")
            return True
        else:
            print(f"✅ Already have today's data ({today}).")
            return False
            
    except Exception as e:
        print(f"⚠️ Error checking data file: {e}")
        return True

def run_analysis(force_update=False):
    """Run the complete daily analysis workflow"""
    
    print("="*70)
    print("📊 DAILY MARKET ANALYSIS WORKFLOW (April 2025 FTD Rally)")
    print("="*70)
    
    # Step 1: Check if we need to update data
    if force_update or needs_data_update():
        print("\n📥 Step 1: Fetching market data from April 2025...")
        print("-"*50)
        
        try:
            # Run the stock scraper
            result = subprocess.run([sys.executable, "stock_scraper.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Data fetch successful!")
                # Optional: show scraper output
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if any(keyword in line for keyword in ['Added today', 'Date range', 'Saved']):
                            print(f"   {line.strip()}")
            else:
                print("❌ Error running scraper:")
                print(result.stderr)
                return
                
        except Exception as e:
            print(f"❌ Failed to run scraper: {e}")
            return
    else:
        print("\n✓ Step 1: Data is already up to date!")
    
    # Step 2: Run the market analyzer
    print("\n📈 Step 2: Running Market School Rules analysis...")
    print("-"*50)
    
    # Import and run the analyzer
    try:
        # Make sure we're importing from the correct path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from hybrid_market_analyzer import HybridMarketAnalyzer
        
        analyzer = HybridMarketAnalyzer()
        analyzer.load_data("IXIC_price_data.csv")
        
        # Analyze from April 1, 2025 (a few days before the April 7 rally low)
        analyzer.analyze_market(start_date="2025-04-01")
        
        # Check for command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "quick":
                analyzer.quick_status()
            elif sys.argv[1] == "report":
                analyzer.generate_report("hybrid_signals_report.tsv")
                print("\n📄 Report saved to hybrid_signals_report.tsv")
            elif sys.argv[1] == "excel":
                analyzer.export_excel_format()
            elif sys.argv[1] == "force":
                # This was a force update, just run normal analysis
                analyzer.print_analysis()
            else:
                analyzer.print_analysis()
        else:
            analyzer.print_analysis()
            
    except Exception as e:
        print(f"❌ Error running analyzer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if user wants to force update
    force_update = len(sys.argv) > 1 and sys.argv[1] == "force"
    
    run_analysis(force_update=force_update)