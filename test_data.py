import yfinance as yf

print("Testing Single Ticker (SPY)...")
spy = yf.Ticker("SPY").history(period="5d")
print(spy)

print("\nTesting Batch Download (AAPL, NVDA)...")
batch = yf.download(["AAPL", "NVDA"], period="5d")
print(batch)