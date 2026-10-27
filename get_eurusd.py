"""
Downloads the last ~60 days of EURUSD 5-minute candles from Yahoo Finance
and saves them as prices.csv (full decimals), ready for fvg_backtest.py.

Setup (once):  pip install yfinance
Run:           python get_eurusd.py
"""
import pandas as pd
import yfinance as yf

df = yf.download("EURUSD=X", interval="5m", period="60d", progress=False)
if df.empty:
    raise SystemExit("No data came back. Check your internet connection and try again.")

if isinstance(df.columns, pd.MultiIndex):       # newer yfinance versions
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()
df = df.rename(columns={df.columns[0]: "time", "Open": "open", "High": "high",
                        "Low": "low", "Close": "close"})
df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
df = df[["time", "open", "high", "low", "close"]].dropna().round(5)

df.to_csv("prices.csv", index=False)
print(f"Saved {len(df)} candles to prices.csv")
print(df.head(3).to_string(index=False))
