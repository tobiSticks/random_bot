"""
Downloads candles for 28 forex pairs plus gold from Yahoo Finance into a folder called 'data'.

Setup (once):  pip install yfinance        (you already did this)
Run:           python get_multi_data.py
Then test:     python multi_backtest.py

To download fewer pairs, delete names from the SYMBOLS list below.
Yahoo gives about 2 years of 1-hour candles, but only about 60 days of 5/15-minute candles.
"""
import os
import time

import pandas as pd
import yfinance as yf

INTERVAL = "1h"     # "1h" (about 2 years), or "15m" / "5m" (about 60 days)
PERIOD = "730d"     # use "60d" when INTERVAL is "15m" or "5m"

FOREX = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF",
         "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
         "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
         "AUDJPY", "AUDCAD", "AUDCHF", "AUDNZD", "NZDJPY", "NZDCAD", "NZDCHF",
         "CADJPY", "CADCHF", "CHFJPY"]
SYMBOLS = {name: [name + "=X"] for name in FOREX}
SYMBOLS["XAUUSD"] = ["XAUUSD=X", "GC=F"]    # spot gold first, gold futures as a fallback

os.makedirs("data", exist_ok=True)
done, failed = [], []

for name, tickers in SYMBOLS.items():
    df = None
    for t in tickers:
        try:
            d = yf.download(t, interval=INTERVAL, period=PERIOD, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if len(d) > 500:
                df = d
                break
        except Exception:
            pass
    if df is None:
        failed.append(name)
        print(f"{name}: no data")
        continue
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "time", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close"})
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df[["time", "open", "high", "low", "close"]].dropna()
    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df[price_cols].round(5 if name != "XAUUSD" else 2)
    df.to_csv(os.path.join("data", f"{name}_{INTERVAL}.csv"), index=False)
    done.append(name)
    print(f"{name}: saved {len(df)} candles")
    time.sleep(1)   # be polite to Yahoo

print(f"\nDone: {len(done)} symbols saved in the 'data' folder.")
if failed:
    print("No data for:", ", ".join(failed))
