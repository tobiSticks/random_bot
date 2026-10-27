"""
Runs the four strategies on every CSV in the 'data' folder (made by get_multi_data.py),
subtracts a trading cost from every trade, and shows whether any strategy has an edge
across many symbols.

Run:  python multi_backtest.py
Needs these files in the same folder (renamed copies like '-2' are fine):
  fvg_backtest.py, liquidity_zone_backtest.py, ifvg_backtest.py, bos_zone_backtest.py
Uses the strategy settings in those files (made for 5-minute candles; 1-hour may need tuning).
"""
import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

# ------------- trading cost assumptions (typical retail spreads, change to match your broker) -------------
SPREAD_PIPS = {"EURUSD": 1.0, "USDJPY": 1.0, "GBPUSD": 1.2, "AUDUSD": 1.2,
               "USDCAD": 1.5, "USDCHF": 1.5, "NZDUSD": 1.5, "EURGBP": 1.5}
DEFAULT_PIPS = 2.5          # every other pair
GOLD_SPREAD = 0.30          # XAUUSD, in dollars per ounce
COST_MULTIPLIER = 1.0       # 0.5 = half the spread (raw account), 2 = double
# ---------------------------------------------------------------------------------------------------------


def spread_price(symbol):
    if symbol == "XAUUSD":
        return GOLD_SPREAD * COST_MULTIPLIER
    pip = 0.01 if "JPY" in symbol else 0.0001
    return SPREAD_PIPS.get(symbol, DEFAULT_PIPS) * pip * COST_MULTIPLIER


def find_module(prefix):
    files = glob.glob(prefix + "*.py")
    if not files:
        return None
    path = max(files, key=os.path.getmtime)
    spec = importlib.util.spec_from_file_location(prefix, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STRATEGIES = [
    ("1 FVG+sweep+BOS", "fvg_backtest", lambda m, d: m.backtest(d)),
    ("2 Zone+tap+sweep", "liquidity_zone_backtest", lambda m, d: m.backtest(d)),
    ("3 Sweep+IFVG", "ifvg_backtest", lambda m, d: m.backtest(d)),
    ("4 BOS zone", "bos_zone_backtest", lambda m, d: m.backtest(d, {**m.CFG, "strict_fresh": False})),
]


def read_prices(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def main():
    files = sorted(glob.glob(os.path.join("data", "*.csv")))
    if not files:
        sys.exit("No CSV files found in the 'data' folder. Run get_multi_data.py first.")
    mods = []
    for label, prefix, run in STRATEGIES:
        m = find_module(prefix)
        if m is None:
            print(f"Skipping '{label}': {prefix}.py is not in this folder.")
        else:
            mods.append((label, m, run))
    if not mods:
        sys.exit("No strategy files found in this folder.")

    all_trades = []
    for f in files:
        symbol = os.path.basename(f).split("_")[0]
        df = read_prices(f)
        print(f"Testing {symbol} ({len(df)} candles)...", flush=True)
        cost = spread_price(symbol)
        for label, m, run in mods:
            tr = run(m, df)
            if tr is None or tr.empty:
                continue
            tr = tr.copy()
            risk = (tr["entry"] - tr["stop"]).abs()
            tr["net"] = tr["r"] - cost / risk
            tr["symbol"], tr["strategy"] = symbol, label
            all_trades.append(tr)
    if not all_trades:
        sys.exit("No trades were found on any symbol.")
    t = pd.concat(all_trades, ignore_index=True)
    t.to_csv("multi_trades.csv", index=False)

    print("\n=== ALL SYMBOLS COMBINED (net = after the assumed spread) ===")
    print(f"{'strategy':20s} {'trades':>7s} {'win%':>5s} {'gross avgR':>11s} {'net avgR':>9s} "
          f"{'95% range':>17s} {'1st half':>9s} {'2nd half':>9s} {'symbols+':>9s}")
    for label, _, _ in mods:
        g = t[t["strategy"] == label]
        if g.empty:
            continue
        n = len(g)
        se = g["net"].std() / np.sqrt(n) if n > 1 else float("nan")
        mid = g["entry_time"].median()
        h1, h2 = g[g["entry_time"] < mid], g[g["entry_time"] >= mid]
        per_sym = g.groupby("symbol")["net"].mean()
        print(f"{label:20s} {n:7d} {(g['r'] > 0).mean() * 100:5.0f} {g['r'].mean():+11.2f} "
              f"{g['net'].mean():+9.2f} {g['net'].mean() - 1.96 * se:+8.2f}..{g['net'].mean() + 1.96 * se:+.2f} "
              f"{h1['net'].mean():+9.2f} {h2['net'].mean():+9.2f} {(per_sym > 0).sum():4d}/{len(per_sym):<4d}")

    print("\n=== NET avgR PER SYMBOL (number of trades in brackets) ===")
    labels = [lab for lab, _, _ in mods]
    print(f"{'symbol':8s}" + "".join(f"{lab[:14]:>20s}" for lab in labels))
    for sym in sorted(t["symbol"].unique()):
        cells = []
        for lab in labels:
            g = t[(t["symbol"] == sym) & (t["strategy"] == lab)]
            cells.append(f"{g['net'].mean():+.2f} ({len(g)})" if len(g) else "-")
        print(f"{sym:8s}" + "".join(f"{c:>20s}" for c in cells))

    print("\nSaved every trade to multi_trades.csv")
    print("Read the 95% range: if it includes 0 or goes below, the result is not convincing.")
    print("Trades on different pairs often overlap in time, so real uncertainty is wider than shown.")


if __name__ == "__main__":
    main()
