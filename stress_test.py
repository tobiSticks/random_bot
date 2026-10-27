"""
Stress test for strategy 2 with the filter that survived robust_test.py:
  trade only between 17:00 and 24:00 (data-file time), planned R:R >= 1.5, spread <= 15% of the stop.

It asks: does the result survive higher costs, different time periods, and removing the
best pairs or the riskiest hours? And what is a fairer uncertainty range?

Run:  python stress_test.py
Needs: the 'data' folder and liquidity_zone_backtest.py (renamed copies like '-2' are fine).
"""
import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

# ------------------------------ settings ------------------------------
STRATEGY_FILE = "liquidity_zone_backtest"
HOURS = (17, 24)          # entry hour window, as in the data files
MIN_RR = 1.5              # planned reward:risk at least this
MAX_COST = 0.15           # spread at most this share of the stop distance
COST_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]
# trading cost assumptions (same as the other scripts; change to match your broker)
SPREAD_PIPS = {"EURUSD": 1.0, "USDJPY": 1.0, "GBPUSD": 1.2, "AUDUSD": 1.2,
               "USDCAD": 1.5, "USDCHF": 1.5, "NZDUSD": 1.5, "EURGBP": 1.5}
DEFAULT_PIPS, GOLD_SPREAD = 2.5, 0.30
# -----------------------------------------------------------------------


def spread_price(symbol):
    if symbol == "XAUUSD":
        return GOLD_SPREAD
    pip = 0.01 if "JPY" in symbol else 0.0001
    return SPREAD_PIPS.get(symbol, DEFAULT_PIPS) * pip


def find_module(prefix):
    files = glob.glob(prefix + "*.py")
    if not files:
        return None
    path = max(files, key=os.path.getmtime)
    spec = importlib.util.spec_from_file_location(prefix, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_prices(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def f(x, n=None):
    return "   n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def week_bootstrap(g, reps=3000):
    """95% range for the average, resampling whole calendar weeks (pairs trade at the same time)."""
    wk = g["entry_time"].dt.to_period("W").astype(str)
    s = g.groupby(wk)["net"].agg(["sum", "count"])
    if len(s) < 5:
        return float("nan"), float("nan")
    sums, counts = s["sum"].to_numpy(), s["count"].to_numpy()
    idx = np.random.default_rng(0).integers(0, len(s), (reps, len(s)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    files = sorted(glob.glob(os.path.join("data", "*.csv")))
    if not files:
        sys.exit("No CSV files in the 'data' folder. Run get_multi_data.py first.")
    mod = find_module(STRATEGY_FILE)
    if mod is None:
        sys.exit(f"{STRATEGY_FILE}.py is not in this folder.")

    rows, tmin, tmax = [], None, None
    for fpath in files:
        symbol = os.path.basename(fpath).split("_")[0]
        df = read_prices(fpath)
        tmin = df["time"].min() if tmin is None else min(tmin, df["time"].min())
        tmax = df["time"].max() if tmax is None else max(tmax, df["time"].max())
        print(f"Testing {symbol} ({len(df)} candles)...", flush=True)
        tr = mod.backtest(df)
        if tr is None or tr.empty:
            continue
        tr = tr.copy()
        risk = (tr["entry"] - tr["stop"]).abs()
        cost = spread_price(symbol)
        tr["cost_frac"] = cost / risk
        tr["plan_rr"] = (tr["tp"] - tr["entry"]).abs() / risk
        tr["entry_time"] = pd.to_datetime(tr["entry_time"])
        tr["hour"] = tr["entry_time"].dt.hour
        tr["symbol"] = symbol
        rows.append(tr)
    if not rows:
        sys.exit("No trades found.")
    t = pd.concat(rows, ignore_index=True)
    sel = ((t["hour"] >= HOURS[0]) & (t["hour"] < HOURS[1]) & (t["plan_rr"] >= MIN_RR)
           & (t["cost_frac"] <= MAX_COST))
    g = t[sel].copy()
    if len(g) < 30:
        sys.exit(f"Only {len(g)} trades match the filter; too few to stress test.")
    g["net"] = g["r"] - g["cost_frac"]

    print(f"\n=== FILTERED TRADES: hours {HOURS[0]}-{HOURS[1]}, R:R >= {MIN_RR}, spread <= {MAX_COST:.0%} of stop ===")
    lo, hi = week_bootstrap(g)
    print(f"{len(g)} trades, win {((g['r'] > 0).mean() * 100):.0f}%, net avgR {f(g['net'].mean())}")
    print(f"Fairer 95% range (resampling whole weeks): {f(lo)} to {f(hi)}")

    print("\n=== 1) HIGHER COSTS (same trades, bigger spread) ===")
    checks = {}
    for m in COST_MULTIPLIERS:
        v = (g["r"] - m * g["cost_frac"]).mean()
        print(f"   spread x{m:<4}  net avgR {f(v)}")
        if m == 2.0:
            checks["Still positive at 2x costs"] = v > 0

    print("\n=== 2) FOUR TIME PERIODS ===")
    edges = [tmin + (tmax - tmin) * k / 4 for k in range(5)]
    pos = 0
    for k in range(4):
        p = g[(g["entry_time"] >= edges[k]) & (g["entry_time"] <= edges[k + 1])]
        v = p["net"].mean() if len(p) else float("nan")
        pos += bool(len(p) >= 20 and v > 0)
        print(f"   {edges[k]:%Y-%m} to {edges[k + 1]:%Y-%m}:  {len(p):4d} trades  net avgR {f(v)}")
    checks["Positive in at least 3 of 4 periods"] = pos >= 3

    print("\n=== 3) PAIRS ===")
    s = g.groupby("symbol")["net"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False)
    print(f"   Profitable pairs: {(s['sum'] > 0).sum()} of {len(s)}")
    print("   Best 5 (trades, net avgR):  " + ", ".join(f"{i} ({r['count']:.0f}, {r['mean']:+.2f})" for i, r in s.head(5).iterrows()))
    print("   Worst 5:                    " + ", ".join(f"{i} ({r['count']:.0f}, {r['mean']:+.2f})" for i, r in s.tail(5).iterrows()))
    top3 = s.head(3)
    rest_n, rest_sum = len(g) - top3["count"].sum(), g["net"].sum() - top3["sum"].sum()
    rest = rest_sum / rest_n if rest_n > 0 else float("nan")
    print(f"   Without the 3 best pairs ({', '.join(top3.index)}): net avgR {f(rest)} over {rest_n:.0f} trades")
    checks["Still positive without the 3 best pairs"] = rest > 0

    print("\n=== 4) HOUR OF ENTRY ===")
    for h, p in g.groupby("hour"):
        print(f"   {h:02d}:00  {len(p):4d} trades  net avgR {f(p['net'].mean())}")
    calm = g[g["hour"] < 21]
    print(f"   Excluding 21:00-24:00 (around the daily rollover): net avgR {f(calm['net'].mean())} over {len(calm)} trades")

    print("\n=== 5) LONGS VS SHORTS ===")
    for side, p in g.groupby("side"):
        print(f"   {side:5s} {len(p):4d} trades  net avgR {f(p['net'].mean())}")

    print("\n=== CHECKS ===")
    checks["Fair 95% range stays above zero"] = (not np.isnan(lo)) and lo > 0
    for k, v in checks.items():
        print(f"   [{'PASS' if v else 'FAIL'}] {k}")
    n_ok = sum(checks.values())
    print(f"\n{n_ok} of {len(checks)} checks passed.")
    print("All passed: worth a slow forward test on a demo account. Fewer: treat the result as fragile or luck.")
    g.to_csv("stress_trades.csv", index=False)
    print("Saved the filtered trades to stress_trades.csv")


if __name__ == "__main__":
    main()
