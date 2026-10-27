"""
Honest test: can a time-of-day, minimum-reward and cost filter rescue strategy 2 or 3?

How it avoids fooling itself:
  * The data is split in half by date. Filters are CHOSEN using only the first half.
  * The chosen filter is then judged on the second half, which it has never seen.
  * If the second-half result is not clearly above zero, no real edge was found.

Run:  python robust_test.py
Needs: the 'data' folder from get_multi_data.py, plus the strategy files in this folder
       (liquidity_zone_backtest.py and ifvg_backtest.py; renamed copies like '-2' are fine).
Times are used exactly as they appear in the data files (Yahoo gives London time for forex).
"""
import glob
import importlib.util
import itertools
import os
import sys

import numpy as np
import pandas as pd

# ------------------------------ settings ------------------------------
STRATEGIES = [   # label, file prefix
    ("2 Zone+tap+sweep", "liquidity_zone_backtest"),
    ("3 Sweep+IFVG", "ifvg_backtest"),
]
SESSIONS = {"all hours": (0, 24), "00-07": (0, 7), "07-12": (7, 12), "12-17": (12, 17), "17-24": (17, 24)}
MIN_RRS = [1.0, 1.5, 2.0]          # skip trades whose planned reward:risk is below this
MAX_COSTS = [None, 0.15, 0.08]     # allow a trade only if the spread is at most this share of its stop
MIN_TRAIN, MIN_TEST = 150, 50      # smallest sample we accept
# trading cost assumptions (same as multi_backtest.py; change to match your broker)
SPREAD_PIPS = {"EURUSD": 1.0, "USDJPY": 1.0, "GBPUSD": 1.2, "AUDUSD": 1.2,
               "USDCAD": 1.5, "USDCHF": 1.5, "NZDUSD": 1.5, "EURGBP": 1.5}
DEFAULT_PIPS, GOLD_SPREAD, COST_MULTIPLIER = 2.5, 0.30, 1.0
# -----------------------------------------------------------------------


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


def read_prices(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def select(df, sess, rr, max_cost):
    a, b = SESSIONS[sess]
    m = (df["hour"] >= a) & (df["hour"] < b) & (df["plan_rr"] >= rr)
    if max_cost is not None:
        m &= df["cost_frac"] <= max_cost
    return df[m]


def stats(g):
    n = len(g)
    if n == 0:
        return n, float("nan"), float("nan"), float("nan")
    mean = g["net"].mean()
    se = g["net"].std() / np.sqrt(n) if n > 1 else float("nan")
    return n, mean, mean - 1.96 * se, mean + 1.96 * se


def main():
    files = sorted(glob.glob(os.path.join("data", "*.csv")))
    if not files:
        sys.exit("No CSV files in the 'data' folder. Run get_multi_data.py first.")
    mods = []
    for label, prefix in STRATEGIES:
        m = find_module(prefix)
        if m is None:
            print(f"Skipping '{label}': {prefix}.py is not in this folder.")
        else:
            mods.append((label, m))
    if not mods:
        sys.exit("No strategy files found.")

    rows, tmin, tmax = [], None, None
    for f in files:
        symbol = os.path.basename(f).split("_")[0]
        df = read_prices(f)
        tmin = df["time"].min() if tmin is None else min(tmin, df["time"].min())
        tmax = df["time"].max() if tmax is None else max(tmax, df["time"].max())
        print(f"Testing {symbol} ({len(df)} candles)...", flush=True)
        cost = spread_price(symbol)
        for label, m in mods:
            tr = m.backtest(df)
            if tr is None or tr.empty:
                continue
            tr = tr.copy()
            risk = (tr["entry"] - tr["stop"]).abs()
            tr["net"] = tr["r"] - cost / risk
            tr["cost_frac"] = cost / risk
            tr["plan_rr"] = (tr["tp"] - tr["entry"]).abs() / risk
            tr["hour"] = pd.to_datetime(tr["entry_time"]).dt.hour
            tr["symbol"], tr["strategy"] = symbol, label
            rows.append(tr)
    if not rows:
        sys.exit("No trades found.")
    trades = pd.concat(rows, ignore_index=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    split = tmin + (tmax - tmin) / 2
    print(f"\nData runs {tmin:%Y-%m-%d} to {tmax:%Y-%m-%d}.")
    print(f"TRAIN = before {split:%Y-%m-%d} (used to choose filters).  TEST = after it (judging only).")
    combos = list(itertools.product(SESSIONS, MIN_RRS, MAX_COSTS))
    print(f"Filter combinations tried on the TRAIN half, per strategy: {len(combos)}")

    grid = []
    for label, _ in mods:
        T = trades[trades["strategy"] == label]
        train, test = T[T["entry_time"] < split], T[T["entry_time"] >= split]
        results = []
        for sess, rr, mc in combos:
            n1, m1, _, _ = stats(select(train, sess, rr, mc))
            n2, m2, lo2, hi2 = stats(select(test, sess, rr, mc))
            results.append(dict(strategy=label, session=sess, min_rr=rr, max_cost=mc,
                                train_n=n1, train_net=m1, test_n=n2, test_net=m2, test_lo=lo2, test_hi=hi2))
        grid += results
        R = pd.DataFrame(results)
        base = R[(R.session == "all hours") & (R.min_rr == 1.0) & (R.max_cost.isna())].iloc[0]
        ok = R[R.train_n >= MIN_TRAIN].sort_values("train_net", ascending=False)

        print(f"\n===== {label} =====")
        print(f"No filters:   TRAIN net avgR {base.train_net:+.3f} ({base.train_n:.0f} trades)   "
              f"TEST net avgR {base.test_net:+.3f} ({base.test_n:.0f} trades)")
        if ok.empty:
            print("No filter combination had enough trades in the TRAIN half.")
            continue
        print("Best 3 filters on the TRAIN half (chosen WITHOUT looking at TEST):")
        for _, r in ok.head(3).iterrows():
            mc = "no limit" if pd.isna(r.max_cost) else f"{r.max_cost:.2f}"
            print(f"   hours {r.session:9s} min R:R {r.min_rr:.1f}  spread/stop {mc:8s} -> TRAIN {r.train_net:+.3f} ({r.train_n:.0f})")
        w = ok.iloc[0]
        mc = "no limit" if pd.isna(w.max_cost) else f"{w.max_cost:.2f}"
        print(f"\nWINNING FILTER: hours {w.session}, min R:R {w.min_rr:.1f}, spread/stop {mc}")
        print(f"   TRAIN: net avgR {w.train_net:+.3f} over {w.train_n:.0f} trades")
        if w.test_n < MIN_TEST:
            print(f"   TEST:  only {w.test_n:.0f} trades, too few to judge")
        else:
            print(f"   TEST:  net avgR {w.test_net:+.3f} over {w.test_n:.0f} trades "
                  f"(95% range {w.test_lo:+.3f} to {w.test_hi:+.3f})")
            if w.test_lo > 0:
                print("   -> The edge held on unseen data. Worth a careful follow-up (more data, other timeframes).")
            elif w.test_net > 0:
                print("   -> Positive on unseen data but not convincing: the range includes zero.")
            else:
                print("   -> It did NOT hold on unseen data. No edge found.")

    pd.DataFrame(grid).to_csv("robust_results.csv", index=False)
    print("\nSaved every combination to robust_results.csv.")
    print("Do not pick a filter from the TEST column: that is the same trap as before.")
    print("Trades on different pairs overlap in time, so real uncertainty is wider than shown.")


if __name__ == "__main__":
    main()
