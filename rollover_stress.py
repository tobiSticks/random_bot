"""
1) Rollover stress: strategy 2's filtered trades, charging bigger spreads on late-evening entries.
2) Time-zone check: which clock your downloaded data uses (matters for the bot's entry hours).

Run:  python rollover_stress.py
Needs: stress_trades.csv (made by stress_test.py) and data/EURUSD_1h.csv, both in this folder tree.
"""
import os
import sys

import numpy as np
import pandas as pd

LATE = (21, 24)                 # the hours that made all the profit
LATE_MULTS = [1, 2, 3, 5, 8]    # spread multiples charged on LATE entries
EARLY_MULTS = [1.0, 1.5, 2.0]   # spread multiples charged on all other entries


def fmt(x):
    return "   n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def week_bootstrap(t, col, reps=3000):
    wk = t["entry_time"].dt.to_period("W").astype(str)
    s = t.groupby(wk)[col].agg(["sum", "count"])
    if len(s) < 5:
        return float("nan"), float("nan")
    sums, counts = s["sum"].to_numpy(), s["count"].to_numpy()
    idx = np.random.default_rng(0).integers(0, len(s), (reps, len(s)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rollover_stress():
    if not os.path.exists("stress_trades.csv"):
        print("stress_trades.csv not found. Run stress_test.py first."); return
    t = pd.read_csv("stress_trades.csv")
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    late = t["hour"] >= LATE[0]
    print(f"=== ROLLOVER STRESS ({len(t)} trades; {late.sum()} enter at {LATE[0]}:00-{LATE[1]}:00) ===")
    print("Average result per trade when late entries pay X times the normal spread")
    print("(rows) and all other entries pay Y times (columns):\n")
    print("            " + "".join(f"  others x{m:<4}" for m in EARLY_MULTS))
    for lm in LATE_MULTS:
        cells = []
        for em in EARLY_MULTS:
            net = t["r"] - np.where(late, lm, em) * t["cost_frac"]
            cells.append(f"   {net.mean():+.3f}    ")
        print(f"  late x{lm:<3}" + "".join(cells))

    L = t[late]
    m_star = L["r"].mean() / L["cost_frac"].mean() if L["cost_frac"].mean() > 0 else float("nan")
    print(f"\nLate entries alone ({len(L)} trades): result before costs {fmt(L['r'].mean())}; "
          f"they stop paying for themselves at about {m_star:.1f}x the assumed spread.")
    scen = t["r"] - np.where(late, 3, 1.5) * t["cost_frac"]
    t2 = t.assign(scen=scen)
    lo, hi = week_bootstrap(t2, "scen")
    print(f"Scenario 'late x3, others x1.5': net avgR {fmt(scen.mean())}, fairer 95% range {fmt(lo)} to {fmt(hi)}")
    print("Real spreads around the daily rollover can be several times normal. Compare these rows with")
    print("what your demo account actually shows at 21:00-24:00 in the data's clock.")


def timezone_check():
    path = os.path.join("data", "EURUSD_1h.csv")
    print("\n=== TIME-ZONE CHECK OF YOUR DATA ===")
    if not os.path.exists(path):
        print("data/EURUSD_1h.csv not found; skipped."); return
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    opens = df.loc[df["time"].diff() > pd.Timedelta(hours=24), "time"]    # first candle after each weekend
    if len(opens) < 10:
        print("Not enough weekend gaps found to tell."); return
    def mode_hour(months):
        h = opens[opens.dt.month.isin(months)].dt.hour
        return int(h.mode().iloc[0]) if len(h) else None
    w, s = mode_hour([12, 1, 2]), mode_hour([6, 7, 8])
    print(f"Weekly open (first candle after the weekend): most common hour is {w} in Dec-Feb and {s} in Jun-Aug.")
    if w is None or s is None:
        print("Could not compare winter and summer."); return
    if w == s:
        print("Same hour all year: the data uses a local clock with daylight saving "
              "(for example London or New York).")
        guess = {22: "London time (as I assumed)", 17: "New York time", 23: "Central European time"}.get(w)
        print(f"Hour {w} matches: {guess}." if guess else f"Hour {w} is unusual; send this output to Claude.")
    elif w - s == 1:
        print("The hour shifts by one with the seasons: the data looks like a fixed clock (UTC).")
    else:
        print("The pattern is unclear; send this output to Claude.")


if __name__ == "__main__":
    rollover_stress()
    timezone_check()
