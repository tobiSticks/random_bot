"""
Which clock does your downloaded data use (London, New York, or fixed UTC)?
It looks at which hours of the day are busiest, in winter and in summer.

Run:  python clock_check.py
Needs data/EURUSD_1h.csv (and data/GBPUSD_1h.csv if you have it).
"""
import os

import numpy as np
import pandas as pd


def busiest(df, months, k=3):
    d = df[df["time"].dt.month.isin(months)]
    rng = ((d["high"] - d["low"]) / d["close"]).groupby(d["time"].dt.hour).mean()
    return list(rng.sort_values(ascending=False).head(k).index)


def check(symbol):
    path = os.path.join("data", f"{symbol}_1h.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    w, s = busiest(df, [12, 1, 2]), busiest(df, [6, 7, 8])
    print(f"{symbol}: busiest hours in winter (Dec-Feb) {sorted(w)}, in summer (Jun-Aug) {sorted(s)}")
    return sorted(w), sorted(s)


def verdict(w, s):
    if sorted(h - 1 for h in w) == s or len(set(w) & set(s)) <= 1 and len(set(h - 1 for h in w) & set(s)) >= 2:
        return "The busy hours shift by one hour with the seasons: a fixed clock (UTC)."
    if len(set(w) & set(s)) >= 2:
        hours = set(w) | set(s)
        if hours & {7, 8, 9} and hours & {13, 14, 15}:
            return "The busy hours stay put all year, around 7-9 and 13-15: London time (as the bot assumes)."
        if hours & {2, 3, 4}:
            return "The busy hours stay put all year, around 2-4 and 8-10: New York time."
        return "The busy hours stay put all year (a local clock with daylight saving), but the hours are unusual."
    return "The pattern is unclear."


if __name__ == "__main__":
    results = [r for r in (check("EURUSD"), check("GBPUSD")) if r]
    if not results:
        print("No data/EURUSD_1h.csv found.")
    else:
        for w, s in results[:1]:
            print("\n" + verdict(w, s))
        print("\nSend this output to Claude if the verdict says 'unusual' or 'unclear'.")
