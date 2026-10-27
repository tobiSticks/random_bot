"""
Backtester for the "liquidity sweep + inverse fair value gap (IFVG)" strategy from the video.

Long setup (shorts are the exact mirror image):
  1. Liquidity: a swing low. Price comes back and sweeps it (trades below it).
  2. Fair value gap: the bearish FVG that formed on the way down into that sweep.
  3. Change in state of delivery (CISD): price closes above the high of the candles
     that created that FVG. The bearish FVG is now an INVERSE FVG (support).
  4. Wait for price to come back into the inverse FVG. Enter at its top.
  5. Stop at the bottom of the inverse FVG. Target the high of the rally ("some highs").

Usage:  python ifvg_backtest.py prices.csv
CSV needs columns: time (or date), open, high, low, close
This only backtests. It does not place any trades.
"""
import sys
import numpy as np
import pandas as pd

CFG = dict(
    swing_n=5,         # bars on each side needed to confirm a swing low (the liquidity)
    atr_period=14,
    fvg_lookback=30,   # how far back from the sweep to look for the FVG
    min_gap_atr=0.3,   # the FVG must be at least this many ATRs tall (filters tiny gaps)
    min_rr=1.0,        # skip trades whose reward:risk is below this
    max_wait=150,      # bars a setup stays valid
)


def swing_lows(l, n):
    """Swing lows, stored at the bar where each one becomes confirmed (no look-ahead)."""
    N = len(l)
    lvl = np.full(N, np.nan); bar = np.full(N, -1)
    for i in range(n, N - n):
        if l[i] <= l[i - n:i + n + 1].min():
            lvl[i + n], bar[i + n] = l[i], i
    return lvl, bar


def run_long(o, h, l, c, cfg):
    N = len(c)
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).rolling(cfg["atr_period"]).mean().to_numpy()
    sl, sl_bar = swing_lows(l, cfg["swing_n"])
    setups, trades, pos = [], [], None
    last_lvl, last_bar, used_bar = None, -1, -1

    for i in range(2, N):
        # manage the open position (stop is checked first)
        if pos is not None:
            px = None
            if l[i] <= pos["stop"]:
                px = min(pos["stop"], o[i])
            elif h[i] >= pos["tp"]:
                px = max(pos["tp"], o[i])
            if px is not None:
                trades.append({**pos, "exit_i": i, "exit": px, "r": (px - pos["entry"]) / pos["risk"]})
                pos = None

        if not np.isnan(sl[i]):
            last_lvl, last_bar = sl[i], sl_bar[i]

        # 1) liquidity sweep: price trades below the latest swing low
        if last_lvl is not None and last_bar != used_bar and l[i] < last_lvl:
            used_bar = last_bar
            # 2) the most recent bearish FVG on the way down (3 candles, gap between
            #    the low of the first and the high of the third)
            lo_j = max(last_bar + 3, i - cfg["fvg_lookback"], 2)
            for j in range(i, lo_j - 1, -1):
                top, bot = l[j - 2], h[j]
                if bot < top and (top - bot) >= cfg["min_gap_atr"] * atr[j]:
                    setups.append(dict(top=top, bot=bot, cisd=h[j - 2:j + 1].max(),
                                       born=i, state="await"))
                    break

        keep = []
        for s in setups:
            expired = i - s["born"] > cfg["max_wait"]
            if s["state"] == "await":
                if expired:
                    continue
                # 3) CISD: a close above the high that created the FVG -> inverse FVG
                if c[i] > s["cisd"]:
                    s["state"] = "ifvg"; s["ifvg_bar"] = i
                keep.append(s)
                continue
            # 4) inverse FVG: enter when price comes back down to its top
            if i > s["ifvg_bar"]:
                if l[i] <= s["top"]:
                    entry = min(o[i], s["top"])
                    stop = s["bot"]
                    tp = h[s["born"]:i].max()      # the high of the rally
                    risk = entry - stop
                    if pos is None and risk > 0 and (tp - entry) / risk >= cfg["min_rr"]:
                        pos = dict(i=i, entry=entry, stop=stop, tp=tp, risk=risk)
                        if l[i] <= stop:           # stopped out on the entry bar (conservative)
                            trades.append({**pos, "exit_i": i, "exit": stop, "r": -1.0})
                            pos = None
                    continue                        # setup used up
            if expired:
                continue
            keep.append(s)
        setups = keep
    return trades


def backtest(df, cfg=CFG):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    times = df["time"].to_numpy() if "time" in df else df.index.to_numpy()
    rows = []
    for side, s in (("long", 1), ("short", -1)):
        # shorts reuse the long logic on a mirrored (negated) price series
        args = (o, h, l, c) if s == 1 else (-o, -l, -h, -c)
        for t in run_long(*args, cfg):
            rows.append(dict(side=side, entry_time=times[t["i"]], exit_time=times[t["exit_i"]],
                             entry=s * t["entry"], stop=s * t["stop"], tp=s * t["tp"],
                             exit=s * t["exit"], r=t["r"]))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


def report(tr):
    if tr.empty:
        print("No trades found."); return
    for name, g in [("all", tr), *tr.groupby("side")]:
        cum = g["r"].cumsum()
        print(f"{name:5s} trades={len(g)}  win%={(g['r'] > 0).mean() * 100:.0f}  "
              f"avgR={g['r'].mean():.2f}  totalR={g['r'].sum():.1f}  "
              f"maxDD(R)={(cum.cummax() - cum).max():.1f}")


def load(path):
    """Reads CSVs from MT5, Dukascopy, Yahoo, etc. (comma or tab separated, any header style)."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(x).strip("<> ").lower() for x in df.columns]
    if "date" in df and "time" in df:          # MT5 style: separate date and time columns
        df["time"] = df["date"].astype(str) + " " + df["time"].astype(str)
    elif "date" in df:
        df["time"] = df["date"]
    elif "datetime" in df:
        df["time"] = df["datetime"]
    if "time" in df:
        df["time"] = pd.to_datetime(df["time"])
    missing = [k for k in ("open", "high", "low", "close") if k not in df]
    if missing:
        sys.exit(f"Missing columns {missing}. Columns found: {list(df.columns)}")
    flat = (df["high"] == df["low"]).mean()
    if flat > 0.5:
        print(f"WARNING: {flat:.0%} of candles are completely flat (high = low). "
              "The prices look rounded or wrong. Re-export the data with full decimals.")
    print(f"Loaded {len(df)} candles, price range {df['low'].min():g} to {df['high'].max():g}")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python ifvg_backtest.py prices.csv")
    trades = backtest(load(sys.argv[1]))
    report(trades)
    trades.to_csv("trades_ifvg.csv", index=False)
    print("Saved trades_ifvg.csv")
