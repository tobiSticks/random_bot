"""
Backtester for the "protected low zone + inducement sweep" strategy from the video.

Long setup (shorts are the exact mirror image):
  1. Price breaks structure: a candle closes above the latest swing high.
     The lowest low behind that break is the "protected low".
  2. Draw a zone from the broken high (top) down to the protected low (bottom).
     Do nothing and wait.
  3. Price returns for a SHALLOW first tap into the top of the zone, then pushes higher.
     The low of that tap is where breakout traders have their stop losses (liquidity).
  4. Enter only once price comes back and sweeps that tap low (inducement).
     Entry at the swept level, stop lower inside the zone for breathing room,
     target the high of the push.

Usage:  python liquidity_zone_backtest.py prices.csv
CSV needs columns: time (or date), open, high, low, close
This only backtests. It does not place any trades.
"""
import sys
import numpy as np
import pandas as pd

CFG = dict(
    swing_n=5,          # bars on each side needed to confirm a swing high
    atr_period=14,
    min_zone_atr=2.0,   # zone must be at least this many ATRs tall (filters tiny zones)
    shallow=0.25,       # first tap may go at most this deep into the zone (0.25 = top quarter)
    stop_frac=0.75,     # stop sits this far down the zone (0 = top, 1 = protected low)
    min_push_atr=1.0,   # price must rally at least this many ATRs above the tap low
    min_rr=1.0,         # skip trades whose reward:risk is below this
    max_wait=300,       # bars a setup stays valid
)


def swing_highs(h, n):
    """Swing highs, stored at the bar where each one becomes confirmed (no look-ahead)."""
    N = len(h)
    lvl = np.full(N, np.nan); bar = np.full(N, -1)
    for i in range(n, N - n):
        if h[i] >= h[i - n:i + n + 1].max():
            lvl[i + n], bar[i + n] = h[i], i
    return lvl, bar


def run_long(o, h, l, c, cfg):
    N = len(c)
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).rolling(cfg["atr_period"]).mean().to_numpy()
    sh, sh_bar = swing_highs(h, cfg["swing_n"])
    zones, trades, pos = [], [], None
    last_lvl, last_bar, used_bar = None, -1, -1

    for i in range(1, N):
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

        if not np.isnan(sh[i]):
            last_lvl, last_bar = sh[i], sh_bar[i]

        # 1) break of structure -> draw the zone: broken high down to the protected low
        if (last_lvl is not None and last_bar != used_bar and c[i] > last_lvl
                and not np.isnan(atr[i])):
            used_bar = last_bar
            top, bot = last_lvl, l[last_bar:i + 1].min()
            if top - bot >= cfg["min_zone_atr"] * atr[i]:
                zones.append(dict(top=top, bot=bot, born=i, state="wait"))

        keep = []
        for z in zones:
            if z["born"] == i:
                keep.append(z); continue
            height = z["top"] - z["bot"]
            st = z["state"]
            expired = i - z["born"] > cfg["max_wait"]

            # 2) wait for price to tap back into the zone
            if st == "wait":
                if expired or c[i] < z["bot"]:
                    continue
                if l[i] <= z["top"]:
                    z["state"] = st = "tap"; z["low"] = l[i]
                else:
                    keep.append(z); continue

            # 3) the first tap must be shallow; once price closes back above the zone, it counts
            if st == "tap":
                z["low"] = min(z["low"], l[i])
                if expired or c[i] < z["bot"] or z["low"] < z["top"] - cfg["shallow"] * height:
                    continue  # tap went too deep: setup is spoiled
                if c[i] > z["top"]:
                    z["state"] = "pushed"; z["liq"] = z["low"]; z["hmax"] = h[i]
                keep.append(z); continue

            # 4) price pushed higher; enter when it sweeps the tap low
            if st == "pushed":
                if l[i] <= z["liq"]:
                    entry = min(o[i], z["liq"])
                    stop = z["top"] - cfg["stop_frac"] * height
                    tp = z["hmax"]
                    risk = entry - stop
                    if (pos is None and risk > 0
                            and (z["hmax"] - z["liq"]) >= cfg["min_push_atr"] * atr[i]
                            and (tp - entry) / risk >= cfg["min_rr"]):
                        pos = dict(i=i, entry=entry, stop=stop, tp=tp, risk=risk)
                        if l[i] <= stop:  # stopped out on the entry bar (conservative)
                            trades.append({**pos, "exit_i": i, "exit": stop, "r": -1.0})
                            pos = None
                    continue  # setup used up
                if expired:
                    continue
                z["hmax"] = max(z["hmax"], h[i])
                keep.append(z)
        zones = keep
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
    trades = backtest(load(sys.argv[1]))
    report(trades)
    trades.to_csv("trades_zone.csv", index=False)
    print("Saved trades_zone.csv")
