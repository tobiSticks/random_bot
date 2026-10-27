"""
Backtester for the "A+ setup" strategy from the video: fresh BOS zone + liquidity sweep.

Long setup (shorts are the exact mirror image):
  1. Clear trend: the high that gets broken is a higher high, and the low behind the
     break is a higher low.
  2. Mark the last clean BOS: price closes above the latest swing high. Draw a zone from
     that broken high (top) down to the low behind the break (bottom).
  3. The area must contain liquidity: a swing low sitting inside the zone.
  4. The zone must still be fresh: price has not closed back inside it since the break.
  5. Wait for the sweep of that liquidity (price trades down through the swing low),
     then enter at that level. Stop at the bottom of the zone, target the recent high.

Usage:  python bos_zone_backtest.py prices.csv
CSV needs columns: time (or date), open, high, low, close
This only backtests. It does not place any trades.
"""
import sys
import numpy as np
import pandas as pd

CFG = dict(
    swing_n=5,           # bars on each side needed to confirm a swing high/low
    atr_period=14,
    min_zone_atr=1.5,    # the zone must be at least this many ATRs tall
    min_rr=1.0,          # skip trades whose reward:risk is below this
    max_wait=300,        # bars a zone stays valid
    require_trend=True,  # require a higher high and a higher low at the break
    trend_lookback=300,  # how far back to look for the previous swing points
    liq_n=2,             # swing size for the liquidity lows inside the zone (smaller = more lows)
    strict_fresh=True,   # True: any close back inside the zone before the sweep spoils it (as in the video)
                         # False: only a close below the zone bottom spoils it (gives many more trades)
)


def swings(h, l, n):
    """Swing highs/lows. sh/sh_bar are stored at the bar where the swing is confirmed
    (no look-ahead); own_hi/own_lo are stored at the swing's own bar."""
    N = len(h)
    sh = np.full(N, np.nan); sh_bar = np.full(N, -1)
    own_hi = np.full(N, np.nan); own_lo = np.full(N, np.nan)
    for i in range(n, N - n):
        if h[i] >= h[i - n:i + n + 1].max():
            sh[i + n], sh_bar[i + n], own_hi[i] = h[i], i, h[i]
        if l[i] <= l[i - n:i + n + 1].min():
            own_lo[i] = l[i]
    return sh, sh_bar, own_hi, own_lo


def run_long(o, h, l, c, cfg):
    N, n = len(c), cfg["swing_n"]
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).rolling(cfg["atr_period"]).mean().to_numpy()
    sh, sh_bar, own_hi, own_lo = swings(h, l, n)
    liq_lo = swings(h, l, cfg["liq_n"])[3]   # minor swing lows = the liquidity
    zones, trades, pos = [], [], None
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

        if not np.isnan(sh[i]):
            last_lvl, last_bar = sh[i], sh_bar[i]

        # 1+2) clean break of structure -> zone from the broken high down to the low behind it
        if (last_lvl is not None and last_bar != used_bar and c[i] > last_lvl
                and not np.isnan(atr[i])):
            used_bar = last_bar
            top, bot = last_lvl, l[last_bar:i + 1].min()
            ok = top - bot >= cfg["min_zone_atr"] * atr[i]
            if ok and cfg["require_trend"]:
                lb = max(0, last_bar - cfg["trend_lookback"])
                prev_h = next((own_hi[k] for k in range(last_bar - 1, lb - 1, -1)
                               if not np.isnan(own_hi[k])), None)
                prev_l = next((own_lo[k] for k in range(last_bar - 1, lb - 1, -1)
                               if not np.isnan(own_lo[k])), None)
                ok = prev_h is not None and prev_l is not None and prev_h < top and bot > prev_l
            if ok:
                zones.append(dict(top=top, bot=bot, hb=last_bar, bos=i, Q=None))

        keep = []
        for z in zones:
            if z["bos"] == i:
                keep.append(z); continue
            if i - z["bos"] > cfg["max_wait"]:
                continue
            # 3) liquidity inside the zone: the highest minor swing low strictly between bottom and top
            if z["Q"] is None and i - z["bos"] >= n:
                cands = [liq_lo[k] for k in range(z["hb"] + 1, z["bos"])
                         if not np.isnan(liq_lo[k]) and z["bot"] < liq_lo[k] < z["top"]]
                if not cands:
                    continue  # no liquidity in the area -> skip this zone
                z["Q"] = max(cands)
            # 5) the sweep of that liquidity -> enter at the swept level
            if z["Q"] is not None and l[i] <= z["Q"]:
                entry = min(o[i], z["Q"])
                stop = z["bot"]
                tp = h[z["bos"]:i].max()         # the recent high since the break
                risk = entry - stop
                if pos is None and risk > 0 and (tp - entry) / risk >= cfg["min_rr"]:
                    pos = dict(i=i, entry=entry, stop=stop, tp=tp, risk=risk)
                    if l[i] <= stop:             # stopped out on the entry bar (conservative)
                        trades.append({**pos, "exit_i": i, "exit": stop, "r": -1.0})
                        pos = None
                continue                          # setup used up
            # 4) fresh zone: a close back inside it before the sweep spoils the setup
            if (cfg["strict_fresh"] and c[i] <= z["top"]) or c[i] < z["bot"]:
                continue
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
    if len(sys.argv) < 2:
        sys.exit("Usage: python bos_zone_backtest.py prices.csv")
    data = load(sys.argv[1])
    # Strict = exactly as in the video (rare setups). Relaxed = many more trades, so the
    # sample is big enough to judge. Both are shown so you can compare.
    for label, strict in (("STRICT (as in the video)", True), ("RELAXED (more trades)", False)):
        print(f"\n--- {label} ---")
        trades = backtest(data, {**CFG, "strict_fresh": strict})
        report(trades)
        if not trades.empty:
            name = "trades_bos_zone_strict.csv" if strict else "trades_bos_zone_relaxed.csv"
            trades.to_csv(name, index=False)
            print("Saved", name)
