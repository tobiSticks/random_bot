"""
Backtester for the 3-step model from the video:
  1. Fair value gap (FVG) created by a strong push
  2. Liquidity: a swing low that forms just above the FVG
  3. Break of structure (BOS): price closes above the high that created that liquidity
Then price returns to the FVG and gives a bullish reaction -> enter on that candle's close.
Stop below the deepest wick, target 1:3, stop moved to breakeven at 1:1.5.
Shorts are the exact mirror image.

Usage:  python fvg_backtest.py prices.csv
CSV needs columns: time, open, high, low, close
This only backtests. It does not place any trades.
"""
import sys
import numpy as np
import pandas as pd

CFG = dict(
    ema_period=None,   # optional trend filter, e.g. 50. None = off (the video doesn't use one)
    swing_n=3,         # bars on each side needed to confirm a swing high/low
    atr_period=14,
    impulse_atr=1.2,   # the "strong push" candle must be >= this x ATR
    rr=3.0,            # take profit at 1:3
    be_rr=1.5,         # move stop to breakeven at 1:1.5 (None = off)
    stop_buffer=0.0,   # extra distance below the deepest wick (price units)
    react_bars=2,      # the bullish candle (or the one before it) must touch the FVG
    max_wait=100,      # bars a setup stays valid
)


def swings(h, l, n):
    """Swing levels, stored at the bar where each swing becomes confirmed."""
    N = len(h)
    sh = np.full(N, np.nan); sh_bar = np.full(N, -1)
    sl = np.full(N, np.nan); sl_bar = np.full(N, -1)
    sh_own = np.full(N, np.nan)  # swing high level at its own bar
    for i in range(n, N - n):
        j = i + n  # only known n bars later, so there is no look-ahead
        if h[i] >= h[i - n:i + n + 1].max():
            sh[j], sh_bar[j], sh_own[i] = h[i], i, h[i]
        if l[i] <= l[i - n:i + n + 1].min():
            sl[j], sl_bar[j] = l[i], i
    return sh, sh_bar, sl, sl_bar, sh_own


def try_exit(pos, o, h, l, i, be_rr):
    """Stop is checked first (conservative if both are hit in one bar)."""
    if l[i] <= pos["stop"]:
        px = min(pos["stop"], o[i])
    elif h[i] >= pos["tp"]:
        px = max(pos["tp"], o[i])
    else:
        # breakeven applies from the next bar (order inside a bar is unknown)
        if be_rr is not None and h[i] >= pos["entry"] + be_rr * pos["risk"]:
            pos["stop"] = pos["entry"]
        return None
    return {**pos, "exit_i": i, "exit": px, "r": (px - pos["entry"]) / pos["risk"]}


def run_long(o, h, l, c, cfg):
    N = len(c)
    use_ema = cfg["ema_period"] is not None
    ema = pd.Series(c).ewm(span=cfg["ema_period"], adjust=False).mean().to_numpy() if use_ema else None
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).rolling(cfg["atr_period"]).mean().to_numpy()
    sh, sh_bar, sl, sl_bar, sh_own = swings(h, l, cfg["swing_n"])
    zones, trades, pos = [], [], None

    for i in range(2, N):
        if pos is not None:
            t = try_exit(pos, o, h, l, i, cfg["be_rr"])
            if t:
                trades.append(t); pos = None

        # 1) fair value gap made by a strong push up
        if (l[i] > h[i - 2] and (h[i - 1] - l[i - 1]) >= cfg["impulse_atr"] * atr[i - 1]
                and (not use_ema or c[i] > ema[i])):
            zones.append(dict(lo=h[i - 2], hi=l[i], born=i, bos_level=None, bos_bar=None))

        keep = []
        for z in zones:
            if z["born"] == i:
                keep.append(z); continue
            if z["bos_bar"] is None:
                # 2) liquidity: newest swing low above the FVG, and the high that created it
                if not np.isnan(sl[i]) and sl_bar[i] > z["born"] and sl[i] > z["hi"]:
                    k = next((k for k in range(sl_bar[i] - 1, z["born"] - 1, -1)
                              if not np.isnan(sh_own[k])), None)
                    z["bos_level"] = sh_own[k] if k is not None else h[z["born"]:sl_bar[i]].max()
                # 3) break of structure: a close above that high
                if z["bos_level"] is not None and c[i] > z["bos_level"]:
                    z["bos_bar"] = i
            elif pos is None:
                # 4) back in the FVG with a bullish reaction candle -> enter on its close
                b = z["bos_bar"]
                touched = l[max(b + 1, i - cfg["react_bars"] + 1):i + 1].min() <= z["hi"]
                if touched and c[i] > o[i] and c[i] > z["lo"]:
                    stop = l[b + 1:i + 1].min() - cfg["stop_buffer"]  # below the deepest wick
                    risk = c[i] - stop
                    if risk > 0:
                        pos = dict(i=i, entry=c[i], stop=stop, stop0=stop,
                                   tp=c[i] + cfg["rr"] * risk, risk=risk)
                        continue  # setup used up
            if i - z["born"] > cfg["max_wait"] or c[i] < z["lo"]:
                continue  # expired or invalidated
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
                             entry=s * t["entry"], stop=s * t["stop0"], tp=s * t["tp"],
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


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    df.columns = [x.lower() for x in df.columns]
    if "time" in df:
        df["time"] = pd.to_datetime(df["time"])
    trades = backtest(df)
    report(trades)
    trades.to_csv("trades.csv", index=False)
    print("Saved trades.csv")
