"""
DEMO-ACCOUNT bot for strategy 2: protected-low zone + shallow tap + liquidity sweep.

How it works: after every closed candle it re-reads recent prices, looks for setups that
are waiting for the sweep, and keeps a LIMIT order at the sweep level with the stop and
target already attached. When one fills, it cancels the others (one trade at a time).

Safety:
  * It refuses to run unless the MT5 account is a DEMO account.
  * DRY_RUN = True (default) only prints what it WOULD do. Nothing is placed.
  * Every order has a stop and a target attached on the broker's side.
  * It stops placing orders after a daily loss limit and when the spread is too wide.

Needs: Windows, MetaTrader 5 open and logged in to a demo account, with Algo Trading ON.
Setup: pip install MetaTrader5 pandas numpy
Run:   python demo_bot.py        (Ctrl+C to stop; open orders/trades stay on the account)
Keep liquidity_zone_backtest.py in the same folder (use the newest version).
"""
import csv
import math
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import liquidity_zone_backtest as lz

# ------------------------ settings you can change ------------------------
SYMBOL = "EURUSD"          # your broker may add a suffix, e.g. "EURUSD.m" or "EURUSDm"
TIMEFRAME = "M5"           # "M5", "M15" or "H1"
DRY_RUN = True             # True = only print. Change to False to place DEMO orders.
RISK_PCT = 0.5             # percent of account equity risked per trade
MAX_LOTS = 1.0             # never trade more than this many lots
MAX_SPREAD_PIPS = 2.0      # skip new orders when the spread is wider than this
MAX_PENDING = 2            # at most this many waiting orders at once
MAX_DAILY_LOSS_PCT = 3.0   # stop placing orders after losing this much today
HISTORY_BARS = 2000        # candles re-read after every new candle
MAGIC = 20261024           # tags this bot's orders so it never touches yours
LOG_FILE = "bot_log.csv"
STRATEGY = dict(lz.CFG)    # strategy settings (same as the backtest)
# --------------------------------------------------------------------------


def log(event, detail=""):
    row = [datetime.now().isoformat(timespec="seconds"), event, detail]
    print(" | ".join(row))
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)


def plan_orders(o, h, l, c, cfg):
    """Setups that already have their tap and push, and are waiting for the liquidity sweep.
    Returns a list of dicts with side, entry (limit price), sl and tp."""
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).rolling(cfg["atr_period"]).mean().iloc[-1]
    plans = []
    for side, s in (("buy", 1), ("sell", -1)):
        # sells reuse the buy logic on a mirrored (negated) price series
        args = (o, h, l, c) if s == 1 else (-o, -l, -h, -c)
        _, zones, _ = lz.run_long(*args, cfg, return_state=True)
        for z in zones:
            if z["state"] != "pushed":
                continue
            entry = z["liq"]
            stop = z["top"] - cfg["stop_frac"] * (z["top"] - z["bot"])
            tp = z["hmax"]
            risk = entry - stop
            if risk <= 0 or np.isnan(atr):
                continue
            if (tp - entry) < cfg["min_push_atr"] * atr or (tp - entry) / risk < cfg["min_rr"]:
                continue
            plans.append(dict(side=side, entry=float(s * entry), sl=float(s * stop), tp=float(s * tp)))
    return plans


def check_plan(p, info, tick):
    """Broker-side sanity checks. Returns (ok, reason)."""
    pip = info.point * 10 if info.digits in (3, 5) else info.point
    spread = (tick.ask - tick.bid) / pip
    if spread > MAX_SPREAD_PIPS:
        return False, f"spread is {spread:.1f} pips (limit {MAX_SPREAD_PIPS})"
    min_dist = info.trade_stops_level * info.point
    if p["side"] == "buy" and p["entry"] > tick.ask - min_dist:
        return False, "price is already at or below the entry level"
    if p["side"] == "sell" and p["entry"] < tick.bid + min_dist:
        return False, "price is already at or above the entry level"
    if abs(p["entry"] - p["sl"]) < min_dist or abs(p["tp"] - p["entry"]) < min_dist:
        return False, "stop or target is closer than this broker allows"
    return True, ""


def lot_size(p, info, equity):
    """Lots so that a stop-out loses about RISK_PCT of equity (works for any account currency)."""
    typ = mt5.ORDER_TYPE_BUY if p["side"] == "buy" else mt5.ORDER_TYPE_SELL
    loss = mt5.order_calc_profit(typ, SYMBOL, 1.0, p["entry"], p["sl"])
    if not loss:
        return None
    lots = (equity * RISK_PCT / 100) / abs(loss)
    step = info.volume_step
    lots = math.floor(lots / step + 1e-9) * step
    lots = min(lots, MAX_LOTS, info.volume_max)
    return round(max(lots, info.volume_min), 2)


def todays_loss():
    """Profit/loss of this bot's closed trades today (negative = loss)."""
    now = datetime.now()
    deals = mt5.history_deals_get(datetime(now.year, now.month, now.day), now + timedelta(days=1)) or []
    return sum(d.profit + d.commission + d.swap for d in deals
               if d.magic == MAGIC and d.entry == mt5.DEAL_ENTRY_OUT)


def cancel(order, why):
    if DRY_RUN:
        log("WOULD CANCEL", f"order {order.ticket}: {why}")
        return
    res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket})
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    log("CANCEL" if ok else "CANCEL FAILED", f"order {order.ticket}: {why}")


def place(p, info, equity):
    lots = lot_size(p, info, equity)
    if lots is None:
        log("SKIP", "could not work out the lot size")
        return
    digits = info.digits
    desc = (f"{p['side'].upper()} LIMIT {lots} lots at {p['entry']:.{digits}f}, "
            f"SL {p['sl']:.{digits}f}, TP {p['tp']:.{digits}f}")
    if DRY_RUN:
        log("WOULD PLACE", desc)
        return
    res = mt5.order_send({
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": SYMBOL,
        "volume": lots,
        "type": mt5.ORDER_TYPE_BUY_LIMIT if p["side"] == "buy" else mt5.ORDER_TYPE_SELL_LIMIT,
        "price": round(p["entry"], digits),
        "sl": round(p["sl"], digits),
        "tp": round(p["tp"], digits),
        "magic": MAGIC,
        "comment": "zone-sweep demo",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    })
    if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
        log("PLACED", desc)
    else:
        log("PLACE FAILED", f"{desc} | {getattr(res, 'retcode', None)} {getattr(res, 'comment', '')}")


def cycle(rates):
    """Runs once after each closed candle."""
    info = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)
    acc = mt5.account_info()
    positions = [p for p in (mt5.positions_get(symbol=SYMBOL) or []) if p.magic == MAGIC]
    orders = [o for o in (mt5.orders_get(symbol=SYMBOL) or []) if o.magic == MAGIC]

    if positions:                                   # one trade at a time
        for od in orders:
            cancel(od, "a trade is already open")
        log("HOLD", f"{len(positions)} trade(s) open; stop and target are attached")
        return
    if todays_loss() <= -acc.balance * MAX_DAILY_LOSS_PCT / 100:
        for od in orders:
            cancel(od, "daily loss limit reached")
        log("DAILY LIMIT", f"lost more than {MAX_DAILY_LOSS_PCT}% today; no new orders")
        return

    o, h, l, c = (np.asarray(rates[k], dtype=float) for k in ("open", "high", "low", "close"))
    plans = plan_orders(o, h, l, c, STRATEGY)
    mid = (tick.bid + tick.ask) / 2
    plans.sort(key=lambda p: abs(p["entry"] - mid))  # nearest to the current price first
    desired = []
    for p in plans:
        ok, why = check_plan(p, info, tick)
        if ok:
            desired.append(p)
        else:
            log("SKIP", f"{p['side']} at {p['entry']:.{info.digits}f}: {why}")
    desired = desired[:MAX_PENDING]

    kept = set()                                    # keep matching orders, cancel the rest
    for od in orders:
        side = "buy" if od.type == mt5.ORDER_TYPE_BUY_LIMIT else "sell"
        hit = next((i for i, p in enumerate(desired) if i not in kept and p["side"] == side
                    and abs(od.price_open - p["entry"]) < info.point
                    and abs(od.sl - p["sl"]) < info.point and abs(od.tp - p["tp"]) < info.point), None)
        if hit is None:
            cancel(od, "setup changed or expired")
        else:
            kept.add(hit)
    for i, p in enumerate(desired):
        if i not in kept:
            place(p, info, acc.equity)
    if not desired:
        log("WAIT", "no setup waiting for a sweep")


def main():
    if not mt5.initialize():
        sys.exit(f"Could not connect to MetaTrader 5 ({mt5.last_error()}). "
                 "Open MT5, log in to a DEMO account, then try again.")
    acc = mt5.account_info()
    if acc is None:
        mt5.shutdown(); sys.exit("No account found. Log in to a demo account in MT5 first.")
    if acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        mt5.shutdown(); sys.exit("STOP: this is NOT a demo account. This bot only runs on demo accounts.")
    if not mt5.symbol_select(SYMBOL, True):
        mt5.shutdown(); sys.exit(f"Symbol {SYMBOL} was not found. Check the name in MT5's Market Watch.")
    tf = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}[TIMEFRAME]
    log("START", f"demo account {acc.login}, {SYMBOL} {TIMEFRAME}, dry_run={DRY_RUN}, risk {RISK_PCT}%")
    last = None
    try:
        while True:
            rates = mt5.copy_rates_from_pos(SYMBOL, tf, 1, HISTORY_BARS)   # closed candles only
            if rates is not None and len(rates) > 100 and rates[-1]["time"] != last:
                last = rates[-1]["time"]
                try:
                    cycle(rates)
                except Exception as e:                                       # keep running
                    log("ERROR", repr(e))
            time.sleep(5)
    except KeyboardInterrupt:
        log("STOP", "stopped by you; any open orders and trades stay on the account")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
