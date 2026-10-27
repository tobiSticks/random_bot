"""
DEMO-ACCOUNT bot for strategy 2 (protected-low zone + shallow tap + liquidity sweep)
on 28 forex pairs and gold, 1-hour candles, with the filters from robust_test.py:
  * only enter during the late-evening window (default 17:00-24:00, London clock)
  * planned reward:risk at least 1.5
  * the live spread must be at most 15% of the stop distance

How it works: after every closed candle, for every symbol, it re-reads recent prices, finds
setups that are waiting for the liquidity sweep, and keeps a LIMIT order at the sweep level with
the stop and target attached. When one fills, it cancels that symbol's other orders.

Safety:
  * It refuses to run unless the MT5 account is a DEMO account.
  * DRY_RUN = True (default) only prints what it WOULD do. Nothing is placed.
  * Every order has a stop and a target attached on the broker's side.
  * Limits: one trade per symbol, MAX_OPEN_TRADES in total, a daily loss limit.
  * The spread seen at every order is written to bot_log.csv, so you can compare it with
    the spreads the backtests assumed.

Needs: Windows, MetaTrader 5 open and logged in to a demo account, Algo Trading ON.
Setup: pip install MetaTrader5 pandas numpy tzdata
Run:   python demo_bot.py        (Ctrl+C to stop; open orders/trades stay on the account)
Keep liquidity_zone_backtest.py (the newest version) in the same folder.
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
SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF",
           "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
           "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
           "AUDJPY", "AUDCAD", "AUDCHF", "AUDNZD", "NZDJPY", "NZDCAD", "NZDCHF",
           "CADJPY", "CADCHF", "CHFJPY", "XAUUSD"]
SUFFIX = ""                # if your broker adds one to every name, e.g. "m" or ".a"
TIMEFRAME = "H1"           # the filters were tested on "H1" only
DRY_RUN = True             # True = only print. Change to False to place DEMO orders.
RISK_PCT = 0.5             # percent of account equity risked per trade
MAX_LOTS = 1.0             # never trade more than this many lots
MAX_OPEN_TRADES = 4        # open trades allowed at the same time, across all symbols
MAX_DAILY_LOSS_PCT = 3.0   # stop placing orders after losing this much today
HOURS = (17, 24)           # entry window, on the FILTER_TZ clock
FILTER_TZ = "Europe/London"  # the clock your downloaded data used (see rollover_stress.py)
MIN_RR = 1.5               # skip setups with planned reward:risk below this
MAX_COST_FRAC = 0.15       # skip setups when the spread is more than this share of the stop
SERVER_UTC_OFFSET = None   # your broker's clock vs UTC in hours (e.g. 2 or 3). None = detect it
HISTORY_BARS = 2000        # candles re-read after every new candle
MAGIC = 20261024           # tags this bot's orders so it never touches yours
LOG_FILE = "bot_log.csv"
STRATEGY = dict(lz.CFG)    # strategy settings (same as the backtest)
# --------------------------------------------------------------------------

BAR_SECONDS = {"M5": 300, "M15": 900, "H1": 3600}[TIMEFRAME]


def log(event, detail=""):
    row = [datetime.now().isoformat(timespec="seconds"), event, detail]
    print(" | ".join(row))
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)


def server_offset_hours(symbol):
    """Hours between the broker's clock and UTC. None if it cannot be told (market closed)."""
    if SERVER_UTC_OFFSET is not None:
        return SERVER_UTC_OFFSET
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    diff = (tick.time - time.time()) / 3600
    off = round(diff)
    if abs(diff - off) > 0.05 or abs(off) > 14:   # an old price: the market is probably closed
        return None
    return off


def next_bar_hour(last_bar_time, offset):
    """Clock hour (on FILTER_TZ) of the candle that has just started."""
    from zoneinfo import ZoneInfo
    utc_ts = int(last_bar_time) + BAR_SECONDS - offset * 3600
    return datetime.fromtimestamp(utc_ts, tz=ZoneInfo(FILTER_TZ)).hour


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
    """Filters and broker-side sanity checks. Returns (ok, reason)."""
    stop_dist = abs(p["entry"] - p["sl"])
    rr = abs(p["tp"] - p["entry"]) / stop_dist
    if rr < MIN_RR:
        return False, f"reward:risk {rr:.2f} is below {MIN_RR}"
    cost_frac = (tick.ask - tick.bid) / stop_dist
    if cost_frac > MAX_COST_FRAC:
        return False, f"spread is {cost_frac:.0%} of the stop (limit {MAX_COST_FRAC:.0%})"
    min_dist = info.trade_stops_level * info.point
    if p["side"] == "buy" and p["entry"] > tick.ask - min_dist:
        return False, "price is already at or below the entry level"
    if p["side"] == "sell" and p["entry"] < tick.bid + min_dist:
        return False, "price is already at or above the entry level"
    if stop_dist < min_dist or abs(p["tp"] - p["entry"]) < min_dist:
        return False, "stop or target is closer than this broker allows"
    return True, ""


def lot_size(p, symbol, info, equity):
    """Lots so that a stop-out loses about RISK_PCT of equity (works for any account currency)."""
    typ = mt5.ORDER_TYPE_BUY if p["side"] == "buy" else mt5.ORDER_TYPE_SELL
    loss = mt5.order_calc_profit(typ, symbol, 1.0, p["entry"], p["sl"])
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
        log("WOULD CANCEL", f"{order.symbol} order {order.ticket}: {why}")
        return
    res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket})
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    log("CANCEL" if ok else "CANCEL FAILED", f"{order.symbol} order {order.ticket}: {why}")


def place(p, symbol, info, tick, equity):
    lots = lot_size(p, symbol, info, equity)
    if lots is None:
        log("SKIP", f"{symbol}: could not work out the lot size")
        return
    d = info.digits
    spread = tick.ask - tick.bid
    desc = (f"{symbol} {p['side'].upper()} LIMIT {lots} lots at {p['entry']:.{d}f}, "
            f"SL {p['sl']:.{d}f}, TP {p['tp']:.{d}f}, spread {spread:.{d}f} "
            f"({spread / abs(p['entry'] - p['sl']):.0%} of the stop)")
    if DRY_RUN:
        log("WOULD PLACE", desc)
        return
    res = mt5.order_send({
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lots,
        "type": mt5.ORDER_TYPE_BUY_LIMIT if p["side"] == "buy" else mt5.ORDER_TYPE_SELL_LIMIT,
        "price": round(p["entry"], d),
        "sl": round(p["sl"], d),
        "tp": round(p["tp"], d),
        "magic": MAGIC,
        "comment": "zone-sweep demo",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    })
    if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
        log("PLACED", desc)
    else:
        log("PLACE FAILED", f"{desc} | {getattr(res, 'retcode', None)} {getattr(res, 'comment', '')}")


def cycle(symbol, rates):
    """Runs once per symbol after each closed candle."""
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    acc = mt5.account_info()
    if info is None or tick is None or acc is None:
        return
    mine = [p for p in (mt5.positions_get() or []) if p.magic == MAGIC]
    orders = [o for o in (mt5.orders_get(symbol=symbol) or []) if o.magic == MAGIC]

    if any(p.symbol == symbol for p in mine):          # one trade per symbol
        for od in orders:
            cancel(od, "a trade is already open on this symbol")
        log("HOLD", f"{symbol}: trade open, spread now {tick.ask - tick.bid:.{info.digits}f}")
        return
    if len(mine) >= MAX_OPEN_TRADES:
        for od in orders:
            cancel(od, "the open-trades limit is reached")
        return
    if todays_loss() <= -acc.balance * MAX_DAILY_LOSS_PCT / 100:
        for od in orders:
            cancel(od, "daily loss limit reached")
        log("DAILY LIMIT", f"lost more than {MAX_DAILY_LOSS_PCT}% today; no new orders")
        return
    offset = server_offset_hours(symbol)
    if offset is None:
        log("WAIT", f"{symbol}: market closed or prices are stale; no orders")
        return
    hour = next_bar_hour(rates[-1]["time"], offset)
    if not (HOURS[0] <= hour < HOURS[1]):              # no entries outside the tested window
        for od in orders:
            cancel(od, f"outside the entry hours ({hour:02d}:00 {FILTER_TZ})")
        return

    o, h, l, c = (np.asarray(rates[k], dtype=float) for k in ("open", "high", "low", "close"))
    plans = plan_orders(o, h, l, c, STRATEGY)
    mid = (tick.bid + tick.ask) / 2
    plans.sort(key=lambda p: abs(p["entry"] - mid))    # nearest to the current price first
    desired = []
    for p in plans:
        ok, why = check_plan(p, info, tick)
        if ok:
            desired.append(p)
        else:
            log("SKIP", f"{symbol} {p['side']} at {p['entry']:.{info.digits}f}: {why}")
    desired = desired[:1]

    kept = set()                                       # keep matching orders, cancel the rest
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
            place(p, symbol, info, tick, acc.equity)


def main():
    if not mt5.initialize():
        sys.exit(f"Could not connect to MetaTrader 5 ({mt5.last_error()}). "
                 "Open MT5, log in to a DEMO account, then try again.")
    acc = mt5.account_info()
    if acc is None:
        mt5.shutdown(); sys.exit("No account found. Log in to a demo account in MT5 first.")
    if acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        mt5.shutdown(); sys.exit("STOP: this is NOT a demo account. This bot only runs on demo accounts.")
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(FILTER_TZ)
    except Exception:
        mt5.shutdown(); sys.exit("Time zone data is missing. Run: pip install tzdata")
    active, missing = [], []
    for s in SYMBOLS:
        (active if mt5.symbol_select(s + SUFFIX, True) else missing).append(s + SUFFIX)
    if missing:
        log("NOTE", f"not available at your broker, skipped: {', '.join(missing)}")
    if not active:
        mt5.shutdown(); sys.exit("None of the symbols were found. Check SUFFIX and the names in Market Watch.")
    tf = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}[TIMEFRAME]
    log("START", f"demo account {acc.login}, {len(active)} symbols, {TIMEFRAME}, "
                 f"hours {HOURS[0]}-{HOURS[1]} {FILTER_TZ}, dry_run={DRY_RUN}, risk {RISK_PCT}%")
    last = {}
    try:
        while True:
            checked = 0
            for symbol in active:
                rates = mt5.copy_rates_from_pos(symbol, tf, 1, HISTORY_BARS)   # closed candles only
                if rates is not None and len(rates) > 100 and rates[-1]["time"] != last.get(symbol):
                    last[symbol] = rates[-1]["time"]
                    checked += 1
                    try:
                        cycle(symbol, rates)
                    except Exception as e:                                       # keep running
                        log("ERROR", f"{symbol}: {e!r}")
            if checked:
                log("CHECKED", f"{checked} symbols after the latest candle")
            time.sleep(15)
    except KeyboardInterrupt:
        log("STOP", "stopped by you; any open orders and trades stay on the account")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
