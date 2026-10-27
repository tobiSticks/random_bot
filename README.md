# Liquidity Zone Trading Bot

Automated Forex + Gold trading bot for MetaTrader 5 demo accounts.

**Strategy:** Protected low zone + inducement sweep (liquidity sweep entry). Runs on 29 symbols (28 Forex pairs + XAUUSD) on H1 candles.

---

## How the Strategy Works

1. Price breaks structure above a swing high — the lowest low behind the break becomes the "protected low"
2. A zone is drawn from the broken high (top) down to the protected low (bottom)
3. Price returns for a **shallow tap** into the top of the zone, then pushes higher — that tap low is where stop losses sit (liquidity)
4. Bot enters when price sweeps back down through the tap low (inducement)
   - Entry: at the swept level (limit order)
   - Stop: inside the zone for breathing room
   - Target: high of the push

---

## Requirements

- **Windows only** — MetaTrader 5 has no Linux/Mac client
- MetaTrader 5 installed and logged into a **demo account**
- Algo Trading enabled in MT5 (wrench icon → Options → Expert Advisors → Allow)
- Python 3.10+

---

## Setup

### 1. Install dependencies

```bash
pip install MetaTrader5 pandas numpy tzdata
```

### 2. Configure the bot

Open `demo_bot.py` and edit the settings block near the top:

| Setting | Default | What it does |
|---|---|---|
| `DRY_RUN` | `False` | Set to `True` first to test — prints actions without placing orders |
| `SUFFIX` | `""` | Broker's symbol suffix, e.g. `"m"` or `".a"` — check your MT5 Market Watch |
| `RISK_PCT` | `0.5` | % of account equity risked per trade |
| `MAX_LOTS` | `1.0` | Hard cap on lot size per trade |
| `MAX_OPEN_TRADES` | `4` | Max simultaneous open trades across all symbols |
| `MAX_DAILY_LOSS_PCT` | `3.0` | Bot stops placing orders after losing this % in a day |
| `HOURS` | `(0, 24)` | Entry window in London time — narrow this to filter trades |
| `MIN_RR` | `1.0` | Skip setups with reward:risk below this |
| `MAX_COST_FRAC` | `0.5` | Skip if spread > this fraction of the stop distance |

### 3. (Optional) Set up Telegram alerts

Get phone notifications for orders placed, trades opened/closed, errors, and daily heartbeats.

```bash
# Step 1: Create a bot via @BotFather in Telegram, get your TOKEN
# Step 2: Open your new bot in Telegram, press START, send it any message
# Step 3: Run this to find your chat ID
python telegram_setup.py YOUR_TOKEN_HERE
```

Then paste the token and chat ID into `demo_bot.py`:
```python
TELEGRAM_TOKEN = "1234567890:AAH..."
TELEGRAM_CHAT_ID = "987654321"
```

### 4. Run the bot

```bash
python demo_bot.py
```

Stop with `Ctrl+C`. Open orders/trades stay on the account after stopping.

---

## Safety Features

- **Refuses to run on a live account** — exits immediately if MT5 account is not a demo
- **`DRY_RUN` mode** — prints what it would do without touching the broker
- **Every order has SL + TP attached** on the broker's side — not managed in Python
- **One trade per symbol** at a time
- **Daily loss limit** — stops placing orders after `MAX_DAILY_LOSS_PCT` is lost
- **Spread check** — skips setups where spread eats too much of the stop
- **Magic number** (`MAGIC = 20261024`) — bot only touches its own orders, never yours

---

## Files

| File | Purpose |
|---|---|
| `demo_bot.py` | Main bot — run this |
| `liquidity_zone_backtest.py` | Strategy logic (shared by bot + backtests) |
| `telegram_setup.py` | Helper to find your Telegram chat ID |
| `bot_log.csv` | Live log of every event (spreads, orders, errors) |
| `bos_zone_backtest.py` | Backtest for BOS + zone variant |
| `ifvg_backtest.py` | Backtest for inverse FVG variant |
| `multi_backtest.py` | Runs backtests across multiple symbols |
| `robust_test.py` | Stress-tests strategy parameters |
| `stress_test.py` | Monte Carlo / robustness testing |
| `get_multi_data.py` | Downloads historical price data from MT5 |
| `clock_check.py` | Verifies broker clock offset |
| `prices.csv` | Sample price data (EURUSD) |
| `XAUUSD_M5_Data.csv` | Sample XAUUSD M5 data |

---

## Running Backtests

```bash
# Backtest liquidity zone strategy on a CSV file
python liquidity_zone_backtest.py prices.csv

# Download fresh data from MT5 first
python get_multi_data.py

# Run multi-symbol backtest
python multi_backtest.py

# Robustness test (parameter sensitivity)
python robust_test.py
```

CSV files need columns: `time` (or `date`), `open`, `high`, `low`, `close`

---

## Troubleshooting

**Bot exits immediately:** MT5 is not open, or account is not a demo account.

**No symbols found / "symbol not available":** Set `SUFFIX` to match your broker's naming, e.g. `SUFFIX = "m"` for `EURUSDm`.

**Orders not filling:** Check that Algo Trading is enabled in MT5. Also check `DRY_RUN` is `False`.

**Telegram not working:** Run `telegram_setup.py` again — make sure you sent the bot a message in Telegram before running it.

**Spread too high warnings:** Normal during off-hours. Bot skips these setups automatically. Narrow `HOURS` to active market sessions.

**Bot placed trades on a live account:** Not possible — bot exits on startup if account is not demo.
