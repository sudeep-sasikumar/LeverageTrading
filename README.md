# LTS — Leverage Trading Signals

Scans Weex / Binance USDT perpetuals every 5 minutes, finds coins that are mid-move with
momentum still intact, and sends Telegram signals with a volatility-scaled stop, a
target worth about +10% on margin, and a leverage (≤10x) derived from the stop distance.
Every candidate — sent or not — is logged and later labelled with its real outcome, and
a model is retrained nightly on that growing dataset.

**It only sends signals. It never places orders and uses no exchange API keys.**

Design and research notes: [docs/DESIGN.md](docs/DESIGN.md).

## What to expect

- **It starts in paper mode.** On the first 30 days of real data (13 Oct – 12 Nov 2026)
  unfiltered momentum candidates lost about 0.12R each after fees, and the first model's
  out-of-sample result (+0.33R per trade, t = 1.4 over 41 trades) was positive but not
  statistically convincing. So signals are sent marked **PAPER** and tracked, and
  real-money grades (A/B) only switch on after a model passes every validation gate,
  including 20 closed forward signals with a positive average. Details:
  [docs/DESIGN.md §11](docs/DESIGN.md).
- This is a statistical edge-finder, not a guarantee. A 2:1 trade needs roughly a 37–40%
  hit rate after fees to be profitable; many signals will lose.
- Capital preservation outranks signal frequency. On days where nothing clears the bar
  the bot sends a "no-trade day" note instead of a weak signal.
- Leverage is derived from the stop, usually 3–6x; 10x is a cap. Stops are 1–2.5% because
  tighter stops lose most of their edge to fees.

## Quick start (local)

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
cp .env.example .env                               # add Telegram token + chat id (local runs only)
python -m lts backfill --days 30                   # ~30 min, resumable
python -m lts replay                               # rebuild labelled candidates from history
python -m lts research                             # base rates + stop/target grid
python -m lts train                                # walk-forward validation, promote if it passes
python -m lts scan                                 # one live cycle, prints candidates, sends nothing
python -m lts telegram-test
python -m lts run                                  # the live service
```

## Running on the VPS (Hostinger Docker Manager)

Same pattern as CryptoLaunchScanner. Pick a VPS region Binance serves (not the US).

1. Every push to `main` runs the tests, builds the image and publishes
   `ghcr.io/sudeep-sasikumar/leveragetrading:latest` (GitHub Actions).
2. On GitHub: your profile → Packages → `leveragetrading` → Package settings → make it
   public, so the VPS can pull it without a login. Needed once.
3. Hostinger → VPS → Docker Manager → Compose → Compose from URL:
   `https://raw.githubusercontent.com/sudeep-sasikumar/LeverageTrading/main/docker-compose.yaml`,
   project name `leverage-trading`.
4. Enter the environment variables, then Deploy:

   | Variable | Needed | What |
   |---|---|---|
   | `TELEGRAM_BOT_TOKEN` | yes | from @BotFather. For `/status` etc. use a bot no other app polls |
   | `TELEGRAM_CHAT_ID` | yes | your chat id; a group id starts with `-100` |
   | `DASHBOARD_PASSWORD` | for the dashboard | a long password of your choice; without it the dashboard stays off |
   | `ACCOUNT_SIZE_USDT` | optional | quotes each signal's margin in USDT |
   | `RISK_PER_TRADE` | optional | account fraction lost at a stop (default `0.01`) |
   | `MAX_SIGNALS_PER_DAY` | optional | default `3` |
   | `BACKFILL_DAYS` | optional | history downloaded on first start (default `30`) |

**Updating the compose file.** Hostinger keeps the copy of `docker-compose.yaml` from when the project was
created; a redeploy only pulls the newest image. When the compose file changes (it did on 12 Nov, when the
dashboard was added), open the project in Docker Manager, replace its compose content with the current
file from GitHub (or delete the project and re-create it from the URL - the `lts_data` volume and its data
survive as long as you don't delete volumes), keep the environment variables, and deploy. On every start
the bot sends a Telegram message with the dashboard link, or with exactly what is missing.

The first start on an empty volume downloads 30 days of history, rebuilds the labelled
candidates and trains the first model (about 40 minutes), then sends "LTS started" on
Telegram. Data (database, models, logs) lives in the `lts_data` volume and survives
redeploys; a redeploy pulls the newest image and resumes within a minute.

5. Dashboard: `https://srv1479428.hstgr.cloud/lts/` - it shares the hostname and certificate
   that Ignition & Coil and Launch Radar already use (set `DOMAIN` to change the hostname). It is
   read-only: signals with their results, every scanned candidate and why it was skipped, the
   model's validation numbers, and a health panel. It is up during the first-start setup too.

No host ports are opened; the dashboard is reached only through Traefik. On this PC:
`python -m lts dashboard`, then http://127.0.0.1:8020 (no login on localhost).

## Two setups

- **Momentum** (the original): a sharp move with volume that is still intact. Model-ranked; paper until validated.
- **Grind** (steady trend, added 16 Nov): a coin that has moved steadily one way for hours without one candle doing
  the work. It has a wider stop (2.5 × hourly ATR), a 1:1 target, holds up to 24h, and is **paper-only**: over the
  last month it broke even after fees overall, with short grinds positive and long grinds negative
  ([docs/DESIGN.md §12](docs/DESIGN.md)).

**Grind lab** (added 17 Nov): every night LTS re-studies ~4 months of steady grinds and what the ones that kept going
had in common versus lookalikes that fizzled. It looks at multi-timeframe support/resistance, room to run, volume and flow,
funding, open interest, sector, listing age and exchange news. Every hour it scores coins grinding right now against those
traits and paper-trades the best. Everything is on the dashboard's **Grinders** tab; findings are in
[docs/DESIGN.md §13](docs/DESIGN.md).

Every day the bot also reviews the last 24h's steady movers: which coins ground 10%+ one way, what their volume,
open interest and positioning looked like, and whether LTS caught them in time. This is in the daily Telegram report,
on the dashboard's *Daily movers* tab, and via `python -m lts movers`.

## Telegram

Signals, exits (target / stop / time stop), a daily report, the nightly learning report
and health alerts. Commands: `/status`, `/stats`, `/open`, `/pause`, `/resume`.

A signal looks like:

```
🟢 LONG XYZUSDT · Grade A
Entry  0.4105  (limit order preferred; valid 10 min)
Stop   0.4064  (−1.00%)
Target 0.4187  (+2.00%)
Leverage 6x isolated
At target ≈ +11.2% on margin · at stop ≈ −7.1% (after fees)
Size: margin ≤ 14% of account (risks 1.0%)
Time stop: close at market by 21:35 UTC if neither level is hit
Skip if price is already beyond 0.4126
P(target first) 46% · EV +0.28R
```

`R` is the stop distance: +2R is a full target, −1R a full stop (before fees).

## How it improves

| Step | When | What |
|---|---|---|
| Log | every 5 min | each candidate with ~100 features |
| Label | hourly | true outcome (target / stop / time), MFE/MAE, alternative stop/target grid |
| Retrain | nightly 00:40 UTC | purged walk-forward validation, probability calibration, threshold selection |
| Promote | nightly | a model goes live only with significant out-of-sample results, a positive holdout and a positive 20-signal forward record |
| Guard | continuous | daily stop, loss-streak stop, safe mode after a bad week, automatic demotion to paper if the live record turns negative |

## Layout

```
lts/exchanges.py   Binance + Weex public REST adapters, rate limiting
lts/collector.py   5-minute and hourly data collection
lts/features.py    feature engine (shared by live and replay; no look-ahead, tested)
lts/candidates.py  momentum-continuation triggers
lts/labels.py      triple-barrier labels, stop/target/leverage arithmetic
lts/learner.py     walk-forward training, calibration, promotion
lts/signals.py     thresholds, risk gates, Telegram signal text
lts/tracker.py     open-signal tracking and candidate labelling
lts/service.py     the live loops
lts/replay.py      rebuild the labelled dataset from stored history
config.yaml        tunable parameters
```

Run the tests with `python -m pytest tests -q`.

## Disclaimer

Research software, not financial advice. Leveraged trading can lose your entire margin.
