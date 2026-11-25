# LTS — Leverage Trading Signals

Scans Weex / Binance USDT perpetuals every 5 minutes for coins that have been **grinding steadily one way all
day**, and sends PAPER Telegram signals with the trader's plans: **5x, +2% target, −1% stop** (tracked), plus
10x +1% / −0.5%, 3x +3.33% / −1% and 2x +5% / −0.75%, entered at market, closed after 24h. Every detected
grind — sent or not — is recorded with its factors and labelled with what every plan did.

**It only sends signals. It never places orders and uses no exchange API keys.**

Design and research notes: [docs/DESIGN.md](docs/DESIGN.md) (read §25–28 first). Pre-registered studies:
[docs/HYPOTHESES.md](docs/HYPOTHESES.md); their outputs: [docs/results/](docs/results/).

## What to expect

- **Everything is PAPER.** No rule tested so far has shown an edge after fees: on the past year the all-day
  steady rule (the one that sends) won 38% at 5x +2% / −1% against about 41% needed, a small loss per trade
  (DESIGN.md §25, reproduced in §28). Real-money grades need the rule to pass the pre-registered bar AND 30
  closed live paper trades under that exact rule (`lts/gate.py`).
- **Signals are only sent when the rule's tested net after costs is positive** (`grindsignal.require_positive_ev`,
  on by default since 25 Nov 2026). With the current record that means **no grind signals are sent**; every
  grind keeps being recorded and the dashboard shows the expected net and its 90% interval. Set
  `require_positive_ev: false` under `grindsignal:` in config.yaml to receive the paper calls anyway.
- **Portfolio limits apply to grind signals again** (since 25 Nov 2026): stop for the day at −2R, after two
  losses in a row, and at most 2 open-or-today calls in one direction while the market moves that way.
  Switches: `grindsignal.daily_stop`, `streak_stop`, `max_same_direction` (0 = off).
- The steady rule is checked **as the year test did** (`grindsignal.steady_eval: hour_close_first`): at
  clock-hour closes, at the first early-grind detection per coin and side in 12h, with no second call inside
  that window. `rolling_5m` is the old every-5-minutes behaviour.
- The momentum setup (the original design) is logged and labelled for learning only (`signal.momentum_send`).
  Its model retrains weekly (`learn.retrain_schedule`); the grind-lab study also runs weekly
  (`grindlab.study_schedule`), while its data refresh runs nightly.

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

## What runs

| Track | Sends? | What |
|---|---|---|
| **Grind signals** (`lts/grindsignal.py`, `lts/gate.py`) | PAPER, only if the rule's expected net > 0 | all-day steady grinds (DESIGN.md §25, §28) with the trader's plans |
| Grind recording (`grind_live`) | no | every early grind every 5 minutes: factors, levels, liquidations, book, every plan's outcome, shadow early exits, trade anatomy |
| Momentum (`lts/candidates.py`, `lts/learner.py`) | no (logged) | the original sharp-move setup; model retrained weekly |
| Grind lab (`lts/grindlab.py`) | no | what successful grinders share; study weekly, data nightly |
| Daily movers review (`lts/movers.py`) | in the daily report | the day's steady movers and whether LTS caught them |
| Liquidations + order books (`lts/microstructure.py`) | no | collected since 16–17 Nov 2026 for study B4 (from about 16 Jan 2027) |

Every coin name on the dashboard and in Telegram links to its Weex chart. The trader's own trades are logged on
the dashboard's *My trades* tab (`lts/journal.py`).

## Telegram

Signals, exits (target / stop / time stop), a daily report, the nightly learning report
and health alerts. Commands: `/status`, `/stats`, `/open`, `/pause`, `/resume`.

A grind signal shows each plan's target and stop prices, the result on margin at target and at stop after fees,
the rule's year record per plan, the expected net per trade with its 90% interval, daily/weekly/monthly levels
between entry and target or stop, and the steadiness numbers that qualified it. `R` is the stop distance.

## How it improves

| Step | When | What |
|---|---|---|
| Record | every 5 min | every early grind with ~150 factors (and every momentum candidate) |
| Label | hourly | every plan, a target/stop grid, price for 24h after each exit, shadow early exits, trade anatomy |
| Study | pre-registered | `docs/HYPOTHESES.md` first, then `research/*.py` → `docs/results/`, judged by `lts/validate.py` |
| Gate | continuous | a rule sends only with positive tested expected net; goes live only after the bar + 30 live paper trades |
| Guard | continuous | daily −2R stop, two-loss stop, same-direction limit, BTC-shock and crowded-funding vetoes |

## Layout

```
lts/exchanges.py   Binance + Weex public REST adapters, rate limiting
lts/collector.py   5-minute and hourly data collection
lts/features.py    feature engine (shared by live and replay; no look-ahead, tested)
lts/candidates.py  momentum-continuation triggers
lts/labels.py      triple-barrier labels, stop/target/leverage arithmetic
lts/learner.py     walk-forward training, calibration, promotion
lts/signals.py     thresholds, risk gates, Telegram signal text
lts/tracker.py     open-signal tracking (1-minute candles, gaps paged and recorded) and candidate labelling
lts/grindlab.py    early-grind detection (5-minute scanner) and the grind-lab study
lts/grindsignal.py grind signals: the steady rule, plans, labelling, recording
lts/gate.py        signal policy: expected net and the go-live gate
lts/fees.py        trading costs (from config.trade only)
lts/validate.py    shared study validation (day-clustered t, Holm, locked confirmation)
research/          the scripts behind every published number (docs/results/)
lts/service.py     the live loops
lts/replay.py      rebuild the labelled dataset from stored history
config.yaml        tunable parameters
```

Run the tests with `python -m pytest -q`.

## Disclaimer

Research software, not financial advice. Leveraged trading can lose your entire margin.
