# Leverage Trading Signal System — Design

Status: v1, implemented. Section 11 records what 30 days of real data showed — read it
before trusting any signal.

## 1. Goal and honest framing

Scan Weex / Binance USDT perpetuals continuously, detect coins that are **mid-move with
sustained momentum**, and send Telegram signals where a tight stop and a target worth
about **+10% on margin at up to 10x leverage** is statistically justified.

What the four aims translate to in engineering terms:

| Aim | Design consequence |
|---|---|
| ≥1 signal/day worth 10% on margin | Target = 1–3% underlying move; leverage is *derived* from the stop distance, never fixed at 10x |
| Low drawdown, tight stop, capital preservation | Volatility-scaled stop, fixed-fractional risk (1% of account per trade), daily/weekly circuit breakers, hard vetoes |
| Pattern recognition across regimes | ~90 features per candidate (price, volume, flow, OI, funding, positioning, market regime) + gradient-boosted classifier |
| Improves with data | Every candidate (sent or not) is logged with features and later labelled with its true outcome; nightly walk-forward retrain with champion/challenger promotion |

Two things the system does **not** promise, by design:

1. **A signal every single day.** When aim 1 and aim 2 conflict, aim 2 wins ("non
   negotiable"). If no candidate has positive expected value, the bot sends a
   "no-trade day" note instead of a weak signal.
2. **That every signal wins.** A 2:1 reward/risk trade needs roughly a 40% hit rate
   after fees to make money. The realistic objective is a positive expectancy with
   controlled drawdown, not a high win rate.

The system only *signals*. It never places orders and holds no exchange API keys.

## 2. Research summary → design decisions

| Finding | Source type | Decision |
|---|---|---|
| Liquid coins show short-term momentum; small illiquid coins show *reversal* | Academic (Fičura 2023; "Up or down?" IRFA 2021) | Liquidity floor on Binance 24h volume; liquidity is a model feature |
| Price up + OI up + sustained taker buying = new leverage entering (continuation); price up + OI down = short-covering bounce (fragile) | Practitioner consensus; OI was the strongest stop-survival predictor in a 94k-observation study | OI change over 30m/1h/4h/24h and taker imbalance are core features |
| Funding rate has weak overall predictive power, but extremes mark crowding | Mixed empirical | Funding is a feature and an extreme-value veto, not a trigger |
| Order-flow imbalance IC is small (0.01–0.05); impulsive flow-driven bars tend to mean-revert at very short horizons | arXiv 2026 preprints | Do not chase a single impulsive bar: over-extension and wick features, entry validity window |
| Volatility spikes at funding times (00/08/16 UTC) and top of hour | Hansen, Kim & Kimbrough | Time-of-day and minutes-to-funding features |
| Base rate of TP-before-SL at 2:1 is ~30–34% (≈ breakeven) | Industry study, 656 coins | Edge must come from *selection*; measure everything against the unfiltered base rate |
| Fixed-horizon labels and plain k-fold CV leak and overfit | López de Prado (AFML) | Triple-barrier labels, meta-labelling, purged walk-forward validation, untouched holdout |
| Weex API v3 mirrors Binance's schema; 515 crypto perps exist on both; Binance has 10–50× the volume | Live probing (12 Nov 2026) | Binance data drives analytics; Weex data confirms tradability and sets the actual price levels |

Measured on the day of design: 16 of 522 Binance USDT perps moved more than 10% in 24h
and 6 moved more than 20% — consistent with the observation that motivated the project.

## 3. Architecture

```
            ┌──────────────── every 5 min (bar close + 4s) ────────────────┐
 Binance ──►│ Collector: klines(5m, taker vol, trades), OI, premium/funding │
 Weex    ──►│            Weex tickers; hourly: long/short ratios, universe  │
            └──────────────────────────┬────────────────────────────────────┘
                                       ▼
                               SQLite (WAL) store
                                       ▼
   Feature engine ─► Candidate detector ─► Scorer (rules + ML model) ─► Risk manager
   (same code live       (broad momentum       (calibrated P(TP first))    (stops, leverage,
    and in replay)        triggers)                                         vetoes, breakers)
                                       │                                        │
                 every candidate logged with features                    Telegram signal
                                       ▼                                        ▼
                     Outcome tracker (1-min loop): TP / SL / time exit, MFE / MAE
                                       ▼
                 Labeller: triple-barrier outcome for *every* candidate
                                       ▼
        Nightly learner: purged walk-forward → challenger vs champion → promote
```

Single Python process (asyncio), one SQLite file, no external services except the two
public exchange APIs and Telegram. Runs comfortably on a 1 vCPU / 4 GB VPS.

### Why these choices

- **Polling, not websockets.** Signals are decided on closed 5-minute bars and held for
  hours; REST polling costs ~1,050 of Binance's 12,000 weight per 5 minutes and is far
  easier to keep correct after a restart or network drop.
- **SQLite.** ~150k kline rows/day; 5-minute data is pruned after 120 days, while
  candidates, labels and signals (the learning data) are kept forever.
- **One feature code path.** The replay (backtest) and live scan call the same function
  on the same tables, and a unit test asserts that features at time *t* are identical
  whether or not future bars exist. This is the main defence against look-ahead bias.

## 4. Universe

- Crypto perps listed on **both** Weex and Binance (≈515), analysed with Binance data.
- Weex-only crypto perps are included only above a Weex volume floor (they are mostly
  below $1M/day, where tight stops get slipped). Stocks / indices / metals perps are
  excluded by default (`underlying_types: [COIN]`).
- Signal eligibility: Binance 24h quote volume ≥ $10M, listed ≥ 3 days, Weex 24h volume
  ≥ $300k, Weex spread small relative to the stop (checked at signal time).

## 5. Data captured

Per symbol, every 5 minutes unless noted:

| Table | Fields | Why |
|---|---|---|
| `klines` | OHLC, base/quote volume, trade count, taker-buy volume | Price, volume surge, taker flow (CVD proxy), average trade size |
| `oi` | Open interest (contracts and USD) | New leverage vs covering |
| `premium` | Mark, index, current funding, next funding time | Basis and crowding |
| `funding` | Settled funding history | Crowding baseline |
| `lsr` (hourly) | Top-trader position L/S, global account L/S | Smart vs crowd positioning |
| `weex_ticker` | Weex last, 24h volume, mark | Tradability, cross-venue deviation |
| `book` (at candidate time) | Weex spread, top-15 depth imbalance | Execution quality; live-only features |
| `candidates` | Trigger, full feature JSON, rule score, model probability, labels | **The learning dataset** |
| `signals` | Levels, leverage, lifecycle, outcome, MFE/MAE | Live track record |
| `models` | Version, metrics, threshold, status | Champion/challenger registry |

## 6. Features (~90, direction-aligned)

All directional features are multiplied by the trade side, so one model learns
"with-the-move" patterns from longs and shorts alike.

- **Momentum**: returns 5m…3d; volatility-normalised z-scores (1h/4h/24h vs 7-day vol).
- **Trend quality**: Kaufman efficiency ratio (1h/4h), share of up-bars, EMA20/50/200
  distance in ATRs, EMA stack, EMA20 slope.
- **Volume**: relative volume 5m/1h/24h vs 7-day baseline, volume acceleration, log
  dollar volume, trade-count surge, average trade size ratio.
- **Flow**: taker buy imbalance over 15m/1h/4h/24h.
- **Leverage**: OI change 30m/1h/4h/24h, OI-to-volume, funding level and deviation from
  3-day mean, basis (live), minutes to funding.
- **Positioning**: top-trader and global long/short ratio, 4h and 24h deltas.
- **Location / exhaustion**: position in 1h/4h/24h/7d range, distance to the extreme in
  ATRs, bars since the extreme, run from the opposite extreme, distance from 24h/4h
  VWAP, RSI (5m and ~1h), Bollinger z, adverse/favourable wick, bar range expansion.
- **Volatility regime**: ATR%, realised vol 1h/24h/7d and their ratios.
- **Market regime**: BTC returns 15m/1h/4h/24h and trend, breadth (share of coins up
  1h/24h), median alt return, cross-sectional dispersion, share of >10% movers,
  relative strength vs BTC and vs the median alt, cross-sectional ranks, 24h
  correlation to BTC.
- **Calendar / meta**: hour (sin/cos), day of week, weekend, listing age.
- **Live-only (accumulate over time)**: Weex spread, Weex depth imbalance, Weex volume,
  Weex–Binance price deviation, basis. These are NaN in backfilled rows; the model
  starts using them once enough live rows exist — one concrete way the system gets
  better with age.

## 7. Candidate detection ("mid-breakout, momentum intact")

Evaluated on every closed 5m bar, for each side:

- **Momentum**: 1h z-score ≥ 2.0, or 4h z-score ≥ 2.0, or 24h move ≥ 10% with a
  with-trend last hour.
- **Participation**: 1h relative volume ≥ 1.3 (or 24h relative volume ≥ 2).
- **Still intact**: close in the favourable half of the last hour's range and on the
  right side of EMA50.
- **Tradable**: liquidity floor, enough history, and 5m ATR between 0.5% and 1.25% — low
  enough for a stop of at most 2.5%, high enough that the stop is at least 1% (below
  that, fees and slippage are more than a quarter of the risk; see section 11).
- 30-minute cooldown per symbol and side.

The trigger is deliberately broad (meta-labelling): it produces many candidates per
day, and the model's job is to decide which few are worth sending.

## 8. Trade construction and risk

```
stop%      = clamp(2 × ATR%(5m,14), 1.0%, 2.5%)            (skip if above 2.5%)
target%    = rr × stop%
leverage   = min(10, ceil(10% / (target% − fees%)))        (integer, isolated)
margin     = account × 1% / (leverage × (stop% + fees%))   (1% of account at risk)
time stop  = exit at market after the horizon if neither level is hit
```

Fees are modelled from Weex VIP0 (taker 0.08%, maker 0.02%) plus slippage, and are part
of every label. With stops of 1–2.5% the derived leverage is typically 3–6x; 10x is a
cap, not a default. "Tight stop" is enforced where it matters — 1% of the account at
risk per trade — rather than as a tiny price distance that fees would dominate.
Signals recommend a **limit entry** at the signal price: it roughly halves fee drag.

Hard vetoes (never overridden by the model): Weex spread too wide for the stop, BTC
shock against the trade, extreme same-side funding, illiquidity, symbol in cooldown
after a stop-out, entry no longer valid after 10 minutes.

Circuit breakers: max 2 open signals, max 3 per day, stop for the day at −2R or two
consecutive losses, "safe mode" (higher threshold) when the rolling 7-day result is
below −4R.

Signal frequency: grade **A** signals fire whenever the model's ranking score ≥ the
learned threshold. If the day has no signal by 14:00 UTC, the best candidate with
positive expected value may be sent as grade **B**. Below positive EV, nothing is sent.

**Paper mode.** Grades A and B exist only while a model that passed validation
(section 9) is the champion. Without one, the same pipeline still runs end to end, but
every signal is graded **P** and the message says plainly that it is a paper signal not
to be traded with real money. Paper signals are tracked to exit like real ones, so a
forward track record accumulates either way.

## 9. Learning loop

1. **Log** every candidate with its full feature vector (sent or not — avoids
   selection bias and yields hundreds of labelled examples per day instead of one).
2. **Label** after the horizon with the triple-barrier outcome net of fees, plus
   MFE/MAE and outcomes for a small grid of stop/target/horizon alternatives.
3. **Retrain nightly**: histogram gradient boosting (handles missing features
   natively), shallow trees, strong regularisation, time-decay and uniqueness sample
   weights, live rows weighted above replayed rows. Two guards against learning "what
   kind of day it was" instead of "which coin is better":
   - the default objective is **cross-sectional** — the target is the outcome minus the
     average outcome of all candidates in the same 4-hour block;
   - **market-wide and calendar features are withheld** from the model until the
     labelled history spans 120 days, because they have one independent observation
     per day, not per candidate. They are still recorded from day one.
4. **Validate** with expanding walk-forward folds, purged by the label horizon, and
   score on what matters: a simulation of the actual signal policy (cooldowns, max
   open, daily stop) → expectancy in R, win rate, drawdown, signals/day.
5. **Promote** a challenger to real-money signals only if **all** of these hold:
   - same-moment selection skill ≥ 0.03;
   - out-of-sample policy expectancy positive *and significant* (t ≥ 2 over ≥ 40
     trades — the threshold is the best of a dozen tried, so a small positive average
     is not evidence);
   - positive expectancy on the untouched most-recent holdout;
   - a **forward record** of at least 20 closed paper signals with positive average R;
   - not worse than the champion on candidates the champion has never seen.
   A model that fails any gate becomes the *shadow* model and drives paper signals only.
6. **Threshold**: chosen on the model's raw ranking score to maximise expectancy subject
   to the signal-frequency target; probabilities are isotonic-calibrated for display
   and expected-value checks.
7. **Guard**: a champion whose recent live record falls below −0.10R per signal (over ≥
   20 signals) is demoted automatically and the bot returns to paper mode. Safe mode
   (stricter threshold) applies after a −4R week.
8. **Report**: daily Telegram summary and nightly learning report; `python -m lts
   research` prints base rates and the stop/target grid so parameter changes are a
   reviewed decision, not silent drift.

Cold start: `backfill` pulls 30 days of history (the limit for Binance OI / positioning
data), `replay` regenerates every candidate that would have fired, and `train` builds
the first model — so the bot does not start blind.

## 10. Operations

- `python -m lts backfill | replay | train | run | report | status`
- Telegram: signals, exits, daily summary, retrain report, health alerts;
  commands `/status /stats /open /pause /resume`.
- Deployment: GitHub Actions builds a Docker image to GHCR on every push; Hostinger
  Docker Manager deploys `docker-compose.yaml` from its raw URL. The Telegram token and
  chat id are environment variables set in Docker Manager. Data lives in a named volume.
  The VPS must be in a region Binance serves (not the US).
- Secrets never enter the repo or the image; `.env` is git-ignored and docker-ignored.

## 11. What 30 days of real data showed (13 Oct – 12 Nov 2026)

Dataset: 4.5M five-minute bars for 519 Binance USDT perps, with open interest, funding
and long/short ratios. Replaying the trigger produced **9,790 labelled candidates**
(355/day, 319 symbols). Reproduce with `python -m lts replay && python -m lts research`.

### Findings

1. **Unfiltered, momentum continuation had no edge in this window.** 31.2% of
   candidates reached the 2:1 target before the stop, against a 33.3% break-even before
   costs. Gross of fees the candidates were a coin flip; net they averaged **−0.12R**.
2. **Fees are the dominant term.** The hit rate was ~31% for every stop width, but net
   results ran from −0.23R for stops under 0.95% to −0.02R for stops of 2–2.5%. A 0.6%
   stop pays about 40% of its risk in taker fees and slippage. → minimum stop raised to
   1%, low-volatility candidates dropped.
3. **A limit entry roughly halves the drag**: −0.066R per signal vs −0.122R at market,
   with ~99% of limit orders at the signal price filling within 15 minutes in the
   replay (real fills will be somewhat lower). → signals recommend a limit entry.
4. **No stop/target/horizon combination rescues it.** All 18 grid cells sat between
   −0.11R and −0.21R, so the 2×ATR / 2:1 / 8h default was kept rather than tuned.
5. **Outcomes depend mostly on what kind of day it is.** On one sell-off day 58% of
   candidates hit target; on pump days about 21–25% did. Shorts (33.4%, −0.035R) beat
   longs (28.8%, −0.215R) in what was a falling market. With 29 days there are only 29
   independent observations of "kind of day", which is why market-level features are
   withheld from the model for now.
6. **The hand-written rule score has no predictive value** (AUC 0.51; its top band was
   the worst). It is kept only as descriptive context in messages.
7. **The model shows a small, real-looking selection skill but not enough.**
   Same-moment selection correlation was +0.04 to +0.09 across model families. Its
   top-ranked candidates returned **+0.33R per trade out-of-sample (t = 1.4, 41 trades)**
   and +0.09R on the 13-trade holdout. Positive, but not statistically distinguishable
   from luck — so the model was **not promoted** and the system starts in paper mode.
8. A first attempt with all features scored AUC 0.44 out-of-sample — worse than chance —
   because it memorised the regime of the first weeks. The validation harness caught
   it; the cross-sectional objective and feature withholding are the response.

### Timing check on Binance statistics

An `openInterestHist` record stamped T matches the live open interest at T and is
published 1–3 minutes later. The live collector reads OI directly at bar close, so OI
needs no shift; hourly long/short ratios are stored as available at T+10 min.

### Hypotheses being tracked, not yet rules

- Candidates firing during a moderate market-wide impulse (many coins moving the same
  way) did better (35–37% hit rate) than isolated ones (28%), but extreme impulse was
  worst (24%).
- Trades against BTC's 3-day trend beat trades with it (34.0% vs 29.3%) — mean reversion
  that is probably specific to this choppy window.

Both rest on fewer than 30 independent days and are recorded as features only.

### What this means in practice

The honest state on day one is: **infrastructure proven, edge not proven.** The bot
sends paper signals, tracks them to exit, and retrains nightly. Real-money grades appear
only after the gates in section 9 pass. More days of data are the main thing that can
change the picture: they are what unlock the regime features and make the significance
tests meaningful.
