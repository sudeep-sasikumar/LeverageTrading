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

## 12. Steady trends ("grinds") — study of 16 Nov 2026

Prompted by the observation that, even on flat days for BTC and ETH, several coins grind
10–100% one way over a day without a single breakout candle, so a leveraged position
entered mid-move could take 10% on margin.

### The shape

Measured on the user's examples (AIOT, SOPH, BTW, ZIL down; BLESS, SPACE, CROSS,
PIEVERSE up) over the 24 hours to 16 Nov 18:35 UTC:

- **Straight path:** R² of the 15-minute log-price path was 0.54–0.97, and 62–75% of
  hours moved with the trend; no single hour did more than about a third of the move
  (except AIOT's final crash hour, 48%).
- **Most of the move came late:** for most examples 70–100% of the 24h move happened
  *after* the first 6 hours, so it was catchable mid-move in hindsight.
- **The up-grinders had new money behind them:** volume 13–54× normal and open interest
  +40% to +100%, while the crowd's long/short ratio fell (shorts leaning against the move).
  **The down-grinders were quiet:** volume about 1–3× normal.
- Taker buy/sell imbalance was near zero for all eight; it did not distinguish them.

### Does the shape predict continuation? (31 days, 89,069 coin-hours)

Rule tested: at an hour close, the coin moved ≥ 4% over the last 6h with R² ≥ 0.6,
15m efficiency ≥ 0.25, ≥ 4 of 6 hours with the trend and no hour > 45% of the move;
enter on the next bar; stop 2.5 × hourly ATR, 1:1 target, 24h limit, Weex costs.

| Group | Trades | Avg R after costs |
|---|---|---|
| All liquid coin-hours (same direction as their last 6h) | 15,246 | −0.065 |
| Spiky movers (≥ 4%, one hour > 60% of the move) | 3,344 | −0.081 |
| **Grinders** | 1,983 | **+0.015** |
| Grinders, shorts | 1,181 | +0.129 (t = 5.4) |
| Grinders, longs | 802 | −0.154 |

- **Shape matters:** grinders beat other movers in the same direction by about +0.07R,
  for longs and shorts alike, and spiky movers were the worst group.
- **Direction dominated:** in a month where the median altcoin fell on most days, short
  grinds paid and long grinds lost, in both halves of the month. Requiring the grind to
  go with the broad market did not rescue the longs.
- **Ranking grinds by smoothness or size did not reliably help:** in the full replay,
  higher-ranked grinds did slightly worse.
- **"Trade the side that has been working" is harmful:** gating each side on its own
  last 3–10 days of grind results gave −0.08R to −0.21R per trade (t ≈ −3 to −7). Grind
  performance reverses over days, so the system must not chase recent winners.

Live replay of the implemented trigger (market-aligned): 2,132 candidates, 69/day,
target-first 38.8%, **−0.02R per trade** (shorts +0.09R, t = 3.7; longs −0.23R).

On 16 Nov the grind trigger flagged 3 of the day's 11 steady movers: BLESS with 94% of
the move still ahead, NIGHT 46%, PIEVERSE 42%. The momentum trigger flagged 5, at
3–89% left. The misses were mostly blocked by the stop cap (moves too volatile for a
stop under 8%) or by one hour carrying too much of a 6-hour window.

### What was built

- **Grind features** for every candidate: 6h move, 6h and 12h straightness, 15m
  efficiency, hours with the trend, biggest-hour share, hourly ATR. They are recorded
  for momentum candidates too, so the main model can learn from them.
- **A grind setup as a separate, paper-only track:**
  - stop 2.5 × hourly ATR (1.5–8%), 1:1 target, up to 24h, leverage chosen for +10% on
    margin at target;
  - at most 3 signals a day, 2 open at once, 4h apart, with limits separate from the
    momentum track;
  - labelled with its own trade plan and kept out of the momentum model's training set.
- **The daily grinder review** (`python -m lts movers`, in the daily Telegram report,
  and on the dashboard's *Daily movers* tab):
  - finds the last 24h's coins that moved 10%+ in a steady line;
  - records their volume, open interest, taker flow, funding and positioning;
  - records whether LTS flagged each one, and how much of the move was still ahead when
    it did.
  - These rows accumulate in `daily_movers`: the growing library used to judge the grind
    setup and decide when its definition should change.

The grind track stays paper (`grind.live: false`) until its forward record shows an
edge. The bar is at least 30 closed signals averaging +0.1R or better, with t ≥ 2.
Today it is break-even overall, and its only positive slice, short grinds, depends on
this month's falling market.

## 13. Grind lab: what successful grinders have in common (16–17 Nov 2026)

Section 12 tested *every* coin showing the grind shape. This section answers the real
question: among coins that were grinding early enough to enter, what did the ones that
**kept going** share, compared with **lookalikes that fizzled**? A trait only counts if it
separates the two groups, not merely if winners have it, and only if the separation holds
on later days the study did not see.

### Data (`lts/grindlab_data.py`)

- 125 days of hourly bars and full daily history for all 519 Binance perps.
- 4 months of funding history, plus open interest and long/short ratios for the last
  30 days (Binance keeps no more).
- Binance sector tags.
- 198 coin-specific exchange announcements: Binance listings, futures launches,
  delistings, monitoring tags and Alpha listings; Upbit listings, caution warnings and
  delistings.
- No per-coin general-news feed was available, so general news is not covered.

### Method (`lts/grindlab.py`)

1. **Catalogue:** every 24h window that moved ≥ 10% with R² ≥ 0.6, at least 14 of 24
   hours the same way and no single hour > 45% of the move. This gave 7,350 steady grinds
   in 120 days (≈ 61 a day; 3,951 up, 3,399 down; median move 13.6%).
2. **Entry moments:** an hour close after ≥ 3% in 6h with R² ≥ 0.7, at least 4 of 6 hours
   the same way, no hour > 50% of it, and $5M+ daily volume. This gave 16,554 early-grind
   moments, one per coin and side per 12h.
3. **Outcome:** success = +6% before −3% within 24h. **25.7%** succeeded; the rest are the
   lookalikes. Seven alternative exits and a trailing stop are recorded too.
4. **~100 factors at the entry moment**, all using only data available then:
   - multi-timeframe support/resistance (1h/2-day, 4h/7-day, 30-day, 90-day, 52-week,
     all-time);
   - daily pivot levels and the room to the next one;
   - compression before the move;
   - volume and trade-count surges, taker flow, and volume on with-trend hours;
   - funding, open interest and positioning;
   - BTC and market moves, and sector peers;
   - sector tags, listing age, size;
   - exchange announcements in the past week.
5. **Validation:** factors are compared within volatility bands, then a factor checklist and
   a model are fitted on the first 60% of days and judged on the last 40%.

### Findings

**Traits of successful grinders that held up on unseen days** (success with vs without
the trait on the held-out 40%):

| Trait | With | Without |
|---|---|---|
| Down-grind in the week after an Upbit listing or Binance futures launch | 43% | 24% (n = 61) |
| No daily support/resistance within 10% ahead | 28% | 23% |
| Breaking out of / at the edge of its 30-day range | 27% | 23% |
| BTC's 7-day trend in the same direction | 26% | 22% |
| The coin's own week in the same direction | 26% | 22% |
| Coin-specific move (the market *not* moving the same way in the last 6h) | 26% | 23% |
| Younger listing | 26% | 23% |
| Stronger than the market over 24h | 25% | 23% |

**Faded on unseen days:** sector peers moving the same way, funding not against the trade,
light taker flow.

**The important caveat.** The more of these traits an early grind has, the further it
moves, *in both directions*. With 9+ traits, half reached +6% and about a third reached
+10% within 24h, but the typical adverse move grew just as much. So the traits mainly
predict **how much** a coin will move, not **which way**.
- Every exit tested (target/stop pairs from +3/−3 to +12/−4, and a trailing stop) lost a
  little after fees, at every trait count.
- A model combining all factors raised success on unseen days only from 25% to 28%.

**The one directional pattern:** in the week after an **Upbit listing** or a **Binance
futures launch**, early *down*-grinds succeeded about 41% of the time and made about
+0.6% per trade after fees (138 cases on 29 coins, t ≈ 1.6). Up-grinds after the same news
lost money. It is a plausible "sell the news" effect but not yet statistically proven.

Other facts:
- Steady grinds are ordinary: most liquid coins have one in any given week.
- Only 2.6% of grinds had an exchange announcement in the week before.
- Coins tagged Alpha are over-represented (1.36×).
- Grind start times are spread evenly over the day.

### What runs in production

- **Nightly (01:10 UTC):** refresh history and announcements, rebuild the catalogue and
  the study, refit the model and the checklist thresholds, store the report, and send a
  short Telegram summary. The definitions update as data accumulates.
- **Hourly (3 min past):** fetch the latest hourly bars, detect early grinds, score each
  against the 12-trait checklist and the model, and record them in `grind_live`.
- **Paper trades:**
  - which ones: news-catalyst down-grinds first, then early grinds with ≥ 8 of 12 traits;
  - plan: the study's own (−3% stop, +6% target, 24h);
  - limits: at most 3 a day and 2 open, ≥ 4h apart;
  - the earlier 5-minute grind trigger now only logs candidates.
- **Dashboard → Grinders tab:**
  - coins grinding right now, with their traits;
  - the grind paper trades;
  - the study ("what successful grinders have in common", trait by trait with held-out
    results, the trait-count table, factors stable in every period);
  - the last 7 days of steady grinds;
  - the daily review of whether LTS caught them.

Paper trades become real-money signals only when the forward record shows an edge (≥ 30
closed trades averaging ≥ +0.1R with t ≥ 2). The news-catalyst shorts are the first
candidate to watch.

## 14. One-year pattern study (17 Nov 2026)

**Question:** do classic chart or smart-money patterns carry a tradable edge on these perps?
Tested: support/resistance, momentum, liquidity sweeps, order blocks, fair value gaps,
structure breaks, and more.

**Data:** 400 days of hourly bars for 518 coins (from 13 Oct 2025), full daily history,
1 year of funding, and 511 Binance/Upbit announcements. Code: `lts/patterns.py`,
`lts/pattern_study.py`, `lts/sweeps.py`.

**Discipline, fixed before any result was seen:**
- **Two stages:** discovery on the first ~9 months, then a single confirmation test on the
  last 120 days with everything frozen.
- **Day-clustered statistics.**
- **Real Weex fees:** taker plus slippage for market orders and stops, maker for limit
  entries and targets.
- **Structural stops:** at the pattern's invalidation point.
- **A discovery pass required all of:** at least 150 trades, at least +0.10R per trade after
  fees, t ≥ 3, and at least 60% of months positive.

### Results

1. **15 pattern families, 32 variants × 2 sides × 6 exits = 491,097 simulated trades.
   None passed discovery.**
   - Families: 20/55-day breakouts with level or ATR stops; breakout retests; 52-week
     highs/lows; multi-touch daily support/resistance bounces; liquidity sweeps of the
     24h, 7-day, previous-day and equal highs/lows, with and without volume; 4h sweeps;
     1h/4h order blocks and fair value gaps after a displacement plus break of structure;
     4h change of character; trend pullbacks; volatility-squeeze breakouts; volume
     climaxes; funding extremes; grind continuation.
   - The best was +0.007R per trade, t = 1.8.
2. **The year was a deep altcoin bear market.** The median liquid altcoin fell **68%**. Long
   setups lost in almost every family and every month (for example, trend-pullback longs
   −0.30R, t = −17). Their short mirrors only broke even, because fees eat the drift.
3. **A regime filter didn't help.** Trading each pattern only in the direction of the
   altcoin index's 50-day trend: zero passed. The plain regime-direction basket also lost
   after daily costs.
4. **Liquidity sweeps carry real timing information.** Against matched random-time trades
   (same coin, ±3 days, same stop size and exit), sweeps did better by +0.19R to +0.37R
   before fees, with t = 3–10 in **both** periods and on **both** sides:
   - shorting a sweep of the 24h high: +0.30R discovery / +0.27R confirmation;
   - with a volume spike: +0.33R / +0.37R;
   - longing a sweep of the 24h low: +0.19R / +0.21R.
5. **But the sweep is not a tradable edge at Weex fees.** A pre-declared grid of 48 ways to
   trade it (market or limit entry; wick or wide stop; 2R, 3R or opposite-liquidity target;
   with or without a volume filter) produced zero passes.
   - The best variant (short, market entry, stop 1 ATR beyond the wick, 3R) made +0.035R
     in discovery and **−0.013R on the unseen months**.
   - One such trade a day at 1% risk: account −25%, worst drawdown −35% over those months.
   - Leverage (3x / 5x / 10x) scales each win and loss (typical +20% / +33% / +66% vs −11% /
     −18% / −37% of margin) but not the near-zero expectancy.

### Conclusion

With price, volume, funding, sector and exchange-announcement data, none of the patterns
tested (support/resistance, momentum, sweeps, order blocks, fair value gaps, structure
breaks) has a robust edge after fees on these coins over the past year. The sweep's
information is real but smaller than trading costs plus the losses built into bracket trades
on volatile altcoins. Nothing from this study was added to live signals.

The data most directly behind sweeps and stop hunts — liquidations and order-book depth —
has no free history, so it has to be collected going forward before it can be studied.

## 15. The aim, tested directly: one grind trade a day for ~10% on margin (17 Nov 2026)

The project's goal is **one trade a day worth about 10% on margin**: a +1% move at 10x, or
+2% at 5x. Sections 12–14 drifted to larger targets. This tests the goal itself.

**Setup:**
- **Candidates:** 63,528 early-grind moments over 389 days (`lts.grindlab`), each followed for
  24h on **5-minute** bars (`lts.aim`). A 5-minute bar touching both target and stop counts as
  a loss.
- **One trade a day, decided in real time:** the first moment of the UTC day whose score
  clears a threshold, and no second trade while one is open.
- **Fees:** Weex taker on entry, stops and time exits; maker on the take-profit.
- **Grid (fixed in advance):** targets +1%, +1.5%, +2%; stops 0.7%, 1%, 1.5%, 2%, 3%; time
  limits 8h and 24h; rankings: first of the day, 12-trait checklist, smoothness, news
  catalyst, and a model trained only on discovery days; five thresholds.
- **Two stages:** selection on 269 days, a single frozen test on the last 120.

**Results:**
- **A grind reaching +1% is a coin flip against an equal stop.** +1% before −1% happened 50%
  of the time (48% in the last 4 months), netting −0.20% to −0.24% per trade after fees. Every
  target/stop pair sat at about −0.16% to −0.22% per trade unselected: the odds are fair and
  fees decide the sign.
- **The checklist, smoothness and "first of the day" rankings added nothing.** Only the model
  helped in discovery: its top 5% with +1.5% target / −3% stop won 76% and made +0.32% per
  trade, at about one trade every two days and t = 1.95. **No combination passed** (t ≥ 3, 60%
  of months positive, a trade at least every other day).
- **On the unseen 4 months, that best combination lost:** 64% wins, −0.31% per trade (−1.6%
  of margin at 5x), worst drawdown −18%. All eight top discovery combinations were negative
  there.

**Conclusion:** with price, volume, flow, funding, open interest, sector and news data, the
"one +1–2% grind a day" trade is about break-even before fees and negative after them.
Leverage multiplies that. The grind is a real, frequent move, but its continuation from the
moment it can be seen is close to a fair coin over 1–2% distances.

**What runs:**
- **Daily-trade track** (`config.aim`): each hour, the first early grind of the day with ≥ 8
  of 12 traits becomes the day's trade (+1% target, −1% stop, 24h, 10x), sent to Telegram as
  🎯 DAILY TRADE and tracked on the dashboard.
- It is **paper** (`aim.live: false`) so the aim is measured live every day. It becomes
  tradeable only if its forward record turns positive with significance.
- **New data accumulating:** liquidations and order-book depth (section 14) are the main
  candidates to change this result.

## 16. Pattern recognition on the grind's minute-level shape (17 Nov 2026)

**Request:** separate winning grinds from losers with pattern recognition, and send every
signal rather than one a day.

**Data:**
- For all 63,528 early-grind moments of the year, the **6 hours of 5-minute bars before the
  decision**: price, volume and taker-buy volume.
- 50 shape features in the trade's direction (`lts/grindshape.py`):
  - the 30-minute path;
  - speeding up or tiring;
  - number, depth and recovery of pullbacks;
  - straightness and efficiency over 6h and 2h;
  - candle bodies and wicks against the move;
  - volume trend, climaxes, and volume on with-trend bars;
  - buy/sell flow over 15 minutes, 1 hour and 6 hours, and its shift.

**Label:** the aim's trade, +1% before −1% within 24h on 5-minute bars, net of fees. Also
+1.5% before −1.5%.

**Test:** monthly walk-forward. Each month is predicted by a model trained only on earlier
months (1-day gap), giving 11 unseen months. Three models: hourly/daily factors, minute shape
only, and both together.

**Results (unseen months):**

| Model | AUC, +1%/−1% | AUC, +1.5%/−1.5% |
|---|---|---|
| Hourly/daily factors | 0.502 | 0.496 |
| Minute shape only | 0.519 | 0.512 |
| Both | 0.508 | 0.496 |

- 0.50 is a coin flip. The factor models' most confident signals won *less* often (44% at
  ≥ 65% confidence vs 49% overall).
- The minute-shape model's best band won 51.9% at about 4 signals a day. A +1%/−1% trade needs
  about 58–61% wins to pay Weex fees, so it still lost −0.15% per trade, and was positive in
  only 2 of 11 months.
- **No band qualified** (net > 0, t ≥ 3, ≥ 70% of months positive).

**Conclusion:** with price, volume and flow data, whether a visible grind runs +1% before
−1% is a fair coin. The fees, about 0.2% per trade, decide the sign.

**What runs (`config.grindsignal`, `lts/grindsignal.py`):**
- **Every hour:** every early grind is scored by the minute-shape model, and every one with
  probability ≥ 0.60 is sent (about 4–5 a day, no daily cap).
- **Each signal:** a +1% target and −1% stop at 10x, with the 5x equivalent and that
  probability band's real record on unseen months.
- **Paper unless the band ever qualifies and `grindsignal.live` is on.**
- **The model:** a seed trained on the full year ships in `lts/models/grind_seed.joblib`.
  Once the server holds 60 days of its own 5-minute data, the model is retrained nightly on it
  and its band table refreshed.
- **Superseded:** the daily-trade track (`aim`) and grind-lab paper trades are switched off.

## 17. The trader's plans: +10% on margin at 5x / 3x / 2x with tight stops (17 Nov 2026)

Section 16 used a +1% / −1% bracket (10x). That was my choice, not the trader's. The trader's
plans are +10% on margin at **5x (+2% move), 3x (+3.33%) or 2x (+5%)**, each with a **tight
stop** so losers are cut fast.

**Test:**
- the same 63,528 grinds, 5-minute paths, 24h limit, Weex fees, stop counted first;
- stops 0.3%, 0.5%, 0.75% and 1%;
- 11 unseen months, by monthly walk-forward.

**All grinds, no filter:** every combination lost almost exactly the fee, about −0.23% per
trade. Before fees, any bracket is about zero.

| Plan | Win rate needed (0.3 / 0.5 / 0.75 / 1% stop) | Actual |
|---|---|---|
| 5x, +2% | 23 / 29 / 35 / 40% | 13 / 19 / 27 / 33% |
| 3x, +3.33% | 15 / 19 / 24 / 28% | 8 / 13 / 18 / 23% |
| 2x, +5% | 10 / 13 / 17 / 20% | 6 / 9 / 14 / 18% |

Tight stops on altcoins are mostly hit by ordinary 5-minute noise: at 0.3–0.5% the median
trade lasts 0–10 minutes.

**With the minute-shape model's top-scored grinds:** the higher the score, the smaller the
loss, but nothing turned profitable.
- Best: **2x, +5% / −0.75% in the top 2%** (about 3 a day): −0.01% per trade, positive in 36%
  of months.
- 5x, +2% / −1% in the top 2%: −0.08%. 3x, +3.33% / −1%: −0.10%.
- No combination qualified.

**What runs (`config.grindsignal`):**
- Every grind in the model's **top 5%** (about 7–8 a day) is sent, with all three plans:
  - 5x: +2% / −1%
  - 3x: +3.33% / −1%
  - 2x: +5% / −0.75%
- These are the stops that held up best per plan. Each plan shows its margin result at
  target and stop, plus its own record in that score band on unseen months.
- The 2x plan is tracked as the signal's result. **Every plan's outcome is recorded** for
  every signal after 24h (`signal_plans`) and shown on the Grinders tab, so live results show
  which plan works.
- PAPER unless the tracked plan's band qualifies and `grindsignal.live` is turned on.

## 18. 10x plan, top 20%, and a full record of every grind (17 Nov 2026)

**10x plan:** +1% target. The trader set no stop, so the tight stops 0.3%, 0.5% and 0.75% were
tested on the year. They are indistinguishable: −0.22% to −0.24% per trade on unseen months,
needing 39–54% wins against 21–42% actual. The trader's rule is a 10x stop of at most 0.5%,
so 0.5% is configured (`grindsignal.plans[0]`; about −7.4% of margin at the stop after fees).
In the top 20% it won 32.4% (about 41% needed), −0.22% per trade.

**Plans now:** 10x +1% / −0.5%, 5x +2% / −1%, 3x +3.33% / −1%, 2x +5% / −0.75% (tracked).

**Sending:** the model's top 20% (score ≥ 0.146 with the seed model, about 30 a day), as the
trader chose. A percentile of a weak model is not a sound filter; the live record decides
later.

**Recording every detected grind** (`grind_live`, sent or not):
- the model score and the threshold at the time, and the decision;
- every hourly/daily factor and all 50 minute-shape features (`all_factors`);
- liquidations in the past 1h and 6h by side, and the latest order-book snapshot (`micro`);
- after 24h (`outcomes`): every plan's result, a 5 × 6 target/stop grid (+1/1.5/2/3.33/5%
  × 0.3/0.5/0.75/1/1.5/2%), best and worst move after 1h/4h/24h, and the 24h close.

Any filter, threshold, plan or stop can be re-tested later on live data without re-collecting
anything.

**Dashboard:**
- The Grinders tab lists every grind of the last 3 days with its score, the threshold, whether
  it was sent, and each plan's outcome.
- Every coin name in every table links to its Weex perpetual page.
- Telegram signals carry the same link.

## 19. Every 5 minutes, 5x +2% tracked, only grind signals sent (18 Nov 2026)

Alignment review against the aim (one +10% trade a day = a 1–2% move at 5–10x):

- **Scan every 5 minutes** (was hourly at hh:03). The last 30 hours are rebuilt as rolling
  1-hour steps from 5-minute candles that end at the latest 5-minute close, so a grind is caught
  within about 6 minutes instead of up to an hour (`grindlab.scan_now`, `FiveMinuteScanner`).
  Older history still comes from the hourly table, which is topped up once an hour. 1–2 minute
  scanning was not adopted: the collector, every study and the model are built on 5-minute candles,
  and over the 5 minutes a typical grind moves about 0.05%.
- **Recording:** a coin and side is recorded at first detection, then about once an hour while it
  keeps grinding, plus whenever its score crosses the sending threshold (it is sent then).
- **Tracked plan and model label: 5x +2% / −1%** (was 2x +5% / −0.75%). The seed model was
  retrained to predict that plan. The threshold for the top 20% is now 0.367, about 31 a day.
  On unseen months, the top 20% won 33.7% against about 40% needed to break even after fees,
  −0.20% per trade.
- **Only grind signals are sent.** The momentum track (stop sized by ATR, so leverage came out as
  3–5x with 1–2.5% stops) is logged for learning only (`signal.momentum_send = False`).
- **Grind signals are not blocked** by the daily −2R stop or the two-losses-in-a-row stop (the trader
  picks manually). They still skip a coin that already has an open signal or lost in the last 4h.
- **Early results on the dashboard:** grinds under 24h old show each plan as soon as its target or
  stop is hit, and the best and worst move so far (`progress` column, `progress_grinds`), instead
  of "pending".

## 20. Pullback entry test (18 Nov 2026)

Idea from the 18 Nov movers: most detected grinds that reach +2% first dip about 1%, so wait for a
dip before entering. Test: at detection, rest a limit order 0.25–2% against the move for 1–8h; if
filled, the trader's plan from the fill price until 24h after detection. Conservative fills (the
fill bar can stop you out but cannot pay the target). Maker fee on the limit entry. 63,528 grinds,
Oct 2025 – Nov 2026; variants picked on the first 7 months, judged on the last 7.

| Plan | Enter at once (last 7 months) | Best dip (0.5%) (last 7 months) |
|---|---|---|
| 10x +1% / −0.5% | won 29.7%, −0.26% per trade | won 30.3%, −0.15% |
| 5x +2% / −1% | won 31.9%, −0.25% | won 31.8%, −0.15% |
| 3x +3.33% / −1% | won 22.5%, −0.25% | won 22.4%, −0.16% |
| 2x +5% / −0.75% | won 13.2%, −0.26% | won 13.1%, −0.17% |

Every variant failed: negative in every month of the second half. The improvement of about 0.1%
per trade is almost exactly the fee saved by entering with a limit order instead of at market;
the win rate does not change. Deeper dips fill less often and do no better. Not adopted for
signals. The useful by-product: entering grind signals with a limit order rather than at market
saves about 0.1% per trade.
