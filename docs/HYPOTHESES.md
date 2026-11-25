# Pre-registered studies

Each study below is written down, with its hypothesis, exact rules, parameter grid and pass criteria, and committed
**before** it is run on the data. Results go to `docs/results/`, produced by the named script in `research/`.
Validation code: `lts/validate.py`. Costs: `lts/fees.py` (config.trade). Nothing here is changed after the run;
a correction would be a new, dated entry.

## Pass bar (all trade-rule studies)

A variant passes only if, in **discovery**:

- at least 150 trades over at least 60 days;
- at least **+0.10R per trade after all costs** (R = the stop distance; fees, slippage and funding included);
- day-clustered t ≥ 3 **after the Holm correction** for every variant of that study's grid;
- at least 60% of calendar months positive;

and then, with its rules frozen (`docs/results/<study>_frozen.json`), in **confirmation**: positive with t ≥ 2.
Every report also gives results by market regime (equal-weight altcoin index above / below its 50-day mean,
`validate.alt_regime`) and by side.

Common data: Binance USDT-M perps; hourly candles, daily candles and settled funding from the local database
(13 Oct 2025 – 16 Nov 2026); 5-minute and 1-minute candles fetched from Binance into `research_data/`.

---

## P0. Reproduce the published year results (registered 25 Nov 2026)

Not a hypothesis test: a check that the rebuilt data and code give the published numbers.

- Events: `grindlab.build(days=390)` with the window ending 16 Nov 2026 23:35 UTC (when the original table was
  built). Published: 63,528 moments, 389 days, 519 coins. **Done before registration: reproduced exactly**
  (`docs/results/build_events.txt`).
- §25 R3 (`research/reproduce_r3.py`): 5x +2% / −1%, 24h, entry at the first 5-minute open after the hour close,
  stop first inside a 5-minute candle, costs from config. Published: 539 grinds; won 38.0%, −0.084% per trade,
  5 of 14 months positive; first 7 months 36.5% / −0.11%, last 7 months 39.6% / −0.01%; all grinds 33.7% /
  −0.20% and 32.3% / −0.24%. Also the other three plans (10x 34.3% / −0.211%, 3x 26.3% / −0.089%,
  2x 15.2% / −0.113%).
- §27 runners (`research/reproduce_runners.py`): the table in DESIGN.md §27 (winners, best point, reach +3/+5/+10%,
  "before falling back to +1%", back to +1%, back to entry, 24h close).
- Any difference from the published numbers is reported, not explained away.
- The same script writes `lts/models/steady_history.json` (review item 13).

## P1. Live-parity replay of the steady rule (registered 25 Nov 2026)

**Question:** the year's 38% for R3 was measured once per coin and side per 12h, at the first early-grind
detection on a clock-hour close. Live (until 25 Nov 2026) checked R3 every 5 minutes on rolling hours and
re-sent. What does the year give under the live behaviour?

Replay (`research/parity.py`), on every coin, for every 5-minute end in the span of the year events (22 Oct 2025 – 15 Nov 2026):

1. Rolling hours: the last 30 one-hour bars ending at that 5-minute end, built from 5-minute candles
   (`grindlab._rolling_hours`; on an hour close these equal the clock hours). Early-grind mask and R3 on them,
   exactly `grindlab.early_grind_mask` and `grindsignal.steady` (R3 needs only the last 25 bars, so older history
   does not change it).
2. Recording, exactly `service.grinds_to_record` with `steady_eval = rolling_5m`: a coin and side is recorded at
   detection when it has no record in the last 55 minutes, or when it meets R3 now and did not at its last record.
3. Sending: a recorded grind that meets R3 is sent unless the coin has an open signal (either side), or the coin
   lost in the last 4h (`RiskState` for grindsig as it was: no daily stop, no streak stop, no cap).
4. Each signal: 5x +2% / −1%, 24h, entry at the open of the 5-minute candle starting at that end (live enters at
   the Weex price about 1–2 minutes later), stop first inside a candle, costs from config. A signal stays open
   until its exit; then the coin can be sent again.
5. Variants reported side by side: (a) as above; (b) (a) plus the level-in-way skip (`skip_if_level_blocks`, on
   live since 19 Nov); (c) `hour_close_first` (the new default) — must reproduce P0's 539-trade R3 result up
   to the entry-candle convention.
6. Reported for each: trades, per day, win rate, net per trade, day-clustered t, months positive, and the share
   of signals that were a repeat on the same coin and side within 24h.

Coverage check (because 5-minute candles are fetched only around coin-hours near the rule — 24h move ≥ 6%,
24h R² ≥ 0.6, ≥ 12/24 hours its way, 6h move ≥ 2% at a neighbouring hour close): on 13 Oct – 16 Nov 2026,
where the local database holds 5-minute candles for every coin, the full replay is also run without the
prefilter, and the number of R3 detections the prefilter would have missed is reported.

## B1. Breakouts with a wide stop and a high reward:risk (registered 25 Nov 2026)

**Hypothesis:** an hourly (or 4-hourly) close beyond a multi-day high/low, the first such break in 72h, is
followed by continuation large enough that a wide-stop, 2–4R bracket is profitable after all costs, and better
than the same bracket at a random time on the same coin.

- Data: hourly candles 13 Oct 2025 – 16 Nov 2026 (`klines_1h`). Universe at the signal: 24h quote volume
  ≥ $5M, ≥ 21 days of hourly history.
- **Entry signal** (long; shorts mirror): the close of bar t is above the highest high of the N bars before
  it, N ∈ {24h, 3d, 7d, 20d}, and no close above that N-high happened in the previous 72h (first break).
  - Variants: all breaks / only with that bar's quote volume ≥ 2 × the mean hourly quote volume of the previous
    168h; hourly close / 4h close (UTC 4h bars, the 4h close above the N-high of the bars before that 4h bar).
  - 16 entry variants (4 levels × 2 volume × 2 timeframes) × 2 sides.
- **Entry:** market at the open of the next hourly candle.
- **Stop:** (a) fixed 5%; (b) structural: the broken level − 0.5 × ATR(14, hourly) (long), skipped when that
  is under 0.5% or over 15% from entry.
- **Targets:** 2R, 3R, 4R (10% / 15% / 20% with the 5% stop). **Time stops:** 24h, 72h, 7 days, at the close.
- If target and stop are both inside one hourly candle, the stop counts. The target is checked from the entry
  candle on (the stop too).
- **Costs:** taker + slippage on entry, stop and time exits; taker on the target (main) and maker on the target
  (reported variant, not a separate hypothesis); funding paid or received at every settlement held (settled
  rates from `funding`; a long pays a positive rate).
- **Control:** for each trade, 5 entries at random hour opens within ±3 days on the same coin and side (not
  inside the 6h around the real entry), with the same stop distance (%) and exit rule. Timing information =
  real minus the mean of its controls (day-clustered t), as in DESIGN.md §14.
- **Grid / family for Holm:** 16 entries × 2 sides × 2 stops × 3 targets × 3 time stops = 576 variants.
- **Split:** discovery = entries before 1 Jun 2026; confirmation = 1 Jun – 16 Nov 2026 (locked).
- **Report** per variant that is in the top 10 of discovery, and for every passing one: win rate vs the
  break-even win rate, net % per trade, % on margin at 5x, max drawdown (cumulative R in time order), t (raw and
  corrected), months positive, trades per day, by regime and side, and the timing-vs-control difference.
- **Break-even win rates to confirm in code (5x, taker TP):** +10%/−5% ≈ 34.8%, +15%/−5% ≈ 26.1%,
  +20%/−5% ≈ 20.9%; current +2%/−1% ≈ 40.8% (`research/b1_breakouts.py` prints them).

## B2. Why the big daily movers are not caught (registered 25 Nov 2026)

**Questions:** for coins that move ≥ 15% in a UTC day, when could the scanner first have flagged them, how much
of the move was still ahead, how many lookalikes did the same logic flag at those moments, and how do the plans
do on each group?

- **Big mover:** a UTC day whose daily candle moved ≥ 15% open to close (either way), coin in the universe
  (24h quote volume ≥ $5M at that day's start), 13 Nov 2025 – 15 Nov 2026.
- **Flag (primary, hour-close logic):** the first clock-hour close inside that day where
  `grindlab.early_grind_mask` fires in the move's direction (hourly bars: complete for every coin).
  **Flag (live logic):** the first 5-minute end inside that day where the mask fires on rolling hours
  (5-minute candles fetched around every big-mover day); reported as how many minutes earlier it is.
- **Still ahead:** side × (day close − price at the flag) / side × (day close − day open); also the share of the
  day's best extreme still ahead.
- **Lookalikes:** at each primary flag time, every other coin whose mask fires at the same hour close in the same
  direction and whose own UTC day moves < 15% that way.
- **Plans replayed on both groups**, entry at the next 5-minute open after the flag: 5x +2% / −1% (24h), and B1's
  fixed-5%-stop brackets (2R, 3R, 4R; 72h), costs and funding as in B1. Descriptive comparison (the mover group is
  defined with hindsight, so its results are not a tradable rule); the lookalike-included group is the tradable one.
- **Stop-first check:** for a random sample of 600 flags (300 movers, 300 lookalikes; seed 25), 1-minute candles for
  the 24h after entry; the 5x plan's win rate on 1-minute candles (stop first inside a 1-minute candle) against the
  same trades on 5-minute candles; reported as the change in win rate and in net per trade.

## B3. Execution costs: limit entries and reduce-only limit take-profits (registered 25 Nov 2026)

**Question:** how much of the fee drag can resting orders remove, and what does adverse selection cost?

- **Weex rules** (official API documentation, read 25 Nov 2026 — see the B3 report): `POST /capi/v3/order`
  accepts `type=LIMIT` with `reduceOnly=true` and `timeInForce=POST_ONLY`; exchange info lists
  makerFeeRate 0.0002 / takerFeeRate 0.0008. Whether a filled reduce-only limit close is charged the maker
  rate is **not documented**; the report says so, and the trader's own fills are the only evidence.
- **Sample:** every all-day steady (R3) grind of P0, and 2,000 random other early grinds (seed 25), 1-minute
  candles from 15 minutes before to 24h after entry.
- **Limit entry:** a limit at the signal price (the hour close), resting 15 minutes; it fills only if price
  trades through it by at least one tick (Binance tick size from `exchangeInfo`, as a proxy for Weex); otherwise
  no trade. **Reduce-only limit TP** at +2%: fills only when price trades at least one tick beyond it.
  Stop: market (taker + slippage) at −1% from the fill.
- **Measured:** fill rate; win rate of filled trades vs all trades (adverse selection); net per trade with
  (a) market in / taker TP (today), (b) limit in / taker TP, (c) market in / maker TP, (d) limit in / maker TP;
  missed winners (target reached but the entry never filled).
- **Also:** the best §14 sweep variant (short after a 24h-high sweep, market entry, stop 1 ATR beyond the wick,
  3R) re-run with the maker-TP fill rule and costs on its hourly candles.
- **Compared with** the trader's fills recorded in `my_trades` (fee per fill vs maker/taker rates).
- Pass bar as above for (b)–(d) as rules (family: 3 variants × 2 samples).

## B4. Sweeps with liquidations and order books (registered 25 Nov 2026; to run from 16 Jan 2027)

- Code now (`lts/sweeps_micro.py`): sweep detection (`lts/sweeps.py`) joined with liquidations (USD liquidated
  on each side in the 15 minutes before / after the sweep candle, and a cascade flag: ≥ 3 × the coin's median
  15-minute liquidations) and order-book snapshots (`book_snapshots`: depth imbalance within 0.5/1/2% and the
  wall nearest the swept level, at the last snapshot before and the first after the sweep); maker entries back
  at the swept level.
- **Refuses to evaluate** before 60 days of liquidation and book data exist (data since 16–17 Nov 2026, so about
  16 Jan 2027). Discovery = the first 60 days; confirmation = everything after, locked until at least 30 more
  days have passed and the discovery rules are frozen.
- Known limits, stated in every report: the Binance liquidation stream sends at most one event per coin per
  second (bursts are undercounted); books are snapshotted every 5 minutes for about 120 coins.
- Feature candidate: Binance → Weex lead-lag — the correlation of Weex 1-minute returns with Binance 1-minute
  returns k minutes earlier (k = 0…5) and the lag with the highest correlation (`lts/leadlag.py`).
- Pass bar as above; its grid is fixed in the code before the first evaluation.

## Part C. Signal policy (registered 25 Nov 2026)

Not a study. Implemented behind config: a grind signal is sent only when the selecting rule's expected net
after costs (its record in `lts/models/steady_history.json`, from P0/P1) is positive; a track may go live only
after it passes the bar above and then has at least 30 closed live paper trades sent under that exact rule.
