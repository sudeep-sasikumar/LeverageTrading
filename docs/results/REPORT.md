# Review of 25 Nov 2026 — final report

Branch `review-parity-studies` (from `main` at 4ed1fe9). Every track stays PAPER (`grindsignal.live`, `grind.live`,
`aim.live` are false; no model was force-promoted). Every figure below comes from a committed script in `research/`
whose full output is in this folder. Studies were pre-registered in [docs/HYPOTHESES.md](../HYPOTHESES.md) (commit
46805f9) before they were run.

## 1. Part A — each item

| # | Finding | Confirmed? | What changed | Test |
|---|---|---|---|---|
| 1a | Live checked the steady rule every 5 min on rolling hours and re-sent; the year test checked it once per coin/side per 12h at the first hour-close detection | Yes (`service.grinds_to_record`, `RiskState` grindsig) | `grindsignal.steady_eval`: `hour_close_first` (default) / `rolling_5m`; first detections tracked in `grind_first` (seeded conservatively from `grind_live` on the first run) | `test_review_fixes.py::test_hour_close_first_*`, `test_rolling_5m_keeps_the_old_behaviour` |
| 1b | Re-run the year under live behaviour | — | `research/parity.py` (section 2 below) | `test_research_parity.py` (its lean feature function equals `hourly_shape`) |
| 2 | `await self.weex.book` unguarded in `grind_signals` | Yes | try/except per pick; the pick is decided `no_weex_book` instead of losing the cycle | `test_grind_signals_survive_a_weex_error_and_need_no_model` |
| 3 | `grind_signals` returned early without the model | Yes | Only `selector=model` needs it; steady signals go out without a score | same test; `test_steady_signal_message_uses_the_rules_year_record` (no model score) |
| 4 | Grade "G" came from the model's score band under `selector=steady` | Yes | Grade from `gate.live_allowed` for the configured rule | `test_expected_net_gate_and_go_live_gate` |
| 5 | BTC-shock and funding vetoes never fired for grind signals | Yes (feature dict lacked `btc_r_15m`, `fund_rate`) | `service.market_now`: side-aligned BTC 15m return, current funding (premium; else settled ≤ 9h) | `test_btc_shock_and_funding_vetoes_apply_to_grind_signals` |
| 6 | Grind signals bypass daily-loss / streak stops | Yes (intentional per DESIGN §19) | **Back on by default**: `grindsignal.daily_stop`, `streak_stop`, new `max_same_direction = 2` (while the median coin moves that way). Turn off: set to `false` / `0` | `test_grind_signals_daily_and_streak_stops_are_back_on_and_switchable`, `test_grind_signals_same_direction_limit_*` |
| 7 | Momentum promotion/demotion used every kind of signal | Yes | Filtered to `kind = momentum` | `test_learning.py::test_momentum_forward_record_ignores_grind_signals` |
| 8 | Tracker start rounded down to the signal's minute | Yes | Starts at the first full minute after the signal | `test_tracker_review.py::test_tracker_starts_*`, `test_tracker_ignores_the_candle_the_signal_was_sent_in` |
| 9 | Tracker capped at 1,000 one-minute bars | Yes | `klines_range` on both venues: Weex `/market/historyKlines` (start/end honoured, 100 per request — `/market/klines` ignores them), Binance fallback; unfilled stretches in `signals.track_gaps` | `test_tracker_fills_a_long_gap_*`, `test_weex_history_klines_pages_backwards` |
| 10 | `plan_outcomes` / `after_exit` overwrote `aim` globals | Yes | `aim.outcomes(paths, side, targets, stops, hours, fee)` | `test_aim.py::test_outcomes_takes_its_grid_as_arguments_*` |
| 11 | Exit type guessed from `net >= target − 0.3%` | Yes | `outcomes` returns `Outcome(net, exit_bar, kind)` | same test (a +4.8% time exit on a +5% target is labelled `time`) |
| 12 | Fees hard-coded in aim, exits, grindlab, patterns | Yes | `lts/fees.py` reads `config.trade` only; all modules use it | `test_fees.py` (fails on any fee literal in `lts/` or `research/`) |
| 13 | `STEADY_HISTORY` hard-coded | Yes | `lts/models/steady_history.json`, written by `research/reproduce_r3.py` and `research/parity.py` | `test_steady_history_is_loaded_from_the_generated_file` |
| 14 | Research scripts not in the repo | Yes | Recovered verbatim from the session transcript into `research/recovered/` (they point at a cleared scratchpad; kept as the record); §25 and §27 recreated as `research/reproduce_r3.py`, `reproduce_runners.py` | section 2 below |
| 15 | `grinds_to_record` / `grind_levels` DB work on the event loop | Yes | Run in a worker thread with their own connection | covered by the service-path tests |
| 16 | Grind-lab bundle reloaded every 5 min; no scan timings | Yes | Cached, reloaded on file change; heartbeat + health panel show collect / momentum scan / grind scan seconds | `test_ops_review.py::test_grindlab_bundle_*`, `test_dashboard_shows_scan_timings` |
| 17 | Nightly retrain replaced the year-trained seed | Yes | `beats_seed`: the new model must net more on months neither model saw (≥ 300 trades each), else it is stored as a challenger | `test_retrained_grind_model_must_beat_the_seed_*` |
| 18 | No login rate limit | **Partly wrong**: a global limit existed (10 failures / 15 min) | Now 5 per client IP (last `X-Forwarded-For` entry, the one Traefik appends) + 30 overall | `test_dashboard.py::test_login_is_required_and_rate_limited` |
| 19 | Non-sending nightly jobs use CPU | Yes | `learn.retrain_schedule`, `grindlab.study_schedule`: daily / weekly (default, Mondays) / off; the data refresh still runs nightly | `test_nightly_schedules`, `test_nightly_refresh_only_skips_the_study` |
| 20 | README describes momentum A/B signals | Yes | README rewritten; DESIGN.md §28 | — |

Part C (signal policy): `grindsignal.require_positive_ev` (on) sends a grind signal only when the rule's tested
expected net after costs is positive, shown with its 90% interval in Telegram and on the dashboard;
`lts/gate.py` lets `live` take effect only after the bar is passed AND 30 closed live paper trades under that rule.
Tests: `test_expected_net_gate_and_go_live_gate`, `test_dashboard_shows_the_signal_policy`.
**With today's record (5x −0.084% per trade) no grind signal is sent.** To keep receiving paper calls, set
`grindsignal.require_positive_ev: false`.

## 2. Parity: the year under the live behaviour vs the year's R3 result

`research/parity.py` → [parity.txt](parity.txt); `research/reproduce_r3.py` → [reproduce_r3.txt](reproduce_r3.txt).
5x +2% / −1%, 24h, taker fees both ways, stop first inside a 5-minute candle.

| How R3 is checked | Signals | Per day | Won | Net / trade | t (day) | Months + | Repeats < 24h |
|---|---|---|---|---|---|---|---|
| (a) every 5 min, rolling hours, re-sends (live until 25 Nov) | 10,469 | 26.9 | 34.3% | −0.198% | −16.1 | 7% | 37% |
| (b) (a) + level-in-way skip (live since 19 Nov) | 7,148 | 18.4 | 35.2% | −0.169% | −11.8 | 7% | 32% |
| (c1) first hour-close detection per 12h = the year test (new default) | 539 | 1.4 | 38.0% | −0.084% | −1.3 | 36% | 1% |
| (c2) (c1) + today's limits (daily stop, streak, direction cap, cooldown) | 399 | 1.0 | 39.1% | −0.051% | −0.8 | 50% | 1% |

Under the live behaviour the steady rule performed like an unfiltered grind (all grinds: 33.0% won, about −0.23%):
checked every 5 minutes it catches coins later and repeatedly, so the year's 38% never applied to the live calls.
Even the year-test behaviour is negative after fees; nothing here passes the bar.
Checks: on the 30 days where every coin's 5-minute candles are stored, 0 of 4,098 R3 detections fell outside the
fetched windows; 300 of 300 random detections gave the same answer when recomputed with the live code's own
functions (`_rolling_hours` + `hourly_shape`, as `scan_now`).

**Reproduction (item 14).** Rebuilt event table: 63,528 moments / 389 days / 519 coins, as published. R3: 539
grinds; every plan's win rate, net and months positive match DESIGN §25 exactly (5x 38.0%, −0.084%, 5/14).
**Mismatch:** the half-year nets come out −0.131% / −0.035% (published −0.11% / −0.01%) and all grinds −0.216% /
−0.257% (published −0.20% / −0.24%); DESIGN §26's table already shows −0.13% / −0.04%, so §25's halves were
probably computed with the earlier maker-fee convention. §27 ([reproduce_runners.txt](reproduce_runners.txt)):
every cell reproduced exactly.

## 3. Studies B1–B3

Pass bar (pre-registered): ≥ 150 trades over ≥ 60 days, ≥ +0.10R per trade after all costs, Holm-corrected
day-clustered t ≥ 3, ≥ 60% months positive; then positive with t ≥ 2 in the locked confirmation period.
Break-even win rates (5x, taker TP; printed by `research/b1_breakouts.py`): +10%/−5% **34.8%**, +15%/−5%
**26.1%**, +20%/−5% **20.9%**, +2%/−1% **40.8%** — as stated in the brief.

### B1 breakouts ([b1_breakouts.txt](b1_breakouts.txt), all 576 variants in [b1_variants.csv](b1_variants.csv))

96,678 breakouts, 13 Oct 2025 – 16 Nov 2026; discovery before 1 Jun 2026. **0 of 576 variants passed.**

| Group (discovery) | Variants | Positive per trade | ≥ +0.10R | Best |
|---|---|---|---|---|
| Long, 5% stop | 144 | 0 | 0 | −0.025R |
| Long, structural stop | 144 | 0 | 0 | −0.034R |
| Short, 5% stop | 144 | 97 | 41 | +0.271R (day-clustered t −1.0) |
| Short, structural stop | 144 | 59 | 7 | +0.174R |

The highest day-clustered t of any variant was +1.02; none reached 2. Shorts were positive per trade but not per
day: a few crash days, when many coins break their lows at once, carry them. Two frozen variants, by period and regime:

| Variant | Period | n | Won (b/e) | Net R | % / trade | 5x margin | t | Months + | / day | vs random-time control |
|---|---|---|---|---|---|---|---|---|---|---|
| 1h 7d vol2x short, structural stop, 4R, 72h | discovery | 1,619 | 30.8% (22.4%) | +0.073 | +0.18% | +0.9% | +1.0 | 43% | 7.7 | −0.333R |
| | confirmation | 1,079 | 20.6% (22.7%) | −0.299 | −0.93% | −4.6% | −1.8 | 17% | 6.7 | −0.624R |
| | alts above 50d | 242 | 31.0% | +0.057 | | | −2.9 | 60% | | |
| | alts below 50d | 2,456 | 26.3% | −0.089 | | | +0.4 | 31% | | |
| 1h 24h vol2x short, 5% stop, 4R, 7d | discovery | 1,425 | 33.5% (20.9%) | +0.080 | +0.40% | +2.0% | +0.7 | 57% | 6.8 | +0.171R (t 2.8) |
| | confirmation | 1,082 | 30.7% (20.9%) | −0.085 | −0.42% | −2.1% | +0.1 | 17% | 6.7 | +0.050R |
| | alts above 50d | 530 | 28.1% | −0.144 | | | +0.8 | 20% | | |
| | alts below 50d | 1,977 | 33.4% | +0.050 | | | +0.4 | 46% | | |

Breakout timing was mostly *worse* than a random hour on the same coin (negative control difference), except the
24h break with a 4R target. Maker TP changes results by less than 0.01R. Drawdowns ran to −100 to −480R.
**Verdict: no breakout rule has an edge.**

### B2 why the big daily movers are not caught ([b2_movers.txt](b2_movers.txt))

4,309 coin-days moved ≥ 15% open to close (2,413 up, 1,896 down).

- **They are caught:** the hour-close logic flagged 3,632 of them (84%), at a median 9h into the day, with a median
  50% of the move to the close still ahead (61% to the day's extreme); 13% were flagged with nothing left.
- **The live 5-minute logic** flags the same mover days a median 95 minutes earlier than the hour close (93% earlier),
  but the 5x plan's problem below is the stop, not the timing.
- **But each comes with 7.7 lookalikes:** 28,068 other coins flagged at the same hour close, same direction, whose day
  moved a median +4.8% that way.

| Group (5x +2%/−1% from the flag) | n | Won | Net / trade | t | Months + |
|---|---|---|---|---|---|
| Movers (picked with hindsight) | 3,632 | 41.1% | +0.010% | −0.3 | 31% |
| Lookalikes | 28,068 | 33.7% | −0.217% | −14.6 | 0% |
| Everything the logic flags | 31,700 | 34.5% | −0.191% | −13.5 | 0% |

| Group (5% stop, 72h, R after costs and funding) | 2R | 3R | 4R |
|---|---|---|---|
| Movers (hindsight) | 63.8% won, +0.860R | 57.2%, +1.184R | 51.7%, +1.337R |
| Lookalikes | 36.6%, −0.073R | 32.3%, −0.119R | 30.8%, −0.158R |
| Everything flagged | 39.7%, +0.034R (t −0.3) | 35.2%, +0.031R (t +0.2) | 33.2%, +0.014R (t +0.1) |

So even the true movers only break even with a 1% stop: the stop is hit by noise before the move. A 5% stop pays
hugely on movers, but because the flag cannot tell movers from lookalikes, the tradable group nets about zero.
By side and regime (5x, everything flagged): longs −0.173%, shorts −0.209%; alts above their 50-day mean
−0.126%, below −0.201%.
**Stop-first check** (600 sampled flags with 1-minute candles): won 38.5% on 5-minute candles vs 39.0% on
1-minute (+0.5 points; 3 of 600 trades flipped from stop to target), about +0.015% per trade. The 5-minute
"stop first" convention barely moves the measured win rates.

### B3 execution costs ([b3_execution.txt](b3_execution.txt))

- **Weex** (official API docs, read 25 Nov 2026): `POST /capi/v3/order` accepts `type=LIMIT` with `reduceOnly=true`
  and `timeInForce=POST_ONLY`; position TP/SL executes at market unless a positive `executePrice` makes it a limit
  (full-position TP/SL: market only); exchange info lists maker 0.02% / taker 0.08%. **Not documented:** whether a
  filled reduce-only limit close is charged the maker rate.
- **The trader's fills** (3 trades, MEGA / ASTER / SKY): every entry and every position TP was charged **0.0798% =
  taker**. Entries were 0.00%, 0.08% better and 0.04% worse than the signal price.
- **1-minute simulation**, 539 steady grinds + 2,000 random grinds, 5x +2% / −1%:

| Sample / variant | Trades | Won | Net / trade | Per signal | t | Months + |
|---|---|---|---|---|---|---|
| Steady (a) market in, taker TP (today) | 539 | 38.4% | −0.073% | −0.073% | −1.15 | 36% |
| Steady (b) limit in (fills 96%), taker TP | 518 | 35.3% | −0.066% | −0.063% | −0.91 | 36% |
| Steady (c) market in, maker TP (through +1 tick) | 539 | 38.0% | −0.061% | −0.061% | −0.95 | 43% |
| Steady (d) limit in, maker TP | 518 | 34.9% | −0.057% | −0.055% | −0.74 | 43% |
| Random (a) | 2,000 | 34.3% | −0.197% | −0.197% | −6.01 | 7% |
| Random (d) limit in (fills 95%), maker TP | 1,898 | 32.2% | −0.142% | −0.135% | −4.44 | 14% |

Adverse selection: filled limits win 3.1 points (steady) and 2.0 points (random) less than all trades; 17 and 64
winners were missed because the limit never filled. Resting orders trim 0.02–0.06% per trade; **0 of 6 variants
pass.** Best §14 sweep variant over the whole year, re-run: taker TP +0.019R, maker TP (through +0.01%) +0.023R,
t ≈ 2.9, 33,485 trades — far below +0.10R.

## 4. B4 status

Code: `lts/sweeps_micro.py` (sweeps × liquidations × order books, maker entries back at the swept level; grid fixed
in the code), runner `research/b4_sweeps.py` (run on the server's database), `lts/leadlag.py`. It refuses to evaluate
before 60 days of liquidation + book data exist: collection started 16–17 Nov 2026, so **discovery can first run
about 16 Jan 2027**, and the confirmation period stays locked for 30 more days (**about 15 Feb 2027**). Stated limits:
the Binance liquidation stream sends at most one event per coin per second; books are snapshotted every 5 minutes for
about 120 coins. Lead-lag measured now ([b4_leadlag.txt](b4_leadlag.txt)): no Binance → Weex lead at 1-minute
resolution (same-minute correlation 0.98–1.00, best lag 0 for 20 of 20 coins).

## 5. What I could not do, and caveats

- **The server's database** (live grinds, signals, liquidations, books) is not reachable from here, so every study
  uses the year's Binance candles and the local copy of hourly/daily/funding history (to 16 Nov 2026). The trader's
  trades were read from the dashboard.
- **The original 5-minute paths** had been deleted with the temporary folder; they were downloaded again from Binance
  (`research_data/`, not committed: 3.3 GB). The original walk-forward model scores used for the "old rule" group in
  §27 survived and were reused, not regenerated.
- **Whether Weex charges maker on a filled reduce-only limit close** is not documented; only a real trade can show it.
- **Proxy data:** Binance candles stand in for Weex prices; live entries happen about 1–2 minutes after the candle
  open used here.
- **Commits:** items 1a, 2–6, 15 and Part C share `grinds_to_record`, `grind_signals` and `SignalEngine.decide`, so
  they landed in one commit, with one test each. One commit (5163d7f) was made while the fee-guard test was failing on
  a quoted string in a print; the next commit (52c7b16) fixed it and the suite passed.
- **B1's first freeze** picked its top 10 by Holm-corrected t alone, and every corrected t was −∞, so the 10 were
  arbitrary. It is kept (`b1_frozen.json`); the corrected pick (by corrected then raw t, discovery only) is
  `b1v2_frozen.json`. No variant passed discovery either way.
- **Not changed:** the trade plans (5x +2% / −1% and the others), and every `live` flag (all off).

## Verdict

Nothing passed. Under the live behaviour the steady rule was just another unfiltered grind (34% won, −0.20% per
trade). Checked the way the year test checked it, it is better (38–39%) but still below the 40.8% that 5x +2% / −1%
needs after fees. Breakouts, limit entries, maker TPs and a hindsight look at the big movers do not change that: the
tight stop is the binding problem, and wider stops only pay on moves that cannot be told apart in advance. With
the new expected-net gate, the bot sends no grind signals on this record.
