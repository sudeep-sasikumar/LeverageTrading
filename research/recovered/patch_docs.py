import pathlib

p = pathlib.Path(r"D:\Leverage trading\docs\DESIGN.md")
s = p.read_text(encoding="utf-8")
pairs = [
("Status: v1 design. Section 11 (empirical calibration) is filled in from real backfilled data.",
 "Status: v1, implemented. Section 11 records what 30 days of real data showed — read it\n"
 "before trusting any signal."),

("- **Tradable**: liquidity floor, ATR% within bounds, enough history.",
 "- **Tradable**: liquidity floor, enough history, and 5m ATR between 0.5% and 1.25% — low\n"
 "  enough for a stop of at most 2.5%, high enough that the stop is at least 1% (below\n"
 "  that, fees and slippage are more than a quarter of the risk; see section 11)."),

("stop%      = clamp(sl_mult × ATR%(5m,14), 0.6%, 2.5%)      (skip if above 2.5%)",
 "stop%      = clamp(2 × ATR%(5m,14), 1.0%, 2.5%)            (skip if above 2.5%)"),

("Fees are modelled from Weex VIP0 (taker 0.08%, maker 0.02%) plus slippage. At 10x,\n"
 "fees alone are ~1.6–2% of margin per round trip, so they are part of every label.",
 "Fees are modelled from Weex VIP0 (taker 0.08%, maker 0.02%) plus slippage, and are part\n"
 "of every label. With stops of 1–2.5% the derived leverage is typically 3–6x; 10x is a\n"
 "cap, not a default. \"Tight stop\" is enforced where it matters — 1% of the account at\n"
 "risk per trade — rather than as a tiny price distance that fees would dominate.\n"
 "Signals recommend a **limit entry** at the signal price: it roughly halves fee drag."),

("Signal frequency: grade **A** signals fire whenever calibrated probability ≥ the\n"
 "learned threshold.",
 "Signal frequency: grade **A** signals fire whenever the model's ranking score ≥ the\n"
 "learned threshold."),

("5. **Promote** a challenger to live signals only if it shows same-moment selection\n"
 "   skill, positive out-of-sample expectancy at its threshold, positive expectancy on\n"
 "   the untouched most-recent holdout, and is not worse than the champion on candidates\n"
 "   the champion has never seen. A model that fails becomes the *shadow* model and\n"
 "   drives paper signals only.\n"
 "6. **Calibrate** probabilities (isotonic) and re-pick the threshold that maximises\n"
 "   expectancy subject to the signal-frequency target.\n"
 "7. **Guard**: live win rate/expectancy is compared to the expected value; sustained\n"
 "   under-performance raises the threshold or pauses signalling and alerts on Telegram.\n"
 "8. **Report**: daily Telegram summary; weekly feature-importance and barrier-grid\n"
 "   report so parameter changes are a reviewed decision, not silent drift.",
 "5. **Promote** a challenger to real-money signals only if **all** of these hold:\n"
 "   - same-moment selection skill ≥ 0.03;\n"
 "   - out-of-sample policy expectancy positive *and significant* (t ≥ 2 over ≥ 40\n"
 "     trades — the threshold is the best of a dozen tried, so a small positive average\n"
 "     is not evidence);\n"
 "   - positive expectancy on the untouched most-recent holdout;\n"
 "   - a **forward record** of at least 20 closed paper signals with positive average R;\n"
 "   - not worse than the champion on candidates the champion has never seen.\n"
 "   A model that fails any gate becomes the *shadow* model and drives paper signals only.\n"
 "6. **Threshold**: chosen on the model's raw ranking score to maximise expectancy subject\n"
 "   to the signal-frequency target; probabilities are isotonic-calibrated for display\n"
 "   and expected-value checks.\n"
 "7. **Guard**: a champion whose recent live record falls below −0.10R per signal (over ≥\n"
 "   20 signals) is demoted automatically and the bot returns to paper mode. Safe mode\n"
 "   (stricter threshold) applies after a −4R week.\n"
 "8. **Report**: daily Telegram summary and nightly learning report; `python -m lts\n"
 "   research` prints base rates and the stop/target grid so parameter changes are a\n"
 "   reviewed decision, not silent drift."),

("## 11. Empirical calibration\n\n_Filled in after the first backfill — see the end of this document._\n",
 """## 11. What 30 days of real data showed (13 Oct – 12 Nov 2026)

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
"""),
]
for a, b in pairs:
    assert a in s, a[:60]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")

p = pathlib.Path(r"D:\Leverage trading\README.md")
s = p.read_text(encoding="utf-8")
pairs = [
("""## What to expect

- This is a statistical edge-finder, not a guarantee. A 2:1 trade needs roughly a 40%
  hit rate after fees to be profitable; many signals will lose.
- Capital preservation outranks signal frequency. On days where nothing has positive
  expected value the bot sends a "no-trade day" note instead of a weak signal.
- The first model is trained on 30 days of history (one market regime). Treat the first
  weeks as paper trading and judge it on the live record in `/stats`.
""",
"""## What to expect

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
"""),
("""Entry  0.4105  (market, valid 10 min)""", """Entry  0.4105  (limit order preferred; valid 10 min)"""),
("""| Promote | nightly | new model replaces the old only if it validates and is not worse on fresh data |
| Guard | continuous | daily stop, loss-streak stop, safe mode after a bad week |""",
"""| Promote | nightly | a model goes live only with significant out-of-sample results, a positive holdout and a positive 20-signal forward record |
| Guard | continuous | daily stop, loss-streak stop, safe mode after a bad week, automatic demotion to paper if the live record turns negative |"""),
]
for a, b in pairs:
    assert a in s, a[:60]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("docs updated")
