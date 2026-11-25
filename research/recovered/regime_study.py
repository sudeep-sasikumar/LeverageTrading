"""Pre-declared follow-up: trade each pattern only in the direction of the altcoin market regime.
Regime on day D = sign(equal-weight alt index vs its 50-day SMA), computed from closes up to D-1.
Benchmark: hold the regime direction on an equal-weight basket with no pattern (daily, net of costs)."""
import sys, time, math
sys.path.insert(0, r"D:\Leverage trading")
import numpy as np, pandas as pd
from lts.config import load_config
from lts.db import DB
from lts import patterns as P, pattern_study as S
from lts.grindlab import _hourly, DAY
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
cfg = load_config(r"D:\Leverage trading\config.yaml"); db = DB(cfg.path(cfg.storage.db_path))
now = int(time.time() * 1000)
since = now - 400 * DAY
closes, qvs = {}, {}
for r in db.query("SELECT DISTINCT symbol FROM klines_1h"):
    k = _hourly(db, r["symbol"], since)
    if k is None:
        continue
    d = P.daily_from_hourly(k)
    closes[r["symbol"]], qvs[r["symbol"]] = d["c"], d["qv"]
C, Q = pd.DataFrame(closes).sort_index(), pd.DataFrame(qvs).sort_index()
rets = C.pct_change()
liquid = Q >= P.MIN_QV24
alt = rets.where(liquid).drop(columns=["BTCUSDT", "ETHUSDT"], errors="ignore").mean(axis=1)    # equal-weight alt index
index = (1 + alt.fillna(0)).cumprod()
regime_d = np.sign(index - index.rolling(50).mean()).shift(1)          # known at the start of day D
split = now - S.CONFIRM_DAYS * DAY
print("alt index change over the window:", f"{index.iloc[-1] / index.iloc[50] - 1:+.1%}",
      "| days in down-regime:", f"{(regime_d < 0).mean():.0%}",
      "| discovery down-days:", f"{(regime_d[regime_d.index < split] < 0).mean():.0%}",
      "| confirm down-days:", f"{(regime_d[regime_d.index >= split] < 0).mean():.0%}")

# benchmark: each day hold the regime direction on an equal-weight basket of liquid alts (taker in/out daily)
cost = 2 * (P.TAKER + P.SLIP)
bench = (regime_d * alt.shift(-1) - cost).dropna()          # decided at day start, earn that day's return
bench = (regime_d * alt - cost).dropna()
for per, b in (("discovery", bench[bench.index < split]), ("confirm", bench[bench.index >= split])):
    print(f"benchmark (regime direction, no pattern) {per}: {len(b)} days, mean {b.mean() * 100:+.3f}%/day, "
          f"t={b.mean() / (b.std(ddof=1) / math.sqrt(len(b))):+.2f}, days won {(b > 0).mean():.0%}")

ev = pd.read_parquet(OUT + r"\pattern_events.parquet")
ev["day0"] = ev.ts // DAY * DAY
ev["regime"] = ev.day0.map(regime_d)
aligned = ev[ev.side == ev.regime]
res = P.evaluate(aligned, split)
chosen = S.select(res)
print(f"\nRegime-aligned patterns: tests {(res.period == 'discovery').sum()}, passed discovery: {len(chosen)}")
d = res[res.period == "discovery"]
top = d.sort_values("t_day", ascending=False).groupby(["pattern", "side"]).head(1).head(15)
print(top[["pattern", "side", "exit", "n", "days", "win", "mean_R", "mean_pct", "t_day", "months_pos", "risk_med"]].round(3).to_string(index=False))
if len(chosen):
    conf = S.confirm(res, chosen)
    conf.to_parquet(OUT + r"\regime_confirm.parquet")
    print("\nCONFIRMATION (last 120 days, frozen exit):")
    print(conf[["pattern", "side", "exit", "n_disc", "mean_R_disc", "t_day_disc", "n_conf", "mean_R_conf", "t_day_conf",
                "win_conf", "mean_pct_conf", "risk_med_disc", "confirmed"]].round(3).to_string(index=False))
# does the pattern add anything beyond the regime? compare each aligned pattern-day to the benchmark on the same days
print("\nPattern vs plain regime on the same days (discovery, 3R/72h exits, shorts in down-regime):")
for pat, g in aligned[(aligned.ts < split) & (aligned.side < 0)].groupby("pattern"):
    days = g.day0.unique()
    if len(days) < 60:
        continue
    b = bench.reindex(days).dropna()
    print(f"  {pat:<20} trades {len(g):6}  pattern {g['pct_3R_72h'].mean() * 100:+.2f}% per trade (72h)   "
          f"plain short basket on those days {b.mean() * 100 * 3:+.2f}% per 3 days")
