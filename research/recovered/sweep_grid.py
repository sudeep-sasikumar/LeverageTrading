"""Pre-declared execution grid for 24h liquidity sweeps: discovery selects, last 120 days confirms (once)."""
import sys, math, time, itertools
sys.path.insert(0, r"D:\Leverage trading")
import numpy as np, pandas as pd
from lts.config import load_config
from lts.db import DB
from lts import sweeps as SW
from lts.grindlab import _hourly, DAY, HOUR
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
cfg = load_config(r"D:\Leverage trading\config.yaml"); db = DB(cfg.path(cfg.storage.db_path))
now = int(time.time() * 1000)
since = now - 400 * DAY
split = now - 120 * DAY
t0 = time.time()
rows = []
for n_sym, r in enumerate(db.query("SELECT DISTINCT symbol FROM klines_1h"), 1):
    k = _hourly(db, r["symbol"], since)
    if k is None:
        continue
    idx = k.index.to_numpy()
    for ev in SW.detect(k):
        ts = int(idx[ev["i"]] + HOUR)
        for ek, sk, tk in itertools.product(SW.ENTRIES, SW.STOPS, SW.TARGETS):
            res = SW.simulate(ev, k, ek, sk, tk)
            if res:
                rows.append({"symbol": r["symbol"], "ts": ts, "side": ev["side"], "rvol": ev["rvol"],
                             "entry": ek, "stop": sk, "target": tk, **res})
    if n_sym % 100 == 0:
        print(f"  {n_sym} symbols, {len(rows):,} simulated trades ({time.time() - t0:.0f}s)", flush=True)
d = pd.DataFrame(rows)
d.to_parquet(OUT + r"\sweep_grid.parquet")
d["period"] = np.where(d.ts < split, "discovery", "confirm")
d["day"] = d.ts // DAY
d["month"] = pd.to_datetime(d.ts, unit="ms").dt.strftime("%Y-%m")

def stats(g):
    daily = g.groupby("day").R.mean()
    t = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))) if len(daily) > 5 and daily.std() > 0 else 0.0
    months = g.groupby("month").R.mean()
    return pd.Series({"n": len(g), "days": len(daily), "win": (g.R > 0).mean(), "mean_R": g.R.mean(),
                      "mean_pct": g.net.mean(), "t_day": t, "months_pos": (months > 0).mean(), "risk_med": g.risk.median(),
                      "reward_med": g.reward.median(), "hold_h_med": g.bars.median()})

rows = []
for flt in SW.FILTERS:
    dd = d if flt == "all" else d[d.rvol >= 2]
    for key, g in dd.groupby(["side", "entry", "stop", "target", "period"]):
        rows.append({"filter": flt, "side": key[0], "entry": key[1], "stop": key[2], "target": key[3], "period": key[4], **stats(g)})
res = pd.DataFrame(rows)
res.to_parquet(OUT + r"\sweep_grid_results.parquet")
disc = res[res.period == "discovery"]
print(f"\nvariants tested in discovery: {len(disc)}")
ok = disc[(disc.n >= 150) & (disc.days >= 60) & (disc.mean_R >= 0.10) & (disc.t_day >= 3.0) & (disc.months_pos >= 0.6)]
print(f"PASSED DISCOVERY: {len(ok)}")
cols = ["filter", "side", "entry", "stop", "target", "n", "days", "win", "mean_R", "mean_pct", "t_day", "months_pos", "risk_med", "reward_med", "hold_h_med"]
print(disc.sort_values("t_day", ascending=False)[cols].head(16).round(3).to_string(index=False))
if len(ok):
    conf = res[res.period == "confirm"]
    m = ok.merge(conf, on=["filter", "side", "entry", "stop", "target"], suffixes=("_disc", "_conf"))
    m["confirmed"] = (m.mean_R_conf > 0) & (m.t_day_conf >= 2.0) & (m.mean_R_conf >= 0.5 * m.mean_R_disc)
    m.to_parquet(OUT + r"\sweep_confirm.parquet")
    print("\nCONFIRMATION (last 120 days, each variant tested once, frozen):")
    print(m[["filter", "side", "entry", "stop", "target", "n_disc", "mean_R_disc", "t_day_disc", "n_conf", "win_conf",
             "mean_R_conf", "mean_pct_conf", "t_day_conf", "months_pos_conf", "risk_med_conf", "confirmed"]].round(3).to_string(index=False))
print(f"\n{time.time() - t0:.0f}s")
