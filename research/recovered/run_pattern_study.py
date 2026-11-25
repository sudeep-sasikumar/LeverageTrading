import sys, time, json
sys.path.insert(0, r"D:\Leverage trading")
import numpy as np, pandas as pd
from lts.config import load_config
from lts.db import DB
from lts import patterns as P, pattern_study as S
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
cfg = load_config(r"D:\Leverage trading\config.yaml"); db = DB(cfg.path(cfg.storage.db_path))
DAY = 86_400_000
t0 = time.time()
ev = P.run(db, days=395)
ev.to_parquet(OUT + r"\pattern_events.parquet")
split = int(time.time() * 1000) - S.CONFIRM_DAYS * DAY
print(f"\n{len(ev):,} events from {ev.symbol.nunique()} coins, {pd.to_datetime(ev.ts.min(), unit='ms'):%Y-%m-%d} to "
      f"{pd.to_datetime(ev.ts.max(), unit='ms'):%Y-%m-%d}; split at {pd.to_datetime(split, unit='ms'):%Y-%m-%d} ({time.time()-t0:.0f}s)")
res = P.evaluate(ev, split)
res.to_parquet(OUT + r"\pattern_results.parquet")
print(f"tests run in discovery: {(res.period == 'discovery').sum()}")
chosen = S.select(res)
print(f"\nPASSED DISCOVERY: {len(chosen)}")
if len(chosen):
    conf = S.confirm(res, chosen)
    conf.to_parquet(OUT + r"\pattern_confirm.parquet")
    cols = ["pattern", "side", "exit", "n_disc", "mean_R_disc", "t_day_disc", "months_pos_disc", "n_conf", "mean_R_conf",
            "t_day_conf", "win_conf", "mean_pct_conf", "risk_med_disc", "confirmed"]
    print(conf[cols].round(3).to_string(index=False))
d = res[res.period == "discovery"]
print("\nTop 25 in discovery by day-clustered t (whatever their other stats):")
top = d.sort_values("t_day", ascending=False).groupby(["pattern", "side"]).head(1).head(25)
print(top[["pattern", "side", "exit", "n", "days", "win", "mean_R", "mean_pct", "t_day", "months_pos", "risk_med"]].round(3).to_string(index=False))
print("\nBottom 10 (consistently losing - candidates for the opposite trade):")
print(d.sort_values("t_day").groupby(["pattern", "side"]).head(1).head(10)[["pattern", "side", "exit", "n", "mean_R", "t_day", "months_pos"]].round(3).to_string(index=False))
cs = S.cross_sectional_study(db, 395, split)
cs.to_parquet(OUT + r"\cross_sectional.parquet")
print("\nCross-sectional (daily rebalanced, top/bottom 10% by past return), discovery:")
cd = cs[cs.period == "discovery"].sort_values("t", ascending=False)
print(cd.head(12).round(4).to_string(index=False))
print(f"\ntotal {time.time()-t0:.0f}s")
