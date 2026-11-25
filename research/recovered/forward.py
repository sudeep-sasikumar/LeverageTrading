"""Forward test: candidates after the saved model's training cut-off, which it has never seen."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, r"D:\Leverage trading")
from lts.config import load_config
from lts.db import DB
from lts import learner as L

cfg = load_config(r"D:\Leverage trading\config.yaml"); db = DB(cfg.path(cfg.storage.db_path))
b = L.load_bundle(cfg, db)
df = L.load_labelled(db)
cut = b["data_until_ts"]
old, new = df[df.exit_ts <= cut], df[df.ts > cut].copy()
DAY = 86_400_000
print(f"model {b['version']} ({b['status']}), trained on data to {pd.to_datetime(cut, unit='ms')} UTC")
print(f"unseen candidates: {len(new)} over {(new.ts.max() - new.ts.min()) / DAY:.1f} days")

def line(name, d):
    if len(d) == 0:
        return f"  {name:<28} n=0"
    return f"  {name:<28} n={len(d):>5}  target-first {d.label.mean():6.1%}  avg {d.ret_r.mean():+.3f}R"

for title, d in (("TRAINING PERIOD (to 12 Nov)", old), ("UNSEEN DAYS (since 12 Nov)", new)):
    print("\n" + title)
    print(line("all", d))
    print(line("longs", d[d.side > 0])); print(line("shorts", d[d.side < 0]))
    print(line("stop >= 2%", d[d.sl_pct >= 0.02])); print(line("stop < 1.3%", d[d.sl_pct < 0.013]))
    imp = d["impulse_same_1h"]
    print(line("moderate market impulse", d[(imp > imp.quantile(.5)) & (imp <= imp.quantile(.95))]))
    print(line("isolated move (low impulse)", d[imp <= imp.quantile(.5)]))
    print(line("against BTC 3-day trend", d[d.btc_r_3d < 0])); print(line("with BTC 3-day trend", d[d.btc_r_3d > 0]))

x = new.reindex(columns=b["features"]).to_numpy()
new["score"] = L.raw_score(b["model"], x)
top = new[new.score >= new.score.quantile(0.9)]
print("\nMODEL ON UNSEEN DAYS")
print(line("model top 10%", top))
print(line("model bottom 50%", new[new.score <= new.score.median()]))
pol = L.simulate_policy(new.sort_values("ts"), "score", b["threshold"], cfg)
print(f"  signal policy at its threshold: {pol['n']} signals, win {pol['win']:.0%}, avg {pol['avg_r']:+.2f}R, "
      f"total {pol['total_r']:+.1f}R, max drawdown {pol['max_dd_r']:.1f}R")
for _, s in new[new.score >= b["threshold"]].sort_values("ts").iterrows():
    pass
