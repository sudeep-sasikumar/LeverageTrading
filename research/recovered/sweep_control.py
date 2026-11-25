"""Matched-control test: is the sweep edge the sweep, or just shorting in a falling market?
For each sweep event: 5 control trades on the same coin, same side, same stop size, same exits,
entered at random hours within +/-3 days (excluding +/-6h around the event). Zero-fee and net results."""
import sys, math, time
sys.path.insert(0, r"D:\Leverage trading")
import numpy as np, pandas as pd
from lts.config import load_config
from lts.db import DB
from lts import patterns as P
from lts.grindlab import _hourly, DAY, HOUR
pd.set_option("display.width", 220)
OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
cfg = load_config(r"D:\Leverage trading\config.yaml"); db = DB(cfg.path(cfg.storage.db_path))
ev = pd.read_parquet(OUT + r"\pattern_events.parquet")
split = ev.ts.max() - 116 * DAY
TEST = [("sweep_24h", -1), ("sweep_24h_vol", -1), ("sweep_prevday", -1), ("sweep_24h", 1), ("sweep_equal", -1)]
EXIT = "3R_72h"
max_cost = 2 * (P.TAKER + P.SLIP)
rng = np.random.default_rng(42)
since = int(time.time() * 1000) - 400 * DAY
sub = ev[ev.set_index(["pattern", "side"]).index.isin(TEST)]
rows = []
for sym, g in sub.groupby("symbol"):
    k = _hourly(db, sym, since)
    if k is None:
        continue
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    idx = k.index.to_numpy()
    pos = {t: j for j, t in enumerate(idx)}
    for _, e in g.iterrows():
        j = pos.get(int(e.ts - HOUR))                          # decision bar of the event
        if j is None:
            continue
        cands = [x for x in range(j - 72, j + 73) if abs(x - j) > 6 and 0 < x < len(c) - 80]
        for x in rng.choice(cands, size=min(5, len(cands)), replace=False):
            entry = o[x + 1]
            stop = entry * (1 + e.side * -1 * -e.risk) if False else entry * (1 - e.side * e.risk)
            r = P.simulate(e.side, entry, stop, x + 1, h, l, c, o, P.TAKER + P.SLIP)
            if r:
                rows.append({"pattern": e.pattern, "side": e.side, "ts": e.ts, "control": True, "R": r[EXIT], "risk": r["risk"]})
        rows.append({"pattern": e.pattern, "side": e.side, "ts": e.ts, "control": False, "R": e[EXIT], "risk": e.risk})
d = pd.DataFrame(rows)
d["gross"] = d.R + max_cost / d.risk
d["period"] = np.where(d.ts < split, "discovery", "confirm")
out = []
for (pat, side, per), g in d.groupby(["pattern", "side", "period"]):
    ev_, ct = g[~g.control], g[g.control]
    diff_by_day = (ev_.groupby(ev_.ts // DAY).gross.mean() - ct.groupby(ct.ts // DAY).gross.mean()).dropna()
    out.append({"pattern": pat, "side": side, "period": per, "n_events": len(ev_), "event_gross_R": ev_.gross.mean(),
                "control_gross_R": ct.gross.mean(), "edge_over_control": diff_by_day.mean(),
                "t_edge": diff_by_day.mean() / (diff_by_day.std(ddof=1) / math.sqrt(len(diff_by_day))) if len(diff_by_day) > 5 else np.nan,
                "event_net_R": ev_.R.mean()})
print(pd.DataFrame(out).round(3).to_string(index=False))
