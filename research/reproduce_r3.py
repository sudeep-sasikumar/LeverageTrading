"""P0 (docs/HYPOTHESES.md): reproduce DESIGN.md section 25's R3 result and write lts/models/steady_history.json.

Run: python -m research.reproduce_r3  ->  docs/results/reproduce_r3.txt
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.common import EVENTS, Tee, halves, load_paths, r3_mask

from lts import grindsignal as GS
from lts import validate as V
from lts.config import Config
from lts.fees import fees

out = Tee("reproduce_r3.txt")
cfg = Config()
gs = cfg.grindsignal
ev = pd.read_parquet(EVENTS).reset_index(drop=True)
post, _ = load_paths(ev[["symbol", "ts"]], 289)
ok = np.isfinite(post[:, :, 1]).mean(1) >= 0.98
out(f"{len(ev):,} events; with a full 24h of 5-minute candles: {ok.sum():,} (published analysis: 63,528 with paths)")
R3 = r3_mask(ev)
out(f"R3 (all-day steady): {R3.sum()} events (published 539)")
side = ev.side.to_numpy()
res = GS.plan_outcomes(post, side, gs.plans, gs.hours)
disc, month = halves(ev.ts.to_numpy())
ts = ev.ts.to_numpy()

PUB = {"5x +2.00% / -1.00%": ("38.0%", "-0.084%", "5/14"), "10x +1.00% / -0.50%": ("34.3%", "-0.211%", "1/14"),
       "3x +3.33% / -1.00%": ("26.3%", "-0.089%", "7/14"), "2x +5.00% / -0.75%": ("15.2%", "-0.113%", "6/14")}
record = {}
for pl in gs.plans:
    name = GS.plan_name(pl)
    net = res[GS.plan_label(pl)]
    m = R3 & np.isfinite(net)
    st = V.stats(net[m], ts[m])
    st_r = V.stats(net[m] / pl["stop"], ts[m])
    pub = PUB.get(name, ("?", "?", "?"))
    out(f"  {name}: n {st['n']} won {st['win']:.1%} net {st['mean']:+.3%} per trade, t {st['t_day']:+.2f}, "
        f"months positive {round(st['months_pos'] * st['n_months'])}/{st['n_months']}   (published: won {pub[0]}, {pub[1]}, {pub[2]})")
    record[name] = {"n": st["n"], "n_days": st["days"], "win": st["win"], "net": st["mean"], "se_day": st["se_day"],
                    "t_day": st["t_day"], "months_pos": st["months_pos"], "n_months": st["n_months"],
                    "net_r": st_r["mean"], "per_day": st["per_day"]}
pl5 = gs.plans[gs.primary]
net5 = res[GS.plan_label(pl5)]
for lab, m in (("R3 first 7 months", R3 & disc), ("R3 last 7 months", R3 & ~disc),
               ("all grinds first 7 months", disc), ("all grinds last 7 months", ~disc)):
    x = net5[m & np.isfinite(net5)]
    out(f"  5x {lab}: n {len(x):,} won {(x > 0).mean():.1%} net {x.mean():+.3%}")
out("  (published: R3 36.5% / -0.11% and 39.6% / -0.01%; all grinds 33.7% / -0.20% and 32.3% / -0.24%)")

# pass bar for the record (family: the ten rules of section 25)
st_r = V.stats(net5[R3 & np.isfinite(net5)] / pl5["stop"], ts[R3 & np.isfinite(net5)])
t_corr = V.holm([st_r["t_day"]] + [0.0] * 9)[0]
passed, why = V.passes_discovery(st_r, t_corr)
out(f"  pass bar (R units, Holm over the 10 rules): {'PASSED' if passed else 'not passed: ' + '; '.join(why)}")

hist_path = GS.HISTORY_PATH
h = GS.load_history() or {}
h.update({"generated_by": "research/reproduce_r3.py",
          "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
          "fees": {"taker": fees().taker, "maker": fees().maker, "slippage": fees().slip}})
h.setdefault("rules", {})
h["rules"]["steady/hour_close_first"] = {
    "source": "steady rule checked at the first hour-close detection per 12h, past year (DESIGN.md section 25)",
    "data": "63,528 early grinds 22 Oct 2025 - 16 Nov 2026, 5m candles, 24h, stop first, config.trade costs",
    "plans": record, "passed_bar": bool(passed), "bar_failures": why, "script": "research/reproduce_r3.py"}
hist_path.write_text(json.dumps(h, indent=1), encoding="utf-8")
out(f"wrote {hist_path.name} (rule steady/hour_close_first)")
