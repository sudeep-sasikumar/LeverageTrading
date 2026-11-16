"""Run the pre-registered pattern study: discovery on the older ~9 months, confirmation on the last ~4.

Decision rules (fixed before results were seen):
  discovery pass : n >= 150, days >= 60, mean R >= +0.10 after fees, day-clustered t >= 3.0,
                   >= 60% of months positive. Best exit per pattern/side is chosen here.
  confirmation   : that single frozen exit on the last 120 days: mean R > 0, t >= 2.0,
                   and at least half the discovery mean R.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from . import patterns as P
from .db import DB
from .grindlab import DAY, HOUR, _hourly

CONFIRM_DAYS = 120
RULES = {"min_n": 150, "min_days": 60, "min_R": 0.10, "min_t": 3.0, "min_months_pos": 0.60,
         "confirm_t": 2.0, "confirm_keep": 0.5}


def select(res: pd.DataFrame) -> pd.DataFrame:
    d = res[res.period == "discovery"]
    ok = d[(d.n >= RULES["min_n"]) & (d.days >= RULES["min_days"]) & (d.mean_R >= RULES["min_R"]) &
           (d.t_day >= RULES["min_t"]) & (d.months_pos >= RULES["min_months_pos"])]
    return ok.sort_values("t_day", ascending=False).groupby(["pattern", "side"]).head(1)


def confirm(res: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    c = res[res.period == "confirm"]
    out = chosen.merge(c, on=["pattern", "side", "exit"], suffixes=("_disc", "_conf"), how="left")
    out["confirmed"] = (out.mean_R_conf > 0) & (out.t_day_conf >= RULES["confirm_t"]) & \
                       (out.mean_R_conf >= RULES["confirm_keep"] * out.mean_R_disc)
    return out


def cross_sectional_study(db: DB, days: float, split_ts: int) -> pd.DataFrame:
    now = int(time.time() * 1000)
    since = now - int(days * DAY)
    closes, qvs = {}, {}
    for r in db.query("SELECT DISTINCT symbol FROM klines_1h"):
        k = _hourly(db, r["symbol"], since)
        if k is None:
            continue
        d = P.daily_from_hourly(k)
        closes[r["symbol"]], qvs[r["symbol"]] = d["c"], d["qv"]
    C, Q = pd.DataFrame(closes).sort_index(), pd.DataFrame(qvs).sort_index()
    cs = P.cross_sectional(C, Q)
    rows = []
    for (lb, hold), g in cs.groupby(["lookback", "hold"]):
        g = g.iloc[::hold]                                  # non-overlapping holding periods for honest t-stats
        for per, gg in (("discovery", g[g.day < split_ts]), ("confirm", g[g.day >= split_ts])):
            for strat in ("momentum", "reversal", "long_top", "short_bottom", "long_bottom", "short_top"):
                x = gg[strat].dropna()
                if len(x) < 10:
                    continue
                rows.append({"strategy": strat, "lookback_d": lb, "hold_d": hold, "period": per, "n_periods": len(x),
                             "mean_pct": float(x.mean()), "t": float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))),
                             "win": float((x > 0).mean())})
    return pd.DataFrame(rows)


def leverage_view(row: dict, lev: int) -> dict:
    """What a pattern means per trade at a given leverage (1% of the account risked per trade)."""
    risk = row["risk_med_disc"] if "risk_med_disc" in row else row["risk_med"]
    mean_pct = row.get("mean_pct_conf", row.get("mean_pct"))
    margin_per_trade = lev * mean_pct
    liq_distance = 1.0 / lev
    return {"leverage": lev, "avg_margin_return": margin_per_trade, "stop_vs_liquidation": risk / liq_distance,
            "account_per_trade_at_1pct_risk": row.get("mean_R_conf", row.get("mean_R")) * 0.01}
