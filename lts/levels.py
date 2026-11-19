"""Major support/resistance levels (daily, weekly, monthly) between a trade's entry and its target or stop.

Levels come only from candles that had closed before the decision time:
  daily   : previous day's high and low; daily swing highs/lows (highest/lowest of 3 days each side), last 180 days
  weekly  : previous week's high and low; weekly swing highs/lows (2 weeks each side), last 104 weeks
  monthly : previous month's high and low; monthly swing highs/lows (1 month each side), last 36 months
Highs and lows both count (an old support above price acts as resistance, and the other way round).
Weeks start Monday 00:00 UTC. Same definitions as the year study in docs/DESIGN.md section 23.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86_400_000
TF = {"daily": (3, 180), "weekly": (2, 104), "monthly": (1, 36)}


def _bars(d: pd.DataFrame) -> dict:
    """d: daily candles with columns ts, h, l (ts = open time, ms). Returns {tf: frame(start, end, h, l, label)}."""
    d = d.sort_values("ts")
    out = {"daily": pd.DataFrame({"start": d.ts.to_numpy(), "end": d.ts.to_numpy() + DAY, "h": d.h.to_numpy(),
                                  "l": d.l.to_numpy()})}
    for tf, key in (("weekly", (d.ts // DAY + 3) // 7), ("monthly", pd.to_datetime(d.ts, unit="ms").dt.strftime("%Y-%m"))):
        g = d.groupby(key.to_numpy())
        out[tf] = pd.DataFrame({"start": g.ts.min(), "end": g.ts.max() + DAY, "h": g.h.max(), "l": g.l.min()}) \
            .sort_values("start").reset_index(drop=True)
    return out


def major_levels(d: pd.DataFrame, t: int) -> list[dict]:
    """Every major level known at time t: {price, tf, what}."""
    if d is None or len(d) < 3:
        return []
    d = d.reset_index() if "ts" not in d.columns else d
    out = []
    for tf, b in _bars(d[["ts", "h", "l"]]).items():
        k, look = TF[tf]
        closed = b[b.end <= t].reset_index(drop=True)
        if not len(closed):
            continue
        last = closed.iloc[-1]
        name = {"daily": "previous day", "weekly": "previous week", "monthly": "previous month"}[tf]
        out += [{"price": float(last.h), "tf": tf, "what": f"{name} high"},
                {"price": float(last.l), "tf": tf, "what": f"{name} low"}]
        h, l = closed.h.to_numpy(), closed.l.to_numpy()
        n = len(closed)
        for i in range(max(k, n - look), n - k):          # a swing is known once k later candles have closed
            if h[i] == h[i - k:i + k + 1].max():
                out.append({"price": float(h[i]), "tf": tf, "what": f"{tf} swing high"})
            if l[i] == l[i - k:i + k + 1].min():
                out.append({"price": float(l[i]), "tf": tf, "what": f"{tf} swing low"})
    return out


def between(levels: list[dict], entry: float, side: int, target: float, stop: float) -> dict:
    """Levels in the way of the target, and levels between entry and the stop (which may hold the stop).
    Each item carries its distance from entry in the trade's direction (+ toward the target)."""
    block, cushion = [], []
    for lv in levels:
        dist = side * (lv["price"] / entry - 1)
        item = {**lv, "dist": dist}
        if 0 < dist <= target:
            block.append(item)
        elif -stop <= dist < 0:
            cushion.append(item)
    rank = {"monthly": 0, "weekly": 1, "daily": 2}
    key = lambda x: (rank[x["tf"]], abs(x["dist"]))
    return {"block": sorted(block, key=key), "cushion": sorted(cushion, key=key),
            "block_tf": sorted({x["tf"] for x in block}, key=rank.get),
            "cushion_tf": sorted({x["tf"] for x in cushion}, key=rank.get)}


def describe(lv: dict, fmt=lambda p: f"{p:g}") -> str:
    return f"{lv['what']} {fmt(lv['price'])} ({lv['dist'] * 100:+.2f}%)"
