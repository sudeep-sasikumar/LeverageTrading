"""Binance -> Weex lead-lag (feature candidate for study B4).

For one coin over a window: the correlation of Weex 1-minute returns with Binance 1-minute returns k minutes
EARLIER (k = 0..max_lag). If Binance leads, the correlation at k = 1 (or more) is clearly above zero, and the
best lag is > 0. Candles are matched on their open time; minutes missing on either venue are dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lead_lag(binance: list[tuple], weex: list[tuple], max_lag: int = 5) -> dict:
    """binance / weex: kline rows (ts, o, h, l, c, ...). Returns {"corr": {k: r}, "best_lag": k, "n": minutes}."""
    b = pd.Series({int(r[0]): float(r[4]) for r in binance}).sort_index()
    w = pd.Series({int(r[0]): float(r[4]) for r in weex}).sort_index()
    idx = b.index.intersection(w.index)
    if len(idx) < 60:
        return {"corr": {}, "best_lag": None, "n": int(len(idx))}
    full = np.arange(idx.min(), idx.max() + 60_000, 60_000)
    rb = np.log(b.reindex(full)).diff()
    rw = np.log(w.reindex(full)).diff()
    corr = {}
    for k in range(max_lag + 1):
        x, y = rb.shift(k), rw
        m = x.notna() & y.notna()
        corr[k] = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 30 and x[m].std() > 0 and y[m].std() > 0 else np.nan
    finite = {k: v for k, v in corr.items() if np.isfinite(v)}
    return {"corr": corr, "best_lag": max(finite, key=finite.get) if finite else None, "n": int(len(idx))}
