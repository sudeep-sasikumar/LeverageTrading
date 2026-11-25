"""Every 5-minute candle the year studies need, fetched once into research_data/candles.sqlite (table k5):

A. each early-grind moment: 6h before (minute shape) to 24h after (the plan) - the original paths/pre chunks;
B. the live-parity replay: around every hour close where a coin was near the steady rule on hourly bars
   (side-aligned 24h move >= 6%, 24h R^2 >= 0.6, >= 12 of 24 hours its way, 6h move >= 2%; the rule itself needs
   8% / 0.8 / 15 / 3%), from 32h before (30 rolling hours need 30h of 5m candles) to 26h after;
C. every big daily mover (a UTC day with a >= 15% move, either way, on daily candles), 32h before the day to 26h
   after it (study B2).
The hourly prefilter in B is checked against a full replay of the last 30 days (research/parity.py).
"""
import asyncio

import numpy as np
import pandas as pd

from research.common import DAY, EVENTS, HOUR, LOCAL_DB, YEAR_NOW, Tee
from research.fetch_candles import run

from lts import grindlab as G
from lts.db import DB

out = Tee("fetch_year_5m.txt")
ev = pd.read_parquet(EVENTS)
ranges = [(s, int(t) - 6 * HOUR, int(t) + 24 * HOUR + 300_000) for s, t in zip(ev.symbol, ev.ts)]
out(f"A. {len(ranges):,} early-grind moments")

db = DB(LOCAL_DB)
start = YEAR_NOW - 390 * DAY
syms = sorted(ev.symbol.unique())
nb = nc = 0
for sym in syms:
    k = G._hourly(db, sym, start - 31 * DAY)
    if k is None:
        continue
    f = G.hourly_shape(k)
    t_close = k.index.to_numpy() + HOUR
    for side in (1, -1):
        hw = (f["up24"] if side > 0 else f["dn24"]).to_numpy()
        with np.errstate(invalid="ignore"):
            near = ((side * f["r_24h"].to_numpy() >= 0.06) & (f["r2_24h"].to_numpy() >= 0.6) & (hw >= 12) &
                    (side * f["r_6h"].to_numpy() >= 0.02) & (t_close >= start) & (t_close <= YEAR_NOW))
        for t in t_close[near]:
            ranges.append((sym, int(t) - 32 * HOUR, int(t) + 26 * HOUR))
            nb += 1
    d = G._daily(db, sym)
    d = d[(d.index >= start) & (d.index < YEAR_NOW)]
    mv = (d["c"] / d["o"] - 1).abs()
    for t in d.index[mv >= 0.15]:
        ranges.append((sym, int(t) - 32 * HOUR, int(t) + DAY + 26 * HOUR))
        nc += 1
out(f"B. {nb:,} coin-hours near the steady rule; C. {nc:,} big-mover coin-days ({len(syms)} coins)")
asyncio.run(run(ranges, "5m", log=out))
