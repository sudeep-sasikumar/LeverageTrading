"""B4 feature candidate: Binance -> Weex lead-lag on recent 1-minute candles (descriptive; lts/leadlag.py).
The 20 most liquid coins on both venues, the last 48 hours. Run: python -m research.b4_leadlag"""
import asyncio
import time

import numpy as np

from research.common import Tee

from lts.exchanges import Binance, Weex
from lts.leadlag import lead_lag

out = Tee("b4_leadlag.txt")
MIN = 60_000


async def main():
    b, w = Binance(), Weex()
    try:
        bt, wt = await b.tickers(), await w.tickers()
        both = sorted((s for s in bt if s in wt and s.endswith("USDT")), key=lambda s: -(bt[s]["qv24"] or 0))[:20]
        end = int(time.time() * 1000) // MIN * MIN
        start = end - 48 * 60 * MIN
        out(f"Binance -> Weex lead-lag, 1-minute candles {time.strftime('%Y-%m-%d %H:%M', time.gmtime(start / 1000))} to "
            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(end / 1000))} UTC, 20 most liquid common coins")
        out("corr(k) = correlation of Weex returns with Binance returns k minutes earlier")
        rows = []
        for s in both:
            kb = await b.klines_range(s, "1m", start, end)
            kw = await w.klines_range(s, "1m", start, end)
            r = lead_lag(kb, kw, 3)
            c = r["corr"]
            rows.append(r)
            out(f"  {s:<12} minutes {r['n']:>5}  corr k=0 {c.get(0, np.nan):+.3f}  k=1 {c.get(1, np.nan):+.3f}  "
                f"k=2 {c.get(2, np.nan):+.3f}  best lag {r['best_lag']}")
        lag1 = np.array([r["corr"].get(1, np.nan) for r in rows])
        out(f"median corr at k=1: {np.nanmedian(lag1):+.3f}; coins whose best lag is > 0: "
            f"{sum(1 for r in rows if (r['best_lag'] or 0) > 0)} of {len(rows)}")
    finally:
        await b.close()
        await w.close()

asyncio.run(main())
