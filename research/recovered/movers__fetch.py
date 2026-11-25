"""Fetch 5m klines (last ~50h) for every Binance USDT perpetual, save as parquet."""
import asyncio, json, sys, time
import aiohttp
import pandas as pd

OUT = sys.argv[1]
END = int(time.time() * 1000)
START = END - 50 * 3600_000


async def main():
    async with aiohttp.ClientSession() as s:
        info = await (await s.get("https://fapi.binance.com/fapi/v1/exchangeInfo")).json()
        syms = [x["symbol"] for x in info["symbols"] if x["quoteAsset"] == "USDT"
                and x["contractType"] == "PERPETUAL" and x["status"] == "TRADING"]
        sem = asyncio.Semaphore(8)
        rows = []

        async def one(sym):
            async with sem:
                out, t = [], START
                while t < END:
                    r = await s.get("https://fapi.binance.com/fapi/v1/klines",
                                    params={"symbol": sym, "interval": "5m", "startTime": t, "limit": 1500})
                    k = await r.json()
                    if not isinstance(k, list) or not k:
                        break
                    out += k
                    t = k[-1][0] + 300_000
                    if len(k) < 1500:
                        break
                for x in out:
                    rows.append((sym, x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[7]), float(x[10])))
        await asyncio.gather(*(one(x) for x in syms))
    df = pd.DataFrame(rows, columns=["symbol", "ts", "o", "h", "l", "c", "qv", "tbqv"])
    df = df[df.ts + 300_000 <= END]
    df.to_parquet(OUT)
    print(len(syms), "symbols", len(df), "bars")


asyncio.run(main())
