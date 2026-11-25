"""5m candles WITH volume and taker-buy volume for 24h after a sample of the year's grinds (for flow-based exits).
Sample: every all-day-steady grind (R3) + 4,000 random others (fixed seed). Slow and steady (Binance weight)."""
import asyncio, os, sys, time
import aiohttp
import numpy as np, pandas as pd

OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
DEST = OUT + r"\flow_paths.npz"
a = pd.read_parquet(OUT + r"\aim_events.parquet").reset_index().rename(columns={"index": "row"})
R3 = (a.r_24h >= 0.08) & (a.r2_24h >= 0.8) & (a.hours_with24 >= 15) & (a.r2_6h >= 0.9) & (a.hours_with6 >= 5) & (a.big6 <= 0.35)
rng = np.random.default_rng(11)
others = a.index[~R3].to_numpy()
pick = np.sort(np.concatenate([a.index[R3].to_numpy(), rng.choice(others, 4000, replace=False)]))
ev = a.loc[pick, ["row", "symbol", "ts", "side"]].reset_index(drop=True)
B = 289
arr = np.full((len(ev), B, 6), np.nan, np.float32)        # o,h,l,c,qv,tbqv
done = np.zeros(len(ev), bool)
if os.path.exists(DEST):
    z = np.load(DEST)
    arr, done = z["arr"], z["done"]


async def main():
    async with aiohttp.ClientSession() as s:
        n = 0
        for i, r in ev.iterrows():
            if done[i]:
                continue
            for attempt in range(5):
                resp = await s.get("https://fapi.binance.com/fapi/v1/klines",
                                   params={"symbol": r.symbol, "interval": "5m", "startTime": int(r.ts), "limit": B})
                if resp.status in (418, 429):
                    print("rate limited, pausing", resp.status, flush=True)
                    np.savez(DEST, arr=arr, done=done)
                    await asyncio.sleep(120)
                    continue
                k = await resp.json()
                break
            else:
                raise SystemExit("gave up")
            if isinstance(k, list):
                for x in k:
                    j = (int(x[0]) - int(r.ts)) // 300_000
                    if 0 <= j < B:
                        arr[i, j] = [float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[7]), float(x[10])]
            done[i] = True
            n += 1
            await asyncio.sleep(0.15)
            if n % 500 == 0:
                np.savez(DEST, arr=arr, done=done)
                print(f"{done.sum()}/{len(ev)}", flush=True)
    np.savez(DEST, arr=arr, done=done)
    ev.to_parquet(OUT + r"\flow_events.parquet")
    print("done", done.sum(), len(ev))

asyncio.run(main())
