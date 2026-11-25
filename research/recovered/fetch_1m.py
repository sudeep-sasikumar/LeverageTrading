"""1-minute candles for 24h after entry, for the year's 5x WINNERS (touched +2% before -1% on 5m data):
every all-day-steady winner + 1,500 random other winners (fixed seed). Throttled: limit 480 (weight 2), 0.2s apart."""
import asyncio, os
import aiohttp
import numpy as np, pandas as pd

OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
DEST = OUT + r"\winners_1m.npz"
r = pd.read_parquet(OUT + r"\runners_results.parquet")
a = pd.read_parquet(OUT + r"\aim_events.parquet")
w = (r.win & r.valid).to_numpy()
R3 = r.R3.to_numpy()
rng = np.random.default_rng(7)
pick = np.sort(np.concatenate([np.where(w & R3)[0], rng.choice(np.where(w & ~R3)[0], 1500, replace=False)]))
ev = a.iloc[pick][["symbol", "ts", "side"]].reset_index(drop=True).assign(row=pick)
ev.to_parquet(OUT + r"\winners_1m_events.parquet")
B = 1440
arr = np.full((len(ev), B, 4), np.nan, np.float32)
done = np.zeros(len(ev), bool)
if os.path.exists(DEST):
    z = np.load(DEST)
    arr, done = z["arr"], z["done"]


async def main():
    async with aiohttp.ClientSession() as s:
        n = 0
        for i, e in ev.iterrows():
            if done[i]:
                continue
            for part in range(3):
                st = int(e.ts) + part * 480 * 60_000
                for attempt in range(6):
                    resp = await s.get("https://fapi.binance.com/fapi/v1/klines",
                                       params={"symbol": e.symbol, "interval": "1m", "startTime": st, "limit": 480})
                    if resp.status in (418, 429):
                        print("rate limited, pausing", resp.status, flush=True)
                        np.savez(DEST, arr=arr, done=done)
                        await asyncio.sleep(180)
                        continue
                    k = await resp.json()
                    break
                else:
                    raise SystemExit("gave up")
                if isinstance(k, list):
                    for x in k:
                        j = (int(x[0]) - int(e.ts)) // 60_000
                        if 0 <= j < B:
                            arr[i, j] = [float(x[1]), float(x[2]), float(x[3]), float(x[4])]
                await asyncio.sleep(0.2)
            done[i] = True
            n += 1
            if n % 200 == 0:
                np.savez(DEST, arr=arr, done=done)
                print(f"{done.sum()}/{len(ev)}", flush=True)
    np.savez(DEST, arr=arr, done=done)
    print("done", done.sum(), len(ev), flush=True)

asyncio.run(main())
