"""Download the 24h of 5-minute bars after every early-grind moment (resumable, chunked)."""
import sys, asyncio, time, os
sys.path.insert(0, r"D:\Leverage trading")
import numpy as np, pandas as pd
from lts.exchanges import Binance

OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
BARS = 289
CHUNK = 4000
ev = pd.read_parquet(OUT + r"\aim_events.parquet")[["symbol", "ts"]].reset_index(drop=True)
os.makedirs(OUT + r"\paths", exist_ok=True)


async def main():
    b = Binance()
    sem = asyncio.Semaphore(12)
    t0 = time.time()
    try:
        for start in range(0, len(ev), CHUNK):
            fn = OUT + rf"\paths\chunk_{start:06d}.npy"
            if os.path.exists(fn):
                continue
            part = ev.iloc[start:start + CHUNK]
            arr = np.full((len(part), BARS, 4), np.nan, dtype=np.float32)

            async def one(j, sym, ts):
                async with sem:
                    for attempt in range(3):
                        try:
                            rows = await b.klines(sym, "5m", start=int(ts), limit=BARS)
                            rows = [r for r in rows if r[0] >= ts][:BARS]
                            if rows:
                                a = np.array([r[1:5] for r in rows], dtype=np.float32)
                                arr[j, :len(a)] = a
                            return
                        except Exception:
                            await asyncio.sleep(2 * (attempt + 1))
            await asyncio.gather(*(one(j, s, t) for j, (s, t) in enumerate(zip(part.symbol, part.ts))))
            np.save(fn, arr)
            done = min(start + CHUNK, len(ev))
            print(f"{done}/{len(ev)} paths ({time.time() - t0:.0f}s), missing in chunk: {int(np.isnan(arr[:, 0, 0]).sum())}", flush=True)
    finally:
        await b.close()

asyncio.run(main())
print("DONE")
