"""Download Binance USDT-M candles for a list of (symbol, start, end) ranges into research_data/candles.sqlite.

Resumable (finished ranges are recorded in `fetched`), merged per symbol, throttled well under Binance's limit:
limit=499 requests (weight 2), at most `WEIGHT_PER_MIN` weight a minute, a long pause on HTTP 418/429.
"""
from __future__ import annotations

import asyncio
import time

import aiohttp

from research.common import MIN, candles_db

WEIGHT_PER_MIN = 1800              # Binance allows 2,400 a minute per IP
WORKERS = 6
LIMIT = 499
STEP = {"5m": 5 * MIN, "1m": MIN}
TABLE = {"5m": "k5", "1m": "k1"}


def merge(ranges: list[tuple[str, int, int]], step: int) -> list[tuple[str, int, int]]:
    by = {}
    for s, a, b in ranges:
        by.setdefault(s, []).append((a // step * step, -(-b // step) * step))
    out = []
    for s, rs in by.items():
        rs.sort()
        cur = list(rs[0])
        for a, b in rs[1:]:
            if a <= cur[1] + step:
                cur[1] = max(cur[1], b)
            else:
                out.append((s, cur[0], cur[1]))
                cur = [a, b]
        out.append((s, cur[0], cur[1]))
    return out


def pending(conn, ranges, interval):
    """Ranges minus the parts already fetched."""
    tbl, step = TABLE[interval], STEP[interval]
    done = {}
    for s, a, b in conn.execute("SELECT symbol, start, end FROM fetched WHERE tbl=?", (tbl,)):
        done.setdefault(s, []).append((a, b))
    out = []
    for s, a, b in merge(ranges, step):
        segs = [(a, b)]
        for da, db_ in sorted(done.get(s, [])):
            nxt = []
            for x, y in segs:
                if db_ <= x or da >= y:
                    nxt.append((x, y))
                    continue
                if da > x:
                    nxt.append((x, da))
                if db_ < y:
                    nxt.append((db_, y))
            segs = nxt
        out += [(s, x, y) for x, y in segs if y - x >= step]
    return out


async def run(ranges: list[tuple[str, int, int]], interval: str = "5m", log=print) -> None:
    conn = candles_db()
    tbl, step = TABLE[interval], STEP[interval]
    todo = pending(conn, ranges, interval)
    n_req = sum(-(-(b - a) // (LIMIT * step)) for _, a, b in todo)
    log(f"{interval}: {len(todo)} ranges to fetch, about {n_req:,} requests "
        f"(~{n_req * 2 / WEIGHT_PER_MIN:.0f} min at {WEIGHT_PER_MIN} weight/min)")
    if not todo:
        return
    gap = 60.0 * 2 / WEIGHT_PER_MIN                   # seconds between request starts (all workers together)
    t0, done = time.time(), 0
    pace = {"next": time.monotonic()}
    lock = asyncio.Lock()
    queue = list(todo)

    async def slot():
        async with lock:
            now = time.monotonic()
            wait = pace["next"] - now
            pace["next"] = max(now, pace["next"]) + gap
        if wait > 0:
            await asyncio.sleep(wait)

    async def worker(http):
        nonlocal done
        while queue:
            s, a, b = queue.pop(0)
            t, got = a, 0
            while t < b:
                params = {"symbol": s, "interval": interval, "startTime": t, "endTime": b - 1, "limit": LIMIT}
                for attempt in range(8):
                    await slot()
                    try:
                        async with http.get("https://fapi.binance.com/fapi/v1/klines", params=params) as r:
                            if r.status in (418, 429):
                                wait = float(r.headers.get("Retry-After", 120))
                                log(f"rate limited ({r.status}); pausing {wait:.0f}s")
                                pace["next"] = time.monotonic() + max(wait, 60)
                                continue
                            if r.status == 400:            # symbol not listed in this period
                                rows = []
                                break
                            r.raise_for_status()
                            rows = await r.json()
                            break
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        await asyncio.sleep(5 * (attempt + 1))
                else:
                    raise SystemExit(f"gave up on {s} {t}")
                done += 1
                if rows:
                    conn.executemany(f"INSERT OR IGNORE INTO {tbl} VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                     [(s, int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]),
                                       float(x[7]), int(x[8]), float(x[9]), float(x[10])) for x in rows])
                    got += len(rows)
                    t = b if len(rows) < LIMIT else int(rows[-1][0]) + step     # a short page reached endTime
                else:
                    t += LIMIT * step                      # nothing in this stretch (not yet listed): skip ahead
                if done % 1000 == 0:
                    conn.commit()
                    el = time.time() - t0
                    log(f"  {done:,}/{n_req:,} requests, {el / 60:.0f} min, ~{(n_req - done) * el / done / 60:.0f} min left")
            conn.execute("INSERT OR REPLACE INTO fetched VALUES (?,?,?,?,?)", (tbl, s, a, b, got))
            conn.commit()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30),
                                     headers={"User-Agent": "lts-research/0.1"}) as http:
        await asyncio.gather(*(worker(http) for _ in range(WORKERS)))
    conn.close()
    log(f"{interval}: done, {done:,} requests in {(time.time() - t0) / 60:.0f} min")
