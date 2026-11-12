"""Historical backfill from Binance (resumable). Weex-only symbols accumulate live instead."""
from __future__ import annotations

import asyncio
import logging
import time

from . import BAR_MS
from .config import Config
from .db import DB
from .exchanges import ApiError, Binance, Weex
from .universe import refresh_universe

log = logging.getLogger("lts.backfill")

PERIOD_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000}
# Measured 12 Nov 2026: an openInterestHist record stamped T matches the live OI at ~T and is
# published 1-3 minutes later. The live collector reads OI directly at bar close, so OI needs no
# shift; the hourly long/short ratios are only read from the statistics endpoint, so they get a
# publication lag (and the live collector fetches them after minute 8 of the hour).
LSR_LAG_MS = 600_000


def _max_ts(db: DB, table: str) -> dict[str, int]:
    return {r["symbol"]: r["m"] for r in db.query(f"SELECT symbol, MAX(ts) AS m FROM {table} GROUP BY symbol")}


async def _run_pool(name: str, symbols: list[str], worker, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    done = 0
    t0 = time.time()

    async def one(sym: str):
        nonlocal done
        async with sem:
            try:
                await worker(sym)
            except ApiError as e:
                log.warning("%s %s failed: %s", name, sym, e)
            except Exception:
                log.exception("%s %s crashed", name, sym)
            done += 1
            if done % 50 == 0 or done == len(symbols):
                log.info("%s: %d/%d symbols (%.0fs)", name, done, len(symbols), time.time() - t0)

    await asyncio.gather(*(one(s) for s in symbols))


async def backfill(cfg: Config, db: DB, days: float = 30.0, oi_period: str = "15m",
                   only: list[str] | None = None) -> None:
    binance, weex = Binance(), Weex()
    try:
        await refresh_universe(cfg, db, binance, weex)
        syms = [r["symbol"] for r in db.query("SELECT symbol FROM symbols WHERE active=1 AND src='binance'")]
        if "BTCUSDT" not in syms:
            syms.append("BTCUSDT")
        if only:
            syms = [s for s in syms if s in set(only)]
        now = int(time.time() * 1000)
        start = (now - int(days * 86_400_000)) // BAR_MS * BAR_MS
        stat_start = max(start, now - int(29.0 * 86_400_000))   # Binance keeps 30 days of OI / ratios
        log.info("backfill %d symbols, %.0f days", len(syms), days)

        k_last = _max_ts(db, "klines")
        oi_last = _max_ts(db, "oi")
        lsr_last = _max_ts(db, "lsr")
        f_last = _max_ts(db, "funding")

        async def klines(sym: str):
            cur = max(start, k_last.get(sym, 0) + BAR_MS)
            while cur < now - BAR_MS:
                rows = await binance.klines(sym, "5m", start=cur, limit=1500)
                rows = [r for r in rows if r[0] + BAR_MS <= now]      # closed bars only
                if not rows:
                    break
                db.insert_klines(sym, rows)
                nxt = rows[-1][0] + BAR_MS
                if nxt <= cur:
                    break
                cur = nxt

        async def oi(sym: str):
            step = PERIOD_MS[oi_period]
            cur = max(stat_start, oi_last.get(sym, 0) + step)
            while cur < now - step:
                end = min(cur + 499 * step, now)
                rows = await binance.oi_hist(sym, oi_period, start=cur, end=end, limit=500)
                if rows:
                    db.insert_rows("oi", ("symbol", "ts", "oi", "oi_value"), [(sym, *r) for r in rows])
                cur = end + step

        async def lsr(sym: str):
            cur = max(stat_start, lsr_last.get(sym, 0) - LSR_LAG_MS + 3_600_000)
            if cur >= now - 3_600_000:
                return
            top: dict[int, float] = {}
            glob: dict[int, float] = {}
            c = cur
            while c < now - 3_600_000:
                top.update(dict(await binance.lsr_top_pos(sym, "1h", start=c, limit=500)))
                glob.update(dict(await binance.lsr_global(sym, "1h", start=c, limit=500)))
                c += 500 * 3_600_000
            # a record stamped T is published a few minutes after T: store it as available at T+10min
            rows = [(sym, ts + LSR_LAG_MS, top.get(ts), glob.get(ts)) for ts in sorted(set(top) | set(glob))]
            db.insert_rows("lsr", ("symbol", "ts", "top_pos", "global_acc"), rows)

        async def funding(sym: str):
            cur = max(start - 4 * 86_400_000, f_last.get(sym, 0) + 1)
            rows = await binance.funding_hist(sym, start=cur, limit=1000)
            db.insert_rows("funding", ("symbol", "ts", "rate"), [(sym, *r) for r in rows])

        await asyncio.gather(
            _run_pool("klines", syms, klines, 6),
            _run_pool("oi", syms, oi, 4),
            _run_pool("lsr", syms, lsr, 3),
            _run_pool("funding", syms, funding, 3),
        )
        log.info("backfill complete")
    finally:
        await binance.close()
        await weex.close()
