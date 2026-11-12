"""Live data collection. One fast cycle per closed 5m bar, one slow cycle per hour."""
from __future__ import annotations

import asyncio
import logging
import time

from . import BAR_MS
from .backfill import LSR_LAG_MS
from .config import Config
from .db import DB
from .exchanges import Binance, Weex

log = logging.getLogger("lts.collector")


def last_closed_bar(now_ms: int) -> int:
    """Open time of the most recent fully closed 5m bar."""
    return (now_ms // BAR_MS) * BAR_MS - BAR_MS


class Collector:
    def __init__(self, cfg: Config, db: DB, binance: Binance, weex: Weex):
        self.cfg, self.db, self.binance, self.weex = cfg, db, binance, weex
        self.weex_tickers: dict[str, dict] = {}
        self.failures = 0
        self.last_errors = 0
        self.last_seconds = 0.0

    def _symbols(self) -> list[dict]:
        rows = [dict(r) for r in self.db.query("SELECT symbol, src, on_weex, on_binance FROM symbols WHERE active=1")]
        if not any(r["symbol"] == "BTCUSDT" for r in rows):
            rows.append({"symbol": "BTCUSDT", "src": "binance", "on_weex": 1, "on_binance": 1})
        return rows

    async def collect(self, now_ms: int | None = None) -> int:
        """Fetch everything for the bar that just closed. Returns that bar's open time."""
        now_ms = now_ms or int(time.time() * 1000)
        bar_ts = last_closed_bar(now_ms)
        close_ts = bar_ts + BAR_MS
        syms = self._symbols()
        last = self.db.last_kline_ts()
        sem = asyncio.Semaphore(20)
        errors = 0

        async def one(row: dict):
            nonlocal errors
            sym = row["symbol"]
            ex = self.binance if row["src"] == "binance" else self.weex
            cap = 1500 if row["src"] == "binance" else 1000
            missing = (bar_ts - last.get(sym, bar_ts - cap * BAR_MS)) // BAR_MS
            async with sem:
                try:
                    if missing > 0:
                        rows = await ex.klines(sym, "5m", limit=int(min(max(missing + 2, 3), cap)))
                        rows = [r for r in rows if r[0] <= bar_ts]
                        if rows:
                            self.db.insert_klines(sym, rows)
                    oi = await ex.open_interest(sym)
                    if oi:
                        # stamp with the bar close so live and historical series line up
                        self.db.insert_rows("oi", ("symbol", "ts", "oi", "oi_value"), [(sym, close_ts, oi[1], None)])
                except Exception as e:
                    errors += 1
                    log.debug("collect %s failed: %s", sym, e)

        async def premium():
            active = {r["symbol"]: r["src"] for r in syms}
            rows = []
            try:
                for sym, _, mark, index, rate, nxt in await self.binance.premium_all():
                    if active.get(sym) == "binance":
                        rows.append((sym, close_ts, mark, index, rate, nxt))
                if any(src == "weex" for src in active.values()):
                    for sym, _, mark, index, rate, nxt in await self.weex.premium_all():
                        if active.get(sym) == "weex":
                            rows.append((sym, close_ts, mark, index, rate, nxt))
                self.db.insert_rows("premium", ("symbol", "ts", "mark", "index_px", "funding_rate", "next_funding_ts"),
                                    rows)
            except Exception as e:
                log.warning("premium collection failed: %s", e)

        async def weex_tickers():
            try:
                t = await self.weex.tickers()
                self.weex_tickers = t
                on_weex = {r["symbol"] for r in syms if r["on_weex"]}
                self.db.insert_rows("weex_ticker", ("symbol", "ts", "last", "qv24", "chg24", "mark"),
                                    [(s, close_ts, v["last"], v["qv24"], v["chg24"], v.get("mark"))
                                     for s, v in t.items() if s in on_weex])
            except Exception as e:
                log.warning("weex tickers failed: %s", e)

        t0 = time.time()
        await asyncio.gather(premium(), weex_tickers(), *(one(r) for r in syms))
        self.failures = self.failures + 1 if errors > len(syms) * 0.2 else 0
        self.last_errors, self.last_seconds = errors, time.time() - t0
        log.info("collected bar %d: %d symbols, %d errors, %.1fs", bar_ts, len(syms), errors, time.time() - t0)
        return bar_ts

    async def collect_slow(self) -> None:
        """Hourly: positioning ratios and settled funding (Binance-sourced symbols)."""
        syms = [r["symbol"] for r in self._symbols() if r["src"] == "binance"]
        sem = asyncio.Semaphore(4)

        async def one(sym: str):
            async with sem:
                try:
                    top = dict(await self.binance.lsr_top_pos(sym, "1h", limit=3))
                    glob = dict(await self.binance.lsr_global(sym, "1h", limit=3))
                    self.db.insert_rows("lsr", ("symbol", "ts", "top_pos", "global_acc"),
                                        [(sym, ts + LSR_LAG_MS, top.get(ts), glob.get(ts))
                                         for ts in sorted(set(top) | set(glob))])
                except Exception as e:
                    log.debug("lsr %s failed: %s", sym, e)

        await asyncio.gather(*(one(s) for s in syms))
        try:
            # without a symbol the endpoint returns recent settlements for all symbols
            raw = await self.binance.http.get("/fapi/v1/fundingRate",
                                              {"startTime": int(time.time() * 1000) - 9 * 3_600_000, "limit": 1000},
                                              limiter=self.binance.fund)
            keep = set(syms)
            self.db.insert_rows("funding", ("symbol", "ts", "rate"),
                                [(r["symbol"], int(r["fundingTime"]), float(r["fundingRate"]))
                                 for r in raw if r["symbol"] in keep])
        except Exception as e:
            log.warning("funding refresh failed: %s", e)
        log.info("slow collection done for %d symbols", len(syms))
