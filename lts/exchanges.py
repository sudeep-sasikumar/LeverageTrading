"""Public (keyless) REST adapters for Binance USDT-M futures and Weex contract API v3.

Both adapters return the same normalised shapes so the collector can treat them alike.
Kline rows: (ts, o, h, l, c, v, qv, trades, tbv, tbqv) with ts = bar open time in ms.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

import aiohttp

log = logging.getLogger("lts.exchanges")


class ApiError(Exception):
    def __init__(self, status: int, body: str, url: str = ""):
        super().__init__(f"HTTP {status} {url}: {body[:200]}")
        self.status = status
        self.body = body


class WeightLimiter:
    """Sliding-window limiter: at most `limit` weight per `window` seconds."""

    def __init__(self, limit: float, window: float):
        self.limit, self.window = limit, window
        self._events: deque[tuple[float, float]] = deque()
        self._used = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self, weight: float = 1.0) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._events and now - self._events[0][0] >= self.window:
                    self._used -= self._events.popleft()[1]
                if self._used + weight <= self.limit or not self._events:
                    self._events.append((now, weight))
                    self._used += weight
                    return
                await asyncio.sleep(max(0.02, self.window - (now - self._events[0][0])))


class Http:
    def __init__(self, base: str):
        self.base = base
        self._session: aiohttp.ClientSession | None = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=25),
                connector=aiohttp.TCPConnector(limit=40, ttl_dns_cache=300),
                headers={"User-Agent": "lts-scanner/0.1"})
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def get(self, path: str, params: dict | None = None, *, limiter: WeightLimiter,
                  weight: float = 1.0, retries: int = 5):
        url = self.base + path
        params = {k: v for k, v in (params or {}).items() if v is not None}
        last: Exception | None = None
        for attempt in range(retries):
            await limiter.acquire(weight)
            try:
                s = await self.session()
                async with s.get(url, params=params) as r:
                    if r.status == 200:
                        return await r.json(content_type=None)
                    body = await r.text()
                    if r.status in (429, 418):
                        wait = float(r.headers.get("Retry-After", 15 * (attempt + 1)))
                        log.warning("rate limited (%s) on %s, sleeping %.0fs", r.status, path, wait)
                        await asyncio.sleep(min(wait, 180))
                        last = ApiError(r.status, body, url)
                        continue
                    if r.status >= 500:
                        last = ApiError(r.status, body, url)
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise ApiError(r.status, body, url)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last = e
                await asyncio.sleep(1.5 * (attempt + 1))
        raise last if last else ApiError(0, "request failed", url)


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _kline_rows(raw: list) -> list[tuple]:
    rows = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]),
             float(k[7]), int(k[8]), float(k[9]), float(k[10])) for k in raw]
    rows.sort(key=lambda r: r[0])
    return rows


def _book_stats(bids: list, asks: list) -> dict | None:
    if not bids or not asks:
        return None
    bid, ask = float(bids[0][0]), float(asks[0][0])
    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    bid_val = sum(float(p) * float(q) for p, q in bids)
    ask_val = sum(float(p) * float(q) for p, q in asks)
    tot = bid_val + ask_val
    return {"bid": bid, "ask": ask, "mid": mid, "spread_bps": (ask - bid) / mid * 1e4,
            "imb": (bid_val - ask_val) / tot if tot > 0 else 0.0, "depth_usd": tot}


class Binance:
    name = "binance"

    def __init__(self):
        self.http = Http("https://fapi.binance.com")
        self.w = WeightLimiter(1700, 60)        # documented 2400 / minute
        self.data = WeightLimiter(800, 300)     # /futures/data: 1000 requests / 5 min
        self.fund = WeightLimiter(400, 300)     # fundingRate: 500 requests / 5 min

    async def close(self):
        await self.http.close()

    async def universe(self) -> list[dict]:
        info = await self.http.get("/fapi/v1/exchangeInfo", limiter=self.w, weight=1)
        out = []
        for s in info["symbols"]:
            if s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING":
                out.append({"symbol": s["symbol"], "base": s.get("baseAsset"),
                            "onboard_ts": s.get("onboardDate"), "underlying_type": s.get("underlyingType"),
                            "price_precision": s.get("pricePrecision")})
        return out

    async def tickers(self) -> dict[str, dict]:
        raw = await self.http.get("/fapi/v1/ticker/24hr", limiter=self.w, weight=40)
        return {r["symbol"]: {"last": _f(r["lastPrice"]), "qv24": _f(r["quoteVolume"]),
                              "chg24": _f(r["priceChangePercent"], 0.0) / 100.0} for r in raw}

    async def klines(self, symbol: str, interval: str = "5m", start: int | None = None,
                     end: int | None = None, limit: int = 500) -> list[tuple]:
        weight = 1 if limit < 100 else 2 if limit < 500 else 5 if limit <= 1000 else 10
        raw = await self.http.get("/fapi/v1/klines", {"symbol": symbol, "interval": interval,
                                                      "startTime": start, "endTime": end, "limit": limit},
                                  limiter=self.w, weight=weight)
        return _kline_rows(raw)

    async def open_interest(self, symbol: str) -> tuple[int, float] | None:
        r = await self.http.get("/fapi/v1/openInterest", {"symbol": symbol}, limiter=self.w, weight=1)
        return (int(r["time"]), float(r["openInterest"]))

    async def oi_hist(self, symbol: str, period: str = "15m", start: int | None = None,
                      end: int | None = None, limit: int = 500) -> list[tuple]:
        raw = await self.http.get("/futures/data/openInterestHist",
                                  {"symbol": symbol, "period": period, "startTime": start, "endTime": end,
                                   "limit": limit}, limiter=self.data)
        return [(int(r["timestamp"]), float(r["sumOpenInterest"]), float(r["sumOpenInterestValue"])) for r in raw]

    async def premium_all(self) -> list[tuple]:
        raw = await self.http.get("/fapi/v1/premiumIndex", limiter=self.w, weight=10)
        return [(r["symbol"], int(r["time"]), _f(r["markPrice"]), _f(r["indexPrice"]),
                 _f(r["lastFundingRate"]), int(r.get("nextFundingTime") or 0)) for r in raw]

    async def funding_hist(self, symbol: str, start: int | None = None, limit: int = 1000) -> list[tuple]:
        raw = await self.http.get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": start, "limit": limit},
                                  limiter=self.fund)
        return [(int(r["fundingTime"]), float(r["fundingRate"])) for r in raw]

    async def _ratio(self, path: str, symbol: str, period: str, start: int | None, limit: int) -> list[tuple]:
        raw = await self.http.get(path, {"symbol": symbol, "period": period, "startTime": start, "limit": limit},
                                  limiter=self.data)
        return [(int(r["timestamp"]), float(r["longShortRatio"])) for r in raw]

    async def lsr_top_pos(self, symbol: str, period: str = "1h", start: int | None = None, limit: int = 500):
        return await self._ratio("/futures/data/topLongShortPositionRatio", symbol, period, start, limit)

    async def lsr_global(self, symbol: str, period: str = "1h", start: int | None = None, limit: int = 500):
        return await self._ratio("/futures/data/globalLongShortAccountRatio", symbol, period, start, limit)

    async def book(self, symbol: str) -> dict | None:
        r = await self.http.get("/fapi/v1/depth", {"symbol": symbol, "limit": 20}, limiter=self.w, weight=2)
        return _book_stats(r.get("bids", []), r.get("asks", []))


class Weex:
    name = "weex"

    def __init__(self):
        self.http = Http("https://api-contract.weex.com")
        self.w = WeightLimiter(250, 10)         # observed header: 500 weight / 10 s

    async def close(self):
        await self.http.close()

    async def universe(self) -> list[dict]:
        info = await self.http.get("/capi/v3/market/exchangeInfo", limiter=self.w, weight=5)
        out = []
        for s in info.get("symbols", []):
            if s.get("quoteAsset") != "USDT" or s.get("contractType") != "PERPETUAL":
                continue
            out.append({"symbol": s["symbol"], "base": s.get("baseAsset"),
                        "underlying_type": s.get("underlyingType"), "weex_max_lev": s.get("maxLeverage"),
                        "weex_taker_fee": s.get("takerFeeRate"), "weex_maker_fee": s.get("makerFeeRate"),
                        "price_precision": s.get("pricePrecision")})
        return out

    async def tickers(self) -> dict[str, dict]:
        raw = await self.http.get("/capi/v3/market/ticker/24hr", limiter=self.w, weight=200)
        # Weex reports priceChangePercent as a fraction already (-0.0627 = -6.27%)
        return {r["symbol"]: {"last": _f(r["lastPrice"]), "qv24": _f(r["quoteVolume"]),
                              "chg24": _f(r["priceChangePercent"], 0.0), "mark": _f(r.get("markPrice"))}
                for r in raw}

    async def klines(self, symbol: str, interval: str = "5m", start: int | None = None,
                     end: int | None = None, limit: int = 500) -> list[tuple]:
        raw = await self.http.get("/capi/v3/market/klines", {"symbol": symbol, "interval": interval,
                                                             "limit": min(limit, 1000)},
                                  limiter=self.w, weight=5)
        rows = _kline_rows(raw)
        if start is not None:
            rows = [r for r in rows if r[0] >= start]
        if end is not None:
            rows = [r for r in rows if r[0] <= end]
        return rows

    async def open_interest(self, symbol: str) -> tuple[int, float] | None:
        r = await self.http.get("/capi/v3/market/openInterest", {"symbol": symbol}, limiter=self.w, weight=10)
        return (int(r["time"]), float(r["openInterest"]))

    async def premium_all(self) -> list[tuple]:
        raw = await self.http.get("/capi/v3/market/premiumIndex", limiter=self.w, weight=25)
        return [(r["symbol"], int(r["time"]), _f(r["markPrice"]), _f(r["indexPrice"]),
                 _f(r.get("forecastFundingRate"), _f(r.get("lastFundingRate"))),
                 int(r.get("nextFundingTime") or 0)) for r in raw]

    async def funding_hist(self, symbol: str, start: int | None = None, limit: int = 100) -> list[tuple]:
        raw = await self.http.get("/capi/v3/market/fundingRate", {"symbol": symbol, "limit": limit},
                                  limiter=self.w, weight=25)
        return [(int(r["fundingTime"]), float(r["fundingRate"])) for r in raw]

    async def book(self, symbol: str) -> dict | None:
        r = await self.http.get("/capi/v3/market/depth", {"symbol": symbol, "limit": 15}, limiter=self.w, weight=5)
        return _book_stats(r.get("bids", []), r.get("asks", []))
