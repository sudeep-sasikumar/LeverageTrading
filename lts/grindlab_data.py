"""Grind lab - data: longer hourly and daily history, sector tags, exchange announcements.

The 5-minute store keeps ~30 days. Studying what successful steady trends ("grinds") have in
common needs more: hourly bars for ~4 months, daily bars back to listing (for daily/weekly
support and resistance, all-time highs/lows), funding history, the sector each coin belongs
to, and coin-specific exchange news (Binance listings / futures launches / delistings /
monitoring tags, Upbit listings / caution warnings / delistings).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from .db import DB
from .exchanges import Binance, Http, WeightLimiter

log = logging.getLogger("lts.grindlab")
DAY = 86_400_000
HOUR = 3_600_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS klines_1h (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL,
    o REAL, h REAL, l REAL, c REAL, v REAL, qv REAL, trades INTEGER, tbv REAL, tbqv REAL,
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS klines_1d (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL,
    o REAL, h REAL, l REAL, c REAL, v REAL, qv REAL, trades INTEGER, tbv REAL, tbqv REAL,
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS announcements (
    source TEXT NOT NULL,            -- binance / upbit
    ts INTEGER NOT NULL,
    base TEXT NOT NULL,              -- coin ticker, e.g. CROSS
    kind TEXT NOT NULL,              -- list / futures_launch / delist / monitoring / alpha / airdrop / news /
                                     -- upbit_list / upbit_warning / upbit_warning_lifted / upbit_delist
    title TEXT,
    PRIMARY KEY (source, ts, base, kind)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS symbol_tags (
    symbol TEXT NOT NULL, tag TEXT NOT NULL, PRIMARY KEY (symbol, tag)
) WITHOUT ROWID;
"""

BINANCE_CATALOGS = {48: "New Cryptocurrency Listing", 161: "Delisting", 49: "Latest Binance News",
                    128: "Crypto Airdrop", 93: "Latest Activities"}


def ensure_schema(db: DB) -> None:
    db.conn.executescript(SCHEMA)
    db.conn.commit()


def clean_base(base: str) -> str:
    """1000PEPE -> PEPE, 1MBABYDOGE -> BABYDOGE: the ticker as announcements write it."""
    return re.sub(r"^(1000000|100000|10000|1000|1M)", "", base or "")


# ---- announcement parsing ---------------------------------------------------
def binance_kind(title: str) -> str:
    t = title.lower()
    if "monitoring tag" in t:
        return "monitoring"
    if "delist" in t or "removal of" in t or "cease" in t:
        return "delist"
    if "futures will launch" in t or "perpetual contract" in t:
        return "futures_launch"
    if "alpha" in t:
        return "alpha"
    if "hodler" in t or "airdrop" in t or "launchpool" in t or "megadrop" in t:
        return "airdrop"
    if ("will list" in t or "will add" in t or "adds" in t) and "collateral" not in t:
        return "list"
    return "news"


def upbit_kind(title: str) -> str | None:
    if "해제" in title:                                    # designation lifted
        return "upbit_warning_lifted"
    if "유의" in title:                                    # caution / warning designation
        return "upbit_warning"
    if "거래지원 종료" in title:
        return "upbit_delist"
    if "신규 거래지원" in title or "디지털 자산 추가" in title:
        return "upbit_list"
    return None


def tickers_in(title: str, bases: set[str]) -> set[str]:
    """Coin tickers mentioned in an announcement title, restricted to coins we trade."""
    found = set(re.findall(r"\(([A-Z0-9]{2,15})\)", title))
    found |= {m for m in re.findall(r"\b([A-Z0-9]{2,15})USD[TC]\b", title)}
    m = re.search(r"(?:Delist|Remove|Monitoring Tag[^:]*?Include|Removal of[^:]*?:)\s*([A-Z0-9 ,&and]+)", title)
    if m:
        found |= {w for w in re.split(r"[ ,&]+|and", m.group(1)) if w}
    return {f for f in found if f in bases}


async def fetch_announcements(db: DB, since_ts: int, bases: set[str]) -> int:
    http = Http("https://www.binance.com")
    lim = WeightLimiter(4, 1.0)
    rows = []
    try:
        for cat in BINANCE_CATALOGS:
            for page in range(1, 60):
                try:
                    data = await http.get("/bapi/composite/v1/public/cms/article/list/query",
                                          {"type": 1, "catalogId": cat, "pageNo": page, "pageSize": 50}, limiter=lim)
                except Exception as e:
                    log.warning("binance announcements %s page %d: %s", cat, page, e)
                    break
                arts = ((data.get("data") or {}).get("catalogs") or [{}])[0].get("articles") or []
                if not arts:
                    break
                for a in arts:
                    ts, title = int(a["releaseDate"]), a["title"]
                    kind = binance_kind(title)
                    for b in tickers_in(title, bases):
                        rows.append(("binance", ts, b, kind, title[:300]))
                if int(arts[-1]["releaseDate"]) < since_ts:
                    break
    finally:
        await http.close()
    up = Http("https://api-manager.upbit.com")
    try:
        for page in range(1, 60):
            try:
                data = await up.get("/api/v1/announcements", {"os": "web", "page": page, "per_page": 20,
                                                              "category": "trade"}, limiter=lim)
            except Exception as e:
                log.warning("upbit notices page %d: %s", page, e)
                break
            notes = (data.get("data") or {}).get("notices") or []
            if not notes:
                break
            oldest = None
            for n in notes:
                ts = _kst_to_utc_ms(n["listed_at"])            # e.g. 2026-11-16T20:30:00+09:00
                oldest = ts
                kind = upbit_kind(n["title"])
                if not kind:
                    continue
                for b in set(re.findall(r"\(([A-Z0-9]{2,15})\)", n["title"])) & bases:
                    rows.append(("upbit", ts, b, kind, n["title"][:300]))
            if oldest is not None and oldest < since_ts:
                break
    finally:
        await up.close()
    db.insert_rows("announcements", ("source", "ts", "base", "kind", "title"), rows)
    return len(rows)


def _kst_to_utc_ms(s: str) -> int:
    from datetime import datetime
    return int(datetime.fromisoformat(s).timestamp() * 1000)          # string carries its +09:00 offset


# ---- price history ----------------------------------------------------------
async def _bars(db: DB, binance: Binance, table: str, interval: str, step: int, sym: str, start: int, now: int) -> None:
    """Fill [start, now): backfill before the earliest stored bar if needed, then extend after the latest."""
    row = db.query(f"SELECT MIN(ts) AS a, MAX(ts) AS m FROM {table} WHERE symbol=?", (sym,))[0]
    first, last = row["a"], row["m"]
    if first is not None and first > start + step:
        await _fetch_range(db, binance, table, interval, step, sym, start, first)
    await _fetch_range(db, binance, table, interval, step, sym, max(start, (last or 0) + step), now)


async def _fetch_range(db: DB, binance: Binance, table: str, interval: str, step: int, sym: str, cur: int,
                       stop: int) -> None:
    now = int(time.time() * 1000)
    while cur < min(stop, now - step):
        rows = await binance.klines(sym, interval, start=cur, limit=1500)
        rows = [r for r in rows if r[0] + step <= now and r[0] < stop]      # closed bars, inside the range
        if not rows:
            break
        db.insert_rows(table, ("symbol", "ts", "o", "h", "l", "c", "v", "qv", "trades", "tbv", "tbqv"),
                       [(sym, *r) for r in rows])
        nxt = rows[-1][0] + step
        if nxt <= cur:
            break
        cur = nxt


async def refresh_recent(db: DB, symbols: list[str], bars: int = 3) -> int:
    """Hourly: fetch the last few 1h bars for each symbol (cheap: one small request per coin)."""
    ensure_schema(db)
    binance = Binance()
    now = int(time.time() * 1000)
    sem = asyncio.Semaphore(10)
    n = 0

    async def one(sym):
        nonlocal n
        async with sem:
            try:
                rows = [r for r in await binance.klines(sym, "1h", limit=bars + 1) if r[0] + HOUR <= now]
                db.insert_rows("klines_1h", ("symbol", "ts", "o", "h", "l", "c", "v", "qv", "trades", "tbv", "tbqv"),
                               [(sym, *r) for r in rows])
                n += 1
            except Exception as e:
                log.debug("hourly bars %s: %s", sym, e)
    try:
        await asyncio.gather(*(one(s) for s in symbols))
    finally:
        await binance.close()
    return n


async def refresh(db: DB, hourly_days: float = 125, funding_days: float = 125, announcements: bool = True) -> dict:
    """Bring the grind-lab history up to date (incremental after the first run)."""
    ensure_schema(db)
    binance = Binance()
    t0 = time.time()
    try:
        info = await binance.http.get("/fapi/v1/exchangeInfo", limiter=binance.w, weight=1)
        tags = []
        for s in info["symbols"]:
            if s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL":
                tags += [(s["symbol"], t) for t in (s.get("underlyingSubType") or [])]
        db.execute("DELETE FROM symbol_tags")
        db.insert_rows("symbol_tags", ("symbol", "tag"), tags)
        syms = [r["symbol"] for r in db.query("SELECT symbol FROM symbols WHERE active=1 AND src='binance'")]
        if "BTCUSDT" not in syms:
            syms.append("BTCUSDT")
        now = int(time.time() * 1000)
        sem = asyncio.Semaphore(6)

        async def one(sym):
            async with sem:
                try:
                    await _bars(db, binance, "klines_1h", "1h", HOUR, sym, now - int(hourly_days * DAY), now)
                    await _bars(db, binance, "klines_1d", "1d", DAY, sym, 0, now)
                    f_row = db.query("SELECT MIN(ts) AS a, MAX(ts) AS m FROM funding WHERE symbol=?", (sym,))[0]
                    want = now - int(funding_days * DAY)
                    # backfill the full window if the stored history starts later than wanted, else just extend it
                    f_start = want if (f_row["a"] is None or f_row["a"] > want + 9 * HOUR) else f_row["m"] + 1
                    while f_start < now - 8 * HOUR:
                        rows = await binance.funding_hist(sym, start=f_start, limit=1000)
                        if not rows:
                            break
                        db.insert_rows("funding", ("symbol", "ts", "rate"), [(sym, *r) for r in rows])
                        if len(rows) < 1000:
                            break
                        f_start = rows[-1][0] + 1
                except Exception as e:
                    log.warning("grindlab history %s: %s", sym, e)

        done = 0
        for chunk in range(0, len(syms), 50):
            await asyncio.gather(*(one(s) for s in syms[chunk:chunk + 50]))
            done += len(syms[chunk:chunk + 50])
            log.info("grindlab history: %d/%d symbols (%.0fs)", done, len(syms), time.time() - t0)
        n_ann = 0
        if announcements:
            bases = {clean_base(r["base"]) for r in db.query("SELECT base FROM symbols WHERE base IS NOT NULL")}
            n_ann = await fetch_announcements(db, now - int(hourly_days * DAY), bases)
        return {"symbols": len(syms), "announcements": n_ann, "seconds": time.time() - t0}
    finally:
        await binance.close()
