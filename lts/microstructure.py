"""Liquidations and order-book depth, collected live (no free history exists).

- Liquidations: Binance USDS-M all-market force-order stream (!forceOrder@arr). Binance pushes at most the
  latest liquidation per symbol per second, so this is a lower bound on liquidation flow, but it shows where and
  when positions were force-closed - the fuel behind liquidity sweeps.
- Order book: every few minutes, top-100 levels for the most liquid coins (and any coin currently grinding),
  reduced to depth within 0.5% / 1% / 2% of mid, imbalance, and the largest bid/ask walls nearby.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import aiohttp
import numpy as np

from .db import DB

log = logging.getLogger("lts.micro")
# Binance moved USDS-M market streams under /market/ (the old /ws path still connects but stays silent)
WS_URL = "wss://fstream.binance.com/market/ws/!forceOrder@arr"
SILENCE_S = 600                     # reconnect if nothing arrives for this long (liquidations happen every few seconds)

SCHEMA = """
CREATE TABLE IF NOT EXISTS liquidations (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL,
    side TEXT NOT NULL,                 -- long_liq (a long was force-sold) / short_liq (a short was force-bought)
    price REAL, qty REAL, usd REAL
);
CREATE INDEX IF NOT EXISTS idx_liq_sym_ts ON liquidations (symbol, ts);
CREATE INDEX IF NOT EXISTS idx_liq_ts ON liquidations (ts);
CREATE TABLE IF NOT EXISTS book_snapshots (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL,
    mid REAL, spread_bps REAL,
    bid05 REAL, ask05 REAL, bid1 REAL, ask1 REAL, bid2 REAL, ask2 REAL,     -- USD depth within 0.5% / 1% / 2%
    big_bid_dist REAL, big_bid_usd REAL, big_bid_x REAL,                   -- largest bid level within 2%:
    big_ask_dist REAL, big_ask_usd REAL, big_ask_x REAL,                   --   distance (fraction), USD, x median level
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;
"""


def ensure_schema(db: DB) -> None:
    db.conn.executescript(SCHEMA)
    db.conn.commit()


def parse_liquidation(msg: dict) -> tuple | None:
    o = (msg or {}).get("o")
    if not o:
        return None
    try:
        price = float(o.get("ap") or o.get("p"))
        qty = float(o.get("z") or o.get("q"))
        side = "long_liq" if o["S"] == "SELL" else "short_liq"
        return (o["s"], int(o.get("T") or msg.get("E")), side, price, qty, price * qty)
    except (KeyError, TypeError, ValueError):
        return None


def book_metrics(bids: list, asks: list) -> dict | None:
    if not bids or not asks:
        return None
    b = np.array(bids, dtype=float)
    a = np.array(asks, dtype=float)
    best_bid, best_ask = b[0, 0], a[0, 0]
    mid = (best_bid + best_ask) / 2
    if mid <= 0 or best_ask < best_bid:
        return None
    bd, ad = 1 - b[:, 0] / mid, a[:, 0] / mid - 1                      # distance from mid (fraction)
    bu, au = b[:, 0] * b[:, 1], a[:, 0] * a[:, 1]                        # USD per level
    out = {"mid": mid, "spread_bps": (best_ask - best_bid) / mid * 1e4}
    for w, name in ((0.005, "05"), (0.01, "1"), (0.02, "2")):
        out[f"bid{name}"] = float(bu[bd <= w].sum())
        out[f"ask{name}"] = float(au[ad <= w].sum())
    for side, d, u in (("bid", bd, bu), ("ask", ad, au)):
        m = d <= 0.02
        if m.any():
            j = int(np.argmax(np.where(m, u, -1)))
            med = float(np.median(u[m])) if m.sum() > 1 else float(u[j])
            out[f"big_{side}_dist"], out[f"big_{side}_usd"] = float(d[j]), float(u[j])
            out[f"big_{side}_x"] = float(u[j] / med) if med > 0 else None
        else:
            out[f"big_{side}_dist"] = out[f"big_{side}_usd"] = out[f"big_{side}_x"] = None
    return out


class LiquidationStream:
    """Listens to the all-market liquidation stream and writes batches to the database."""

    def __init__(self, db: DB):
        self.db = db
        self.buffer: list[tuple] = []
        self.received = 0
        self.connected = False

    def flush(self) -> None:
        if self.buffer:
            rows, self.buffer = self.buffer, []
            self.db.insert_rows("liquidations", ("symbol", "ts", "side", "price", "qty", "usd"), rows)

    async def run(self) -> None:
        ensure_schema(self.db)
        backoff = 5
        while True:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(WS_URL, heartbeat=60, timeout=30) as ws:
                        self.connected, backoff = True, 5
                        log.info("liquidation stream connected")
                        last_flush = time.time()
                        while True:
                            try:
                                msg = await asyncio.wait_for(ws.receive(), timeout=SILENCE_S)
                            except asyncio.TimeoutError:
                                log.warning("liquidation stream silent for %ds - reconnecting", SILENCE_S)
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                for item in data if isinstance(data, list) else [data]:
                                    row = parse_liquidation(item)
                                    if row:
                                        self.buffer.append(row)
                                        self.received += 1
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                            if time.time() - last_flush >= 10:
                                self.flush()
                                last_flush = time.time()
            except asyncio.CancelledError:
                self.flush()
                raise
            except Exception as e:
                log.warning("liquidation stream error: %s", e)
            self.connected = False
            self.flush()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 300)


async def snapshot_books(db: DB, binance, symbols: list[str], ts: int) -> int:
    """Top-100 depth for each symbol -> one row of metrics each."""
    ensure_schema(db)
    sem = asyncio.Semaphore(8)
    rows = []

    async def one(sym):
        async with sem:
            try:
                r = await binance.http.get("/fapi/v1/depth", {"symbol": sym, "limit": 100}, limiter=binance.w, weight=5)
            except Exception as e:
                log.debug("depth %s: %s", sym, e)
                return
            m = book_metrics(r.get("bids", []), r.get("asks", []))
            if m:
                rows.append((sym, ts, m["mid"], m["spread_bps"], m["bid05"], m["ask05"], m["bid1"], m["ask1"],
                             m["bid2"], m["ask2"], m["big_bid_dist"], m["big_bid_usd"], m["big_bid_x"],
                             m["big_ask_dist"], m["big_ask_usd"], m["big_ask_x"]))
    await asyncio.gather(*(one(s) for s in symbols))
    db.insert_rows("book_snapshots", ("symbol", "ts", "mid", "spread_bps", "bid05", "ask05", "bid1", "ask1", "bid2",
                                      "ask2", "big_bid_dist", "big_bid_usd", "big_bid_x", "big_ask_dist", "big_ask_usd",
                                      "big_ask_x"), rows)
    return len(rows)


def book_symbols(db: DB, n: int) -> list[str]:
    """The n most liquid coins by the latest 24h volume, plus anything grinding in the last 12h."""
    now = int(time.time() * 1000)
    rows = db.query("""SELECT symbol, SUM(qv) AS v FROM klines WHERE ts >= ? GROUP BY symbol ORDER BY v DESC LIMIT ?""",
                    (now - 86_400_000, n))
    syms = [r["symbol"] for r in rows]
    try:
        syms += [r["symbol"] for r in db.query("SELECT DISTINCT symbol FROM grind_live WHERE ts >= ?", (now - 12 * 3_600_000,))]
    except Exception:
        pass
    return list(dict.fromkeys(syms))


def prune(db: DB, keep_days: int) -> None:
    cutoff = int(time.time() * 1000) - keep_days * 86_400_000
    db.execute("DELETE FROM liquidations WHERE ts < ?", (cutoff,))
    db.execute("DELETE FROM book_snapshots WHERE ts < ?", (cutoff,))
