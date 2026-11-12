"""Tradable universe: merge Binance and Weex listings, decide the data source per symbol."""
from __future__ import annotations

import logging

from .config import Config
from .db import DB
from .exchanges import Binance, Weex

log = logging.getLogger("lts.universe")


def merge_universe(cfg: Config, b_list: list[dict], w_list: list[dict], w_tickers: dict[str, dict]) -> list[dict]:
    b = {s["symbol"]: s for s in b_list}
    w = {s["symbol"]: s for s in w_list}
    rows = []
    for sym in sorted(set(b) | set(w)):
        bs, ws = b.get(sym), w.get(sym)
        utype = (ws or {}).get("underlying_type") or (bs or {}).get("underlying_type") or "COIN"
        active = utype in cfg.universe.underlying_types and sym not in cfg.universe.exclude
        if bs is None:
            # Weex-only: analysed from Weex data, only if liquid enough there
            wq = (w_tickers.get(sym) or {}).get("qv24") or 0.0
            active = active and wq >= cfg.universe.weex_only_min_qvol
        rows.append({
            "symbol": sym, "src": "binance" if bs else "weex",
            "on_binance": bs is not None, "on_weex": ws is not None,
            "base": (bs or ws).get("base"), "underlying_type": utype,
            "onboard_ts": (bs or {}).get("onboard_ts"),
            "weex_max_lev": (ws or {}).get("weex_max_lev"),
            "weex_taker_fee": (ws or {}).get("weex_taker_fee"),
            "weex_maker_fee": (ws or {}).get("weex_maker_fee"),
            "price_precision": (ws or bs).get("price_precision"),
            "active": active,
        })
    return rows


async def refresh_universe(cfg: Config, db: DB, binance: Binance, weex: Weex) -> int:
    b_list = await binance.universe()
    try:
        w_list = await weex.universe()
        w_tickers = await weex.tickers()
    except Exception as e:  # Weex outage must not stop Binance-driven analysis
        log.warning("weex universe unavailable: %s", e)
        if db.query("SELECT 1 FROM symbols LIMIT 1"):
            return 0
        w_list, w_tickers = [], {}
    rows = merge_universe(cfg, b_list, w_list, w_tickers)
    seen = {r["symbol"] for r in rows}
    db.upsert_symbols(rows)
    # delisted symbols
    for r in db.query("SELECT symbol FROM symbols WHERE active=1"):
        if r["symbol"] not in seen:
            db.execute("UPDATE symbols SET active=0 WHERE symbol=?", (r["symbol"],))
    n = sum(1 for r in rows if r["active"])
    log.info("universe: %d active symbols (%d both venues)", n,
             sum(1 for r in rows if r["active"] and r["on_binance"] and r["on_weex"]))
    return n
