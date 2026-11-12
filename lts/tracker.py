"""Outcome tracking: open signals (1-minute resolution) and labelling of matured candidates."""
from __future__ import annotations

import json
import logging
import time

from . import BAR_MS
from .config import Config
from .db import DB
from .labels import label_path
from .signals import close_signal, exit_message

log = logging.getLogger("lts.tracker")
MIN = 60_000


def evaluate_bars(sig: dict, bars: list[tuple]) -> tuple[str | None, float, int, float, float]:
    """Walk 1m bars (ts, o, h, l, c, ...) after entry.

    Returns (status or None, exit_price, exit_ts, mfe, mae). A bar touching both levels
    counts as a stop (conservative).
    """
    side, entry = sig["side"], sig["entry"]
    mfe, mae = sig.get("mfe") or 0.0, sig.get("mae") or 0.0
    for ts, _o, h, l, _c, *_ in bars:
        up, down = h / entry - 1, 1 - l / entry
        fav, adv = (up, down) if side > 0 else (down, up)
        mfe, mae = max(mfe, fav), max(mae, adv)
        hit_sl = l <= sig["sl"] if side > 0 else h >= sig["sl"]
        hit_tp = h >= sig["tp"] if side > 0 else l <= sig["tp"]
        if hit_sl:
            return "sl", sig["sl"], ts + MIN, mfe, mae
        if hit_tp:
            return "tp", sig["tp"], ts + MIN, mfe, mae
    return None, 0.0, 0, mfe, mae


async def track_open_signals(cfg: Config, db: DB, binance, weex, notifier) -> None:
    now = int(time.time() * 1000)
    for row in db.query("SELECT s.*, y.on_weex FROM signals s LEFT JOIN symbols y ON y.symbol=s.symbol "
                        "WHERE s.status='open'"):
        sig = dict(row)
        ex = weex if sig.get("on_weex") else binance          # track on the venue the user trades
        try:
            start = max(sig["last_checked_ts"] or sig["ts"], sig["ts"]) // MIN * MIN
            need = int(min(1000, (now - start) // MIN + 2))
            bars = [b for b in await ex.klines(sig["symbol"], "1m", limit=max(need, 2)) if b[0] >= start]
        except Exception as e:
            log.warning("tracking %s failed: %s", sig["symbol"], e)
            continue
        status, px, ts, mfe, mae = evaluate_bars(sig, bars)
        if status is None and now >= sig["expires_ts"] and bars:
            status, px, ts = "time", bars[-1][4], now
        if status:
            closed = close_signal(cfg, db, sig, status, px, ts, mfe, mae)
            await notifier.send(exit_message(cfg, db, closed))
            log.info("signal #%d %s closed: %s %+.2fR", sig["id"], sig["symbol"], status, closed["ret_r"])
        else:
            closed_bars = [b for b in bars if b[0] + MIN <= now]
            last = closed_bars[-1][0] + MIN if closed_bars else sig["last_checked_ts"]
            db.execute("UPDATE signals SET mfe=?, mae=?, last_checked_ts=? WHERE id=?", (mfe, mae, last, sig["id"]))


def label_matured(cfg: Config, db: DB) -> int:
    """Label live candidates whose full horizon is now in the kline store."""
    tc = cfg.trade
    max_h = max([tc.horizon_bars] + list(tc.grid_horizon))
    last = db.last_kline_ts()
    now = int(time.time() * 1000)
    done = 0
    for r in db.query("SELECT id, symbol, ts, side, atr_pct, ref_price FROM candidates "
                      "WHERE labeled_ts IS NULL AND ts<=?",
                      (now - max_h * BAR_MS,)):
        sym, ts = r["symbol"], r["ts"]
        if last.get(sym, 0) < ts + (max_h - 1) * BAR_MS:
            if now - ts > (max_h + 600) * BAR_MS:            # data never arrived: give up on this row
                db.execute("UPDATE candidates SET labeled_ts=?, outcome='nodata' WHERE id=?", (now, r["id"]))
            continue
        k = db.klines(sym, since_ts=ts, until_ts=ts + (max_h - 1) * BAR_MS)
        if len(k) < max_h * 0.98:
            db.execute("UPDATE candidates SET labeled_ts=?, outcome='nodata' WHERE id=?", (now, r["id"]))
            continue
        # tolerate a rare missing bar by padding with the last close
        o, h, l, c = (k[x].to_numpy() for x in ("o", "h", "l", "c"))
        if len(k) < max_h:
            import numpy as np
            pad = max_h - len(k)
            o, h, l, c = (np.concatenate([a, np.full(pad, c[-1])]) for a in (o, h, l, c))
        lab = label_path(tc, r["side"], r["atr_pct"], o, h, l, c, ref_price=r["ref_price"])
        if lab is None:
            db.execute("UPDATE candidates SET labeled_ts=?, outcome='nodata' WHERE id=?", (now, r["id"]))
            continue
        db.execute("""UPDATE candidates SET label=?, outcome=?, ret_r=?, exit_ts=?, mfe=?, mae=?, grid=?, sl_pct=?,
                      labeled_ts=? WHERE id=?""",
                   (lab["label"], lab["outcome"], lab["ret_r"], ts + lab["bars"] * BAR_MS, lab["mfe"], lab["mae"],
                    json.dumps(lab["grid"], separators=(",", ":")), lab["sl_pct"], now, r["id"]))
        done += 1
    if done:
        log.info("labelled %d matured candidates", done)
    return done
