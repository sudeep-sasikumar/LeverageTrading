"""Outcome tracking: open signals (1-minute resolution) and labelling of matured candidates."""
from __future__ import annotations

import json
import logging
import time

from . import BAR_MS
from .config import Config
from .db import DB
from .labels import label_grind, label_path
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


def first_minute(sig: dict) -> int:
    """First 1-minute candle to check: the first FULL minute after the signal was sent (the candle the signal was
    sent in opened before entry, so its high/low may be prices from before the trade), or the minute after the
    last one already checked."""
    t = max(sig.get("last_checked_ts") or sig["ts"], sig["ts"])
    return -(-t // MIN) * MIN


def add_gaps(sig: dict, gaps: list[list[int]]) -> str | None:
    old = json.loads(sig.get("track_gaps") or "[]")
    return json.dumps(old + gaps) if gaps else sig.get("track_gaps")


def missing(bars: list[tuple], start: int, end: int) -> list[list[int]]:
    """Stretches of [start, end) with no 1-minute candle (gaps of 3+ minutes; a quiet minute can be absent)."""
    out, t = [], start
    for b in bars:
        if b[0] - t >= 3 * MIN:
            out.append([t, b[0]])
        t = max(t, b[0] + MIN)
    if end - t >= 3 * MIN:
        out.append([t, end])
    return out


async def fetch_minutes(sym: str, start: int, end: int, primary, fallback) -> tuple[list[tuple], list[list[int]]]:
    """1-minute candles for [start, end) from the venue the user trades, paging back as far as needed (after a
    long downtime too); stretches it cannot cover are filled from the other venue, and anything still missing is
    returned as gaps."""
    bars = []
    try:
        bars = await primary.klines_range(sym, "1m", start, end)
    except Exception as e:
        log.warning("1m candles %s from %s failed: %s", sym, getattr(primary, "name", "?"), e)
    gaps = missing(bars, start, end)
    if gaps and fallback is not None:
        extra = []
        for a, b in gaps:
            try:
                extra += await fallback.klines_range(sym, "1m", a, b)
            except Exception as e:
                log.warning("1m candles %s fallback failed: %s", sym, e)
        if extra:
            bars = sorted({b[0]: b for b in bars + extra}.values())
            gaps = missing(bars, start, end)
    return bars, gaps


async def track_open_signals(cfg: Config, db: DB, binance, weex, notifier) -> None:
    now = int(time.time() * 1000)
    for row in db.query("SELECT s.*, y.on_weex FROM signals s LEFT JOIN symbols y ON y.symbol=s.symbol "
                        "WHERE s.status='open'"):
        sig = dict(row)
        ex, other = (weex, binance) if sig.get("on_weex") else (binance, None)   # track on the venue the user trades
        start = first_minute(sig)
        end = now // MIN * MIN                                                  # candles that have closed
        if end <= start:
            continue
        bars, gaps = await fetch_minutes(sig["symbol"], start, end, ex, other)
        if not bars and not gaps:
            continue
        status, px, ts, mfe, mae = evaluate_bars(sig, bars)
        if status is None and now >= sig["expires_ts"] and bars:
            status, px, ts = "time", bars[-1][4], now
        track_gaps = add_gaps(sig, gaps)
        if gaps:
            log.warning("signal #%d %s: no 1m candles for %d stretch(es), e.g. %s", sig["id"], sig["symbol"], len(gaps), gaps[0])
        if status:
            db.execute("UPDATE signals SET track_gaps=? WHERE id=?", (track_gaps, sig["id"]))
            closed = close_signal(cfg, db, sig, status, px, ts, mfe, mae)
            await notifier.send(exit_message(cfg, db, closed))
            log.info("signal #%d %s closed: %s %+.2fR", sig["id"], sig["symbol"], status, closed["ret_r"])
        else:
            db.execute("UPDATE signals SET mfe=?, mae=?, last_checked_ts=?, track_gaps=? WHERE id=?",
                       (mfe, mae, end, track_gaps, sig["id"]))


def label_matured(cfg: Config, db: DB) -> int:
    """Label live candidates whose full horizon is now in the kline store."""
    tc = cfg.trade
    mom_h = max([tc.horizon_bars] + list(tc.grid_horizon))
    last = db.last_kline_ts()
    now = int(time.time() * 1000)
    done = 0
    for r in db.query("SELECT id, symbol, ts, side, atr_pct, ref_price, kind FROM candidates "
                      "WHERE labeled_ts IS NULL AND ts<=?",
                      (now - min(mom_h, cfg.grind.horizon_bars) * BAR_MS,)):
        sym, ts = r["symbol"], r["ts"]
        grind = r["kind"] == "grind"
        max_h = cfg.grind.horizon_bars if grind else mom_h
        if now - ts < max_h * BAR_MS:
            continue
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
        if grind:
            lab = label_grind(tc, cfg.grind, r["side"], r["atr_pct"], o, h, l, c, ref_price=r["ref_price"])
        else:
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
