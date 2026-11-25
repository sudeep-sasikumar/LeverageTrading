"""Study B4 (docs/HYPOTHESES.md): liquidity sweeps joined with liquidations and order-book depth, with maker
entries back at the swept level. Built now; it REFUSES to evaluate until enough data exists.

Data limits, stated in every report:
- the Binance liquidation stream (!forceOrder@arr) sends at most one event per coin per second, so bursts are
  undercounted;
- order books are snapshotted every 5 minutes for about 120 coins (micro.book_symbols), so the book "at" a sweep
  is the last snapshot before it and the first after it, up to 5 minutes away, and many coins have none.

Periods: discovery = the first DISCOVERY_DAYS days of liquidation + book data; confirmation = everything after,
locked until it spans at least CONF_MIN_DAYS days AND the discovery rules have been frozen (lts.validate.Split).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .db import DB
from .fees import fees
from .sweeps import detect, plan

DAY, HOUR, MIN = 86_400_000, 3_600_000, 60_000
DISCOVERY_DAYS = 60
CONF_MIN_DAYS = 30
LIQ_WINDOW = 15 * MIN
CASCADE_X = 3.0
LIMITS_NOTE = ("Binance liquidation stream: at most one event per coin per second (bursts undercounted); "
               "order books snapshotted every 5 minutes for about 120 coins.")
# The grid, fixed before the first evaluation (family for the Holm correction).
GRID = {
    "liq_filter": ("any", "cascade_with_sweep"),      # liquidations of the swept side in the 15 min around the sweep
    "book_filter": ("any", "absorbing"),              # depth on the trade's side within 1% >= the other side's
    "stop": ("wick", "wide"),                         # sweeps.plan stop kinds
    "target": ("2R", "3R", "pool"),
    "side": (1, -1),
}


class NotEnoughData(RuntimeError):
    pass


def data_span(db: DB) -> tuple[int | None, int | None]:
    """First and last timestamps where BOTH liquidations and book snapshots exist."""
    a = db.query("SELECT MIN(ts) AS a, MAX(ts) AS b FROM liquidations")
    b = db.query("SELECT MIN(ts) AS a, MAX(ts) AS b FROM book_snapshots")
    if not a or not b or a[0]["a"] is None or b[0]["a"] is None:
        return None, None
    return max(a[0]["a"], b[0]["a"]), min(a[0]["b"], b[0]["b"])


def periods(db: DB, now: int | None = None) -> dict:
    """{"start", "conf_start", "end", "can_discover", "can_confirm", "first_run_date"}; never raises."""
    start, end = data_span(db)
    now = now or int(time.time() * 1000)
    if start is None:
        return {"start": None, "can_discover": False, "can_confirm": False, "first_run_date": None}
    conf_start = start + DISCOVERY_DAYS * DAY
    return {"start": start, "conf_start": conf_start, "end": end,
            "can_discover": end >= conf_start, "can_confirm": end >= conf_start + CONF_MIN_DAYS * DAY,
            "first_run_date": pd.to_datetime(conf_start, unit="ms").strftime("%Y-%m-%d"),
            "confirm_date": pd.to_datetime(conf_start + CONF_MIN_DAYS * DAY, unit="ms").strftime("%Y-%m-%d")}


def require(db: DB, phase: str = "discovery") -> dict:
    p = periods(db)
    if phase == "discovery" and not p["can_discover"]:
        raise NotEnoughData(f"B4 needs {DISCOVERY_DAYS} days of liquidation + order-book data; first run on "
                            f"{p.get('first_run_date') or 'unknown (no data yet)'}")
    if phase == "confirmation" and not p["can_confirm"]:
        raise NotEnoughData(f"B4 confirmation is locked until {CONF_MIN_DAYS} days after discovery "
                            f"({p.get('confirm_date')})")
    return p


def liq_context(liq: pd.DataFrame, t: int, side: int) -> dict:
    """USD liquidated on the swept side (longs for a low sweep, shorts for a high sweep) and the other side,
    15 minutes before / after t; liq has ts, side, usd for one coin."""
    swept = "long_liq" if side > 0 else "short_liq"
    other = "short_liq" if side > 0 else "long_liq"
    out = {}
    for name, a, b in (("before", t - LIQ_WINDOW, t), ("after", t, t + LIQ_WINDOW)):
        w = liq[(liq["ts"] > a) & (liq["ts"] <= b)]
        out[f"swept_{name}"] = float(w.loc[w["side"] == swept, "usd"].sum())
        out[f"other_{name}"] = float(w.loc[w["side"] == other, "usd"].sum())
    return out


def book_context(books: pd.DataFrame, t: int, side: int) -> dict:
    """Last snapshot before t and first after: depth on the trade's side vs the other within 0.5/1/2%."""
    out = {}
    before = books[books["ts"] <= t].tail(1)
    after = books[books["ts"] > t].head(1)
    for name, b in (("before", before), ("after", after)):
        if not len(b):
            continue
        r = b.iloc[0]
        for w in ("05", "1", "2"):
            mine, theirs = (r[f"bid{w}"], r[f"ask{w}"]) if side > 0 else (r[f"ask{w}"], r[f"bid{w}"])
            tot = (mine or 0) + (theirs or 0)
            out[f"imb{w}_{name}"] = float((mine - theirs) / tot) if tot else np.nan
        out[f"age_{name}_min"] = float(abs(r["ts"] - t) / MIN)
    return out


def events(db: DB, sym: str, since: int, until: int) -> list[dict]:
    """Sweeps on hourly bars in [since, until) with their liquidation and book context and maker-entry trades."""
    k = db.frame("SELECT ts,o,h,l,c,qv FROM klines_1h WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts",
                 [sym, since - 10 * DAY, until + 3 * DAY]).drop_duplicates("ts").reset_index(drop=True)
    if len(k) < 200:
        return []
    liq = db.frame("SELECT ts, side, usd FROM liquidations WHERE symbol=? AND ts>=? AND ts<?", [sym, since - DAY, until + DAY])
    books = db.frame("SELECT * FROM book_snapshots WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts", [sym, since - DAY, until + DAY])
    med15 = 0.0
    if len(liq):
        b = liq.assign(bin=liq["ts"] // LIQ_WINDOW).groupby("bin")["usd"].sum()
        full = (until - since) // LIQ_WINDOW
        med15 = float(np.median(np.r_[b.to_numpy(), np.zeros(max(0, int(full) - len(b)))])) if full > 0 else 0.0
    out = []
    fee = fees()
    for ev in detect(k):
        t = int(k["ts"].iloc[ev["i"]]) + HOUR                 # the sweep candle's close
        if not (since <= t < until):
            continue
        ctx = liq_context(liq, t, ev["side"]) if len(liq) else {}
        swept = ctx.get("swept_before", 0.0) + ctx.get("swept_after", 0.0)
        ctx["cascade"] = bool(swept >= CASCADE_X * max(med15, 1.0) and swept > 0)
        ctx.update(book_context(books, t, ev["side"]) if len(books) else {})
        out.append({"symbol": sym, "ts": t, "side": ev["side"], "level": float(ev["level"]), "wick": float(ev["wick"]),
                    "atr": float(ev["atr"]), "pool": float(ev["pool"]), "i": ev["i"], **ctx, "_k": k, "_fee": fee})
    return out


def maker_trade(ev: dict, stop_kind: str, target_kind: str, wait_h: int = 24, horizon_h: int = 48) -> dict | None:
    """Resting limit back at the swept level for up to wait_h hours; fills only if price trades THROUGH it
    (one tick is approximated by 0.01%); maker fee in, taker + slippage on stop/time, maker on the target."""
    k, fee = ev["_k"], ev["_fee"]
    o, h, l, c = (k[x].to_numpy(float) for x in "ohlc")
    s, lvl, i = ev["side"], ev["level"], ev["i"]
    for m in range(i + 1, min(i + 1 + wait_h, len(o))):
        p = plan(ev, lvl, stop_kind, target_kind)
        if p is None:
            return None
        stop, tgt = p
        through = (l[m] < lvl * (1 - 1e-4)) if s > 0 else (h[m] > lvl * (1 + 1e-4))
        if through:
            if (s > 0 and o[m] <= stop) or (s < 0 and o[m] >= stop):
                return None                                     # gapped through the stop
            risk = abs(lvl - stop) / lvl
            for j in range(m, min(m + horizon_h, len(o))):
                if (s > 0 and l[j] <= stop) or (s < 0 and h[j] >= stop):
                    return {"r": (-risk - fee.entry_maker - fee.stop) / risk, "exit": "stop", "fill_ts": int(k["ts"].iloc[m])}
                if j > m and ((s > 0 and h[j] >= tgt * (1 + 1e-4)) or (s < 0 and l[j] <= tgt * (1 - 1e-4))):
                    g = abs(tgt - lvl) / lvl
                    return {"r": (g - fee.entry_maker - fee.target_maker) / risk, "exit": "target", "fill_ts": int(k["ts"].iloc[m])}
            j = min(m + horizon_h, len(o)) - 1
            return {"r": (s * (c[j] / lvl - 1) - fee.entry_maker - fee.time) / risk, "exit": "time",
                    "fill_ts": int(k["ts"].iloc[m])}
        if (s > 0 and h[m] >= ev.get("pool", np.inf)) or (s < 0 and l[m] <= ev.get("pool", -np.inf)):
            return None
    return None


def evaluate(db: DB, phase: str = "discovery") -> pd.DataFrame:
    """Every sweep x grid cell in the phase's period. Raises NotEnoughData until the data allows it."""
    p = require(db, phase)
    since, until = (p["start"], p["conf_start"]) if phase == "discovery" else (p["conf_start"], p["end"])
    rows = []
    syms = [r["symbol"] for r in db.query("SELECT DISTINCT symbol FROM book_snapshots")]
    for sym in syms:
        for ev in events(db, sym, since, until):
            for stop in GRID["stop"]:
                for tgt in GRID["target"]:
                    tr = maker_trade(ev, stop, tgt)
                    if tr is None:
                        continue
                    absorbing = (ev.get("imb1_before", np.nan) >= 0)
                    rows.append({k: v for k, v in ev.items() if not k.startswith("_")} |
                                {"stop": stop, "target": tgt, "absorbing": bool(absorbing), **tr})
    return pd.DataFrame(rows)
