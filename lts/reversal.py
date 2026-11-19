"""Anatomy of a trade: what happened between entry, the best point (peak) and the exit.

Recorded for every grind (on the tracked plan) and every sent signal, winners and losers alike, so that
"went our way, reversed, hit the stop" trades can be compared with ones that reached the target.

Phases (5-minute candles):
  run   entry -> peak (the best price before the exit)
  turn  30 minutes either side of the peak
  fall  peak -> exit
For each phase: price speed, candle shape, volume vs the 6h before entry, taker flow (in the trade's
direction), BTC and the whole market, liquidations, open interest. Snapshots of the order book and
premium at entry, peak and exit. Levels near the peak. And exit clues: when price first closed back at
entry, below the 5m EMA20, or broke the last swing low of the run (for a long) before the stop.

All moves are side-aligned: + = in the trade's direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .db import DB

BAR, HOUR, DAY = 300_000, 3_600_000, 86_400_000
REVERSAL_MIN_MFE = 0.005           # "went our way first": at least +0.5% before the stop


def _k(db: DB, sym: str, a: int, b: int) -> pd.DataFrame:
    return db.frame("SELECT ts,o,h,l,c,v,qv,trades,tbqv FROM klines WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts",
                    [sym, a, b])


def _ret(db: DB, sym: str, a: int, b: int, side: int) -> float | None:
    """Side-aligned close-to-close return of another symbol between bar opens a and b."""
    r = db.query("SELECT ts, c FROM klines WHERE symbol=? AND ts IN (?, ?)", (sym, a, b))
    px = {x["ts"]: x["c"] for x in r}
    return side * (px[b] / px[a] - 1) if a in px and b in px and px[a] else None


def _market(db: DB, a: int, b: int, side: int, skip: str) -> dict:
    """Median side-aligned return of every coin between bars a and b, and the share moving against the trade."""
    if b <= a:
        return {}
    r = db.frame("SELECT symbol, ts, c FROM klines WHERE ts IN (?, ?)", [a, b])
    if not len(r):
        return {}
    w = r.pivot_table(index="symbol", columns="ts", values="c")
    if a not in w.columns or b not in w.columns:
        return {}
    ret = side * (w[b] / w[a] - 1)
    ret = ret.drop(labels=[skip], errors="ignore").dropna()
    if len(ret) < 20:
        return {}
    return {"mkt_med": float(ret.median()), "mkt_against_share": float((ret < 0).mean()), "mkt_n": int(len(ret))}


def _liq(db: DB, sym: str | None, a: int, b: int, side: int) -> dict:
    """Liquidations (USD) between a and b. 'with' = shorts squeezed for a long (the move's fuel);
    'against' = longs liquidated for a long (forced selling into the trade)."""
    try:
        q = "SELECT side, COALESCE(SUM(usd),0) u, COUNT(*) n FROM liquidations WHERE ts>=? AND ts<?"
        p: list = [a, b]
        if sym:
            q += " AND symbol=?"
            p.append(sym)
        rows = {r["side"]: (float(r["u"]), int(r["n"])) for r in db.query(q + " GROUP BY side", tuple(p))}
    except Exception:
        return {}
    with_side, against = ("short_liq", "long_liq") if side > 0 else ("long_liq", "short_liq")
    return {"liq_with_usd": rows.get(with_side, (0.0, 0))[0], "liq_against_usd": rows.get(against, (0.0, 0))[0],
            "liq_n": sum(v[1] for v in rows.values())}


def _book(db: DB, sym: str, t: int, side: int) -> dict | None:
    try:
        b = db.query("SELECT * FROM book_snapshots WHERE symbol=? AND ts<=? AND ts>? ORDER BY ts DESC LIMIT 1",
                     (sym, t, t - 15 * 60_000))
    except Exception:
        return None
    if not b:
        return None
    b = dict(b[0])
    def imb(x, y):
        return (x - y) / (x + y) if x is not None and y is not None and (x + y) > 0 else None
    out = {"age_min": (t - b["ts"]) / 60_000, "spread_bps": b["spread_bps"]}
    for w in ("05", "1", "2"):                                     # depth imbalance, + = our side heavier
        i = imb(b.get(f"bid{w}"), b.get(f"ask{w}"))
        out[f"imb_{w}"] = None if i is None else side * i
    # the wall in the way (asks for a long) and the wall behind (bids for a long)
    ahead, behind = ("ask", "bid") if side > 0 else ("bid", "ask")
    out["wall_ahead_dist"], out["wall_ahead_x"] = b.get(f"big_{ahead}_dist"), b.get(f"big_{ahead}_x")
    out["wall_behind_dist"], out["wall_behind_x"] = b.get(f"big_{behind}_dist"), b.get(f"big_{behind}_x")
    return out


def _oi(db: DB, sym: str, t: int) -> float | None:
    r = db.query("SELECT oi FROM oi WHERE symbol=? AND ts<=? AND ts>? ORDER BY ts DESC LIMIT 1", (sym, t, t - 2 * HOUR))
    return float(r[0]["oi"]) if r and r[0]["oi"] else None


def _premium(db: DB, sym: str, t: int, side: int) -> dict | None:
    r = db.query("SELECT mark, index_px, funding_rate FROM premium WHERE symbol=? AND ts<=? AND ts>? ORDER BY ts DESC LIMIT 1",
                 (sym, t, t - 30 * 60_000))
    if not r or not r[0]["index_px"]:
        return None
    return {"basis_bps": side * (r[0]["mark"] / r[0]["index_px"] - 1) * 1e4,
            "funding": None if r[0]["funding_rate"] is None else side * r[0]["funding_rate"]}


def _phase(k: pd.DataFrame, i0: int, i1: int, e: float, side: int, base_qv: float) -> dict:
    """Bars i0..i1 inclusive."""
    w = k.iloc[i0:i1 + 1]
    if not len(w):
        return {}
    o, h, l, c = (w[x].to_numpy(float) for x in ("o", "h", "l", "c"))
    qv, tb = w["qv"].to_numpy(float), w["tbqv"].to_numpy(float)
    body = side * (c - o) / e
    with_bar = body > 0
    buy_share = tb.sum() / qv.sum() if qv.sum() > 0 else np.nan
    flow = (2 * buy_share - 1) * side if np.isfinite(buy_share) else np.nan       # + = takers pushing our way
    adverse_run = 0
    best_run = 0
    for b in body:
        adverse_run = adverse_run + 1 if b < 0 else 0
        best_run = max(best_run, adverse_run)
    return {"minutes": len(w) * 5, "move": float(side * (c[-1] / o[0] - 1)),
            "speed_per_h": float(side * (c[-1] / o[0] - 1) / (len(w) / 12)),
            "rvol": float(qv.mean() / base_qv) if base_qv > 0 else None,
            "taker_flow": None if not np.isfinite(flow) else float(flow),
            "vol_share_against_candles": float(qv[~with_bar].sum() / qv.sum()) if qv.sum() > 0 else None,
            "biggest_candle_against": float(max(0.0, -body.min())),
            "most_candles_against_in_a_row": int(best_run)}


def anatomy(db: DB, sym: str, side: int, entry_ts: int, plan: dict, hours: int = 24, entry_px: float | None = None,
            levels: list | None = None) -> dict | None:
    """The trade from the first full 5m bar at/after entry_ts. None when the candles are not there yet."""
    start = (entry_ts + BAR - 1) // BAR * BAR
    pre = _k(db, sym, start - 6 * HOUR, start)
    k = _k(db, sym, start, start + hours * HOUR)
    if len(k) < 12:
        return None
    e = float(entry_px or k["o"].iloc[0])
    hi, lo, cl = k["h"].to_numpy(float), k["l"].to_numpy(float), k["c"].to_numpy(float)
    fav = (hi / e - 1) if side > 0 else (1 - lo / e)
    adv = (1 - lo / e) if side > 0 else (hi / e - 1)
    hs, ht = np.flatnonzero(adv >= plan["stop"]), np.flatnonzero(fav >= plan["target"])
    i_s, i_t = (hs[0] if len(hs) else 10 ** 9), (ht[0] if len(ht) else 10 ** 9)
    if i_s <= i_t and i_s < 10 ** 9:
        exit_kind, xi = "stop", int(i_s)
    elif i_t < 10 ** 9:
        exit_kind, xi = "target", int(i_t)
    elif len(k) >= hours * 12 * 0.98:
        exit_kind, xi = "time", len(k) - 1
    else:
        return None                                              # still open
    # the peak: best point strictly before the exit bar (the exit bar itself can hold both)
    pi = int(np.argmax(fav[:xi])) if xi > 0 else 0
    mfe = float(fav[:xi].max()) if xi > 0 else 0.0
    base_qv = float(pre["qv"].mean()) if len(pre) else float(k["qv"].iloc[:12].mean())
    ts = k["ts"].to_numpy()
    t_entry, t_peak, t_exit = int(ts[0]), int(ts[pi]), int(ts[xi])
    out = {"plan": f"{plan.get('leverage', 0)}x +{plan['target']:.2%} / -{plan['stop']:.2%}", "exit": exit_kind,
           "entry": e, "mfe": mfe, "mae_before_peak": float(adv[:pi + 1].max()),
           "reversal": bool(exit_kind == "stop" and mfe >= REVERSAL_MIN_MFE),
           "minutes_to_peak": (pi + 1) * 5, "minutes_peak_to_exit": (xi - pi) * 5, "minutes_to_exit": (xi + 1) * 5,
           "peak_hour_utc": (t_peak % DAY) / HOUR, "exit_hour_utc": (t_exit % DAY) / HOUR}
    # phases
    t0, t1 = max(0, pi - 6), min(xi, pi + 6)
    out["run"] = _phase(k, 0, pi, e, side, base_qv)
    out["turn"] = _phase(k, t0, t1, e, side, base_qv)
    out["fall"] = _phase(k, pi + 1, xi, e, side, base_qv) if xi > pi else {}
    # candle at the peak: wick against us (selling into the high for a long)
    o_p, h_p, l_p, c_p = (float(k[x].iloc[pi]) for x in ("o", "h", "l", "c"))
    rng = h_p - l_p
    if rng > 0:
        out["peak_wick_against"] = ((h_p - max(o_p, c_p)) if side > 0 else (min(o_p, c_p) - l_p)) / rng
    # BTC and the whole market in each phase
    for name, a, b in (("run", t_entry, t_peak), ("fall", t_peak, t_exit)):
        if b > a:
            out[name]["btc"] = _ret(db, "BTCUSDT", a, b, side)
            out[name].update(_market(db, a, b, side, sym))
            out[name].update(_liq(db, sym, a, b + BAR, side))
            m = _liq(db, None, a, b + BAR, side)
            out[name]["mkt_liq_against_usd"], out[name]["mkt_liq_with_usd"] = m.get("liq_against_usd"), m.get("liq_with_usd")
    # open interest, order book, premium at entry / peak / exit
    oi = {n: _oi(db, sym, t) for n, t in (("entry", t_entry), ("peak", t_peak), ("exit", t_exit))}
    if oi["entry"] and oi["peak"]:
        out["oi_chg_run"] = oi["peak"] / oi["entry"] - 1
    if oi["peak"] and oi["exit"]:
        out["oi_chg_fall"] = oi["exit"] / oi["peak"] - 1
    out["book"] = {n: _book(db, sym, t, side) for n, t in (("entry", t_entry), ("peak", t_peak), ("exit", t_exit))}
    out["premium"] = {n: _premium(db, sym, t, side) for n, t in (("entry", t_entry), ("peak", t_peak), ("exit", t_exit))}
    # levels: the nearest major level to the peak price (did it turn at one?)
    if levels:
        peak_px = e * (1 + side * mfe)
        near = sorted(levels, key=lambda x: abs(x["price"] / peak_px - 1))[0]
        out["peak_level"] = {"what": near["what"], "tf": near["tf"], "dist_from_peak": side * (near["price"] / peak_px - 1)}
    # exit clues after the peak (the first time each happened, minutes after entry; None = never before the exit)
    after = range(pi + 1, xi + 1)
    back = next((j for j in after if side * (cl[j] / e - 1) <= 0), None)
    out["first_close_back_at_entry_min"] = None if back is None else (back + 1) * 5
    ema = pd.Series(np.concatenate([pre["c"].to_numpy(float), cl])).ewm(span=20, adjust=False).mean().to_numpy()[len(pre):]
    below = next((j for j in after if side * (cl[j] - ema[j]) < 0), None)
    out["first_close_through_ema20_min"] = None if below is None else (below + 1) * 5
    # last swing low of the run (for a long; swing high for a short), 2 bars each side
    swing = None
    for j in range(2, pi - 1):
        x = lo if side > 0 else -hi
        if x[j] == x[j - 2:j + 3].min():
            swing = j
    if swing is not None:
        lvl = lo[swing] if side > 0 else hi[swing]
        brk = next((j for j in after if (cl[j] < lvl if side > 0 else cl[j] > lvl)), None)
        out["swing_of_run"] = side * (lvl / e - 1)
        out["first_close_through_swing_min"] = None if brk is None else (brk + 1) * 5
    # announcements for this coin from entry to exit
    base = sym[:-4] if sym.endswith("USDT") else sym
    try:
        ann = db.query("SELECT source, ts, kind, title FROM announcements WHERE base=? AND ts>=? AND ts<=?",
                       (base, t_entry - HOUR, t_exit + BAR))
        out["announcements"] = [{"source": a["source"], "min": (a["ts"] - t_entry) / 60_000, "kind": a["kind"],
                                 "title": (a["title"] or "")[:120]} for a in ann]
    except Exception:
        out["announcements"] = []
    return out
