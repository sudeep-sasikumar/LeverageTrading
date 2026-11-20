"""Early-exit rules, run in the shadow on every recorded grind (they never close a signal).

Tested on the year (DESIGN.md section 26): every rule cut the share of full -1% stops, but none improved the
result per trade, because the early exits also closed trades that went on to the target and paid fees on
many small losses. They are recorded live, per grind, so that verdict can be re-checked on live trades.

Plan: target and stop checked inside each 5-minute candle first (stop first). Rules decide on a candle's close
and exit at that close (taker fee + slippage). Taker flow = (2 x taker-buy volume / volume - 1) x side.
"""
from __future__ import annotations

import numpy as np

ENTRY = 0.0008 + 0.0004          # taker + slippage in
MKT = 0.0008 + 0.0004            # market exit
TGT = 0.0008                     # Weex TP fills as taker
RULES = {
    "none": "no early exit (target or stop or 24h)",
    "back_0.5": "after +0.5%, close back at entry",
    "ema_1": "after +1%, close through the 5m EMA20",
    "wick": "after +0.5%, a candle with a 60%+ wick against us (range 0.3%+)",
    "give_0.75": "after +1%, close 0.75% below the best point",
    "lock": "after +1.5%, stop moves to +0.5%",
    "flow15": "after +0.5%, 15-min taker flow -0.2 or worse and a lower close",
    "flow_ema": "after +0.5%, close through the EMA20 with 15-min taker flow against",
}


def shadow(post: np.ndarray, pre_close: np.ndarray, side: int, target: float, stop: float, hours: int = 24) -> dict:
    """post: (bars, 6) o,h,l,c,qv,tbqv from entry; pre_close: closes before entry (EMA seed).
    Returns {rule: {"net": ..., "exit": "target|stop|rule|time", "min": minutes}} or {} if data is missing."""
    B = min(hours * 12, len(post))
    if B < hours * 12 * 0.98 or not np.isfinite(post[0, 0]):
        return {}
    o, h, l, c, qv, tb = (post[:B, k].astype(float) for k in range(6))
    e = o[0]
    fav = (h / e - 1) if side > 0 else (1 - l / e)
    adv = (1 - l / e) if side > 0 else (h / e - 1)
    cl = side * (c / e - 1)
    flow = np.where(qv > 0, 2 * tb / np.where(qv > 0, qv, 1) - 1, 0.0) * side
    rng = h - l
    wick = np.where(rng > 0, ((h - np.maximum(o, c)) if side > 0 else (np.minimum(o, c) - l)) / np.where(rng > 0, rng, 1), 0)
    k = 2 / 21
    ema0 = None
    for x in pre_close:
        if np.isfinite(x):
            ema0 = x if ema0 is None else ema0 + k * (x - ema0)
    out = {}
    for rule in RULES:
        ema = ema0 if ema0 is not None else c[0]
        mfe, st, res = 0.0, -stop, None
        for j in range(B):
            if not np.isfinite(h[j]):
                continue
            if -adv[j] <= st:
                res = (st - ENTRY - MKT, "stop" if st <= -stop + 1e-12 else "rule", j)
                break
            if fav[j] >= target:
                res = (target - ENTRY - TGT, "target", j)
                break
            mfe = max(mfe, fav[j])
            ema = ema + k * (c[j] - ema)
            ema_tr = side * (ema / e - 1)
            f3 = float(np.mean(flow[max(0, j - 2):j + 1]))
            prev = cl[j - 1] if j > 0 else 0.0
            hit = False
            if rule == "back_0.5":
                hit = mfe >= 0.005 and cl[j] <= 0
            elif rule == "ema_1":
                hit = mfe >= 0.01 and cl[j] < ema_tr
            elif rule == "wick":
                hit = mfe >= 0.005 and wick[j] >= 0.6 and rng[j] / e >= 0.003
            elif rule == "give_0.75":
                hit = mfe >= 0.01 and cl[j] <= mfe - 0.0075
            elif rule == "flow15":
                hit = mfe >= 0.005 and f3 <= -0.2 and cl[j] < prev
            elif rule == "flow_ema":
                hit = mfe >= 0.005 and cl[j] < ema_tr and f3 < 0
            if hit:
                res = (cl[j] - ENTRY - MKT, "rule", j)
                break
            if rule == "lock" and mfe >= 0.015:
                st = max(st, 0.005)
        if res is None:
            res = (cl[B - 1] - ENTRY - MKT, "time", B - 1)
        out[rule] = {"net": float(res[0]), "exit": res[1], "min": int((res[2] + 1) * 5)}
    return out
