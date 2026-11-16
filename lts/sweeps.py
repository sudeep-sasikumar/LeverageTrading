"""Liquidity-sweep setups (the one pattern family with a confirmed timing edge, docs/DESIGN.md section 14).

A sweep: an hourly candle trades beyond the prior 24h high (or low) by at least 0.1 x ATR and closes back
inside. Shorts after high-sweeps, longs after low-sweeps. This module detects sweeps and simulates a small,
pre-declared grid of ways to execute them; the same detection drives the live scanner.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .patterns import MAKER, SLIP, TAKER, atr

HORIZON_H = 48
ENTRIES = ("market", "limit")          # next open, or a resting limit back at the swept level (24h)
STOPS = ("wick", "wide")               # wick + 0.1 ATR, or wick + 1.0 ATR
TARGETS = ("2R", "3R", "pool")         # fixed R, or the opposite side's prior-24h extreme
FILTERS = ("all", "vol2")              # every sweep, or only with hourly volume >= 2x normal


def detect(k: pd.DataFrame, min_qv24: float = 5_000_000) -> list[dict]:
    """All 24h-pool sweeps on hourly bars (decision at the sweep candle's close)."""
    o, h, l, c, qv = (k[x].to_numpy() for x in ("o", "h", "l", "c", "qv"))
    a1 = atr(h, l, c, 14)
    hi24 = pd.Series(h).shift(1).rolling(24).max().to_numpy()
    lo24 = pd.Series(l).shift(1).rolling(24).min().to_numpy()
    liquid = pd.Series(qv).rolling(24).sum().to_numpy() >= min_qv24
    rvol = qv / pd.Series(qv).rolling(168, min_periods=72).mean().shift(1).to_numpy()
    out, last = [], {1: -10 ** 9, -1: -10 ** 9}
    for i in range(24, len(c)):
        if not liquid[i] or not np.isfinite(a1[i]):
            continue
        buf = 0.1 * a1[i]
        if np.isfinite(hi24[i]) and h[i] >= hi24[i] + buf and c[i] < hi24[i] and i - last[-1] > 12:
            last[-1] = i
            out.append({"i": i, "side": -1, "level": hi24[i], "wick": h[i], "pool": lo24[i], "atr": a1[i],
                        "rvol": rvol[i], "close": c[i]})
        if np.isfinite(lo24[i]) and l[i] <= lo24[i] - buf and c[i] > lo24[i] and i - last[1] > 12:
            last[1] = i
            out.append({"i": i, "side": 1, "level": lo24[i], "wick": l[i], "pool": hi24[i], "atr": a1[i],
                        "rvol": rvol[i], "close": c[i]})
    return out


def plan(ev: dict, entry_px: float, stop_kind: str, target_kind: str) -> tuple[float, float] | None:
    """(stop price, target price) for a sweep at a given entry."""
    s = ev["side"]
    stop = ev["wick"] - s * (0.1 if stop_kind == "wick" else 1.0) * ev["atr"]
    risk = abs(entry_px - stop)
    if (stop - entry_px) * s >= 0 or risk <= 0:
        return None
    if target_kind == "pool":
        tgt = ev["pool"]
        if (tgt - entry_px) * s <= 0:
            return None
    else:
        tgt = entry_px + s * float(target_kind[0]) * risk
    return stop, tgt


def _run(side, entry, stop, tgt, j, h, l, c, entry_cost, horizon=HORIZON_H):
    """From bar j (entry bar). Stop first in a shared bar; target counted from the next bar."""
    n = min(horizon, len(c) - j)
    if n < horizon:
        return None
    risk = abs(entry - stop) / entry
    for m in range(n):
        hb, lb = h[j + m], l[j + m]
        if (side > 0 and lb <= stop) or (side < 0 and hb >= stop):
            return -risk - entry_cost - TAKER - SLIP, risk, "stop", m
        if m > 0 and ((side > 0 and hb >= tgt) or (side < 0 and lb <= tgt)):
            return abs(tgt - entry) / entry - entry_cost - MAKER, risk, "target", m
    return side * (c[j + n - 1] / entry - 1) - entry_cost - TAKER - SLIP, risk, "time", n


def simulate(ev: dict, k: pd.DataFrame, entry_kind: str, stop_kind: str, target_kind: str,
             min_risk=0.004, max_risk=0.10) -> dict | None:
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    i, s = ev["i"], ev["side"]
    if entry_kind == "market":
        if i + 1 >= len(o):
            return None
        j, entry, cost = i + 1, o[i + 1], TAKER + SLIP
    else:                                                    # limit back at the swept level, waits up to 24h
        lvl = ev["level"]
        j = None
        for m in range(i + 1, min(i + 25, len(o))):
            p = plan(ev, lvl, stop_kind, target_kind)
            if p is None:
                return None
            if (s < 0 and h[m] >= lvl * 1.0005) or (s > 0 and l[m] <= lvl * 0.9995):
                if (s < 0 and o[m] >= p[0]) or (s > 0 and o[m] <= p[0]):
                    return None                              # gapped through the stop
                j, entry, cost = m, (max(lvl, o[m]) if s < 0 else min(lvl, o[m])), MAKER
                break
            if (s < 0 and l[m] <= p[1]) or (s > 0 and h[m] >= p[1]):
                return None                                  # went to target without us
        if j is None:
            return None
    p = plan(ev, entry, stop_kind, target_kind)
    if p is None:
        return None
    r = _run(s, entry, p[0], p[1], j, h, l, c, cost)
    if r is None:
        return None
    net, risk, how, bars = r
    if not (min_risk <= risk <= max_risk):
        return None
    return {"net": net, "risk": risk, "R": net / risk, "exit": how, "bars": bars, "reward": abs(p[1] - entry) / entry}
