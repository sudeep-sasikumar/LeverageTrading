"""Triple-barrier labelling and the trade plan arithmetic (stop, target, leverage, costs)."""
from __future__ import annotations

import math

import numpy as np

from .config import Trade


def costs(tc: Trade) -> dict:
    """Round-trip cost as a fraction of notional, by exit type."""
    market = tc.taker_fee + tc.slippage
    return {"tp": market + tc.maker_fee,      # market entry, limit take-profit
            "sl": 2 * market,                 # market entry, stop-market exit
            "time": 2 * market}


def stop_pct(tc: Trade, atr_pct: float, sl_mult: float | None = None) -> float | None:
    """Volatility-scaled stop distance; None when the coin is too volatile for a tight stop."""
    if atr_pct is None or not np.isfinite(atr_pct) or atr_pct <= 0:
        return None
    raw = (sl_mult if sl_mult is not None else tc.sl_atr_mult) * atr_pct
    if raw > tc.max_sl_pct:
        return None
    return max(raw, tc.min_sl_pct)


def plan_trade(tc: Trade, atr_pct: float, risk_per_trade: float = 0.01) -> dict | None:
    sl = stop_pct(tc, atr_pct)
    if sl is None:
        return None
    return plan_from_stop(tc, sl, tc.rr, risk_per_trade, tc.horizon_bars)


def grind_stop_pct(gc, atr_1h: float, sl_mult: float | None = None) -> float | None:
    if atr_1h is None or not np.isfinite(atr_1h) or atr_1h <= 0:
        return None
    raw = (sl_mult if sl_mult is not None else gc.sl_atr1h_mult) * atr_1h
    if raw > gc.max_sl_pct:
        return None
    return max(raw, gc.min_sl_pct)


def plan_grind(tc: Trade, gc, atr_1h: float, risk_per_trade: float = 0.01) -> dict | None:
    sl = grind_stop_pct(gc, atr_1h)
    return None if sl is None else plan_from_stop(tc, sl, gc.rr, risk_per_trade, gc.horizon_bars)


def plan_from_stop(tc: Trade, sl: float, rr: float, risk_per_trade: float, horizon_bars: int) -> dict:
    cst = costs(tc)
    tp = rr * sl
    lev = int(min(tc.max_leverage, max(1, math.ceil(tc.target_margin_return / max(tp - cst["tp"], 1e-6)))))
    # never let the stop sit beyond 60% of the way to liquidation
    while lev > 1 and sl > 0.6 / lev:
        lev -= 1
    loss_margin = lev * (sl + cst["sl"])
    return {"sl_pct": sl, "tp_pct": tp, "leverage": lev, "horizon_bars": horizon_bars,
            "gain_margin": lev * (tp - cst["tp"]), "loss_margin": loss_margin,
            "win_r": (tp - cst["tp"]) / sl, "loss_r": -(sl + cst["sl"]) / sl,
            "margin_frac": min(1.0, risk_per_trade / loss_margin)}


def barrier(side: int, entry: float, sl_pct: float, tp_pct: float, h: np.ndarray, l: np.ndarray,
            c: np.ndarray, horizon: int, cst: dict) -> dict | None:
    """First-touch outcome over bars [0, horizon). Bar 0 is the entry bar (entry at its open).

    When a bar touches both levels the stop is assumed to have been hit first (conservative).
    Returns None when the path is too short to decide.
    """
    n = min(horizon, len(h))
    if side > 0:
        sl_px, tp_px = entry * (1 - sl_pct), entry * (1 + tp_pct)
        hit_sl, hit_tp = l[:n] <= sl_px, h[:n] >= tp_px
    else:
        sl_px, tp_px = entry * (1 + sl_pct), entry * (1 - tp_pct)
        hit_sl, hit_tp = h[:n] >= sl_px, l[:n] <= tp_px
    i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else -1
    i_tp = int(np.argmax(hit_tp)) if hit_tp.any() else -1
    if i_sl >= 0 and (i_tp < 0 or i_sl <= i_tp):
        return {"outcome": "sl", "bars": i_sl + 1, "ret": -sl_pct, "ret_r": -(sl_pct + cst["sl"]) / sl_pct}
    if i_tp >= 0:
        return {"outcome": "tp", "bars": i_tp + 1, "ret": tp_pct, "ret_r": (tp_pct - cst["tp"]) / sl_pct}
    if n < horizon:
        return None
    ret = side * (c[n - 1] / entry - 1)
    return {"outcome": "time", "bars": n, "ret": ret, "ret_r": (ret - cst["time"]) / sl_pct}


def limit_entry(tc: Trade, side: int, limit_px: float, sl_pct: float, o: np.ndarray, h: np.ndarray,
                l: np.ndarray, c: np.ndarray, wait_bars: int = 3) -> list:
    """Research variant: a resting limit order at the trigger-bar close instead of a market entry.

    Pays the maker fee and no slippage on entry, but only fills if price trades back to the
    limit within `wait_bars` — so it misses the moves that run away immediately. Conservative
    fill-bar handling: a stop touched in the fill bar counts, a target touched there does not.
    Returns [outcome, net R, bars] with outcome 'nofill' when the order never fills.
    """
    touched = (l[:wait_bars] <= limit_px) if side > 0 else (h[:wait_bars] >= limit_px)
    if not touched.any():
        return ["nofill", 0.0, 0]
    j = int(np.argmax(touched))
    market = tc.taker_fee + tc.slippage
    cst = {"tp": 2 * tc.maker_fee, "sl": tc.maker_fee + market, "time": tc.maker_fee + market}
    hh, ll = h[j:].copy(), l[j:].copy()
    if side > 0:
        hh[0] = min(hh[0], limit_px)          # ignore favourable extremes of the fill bar
    else:
        ll[0] = max(ll[0], limit_px)
    out = barrier(side, limit_px, sl_pct, tc.rr * sl_pct, hh, ll, c[j:], tc.horizon_bars - wait_bars, cst)
    if out is None:
        return ["nofill", 0.0, 0]
    return [out["outcome"], round(out["ret_r"], 4), out["bars"] + j]


def excursions(side: int, entry: float, h: np.ndarray, l: np.ndarray) -> tuple[float, float]:
    """(max favourable, max adverse) excursion as positive fractions of entry."""
    if len(h) == 0:
        return 0.0, 0.0
    up, down = h.max() / entry - 1, 1 - l.min() / entry
    return (max(up, 0.0), max(down, 0.0)) if side > 0 else (max(down, 0.0), max(up, 0.0))


def label_grind(tc: Trade, gc, side: int, atr_1h: float, o: np.ndarray, h: np.ndarray, l: np.ndarray,
                c: np.ndarray, ref_price: float | None = None) -> dict | None:
    """Triple-barrier label for a grind candidate: 1h-ATR stop, gc.rr target, gc.horizon_bars."""
    n = gc.horizon_bars
    if len(o) < n:
        return None
    sl = grind_stop_pct(gc, atr_1h)
    entry = float(o[0])
    if sl is None or not np.isfinite(entry) or entry <= 0:
        return None
    cst = costs(tc)
    main = barrier(side, entry, sl, gc.rr * sl, h, l, c, n, cst)
    if main is None:
        return None
    mfe, mae = excursions(side, entry, h[:n], l[:n])
    grid = {}
    for m in gc.grid_sl_mult:
        s = grind_stop_pct(gc, atr_1h, m)
        if s is None:
            continue
        for rr in gc.grid_rr:
            g = barrier(side, entry, s, rr * s, h, l, c, n, cst)
            if g is not None:
                grid[f"{m}|{rr}|{n}"] = [g["outcome"], round(g["ret_r"], 4), g["bars"]]
    return {"label": int(main["outcome"] == "tp"), "outcome": main["outcome"], "ret_r": main["ret_r"],
            "bars": main["bars"], "mfe": mfe, "mae": mae, "sl_pct": sl, "entry": entry, "grid": grid}


def label_path(tc: Trade, side: int, atr_pct: float, o: np.ndarray, h: np.ndarray, l: np.ndarray,
               c: np.ndarray, ref_price: float | None = None) -> dict | None:
    """Label one candidate from the bars AFTER its trigger bar. Entry = open of the first bar."""
    max_h = max([tc.horizon_bars] + list(tc.grid_horizon))
    if len(o) < max_h:
        return None
    sl = stop_pct(tc, atr_pct)
    if sl is None:
        return None
    entry = float(o[0])
    if not np.isfinite(entry) or entry <= 0:
        return None
    cst = costs(tc)
    main = barrier(side, entry, sl, tc.rr * sl, h, l, c, tc.horizon_bars, cst)
    if main is None:
        return None
    mfe, mae = excursions(side, entry, h[:tc.horizon_bars], l[:tc.horizon_bars])
    grid = {}
    for m in tc.grid_sl_mult:
        s = stop_pct(tc, atr_pct, m)
        if s is None:
            continue
        for rr in tc.grid_rr:
            for hz in tc.grid_horizon:
                g = barrier(side, entry, s, rr * s, h, l, c, hz, cst)
                if g is not None:
                    grid[f"{m}|{rr}|{hz}"] = [g["outcome"], round(g["ret_r"], 4), g["bars"]]
    if ref_price:
        grid["limit"] = limit_entry(tc, side, float(ref_price), sl, o, h, l, c)
    return {"label": int(main["outcome"] == "tp"), "outcome": main["outcome"], "ret_r": main["ret_r"],
            "bars": main["bars"], "mfe": mfe, "mae": mae, "sl_pct": sl, "entry": entry, "grid": grid}
