"""Pattern study: do classic chart / smart-money patterns carry an edge on crypto perps?

Every pattern, variant and exit plan is defined here BEFORE results are seen, and judged in two
stages: discovery on the first ~8 months, then confirmation on the last ~4 months with
everything frozen. Statistics are clustered by day (many coins trade the same market).

Each event: decision at a bar close (no look-ahead), entry at the next bar's open (market) or
at a level (limit, filled only if price trades through it), a structural stop (where the
pattern is proven wrong), targets of 1R/2R/3R and a time stop. Weex fees and slippage are
included: taker for market entries and stops, maker for limit entries and targets.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from .db import DB
from .grindlab import HOUR, DAY, _hourly, early_grind_mask, hourly_shape

from .fees import fees as _fees

_FEE = _fees()                                      # config.trade, the one source of fees
TAKER, MAKER, SLIP = _FEE.taker, _FEE.maker, _FEE.slip
MIN_STOP, MAX_STOP = 0.006, 0.08          # stop distance bounds (fraction of price)
MIN_QV24 = 5_000_000
EXITS = [(1.0, 24), (2.0, 24), (3.0, 24), (1.0, 72), (2.0, 72), (3.0, 72)]   # (target R, time stop hours)
LIMIT_WAIT_H = 48                          # a limit entry must fill within this many hours of the setup


# ---- helpers ----------------------------------------------------------------
def atr(h, l, c, n=14):
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    return pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()


def pivots(h, l, k):
    """Fractal swing highs/lows. Returns (is_high, is_low) at the pivot bar; usable only k bars later."""
    n = len(h)
    is_h, is_l = np.zeros(n, bool), np.zeros(n, bool)
    from numpy.lib.stride_tricks import sliding_window_view as sw
    if n > 2 * k:
        wh, wl = sw(h, 2 * k + 1), sw(l, 2 * k + 1)
        is_h[k:n - k] = h[k:n - k] == wh.max(axis=1)
        is_l[k:n - k] = l[k:n - k] == wl.min(axis=1)
    return is_h, is_l


def to_4h(k: pd.DataFrame) -> pd.DataFrame:
    """Hourly -> 4h bars aligned to 00/04/08/12/16/20 UTC (index = 4h bar open)."""
    g = (k.index.to_numpy() // (4 * HOUR)) * 4 * HOUR
    agg = k.groupby(g).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"),
                           qv=("qv", "sum"), n=("c", "size"))
    return agg[agg["n"] == 4].drop(columns="n")


def daily_from_hourly(k: pd.DataFrame) -> pd.DataFrame:
    g = (k.index.to_numpy() // DAY) * DAY
    agg = k.groupby(g).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), qv=("qv", "sum"), n=("c", "size"))
    return agg[agg["n"] == 24].drop(columns="n")


# ---- outcome simulation -----------------------------------------------------
def simulate(side: int, entry: float, stop: float, fill_i: int, h: np.ndarray, l: np.ndarray, c: np.ndarray,
             o: np.ndarray, entry_cost: float) -> dict | None:
    """Outcomes for every exit plan, starting at hourly bar fill_i (the entry bar).
    For a limit fill the entry bar is included (the stop can be hit in the same bar); a target is
    only counted from the next bar (conservative)."""
    risk = abs(entry - stop) / entry
    if not (MIN_STOP <= risk <= MAX_STOP):
        return None
    n = len(c)
    out = {"risk": risk}
    max_h = max(e[1] for e in EXITS)
    if fill_i + max_h > n:
        return None
    hh, ll, cc = h[fill_i:fill_i + max_h], l[fill_i:fill_i + max_h], c[fill_i:fill_i + max_h]
    if side > 0:
        fav, adv = hh / entry - 1, 1 - ll / entry
    else:
        fav, adv = 1 - ll / entry, hh / entry - 1
    stop_hits = np.flatnonzero(adv >= risk)
    first_stop = stop_hits[0] if len(stop_hits) else 10 ** 9
    for r_mult, hours in EXITS:
        tgt = r_mult * risk
        th = np.flatnonzero(fav[1:hours] >= tgt) + 1             # targets from the next bar on
        first_t = th[0] if len(th) else 10 ** 9
        fs = first_stop if first_stop < hours else 10 ** 9
        if fs <= first_t and fs < 10 ** 9:
            net = -risk - entry_cost - TAKER - SLIP
        elif first_t < 10 ** 9:
            net = tgt - entry_cost - MAKER
        else:
            net = side * (cc[hours - 1] / entry - 1) - entry_cost - TAKER - SLIP
        out[f"{r_mult:g}R_{hours}h"] = net / risk               # in R (risk units)
        out[f"pct_{r_mult:g}R_{hours}h"] = net                  # in % of price
    out["mfe72"], out["mae72"] = float(fav.max()), float(adv.max())
    return out


def market_entry(side, i, stop, h, l, c, o):
    """Decision at the close of bar i; enter at the open of bar i+1."""
    if i + 1 >= len(o):
        return None
    return simulate(side, o[i + 1], stop, i + 1, h, l, c, o, TAKER + SLIP)


def limit_entry(side, i, price, stop, h, l, c, o, wait=LIMIT_WAIT_H, invalid_beyond=None):
    """Setup known at the close of bar i; a resting limit at `price` fills on the first later bar that
    trades through it by 0.05% (i.e. a long fills when low <= price*0.9995). Cancelled if the stop
    level is crossed first or `invalid_beyond` (e.g. the target side) is reached first."""
    for j in range(i + 1, min(i + 1 + wait, len(o))):
        if side > 0:
            if invalid_beyond is not None and h[j] >= invalid_beyond:
                return None
            if l[j] <= price * 0.9995:
                if o[j] < stop:                                  # gapped through the stop: no fill
                    return None
                return simulate(side, min(price, o[j]), stop, j, h, l, c, o, MAKER)
        else:
            if invalid_beyond is not None and l[j] <= invalid_beyond:
                return None
            if h[j] >= price * 1.0005:
                if o[j] > stop:
                    return None
                return simulate(side, max(price, o[j]), stop, j, h, l, c, o, MAKER)
    return None


# ---- pattern detectors ------------------------------------------------------
# Each yields (pattern_id, side, decision_bar_index, result_dict). All levels use only past data.

def p_range_breakout(k, d_hi, d_lo, a1, liquid):
    """Close beyond the prior N-day high/low (first break in 3 days). Stop: back inside by 0.5 x 1h-ATR beyond the level,
    or 2 x 1h-ATR (variant). Market entry."""
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    for n_days, hi, lo in d_hi:
        lvl_hi, lvl_lo = hi, lo
        last_up = last_dn = -10 ** 9
        for i in range(1, len(c)):
            if not liquid[i] or not np.isfinite(a1[i]):
                continue
            if np.isfinite(lvl_hi[i]) and c[i] > lvl_hi[i] and c[i - 1] <= lvl_hi[i] and i - last_up > 72:
                last_up = i
                for sv, stop in (("lvl", lvl_hi[i] - 0.5 * a1[i]), ("atr", c[i] - 2 * a1[i])):
                    r = market_entry(1, i, stop, h, l, c, o)
                    if r:
                        yield f"breakout_{n_days}d_{sv}", 1, i, r
            if np.isfinite(lvl_lo[i]) and c[i] < lvl_lo[i] and c[i - 1] >= lvl_lo[i] and i - last_dn > 72:
                last_dn = i
                for sv, stop in (("lvl", lvl_lo[i] + 0.5 * a1[i]), ("atr", c[i] + 2 * a1[i])):
                    r = market_entry(-1, i, stop, h, l, c, o)
                    if r:
                        yield f"breakout_{n_days}d_{sv}", -1, i, r


def p_breakout_retest(k, d_hi, d_lo, a1, liquid):
    """After a close beyond the N-day level, a limit order back at the level (retest), stop 1 x 1h-ATR beyond it."""
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    for n_days, hi, lo in d_hi:
        last_up = last_dn = -10 ** 9
        for i in range(1, len(c)):
            if not liquid[i] or not np.isfinite(a1[i]):
                continue
            if np.isfinite(hi[i]) and c[i] > hi[i] and c[i - 1] <= hi[i] and i - last_up > 72:
                last_up = i
                lvl = hi[i]
                r = limit_entry(1, i, lvl * 1.001, lvl - a1[i], h, l, c, o, invalid_beyond=None)
                if r:
                    yield f"retest_{n_days}d", 1, i, r
            if np.isfinite(lo[i]) and c[i] < lo[i] and c[i - 1] >= lo[i] and i - last_dn > 72:
                last_dn = i
                lvl = lo[i]
                r = limit_entry(-1, i, lvl * 0.999, lvl + a1[i], h, l, c, o)
                if r:
                    yield f"retest_{n_days}d", -1, i, r


def p_52w(k, hi365, lo365, a1, liquid):
    """New 52-week high (long) / low (short) on an hourly close, first in 7 days. Stop 2 x 1h-ATR."""
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    last_up = last_dn = -10 ** 9
    for i in range(1, len(c)):
        if not liquid[i] or not np.isfinite(a1[i]):
            continue
        if np.isfinite(hi365[i]) and c[i] > hi365[i] and i - last_up > 168:
            last_up = i
            r = market_entry(1, i, c[i] - 2 * a1[i], h, l, c, o)
            if r:
                yield "52w_extreme", 1, i, r
        if np.isfinite(lo365[i]) and c[i] < lo365[i] and i - last_dn > 168:
            last_dn = i
            r = market_entry(-1, i, c[i] + 2 * a1[i], h, l, c, o)
            if r:
                yield "52w_extreme", -1, i, r


def p_sr_bounce(k, dly, a1, liquid):
    """Multi-touch daily S/R zone (>= 3 daily pivots within 0.5 x daily-ATR in 90 days). An hourly candle wicks into
    the zone and closes back out with a rejection wick >= 50% of its range -> trade away from the zone.
    Stop beyond the wick + 0.1 x 1h-ATR. Market entry."""
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    idx = k.index.to_numpy()
    dh, dl = dly["h"].to_numpy(), dly["l"].to_numpy()
    dts = dly.index.to_numpy()
    da = atr(dh, dl, dly["c"].to_numpy(), 14)
    ph, pl = pivots(dh, dl, 3)
    last = {}
    d_ptr = 0
    zones_by_day = {}
    for i in range(len(c)):
        if not liquid[i] or not np.isfinite(a1[i]):
            continue
        day_i = np.searchsorted(dts, idx[i] - DAY, side="right") - 1         # last daily bar fully closed
        if day_i < 30:
            continue
        if day_i not in zones_by_day:
            lo_j = max(0, day_i - 90)
            usable = np.arange(lo_j, day_i - 3 + 1)                            # pivots confirmed 3 days later
            lv = np.concatenate([dh[usable][ph[usable]], dl[usable][pl[usable]]])
            w = 0.5 * da[day_i] if np.isfinite(da[day_i]) else np.nan
            zones = []
            if len(lv) and np.isfinite(w):
                lv = np.sort(lv)
                for x in lv:
                    cnt = int(np.sum(np.abs(lv - x) <= w))
                    if cnt >= 3 and not any(abs(x - z) <= w for z in zones):
                        zones.append(float(x))
            zones_by_day = {day_i: (zones, w)}
        zones, w = zones_by_day[day_i]
        rng_ = h[i] - l[i]
        if rng_ <= 0:
            continue
        for z in zones:
            if l[i] <= z + w * 0.5 and c[i] > z + w * 0.5 and (min(o[i], c[i]) - l[i]) / rng_ >= 0.5 and c[i] > z:
                if i - last.get((z, 1), -10 ** 9) > 24:
                    last[(z, 1)] = i
                    r = market_entry(1, i, l[i] - 0.1 * a1[i], h, l, c, o)
                    if r:
                        yield "sr_bounce", 1, i, r
            if h[i] >= z - w * 0.5 and c[i] < z - w * 0.5 and (h[i] - max(o[i], c[i])) / rng_ >= 0.5 and c[i] < z:
                if i - last.get((z, -1), -10 ** 9) > 24:
                    last[(z, -1)] = i
                    r = market_entry(-1, i, h[i] + 0.1 * a1[i], h, l, c, o)
                    if r:
                        yield "sr_bounce", -1, i, r


def p_sweep(k, a1, liquid, rvol1):
    """Liquidity sweep: a candle trades beyond a liquidity pool (prior 24h high/low, prior 7-day high/low, equal
    highs/lows from the last 72h, or the previous UTC day's high/low) by >= 0.1 x ATR and closes back inside ->
    reversal. Stop beyond the sweep wick + 0.1 x ATR. Variants: pool type, with a volume spike (>= 2x)."""
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    n = len(c)
    hs, ls = pd.Series(h), pd.Series(l)
    pools = {"24h": (hs.shift(1).rolling(24).max().to_numpy(), ls.shift(1).rolling(24).min().to_numpy()),
             "7d": (hs.shift(1).rolling(168).max().to_numpy(), ls.shift(1).rolling(168).min().to_numpy())}
    idx = k.index.to_numpy()
    day = idx // DAY
    dfh = pd.DataFrame({"day": day, "h": h, "l": l})
    dh = dfh.groupby("day")["h"].max().shift(1)
    dl = dfh.groupby("day")["l"].min().shift(1)
    pools["prevday"] = (dh.reindex(day).to_numpy(), dl.reindex(day).to_numpy())
    # equal highs/lows: two 1h swing pivots (k=3) within 0.15% of each other in the last 72h, confirmed
    ph, pl = pivots(h, l, 3)
    eqh, eql = np.full(n, np.nan), np.full(n, np.nan)
    piv_h_idx, piv_l_idx = np.flatnonzero(ph), np.flatnonzero(pl)
    for i in range(n):
        ch = piv_h_idx[(piv_h_idx <= i - 3) & (piv_h_idx >= i - 72)]
        if len(ch) >= 2:
            v = np.sort(h[ch])
            d = np.diff(v) / v[:-1]
            j = np.flatnonzero(d <= 0.0015)
            if len(j):
                eqh[i] = v[j[-1] + 1]
        cl = piv_l_idx[(piv_l_idx <= i - 3) & (piv_l_idx >= i - 72)]
        if len(cl) >= 2:
            v = np.sort(l[cl])
            d = np.diff(v) / v[:-1]
            j = np.flatnonzero(d <= 0.0015)
            if len(j):
                eql[i] = v[j[0]]
    pools["equal"] = (eqh, eql)
    for name, (ph_, pl_) in pools.items():
        last_s = last_l = -10 ** 9
        for i in range(n):
            if not liquid[i] or not np.isfinite(a1[i]):
                continue
            buf = 0.1 * a1[i]
            if np.isfinite(ph_[i]) and h[i] >= ph_[i] + buf and c[i] < ph_[i] and i - last_s > 12:
                last_s = i
                r = market_entry(-1, i, h[i] + buf, h, l, c, o)
                if r:
                    yield f"sweep_{name}", -1, i, r
                    if rvol1[i] >= 2:
                        yield f"sweep_{name}_vol", -1, i, r
            if np.isfinite(pl_[i]) and l[i] <= pl_[i] - buf and c[i] > pl_[i] and i - last_l > 12:
                last_l = i
                r = market_entry(1, i, l[i] - buf, h, l, c, o)
                if r:
                    yield f"sweep_{name}", 1, i, r
                    if rvol1[i] >= 2:
                        yield f"sweep_{name}_vol", 1, i, r


def p_sweep_4h(k4, k, a1h_at, liquid_at):
    """Sweep of the prior 7-day (42 x 4h) high/low on a 4h candle that closes back inside -> reversal (entry on the
    next hourly open after the 4h close). Stop beyond the 4h wick."""
    o4, h4, l4, c4 = (k4[x].to_numpy() for x in "ohlc")
    hi = pd.Series(h4).shift(1).rolling(42).max().to_numpy()
    lo = pd.Series(l4).shift(1).rolling(42).min().to_numpy()
    a4 = atr(h4, l4, c4, 14)
    idx4 = k4.index.to_numpy()
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    pos = {t: j for j, t in enumerate(k.index.to_numpy())}
    for i in range(len(c4)):
        j = pos.get(int(idx4[i] + 3 * HOUR))                      # last hourly bar of this 4h bar
        if j is None or not liquid_at[j] or not np.isfinite(a4[i]):
            continue
        buf = 0.1 * a4[i]
        if np.isfinite(hi[i]) and h4[i] >= hi[i] + buf and c4[i] < hi[i]:
            r = market_entry(-1, j, h4[i] + buf, h, l, c, o)
            if r:
                yield "sweep4h_7d", -1, j, r
        if np.isfinite(lo[i]) and l4[i] <= lo[i] - buf and c4[i] > lo[i]:
            r = market_entry(1, j, l4[i] - buf, h, l, c, o)
            if r:
                yield "sweep4h_7d", 1, j, r


def _ob_fvg(tf_name, o, h, l, c, a, idx_map, k, liquid_at, disp_mult):
    """Order blocks and fair value gaps on one timeframe. Displacement: a candle with body >= disp_mult x ATR
    that closes beyond the most recent confirmed swing high (bullish) / low (bearish) = break of structure.
      OB : the last opposite-colour candle before the displacement; limit at the OB's near edge, stop beyond it.
      FVG: gap between candle[i-2] high and candle[i] low (bullish); limit at the gap's near edge, stop below
           candle[i-1] low (bullish) / above its high (bearish).
    Entries are hourly limit orders placed after the displacement candle closes."""
    ho, hh, hl, hc = (k[x].to_numpy() for x in "ohlc")
    ph, pl = pivots(h, l, 3)
    last_ph = last_pl = np.nan
    n = len(c)
    for i in range(4, n):
        if ph[i - 3]:
            last_ph = h[i - 3]
        if pl[i - 3]:
            last_pl = l[i - 3]
        j = idx_map(i)
        if j is None or not liquid_at[j] or not np.isfinite(a[i]):
            continue
        body = c[i] - o[i]
        if body >= disp_mult * a[i] and np.isfinite(last_ph) and c[i] > last_ph:
            # bullish displacement + BOS
            ob = next((m for m in range(i - 1, max(i - 6, 0), -1) if c[m] < o[m]), None)
            if ob is not None:
                r = limit_entry(1, j, h[ob], l[ob] - 0.1 * a[i], hh, hl, hc, ho, invalid_beyond=None)
                if r:
                    yield f"ob_{tf_name}_{disp_mult:g}", 1, j, r
            if l[i] > h[i - 2]:
                r = limit_entry(1, j, l[i], l[i - 1] - 0.1 * a[i] if l[i - 1] < h[i - 2] else h[i - 2] - 0.1 * a[i],
                                hh, hl, hc, ho)
                if r:
                    yield f"fvg_{tf_name}_{disp_mult:g}", 1, j, r
        if -body >= disp_mult * a[i] and np.isfinite(last_pl) and c[i] < last_pl:
            ob = next((m for m in range(i - 1, max(i - 6, 0), -1) if c[m] > o[m]), None)
            if ob is not None:
                r = limit_entry(-1, j, l[ob], h[ob] + 0.1 * a[i], hh, hl, hc, ho)
                if r:
                    yield f"ob_{tf_name}_{disp_mult:g}", -1, j, r
            if h[i] < l[i - 2]:
                r = limit_entry(-1, j, h[i], h[i - 1] + 0.1 * a[i] if h[i - 1] > l[i - 2] else l[i - 2] + 0.1 * a[i],
                                hh, hl, hc, ho)
                if r:
                    yield f"fvg_{tf_name}_{disp_mult:g}", -1, j, r


def p_ob_fvg(k, k4, a1, liquid):
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    for m in (1.5, 2.5):
        yield from _ob_fvg("1h", o, h, l, c, a1, lambda i: i, k, liquid, m)
    o4, h4, l4, c4 = (k4[x].to_numpy() for x in "ohlc")
    a4 = atr(h4, l4, c4, 14)
    pos = {t: j for j, t in enumerate(k.index.to_numpy())}
    idx4 = k4.index.to_numpy()
    for m in (1.5, 2.5):
        yield from _ob_fvg("4h", o4, h4, l4, c4, a4, lambda i: pos.get(int(idx4[i] + 3 * HOUR)), k, liquid, m)


def p_choch_4h(k, k4, liquid):
    """4h change of character: after >= 2 lower highs, a 4h close above the last swing high -> long (mirror for short).
    Stop below the most recent 4h swing low. Market entry on the next hourly open."""
    o4, h4, l4, c4 = (k4[x].to_numpy() for x in "ohlc")
    ph, pl = pivots(h4, l4, 2)
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    pos = {t: j for j, t in enumerate(k.index.to_numpy())}
    idx4 = k4.index.to_numpy()
    highs, lows = [], []
    for i in range(3, len(c4)):
        if ph[i - 2]:
            highs.append(h4[i - 2])
        if pl[i - 2]:
            lows.append(l4[i - 2])
        j = pos.get(int(idx4[i] + 3 * HOUR))
        if j is None or not liquid[j] or len(highs) < 3 or len(lows) < 3:
            continue
        if highs[-1] < highs[-2] < highs[-3] and c4[i] > highs[-1] and c4[i - 1] <= highs[-1]:
            r = market_entry(1, j, lows[-1], h, l, c, o)
            if r:
                yield "choch_4h", 1, j, r
        if lows[-1] > lows[-2] > lows[-3] and c4[i] < lows[-1] and c4[i - 1] >= lows[-1]:
            r = market_entry(-1, j, highs[-1], h, l, c, o)
            if r:
                yield "choch_4h", -1, j, r


def p_trend_pullback(k, k4, a1, liquid):
    """Momentum pullback: 4h uptrend (close > EMA50 and EMA20 > EMA50 on 4h); the hourly price had been >= 2 x ATR above
    its EMA20, pulls back to touch the EMA20, and closes back above it -> long. Stop below the pullback low. Mirror short."""
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    e20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    c4 = k4["c"]
    e20_4 = c4.ewm(span=20, adjust=False).mean()
    e50_4 = c4.ewm(span=50, adjust=False).mean()
    trend = np.sign((c4 > e50_4).astype(int) - (c4 < e50_4).astype(int)) * ((e20_4 > e50_4) == (c4 > e50_4))
    # 4h state known after the 4h bar closes: map to the hourly bars of the NEXT 4h bar
    t_idx = (k.index.to_numpy() // (4 * HOUR)) * 4 * HOUR - 4 * HOUR
    tr = pd.Series(trend.to_numpy(), index=k4.index).reindex(t_idx).to_numpy()
    stretched_up = pd.Series((c - e20) >= 2 * a1).rolling(24).max().to_numpy()
    stretched_dn = pd.Series((e20 - c) >= 2 * a1).rolling(24).max().to_numpy()
    last = {1: -10 ** 9, -1: -10 ** 9}
    for i in range(1, len(c)):
        if not liquid[i] or not np.isfinite(a1[i]) or not np.isfinite(tr[i]):
            continue
        if tr[i] > 0 and stretched_up[i - 1] == 1 and l[i] <= e20[i] and c[i] > e20[i] and i - last[1] > 24:
            last[1] = i
            r = market_entry(1, i, l[i] - 0.1 * a1[i], h, l, c, o)
            if r:
                yield "trend_pullback", 1, i, r
        if tr[i] < 0 and stretched_dn[i - 1] == 1 and h[i] >= e20[i] and c[i] < e20[i] and i - last[-1] > 24:
            last[-1] = i
            r = market_entry(-1, i, h[i] + 0.1 * a1[i], h, l, c, o)
            if r:
                yield "trend_pullback", -1, i, r


def p_squeeze(k, k4, liquid, a1):
    """Compression breakout: 4h Bollinger width in the lowest 10% of the last 90 days, then a 4h close beyond the
    prior 5-day (30 x 4h) range -> trade the breakout. Stop at the middle of that range."""
    o4, h4, l4, c4 = (k4[x].to_numpy() for x in "ohlc")
    s = pd.Series(c4)
    bw = (s.rolling(20).std() / s.rolling(20).mean()).to_numpy()
    pct = pd.Series(bw).rolling(540, min_periods=180).rank(pct=True).to_numpy()
    hi = pd.Series(h4).shift(1).rolling(30).max().to_numpy()
    lo = pd.Series(l4).shift(1).rolling(30).min().to_numpy()
    squeezed = pd.Series(pct <= 0.10).rolling(6).max().to_numpy()
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    pos = {t: j for j, t in enumerate(k.index.to_numpy())}
    idx4 = k4.index.to_numpy()
    for i in range(1, len(c4)):
        j = pos.get(int(idx4[i] + 3 * HOUR))
        if j is None or not liquid[j] or squeezed[i - 1] != 1:
            continue
        mid = (hi[i] + lo[i]) / 2
        if c4[i] > hi[i] and c4[i - 1] <= hi[i]:
            r = market_entry(1, j, mid, h, l, c, o)
            if r:
                yield "squeeze_breakout", 1, j, r
        if c4[i] < lo[i] and c4[i - 1] >= lo[i]:
            r = market_entry(-1, j, mid, h, l, c, o)
            if r:
                yield "squeeze_breakout", -1, j, r


def p_climax(k, a1, liquid, rvol1):
    """Volume climax reversal: hourly volume >= 5x normal, range >= 3 x ATR, close in the far third against the move
    (long lower wick after a dump -> long). Stop beyond the candle."""
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    for i in range(len(c)):
        if not liquid[i] or not np.isfinite(a1[i]) or rvol1[i] < 5:
            continue
        rng_ = h[i] - l[i]
        if rng_ < 3 * a1[i]:
            continue
        if (c[i] - l[i]) / rng_ >= 0.67 and (min(o[i], c[i]) - l[i]) / rng_ >= 0.5:
            r = market_entry(1, i, l[i] - 0.1 * a1[i], h, l, c, o)
            if r:
                yield "volume_climax", 1, i, r
        if (h[i] - c[i]) / rng_ >= 0.67 and (h[i] - max(o[i], c[i])) / rng_ >= 0.5:
            r = market_entry(-1, i, h[i] + 0.1 * a1[i], h, l, c, o)
            if r:
                yield "volume_climax", -1, i, r


def p_funding(k, fund, a1, liquid):
    """Funding extreme (contrarian): at a settlement where the rate <= -thr (shorts paying) -> long; >= +thr -> short.
    Stop 2 x 1h-ATR. Market entry on the next hour."""
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    if fund is None or not len(fund):
        return
    idx = k.index.to_numpy()
    pos = {t: j for j, t in enumerate(idx)}
    for ts, rate in zip(fund["ts"].to_numpy(), fund["rate"].to_numpy()):
        j = pos.get(int(ts // HOUR * HOUR - HOUR))                  # hourly bar that closes at the settlement
        if j is None or not liquid[j] or not np.isfinite(a1[j]):
            continue
        for thr in (0.0005, 0.001):
            if rate <= -thr:
                r = market_entry(1, j, c[j] - 2 * a1[j], h, l, c, o)
                if r:
                    yield f"funding_{thr * 100:g}pct", 1, j, r
            elif rate >= thr:
                r = market_entry(-1, j, c[j] + 2 * a1[j], h, l, c, o)
                if r:
                    yield f"funding_{thr * 100:g}pct", -1, j, r


def p_grind(k, a1, liquid):
    """Baseline from the grind lab: early-grind continuation. Stop 3% / market entry (as the lab plan)."""
    o, h, l, c = (k[x].to_numpy() for x in "ohlc")
    f = hourly_shape(k)
    for side in (1, -1):
        last = -10 ** 9
        for i in np.flatnonzero(early_grind_mask(f, side)):
            if not liquid[i] or i - last < 12:
                continue
            last = i
            r = market_entry(side, i, c[i] * (1 - side * 0.03), h, l, c, o)
            if r:
                yield "grind_continuation", side, i, r


# ---- cross-sectional momentum / reversal (daily, portfolio) ------------------
def cross_sectional(closes: pd.DataFrame, qv: pd.DataFrame, lookbacks=(1, 3, 7, 30), holds=(1, 3)) -> pd.DataFrame:
    """Each UTC day: rank liquid coins by their past N-day return; long the top 10% and short the bottom 10%
    (momentum) - the reversal strategy is its mirror. Returns one row per (day, lookback, hold) with the daily
    long-short return net of taker costs on both legs. No stops (portfolio)."""
    rows = []
    daily_c = closes
    liquid = qv.rolling(1).sum() >= MIN_QV24
    for lb in lookbacks:
        past = daily_c / daily_c.shift(lb) - 1
        for hold in holds:
            fwd = daily_c.shift(-hold) / daily_c - 1
            for t in daily_c.index[lb:-hold]:
                p, f, lq = past.loc[t], fwd.loc[t], liquid.loc[t]
                ok = p.notna() & f.notna() & lq
                if ok.sum() < 40:
                    continue
                pr = p[ok]
                top = pr[pr >= pr.quantile(0.9)].index
                bot = pr[pr <= pr.quantile(0.1)].index
                cost = 2 * (TAKER + SLIP)
                mom = f[top].mean() - f[bot].mean() - 2 * cost
                rows.append({"day": int(t), "lookback": lb, "hold": hold, "momentum": mom,
                             "reversal": -(f[top].mean() - f[bot].mean()) - 2 * cost,
                             "long_top": f[top].mean() - cost, "short_bottom": -f[bot].mean() - cost,
                             "long_bottom": f[bot].mean() - cost, "short_top": -f[top].mean() - cost})
    return pd.DataFrame(rows)


# ---- driver -----------------------------------------------------------------
def run(db: DB, days: float = 395.0, symbols: list[str] | None = None, log_every: int = 50) -> pd.DataFrame:
    t0 = time.time()
    now = int(time.time() * 1000)
    since = now - int(days * DAY)
    syms = symbols or [r["symbol"] for r in db.query("SELECT DISTINCT symbol FROM klines_1h")]
    fund_all = db.frame("SELECT symbol, ts, rate FROM funding WHERE ts>=?", [since])
    fund_by = {s: g.sort_values("ts") for s, g in fund_all.groupby("symbol")}
    events = []
    for n, sym in enumerate(syms, 1):
        k = _hourly(db, sym, since)
        if k is None or len(k) < 24 * 40:
            continue
        o, h, l, c = (k[x].to_numpy() for x in "ohlc")
        qv = k["qv"].to_numpy()
        a1 = atr(h, l, c, 14)
        liquid = pd.Series(qv).rolling(24).sum().to_numpy() >= MIN_QV24
        rvol1 = qv / pd.Series(qv).rolling(168, min_periods=72).mean().shift(1).to_numpy()
        k4 = to_4h(k)
        dly = daily_from_hourly(k)
        # prior N-day high/low known at each hour (from fully closed UTC days only)
        day_open = (k.index.to_numpy() // DAY) * DAY
        d_hi = []
        for nd in (20, 55):
            hi = dly["h"].rolling(nd).max()
            lo = dly["l"].rolling(nd).min()
            # value for day D uses days up to D-1
            hi_prev = hi.shift(1).reindex(day_open).to_numpy()
            lo_prev = lo.shift(1).reindex(day_open).to_numpy()
            d_hi.append((nd, hi_prev, lo_prev))
        # 52-week extremes from the daily store (full history), days strictly before today
        dfull = db.frame("SELECT ts, h, l FROM klines_1d WHERE symbol=? ORDER BY ts", [sym]).set_index("ts")
        hi365 = dfull["h"].rolling(365, min_periods=200).max().shift(1).reindex(day_open).to_numpy()
        lo365 = dfull["l"].rolling(365, min_periods=200).min().shift(1).reindex(day_open).to_numpy()
        gens = [p_range_breakout(k, d_hi, None, a1, liquid), p_breakout_retest(k, d_hi, None, a1, liquid),
                p_52w(k, hi365, lo365, a1, liquid), p_sr_bounce(k, dly, a1, liquid), p_sweep(k, a1, liquid, rvol1),
                p_sweep_4h(k4, k, a1, liquid), p_ob_fvg(k, k4, a1, liquid), p_choch_4h(k, k4, liquid),
                p_trend_pullback(k, k4, a1, liquid), p_squeeze(k, k4, liquid, a1), p_climax(k, a1, liquid, rvol1),
                p_funding(k, fund_by.get(sym), a1, liquid), p_grind(k, a1, liquid)]
        idx = k.index.to_numpy()
        for g in gens:
            for pid, side, i, r in g:
                events.append({"pattern": pid, "symbol": sym, "side": side, "ts": int(idx[i] + HOUR), **r})
        if n % log_every == 0:
            print(f"  patterns: {n}/{len(syms)} symbols, {len(events):,} events ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(events)


# ---- evaluation -------------------------------------------------------------
def evaluate(ev: pd.DataFrame, split_ts: int) -> pd.DataFrame:
    """Per pattern x side x exit, separately for discovery (< split) and confirmation (>= split):
    n, days, win rate, mean R, mean % of price, day-clustered t, share of positive months."""
    ev = ev.copy()
    ev["period"] = np.where(ev.ts < split_ts, "discovery", "confirm")
    ev["day"] = ev.ts // DAY
    ev["month"] = pd.to_datetime(ev.ts, unit="ms").dt.strftime("%Y-%m")
    rows = []
    for (pat, side, per), g in ev.groupby(["pattern", "side", "period"]):
        for tgt, hrs in EXITS:
            col = f"{tgt:g}R_{hrs}h"
            x = g[col].dropna()
            if len(x) < 30:
                continue
            daily = g.groupby("day")[col].mean()
            t_day = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))) if len(daily) > 5 and daily.std() > 0 else 0.0
            months = g.groupby("month")[col].mean()
            rows.append({"pattern": pat, "side": side, "period": per, "exit": col, "n": len(x), "days": len(daily),
                         "win": float((x > 0).mean()), "mean_R": float(x.mean()),
                         "mean_pct": float(g[f"pct_{col}"].mean()), "t_day": float(t_day),
                         "months_pos": float((months > 0).mean()), "n_months": int(len(months)),
                         "risk_med": float(g["risk"].median())})
    return pd.DataFrame(rows)
