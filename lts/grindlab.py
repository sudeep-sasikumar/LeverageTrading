"""Grind lab: what do successful steady trends have in common?

Method (case-control at the moment you could have entered):
  1. Catalogue every steady 24h grind (>=10%, straight path, most hours one way, no single
     hour doing the work) over the stored hourly history - the "hindsight" list.
  2. Find every *early-grind* moment: an hour close where a coin has been moving steadily one
     way for ~6h. This is when a trader could enter.
  3. Label what happened next: did it go on to +TARGET before -STOP within 24h (a successful
     grinder) or not (a lookalike that fizzled)?
  4. Measure ~80 factors at that moment using only information available then: multi-timeframe
     support/resistance and room to run, compression before the move, volume and taker flow,
     funding, open interest, sector momentum, listing age, exchange announcements, market.
  5. Compare successes with lookalikes factor by factor (and stability over time), then fit a
     model on earlier days and test it on later days it never saw.
Results are stored for the dashboard and drive the paper-trading grind track.
"""
from __future__ import annotations

import json
import logging
import math
import time

import numpy as np
import pandas as pd

from .db import DB
from .features import strided_r2
from .grindlab_data import DAY, HOUR, clean_base, ensure_schema

log = logging.getLogger("lts.grindlab")

TARGET, STOP, HORIZON_H = 0.06, 0.03, 24       # success = +6% before -3% within 24h
TRAIL = 0.03                                    # trailing exit: give back 3% from the best point once 3% up
EXIT_GRID = [(0.03, 0.03), (0.04, 0.02), (0.05, 0.025), (0.06, 0.03), (0.08, 0.04), (0.10, 0.05), (0.12, 0.04)]
COST = 2 * (0.0008 + 0.0004)                   # Weex taker in + out with slippage
MIN_QV24 = 5_000_000                            # liquidity floor (24h quote volume)
DEDUPE_H = 12                                   # one early-grind decision per coin and side per 12h
MAIN_TAGS = ["AI", "Meme", "DeFi", "Layer-1", "Layer-2", "Gaming", "Infrastructure", "Alpha", "NFT", "Metaverse",
             "PoW", "Payment", "Storage", "RWA"]


# ---- loading ----------------------------------------------------------------
def _hourly(db: DB, sym: str, since: int) -> pd.DataFrame | None:
    k = db.frame("SELECT ts,o,h,l,c,v,qv,trades,tbqv FROM klines_1h WHERE symbol=? AND ts>=? ORDER BY ts", [sym, since])
    if len(k) < 24 * 10:
        return None
    k = k.drop_duplicates("ts").set_index("ts")
    full = np.arange(k.index[0], k.index[-1] + HOUR, HOUR, dtype=np.int64)
    if len(full) != len(k):
        k = k.reindex(full)
        c = k["c"].ffill()
        for col in ("o", "h", "l", "c"):
            k[col] = k[col].fillna(c)
        for col in ("v", "qv", "trades", "tbqv"):
            k[col] = k[col].fillna(0.0)
    return k.astype("float64")


def _daily(db: DB, sym: str) -> pd.DataFrame:
    return db.frame("SELECT ts,o,h,l,c,qv FROM klines_1d WHERE symbol=? ORDER BY ts", [sym]).set_index("ts")


# ---- shape on the hourly grid -----------------------------------------------
def hourly_shape(k: pd.DataFrame) -> pd.DataFrame:
    """Per-hour trailing-shape columns (raw, not side-aligned). Index = bar open; values at bar close."""
    c, h, l, qv = k["c"], k["h"], k["l"], k["qv"]
    y = np.log(c.to_numpy())
    f = pd.DataFrame(index=k.index)
    d1 = np.diff(y, prepend=np.nan)
    for n in (1, 3, 6, 12, 24, 72, 168, 720):
        f[f"r_{n}h"] = c / c.shift(n) - 1
    f["r2_6h"] = strided_r2(y, 7, 1)                    # open-of-window + 6 hourly closes
    f["r2_24h"] = strided_r2(y, 25, 1)
    up, dn = pd.Series(d1 > 0, index=k.index).astype(float), pd.Series(d1 < 0, index=k.index).astype(float)
    f["up6"], f["dn6"] = up.rolling(6).sum(), dn.rolling(6).sum()
    f["up24"], f["dn24"] = up.rolling(24).sum(), dn.rolling(24).sum()
    r6 = (np.log(c) - np.log(c.shift(6))).abs()
    f["big_up6"] = pd.Series(d1, index=k.index).rolling(6).max() / r6
    f["big_dn6"] = (-pd.Series(d1, index=k.index)).rolling(6).max() / r6
    r24 = (np.log(c) - np.log(c.shift(24))).abs()
    f["big_up24"] = pd.Series(d1, index=k.index).rolling(24).max() / r24
    f["big_dn24"] = (-pd.Series(d1, index=k.index)).rolling(24).max() / r24
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    f["atr_1h"] = tr.rolling(14).mean() / c
    f["atr_1h_30d"] = (tr / c).rolling(720, min_periods=168).mean()
    f["vol_exp"] = f["atr_1h"] / f["atr_1h_30d"]                               # volatility expansion
    lr = pd.Series(d1, index=k.index)
    f["rv_24h"] = lr.rolling(24).std()
    # compression BEFORE the last 6h: range of the prior 24h vs the 30-day typical 24h range
    rng24 = (h.rolling(24).max() - l.rolling(24).min()) / c
    f["coil_before"] = rng24.shift(6) / rng24.rolling(720, min_periods=168).median()
    f["bbw_pct"] = ((c.rolling(20).std() / c.rolling(20).mean())
                    .rolling(720, min_periods=168).rank(pct=True))                # BB width percentile (1h, 30d)
    f["qv_24h"] = qv.rolling(24).sum()
    base_h = qv.rolling(168, min_periods=72).mean().shift(6)
    f["rvol_6h"] = qv.rolling(6).sum() / (base_h * 6)
    f["rvol_24h"] = f["qv_24h"] / (qv.rolling(720, min_periods=168).mean().shift(24) * 24)
    f["vol_trend_3d"] = qv.rolling(72).sum() / (qv.rolling(720, min_periods=168).mean() * 72)
    tb = k["tbqv"]
    for n in (6, 24):
        tot = qv.rolling(n).sum().replace(0, np.nan)
        f[f"imb_{n}h"] = (2 * tb.rolling(n).sum() - tot) / tot
    ups = pd.Series(np.where(d1 > 0, qv, 0.0), index=k.index)
    f["upvol_share_24h"] = ups.rolling(24).sum() / qv.rolling(24).sum().replace(0, np.nan)
    tr_n = k["trades"]
    f["trades_rvol_6h"] = tr_n.rolling(6).sum() / (tr_n.rolling(168, min_periods=72).mean().shift(6) * 6)
    f["ats_ratio_6h"] = (qv.rolling(6).sum() / tr_n.rolling(6).sum().replace(0, np.nan)) / \
                        (qv.rolling(168, min_periods=72).sum() / tr_n.rolling(168, min_periods=72).sum().replace(0, np.nan))
    vwap24 = qv.rolling(24).sum() / k["v"].rolling(24).sum().replace(0, np.nan)
    f["d_vwap24"] = (c - vwap24) / (f["atr_1h"] * c)
    for n, name in ((48, "2d"), (168, "7d"), (720, "30d")):
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        f[f"pos_{name}"] = (c - lo) / (hi - lo).replace(0, np.nan)
    # 4h structure: highs/lows of the last 42 4h bars (7 days) excluding the last 6h
    f["hi_7d_prior"], f["lo_7d_prior"] = h.shift(6).rolling(162).max(), l.shift(6).rolling(162).min()
    f["hi_2d_prior"], f["lo_2d_prior"] = h.shift(6).rolling(42).max(), l.shift(6).rolling(42).min()
    f["hour"] = ((k.index.to_numpy() + HOUR) % DAY) / HOUR
    f["dow"] = (((k.index.to_numpy() + HOUR) // DAY) + 3) % 7
    return f


def early_grind_mask(f: pd.DataFrame, side: int, min_r6=0.03, min_r2=0.70, min_hours=4, max_big=0.5) -> np.ndarray:
    r6 = f["r_6h"].to_numpy() * side
    hours = (f["up6"] if side > 0 else f["dn6"]).to_numpy()
    big = (f["big_up6"] if side > 0 else f["big_dn6"]).to_numpy()
    with np.errstate(invalid="ignore"):
        return (r6 >= min_r6) & (f["r2_6h"].to_numpy() >= min_r2) & (hours >= min_hours) & (big <= max_big) & \
               (f["qv_24h"].to_numpy() >= MIN_QV24)


def hindsight_mask(f: pd.DataFrame, side: int) -> np.ndarray:
    """24h windows that were a steady grind of 10%+ (the daily-movers definition, on hourly bars)."""
    with np.errstate(invalid="ignore"):
        return (f["r_24h"].to_numpy() * side >= 0.10) & (f["r2_24h"].to_numpy() >= 0.60) & \
               ((f["up24"] if side > 0 else f["dn24"]).to_numpy() >= 14) & \
               ((f["big_up24"] if side > 0 else f["big_dn24"]).to_numpy() <= 0.45) & \
               (f["qv_24h"].to_numpy() >= MIN_QV24)


def outcome(side: int, entry: float, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> dict:
    """Path after entry (hourly bars, entry at the next bar's open). Stop counts first in a shared bar."""
    if side > 0:
        fav, adv = h / entry - 1, 1 - l / entry
    else:
        fav, adv = 1 - l / entry, h / entry - 1
    hit_t = np.flatnonzero(fav >= TARGET)
    hit_s = np.flatnonzero(adv >= STOP)
    it = hit_t[0] if len(hit_t) else 10 ** 9
    is_ = hit_s[0] if len(hit_s) else 10 ** 9
    if is_ <= it and is_ < 10 ** 9:
        res, r = "stop", (-STOP - COST) / STOP
    elif it < 10 ** 9:
        res, r = "target", (TARGET - COST) / STOP
    else:
        ret = side * (c[-1] / entry - 1)
        res, r = "time", (ret - COST) / STOP
    out = {"success": int(res == "target"), "result": res, "ret_r": r, "mfe": float(fav.max()), "mae": float(adv.max()),
           "fwd24": float(side * (c[-1] / entry - 1)), "hours_to_target": int(it + 1) if it < 10 ** 9 else None,
           "fwd12": float(side * (c[min(11, len(c) - 1)] / entry - 1))}
    # alternative exits, in % of the move (net of costs) - to judge exits on held-out days only
    for tgt, stp in EXIT_GRID:
        ht, hs = np.flatnonzero(fav >= tgt), np.flatnonzero(adv >= stp)
        i_t = ht[0] if len(ht) else 10 ** 9
        i_s = hs[0] if len(hs) else 10 ** 9
        if i_s <= i_t and i_s < 10 ** 9:
            out[f"x_{tgt:g}_{stp:g}"] = -stp - COST
        elif i_t < 10 ** 9:
            out[f"x_{tgt:g}_{stp:g}"] = tgt - COST
        else:
            out[f"x_{tgt:g}_{stp:g}"] = float(side * (c[-1] / entry - 1)) - COST
    # trailing stop: exit when price gives back TRAIL from its best close (or the initial STOP), 24h max
    best, exit_ret = 0.0, None
    closes = side * (c / entry - 1)
    for j in range(len(c)):
        if adv[j] >= STOP and best < STOP:
            exit_ret = -STOP
            break
        best = max(best, fav[j])
        if best >= STOP and closes[j] <= best - TRAIL:
            exit_ret = closes[j]
            break
    out["x_trail"] = (exit_ret if exit_ret is not None else float(closes[-1])) - COST
    return out


# ---- daily / weekly structure -----------------------------------------------
def daily_levels(d: pd.DataFrame, t_close: int, price: float, side: int, atr_abs: float) -> dict:
    """Support/resistance from daily and weekly history, using only daily bars closed before t."""
    d = d[d.index + DAY <= t_close]
    out = {}
    if len(d) < 10:
        return out
    hi, lo, cl = d["h"].to_numpy(), d["l"].to_numpy(), d["c"].to_numpy()
    out["listing_age_d"] = float(len(d))
    ath, atl = hi.max(), lo.min()
    for n, name in ((7, "7d"), (30, "30d"), (90, "90d"), (365, "52w")):
        if len(d) >= min(n, 10):
            h_n, l_n = hi[-n:].max(), lo[-n:].min()
            out[f"dpos_{name}"] = (price - l_n) / (h_n - l_n) if h_n > l_n else np.nan
            # beyond the prior range in the trade direction (a breakout / breakdown of that range)
            out[f"brk_{name}"] = float(price > h_n) if side > 0 else float(price < l_n)
            out[f"dist_{name}_atr"] = ((h_n - price) if side > 0 else (price - l_n)) / atr_abs
    out["dpos_ath"] = (price - atl) / (ath - atl) if ath > atl else np.nan
    out["beyond_ath"] = float(price > ath) if side > 0 else float(price < atl)
    out["from_ath"] = price / ath - 1
    # pivot levels: daily fractal highs/lows (confirmed 3 days later), last 365 days
    n = len(d)
    piv_hi = [hi[i] for i in range(max(3, n - 365), n - 3) if hi[i] == hi[i - 3:i + 4].max()]
    piv_lo = [lo[i] for i in range(max(3, n - 365), n - 3) if lo[i] == lo[i - 3:i + 4].min()]
    if side > 0:
        ahead = [p for p in piv_hi if p > price * 1.002]
        behind = [p for p in piv_hi + piv_lo if p < price]
        room = (min(ahead) / price - 1) if ahead else np.nan
    else:
        ahead = [p for p in piv_lo if p < price * 0.998]
        behind = [p for p in piv_hi + piv_lo if p > price]
        room = (1 - max(ahead) / price) if ahead else np.nan
    out["room_ahead"] = room if ahead else 1.0                                 # 1.0 = no level ahead ("blue sky")
    out["blue_sky"] = float(not ahead)
    out["levels_ahead_10pct"] = float(sum(1 for p in ahead if abs(p / price - 1) <= 0.10))
    out["level_behind_dist"] = (min(abs(price / p - 1) for p in behind) if behind else np.nan)
    # touches of the 30d extreme before now (a well-tested level that just broke tends to matter)
    ext = hi[-30:].max() if side > 0 else lo[-30:].min()
    touch = (np.abs(hi[-30:] / ext - 1) <= 0.015) if side > 0 else (np.abs(lo[-30:] / ext - 1) <= 0.015)
    out["touches_30d_ext"] = float(touch.sum())
    for m in (20, 50, 200):
        if len(cl) >= m:
            ma = cl[-m:].mean()
            out[f"d_ma{m}"] = side * (price / ma - 1)
            if len(cl) >= m + 5:
                out[f"ma{m}_slope"] = side * (ma / cl[-m - 5:-5].mean() - 1)
    streak = 0
    for x in np.diff(cl[-15:])[::-1]:
        if np.sign(x) == side:
            streak += 1
        else:
            break
    out["daily_streak"] = float(streak)
    wk = cl[-7:]
    out["week_ret"] = side * (price / cl[-7] - 1) if len(cl) >= 7 else np.nan
    if len(cl) >= 30:
        dr = np.abs(hi[-30:] - lo[-30:]) / cl[-30:]
        out["daily_range_30d"] = float(np.median(dr))
    return out


# ---- dataset ----------------------------------------------------------------
def load_context(db: DB, since: int) -> dict:
    """Everything shared across coins: market/sector medians per hour, tags, announcements, funding, OI.
    Built in a narrow first pass (3 columns per coin) so the nightly study fits a small VPS."""
    syms = [r["symbol"] for r in db.query("SELECT DISTINCT symbol FROM klines_1h")]
    tags = {}
    for r in db.query("SELECT symbol, tag FROM symbol_tags"):
        tags.setdefault(r["symbol"], set()).add(r["tag"])
    r24, r6 = {}, {}
    btc = None
    for sym in syms:
        k = _hourly(db, sym, since)
        if k is None:
            continue
        c, qv = k["c"], k["qv"]
        liquid = qv.rolling(24).sum() >= MIN_QV24
        r24[sym] = (c / c.shift(24) - 1).where(liquid)
        r6[sym] = (c / c.shift(6) - 1).where(liquid)
        if sym == "BTCUSDT":
            btc = pd.DataFrame({"r_6h": c / c.shift(6) - 1, "r_24h": c / c.shift(24) - 1, "r_168h": c / c.shift(168) - 1})
    w24, w6 = pd.DataFrame(r24), pd.DataFrame(r6)
    sector24 = {}
    for tag in MAIN_TAGS:
        members = [s for s in w24.columns if tag in tags.get(s, set())]
        if len(members) >= 4:
            sector24[tag] = w24[members]
    ann = db.frame("SELECT source, ts, base, kind FROM announcements")
    oi = db.frame("SELECT symbol, ts, oi FROM oi WHERE ts>=?", [since])
    lsr = db.frame("SELECT symbol, ts, top_pos, global_acc FROM lsr WHERE ts>=?", [since])
    fund = db.frame("SELECT symbol, ts, rate FROM funding WHERE ts>=?", [since - 3 * DAY])
    return {"syms": list(w24.columns), "tags": tags, "mkt24": w24.median(axis=1), "mkt6": w6.median(axis=1),
            "breadth": (w24 > 0).sum(axis=1) / w24.notna().sum(axis=1).replace(0, np.nan),
            "sector24": sector24, "btc": btc,
            "meta": {r["symbol"]: dict(r) for r in db.query("SELECT symbol, base FROM symbols")},
            "ann": {b: g.sort_values("ts") for b, g in ann.groupby("base")} if len(ann) else {},
            "oi": {s: g.sort_values("ts") for s, g in oi.groupby("symbol")},
            "lsr": {s: g.sort_values("ts") for s, g in lsr.groupby("symbol")},
            "fund": {s: g.sort_values("ts") for s, g in fund.groupby("symbol")}}


ANN_KINDS = ("list", "futures_launch", "delist", "monitoring", "alpha", "airdrop", "upbit_list", "upbit_warning",
             "upbit_delist")


def decision_row(sym: str, side: int, i: int, k: pd.DataFrame, f: pd.DataFrame, d: pd.DataFrame, ctx: dict,
                 ctx_ts: int | None = None) -> dict:
    """All factors for an early-grind decision at hourly bar i (values known at that bar's close).
    The nightly study and the live detector both use this function."""
    idx = k.index.to_numpy()
    t_close = int(idx[i] + HOUR)
    price = float(k["c"].iloc[i])
    fr = f.iloc[i]
    row = {"symbol": sym, "ts": t_close, "side": side, "price": price}
    for col in ("r_1h", "r_3h", "r_6h", "r_12h", "r_24h", "r_72h", "r_168h", "r_720h", "imb_6h", "imb_24h", "d_vwap24"):
        row[col] = side * fr[col]
    for col in ("r2_6h", "r2_24h", "atr_1h", "vol_exp", "rv_24h", "coil_before", "bbw_pct", "rvol_6h", "rvol_24h",
                "vol_trend_3d", "trades_rvol_6h", "ats_ratio_6h", "hour", "dow"):
        row[col] = fr[col]
    row["hours_with6"] = fr["up6"] if side > 0 else fr["dn6"]
    row["hours_with24"] = fr["up24"] if side > 0 else fr["dn24"]
    row["big6"] = fr["big_up6"] if side > 0 else fr["big_dn6"]
    row["upvol_share_24h"] = fr["upvol_share_24h"] if side > 0 else 1 - fr["upvol_share_24h"]
    for name in ("2d", "7d", "30d"):
        p = fr[f"pos_{name}"]
        row[f"pos_{name}"] = p if side > 0 else 1 - p
    hi7, lo7, hi2, lo2 = fr["hi_7d_prior"], fr["lo_7d_prior"], fr["hi_2d_prior"], fr["lo_2d_prior"]
    row["brk_2d_4h"] = float(price > hi2) if side > 0 else float(price < lo2)
    row["brk_7d_4h"] = float(price > hi7) if side > 0 else float(price < lo7)
    row["log_qv24"] = math.log10(max(fr["qv_24h"], 1.0)) if np.isfinite(fr["qv_24h"]) else np.nan
    row["log_price"] = math.log10(max(price, 1e-12))
    atr_abs = fr["atr_1h"] * price
    if np.isfinite(atr_abs) and atr_abs > 0:
        row.update(daily_levels(d, t_close, price, side, atr_abs))
    ts_i = idx[i] if ctx_ts is None else ctx_ts      # market context: the latest whole hour on a 5-minute scan
    row["mkt_r24"] = side * ctx["mkt24"].get(ts_i, np.nan)
    row["mkt_r6"] = side * ctx["mkt6"].get(ts_i, np.nan)
    b = ctx["breadth"].get(ts_i, np.nan)
    row["breadth"] = b if side > 0 else 1 - b
    btc = ctx["btc"]
    if btc is not None and ts_i in btc.index:
        row["btc_r6"], row["btc_r24"], row["btc_r168"] = (side * btc.at[ts_i, c] for c in ("r_6h", "r_24h", "r_168h"))
    row["rs_mkt24"] = row["r_24h"] - row["mkt_r24"] if np.isfinite(row["mkt_r24"]) else np.nan
    my_tags = ctx["tags"].get(sym, set())
    sec_vals, sec_same = [], []
    for tag in my_tags & set(ctx["sector24"]):
        w = ctx["sector24"][tag]
        if ts_i in w.index:
            peers = w.loc[ts_i].drop(labels=[sym], errors="ignore").dropna()
            if len(peers) >= 3:
                sec_vals.append(side * peers.median())
                sec_same.append(float((side * peers > 0.03).mean()))
    row["sector_r24"] = float(np.mean(sec_vals)) if sec_vals else np.nan
    row["sector_peers_moving"] = float(np.mean(sec_same)) if sec_same else np.nan
    for tag in MAIN_TAGS:
        row[f"tag_{tag}"] = float(tag in my_tags)
    fz = ctx["fund"].get(sym)
    if fz is not None:
        fz = fz[fz["ts"] <= t_close]
        if len(fz):
            row["funding"] = -side * fz["rate"].iloc[-1]
            row["funding_3d"] = -side * fz["rate"].iloc[-9:].mean()
            row["funding_chg"] = -side * (fz["rate"].iloc[-1] - fz["rate"].iloc[-9:].mean())
    osym = ctx["oi"].get(sym)
    if osym is not None and len(osym):
        o_now = osym[osym["ts"] <= t_close]
        if len(o_now) and t_close - o_now["ts"].iloc[-1] <= 2 * HOUR:
            for hrs in (6, 24):
                o_then = osym[osym["ts"] <= t_close - hrs * HOUR]
                if len(o_then) and t_close - hrs * HOUR - o_then["ts"].iloc[-1] <= 2 * HOUR:
                    row[f"oi_chg_{hrs}h"] = side * (o_now["oi"].iloc[-1] / o_then["oi"].iloc[-1] - 1)
    lsym = ctx["lsr"].get(sym)
    if lsym is not None and len(lsym):
        z = lsym[lsym["ts"] <= t_close]
        if len(z) >= 25 and t_close - z["ts"].iloc[-1] <= 2 * HOUR:
            row["lsr_glob_chg24"] = side * math.log(z["global_acc"].iloc[-1] / z["global_acc"].iloc[-25])
            row["lsr_top_chg24"] = side * math.log(z["top_pos"].iloc[-1] / z["top_pos"].iloc[-25])
            row["lsr_glob"] = side * math.log(z["global_acc"].iloc[-1])
    base = clean_base(ctx["meta"].get(sym, {}).get("base") or sym.replace("USDT", ""))
    for kind_ in ANN_KINDS:
        row[f"ann_{kind_}_7d"] = 0.0
    row["ann_any_48h"] = 0.0
    a_sym = ctx["ann"].get(base)
    if a_sym is not None:
        recent = a_sym[(a_sym["ts"] <= t_close) & (a_sym["ts"] > t_close - 7 * DAY)]
        for kind_ in set(recent["kind"]):
            if f"ann_{kind_}_7d" in row:
                row[f"ann_{kind_}_7d"] = 1.0
        row["ann_any_48h"] = float(((a_sym["ts"] <= t_close) & (a_sym["ts"] > t_close - 2 * DAY)).any())
    # catalyst found in the study: steady DOWN-grinds in the week after an Upbit listing / Binance futures launch
    row["catalyst_short"] = float(side < 0 and (row["ann_upbit_list_7d"] or row["ann_futures_launch_7d"]))
    return row


def build(db: DB, days: float = 120.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(early-grind decisions with factors and outcomes, hindsight grind episodes). One coin at a time."""
    ensure_schema(db)
    t0 = time.time()
    now = int(time.time() * 1000)
    since = now - int((days + 31) * DAY)
    start_ts = now - int(days * DAY)
    ctx = load_context(db, since)
    log.info("grindlab: context for %d symbols (%.0fs)", len(ctx["syms"]), time.time() - t0)
    rows, episodes = [], []
    for sym in ctx["syms"]:
        k = _hourly(db, sym, since)
        if k is None:
            continue
        f = hourly_shape(k)
        d = _daily(db, sym)
        idx = k.index.to_numpy()
        o, h, l, c = (k[x].to_numpy() for x in ("o", "h", "l", "c"))
        tag_str = ",".join(sorted(ctx["tags"].get(sym, set()) - {"Crypto"}))
        for side in (1, -1):
            hits = np.flatnonzero(hindsight_mask(f, side))
            if len(hits):
                for g_ in np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1):
                    a_, b_ = g_[0], g_[-1]
                    if idx[b_] < start_ts:
                        continue
                    s0 = max(a_ - 23, 0)
                    episodes.append({"symbol": sym, "side": side, "start": int(idx[s0]), "end": int(idx[b_] + HOUR),
                                     "move": float(side * (c[b_] / o[s0] - 1)), "hours": int(b_ - s0 + 1),
                                     "best24": float(np.nanmax(side * f["r_24h"].to_numpy()[g_])), "tags": tag_str})
            last_t = -10 ** 18
            for i in np.flatnonzero(early_grind_mask(f, side)):
                t_close = int(idx[i] + HOUR)
                if t_close < start_ts or i + 1 + HORIZON_H > len(c) or t_close - last_t < DEDUPE_H * HOUR:
                    continue
                last_t = t_close
                row = decision_row(sym, side, i, k, f, d, ctx)
                row.update(outcome(side, o[i + 1], h[i + 1:i + 1 + HORIZON_H], l[i + 1:i + 1 + HORIZON_H],
                                   c[i + 1:i + 1 + HORIZON_H]))
                rows.append(row)
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True) if rows else pd.DataFrame()
    log.info("grindlab: %d early-grind decisions, %d hindsight episodes (%.0fs)", len(df), len(episodes), time.time() - t0)
    return df, pd.DataFrame(episodes)


# ---- analysis ---------------------------------------------------------------
NON_FACTORS = {"symbol", "ts", "side", "price", "success", "result", "ret_r", "mfe", "mae", "fwd24", "fwd12", "hours_to_target",
               "fold", "x_trail"} | {f"x_{t:g}_{s:g}" for t, s in EXIT_GRID}


def factor_table(df: pd.DataFrame, n_folds: int = 3) -> pd.DataFrame:
    """For each factor: success rate in its top vs bottom fifth (or with vs without), and whether the
    direction of the effect held in every time period."""
    df = df.copy()
    df["fold"] = pd.qcut(df["ts"].rank(method="first"), n_folds, labels=False)
    base = df["success"].mean()
    out = []
    for col in [c for c in df.columns if c not in NON_FACTORS]:
        x = df[col]
        if x.notna().mean() < 0.15 or x.nunique() < 2:
            continue
        binary = set(x.dropna().unique()) <= {0.0, 1.0}
        def split(d):
            v = d[col]
            if binary:
                return d[v == 1], d[v == 0]
            v = v.dropna()
            if v.nunique() < 5:
                return None, None
            q_lo, q_hi = v.quantile(0.2), v.quantile(0.8)
            return d[d[col] >= q_hi], d[d[col] <= q_lo]
        hi, lo = split(df)
        if hi is None or len(hi) < 25 or len(lo) < 25:
            continue
        diffs = []
        for _, g in df.groupby("fold"):
            a, b = split(g)
            if a is None or len(a) < 8 or len(b) < 8:
                diffs.append(np.nan)
            else:
                diffs.append(a["success"].mean() - b["success"].mean())
        diff = hi["success"].mean() - lo["success"].mean()
        p1, p2, n1, n2 = hi["success"].mean(), lo["success"].mean(), len(hi), len(lo)
        pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(max(pooled * (1 - pooled) * (1 / n1 + 1 / n2), 1e-12))
        same_sign = np.nansum(np.sign(diffs) == np.sign(diff)) if np.isfinite(diff) else 0
        out.append({"factor": col, "binary": binary, "n_high": n1, "n_low": n2,
                    "success_high": p1, "success_low": p2, "lift": diff, "z": diff / se,
                    "folds_agree": f"{int(same_sign)}/{int(np.sum(np.isfinite(diffs)))}",
                    "stable": bool(same_sign == np.sum(np.isfinite(diffs)) and np.sum(np.isfinite(diffs)) >= 2),
                    "median_success": float(df.loc[df.success == 1, col].median()) if not binary else float(df.loc[df.success == 1, col].mean()),
                    "median_fail": float(df.loc[df.success == 0, col].median()) if not binary else float(df.loc[df.success == 0, col].mean()),
                    "cut_high": None if binary else float(df[col].quantile(0.8)),
                    "cut_low": None if binary else float(df[col].quantile(0.2)),
                    "base": base})
    t = pd.DataFrame(out)
    if t.empty:
        return t
    return t.reindex(t["z"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def walk_forward(df: pd.DataFrame, feats: list[str], train_frac=(0.5, 0.65, 0.8)) -> dict:
    """Fit on earlier days, score later days it never saw (expanding window, purged by the 24h horizon)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    df = df.sort_values("ts").reset_index(drop=True)
    tmin, tmax = df.ts.min(), df.ts.max()
    parts = []
    edges = [tmin + f * (tmax - tmin) for f in train_frac] + [tmax + 1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        tr = df[df.ts < lo - HORIZON_H * HOUR]
        te = df[(df.ts >= lo) & (df.ts < hi)]
        if len(tr) < 300 or len(te) < 50 or tr.success.nunique() < 2:
            continue
        use = usable(tr, feats)                     # e.g. open interest only exists for the last ~30 days
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=60,
                                           l2_regularization=5.0, random_state=7)
        m.fit(tr[use].to_numpy(float), tr.success.to_numpy())
        parts.append(te.assign(score=m.predict_proba(te[use].to_numpy(float))[:, 1]))
    if not parts:
        return {"ok": False}
    oos = pd.concat(parts)
    res = {"ok": True, "n_oos": len(oos), "auc": float(roc_auc_score(oos.success, oos.score)) if oos.success.nunique() == 2 else float("nan"),
           "base_success": float(oos.success.mean()), "base_r": float(oos.ret_r.mean())}
    for q, name in ((0.8, "top20"), (0.9, "top10"), (0.95, "top5")):
        top = oos[oos.score >= oos.score.quantile(q)]
        res[name] = {"n": len(top), "success": float(top.success.mean()), "avg_r": float(top.ret_r.mean()),
                     "t": float(top.ret_r.mean() / (top.ret_r.std(ddof=1) / math.sqrt(len(top)))) if len(top) > 2 else 0.0}
    day = oos.ts // DAY
    best = oos.loc[oos.groupby(day).score.idxmax()]
    res["best_per_day"] = {"n": len(best), "success": float(best.success.mean()), "avg_r": float(best.ret_r.mean()),
                           "days_win": float((best.ret_r > 0).mean())}
    best2 = oos.sort_values("score", ascending=False).groupby(day).head(2)
    res["best2_per_day"] = {"n": len(best2), "success": float(best2.success.mean()), "avg_r": float(best2.ret_r.mean())}
    res["oos"] = oos[["symbol", "ts", "side", "score", "success", "ret_r"]]
    return res


def model_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FACTORS and df[c].notna().mean() >= 0.15 and df[c].nunique() > 1]


def usable(df: pd.DataFrame, feats: list[str]) -> list[str]:
    return [f for f in feats if df[f].notna().sum() >= 50 and df[f].nunique() > 1]


def fit_final(df: pd.DataFrame, feats: list[str]):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=60,
                                       l2_regularization=5.0, random_state=7)
    m.fit(df[feats].to_numpy(float), df.success.to_numpy())
    return m


def permutation_importance(df: pd.DataFrame, feats: list[str], n_repeats: int = 3) -> pd.DataFrame:
    """Importance on the last third (trained on the first two thirds): drop in AUC when a factor is shuffled."""
    from sklearn.metrics import roc_auc_score
    df = df.sort_values("ts")
    cut = df.ts.quantile(2 / 3)
    tr, te = df[df.ts < cut - HORIZON_H * HOUR], df[df.ts >= cut]
    feats = usable(tr, feats)
    m = fit_final(tr, feats)
    X = te[feats].to_numpy(float)
    base = roc_auc_score(te.success, m.predict_proba(X)[:, 1])
    rng = np.random.default_rng(0)
    rows = []
    for j, ft in enumerate(feats):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - roc_auc_score(te.success, m.predict_proba(Xp)[:, 1]))
        rows.append({"factor": ft, "auc_drop": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True).assign(base_auc=base)


# ---- checklist: the stable traits of successful grinders ---------------------
# (factor, direction, label). Thresholds are learned nightly from the data (training part only).
CHECKS = [
    ("btc_r168", "pos", "BTC's week is going the same way"),
    ("brk_30d", "brk", "breaking out of / at the edge of its 30-day range"),
    ("week_ret", "pos", "the coin's own week is going the same way"),
    ("rs_mkt24", "pos", "stronger than the market over 24h"),
    ("imb_6h", "low", "no heavy aggressive buying/selling in the last 6h"),
    ("mkt_r6", "low", "the move is coin-specific (market not moving the same way)"),
    ("levels_ahead_10pct", "zero", "no daily support/resistance within 10% ahead"),
    ("sector_r24", "pos", "its sector is moving the same way"),
    ("funding", "nonneg", "funding is not against the trade"),
    ("listing_age_d", "low", "younger listing"),
    ("rvol_6h", "high", "volume above normal"),
    ("catalyst_short", "one", "down-grind after an Upbit listing / Binance futures launch"),
]


def learn_thresholds(train: pd.DataFrame) -> dict:
    return {c: float(train[c].median()) for c, mode, _ in CHECKS if mode in ("low", "high") and c in train}


def checklist(d: pd.DataFrame, thr: dict) -> pd.DataFrame:
    out = {}
    nan = pd.Series(np.nan, index=d.index)
    for col, mode, label in CHECKS:
        x = d[col] if col in d else nan
        if col == "brk_30d":
            ok = (x == 1) | ((d["pos_30d"] if "pos_30d" in d else nan) >= 0.8)
        elif mode == "pos":
            ok = x.fillna(0) > 0
        elif mode == "low":
            ok = x <= thr.get(col, np.inf)
        elif mode == "high":
            ok = x >= thr.get(col, -np.inf)
        elif mode == "zero":
            ok = x == 0
        elif mode == "nonneg":
            ok = x.fillna(0) >= 0
        else:
            ok = x == 1
        out[label] = ok.fillna(False).astype(bool)
    return pd.DataFrame(out, index=d.index)


EXIT_LABELS = {"x_0.06_0.03": "+6% / -3%", "x_0.1_0.05": "+10% / -5%", "x_0.04_0.02": "+4% / -2%", "x_trail": "trailing 3%"}


def _exit_stats(g: pd.DataFrame) -> dict:
    out = {"n": int(len(g)), "success": float(g.success.mean()) if len(g) else None,
           "best_med": float(g.mfe.median()) if len(g) else None, "worst_med": float(g.mae.median()) if len(g) else None}
    for col in EXIT_LABELS:
        if col in g and len(g):
            x = g[col]
            out[col] = float(x.mean())
            out[col + "_t"] = float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0
    return out


def study(df: pd.DataFrame, ep: pd.DataFrame) -> dict:
    """The nightly report: what successful grinders share, and whether it held on later, unseen days."""
    df = df.sort_values("ts").reset_index(drop=True)
    cut = df.ts.quantile(0.6)
    train, test = df[df.ts < cut - DAY], df[df.ts >= cut]
    thr = learn_thresholds(train)
    ck_tr, ck_te = checklist(train, thr), checklist(test, thr)
    traits = []
    for label in ck_tr.columns:
        a, b = train[ck_tr[label]], train[~ck_tr[label]]
        c, e = test[ck_te[label]], test[~ck_te[label]]
        ok = len(a) and len(b) and len(c) and len(e)
        traits.append({"trait": label,
                       "train_with": float(a.success.mean()) if len(a) else None,
                       "train_without": float(b.success.mean()) if len(b) else None,
                       "test_with": float(c.success.mean()) if len(c) else None,
                       "test_without": float(e.success.mean()) if len(e) else None,
                       "n_test_with": int(len(c)),
                       "held": bool(ok and (a.success.mean() - b.success.mean()) * (c.success.mean() - e.success.mean()) > 0)})
    score_te = ck_te.sum(axis=1)
    by_score = [{"score": int(s_), **_exit_stats(g)} for s_, g in test.groupby(np.minimum(score_te, 9))]
    cat = {}
    for side, name in ((-1, "short"), (1, "long")):
        news = (df["ann_upbit_list_7d"] == 1) | (df["ann_futures_launch_7d"] == 1)
        g = df[news & (df.side == side)]
        cat[name] = {**_exit_stats(g), "coins": int(g.symbol.nunique()) if len(g) else 0}
    ft = factor_table(df, n_folds=4)
    stable = ft[ft.stable & (ft.z.abs() >= 3)].head(25) if len(ft) else pd.DataFrame(
        columns=["factor", "success_high", "success_low", "lift", "z", "folds_agree", "median_success", "median_fail"])
    wf = walk_forward(df, model_features(df))
    days = (df.ts.max() - df.ts.min()) / DAY
    return {
        "built_ts": int(time.time() * 1000), "days": float(days), "n_decisions": int(len(df)),
        "decisions_per_day": float(len(df) / max(days, 1)), "base_success": float(df.success.mean()),
        "target": TARGET, "stop": STOP, "horizon_h": HORIZON_H,
        "n_episodes": int(len(ep)), "episodes_per_day": float(len(ep) / max(days, 1)),
        "episodes_up": int((ep.side > 0).sum()) if len(ep) else 0,
        "episodes_down": int((ep.side < 0).sum()) if len(ep) else 0,
        "episode_median_move": float(ep.move.abs().median()) if len(ep) else None,
        "thresholds": thr, "traits": traits, "by_score": by_score, "catalyst": cat,
        "test_from_ts": int(cut), "base_test": _exit_stats(test),
        "stable_factors": stable[["factor", "success_high", "success_low", "lift", "z", "folds_agree",
                                  "median_success", "median_fail"]].to_dict("records"),
        "model": {k: v for k, v in wf.items() if k != "oos"} if wf.get("ok") else {"ok": False},
    }


# ---- live detection ----------------------------------------------------------
def live_scan(db: DB, ctx: dict | None = None, now_ms: int | None = None) -> list[dict]:
    """Early-grind detections on the latest closed hourly bar, with all factors (same code as the study)."""
    now_ms = now_ms or int(time.time() * 1000)
    since = now_ms - 40 * DAY
    ctx = ctx or load_context(db, since)
    last_bar = (now_ms // HOUR) * HOUR - HOUR
    out = []
    for sym in ctx["syms"]:
        k = _hourly(db, sym, since)
        if k is None or k.index[-1] != last_bar:
            continue
        f = hourly_shape(k)
        i = len(k) - 1
        for side in (1, -1):
            if early_grind_mask(f.iloc[[i]], side)[0]:
                out.append(decision_row(sym, side, i, k, f, _daily(db, sym), ctx))
    return out


RECENT_H = 30                                   # hours rebuilt from 5-minute candles on each 5-minute scan


def _rolling_hours(db: DB, sym: str, end: int, n: int = RECENT_H) -> pd.DataFrame | None:
    """n one-hour bars ending exactly at `end` (a 5-minute boundary), built from the 5-minute candles."""
    k = db.frame("SELECT ts,o,h,l,c,v,qv,trades,tbqv FROM klines WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts",
                 [sym, end - n * HOUR, end])
    if len(k) < n * 12 * 0.9:
        return None
    k["hb"] = end - n * HOUR + ((k["ts"] - (end - n * HOUR)) // HOUR) * HOUR
    g = k.groupby("hb")
    out = pd.DataFrame({"o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last(),
                        "v": g["v"].sum(), "qv": g["qv"].sum(), "trades": g["trades"].sum(), "tbqv": g["tbqv"].sum()})
    out.index.name = "ts"
    return out.astype("float64") if len(out) == n else None


def scan_now(db: DB, ctx: dict, end: int) -> list[dict]:
    """Early-grind detections as of `end` (any 5-minute boundary): older history from the hourly table,
    the last RECENT_H hours rebuilt from 5-minute candles so a grind is caught within 5 minutes."""
    since = end - 40 * DAY
    ctx_ts = (end // HOUR) * HOUR - HOUR            # latest whole hourly bar for market/sector context
    out = []
    for sym in ctx["syms"]:
        old = _hourly(db, sym, since)
        recent = _rolling_hours(db, sym, end)
        if old is None or recent is None:
            continue
        old = old[old.index + HOUR <= recent.index[0]]
        k = pd.concat([old, recent])
        f = hourly_shape(k)
        i = len(k) - 1
        for side in (1, -1):
            if early_grind_mask(f.iloc[[i]], side)[0]:
                out.append(decision_row(sym, side, i, k, f, _daily(db, sym), ctx, ctx_ts=ctx_ts))
    return out


def score_live(rows: list[dict], bundle: dict | None) -> list[dict]:
    if not rows:
        return rows
    d = pd.DataFrame(rows)
    ck = checklist(d, (bundle or {}).get("thresholds", {}))
    d["score"] = ck.sum(axis=1)
    if bundle and bundle.get("model") is not None:
        d["model_p"] = bundle["model"].predict_proba(d.reindex(columns=bundle["features"]).to_numpy(float))[:, 1]
    else:
        d["model_p"] = np.nan
    for i, r in enumerate(rows):
        r["score"] = int(d["score"].iloc[i])
        r["model_p"] = None if np.isnan(d["model_p"].iloc[i]) else float(d["model_p"].iloc[i])
        r["checks"] = {k: bool(v) for k, v in ck.iloc[i].items()}
    return rows


# ---- jobs: nightly study and hourly detection --------------------------------
LIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS grind_live (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL, side INTEGER NOT NULL,
    price REAL, score INTEGER, model_p REAL, catalyst INTEGER,
    checks TEXT, factors TEXT, decision TEXT,
    all_factors TEXT,                  -- every factor + minute-shape feature known at detection (JSON)
    micro TEXT,                        -- liquidations / order book around detection (JSON)
    threshold REAL,                    -- the model score needed to be sent at the time
    outcomes TEXT,                     -- after 24h: every plan + a target/stop grid + best/worst moves (JSON)
    outcome_ts INTEGER,
    progress TEXT,                     -- under 24h: each plan once decided, best/worst so far (JSON)
    PRIMARY KEY (symbol, ts, side)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS grind_episodes (
    symbol TEXT NOT NULL, side INTEGER NOT NULL, start INTEGER NOT NULL, end INTEGER, move REAL, hours INTEGER,
    best24 REAL, tags TEXT, PRIMARY KEY (symbol, side, start)
) WITHOUT ROWID;
"""
SHOW_FACTORS = ["r_6h", "r_24h", "r2_6h", "hours_with6", "week_ret", "rs_mkt24", "btc_r168", "dpos_30d", "brk_30d",
                "levels_ahead_10pct", "room_ahead", "imb_6h", "mkt_r6", "rvol_6h", "funding", "oi_chg_24h",
                "listing_age_d", "sector_r24", "atr_1h"]


def ensure_live_schema(db: DB) -> None:
    db.conn.executescript(LIVE_SCHEMA)
    have = {r[1] for r in db.conn.execute("PRAGMA table_info(grind_live)")}
    for col, typ in (("all_factors", "TEXT"), ("micro", "TEXT"), ("threshold", "REAL"), ("outcomes", "TEXT"),
                     ("outcome_ts", "INTEGER"), ("progress", "TEXT")):
        if col not in have:
            db.conn.execute(f"ALTER TABLE grind_live ADD COLUMN {col} {typ}")
    db.conn.commit()


def bundle_path(cfg) -> str:
    return str(cfg.path(cfg.storage.model_dir) / "grindlab.joblib")


def load_bundle(cfg) -> dict | None:
    import joblib
    from pathlib import Path
    p = Path(bundle_path(cfg))
    return joblib.load(p) if p.exists() else None


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def nightly(cfg, db: DB) -> dict:
    """Refresh history, rebuild the study, refit the model, store the report. Run in a worker thread."""
    import asyncio
    import joblib
    from .grindlab_data import refresh
    db.conn.executescript(LIVE_SCHEMA)
    asyncio.run(refresh(db, hourly_days=cfg.grindlab.days + 5, funding_days=cfg.grindlab.days + 5))
    df, ep = build(db, days=cfg.grindlab.days)
    if len(df) < 500:
        rep = {"built_ts": int(time.time() * 1000), "n_decisions": int(len(df)), "insufficient": True}
        db.set("grindlab_report", rep)
        return rep
    rep = study(df, ep)
    feats = usable(df, model_features(df))
    bundle = {"model": fit_final(df, feats), "features": feats, "thresholds": rep["thresholds"],
              "built_ts": rep["built_ts"], "n": int(len(df))}
    cfg.path(cfg.storage.model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path(cfg), compress=3)
    db.execute("DELETE FROM grind_episodes")
    recent = ep[ep["start"] >= int(time.time() * 1000) - 14 * DAY] if len(ep) else ep
    db.insert_rows("grind_episodes", ("symbol", "side", "start", "end", "move", "hours", "best24", "tags"),
                   [tuple(_clean(r[c]) for c in ("symbol", "side", "start", "end", "move", "hours", "best24", "tags"))
                    for r in recent.to_dict("records")])
    db.set("grindlab_report", _clean(rep))
    return rep


def nightly_text(rep: dict) -> str:
    from .telegram import esc
    if rep.get("insufficient"):
        return f"<b>Grind lab</b>\nNot enough history yet ({rep.get('n_decisions', 0)} early-grind moments)."
    held = [t for t in rep["traits"] if t["held"]]
    cat = rep["catalyst"].get("short", {})
    lines = [f"<b>Grind lab (nightly)</b> — {rep['n_episodes']} steady grinds in {rep['days']:.0f} days "
             f"({rep['episodes_per_day']:.0f}/day); {rep['n_decisions']} early-grind moments, "
             f"{rep['base_success']:.0%} went on to +{rep['target']:.0%} before -{rep['stop']:.0%}.",
             f"Traits of successful grinders that held on unseen days: {len(held)} of {len(rep['traits'])}"]
    for t in sorted(held, key=lambda t: (t["test_with"] or 0) - (t["test_without"] or 0), reverse=True)[:5]:
        lines.append(f"  • {esc(t['trait'])}: {t['test_with']:.0%} vs {t['test_without']:.0%}")
    if cat.get("n"):
        lines.append(f"News catalyst (down-grinds after Upbit listing / futures launch): {cat['n']} cases, "
                     f"{cat['success']:.0%} success, {cat.get('x_0.06_0.03', 0) * 100:+.2f}% per trade")
    m = rep.get("model", {})
    if m.get("ok"):
        lines.append(f"Model on unseen days: top 10% succeed {m['top10']['success']:.0%} vs {m['base_success']:.0%} base, "
                     f"{m['top10']['avg_r']:+.2f}R per trade")
    return "\n".join(lines)


def hourly(cfg, db: DB) -> list[dict]:
    """Refresh the latest hourly bars, detect early grinds, score them, record them. Run in a worker thread."""
    import asyncio
    from .grindlab_data import refresh_recent
    ensure_live_schema(db)
    syms = [r["symbol"] for r in db.query("SELECT symbol FROM symbols WHERE active=1 AND src='binance'")]
    asyncio.run(refresh_recent(db, syms + ["BTCUSDT"]))
    rows = score_live(live_scan(db), load_bundle(cfg))
    _insert_live(db, rows)
    return rows


def _insert_live(db: DB, rows: list[dict]) -> None:
    for r in rows:
        r["catalyst"] = int(r.get("catalyst_short") or 0)
        db.execute("INSERT OR REPLACE INTO grind_live(symbol, ts, side, price, score, model_p, catalyst, checks, factors, decision) "
                   "VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (r["symbol"], r["ts"], r["side"], r["price"], r["score"], r["model_p"], r["catalyst"],
                    json.dumps(r["checks"]), json.dumps(_clean({k: r.get(k) for k in SHOW_FACTORS})), None))


class FiveMinuteScanner:
    """Every 5 minutes: detect early grinds from the latest 5-minute candles. The hourly table is topped up
    once an hour, and the market context (slow, all coins) is rebuilt once an hour."""

    def __init__(self, cfg):
        self.cfg, self.ctx, self.ctx_hour, self.refreshed_hour = cfg, None, None, None

    def scan(self, db: DB, end: int) -> list[dict]:
        import asyncio
        from .grindlab_data import refresh_recent
        ensure_live_schema(db)
        hour = end // HOUR
        if self.refreshed_hour != hour and end % HOUR >= 5 * 60_000:      # hourly bars are published by then
            syms = [r["symbol"] for r in db.query("SELECT symbol FROM symbols WHERE active=1 AND src='binance'")]
            asyncio.run(refresh_recent(db, syms + ["BTCUSDT"]))
            self.refreshed_hour, self.ctx = hour, None
        if self.ctx is None or self.ctx_hour != hour:
            self.ctx, self.ctx_hour = load_context(db, end - 40 * DAY), hour
        return score_live(scan_now(db, self.ctx, end), load_bundle(self.cfg))