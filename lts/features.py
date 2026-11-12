"""Feature engine. The SAME functions serve the live scan and the historical replay.

Every value at bar i uses only data available at that bar's close (no look-ahead):
rolling windows look backwards, and slower series (OI, funding, ratios) are joined
as-of the bar close time.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

from . import BAR_MS

H, D, W = 12, 288, 2016          # bars per hour / day / week
STALE_MS = 2 * 3_600_000         # slower series older than this are treated as missing

RET_WINDOWS = [("5m", 1), ("15m", 3), ("30m", 6), ("1h", 12), ("2h", 24), ("4h", 48),
               ("8h", 96), ("24h", 288), ("3d", 864)]

# ---- feature alignment tables (raw column -> aligned behaviour) -------------
DIRECTIONAL = [
    "r_5m", "r_15m", "r_30m", "r_1h", "r_2h", "r_4h", "r_8h", "r_24h", "r_3d",
    "z_1h", "z_4h", "z_24h", "imb_15m", "imb_1h", "imb_4h", "imb_24h",
    "d_ema20", "d_ema50", "d_ema200", "ema20_slope", "ema_stack", "d_vwap_24h", "d_vwap_4h",
    "clv", "body_frac", "bb_z",
    "rs_btc_1h", "rs_btc_4h", "rs_btc_24h", "rs_mkt_1h", "rs_mkt_24h",
    "btc_r_15m", "btc_r_1h", "btc_r_4h", "btc_r_24h", "btc_r_3d", "btc_d_ema200",
    "mkt_r_1h", "mkt_r_4h", "mkt_r_24h", "mkt_r_3d",
    "fund_rate", "fund_dev", "fund_pred", "basis_bps",
    "lsr_top", "lsr_glob", "lsr_top_chg_4h", "lsr_top_chg_24h", "lsr_glob_chg_4h", "lsr_glob_chg_24h",
    "book_imb", "weex_dev_bps",
]
FLIP_UNIT = ["pos_1h", "pos_4h", "pos_24h", "pos_7d", "up_frac_1h", "breadth_1h", "breadth_24h",
             "rank_r_1h", "rank_r_24h"]
FLIP_RSI = ["rsi14", "rsi_1h"]
SWAP = [  # (aligned name for the trade-direction side, aligned name for the other side, long col, short col)
    ("dist_ext", "dist_opp", "dist_hi_24h", "dist_lo_24h"),
    ("bars_since_ext", "bars_since_opp", "bars_since_hi", "bars_since_lo"),
    ("adverse_wick", "fav_wick", "upper_wick", "lower_wick"),
    ("pullback_24h", "run_24h", "run_from_hi_24h", "run_from_lo_24h"),
    ("new_ext", "new_opp", "new_hi_24h", "new_lo_24h"),
    ("impulse_same_1h", "impulse_opp_1h", "impulse_up_1h", "impulse_dn_1h"),
    ("impulse_same_4h", "impulse_opp_4h", "impulse_up_4h", "impulse_dn_4h"),
]
# Features that are the same for every coin at a given moment (up to the side sign). Their
# effective sample size is the number of DAYS, not candidates, so the learner only admits
# them once the dataset spans enough days (see Learn.market_features_min_days).
MARKET_LEVEL = [
    "btc_r_15m", "btc_r_1h", "btc_r_4h", "btc_r_24h", "btc_r_3d", "btc_d_ema200", "btc_atr_pct",
    "btc_vol_ratio", "btc_er_4h", "mkt_r_1h", "mkt_r_4h", "mkt_r_24h", "mkt_r_3d",
    "breadth_1h", "breadth_24h", "disp_24h", "movers",
    "impulse_same_1h", "impulse_opp_1h", "impulse_same_4h", "impulse_opp_4h",
    "hour_sin", "hour_cos", "dow", "weekend", "mins_to_funding", "side",
]
NEUTRAL = [
    "atr_pct", "rv_1h", "rv_24h", "vol_ratio_1h_24h", "vol_ratio_24h_7d",
    "rvol_5m", "rvol_1h", "rvol_24h", "vol_accel", "log_qv_24h", "ats_ratio", "trades_rvol",
    "er_1h", "er_4h", "range_atr", "range_atr_3",
    "oi_chg_30m", "oi_chg_1h", "oi_chg_4h", "oi_chg_24h", "oi_to_vol",
    "mins_to_funding", "corr_btc_24h", "btc_atr_pct", "btc_vol_ratio", "btc_er_4h",
    "disp_24h", "movers", "rank_rvol", "hour_sin", "hour_cos", "dow", "weekend", "listing_age_log",
    "weex_spread_bps", "weex_log_qv", "book_depth_log",
]
LIVE_ONLY = ["fund_pred", "basis_bps", "book_imb", "weex_dev_bps", "weex_spread_bps", "weex_log_qv",
             "book_depth_log"]


def feature_names() -> list[str]:
    names = list(DIRECTIONAL) + list(FLIP_UNIT) + list(FLIP_RSI) + list(NEUTRAL)
    for ext, opp, _, _ in SWAP:
        names += [ext, opp]
    return names + ["side"]


# ---- helpers ----------------------------------------------------------------
def regularize(k: pd.DataFrame) -> pd.DataFrame:
    """Index by ts on a continuous 5m grid; missing bars are flat with zero volume."""
    k = k.drop_duplicates("ts").set_index("ts").sort_index()
    full = np.arange(k.index[0], k.index[-1] + BAR_MS, BAR_MS, dtype=np.int64)
    if len(full) != len(k):
        k = k.reindex(full)
        c = k["c"].ffill()
        for col in ("o", "h", "l", "c"):
            k[col] = k[col].fillna(c)
        for col in ("v", "qv", "trades", "tbv", "tbqv"):
            k[col] = k[col].fillna(0.0)
    return k.astype("float64")


def _asof(index_close_ts: np.ndarray, src: pd.DataFrame | None, col: str,
          stale_ms: int = STALE_MS) -> np.ndarray:
    """Latest value of src[col] with src.ts <= bar close time; NaN when stale or absent."""
    out = np.full(len(index_close_ts), np.nan)
    if src is None or len(src) == 0 or col not in src:
        return out
    s = src[["ts", col]].dropna()
    if s.empty:
        return out
    ts = s["ts"].to_numpy(dtype=np.int64)
    vals = s[col].to_numpy(dtype=np.float64)
    pos = np.searchsorted(ts, index_close_ts, side="right") - 1
    ok = pos >= 0
    safe = np.where(ok, pos, 0)
    fresh = ok & (index_close_ts - ts[safe] <= stale_ms)
    out[fresh] = vals[safe[fresh]]
    return out


def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up, dn = _wilder(d.clip(lower=0), n), _wilder((-d).clip(lower=0), n)
    return 100.0 - 100.0 / (1.0 + up / dn.replace(0, np.nan))


def _bars_since_extreme(x: np.ndarray, n: int, use_max: bool) -> np.ndarray:
    """Fraction of the window elapsed since the most recent extreme (0 = current bar)."""
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        rev = sliding_window_view(x, n)[:, ::-1]          # column 0 = current bar
        out[n - 1:] = (np.argmax(rev, axis=1) if use_max else np.argmin(rev, axis=1)) / n
    return out


def panel_inputs(k: pd.DataFrame) -> pd.DataFrame:
    """Cheap per-symbol columns needed for cross-sectional context."""
    c, qv = k["c"], k["qv"]
    out = pd.DataFrame(index=k.index)
    out["r_1h"] = c / c.shift(H) - 1
    out["r_4h"] = c / c.shift(4 * H) - 1
    out["r_24h"] = c / c.shift(D) - 1
    out["r_3d"] = c / c.shift(3 * D) - 1
    out["rvol_1h"] = qv.rolling(H).sum() / (qv.rolling(W, min_periods=2 * D).mean() * H).shift(H)
    out["qv_24h"] = qv.rolling(D, min_periods=D // 2).sum()
    base = (np.log(c).diff().rolling(W, min_periods=2 * D).std() * math.sqrt(H)).shift(H).replace(0, np.nan)
    out["z_1h"] = out["r_1h"] / base                # same definition as symbol_frame
    out["z_4h"] = out["r_4h"] / (base * 2.0)
    return out


# ---- per-symbol raw frame ---------------------------------------------------
def symbol_frame(k: pd.DataFrame, oi: pd.DataFrame | None = None, funding: pd.DataFrame | None = None,
                 lsr: pd.DataFrame | None = None, premium: pd.DataFrame | None = None,
                 onboard_ts: float | None = None) -> pd.DataFrame:
    """Raw (not side-aligned) features for every bar of a regularized kline frame."""
    o, h, l, c, v, qv = k["o"], k["h"], k["l"], k["c"], k["v"], k["qv"]
    trades, tbqv = k["trades"], k["tbqv"]
    f = pd.DataFrame(index=k.index)
    f["close"] = c
    close_ts = k.index.to_numpy(dtype=np.int64) + BAR_MS

    for name, n in RET_WINDOWS:
        f[f"r_{name}"] = c / c.shift(n) - 1

    # volatility
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, 14)
    f["atr_pct"] = atr / c
    lr = np.log(c).diff()
    rv_1h = lr.rolling(H).std() * math.sqrt(H)
    rv_24h = lr.rolling(D, min_periods=D // 2).std() * math.sqrt(H)
    rv_7d = lr.rolling(W, min_periods=2 * D).std() * math.sqrt(H)
    f["rv_1h"], f["rv_24h"] = rv_1h, rv_24h
    f["vol_ratio_1h_24h"] = rv_1h / rv_24h
    f["vol_ratio_24h_7d"] = rv_24h / rv_7d
    base = rv_7d.shift(H).replace(0, np.nan)        # hourly vol baseline excluding the current hour
    f["z_1h"] = f["r_1h"] / base
    f["z_4h"] = f["r_4h"] / (base * 2.0)
    f["z_24h"] = f["r_24h"] / (base * math.sqrt(24))

    # volume / participation
    qv_1h = qv.rolling(H).sum()
    qv_24h = qv.rolling(D, min_periods=D // 2).sum()
    qv_bar_7d = qv.rolling(W, min_periods=2 * D).mean()
    f["qv_24h"] = qv_24h
    f["log_qv_24h"] = np.log10(qv_24h.clip(lower=1.0))
    f["rvol_1h"] = qv_1h / (qv_bar_7d * H).shift(H)
    f["rvol_5m"] = qv / qv.rolling(D, min_periods=D // 2).mean().shift(1)
    f["rvol_24h"] = qv_24h / (qv_bar_7d * D).shift(D)
    f["vol_accel"] = qv_1h / qv_1h.shift(H).replace(0, np.nan)
    tr_1h = trades.rolling(H).sum()
    tr_24h = trades.rolling(D, min_periods=D // 2).sum()
    f["ats_ratio"] = (qv_1h / tr_1h.replace(0, np.nan)) / (qv_24h / tr_24h.replace(0, np.nan))
    f["trades_rvol"] = tr_1h / (trades.rolling(W, min_periods=2 * D).mean() * H).shift(H)
    for name, n in (("15m", 3), ("1h", H), ("4h", 4 * H), ("24h", D)):
        tot = qv.rolling(n).sum().replace(0, np.nan)
        f[f"imb_{name}"] = (2.0 * tbqv.rolling(n).sum() - tot) / tot

    # trend
    ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = c.ewm(span=200, adjust=False, min_periods=200).mean()
    atr_nz = atr.replace(0, np.nan)
    f["d_ema20"] = (c - ema20) / atr_nz
    f["d_ema50"] = (c - ema50) / atr_nz
    f["d_ema200"] = (c - ema200) / atr_nz
    f["ema20_slope"] = (ema20 - ema20.shift(6)) / atr_nz
    f["ema_stack"] = np.sign(ema20 - ema50) + np.sign(ema50 - ema200)
    absd = c.diff().abs()
    f["er_1h"] = (c - c.shift(H)).abs() / absd.rolling(H).sum().replace(0, np.nan)
    f["er_4h"] = (c - c.shift(4 * H)).abs() / absd.rolling(4 * H).sum().replace(0, np.nan)
    f["up_frac_1h"] = (c.diff() > 0).astype(float).rolling(H).mean()

    # location in range
    for name, n in (("1h", H), ("4h", 4 * H), ("24h", D), ("7d", W)):
        mp = n if n <= D else 2 * D
        hi, lo = h.rolling(n, min_periods=mp).max(), l.rolling(n, min_periods=mp).min()
        f[f"pos_{name}"] = (c - lo) / (hi - lo).replace(0, np.nan)
        if name == "24h":
            f["dist_hi_24h"] = (hi - c) / atr_nz
            f["dist_lo_24h"] = (c - lo) / atr_nz
            f["run_from_lo_24h"] = c / lo - 1
            f["run_from_hi_24h"] = hi / c - 1
            f["new_hi_24h"] = (h >= hi.shift(1)).astype(float)
            f["new_lo_24h"] = (l <= lo.shift(1)).astype(float)
    f["bars_since_hi"] = _bars_since_extreme(h.to_numpy(), D, True)
    f["bars_since_lo"] = _bars_since_extreme(l.to_numpy(), D, False)
    for name, n in (("24h", D), ("4h", 4 * H)):
        mp = max(n // 2, H)
        vwap = qv.rolling(n, min_periods=mp).sum() / v.rolling(n, min_periods=mp).sum().replace(0, np.nan)
        f[f"d_vwap_{name}"] = (c - vwap) / atr_nz

    # candle structure / exhaustion
    rng = (h - l).replace(0, np.nan)
    f["clv"] = ((2 * c - h - l) / rng).fillna(0.0)
    f["body_frac"] = ((c - o) / rng).fillna(0.0)
    top = pd.concat([o, c], axis=1).max(axis=1)
    bot = pd.concat([o, c], axis=1).min(axis=1)
    f["upper_wick"] = ((h - top) / rng).fillna(0.0).rolling(3).mean()
    f["lower_wick"] = ((bot - l) / rng).fillna(0.0).rolling(3).mean()
    f["range_atr"] = (h - l) / atr_nz
    f["range_atr_3"] = (h.rolling(3).max() - l.rolling(3).min()) / atr_nz
    f["rsi14"] = _rsi(c, 14)
    f["rsi_1h"] = _rsi(c, 14 * H)
    f["bb_z"] = (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)

    # open interest (as-of bar close)
    oi_s = pd.Series(_asof(close_ts, oi, "oi"), index=k.index)
    for name, n in (("30m", 6), ("1h", H), ("4h", 4 * H), ("24h", D)):
        f[f"oi_chg_{name}"] = oi_s / oi_s.shift(n) - 1
    f["oi_to_vol"] = oi_s * c / qv_24h.replace(0, np.nan)

    # funding (settled history) and live premium snapshots
    if funding is not None and len(funding):
        fd = funding.sort_values("ts").copy()
        fd["dev"] = fd["rate"] - fd["rate"].rolling(9, min_periods=3).mean()
        f["fund_rate"] = _asof(close_ts, fd, "rate", stale_ms=9 * 3_600_000)
        f["fund_dev"] = _asof(close_ts, fd, "dev", stale_ms=9 * 3_600_000)
    else:
        f["fund_rate"] = np.nan
        f["fund_dev"] = np.nan
    f["fund_pred"] = _asof(close_ts, premium, "funding_rate")
    if premium is not None and len(premium):
        pm = premium.copy()
        pm["basis"] = (pm["mark"] - pm["index_px"]) / pm["index_px"].replace(0, np.nan) * 1e4
        f["basis_bps"] = _asof(close_ts, pm, "basis")
    else:
        f["basis_bps"] = np.nan
    f["mins_to_funding"] = (8 * 3_600_000 - (close_ts % (8 * 3_600_000))) / 60_000.0

    # positioning
    top_s = pd.Series(np.log(_asof(close_ts, lsr, "top_pos")), index=k.index)
    glob_s = pd.Series(np.log(_asof(close_ts, lsr, "global_acc")), index=k.index)
    f["lsr_top"], f["lsr_glob"] = top_s, glob_s
    for name, n in (("4h", 4 * H), ("24h", D)):
        f[f"lsr_top_chg_{name}"] = top_s - top_s.shift(n)
        f[f"lsr_glob_chg_{name}"] = glob_s - glob_s.shift(n)

    # calendar / meta
    hours = (close_ts % 86_400_000) / 3_600_000.0
    f["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    dow = ((close_ts // 86_400_000) + 3) % 7           # 0 = Monday
    f["dow"] = dow.astype(float)
    f["weekend"] = (dow >= 5).astype(float)
    if onboard_ts and not (isinstance(onboard_ts, float) and math.isnan(onboard_ts)):
        age_days = np.maximum((close_ts - float(onboard_ts)) / 86_400_000.0, 0.0)
        f["listing_age_log"] = np.log10(age_days + 1.0)
        f["listing_age_days"] = age_days
    else:
        f["listing_age_log"] = np.nan
        f["listing_age_days"] = np.nan
    f["bars_seen"] = np.arange(1, len(f) + 1, dtype=float)
    f["lr"] = lr
    return f.copy()          # defragment after many column inserts


# ---- market context ---------------------------------------------------------
def market_context(panels: dict[str, pd.DataFrame], btc: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Cross-sectional context from per-symbol panel_inputs frames.

    Returns (market frame indexed by ts, {name: wide rank frame}).
    """
    wide = {col: pd.DataFrame({s: p[col] for s, p in panels.items()})
            for col in ("r_1h", "r_4h", "r_24h", "r_3d", "rvol_1h", "z_1h", "z_4h")}
    r1, r4, r24 = wide["r_1h"], wide["r_4h"], wide["r_24h"]
    m = pd.DataFrame(index=r1.index)
    n1 = r1.notna().sum(axis=1).replace(0, np.nan)
    n24 = r24.notna().sum(axis=1).replace(0, np.nan)
    m["breadth_1h"] = (r1 > 0).sum(axis=1) / n1
    m["breadth_24h"] = (r24 > 0).sum(axis=1) / n24
    m["mkt_r_1h"] = r1.median(axis=1)
    m["mkt_r_4h"] = r4.median(axis=1)
    m["mkt_r_24h"] = r24.median(axis=1)
    m["mkt_r_3d"] = wide["r_3d"].median(axis=1)
    m["disp_24h"] = r24.std(axis=1)
    m["movers"] = (r24.abs() > 0.10).sum(axis=1) / n24
    # market-wide impulse: share of coins making an abnormal move right now (trend-day detector)
    for name, z in (("1h", wide["z_1h"]), ("4h", wide["z_4h"])):
        nz = z.notna().sum(axis=1).replace(0, np.nan)
        m[f"impulse_up_{name}"] = (z >= 2.0).sum(axis=1) / nz
        m[f"impulse_dn_{name}"] = (z <= -2.0).sum(axis=1) / nz
    ranks = {"rank_r_1h": r1.rank(axis=1, pct=True), "rank_r_24h": r24.rank(axis=1, pct=True),
             "rank_rvol": wide["rvol_1h"].rank(axis=1, pct=True)}
    if btc is not None and len(btc):
        c, h, l = btc["c"], btc["h"], btc["l"]
        b = pd.DataFrame(index=btc.index)
        for name, n in (("15m", 3), ("1h", H), ("4h", 4 * H), ("24h", D), ("3d", 3 * D)):
            b[f"btc_r_{name}"] = c / c.shift(n) - 1
        prev = c.shift(1)
        tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
        atr = _wilder(tr, 14)
        b["btc_atr_pct"] = atr / c
        lr = np.log(c).diff()
        b["btc_vol_ratio"] = (lr.rolling(H).std() / lr.rolling(D, min_periods=D // 2).std())
        b["btc_d_ema200"] = (c - c.ewm(span=200, adjust=False, min_periods=200).mean()) / atr.replace(0, np.nan)
        b["btc_er_4h"] = (c - c.shift(4 * H)).abs() / c.diff().abs().rolling(4 * H).sum().replace(0, np.nan)
        b["btc_lr"] = lr
        m = m.join(b, how="left")
    return m, ranks


def add_context(f: pd.DataFrame, symbol: str, market: pd.DataFrame, ranks: dict) -> pd.DataFrame:
    ctx = market.reindex(f.index)
    new = {col: ctx[col].to_numpy() for col in ctx.columns if col != "btc_lr"}
    for name, wide in ranks.items():
        new[name] = wide[symbol].reindex(f.index).to_numpy() if symbol in wide else np.full(len(f), np.nan)
    if "btc_r_1h" in new:
        new["rs_btc_1h"] = f["r_1h"].to_numpy() - new["btc_r_1h"]
        new["rs_btc_4h"] = f["r_4h"].to_numpy() - new["btc_r_4h"]
        new["rs_btc_24h"] = f["r_24h"].to_numpy() - new["btc_r_24h"]
        new["corr_btc_24h"] = f["lr"].rolling(D, min_periods=D // 2).corr(ctx["btc_lr"]).to_numpy()
    new["rs_mkt_1h"] = f["r_1h"].to_numpy() - new["mkt_r_1h"]
    new["rs_mkt_24h"] = f["r_24h"].to_numpy() - new["mkt_r_24h"]
    return pd.concat([f, pd.DataFrame(new, index=f.index)], axis=1)


# ---- alignment --------------------------------------------------------------
def align(rows: pd.DataFrame, side: int) -> pd.DataFrame:
    """Turn raw rows into the side-aligned model feature frame (see module tables)."""
    out = {}
    n = len(rows)

    def col(name: str) -> np.ndarray:
        return rows[name].to_numpy(dtype=np.float64) if name in rows else np.full(n, np.nan)

    for name in DIRECTIONAL:
        out[name] = col(name) * side
    for name in FLIP_UNIT:
        x = col(name)
        out[name] = x if side > 0 else 1.0 - x
    for name in FLIP_RSI:
        x = col(name)
        out[name] = x if side > 0 else 100.0 - x
    for name in NEUTRAL:
        out[name] = col(name)
    for ext, opp, long_col, short_col in SWAP:
        a, b = col(long_col), col(short_col)
        out[ext], out[opp] = (a, b) if side > 0 else (b, a)
    out["side"] = np.full(n, float(side))
    return pd.DataFrame(out, index=rows.index)[feature_names()]
