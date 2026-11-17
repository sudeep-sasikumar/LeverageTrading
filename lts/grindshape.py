"""Minute-level shape of a grind: what the last 6 hours of 5-minute bars looked like before a decision.

Input: an array (72, 6) of 5m bars ending at the decision time: open, high, low, close, quote volume,
taker-buy quote volume. Everything is measured in the trade's direction (side), so one model learns
long and short grinds together. Used identically by the study and by the live scanner.
"""
from __future__ import annotations

import numpy as np

BARS = 72


def _r2(y: np.ndarray) -> float:
    if len(y) < 3 or not np.all(np.isfinite(y)) or np.std(y) == 0:
        return np.nan
    t = np.arange(len(y), dtype=float)
    return float(np.corrcoef(t, y)[0, 1] ** 2)


def shape_features(pre: np.ndarray, side: int) -> dict:
    """pre: (72, 6) o,h,l,c,qv,tbqv (oldest first). Returns a dict of side-aligned features (NaN when missing)."""
    out: dict = {}
    if pre is None or pre.shape[0] < BARS or not np.isfinite(pre[-1, 3]):
        return out
    o, h, l, c, qv, tb = (pre[:, k].astype(np.float64) for k in range(6))
    ok = np.isfinite(c)
    if ok.sum() < BARS * 0.8:
        return out
    c = np.where(ok, c, np.nan)
    last = c[-1]
    s = float(side)
    y = s * np.log(c / last)                       # path relative to now; in-trend history is negative
    # 1) the raw path, every 30 minutes (12 points: -6h ... -30m)
    for k, i in enumerate(range(0, BARS, 6)):
        out[f"sh_path_{k:02d}"] = float(y[i]) if np.isfinite(y[i]) else np.nan
    lr = s * np.diff(np.log(c))                    # aligned 5m log returns
    lr = np.where(np.isfinite(lr), lr, 0.0)
    for n, name in ((1, "5m"), (3, "15m"), (6, "30m"), (12, "1h"), (24, "2h")):
        out[f"sh_ret_{name}"] = float(lr[-n:].sum())
    total = float(lr.sum())
    out["sh_ret_6h"] = total
    hourly = total / 6.0
    out["sh_accel_1h"] = float(lr[-12:].sum() - hourly)                 # last hour vs average hour
    out["sh_accel_2v4"] = float(lr[-24:].sum() / 2 - lr[:-24].sum() / 4)  # per-hour pace: last 2h vs first 4h
    # 2) pullbacks inside the move
    cum = np.concatenate([[0.0], np.cumsum(lr)])
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    out["sh_max_pullback"] = float(dd.max())
    out["sh_pullback_now"] = float(dd[-1])
    out["sh_pullback_ratio"] = float(dd.max() / total) if total > 0 else np.nan
    in_pb = dd > 0.003
    starts = np.flatnonzero(in_pb[1:] & ~in_pb[:-1])
    out["sh_n_pullbacks"] = float(len(starts))
    out["sh_bars_since_high"] = float(len(cum) - 1 - int(np.argmax(cum)))
    # 3) steadiness at minute level
    out["sh_r2_6h"] = _r2(cum)
    out["sh_r2_2h"] = _r2(cum[-25:])
    absum = np.abs(lr).sum()
    out["sh_er_6h"] = float(abs(total) / absum) if absum > 0 else np.nan
    a2 = np.abs(lr[-24:]).sum()
    out["sh_er_2h"] = float(abs(lr[-24:].sum()) / a2) if a2 > 0 else np.nan
    out["sh_up_share"] = float((lr > 0).mean())
    out["sh_up_share_1h"] = float((lr[-12:] > 0).mean())
    vol = np.std(lr)
    out["sh_vol_5m"] = float(vol)
    out["sh_vol_ratio_1h"] = float(np.std(lr[-12:]) / vol) if vol > 0 else np.nan
    # 4) candles
    rng = np.maximum(h - l, 1e-12)
    body = s * (c - o) / rng
    adverse_wick = np.where(s > 0, (np.minimum(o, c) - l) / rng, (h - np.maximum(o, c)) / rng)
    with_wick = np.where(s > 0, (h - np.maximum(o, c)) / rng, (np.minimum(o, c) - l) / rng)
    out["sh_body_1h"] = float(np.nanmean(body[-12:]))
    out["sh_body_last"] = float(body[-1])
    out["sh_adv_wick_1h"] = float(np.nanmean(adverse_wick[-12:]))
    out["sh_with_wick_1h"] = float(np.nanmean(with_wick[-12:]))        # upper wicks on a long = selling into it
    out["sh_strong_bars"] = float((body > 0.6).sum())
    rng_last = (h[-12:].max() - l[-12:].min())
    pos = (c[-1] - l[-12:].min()) / rng_last if rng_last > 0 else np.nan
    out["sh_pos_1h"] = float(pos if s > 0 else 1 - pos) if np.isfinite(pos) else np.nan
    out["sh_range_last3"] = float(np.nanmean(rng[-3:] / c[-3:]) / np.nanmean(rng / c)) if np.nanmean(rng / c) > 0 else np.nan
    # 5) volume and flow
    qv = np.where(np.isfinite(qv), qv, 0.0)
    tb = np.where(np.isfinite(tb), tb, 0.0)
    mq = qv.mean()
    if mq > 0:
        out["sh_vol_1h_vs_6h"] = float(qv[-12:].mean() / mq)
        out["sh_vol_15m_vs_6h"] = float(qv[-3:].mean() / mq)
        out["sh_vol_slope"] = float(np.polyfit(np.arange(BARS), qv / mq, 1)[0] * BARS)
        out["sh_vol_climax"] = float(qv.max() / np.median(qv)) if np.median(qv) > 0 else np.nan
        with_trend = lr > 0
        out["sh_vol_with_share"] = float(qv[1:][with_trend].sum() / qv[1:].sum()) if qv[1:].sum() > 0 else np.nan
        imb = np.where(qv > 0, 2 * tb / np.maximum(qv, 1e-12) - 1, 0.0) * s
        out["sh_flow_6h"] = float((imb * qv).sum() / qv.sum())
        out["sh_flow_1h"] = float((imb[-12:] * qv[-12:]).sum() / max(qv[-12:].sum(), 1e-12))
        out["sh_flow_15m"] = float((imb[-3:] * qv[-3:]).sum() / max(qv[-3:].sum(), 1e-12))
        out["sh_flow_shift"] = out["sh_flow_1h"] - out["sh_flow_6h"]
        ar = np.abs(lr)
        if np.std(ar) > 0 and np.std(qv[1:]) > 0:
            out["sh_vol_move_corr"] = float(np.corrcoef(ar, qv[1:])[0, 1])
    return out


def shape_matrix(pre: np.ndarray, sides: np.ndarray) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame([shape_features(pre[i], int(sides[i])) for i in range(len(sides))])
