"""Candidate detection: broad momentum-continuation triggers evaluated on closed 5m bars."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def trigger_mask(f: pd.DataFrame, cfg: Config, side: int) -> np.ndarray:
    """Boolean array: bars where a `side` candidate fires (before cooldown)."""
    t, u, tc = cfg.trigger, cfg.universe, cfg.trade

    def a(name: str) -> np.ndarray:
        return f[name].to_numpy(dtype=np.float64)

    with np.errstate(invalid="ignore"):
        mom = (side * a("z_1h") >= t.z_1h) | (side * a("z_4h") >= t.z_4h) | \
              ((side * a("r_24h") >= t.r_24h) & (side * a("r_1h") > 0))
        participation = (a("rvol_1h") >= t.rvol_1h) | (a("rvol_24h") >= t.rvol_24h)
        pos = a("pos_1h") if side > 0 else 1.0 - a("pos_1h")
        intact = (pos >= t.min_pos_1h) & (side * a("d_ema50") > 0)
        atr = a("atr_pct")
        tradable = (a("qv_24h") >= u.min_qvol_24h) & (atr >= t.min_atr_pct) & (atr <= t.max_atr_pct) & \
                   (tc.sl_atr_mult * atr <= tc.max_sl_pct) & (a("bars_seen") >= t.min_history_bars)
        age = a("listing_age_days")
        old_enough = np.isnan(age) | (age >= u.min_listing_days)
    return mom & participation & intact & tradable & old_enough


def apply_cooldown(mask: np.ndarray, cooldown: int) -> np.ndarray:
    """Keep the first trigger of each cluster; suppress the next `cooldown` bars."""
    out = np.zeros(len(mask), dtype=bool)
    last = -10 ** 9
    for i in np.flatnonzero(mask):
        if i - last > cooldown:
            out[i] = True
            last = i
    return out


def kind_of(row: dict) -> str:
    """Human label for the setup, from aligned features."""
    if row.get("new_ext", 0) >= 1:
        return "breakout"
    if row.get("dist_ext", 0) is not None and row.get("dist_ext", 0) > 3:
        return "pullback"
    return "trend"
