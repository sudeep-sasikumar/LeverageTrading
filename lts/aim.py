"""The project's aim, tested directly: one trade a day that makes ~10% on margin.

At 10x that is a +1% move in the coin; at 5x, +2%. Candidates are early grinds (lts.grindlab). Each UTC day
the system takes the FIRST candidate whose ranking score clears a threshold (decided in real time - never the
best of the day in hindsight) and holds it until target, stop or time limit. Outcomes use 5-minute paths:
if one 5-minute bar touches both target and stop, the stop counts. Weex costs (lts.fees, from config.trade): taker +
slippage on entry and on stop/time exits, taker on the take-profit.

Everything in GRID / RANKERS / QUANTILES is fixed before results are seen. Selection on discovery (older ~9
months), one frozen test on the last 120 days.
"""
from __future__ import annotations

import math

from typing import NamedTuple

import numpy as np
import pandas as pd

from .fees import Fees, fees

TARGETS = (0.01, 0.015, 0.02)
STOPS = (0.007, 0.01, 0.015, 0.02, 0.03)
HOURS = (8, 24)
QUANTILES = (0.0, 0.5, 0.7, 0.85, 0.95)        # threshold = this quantile of the score on discovery days
RANKERS = ("first", "checklist", "smooth", "model", "catalyst")
DAY, HOUR = 86_400_000, 3_600_000
TARGET, STOP, TIME = 1, -1, 0                   # exit kinds returned by outcomes()
EXIT_NAMES = {TARGET: "target", STOP: "stop", TIME: "time"}


class Outcome(NamedTuple):
    net: np.ndarray          # net return per trade (fraction of the entry price), NaN if the path is incomplete
    exit_bar: np.ndarray     # 5m bar index of the exit
    kind: np.ndarray         # TARGET / STOP / TIME


def outcomes(paths: np.ndarray, side: np.ndarray, targets=TARGETS, stops=STOPS, hours=HOURS, fee: Fees | None = None,
             tp_maker: bool = False) -> dict:
    """paths: (N, bars, 4) o,h,l,c of 5m bars starting at entry (entry = first bar's open).
    Returns {(T, S, H): Outcome}. A bar touching both levels counts as the stop. Fees from config.trade:
    entry taker + slippage; stop and time exits taker + slippage; target taker (or maker with tp_maker)."""
    fee = fee or fees()
    tgt_cost = fee.target_maker if tp_maker else fee.target
    o0 = paths[:, 0, 0].astype(np.float64)
    hi, lo, cl = paths[:, :, 1].astype(np.float64), paths[:, :, 2].astype(np.float64), paths[:, :, 3].astype(np.float64)
    s = np.asarray(side)[:, None]
    fav = np.where(s > 0, hi / o0[:, None] - 1, 1 - lo / o0[:, None])
    adv = np.where(s > 0, 1 - lo / o0[:, None], hi / o0[:, None] - 1)
    close_ret = np.where(s > 0, cl / o0[:, None] - 1, 1 - cl / o0[:, None])
    out = {}
    n_bars = paths.shape[1]
    for H in hours:
        nb = min(H * 12, n_bars)
        bad = ~np.isfinite(paths[:, :nb, 1]).all(1)
        for S in stops:
            hit_s = adv[:, :nb] >= S
            i_s = np.where(hit_s.any(1), hit_s.argmax(1), 10 ** 6)
            for T in targets:
                hit_t = fav[:, :nb] >= T
                i_t = np.where(hit_t.any(1), hit_t.argmax(1), 10 ** 6)
                stop_first = (i_s <= i_t) & (i_s < 10 ** 6)
                tgt = (i_t < i_s) & (i_t < 10 ** 6)
                time_ret = close_ret[:, nb - 1]
                net = np.where(stop_first, -S - fee.entry - fee.stop,
                               np.where(tgt, T - fee.entry - tgt_cost, time_ret - fee.entry - fee.time))
                exit_bar = np.where(stop_first, i_s, np.where(tgt, i_t, nb - 1))
                kind = np.where(stop_first, STOP, np.where(tgt, TARGET, TIME))
                net[bad] = np.nan
                out[(T, S, H)] = Outcome(net, exit_bar, kind)
    return out


def daily_policy(ts: np.ndarray, score: np.ndarray, thr: float, net: np.ndarray, exit_bar: np.ndarray) -> np.ndarray:
    """Indices of trades taken: per UTC day, the first candidate (in time) with score >= thr, only if no trade
    is still open. Events must be sorted by ts."""
    taken, busy_until, day_done = [], -1, None
    for i in range(len(ts)):
        d = ts[i] // DAY
        if d == day_done or ts[i] < busy_until or not np.isfinite(net[i]) or not (score[i] >= thr):
            continue
        taken.append(i)
        day_done = d
        busy_until = ts[i] + (exit_bar[i] + 1) * 300_000
    return np.array(taken, dtype=int)


def summarize(ts: np.ndarray, net: np.ndarray, days_total: float) -> dict:
    if len(net) == 0:
        return {"n": 0}
    d = pd.Series(net, index=pd.to_datetime(ts, unit="ms"))
    months = d.groupby(d.index.strftime("%Y-%m")).sum()
    eq5 = np.cumprod(1 + 0.2 * np.clip(5 * net, -1, None))      # 20% of the account as margin at 5x
    dd5 = float((eq5 / np.maximum.accumulate(eq5) - 1).min())
    streak = cur = 0
    for x in net:
        cur = cur + 1 if x <= 0 else 0
        streak = max(streak, cur)
    sd = net.std(ddof=1) if len(net) > 1 else 0.0
    return {"n": int(len(net)), "per_day": len(net) / days_total, "win": float((net > 0).mean()),
            "mean_pct": float(net.mean()), "t": float(net.mean() / (sd / math.sqrt(len(net)))) if sd > 0 else 0.0,
            "months_pos": float((months > 0).mean()), "n_months": int(len(months)),
            "margin_5x": float(5 * net.mean()), "margin_10x": float(10 * net.mean()),
            "month_5x_median": float(5 * months.median()), "dd_5x_20pct_margin": dd5, "worst_streak": int(streak)}


def live_score(row: dict, ranker: str) -> float:
    """The daily-trade ranking score for a live early-grind row (same definitions as the study)."""
    if ranker == "checklist":
        return float(row.get("score") or 0)
    if ranker == "smooth":
        return float((row.get("r2_6h") or 0) * (row.get("hours_with6") or 0))
    if ranker == "catalyst":
        return float(row.get("catalyst_short") or 0)
    return 1.0                                      # "first": any early grind
