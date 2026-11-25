"""B2 (docs/HYPOTHESES.md): why the big daily movers are not caught.

Needs research/parity.py's detections (live 5-minute logic) for the live-logic flag time.
Run: python -m research.b2_movers  ->  docs/results/b2_movers.txt (fetches the extra 5m / 1m candles it needs)
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from research.common import BAR, DAY, HOUR, LOCAL_DB, MIN, RD, YEAR_NOW, Tee, load_paths
from research.fetch_candles import run as fetch

from lts import grindlab as G
from lts import grindsignal as GS
from lts import validate as V
from lts.db import DB
from lts.fees import fees

out = Tee("b2_movers.txt")
FEE = fees()
START = YEAR_NOW - 368 * DAY                     # 13 Nov 2025 (21 days of hourly history before)
MOVE = 0.15
PL5 = {"leverage": 5, "target": 0.02, "stop": 0.01}
B1_PLANS = [(2, 72), (3, 72), (4, 72)]           # fixed 5% stop, R multiple, hours
RNG = np.random.default_rng(25)

db = DB(LOCAL_DB)
syms = [r["symbol"] for r in db.query("SELECT DISTINCT symbol FROM klines_1h")]
fund_all = db.frame("SELECT symbol, ts, rate FROM funding")
fund = {s: g[["ts", "rate"]].to_numpy() for s, g in fund_all.groupby("symbol")}
d1 = db.frame("SELECT symbol, ts, o, h, l, c FROM klines_1d WHERE ts >= ?", [START - 60 * DAY])
regime = V.alt_regime(d1.pivot_table(index="ts", columns="symbol", values="c"))

# ---- hour-close early-grind detections (every hour, not deduplicated) and the hourly bars ------------------
det, hourly = [], {}
for sym in syms:
    k = G._hourly(db, sym, START - 31 * DAY)
    if k is None:
        continue
    f = G.hourly_shape(k)
    t_close = k.index.to_numpy() + HOUR
    hourly[sym] = k
    for side in (1, -1):
        m = G.early_grind_mask(f, side) & (t_close >= START) & (t_close <= YEAR_NOW)
        for i in np.flatnonzero(m):
            det.append((sym, int(t_close[i]), side, float(k["c"].iloc[i]), float(f["qv_24h"].iloc[i])))
det = pd.DataFrame(det, columns=["symbol", "ts", "side", "price", "qv24"])
det["day"] = det.ts // DAY * DAY
det.loc[det.ts % DAY == 0, "day"] -= DAY              # a close at 00:00 belongs to the day that just ended
out(f"hour-close early-grind detections: {len(det):,} ({det.symbol.nunique()} coins)")

# ---- big movers -----------------------------------------------------------------------------------------------
dd = d1[(d1.ts >= START) & (d1.ts < YEAR_NOW - DAY)].copy()
dd["move"] = dd.c / dd.o - 1
qv_start = {}
for sym, k in hourly.items():
    q = k["qv"].rolling(24).sum()
    qv_start[sym] = pd.Series(q.to_numpy(), index=k.index.to_numpy() + HOUR)
dd["qv24"] = [qv_start[s].get(t, np.nan) if s in qv_start else np.nan for s, t in zip(dd.symbol, dd.ts)]
movers = dd[(dd.move.abs() >= MOVE) & (dd.qv24 >= 5_000_000)].copy()
movers["side"] = np.sign(movers.move).astype(int)
out(f"big movers (UTC day >= 15% open to close, $5M+ volume): {len(movers):,} coin-days "
    f"({(movers.side > 0).sum():,} up, {(movers.side < 0).sum():,} down)")
day_move = {(s, t): m for s, t, m in zip(dd.symbol, dd.ts, dd.move)}

# ---- primary flag: first hour close in the day where the mask fires the move's way --------------------------------
first = det.loc[det.groupby(["symbol", "day", "side"]).ts.idxmin()].set_index(["symbol", "day", "side"])
flags = []
for r in movers.itertuples(index=False):
    if (r.symbol, r.ts, r.side) not in first.index:
        continue
    x = first.loc[(r.symbol, r.ts, r.side)]
    k = hourly[r.symbol]
    day = k[(k.index >= r.ts) & (k.index < r.ts + DAY)]
    best = (day.h.max() if r.side > 0 else day.l.min())
    ahead = r.side * (r.c - x.price) / (r.side * (r.c - r.o))
    ahead_best = r.side * (best - x.price) / (r.side * (best - r.o))
    flags.append({"symbol": r.symbol, "day": r.ts, "side": r.side, "ts": int(x.ts), "price": x.price, "move": r.move,
                  "ahead": ahead, "ahead_best": ahead_best, "group": "mover"})
flags = pd.DataFrame(flags)
out(f"\nflagged by the hour-close logic during the day: {len(flags):,} of {len(movers):,} ({len(flags) / len(movers):.0%})")
for lab, m in (("up", flags.side > 0), ("down", flags.side < 0), ("all", flags.side != 0)):
    f_ = flags[m]
    out(f"  {lab:<4} n {len(f_):>5}: hour of the day flagged (median) {np.median((f_.ts - f_.day) / HOUR):.0f}h; "
        f"move still ahead to the day's close: median {f_.ahead.median():.0%}, mean {f_.ahead.mean():.0%}; "
        f"to the day's extreme: median {f_.ahead_best.median():.0%}; flagged with <= 0% still ahead: {(f_.ahead <= 0).mean():.0%}")

# ---- lookalikes at the same moments ------------------------------------------------------------------------------
mover_keys = set(zip(movers.symbol, movers.ts))
by_moment = {k: g for k, g in det.groupby(["ts", "side"])}
looks = []
for r in flags.itertuples(index=False):
    x = by_moment.get((r.ts, r.side))
    if x is None:
        continue
    for y in x.itertuples(index=False):
        if y.symbol == r.symbol or (y.symbol, r.day) in mover_keys:
            continue
        mv = day_move.get((y.symbol, r.day), np.nan)
        if np.isfinite(mv) and r.side * mv < MOVE:
            looks.append({"symbol": y.symbol, "day": r.day, "side": r.side, "ts": r.ts, "price": y.price, "move": mv,
                          "group": "lookalike"})
looks = pd.DataFrame(looks).drop_duplicates(["symbol", "ts", "side"])
out(f"lookalikes (same hour close, same direction, own day < 15%): {len(looks):,} "
    f"({len(looks) / max(len(flags), 1):.1f} per mover flag); their own day moved a median "
    f"{(looks.side * looks.move).median():+.1%} their way")

# ---- 5m candles for both groups, 1m for the stop-first sample ----------------------------------------------------
allf = pd.concat([flags, looks], ignore_index=True)
asyncio.run(fetch([(s, int(t), int(t) + 24 * HOUR + BAR) for s, t in zip(allf.symbol, allf.ts)], "5m", log=out))
samp = pd.concat([flags.sample(min(300, len(flags)), random_state=25), looks.sample(min(300, len(looks)), random_state=25)])
asyncio.run(fetch([(s, int(t), int(t) + 24 * HOUR) for s, t in zip(samp.symbol, samp.ts)], "1m", log=out))

post, _ = load_paths(allf[["symbol", "ts"]], 289)
o5 = GS.plan_outcomes(post, allf.side.to_numpy(), [PL5], 24, with_kind=True)[GS.plan_label(PL5)]
allf["net5x"] = o5.net
allf["regime"] = regime.reindex(allf.day).to_numpy()


def b1_plan(sym, t, side, tr, th):
    """Fixed 5% stop, tr x 5% target, th hours, on hourly bars from the hour open at t; funding included."""
    k = hourly.get(sym)
    if k is None:
        return np.nan
    j = np.searchsorted(k.index.to_numpy(), t)
    if j + th > len(k) or k.index[j] != t:
        return np.nan
    e = k.o.iloc[j]
    h, l, c = (k[x].to_numpy()[j:j + th] for x in "hlc")
    fav = (h / e - 1) if side > 0 else (1 - l / e)
    adv = (1 - l / e) if side > 0 else (h / e - 1)
    risk, tgt = 0.05, tr * 0.05
    hs, ht = np.flatnonzero(adv >= risk), np.flatnonzero(fav >= tgt)
    i_s = hs[0] if len(hs) else 10 ** 9
    i_t = ht[0] if len(ht) else 10 ** 9
    if i_s <= i_t and i_s < 10 ** 9:
        g, xc, kk = -risk, FEE.stop, i_s
    elif i_t < 10 ** 9:
        g, xc, kk = tgt, FEE.target, i_t
    else:
        kk = th - 1
        g, xc = side * (c[kk] / e - 1), FEE.time
    fs = fund.get(sym)
    fd = 0.0
    if fs is not None:
        mm = (fs[:, 0] > t) & (fs[:, 0] <= t + (kk + 1) * HOUR)
        fd = float(side * fs[mm, 1].sum())
    return (g - FEE.entry - xc - fd) / risk


for tr, th in B1_PLANS:
    allf[f"b1_{tr}R"] = [b1_plan(s, int(t), int(sd), tr, th) for s, t, sd in zip(allf.symbol, allf.ts, allf.side)]

out("\nPlans from the flag (entry at the next 5-minute open; B1 plans on hourly candles from the next hour open):")
for gname, g in (("movers (hindsight group)", allf[allf.group == "mover"]), ("lookalikes", allf[allf.group == "lookalike"]),
                 ("both (what the logic flags)", allf)):
    out(f"  {gname}:")
    st = V.stats(g.net5x.to_numpy(), g.ts.to_numpy())
    out(f"    5x +2%/-1%: n {st['n']:,} won {st['win']:.1%} net {st['mean']:+.3%} t {st['t_day']:+.1f} months+ {st['months_pos']:.0%}")
    for tr, _ in B1_PLANS:
        st = V.stats(g[f"b1_{tr}R"].to_numpy(), g.ts.to_numpy())
        out(f"    5% stop {tr}R 72h: n {st['n']:,} won {st['win']:.1%} {st['mean']:+.3f}R t {st['t_day']:+.1f} months+ {st['months_pos']:.0%}")
    for lab, m in (("long", g.side > 0), ("short", g.side < 0), ("regime +", g.regime > 0), ("regime -", g.regime < 0)):
        st = V.stats(g.net5x[m].to_numpy(), g.ts[m].to_numpy())
        if st.get("n"):
            out(f"      5x, {lab:<9}: n {st['n']:,} won {st['win']:.1%} net {st['mean']:+.3%} t {st['t_day']:+.1f}")

# ---- live (5-minute rolling) flag: how much earlier ------------------------------------------------------------------
pd_path = RD / "parity_detections.parquet"
if pd_path.exists():
    pdet = pd.read_parquet(pd_path, columns=["symbol", "ts", "side"])
    pdet["day"] = (pdet.ts - 1) // DAY * DAY
    first5 = pdet.groupby(["symbol", "day", "side"]).ts.min()
    earlier = []
    for r in flags.itertuples(index=False):
        t5 = first5.get((r.symbol, r.day, r.side))
        if t5 is not None:
            earlier.append((r.ts - t5) / MIN)
    earlier = np.array(earlier)
    out(f"\nLive 5-minute logic on the same mover days: flags {len(earlier):,} of them; it flags a median "
        f"{np.median(earlier):.0f} minutes earlier than the hour close (mean {earlier.mean():.0f}; "
        f"{(earlier > 0).mean():.0%} earlier, {(earlier == 0).mean():.0%} same time)")

# ---- stop-first check on 1-minute candles -------------------------------------------------------------------------------
p1, _ = load_paths(samp[["symbol", "ts"]], 1440, table="k1", step=MIN)
p5, _ = load_paths(samp[["symbol", "ts"]], 289)


def plan_on(paths, side, bars_per_h):
    import lts.aim as A
    e = paths[:, 0, 0].astype(float)
    hi, lo = paths[:, :, 1].astype(float), paths[:, :, 2].astype(float)
    nb = 24 * bars_per_h
    s = side[:, None]
    fav = np.where(s > 0, hi / e[:, None] - 1, 1 - lo / e[:, None])[:, :nb]
    adv = np.where(s > 0, 1 - lo / e[:, None], hi / e[:, None] - 1)[:, :nb]
    out_ = np.full(len(e), np.nan)
    for i in range(len(e)):
        if np.isfinite(hi[i, :nb]).mean() < 0.97:
            continue
        hs, ht = np.flatnonzero(adv[i] >= 0.01), np.flatnonzero(fav[i] >= 0.02)
        a, b = (hs[0] if len(hs) else 10 ** 9), (ht[0] if len(ht) else 10 ** 9)
        out_[i] = A.TARGET if (b < a) else (A.STOP if a < 10 ** 9 else A.TIME)
    return out_


sd = samp.side.to_numpy()
k1, k5 = plan_on(p1, sd, 60), plan_on(p5, sd, 12)
both = np.isfinite(k1) & np.isfinite(k5)
import lts.aim as A
w1, w5 = (k1[both] == A.TARGET).mean(), (k5[both] == A.TARGET).mean()
flip = ((k5[both] == A.STOP) & (k1[both] == A.TARGET)).sum()
out(f"\nStop-first check, 5x +2%/-1%, {both.sum()} sampled flags with both 1m and 5m candles: won {w5:.1%} on 5m vs "
    f"{w1:.1%} on 1m candles (change {100 * (w1 - w5):+.1f} points); trades counted as stops on 5m that 1m shows "
    f"reached the target first: {flip}; net per trade change about {(w1 - w5) * 0.03:+.3%}")
