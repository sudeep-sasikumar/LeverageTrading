"""B3 (docs/HYPOTHESES.md): limit entries and reduce-only limit take-profits on 1-minute candles.

Run: python -m research.b3_execution  ->  docs/results/b3_execution.txt (fetches the 1-minute candles it needs)
"""
from __future__ import annotations

import asyncio
import json

import aiohttp
import numpy as np
import pandas as pd

from research.common import EVENTS, HOUR, LOCAL_DB, MIN, RD, Tee, load_paths, r3_mask
from research.fetch_candles import run as fetch

from lts import sweeps as SW
from lts import validate as V
from lts.db import DB
from lts.fees import fees

out = Tee("b3_execution.txt")
F = fees()
T, S, WAIT = 0.02, 0.01, 15
PRE = 15                                          # minutes of candles before the signal

out("Weex rules (official API docs, read 25 Nov 2026, www.weex.com/api-doc/catalog/core-trading-futures/api/rest-api/"
    "transaction-api): POST /capi/v3/order takes type=LIMIT, reduceOnly=true and timeInForce GTC/IOC/FOK/POST_ONLY;")
out("  position TP/SL (/capi/v3/placeTpSlOrder) executes at market when executePrice is 0 or omitted, or as a limit when"
    " it is positive (full-position TP/SL: market only). Exchange info: makerFeeRate 0.0002, takerFeeRate 0.0008.")
out("  NOT documented: whether a filled reduce-only limit close is charged the maker rate. Evidence below = the trader's fills.")

# ---- the trader's fills ----------------------------------------------------------------------------------------------
mt = json.loads((RD / "my_trades.json").read_text(encoding="utf-8"))
out(f"\nThe trader's recorded trades ({len(mt)}): fee as a share of the filled notional")
for t in mt:
    notional_in = t["margin"] * t["leverage"]
    notional_out = notional_in * t["exit_price"] / t["entry_price"]
    slip = t["side"] * (t["entry_price"] / t["signal_entry"] - 1)
    out(f"  {t['symbol']:<10} entry fee {t['open_fee'] / notional_in:.4%}, TP fee {t['close_fee'] / notional_out:.4%} "
        f"(exit {t['exit_reason']}); entry vs signal price {slip:+.3%} ({'worse' if slip > 0 else 'better or equal'})")
out("  -> every fill, the position TP included, was charged the taker rate (0.08%); none was a maker fill.")

# ---- sample ------------------------------------------------------------------------------------------------------------
ev = pd.read_parquet(EVENTS).reset_index(drop=True)
R3 = r3_mask(ev)
rest = np.flatnonzero(~R3)
pick = np.r_[np.flatnonzero(R3), np.random.default_rng(25).choice(rest, 2000, replace=False)]
sm = ev.iloc[pick][["symbol", "ts", "side", "price"]].reset_index(drop=True)
sm["steady"] = np.r_[np.ones(R3.sum(), bool), np.zeros(2000, bool)]
asyncio.run(fetch([(s, int(t) - PRE * MIN, int(t) + 24 * HOUR) for s, t in zip(sm.symbol, sm.ts)], "1m", log=out))


async def ticks():
    async with aiohttp.ClientSession() as s:
        async with s.get("https://fapi.binance.com/fapi/v1/exchangeInfo") as r:
            info = await r.json()
    return {x["symbol"]: float(next(f["tickSize"] for f in x["filters"] if f["filterType"] == "PRICE_FILTER"))
            for x in info["symbols"]}
tick = asyncio.run(ticks())
start = sm.assign(ts=sm.ts - PRE * MIN)
p1, _ = load_paths(start[["symbol", "ts"]], PRE + 1440, table="k1", step=MIN)

rows = []
for i, r in sm.iterrows():
    P = p1[i, PRE:].astype(float)                    # from the signal minute on
    if not np.isfinite(P[0, 0]) or np.isfinite(P[:, 1]).mean() < 0.97:
        continue
    s, tk = r.side, tick.get(r.symbol, np.nan)
    if not np.isfinite(tk):
        tk = abs(P[0, 0]) * 1e-4
    sig = float(r.price)                             # the hour close the signal was based on
    o, h, l, c = P[:, 0], P[:, 1], P[:, 2], P[:, 3]

    def run(entry, j0, tp_through):
        tp, sl = entry * (1 + s * T), entry * (1 - s * S)
        for j in range(j0, len(o)):
            if not np.isfinite(h[j]):
                continue
            if (s > 0 and l[j] <= sl) or (s < 0 and h[j] >= sl):
                return "stop", j
            hit = (h[j] >= tp + (tk if tp_through else 0)) if s > 0 else (l[j] <= tp - (tk if tp_through else 0))
            if hit and j > j0 - (0 if tp_through else 1):
                return "target", j
        return "time", len(o) - 1

    def net(kind, entry, j, ein, tp_cost):
        if kind == "stop":
            return -S - ein - F.stop
        if kind == "target":
            return T - ein - tp_cost
        last = c[np.isfinite(c)][-1]
        return s * (last / entry - 1) - ein - F.time

    # market entry at the open of the signal minute
    e_m = o[0]
    km, jm = run(e_m, 0, False)
    kmm, jmm = run(e_m, 0, True)
    # limit at the signal price for WAIT minutes: fills only if price trades through it by a tick
    fill = None
    for j in range(min(WAIT, len(o))):
        if (s > 0 and l[j] <= sig - tk) or (s < 0 and h[j] >= sig + tk):
            fill = j
            break
    row = {"symbol": r.symbol, "ts": int(r.ts), "side": s, "steady": r.steady,
           "a": net(km, e_m, jm, F.entry, F.target), "c": net(kmm, e_m, jmm, F.entry, F.target_maker),
           "win_all": km == "target", "filled": fill is not None}
    if fill is not None:
        kl, jl = run(sig, fill, False)
        klm, jlm = run(sig, fill, True)
        row["b"] = net(kl, sig, jl, F.entry_maker, F.target)
        row["d"] = net(klm, sig, jlm, F.entry_maker, F.target_maker)
        row["win_filled"] = kl == "target"
    rows.append(row)
d = pd.DataFrame(rows)
out(f"\n1-minute sample: {len(d):,} trades with candles ({int(d.steady.sum())} all-day steady, {int((~d.steady).sum())} random)")
pass_rows = []
for lab, g in (("all-day steady (R3)", d[d.steady]), ("random early grinds", d[~d.steady])):
    fr = g.filled.mean()
    wa = g.win_all.mean()
    wf = g.loc[g.filled, "win_filled"].mean()
    missed = (g.win_all & ~g.filled).sum()
    out(f"\n{lab}: {len(g):,} signals; limit filled {fr:.0%}; won {wa:.1%} of all (market entry) vs {wf:.1%} of filled "
        f"(adverse selection {100 * (wf - wa):+.1f} points); winners missed because the limit never filled: {missed}")
    for v, name in (("a", "(a) market in, taker TP (today)"), ("b", "(b) limit in, taker TP"),
                    ("c", "(c) market in, maker TP (fills only through +1 tick)"), ("d", "(d) limit in, maker TP")):
        x = g[v].dropna()
        st = V.stats(x.to_numpy() / S, g.loc[x.index, "ts"].to_numpy())
        per_signal = x.sum() / len(g)
        out(f"  {name:<54} trades {st['n']:>5} won {st['win']:.1%} net {x.mean():+.3%} per trade "
            f"({x.mean() / S:+.3f}R), {per_signal:+.3%} per signal; t {st['t_day']:+.2f}; months+ {st['months_pos']:.0%}")
        if v != "a":
            pass_rows.append((f"{lab} {name}", st))
tc = V.holm([st["t_day"] for _, st in pass_rows])
out("\nPass bar (family of 6: variants b-d x 2 samples, R = the 1% stop):")
for (name, st), t_ in zip(pass_rows, tc):
    ok, why = V.passes_discovery(st, t_)
    out(f"  {name}: {'PASSED' if ok else 'not passed: ' + '; '.join(why)}")

# ---- the best section 14 sweep variant with the maker TP fill rule --------------------------------------------------------
db = DB(LOCAL_DB)
res = {"taker TP": [], "maker TP (through +0.01%)": []}
tss = []
for sym in [r["symbol"] for r in db.query("SELECT DISTINCT symbol FROM klines_1h")]:
    k = db.frame("SELECT ts,o,h,l,c,qv FROM klines_1h WHERE symbol=? ORDER BY ts", [sym]).drop_duplicates("ts").reset_index(drop=True)
    if len(k) < 500:
        continue
    o, h, l, c = (k[x].to_numpy(float) for x in "ohlc")
    for ev_ in SW.detect(k):
        if ev_["side"] != -1 or ev_["i"] + 1 + SW.HORIZON_H > len(o):
            continue
        j = ev_["i"] + 1
        e = o[j]
        p = SW.plan(ev_, e, "wide", "3R")
        if p is None:
            continue
        stop, tgt = p
        risk = abs(e - stop) / e
        if not (0.004 <= risk <= 0.10):
            continue
        for name, through in (("taker TP", 0.0), ("maker TP (through +0.01%)", 1e-4)):
            r_ = None
            for m in range(SW.HORIZON_H):
                if h[j + m] >= stop:
                    r_ = (-risk - F.entry - F.stop) / risk
                    break
                if m > 0 and l[j + m] <= tgt * (1 - through):
                    r_ = (abs(tgt - e) / e - F.entry - (F.target if through == 0 else F.target_maker)) / risk
                    break
            if r_ is None:
                r_ = (-(c[j + SW.HORIZON_H - 1] / e - 1) - F.entry - F.time) / risk
            res[name].append(r_)
        tss.append(int(k.ts.iloc[j]))
out("\nBest section 14 sweep variant (short after a 24h-high sweep, market entry, stop 1 ATR beyond the wick, 3R, 48h):")
for name, v in res.items():
    st = V.stats(np.array(v), np.array(tss))
    out(f"  {name:<28} n {st['n']:,} won {st['win']:.1%} {st['mean']:+.3f}R t {st['t_day']:+.2f} months+ {st['months_pos']:.0%}")
