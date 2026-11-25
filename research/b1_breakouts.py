"""B1 (docs/HYPOTHESES.md): breakouts with a wide stop and a high reward:risk, against matched random-time controls.

Run: python -m research.b1_breakouts   ->  docs/results/b1_breakouts.txt, b1_variants.csv, b1_frozen.json
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.common import DAY, HOUR, LOCAL_DB, RESULTS, Tee
from lts import validate as V
from lts.db import DB
from lts.fees import fees

out = Tee("b1_breakouts.txt")
FEE = fees()
LEVELS = {"24h": 24, "3d": 72, "7d": 168, "20d": 480}
VOL = ("all", "vol2x")
TF = ("1h", "4h")
STOPS = ("fixed5", "struct")
TARGETS_R = (2, 3, 4)
TIMES_H = (24, 72, 168)
FIRST_GAP_H = 72
MIN_QV24, MIN_HIST = 5_000_000, 21 * 24
CONF_START = int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
N_CTRL, CTRL_WIN, CTRL_EXCL = 5, 72, 6
RNG = np.random.default_rng(25)
MAXH = max(TIMES_H)


def break_even(t, s, tp_cost):
    loss = s + FEE.entry + FEE.stop
    win = t - FEE.entry - tp_cost
    return loss / (win + loss)


out(f"B1 breakouts - run {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC; costs from config: taker {FEE.taker:.4%}, "
    f"maker {FEE.maker:.4%}, slippage {FEE.slip:.4%}")
for t, s in ((0.10, 0.05), (0.15, 0.05), (0.20, 0.05), (0.02, 0.01)):
    out(f"  break-even win rate +{t:.0%} / -{s:.0%}: {break_even(t, s, FEE.target):.1%} (taker TP), "
        f"{break_even(t, s, FEE.target_maker):.1%} (maker TP)")

db = DB(LOCAL_DB)
syms = [r["symbol"] for r in db.query("SELECT DISTINCT symbol FROM klines_1h")]
fund_all = db.frame("SELECT symbol, ts, rate FROM funding")
fund = {s: g[["ts", "rate"]].to_numpy() for s, g in fund_all.groupby("symbol")}
d1 = db.frame("SELECT symbol, ts, c FROM klines_1d WHERE ts >= ?", [int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)])
regime = V.alt_regime(d1.pivot_table(index="ts", columns="symbol", values="c"))


def simulate(o, h, l, c, ts, j, side, risk, tr, th, fsym):
    """Entry at the open of bar j. Returns (net R with taker TP, net R with maker TP, exit bar) or None."""
    if j + th > len(o):
        return None
    e = o[j]
    hi, lo = h[j:j + th], l[j:j + th]
    fav = (hi / e - 1) if side > 0 else (1 - lo / e)
    adv = (1 - lo / e) if side > 0 else (hi / e - 1)
    tgt = tr * risk
    hs, ht = np.flatnonzero(adv >= risk), np.flatnonzero(fav >= tgt)
    i_s = hs[0] if len(hs) else 10 ** 9
    i_t = ht[0] if len(ht) else 10 ** 9
    if i_s <= i_t and i_s < 10 ** 9:
        gross, xc, k = -risk, FEE.stop, i_s
        xcm = xc
    elif i_t < 10 ** 9:
        gross, xc, k = tgt, FEE.target, i_t
        xcm = FEE.target_maker
    else:
        k = th - 1
        gross, xc = side * (c[j + k] / e - 1), FEE.time
        xcm = xc
    t_in, t_out = ts[j], ts[j + k] + HOUR
    fd = 0.0
    if fsym is not None and len(fsym):
        m = (fsym[:, 0] > t_in) & (fsym[:, 0] <= t_out)
        fd = float(side * fsym[m, 1].sum())                     # a long pays a positive funding rate
    n_t = (gross - FEE.entry - xc - fd) / risk
    n_m = (gross - FEE.entry - xcm - fd) / risk
    return n_t, n_m, k


rows = []
t0 = time.time()
TRADES = RESULTS.parent.parent / "research_data" / "b1_trades.parquet"
for si, sym in enumerate([] if TRADES.exists() else syms):    # re-use the simulated trades if already built
    k = db.frame("SELECT ts,o,h,l,c,qv FROM klines_1h WHERE symbol=? ORDER BY ts", [sym]).drop_duplicates("ts")
    if len(k) < MIN_HIST + MAXH:
        continue
    k = k.set_index("ts")
    full = np.arange(k.index[0], k.index[-1] + HOUR, HOUR, dtype=np.int64)
    k = k.reindex(full)
    k["c"] = k["c"].ffill()
    for col in ("o", "h", "l"):
        k[col] = k[col].fillna(k["c"])
    k["qv"] = k["qv"].fillna(0.0)
    ts = k.index.to_numpy()
    o, h, l, c, qv = (k[x].to_numpy(float) for x in ("o", "h", "l", "c", "qv"))
    n = len(ts)
    qv24 = pd.Series(qv).rolling(24).sum().to_numpy()
    qv168 = pd.Series(qv).rolling(168).mean().shift(1).to_numpy()
    prev = np.roll(c, 1)
    prev[0] = c[0]
    tr_ = np.maximum(h - l, np.maximum(abs(h - prev), abs(l - prev)))
    atr = pd.Series(tr_).rolling(14).mean().to_numpy()
    fsym = fund.get(sym)
    for tf in TF:
        if tf == "1h":
            closes = np.arange(n)                                # bar index whose close is the decision
            span = 1
        else:
            closes = np.flatnonzero(((ts + HOUR) % (4 * HOUR)) == 0)  # last hour of each UTC 4h bar
            span = 4
        for lname, N in LEVELS.items():
            hiN = pd.Series(h).rolling(N).max().shift(span).to_numpy()   # N hours before the decision bar
            loN = pd.Series(l).rolling(N).min().shift(span).to_numpy()
            if span == 4:
                bar_qv = pd.Series(qv).rolling(4).sum().to_numpy()
                base = qv168 * 4
            else:
                bar_qv, base = qv, qv168
            for side in (1, -1):
                lvl = hiN if side > 0 else loN
                with np.errstate(invalid="ignore"):
                    brk = (c > lvl) if side > 0 else (c < lvl)
                last_brk = -10 ** 9
                for i in closes:
                    if i < MIN_HIST or i + 1 + MAXH > n or not brk[i]:
                        continue
                    first = (i - last_brk) >= FIRST_GAP_H
                    last_brk = i
                    if not first or not (qv24[i] >= MIN_QV24):
                        continue
                    volok = bool(bar_qv[i] >= 2 * base[i]) if np.isfinite(base[i]) else False
                    j = i + 1
                    e = o[j]
                    lv = lvl[i]
                    for stop in STOPS:
                        if stop == "fixed5":
                            risk = 0.05
                        else:
                            sp = lv - side * 0.5 * atr[i]
                            risk = side * (e - sp) / e
                            if not (0.005 <= risk <= 0.15) or not np.isfinite(risk):
                                continue
                        # controls: random hour opens within +-72h, not within 6h of the real entry
                        cands = np.r_[np.arange(j - CTRL_WIN, j - CTRL_EXCL), np.arange(j + CTRL_EXCL + 1, j + CTRL_WIN + 1)]
                        cands = cands[(cands >= MIN_HIST) & (cands + MAXH <= n)]
                        ctrl = RNG.choice(cands, size=min(N_CTRL, len(cands)), replace=False) if len(cands) else []
                        for tr in TARGETS_R:
                            for th in TIMES_H:
                                r = simulate(o, h, l, c, ts, j, side, risk, tr, th, fsym)
                                if r is None:
                                    continue
                                cr = [simulate(o, h, l, c, ts, int(x), side, risk, tr, th, fsym) for x in ctrl]
                                cr = [x[0] for x in cr if x is not None]
                                rows.append((sym, int(ts[j]), side, tf, lname, volok, stop, tr, th, risk, r[0], r[1],
                                             r[2], float(np.mean(cr)) if cr else np.nan))
    if si % 50 == 0:
        out(f"  {si}/{len(syms)} coins, {len(rows):,} trade rows, {time.time() - t0:.0f}s")

if TRADES.exists() and not rows:
    df = pd.read_parquet(TRADES)
    out("(trades re-used from research_data/b1_trades.parquet, simulated by this script on 25 Nov 2026)")
else:
    df = pd.DataFrame(rows, columns=["symbol", "ts", "side", "tf", "level", "volok", "stop", "tr", "th", "risk",
                                     "r_taker", "r_maker", "bars", "r_ctrl"])
    df.to_parquet(TRADES)
out(f"{len(df):,} trade rows ({df.groupby(['symbol', 'ts', 'side', 'tf', 'level']).ngroups:,} distinct breakouts)")

# ---- variants (volume filter: 'all' includes the vol2x ones) ----
df["regime"] = regime.reindex(df["ts"] // DAY * DAY).to_numpy()
disc = df["ts"] < CONF_START
var = []
for (tf, lname, side, stop, tr, th), g0 in df.groupby(["tf", "level", "side", "stop", "tr", "th"]):
    for vol in VOL:
        g = g0 if vol == "all" else g0[g0["volok"]]
        gd = g[g["ts"] < CONF_START]
        st = V.stats(gd["r_taker"].to_numpy(), gd["ts"].to_numpy())
        var.append({"tf": tf, "level": lname, "vol": vol, "side": side, "stop": stop, "tr": tr, "th": th, **st})
vt = pd.DataFrame(var)
vt["t_corr"] = V.holm(vt["t_day"].fillna(-9).tolist())
vt["name"] = vt.apply(lambda r: f"{r.tf} {r.level} {r.vol} {'long' if r.side > 0 else 'short'} {r.stop} {r.tr}R {r.th}h", axis=1)
vt["pass"] = [V.passes_discovery(r, r["t_corr"])[0] for r in vt.to_dict("records")]
vt.to_csv(RESULTS / "b1_variants.csv", index=False)
out(f"\nDiscovery (entries before {datetime.fromtimestamp(CONF_START / 1000, tz=timezone.utc):%d %b %Y}): "
    f"{len(vt)} variants; passed the bar: {int(vt['pass'].sum())}")
# ties: when Holm sends every corrected t to -inf (adjusted p = 1), rank by the raw day-clustered t
top = vt.sort_values(["t_corr", "t_day"], ascending=False).head(10)
frozen = {r["name"]: {k: r[k] for k in ("tf", "level", "vol", "side", "stop", "tr", "th")}
          for r in pd.concat([vt[vt["pass"]], top]).drop_duplicates("name").to_dict("records")}
# Correction 25 Nov 2026: the first freeze (docs/results/b1_frozen.json) ranked the top 10 by corrected t alone;
# every corrected t was -inf, so it kept 10 arbitrary variants. It is kept as the record; the top 10 by
# (corrected t, raw t) - chosen on discovery only - are frozen as study "b1v2".
sp = V.Split("b1v2", CONF_START, RESULTS)
sp.freeze(frozen)
out(f"Frozen for confirmation ({len(frozen)}: every pass + the discovery top 10 by corrected t, then raw t): "
    f"docs/results/b1v2_frozen.json (b1_frozen.json: the first freeze, see the correction note in the script)")


def table(g, label):
    st = V.stats(g["r_taker"].to_numpy(), g["ts"].to_numpy())
    if not st.get("n"):
        return f"  {label:<26} n 0"
    tr = g["tr"].iloc[0]
    risk = g["risk"].to_numpy()
    be = np.mean([break_even(tr * x, x, FEE.target) for x in risk])
    net_pct = float((g["r_taker"] * g["risk"]).mean())
    cum = g.sort_values("ts")["r_taker"].cumsum().to_numpy()
    dd = float((cum - np.maximum.accumulate(np.r_[0, cum])[1:]).min()) if len(cum) else 0.0
    ctrl = (g["r_taker"] - g["r_ctrl"]).dropna()
    cst = V.stats(ctrl.to_numpy(), g.loc[ctrl.index, "ts"].to_numpy())
    maker = float(g["r_maker"].mean())
    return (f"  {label:<26} n {st['n']:>5} won {st['win']:.1%} (b/e {be:.1%}) {st['mean']:+.3f}R = {net_pct:+.2%} "
            f"(5x margin {5 * net_pct:+.1%}) t {st['t_day']:+.1f} m+ {st['months_pos']:.0%} {st['per_day']:.2f}/day "
            f"maxDD {dd:.1f}R | maker TP {maker:+.3f}R | vs control {cst.get('mean', float('nan')):+.3f}R t {cst.get('t_day', 0):+.1f}")


for name, spec in frozen.items():
    g = df[(df.tf == spec["tf"]) & (df.level == spec["level"]) & (df.side == spec["side"]) & (df.stop == spec["stop"]) &
           (df.tr == spec["tr"]) & (df.th == spec["th"])]
    if spec["vol"] == "vol2x":
        g = g[g["volok"]]
    r = vt[vt["name"] == name].iloc[0]
    tc = "n/a (Holm-adjusted p = 1)" if not np.isfinite(r["t_corr"]) else f"{r['t_corr']:+.2f}"
    out(f"\n{name}   discovery corrected t {tc}  {'PASSED the discovery bar' if r['pass'] else 'did not pass: ' + '; '.join(V.passes_discovery(r.to_dict(), r['t_corr'])[1])}")
    gd, gc = g[g.ts < CONF_START], g[sp.confirmation(g.ts.to_numpy(), name)]
    out(table(gd, "discovery"))
    out(table(gc, "confirmation"))
    if r["pass"]:
        cst = V.stats(gc["r_taker"].to_numpy(), gc["ts"].to_numpy())
        out(f"  confirmation: {'PASSED' if V.passes_confirmation(cst) else 'FAILED'} (positive with t >= 2)")
    for lab, m in (("regime + (alts above 50d)", g.regime > 0), ("regime - (alts below 50d)", g.regime < 0)):
        out(table(g[m], lab))

# whole-family view: how many variants are positive at all, by side and stop
out("\nAll 576 variants, discovery, mean R per trade after costs:")
for (side, stop), g in vt.groupby(["side", "stop"]):
    out(f"  {'long ' if side > 0 else 'short'} {stop:<7} variants {len(g):>3}, positive {int((g['mean'] > 0).sum()):>3}, "
        f">= +0.10R {int((g['mean'] >= 0.10).sum()):>3}, best {g['mean'].max():+.3f}R (t {g.loc[g['mean'].idxmax(), 't_day']:+.1f})")
out(f"Highest raw day-clustered t of any variant in discovery: {vt['t_day'].max():+.2f} "
    f"({vt.loc[vt['t_day'].idxmax(), 'name']}); variants with t >= 2: {int((vt['t_day'] >= 2).sum())}")
out(f"\nverdict: {int(vt['pass'].sum())} of {len(vt)} variants passed the discovery bar.")
