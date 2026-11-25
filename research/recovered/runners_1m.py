"""Exits after +2% on 1-MINUTE candles (5x winners only; losers are unaffected by anything done after +2%).
Both orders inside each 1m candle are run (low first / high first) to show how much the answer depends on it.
Per-trade effect = (winner share) x (average change on winners)."""
import numpy as np, pandas as pd

OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
T, S = 0.02, 0.01
ENTRY, MKT, TGT = 0.0012, 0.0012, 0.0008
z = np.load(OUT + r"\winners_1m.npz")
arr, done = z["arr"], z["done"]
ev = pd.read_parquet(OUT + r"\winners_1m_events.parquet")
r = pd.read_parquet(OUT + r"\runners_results.parquet")
R3 = r.R3.to_numpy()[ev.row]; DISC = r.DISC.to_numpy()[ev.row]
WSHARE = {"all others": 0.330, "new rule": 0.380}
STR = [("trail", 0.0025), ("trail", 0.005), ("trail", 0.0075), ("trail", 0.01), ("hold", 0.03), ("hold", 0.05)]
res = {(k, o): np.full(len(ev), np.nan) for k in STR for o in ("low", "high")}
ok = np.zeros(len(ev), bool)
for i in range(len(ev)):
    P = arr[i].astype(np.float64)
    if not done[i] or not np.isfinite(P[0, 0]) or np.isfinite(P[:, 1]).mean() < 0.97:
        continue
    s = float(ev.side[i]); e = P[0, 0]
    fav = np.nan_to_num((P[:, 1] / e - 1) if s > 0 else (1 - P[:, 2] / e), nan=-9)
    adv = np.nan_to_num((P[:, 2] / e - 1) if s > 0 else (1 - P[:, 1] / e), nan=9)
    cl = s * (P[:, 3] / e - 1); last = cl[np.isfinite(cl)][-1]
    jh = None
    for j in range(len(P)):
        if adv[j] <= -S:
            break
        if fav[j] >= T:
            jh = j; break
    if jh is None:
        continue                                           # 1m says the stop came first: not a winner here
    ok[i] = True
    for (kind, x) in STR:
        for order in ("low", "high"):
            pk, out = T, None
            for k in range(jh + 1, len(P)):
                if kind == "hold":
                    if adv[k] <= 0.01:
                        out = 0.01 - ENTRY - MKT; break
                    if fav[k] >= x:
                        out = x - ENTRY - TGT; break
                    continue
                old = max(0.01, pk - x)
                if order == "low":                          # low first: old stop, then new high
                    if adv[k] <= old:
                        out = old - ENTRY - MKT; break
                    pk = max(pk, fav[k])
                else:                                       # high first: new high, then the low hits the raised stop
                    pk = max(pk, fav[k]); new = max(0.01, pk - x)
                    if adv[k] <= new:
                        out = new - ENTRY - MKT; break
            res[((kind, x), order)][i] = out if out is not None else last - ENTRY - MKT
base = T - ENTRY - TGT
print(f"winners with 1m data: {ok.sum()} of {done.sum()} fetched ({(ok & R3).sum()} new-rule); TP2 banks {base:+.2%}")
for gname, gm in (("ALL OTHER GRINDS", ok & ~R3), ("NEW RULE (all-day steady)", ok & R3)):
    ws = WSHARE["new rule" if "NEW" in gname else "all others"]
    print(f"\n=== {gname}: {gm.sum()} winners ===")
    for k in STR:
        lab = f"trail {k[1]:.2%} (floor +1%)" if k[0] == "trail" else f"stop to +1%, take {k[1]:.0%}"
        line = f"  {lab:<24}"
        for order in ("low", "high"):
            v = res[(k, order)]
            d1, d2 = v[gm & DISC] - base, v[gm & ~DISC] - base
            line += (f" | {order}-first: winners avg {v[gm].mean():+.2%} med {np.median(v[gm]):+.2%} -> per trade "
                     f"first {ws * d1.mean():+.3%} last {ws * d2.mean():+.3%}")
            if k[0] == "hold":
                break
        print(line)
