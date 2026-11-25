"""Neighbourhood check for the trailing stop after +2% (winners only; losers are unchanged by any of these).
TRAIL(start, gap): once +2% is touched the stop moves to max(start, best - gap), checked from the next candle.
Also: what the winners actually banked under each, and how the trailing exit compares to the +2% TP."""
import glob
import numpy as np, pandas as pd

OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
DAY = 86_400_000
T, S = 0.02, 0.01
ENTRY, MKT, TGT = 0.0012, 0.0012, 0.0008
r = pd.read_parquet(OUT + r"\runners_results.parquet")
a = pd.read_parquet(OUT + r"\aim_events.parquet")
month = pd.to_datetime(a.ts, unit="ms").dt.strftime("%Y-%m").to_numpy()
day = (a.ts // DAY).to_numpy()
side = a.side.to_numpy().astype(float)
win, valid, R3, OLD, DISC = (r[c].to_numpy() for c in ("win", "valid", "R3", "OLD", "DISC"))

GRID = [(0.01, 0.0025), (0.01, 0.004), (0.01, 0.005), (0.01, 0.006), (0.01, 0.0075), (0.01, 0.01),
        (0.015, 0.005), (0.0175, 0.0025), (0.005, 0.005), (0.0, 0.005)]
res = {g: np.full(len(a), np.nan) for g in GRID}
start = 0
for fp in sorted(glob.glob(OUT + r"\paths\chunk_*.npy")):
    post = np.load(fp)
    for i in range(len(post)):
        g = start + i
        if g >= len(a) or not win[g]:
            continue
        P = post[i, :288].astype(np.float64)
        s = side[g]; e = P[0, 0]
        fav = np.nan_to_num((P[:, 1] / e - 1) if s > 0 else (1 - P[:, 2] / e), nan=-9)
        adv = np.nan_to_num((P[:, 2] / e - 1) if s > 0 else (1 - P[:, 1] / e), nan=9)
        cl = s * (P[:, 3] / e - 1)
        last = cl[np.isfinite(cl)][-1]
        jh = int(np.argmax(fav >= T))
        for (st0, gap) in GRID:
            pk, out = T, None
            for k in range(jh + 1, 288):
                stp = max(st0, pk - gap)
                if adv[k] <= stp:
                    out = stp - ENTRY - MKT; break
                pk = max(pk, fav[k])
            res[(st0, gap)][g] = out if out is not None else last - ENTRY - MKT
    start += len(post)
print()
base_w = T - ENTRY - TGT


def dstat(v, m):
    m = m & np.isfinite(v)
    x = v[m]
    d = pd.Series(x).groupby(day[m]).mean()
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return f"{x.mean():+.3%} t {t:+.1f}"


for gname, gm in (("ALL GRINDS", valid), ("NEW RULE", valid & R3), ("OLD RULE", valid & OLD)):
    print(f"\n=== {gname} ===  TP2 per trade: first {dstat(r.net_TP2.to_numpy(), gm & DISC)} | last {dstat(r.net_TP2.to_numpy(), gm & ~DISC)}")
    w = gm & win
    print(f"winners {w.sum()}: TP2 banks {base_w:+.2%} each")
    for k in GRID:
        full = r.net_TP2.to_numpy().copy()
        full[w] = res[k][w]
        wv = res[k][w]
        print(f"  trail from +{k[0]:.2%}, gap {k[1]:.2%}: winners avg {wv.mean():+.2%} median {np.median(wv):+.2%} "
              f"(better than TP2 in {np.mean(wv > base_w):.0%}) | per trade first {dstat(full, gm & DISC)} | last {dstat(full, gm & ~DISC)}")

# whole-sample all-grinds descriptive
w = valid & win
print(f"\nALL GRINDS: {valid.sum()} trades, winners {w.sum()} ({w.sum() / valid.sum():.1%}); best move median {np.nanmedian(r.best[w]):.2%} "
      f"p25 {np.nanpercentile(r.best[w], 25):.2%} p75 {np.nanpercentile(r.best[w], 75):.2%}; back to +1% {r.back1[w].mean():.0%}, to entry {r.back_entry[w].mean():.0%}; "
      f"24h close median {np.nanmedian(r.close24[w]):+.2%}")
