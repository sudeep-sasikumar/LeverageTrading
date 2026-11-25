"""P0 (docs/HYPOTHESES.md): reproduce DESIGN.md section 27 (how far 5x winners run after +2%).

Descriptive table on the rebuilt 5-minute paths; the 1-minute exit table on the 1-minute candles kept from the
original run (research_data/winners_1m.npz: every new-rule winner + 1,500 random other winners, fetched 20 Nov).
The old rule (model top 20%) uses the original walk-forward scores (research_data/five_x_scores.parquet).
Run: python -m research.reproduce_runners  ->  docs/results/reproduce_runners.txt
"""
import numpy as np
import pandas as pd

from research.common import EVENTS, RD, Tee, halves, load_paths, r3_mask

from lts.fees import fees

out = Tee("reproduce_runners.txt")
F = fees()
T, S = 0.02, 0.01
LEVELS = (0.03, 0.05, 0.10)
ev = pd.read_parquet(EVENTS).reset_index(drop=True)
sc = pd.read_parquet(RD / "five_x_scores.parquet")[["symbol", "ts", "side", "p_minute shape (today)"]]
a = ev.merge(sc, on=["symbol", "ts", "side"], how="left")
assert len(a) == len(ev)
p = a["p_minute shape (today)"]
OLD = (p.notna() & (p >= p[p.notna()].quantile(0.8))).to_numpy()
R3 = r3_mask(ev)
post, _ = load_paths(ev[["symbol", "ts"]], 289)
side = ev.side.to_numpy().astype(float)
N = len(ev)
win = np.zeros(N, bool); valid = np.zeros(N, bool)
best = np.full(N, np.nan); close24 = np.full(N, np.nan); back1 = np.zeros(N, bool); back0 = np.zeros(N, bool)
reach = {L: np.zeros(N, bool) for L in LEVELS}; safe = {L: np.zeros(N, bool) for L in LEVELS}
t_hit = np.full(N, np.nan); t_peak = np.full(N, np.nan)
for g in range(N):
    P = post[g, :288].astype(np.float64)
    if not np.isfinite(P[0, 0]) or np.isfinite(P[:, 1]).mean() < 0.98:
        continue
    valid[g] = True
    s, e = side[g], P[0, 0]
    fav = np.nan_to_num((P[:, 1] / e - 1) if s > 0 else (1 - P[:, 2] / e), nan=-9)
    adv = np.nan_to_num((P[:, 2] / e - 1) if s > 0 else (1 - P[:, 1] / e), nan=9)
    cl = s * (P[:, 3] / e - 1)
    last = cl[np.isfinite(cl)][-1]
    jh = None
    for j in range(288):
        if adv[j] <= -S:
            break
        if fav[j] >= T:
            jh = j
            break
    if jh is None:
        continue
    win[g] = True
    f2, a2 = fav[jh:], adv[jh + 1:]
    best[g], close24[g] = f2.max(), last
    t_hit[g], t_peak[g] = (jh + 1) * 5, (int(np.argmax(f2)) + jh + 1) * 5
    back1[g], back0[g] = (a2 <= 0.01).any(), (a2 <= 0).any()
    fb = int(np.argmax(a2 <= 0.01)) if back1[g] else 10 ** 9
    for L in LEVELS:
        hit = f2 >= L
        reach[L][g] = hit.any()
        safe[L][g] = hit.any() and int(np.argmax(hit)) <= fb

PUB = {"all": "20,984 (33.0%) | 6.2% (3.8/10.4) | 86/62/26 | 50/24/9 | 89 | 78 | +1.3% | 55/595",
       "old": "3,338 (33.9%) | 6.0% (3.7/10.2) | 85/61/26 | 50/26/9 | 87 | 75 | +1.5% | 60/595",
       "new": "205 (38.0%) | 7.3% (4.3/12.6) | 91/69/34 | 48/25/10 | 94 | 84 | +1.2% | 50/615"}
out("Winners = touched +2% before -1%. Columns: winners | best point median (p25/p75) | reached +3/+5/+10% | ...before "
    "falling back to +1% | back to +1% | back to entry | 24h close median | minutes to +2% / to best (median)")
for key, lab, gm in (("all", "All grinds", valid), ("old", "Old rule (model top 20%)", valid & OLD),
                     ("new", "New rule (all-day steady)", valid & R3)):
    w = gm & win
    row = (f"{w.sum():,} ({w.sum() / gm.sum():.1%}) | {np.median(best[w]):.1%} ({np.percentile(best[w], 25):.1%}/"
           f"{np.percentile(best[w], 75):.1%}) | " + "/".join(f"{reach[L][w].mean():.0%}".rstrip('%') for L in LEVELS) +
           " | " + "/".join(f"{safe[L][w].mean():.0%}".rstrip('%') for L in LEVELS) +
           f" | {back1[w].mean():.0%} | {back0[w].mean():.0%} | {np.median(close24[w]):+.1%} | "
           f"{np.median(t_hit[w]):.0f}/{np.median(t_peak[w]):.0f}")
    out(f"\n{lab}\n  rebuilt:   {row}\n  published: {PUB[key]}")

# ---- 1-minute exit table on the kept 1-minute candles ----
z = np.load(RD / "winners_1m.npz")
arr, done = z["arr"], z["done"]
we = pd.read_parquet(RD / "winners_1m_events.parquet")
same = (ev.loc[we.row, ["symbol", "ts", "side"]].to_numpy() == we[["symbol", "ts", "side"]].to_numpy()).all()
out(f"\n1-minute sample: {len(we):,} winners kept from 20 Nov; rows match the rebuilt event table: {bool(same)}")
disc, _ = halves(ev.ts.to_numpy())
R3w, DISCw = R3[we.row], disc[we.row]
base = T - F.entry - F.target
STR = [("trail", 0.0025), ("trail", 0.005), ("trail", 0.01), ("hold", 0.03), ("hold", 0.05)]
res = {(k, o): np.full(len(we), np.nan) for k in STR for o in ("low", "high")}
ok = np.zeros(len(we), bool)
for i in range(len(we)):
    P = arr[i].astype(np.float64)
    if not done[i] or not np.isfinite(P[0, 0]) or np.isfinite(P[:, 1]).mean() < 0.97:
        continue
    s, e = float(we.side[i]), P[0, 0]
    fav = np.nan_to_num((P[:, 1] / e - 1) if s > 0 else (1 - P[:, 2] / e), nan=-9)
    adv = np.nan_to_num((P[:, 2] / e - 1) if s > 0 else (1 - P[:, 1] / e), nan=9)
    cl = s * (P[:, 3] / e - 1)
    last = cl[np.isfinite(cl)][-1]
    jh = None
    for j in range(len(P)):
        if adv[j] <= -S:
            break
        if fav[j] >= T:
            jh = j
            break
    if jh is None:
        continue
    ok[i] = True
    for kind, x in STR:
        for order in ("low", "high"):
            pk, o_ = T, None
            for k in range(jh + 1, len(P)):
                if kind == "hold":
                    if adv[k] <= 0.01:
                        o_ = 0.01 - F.entry - F.stop
                        break
                    if fav[k] >= x:
                        o_ = x - F.entry - F.target
                        break
                    continue
                old = max(0.01, pk - x)
                if order == "low":
                    if adv[k] <= old:
                        o_ = old - F.entry - F.stop
                        break
                    pk = max(pk, fav[k])
                else:
                    pk = max(pk, fav[k])
                    new = max(0.01, pk - x)
                    if adv[k] <= new:
                        o_ = new - F.entry - F.stop
                        break
            res[((kind, x), order)][i] = o_ if o_ is not None else last - F.entry - F.time
out("Per trade vs taking +2% (winner share x average change on winners), first 7 / last 7 months; "
    "trails: low-first ... high-first inside each 1m candle")
for lab, gm, ws in (("All other grinds", ok & ~R3w, 0.330), ("New rule", ok & R3w, 0.380)):
    out(f"  {lab} ({gm.sum()} winners):")
    for k in STR:
        name = f"trail {k[1]:.2%}" if k[0] == "trail" else f"stop to +1%, take {k[1]:.0%}"
        cells = []
        for half in (DISCw, ~DISCw):
            v = [ws * np.nanmean(res[(k, o)][gm & half] - base) for o in (("low", "high") if k[0] == "trail" else ("low",))]
            cells.append("..".join(f"{x:+.2%}" for x in v))
        out(f"    {name:<22} {cells[0]} / {cells[1]}")
out("  (published, DESIGN.md section 27 table: e.g. trail 0.25% new rule +0.00..+0.03% / +0.02..+0.06%)")
