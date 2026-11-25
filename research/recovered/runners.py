"""How far do the 5x winners run after +2%?

Year data: 63,528 grinds, 289 5m candles (o,h,l,c) after entry (entry = open of candle 0).
Plan: +2% / -1%, 24h, stop checked first inside a candle. A WINNER = touched +2% before -1%.
For winners, from the candle that touched +2% to 24h after entry:
  best: furthest move from entry; reach_L: share that got to L from entry;
  safe_L: share that got to L BEFORE coming back to +1% (could be held with the stop moved to +1%);
  back_entry: share that came all the way back to entry; close_24h: where it was at 24h.
Then whole-plan alternatives (ALL trades, net per trade after fees, first 7 / last 7 months), written before testing:
  TP2        current: take +2%
  HOLD_L     at +2% the stop moves to +1%; take L (3%, 3.33%, 4%, 5%); else 24h close
  TRAIL_g    at +2% the stop moves to +1%, then trails g behind the best point (g = 0.5%, 1%); else 24h close
  HALF       half at +2%, other half as TRAIL_1%
"""
import glob
import numpy as np, pandas as pd

OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
DAY = 86_400_000
T, S = 0.02, 0.01
ENTRY, MKT, TGT = 0.0012, 0.0012, 0.0008
LEVELS = (0.025, 0.03, 0.0333, 0.04, 0.05, 0.07, 0.10)

a = pd.read_parquet(OUT + r"\aim_events.parquet").reset_index().rename(columns={"index": "row"})
sc = pd.read_parquet(OUT + r"\five_x_scores.parquet")[["symbol", "ts", "side", "p_minute shape (today)"]]
a = a.merge(sc, on=["symbol", "ts", "side"], how="left")
R3 = ((a.r_24h >= 0.08) & (a.r2_24h >= 0.8) & (a.hours_with24 >= 15) & (a.r2_6h >= 0.9) & (a.hours_with6 >= 5)
      & (a.big6 <= 0.35)).to_numpy()
p = a["p_minute shape (today)"]
OLD = (p.notna() & (p >= p[p.notna()].quantile(0.8))).to_numpy()
month = pd.to_datetime(a.ts, unit="ms").dt.strftime("%Y-%m").to_numpy()
months = sorted(set(month))
DISC = np.isin(month, months[:len(months) // 2])
day = (a.ts // DAY).to_numpy()
side_all = a.side.to_numpy().astype(float)

STRATS = ["TP2", "HOLD_0.03", "HOLD_0.0333", "HOLD_0.04", "HOLD_0.05", "TRAIL_0.005", "TRAIL_0.01", "HALF"]
N = len(a)
win = np.zeros(N, bool); hitbar = np.full(N, -1)
best = np.full(N, np.nan); close24 = np.full(N, np.nan); back_entry = np.zeros(N, bool); back1 = np.zeros(N, bool)
reach = {L: np.zeros(N, bool) for L in LEVELS}; safe = {L: np.zeros(N, bool) for L in LEVELS}
peak_min = np.full(N, np.nan)
net = {s: np.full(N, np.nan) for s in STRATS}
valid = np.zeros(N, bool)

start = 0
for fp in sorted(glob.glob(OUT + r"\paths\chunk_*.npy")):
    post = np.load(fp)
    for i in range(len(post)):
        g = start + i
        if g >= N:
            break
        P = post[i, :288].astype(np.float64)
        if not np.isfinite(P[0, 0]) or np.isfinite(P[:, 1]).mean() < 0.98:
            continue
        valid[g] = True
        s = side_all[g]; e = P[0, 0]
        fav = (P[:, 1] / e - 1) if s > 0 else (1 - P[:, 2] / e)
        adv = (P[:, 2] / e - 1) if s > 0 else (1 - P[:, 1] / e)        # worst point, trade direction (negative = against)
        cl = s * (P[:, 3] / e - 1)
        fav = np.nan_to_num(fav, nan=-9); adv = np.nan_to_num(adv, nan=9)
        last = cl[np.isfinite(cl)][-1]
        # base plan
        jh = None
        for j in range(288):
            if adv[j] <= -S:
                break
            if fav[j] >= T:
                jh = j
                break
        if jh is None:                                          # stopped or timed out: same for every strategy
            r = (-S - ENTRY - MKT) if (adv <= -S).any() and j < 288 and adv[j] <= -S else last - ENTRY - MKT
            for st in STRATS:
                net[st][g] = r
            continue
        win[g] = True; hitbar[g] = jh
        f2, a2, c2 = fav[jh:], adv[jh + 1:], cl[jh + 1:]
        best[g] = f2.max(); close24[g] = last
        peak_min[g] = (int(np.argmax(f2)) + jh + 1) * 5
        back1[g] = (a2 <= 0.01).any(); back_entry[g] = (a2 <= 0).any()
        first_back1 = int(np.argmax(a2 <= 0.01)) if back1[g] else 10 ** 9
        for L in LEVELS:
            hitL = f2 >= L
            reach[L][g] = hitL.any()
            # safe: reaches L before a later candle trades back to +1% (the hit candle itself counts as reached)
            fL = int(np.argmax(hitL)) if hitL.any() else 10 ** 9
            safe[L][g] = hitL.any() and fL <= first_back1           # f2 index 0 = hit candle, a2 index 0 = next candle
        net["TP2"][g] = T - ENTRY - TGT
        for L in (0.03, 0.0333, 0.04, 0.05):
            r = None
            for k in range(jh + 1, 288):
                if adv[k] <= 0.01:
                    r = 0.01 - ENTRY - MKT; break
                if fav[k] >= L:
                    r = L - ENTRY - TGT; break
            net[f"HOLD_{L}"][g] = r if r is not None else last - ENTRY - MKT
        for gap in (0.005, 0.01):
            pk, r = T, None
            for k in range(jh + 1, 288):
                stp = max(0.01, pk - gap)
                if adv[k] <= stp:
                    r = stp - ENTRY - MKT; break
                pk = max(pk, fav[k])
            net[f"TRAIL_{gap}"][g] = r if r is not None else last - ENTRY - MKT
        net["HALF"][g] = 0.5 * net["TP2"][g] + 0.5 * net["TRAIL_0.01"][g]
    start += len(post)
    print(".", end="", flush=True)
print()


def dstat(v, m):
    m = m & np.isfinite(v)
    x = v[m]
    if len(x) < 30:
        return None
    d = pd.Series(x).groupby(day[m]).mean()
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    mo = pd.Series(x).groupby(month[m]).mean()
    return f"n {len(x):>6} {x.mean():+.3%} t {t:+.1f} m+ {(mo > 0).mean():.0%}"


GROUPS = (("ALL GRINDS", valid), ("NEW RULE (all-day steady)", valid & R3), ("OLD RULE (model top 20%)", valid & OLD))
for gname, gm in GROUPS:
    w = gm & win
    print(f"\n=== {gname}: {gm.sum()} trades, {w.sum()} winners ({w.sum() / gm.sum():.1%}) ===")
    print(f"winners: best move median {np.median(best[w]):.2%}  p25 {np.percentile(best[w], 25):.2%}  "
          f"p75 {np.percentile(best[w], 75):.2%}  p90 {np.percentile(best[w], 90):.2%}")
    print(f"  close at 24h median {np.median(close24[w]):+.2%}; came back to +1%: {back1[w].mean():.0%}; to entry: {back_entry[w].mean():.0%}")
    print(f"  minutes to +2% median {np.median((hitbar[w] + 1) * 5):.0f}; to best point median {np.median(peak_min[w]):.0f}")
    for L in LEVELS:
        print(f"  reached {L:.2%}: {reach[L][w].mean():5.1%}   before falling back to +1%: {safe[L][w].mean():5.1%}"
              f"   (first 7 mo {safe[L][w & DISC].mean():.1%} / last 7 {safe[L][w & ~DISC].mean():.1%})")
    print(f"  {'strategy':<12} {'first 7 months':>36} | {'last 7 months':>36}")
    for st in STRATS:
        print(f"  {st:<12} {dstat(net[st], gm & DISC) or 'n<30':>36} | {dstat(net[st], gm & ~DISC) or 'n<30':>36}")

pd.DataFrame({"win": win, "best": best, "close24": close24, "back1": back1, "back_entry": back_entry, "R3": R3, "OLD": OLD,
              "DISC": DISC, "valid": valid, **{f"net_{s}": net[s] for s in STRATS},
              **{f"safe_{L}": safe[L] for L in LEVELS}}).to_parquet(OUT + r"\runners_results.parquet")
