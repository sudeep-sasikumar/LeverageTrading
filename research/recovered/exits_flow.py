"""Flow-based early exits (taker buy/sell and volume), on a sample of the year's grinds.

Plan as before: 5x, +2% / -1%, 24h, taker fees. Exit rules (written before testing), decided on a 5m close:
  FLOW15      once +0.5%, the last 3 candles' taker flow <= -0.2 against us and the close is lower than the last
  FLOW_EMA    once +0.5%, a close through the 5m EMA20 while the last 3 candles' taker flow is against us
  VOLSPIKE    once +0.5%, a candle against us with >= 2x the average volume since entry, closing in its third against us
  FLOWFLIP    once +0.5%, takers were with us from entry to the best point (flow > +0.1) and the last 6 candles are
              against us (flow < -0.1)
For comparison: BASE, and the best price rules from the price-only test (BACK_0.005, EMA_0.01, WICK).
Taker flow = (2 x taker-buy volume / volume - 1) x side: +1 all takers pushing our way, -1 all against.
"""
import numpy as np, pandas as pd

OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
DAY = 86_400_000
T, S = 0.02, 0.01
ENTRY = MKT = 0.0012
TGT = 0.0008
z = np.load(OUT + r"\flow_paths.npz")
arr, done = z["arr"], z["done"]
ev = pd.read_parquet(OUT + r"\flow_events.parquet")
a = pd.read_parquet(OUT + r"\aim_events.parquet")
R3all = ((a.r_24h >= 0.08) & (a.r2_24h >= 0.8) & (a.hours_with24 >= 15) & (a.r2_6h >= 0.9) & (a.hours_with6 >= 5)
         & (a.big6 <= 0.35)).to_numpy()
R3 = R3all[ev.row.to_numpy()]
import glob
pre = np.concatenate([np.load(f, mmap_mode="r") for f in sorted(glob.glob(OUT + r"\pre\chunk_*.npy"))])[ev.row.to_numpy()]
ok = done & np.isfinite(arr[:, 0, 0]) & (np.isfinite(arr[:, :, 1]).mean(1) > 0.97)
month = pd.to_datetime(ev.ts, unit="ms").dt.strftime("%Y-%m").to_numpy()
months = sorted(set(pd.to_datetime(a.ts, unit="ms").dt.strftime("%Y-%m")))
DISC = np.isin(month, months[:len(months) // 2])
day = (ev.ts // DAY).to_numpy()
side = ev.side.to_numpy().astype(float)
RULES = ["BASE", "BACK_0.005", "EMA_0.01", "WICK", "FLOW15", "FLOW_EMA", "VOLSPIKE", "FLOWFLIP"]


def simulate(rule):
    n, Bn = len(ev), 288
    o, h, l, c, qv, tb = (arr[:, :Bn, k].astype(np.float64) for k in range(6))
    e = o[:, 0]
    s = side
    fav_h = np.where(s[:, None] > 0, h / e[:, None] - 1, 1 - l / e[:, None])
    adv_l = np.where(s[:, None] > 0, 1 - l / e[:, None], h / e[:, None] - 1)
    cl = s[:, None] * (c / e[:, None] - 1)
    flow = np.where(qv > 0, (2 * tb / np.where(qv > 0, qv, 1) - 1), 0.0) * s[:, None]
    rng = h - l
    pos_against = np.where(s[:, None] > 0, (h - c), (c - l)) / np.where(rng > 0, rng, np.nan)   # 1 = closed at the low (long)
    wick = np.where(s[:, None] > 0, h - np.maximum(o, c), np.minimum(o, c) - l) / np.where(rng > 0, rng, np.nan)
    k = 2 / 21
    ema = np.full(n, np.nan)
    for j in range(pre.shape[1]):
        x = pre[:, j, 3].astype(np.float64)
        ema = np.where(np.isnan(ema), x, ema + k * (x - ema))
    net = np.full(n, np.nan)
    open_ = ok.copy()
    mfe = np.zeros(n)
    stop = np.full(n, -S)
    cum_qv = np.zeros(n); cnt = np.zeros(n)
    run_tb = np.zeros(n); run_qv = np.zeros(n)
    for j in range(Bn):
        bad = ~np.isfinite(h[:, j])
        hit_s = open_ & ~bad & (-adv_l[:, j] <= stop)
        hit_t = open_ & ~bad & ~hit_s & (fav_h[:, j] >= T)
        net[hit_s] = stop[hit_s] - ENTRY - MKT
        net[hit_t] = T - ENTRY - TGT
        open_ &= ~(hit_s | hit_t)
        new_best = open_ & ~bad & (fav_h[:, j] > mfe)
        mfe = np.where(open_ & ~bad, np.maximum(mfe, fav_h[:, j]), mfe)
        # flow from entry to the best point so far
        run_tb = np.where(open_ & ~bad, run_tb + np.nan_to_num(tb[:, j]), run_tb)
        run_qv = np.where(open_ & ~bad, run_qv + np.nan_to_num(qv[:, j]), run_qv)
        if j == 0:
            peak_flow = np.zeros(n)
        peak_flow = np.where(new_best, np.where(run_qv > 0, (2 * run_tb / np.where(run_qv > 0, run_qv, 1) - 1) * s, 0), peak_flow)
        avg_qv = np.where(cnt > 0, cum_qv / np.maximum(cnt, 1), np.nan)
        cum_qv += np.nan_to_num(qv[:, j]); cnt += 1
        cj = cl[:, j]
        ema = ema + k * (c[:, j] - ema)
        ema_tr = s * (ema / e - 1)
        f3 = np.nanmean(flow[:, max(0, j - 2):j + 1], 1)
        f6 = np.nanmean(flow[:, max(0, j - 5):j + 1], 1)
        prev = cl[:, j - 1] if j > 0 else np.zeros(n)
        armed = mfe >= 0.005
        ex = np.zeros(n, bool)
        if rule == "BACK_0.005":
            ex = armed & (cj <= 0)
        elif rule == "EMA_0.01":
            ex = (mfe >= 0.01) & (cj < ema_tr)
        elif rule == "WICK":
            ex = armed & (wick[:, j] >= 0.6) & (rng[:, j] / e >= 0.003)
        elif rule == "FLOW15":
            ex = armed & (f3 <= -0.2) & (cj < prev)
        elif rule == "FLOW_EMA":
            ex = armed & (cj < ema_tr) & (f3 < 0)
        elif rule == "VOLSPIKE":
            against = s * (c[:, j] - o[:, j]) < 0
            ex = armed & against & (qv[:, j] >= 2 * avg_qv) & (pos_against[:, j] >= 2 / 3)
        elif rule == "FLOWFLIP":
            ex = armed & (peak_flow > 0.1) & (f6 < -0.1) & (j >= 5)
        ex &= open_ & ~bad
        net[ex] = cj[ex] - ENTRY - MKT
        open_ &= ~ex
    net[open_] = cl[open_, Bn - 1] - ENTRY - MKT
    net[~ok] = np.nan
    return net


res = {r: simulate(r) for r in RULES}


def st(v, m):
    m = m & np.isfinite(v)
    x = v[m]
    if len(x) < 30:
        return None
    d = pd.Series(x).groupby(day[m]).mean()
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    mo = pd.Series(x).groupby(month[m]).mean()
    return {"n": len(x), "win": (x > 0).mean(), "full": (x <= -S - ENTRY - MKT + 1e-9).mean(), "net": x.mean(), "t": t, "mpos": (mo > 0).mean()}


print(f"sample: {ok.sum()} grinds with full candles ({(R3 & ok).sum()} all-day steady)")
for gname, gm in (("RANDOM GRINDS (stands for all grinds)", ~R3), ("NEW RULE (all-day steady)", R3)):
    print(f"\n=== {gname} ===")
    for r in RULES:
        d, c2 = st(res[r], gm & DISC), st(res[r], gm & ~DISC)
        f = lambda s: "n<30" if s is None else f"n {s['n']:>5} won {s['win']:.1%} full stops {s['full']:.1%} {s['net']:+.3%} t {s['t']:+.1f}"
        print(f"{r:<11} {f(d):>56} | {f(c2):>56}")
