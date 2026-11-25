"""Today's grinders vs the early-grind detector, on Binance 5-minute candles.

For every coin: the detector's rule evaluated every 5 minutes on rolling 1-hour steps
(r6 >= 3%, R^2 of the 6h path >= 0.7, >= 4 of 6 hours with the move, no hour > 50% of it, $5M 24h volume).
Then: when it first fired, how much of the move was left, and what each plan did from there.
Also the honest base rate: every detection today, winners and losers alike.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\Leverage trading")
from lts.features import strided_r2

D = sys.argv[1]
k = pd.read_parquet(D + "/k5.parquet").sort_values(["symbol", "ts"])
END = int(k.ts.max()) + 300_000
DAY_START = int(pd.Timestamp("2026-11-17 18:30", tz="UTC").timestamp() * 1000)   # 18 Nov 00:00 IST
NAMED = {"PIPPINUSDT": 1, "PIEVERSEUSDT": 1, "PHAROSUSDT": 1, "PROSUSDT": 1,
         "SKYAIUSDT": -1, "AIOTUSDT": -1, "BREVUSDT": -1, "SYRUPUSDT": -1}
PLANS = [(10, 0.01, 0.005), (5, 0.02, 0.01), (3, 0.0333, 0.01), (2, 0.05, 0.0075)]
TAKER, MAKER, SLIP = 0.0008, 0.0002, 0.0004


def detect(g: pd.DataFrame, side: int) -> np.ndarray:
    c = g.c.to_numpy(float)
    y = np.log(c)
    r6 = (c / pd.Series(c).shift(72).to_numpy() - 1) * side
    r2 = strided_r2(y, 7, 12)
    d = (y - pd.Series(y).shift(12).to_numpy()) * side
    lags = np.stack([pd.Series(d).shift(12 * j).to_numpy() for j in range(6)])
    hours = (lags > 0).sum(0)
    big = np.nanmax(lags, 0) / np.abs(np.log1p(r6 * side))
    qv24 = g.qv.rolling(288).sum().to_numpy()
    with np.errstate(invalid="ignore"):
        return (r6 >= 0.03) & (r2 >= 0.7) & (hours >= 4) & (big <= 0.5) & (qv24 >= 5e6)


def plan(post: np.ndarray, side: int, t: float, s: float):
    """post: (n,4) o,h,l,c from entry. Returns (net, minutes) or (None, None) while open."""
    e = post[0, 0]
    fav = post[:, 1] / e - 1 if side > 0 else 1 - post[:, 2] / e
    adv = 1 - post[:, 2] / e if side > 0 else post[:, 1] / e - 1
    hs, ht = np.flatnonzero(adv >= s), np.flatnonzero(fav >= t)
    i_s, i_t = (hs[0] if len(hs) else 10 ** 9), (ht[0] if len(ht) else 10 ** 9)
    if i_s <= i_t and i_s < 10 ** 9:
        return -s - 2 * TAKER - 2 * SLIP, (i_s + 1) * 5
    if i_t < 10 ** 9:
        return t - TAKER - SLIP - MAKER, (i_t + 1) * 5
    return None, None


rows, dets = [], []
for sym, g in k.groupby("symbol"):
    g = g.reset_index(drop=True)
    if len(g) < 400:
        continue
    ts = g.ts.to_numpy()
    day = ts >= DAY_START
    if day.sum() < 24:
        continue
    o_day = g.o[day].iloc[0]
    for side in (1, -1):
        m = detect(g, side)
        # every detection today (new episode = no detection in the previous 60 minutes)
        idx = np.flatnonzero(m & day)
        last = -10 ** 9
        for i in idx:
            if i - last > 12 and i + 1 < len(g):
                post = g[["o", "h", "l", "c"]].to_numpy(float)[i + 1:i + 1 + 288]
                res = {f"{L}x": plan(post, side, t, s)[0] for L, t, s in PLANS}
                e = post[0, 0]
                fav = (post[:, 1] / e - 1) if side > 0 else (1 - post[:, 2] / e)
                adv = (1 - post[:, 2] / e) if side > 0 else (post[:, 1] / e - 1)
                dets.append({"symbol": sym, "side": side, "ts": int(ts[i]) + 300_000, "hour_aligned": (ts[i] + 300_000) % 3_600_000 == 0,
                             "best": fav.max(), "worst": adv.max(), "hrs_seen": len(post) / 12, **res})
            last = i if i - last > 12 else last
        # the day's move on this side
        cc = g.c[day].to_numpy(float)
        ext = cc.max() if side > 0 else cc.min()
        move = side * (ext / o_day - 1)
        i_ext = int(np.flatnonzero(day)[0] + (np.argmax(cc) if side > 0 else np.argmin(cc)))
        seg = np.log(g.c.to_numpy(float)[np.flatnonzero(day)[0]:i_ext + 1])
        r2 = np.corrcoef(np.arange(len(seg)), seg)[0, 1] ** 2 if len(seg) > 24 else np.nan
        first = np.flatnonzero(m & day)
        first_h = [i for i in first if (ts[i] + 300_000) % 3_600_000 == 0]
        rec = {"symbol": sym, "side": side, "move": move, "r2": r2, "hours_to_ext": (i_ext - np.flatnonzero(day)[0]) / 12,
               "named": NAMED.get(sym) == side}
        if len(first):
            i = first[0]
            p0 = g.c[i]
            rec.update({"det_utc": pd.Timestamp(int(ts[i]) + 300_000, unit="ms").strftime("%H:%M"),
                        "moved_before": side * (p0 / o_day - 1), "left_after": side * (ext / p0 - 1),
                        "det_hourly_utc": pd.Timestamp(int(ts[first_h[0]]) + 300_000, unit="ms").strftime("%H:%M") if first_h else None})
            post = g[["o", "h", "l", "c"]].to_numpy(float)[i + 1:i + 1 + 288]
            for L, t, s in PLANS:
                net, mins = plan(post, side, t, s)
                rec[f"{L}x"] = net
                rec[f"{L}x_min"] = mins
        rows.append(rec)

r = pd.DataFrame(rows)
dets = pd.DataFrame(dets)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
grinders = r[(r.move >= 0.08) & (r.r2 >= 0.6)].sort_values("move", ascending=False)
print(f"window {pd.Timestamp(DAY_START, unit='ms')} -> {pd.Timestamp(END, unit='ms')} UTC; coins {k.symbol.nunique()}")
print("\n=== named coins ===")
cols = ["symbol", "side", "move", "r2", "hours_to_ext", "det_utc", "det_hourly_utc", "moved_before", "left_after", "10x", "10x_min", "5x", "5x_min", "3x", "2x"]
print(r[r.named][[c for c in cols if c in r]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\n=== steady grinders today (move >= 8%, R^2 >= 0.6): {len(grinders)} ===")
print(grinders[[c for c in cols if c in grinders]].head(60).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
det = grinders[grinders.det_utc.notna()]
print(f"\ndetected: {len(det)}/{len(grinders)}; median moved before detection {det.moved_before.median():.1%}, "
      f"median left after {det.left_after.median():.1%}")
for L in ("10x", "5x", "3x", "2x"):
    v = det[L]
    print(f"  {L} from first detection: won {(v > 0).sum()}/{v.notna().sum()}, still open {v.isna().sum()}")
print(f"\n=== every detection today (new episodes): {len(dets)} ===")
done = dets
for L in ("10x", "5x", "3x", "2x"):
    v = done[L].dropna()
    print(f"  {L}: decided {len(v)}, won {(v > 0).mean():.1%}, avg {v.mean() * 100:+.3f}%/trade")
print(f"  median best {dets.best.median():.2%}, median worst {dets.worst.median():.2%}")
print("  longs", (dets.side > 0).sum(), "shorts", (dets.side < 0).sum())
for sd, name in ((1, "longs"), (-1, "shorts")):
    v = dets.loc[dets.side == sd, "5x"].dropna()
    print(f"  5x {name}: won {(v > 0).mean():.1%} of {len(v)}")
dets.to_parquet(D + "/detections.parquet")
r.to_parquet(D + "/coins.parquet")
