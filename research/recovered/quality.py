"""Strict test of 'quality grind' rules for the 5x +2% / -1% plan (taker fees both ways; break-even ~40.8%).

Pre-registered rules (written before looking at their results):
  R1 all-day grinder     side*r_24h >= 8%, R^2 of the 24h path >= 0.8, >= 15 of 24 hours with the move
  R2 very steady 6h      R^2 of the 6h path >= 0.9, >= 5 of 6 hours with the move, no hour > 35% of the move
  R3 R1 and R2
  R4 with the market     the median coin and BTC both moved the trade's way over the last 6h
  R5 own move            the median coin moved AGAINST the trade's way over the last 6h
  R6 not chasing         last 1h move <= 1.5% and the last 5m candle <= 0.4% in the trade's direction
  R7 quiet candles       5m candle range (median, last 2h) <= 0.25% (a 1% stop is >= 4 typical candles away)
  R8 first call per coin one per coin per 24h (the first detection)
  R9 R1 + R6 + R7        (the user's 'actually grinds' picture: all-day, calm, not a spike)
  R10 R3 + R7 + R8
Split: first 7 months (discovery) / last 7 months (confirmation). A rule passes only if, in the confirmation half,
it is positive after fees, day-clustered t >= 2, and positive in >= 60% of months.
"""
import glob
import numpy as np, pandas as pd

OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
DAY = 86_400_000
a = pd.read_parquet(OUT + r"\aim_events.parquet").reset_index().rename(columns={"index": "row"})
sc = pd.read_parquet(OUT + r"\five_x_scores.parquet")[["symbol", "ts", "side", "net", "win", "p_minute shape (today)"]]
ev = a.merge(sc, on=["symbol", "ts", "side"], how="inner")
pre = np.concatenate([np.load(f, mmap_mode="r") for f in sorted(glob.glob(OUT + r"\pre\chunk_*.npy"))])
P = pre[ev.row.to_numpy()].astype(np.float64)                       # (n, 72, 6) o,h,l,c,qv,tbqv before entry
o, h, l, c = P[:, :, 0], P[:, :, 1], P[:, :, 2], P[:, :, 3]
sd = ev.side.to_numpy()
last_c = c[:, -1]
r1h = sd * (last_c / c[:, -13] - 1)
last5 = sd * (c[:, -1] / o[:, -1] - 1)
rng2h = np.nanmedian((h[:, -24:] - l[:, -24:]) / c[:, -24:], axis=1)
ev["r1h_5m"], ev["last5"], ev["rng2h"] = r1h, last5, rng2h
ev["month"] = pd.to_datetime(ev.ts, unit="ms").dt.strftime("%Y-%m")
ev["day"] = ev.ts // DAY
months = sorted(ev.month.unique())
ev["half"] = np.where(ev.month.isin(months[:len(months) // 2]), "disc", "conf")
ev = ev.sort_values("ts").reset_index(drop=True)
first = np.zeros(len(ev), bool)
last_seen = {}
for i, (s_, t_) in enumerate(zip(ev.symbol, ev.ts)):
    if t_ - last_seen.get(s_, -10 ** 15) >= DAY:
        first[i] = True
        last_seen[s_] = t_
ev["first24"] = first

R = {}
R["R1 all-day grinder"] = (ev.r_24h >= 0.08) & (ev.r2_24h >= 0.8) & (ev.hours_with24 >= 15)
R["R2 very steady 6h"] = (ev.r2_6h >= 0.9) & (ev.hours_with6 >= 5) & (ev.big6 <= 0.35)
R["R3 R1+R2"] = R["R1 all-day grinder"] & R["R2 very steady 6h"]
R["R4 with the market"] = (ev.mkt_r6 > 0) & (ev.btc_r6 > 0)
R["R5 own move (market against)"] = ev.mkt_r6 < 0
R["R6 not chasing"] = (ev.r1h_5m <= 0.015) & (ev.last5 <= 0.004)
R["R7 quiet candles"] = ev.rng2h <= 0.0025
R["R8 first call per coin/24h"] = ev.first24
R["R9 R1+R6+R7"] = R["R1 all-day grinder"] & R["R6 not chasing"] & R["R7 quiet candles"]
R["R10 R3+R7+R8"] = R["R3 R1+R2"] & R["R7 quiet candles"] & R["R8 first call per coin/24h"]


def st(m, half):
    x = ev[m & (ev.half == half)]
    if len(x) < 40:
        return None
    dd = x.groupby("day").net.mean()
    mo = x.groupby("month").net.mean()
    t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 2 else np.nan
    days = ev.loc[ev.half == half, "day"].nunique()
    return {"n": len(x), "per_day": len(x) / days, "win": x.win.mean(), "net": x.net.mean(), "t": t, "mpos": (mo > 0).mean()}


print(f"grinds {len(ev):,}  (break-even ~40.8% wins)")
allm = pd.Series(True, index=ev.index)
rows = [("All grinds", allm)] + list(R.items())
print(f"{'rule':<32} {'first 7 months':>44} | {'last 7 months':>52}")
for name, m in rows:
    d, c2 = st(m, "disc"), st(m, "conf")
    f = lambda s: "n<40" if s is None else f"{s['per_day']:5.1f}/day won {s['win']:.1%} {s['net']:+.3%} t {s['t']:+.1f} m+ {s['mpos']:.0%}"
    ok = c2 is not None and c2["net"] > 0 and c2["t"] >= 2 and c2["mpos"] >= 0.6
    print(f"{name:<32} {f(d):>44} | {f(c2):>44}  {'PASS' if ok else 'fail'}")
ev[["symbol", "ts", "side", "net", "win", "half", "r_24h", "r2_24h", "hours_with24", "r2_6h", "hours_with6", "big6", "mkt_r6",
    "btc_r6", "r1h_5m", "last5", "rng2h", "first24"]].to_parquet(OUT + r"\quality_events.parquet")
# volatility buckets (does a 1% stop inside the noise lose more?)
print("\n5m candle range (median, last 2h) -> 5x result, both halves")
ev["rb"] = pd.qcut(ev.rng2h, 5, duplicates="drop")
for b, x in ev.groupby("rb", observed=True):
    s1, s2 = x[x.half == "disc"], x[x.half == "conf"]
    print(f"  {str(b):<22} first: won {s1.win.mean():.1%} {s1.net.mean():+.3%} | last: won {s2.win.mean():.1%} {s2.net.mean():+.3%}")
