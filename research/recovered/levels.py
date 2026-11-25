"""Major levels between entry and target / entry and stop, on the year's early grinds (5x +2% / -1%).

Levels (only from candles that had CLOSED before the grind was detected):
  daily   : previous day's high and low; daily swing highs/lows (highest/lowest of 3 days each side), last 180 days
  weekly  : previous week's high and low; weekly swing highs/lows (2 weeks each side), last 2 years
  monthly : previous month's high and low; monthly swing highs/lows (1 month each side), last 3 years
Both highs and lows count as levels (an old support above price acts as resistance, and the other way round).

For a long: "blocked" = a level above entry, at or below the target (+2%); "cushion" = a level below entry,
at or above the stop (-1%). Mirror image for shorts.
Pre-registered: rules judged on the first 7 months (discovery), confirmed on the last 7.
"""
import sqlite3, sys
import numpy as np, pandas as pd

OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
DAY = 86_400_000
T, S = 0.02, 0.01
ev = pd.read_parquet(OUT + r"\five_x_scores.parquet")
price = pd.read_parquet(OUT + r"\shape_events.parquet")[["symbol", "ts", "side", "price"]]
ev = ev.merge(price, on=["symbol", "ts", "side"], how="left")
conn = sqlite3.connect(r"file:D:\Leverage trading\data\lts.sqlite?mode=ro", uri=True)


def pivots(h, l, k):
    """Swing highs/lows: index i where h[i] is the max of h[i-k..i+k]. Returns (index, level) arrays."""
    n = len(h)
    hi_i = [i for i in range(k, n - k) if h[i] == h[i - k:i + k + 1].max()]
    lo_i = [i for i in range(k, n - k) if l[i] == l[i - k:i + k + 1].min()]
    return np.array(hi_i + lo_i, int), np.concatenate([h[hi_i], l[lo_i]]) if (hi_i or lo_i) else np.array([])


def frame(d, key):
    g = d.groupby(key)
    out = pd.DataFrame({"start": g["ts"].min(), "end": g["ts"].max() + DAY, "h": g["h"].max(), "l": g["l"].min()})
    return out.sort_values("start").reset_index(drop=True)


TF = {"daily": (3, 180), "weekly": (2, 104), "monthly": (1, 36)}
rows = []
for sym, e in ev.groupby("symbol"):
    d = pd.read_sql("SELECT ts,h,l FROM klines_1d WHERE symbol=? ORDER BY ts", conn, params=[sym])
    if len(d) < 10:
        continue
    d["week"] = (d.ts // DAY + 3) // 7                         # weeks start Monday 00:00 UTC
    d["month"] = pd.to_datetime(d.ts, unit="ms").dt.strftime("%Y-%m")
    bars = {"daily": d.assign(start=d.ts, end=d.ts + DAY)[["start", "end", "h", "l"]].reset_index(drop=True),
            "weekly": frame(d, "week"), "monthly": frame(d, "month")}
    piv = {}
    for tf, (k, look) in TF.items():
        b = bars[tf]
        idx, lev = pivots(b.h.to_numpy(), b.l.to_numpy(), k)
        # a swing is known once the k bars after it have closed
        known = b.end.to_numpy()[np.minimum(idx + k, len(b) - 1)] if len(idx) else np.array([])
        piv[tf] = (idx, lev, known)
    for r in e.itertuples():
        t, p, sd = r.ts, r.price, r.side
        if not np.isfinite(p):
            continue
        rec = {"symbol": sym, "ts": t, "side": sd}
        for tf, (k, look) in TF.items():
            b = bars[tf]
            closed = b[b.end <= t]
            if not len(closed):
                continue
            first_ok = closed.start.iloc[max(0, len(closed) - look)]
            idx, lev, known = piv[tf]
            m = (known <= t) & (b.start.to_numpy()[idx] >= first_ok) if len(idx) else np.array([], bool)
            levels = np.concatenate([lev[m] if len(idx) else [], [closed.h.iloc[-1], closed.l.iloc[-1]]])
            if sd > 0:
                rec[f"block_{tf}"] = bool(((levels > p) & (levels <= p * (1 + T))).any())
                rec[f"cush_{tf}"] = bool(((levels < p) & (levels >= p * (1 - S))).any())
            else:
                rec[f"block_{tf}"] = bool(((levels < p) & (levels >= p * (1 - T))).any())
                rec[f"cush_{tf}"] = bool(((levels > p) & (levels <= p * (1 + S))).any())
        rows.append(rec)
lv = pd.DataFrame(rows)
ev = ev.merge(lv, on=["symbol", "ts", "side"], how="inner")
ev["month"] = pd.to_datetime(ev.ts, unit="ms").dt.strftime("%Y-%m")
ev["day"] = ev.ts // DAY
months = sorted(ev.month.unique())
ev["part"] = np.where(ev.month.isin(months[:len(months) // 2]), "first 7 months", "last 7 months")
for tf in TF:
    ev[f"block_{tf}"] = ev[f"block_{tf}"].fillna(False).astype(bool)
    ev[f"cush_{tf}"] = ev[f"cush_{tf}"].fillna(False).astype(bool)
ev["block_any"] = ev[[f"block_{t}" for t in TF]].any(axis=1)
ev["block_wm"] = ev.block_weekly | ev.block_monthly
ev["cush_any"] = ev[[f"cush_{t}" for t in TF]].any(axis=1)
ev["cush_wm"] = ev.cush_weekly | ev.cush_monthly
ev.to_parquet(OUT + r"\levels_events.parquet")
BE = 0.0124 / (0.0180 + 0.0124)
print(f"grinds {len(ev):,}; 5x +2/-1 needs about {BE:.1%} wins to break even (taker fee both ways)")


def line(name, m):
    out = []
    for part in ("first 7 months", "last 7 months"):
        x = ev[m & (ev.part == part)]
        if len(x) < 50:
            out.append(f"{part}: n<50")
            continue
        mo = x.groupby("month").net.mean()
        out.append(f"{part}: n {len(x):>6,} won {x.win.mean():.1%} {x.net.mean():+.3%} (months+ {(mo > 0).mean():.0%})")
    print(f"  {name:<44} " + " | ".join(out))


print("\n=== share of grinds with a level in the way (all months) ===")
for c in ["block_daily", "block_weekly", "block_monthly", "block_any", "cush_daily", "cush_weekly", "cush_monthly", "cush_any"]:
    print(f"  {c:<16} {ev[c].mean():.0%}")
allm = np.ones(len(ev), bool)
print("\n=== resistance (long) / support (short) between entry and the +2% target ===")
line("all grinds", allm)
for tf in ["daily", "weekly", "monthly", "wm", "any"]:
    line(f"no {tf} level in the way", ~ev[f"block_{tf}"])
    line(f"   {tf} level in the way", ev[f"block_{tf}"])
print("\n=== support (long) / resistance (short) between entry and the -1% stop ===")
for tf in ["daily", "weekly", "monthly", "wm", "any"]:
    line(f"{tf} level protecting the stop", ev[f"cush_{tf}"])
    line(f"   no {tf} level protecting the stop", ~ev[f"cush_{tf}"])
print("\n=== combined: clear path to target AND a level protecting the stop ===")
for a, b in [("any", "any"), ("wm", "any"), ("any", "wm"), ("wm", "wm")]:
    line(f"clear of {a} + {b} level protecting", ~ev[f"block_{a}"] & ev[f"cush_{b}"])
print("\n=== with the live model (top 20% by score, scored on unseen months only) ===")
p = ev["p_minute shape (today)"]
ok = p.notna()
thr = p[ok].quantile(0.8)
top = ok & (p >= thr)
line("top 20%", top)
line("top 20% + clear of any level", top & ~ev.block_any)
line("top 20% + clear of weekly/monthly", top & ~ev.block_wm)
line("top 20% + clear of any + level protecting stop", top & ~ev.block_any & ev.cush_any)
for sd, nm in ((1, "longs"), (-1, "shorts")):
    line(f"{nm}: clear of any level", (ev.side == sd) & ~ev.block_any)
    line(f"{nm}: level in the way", (ev.side == sd) & ev.block_any)
