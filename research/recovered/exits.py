"""Early exits: can a winner-turning-loser be spotted and closed before the -1% stop?

Plan: 5x, enter at detection, +2% target, -1% stop, 24h. On top, one exit rule (decided on a 5m candle's
close, exit at that close; taker fee + 0.04% slippage). Stop/target are checked inside each candle first
(stop first if both). Rules written down before testing:
  BE_a     once the best move reached a, the stop moves to entry (from the next candle)
  LOCK     once +1.5%, the stop moves to +0.5% (from the next candle)
  BACK_a   once the best move reached a, exit on a close back at or below entry
  EMA_a    once the best move reached a, exit on a close through the 5m EMA20 (against the trade)
  SWING    once +0.5%, exit on a close through the last swing low of the run (2 candles each side; high for shorts)
  WICK     once +0.5%, exit on a candle whose wick against us is >= 60% of its range and range >= 0.3%
  GIVE_g   once +1%, exit on a close that has given back g from the best point
  TIME_m   exit at the close m minutes after entry unless the best move has reached +0.5%
Groups: all grinds; the new rule (all-day steady, R3); the old rule (model top 20%, unseen months only).
Split: first 7 months (discovery) / last 7 months (confirmation).
"""
import glob
import numpy as np, pandas as pd

OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
DAY = 86_400_000
T, S = 0.02, 0.01
ENTRY, MKT = 0.0008 + 0.0004, 0.0008 + 0.0004          # taker + slippage in; market exit out
TGT = 0.0008                                           # Weex TP fills as taker

a = pd.read_parquet(OUT + r"\aim_events.parquet").reset_index().rename(columns={"index": "row"})
sc = pd.read_parquet(OUT + r"\five_x_scores.parquet")[["symbol", "ts", "side", "p_minute shape (today)"]]
a = a.merge(sc, on=["symbol", "ts", "side"], how="left")
R3 = ((a.r_24h >= 0.08) & (a.r2_24h >= 0.8) & (a.hours_with24 >= 15) & (a.r2_6h >= 0.9) & (a.hours_with6 >= 5)
      & (a.big6 <= 0.35)).to_numpy()
p = a["p_minute shape (today)"]
OLD = (p.notna() & (p >= p[p.notna()].quantile(0.8))).to_numpy()
month = pd.to_datetime(a.ts, unit="ms").dt.strftime("%Y-%m")
months = sorted(month.unique())
DISC = month.isin(months[:len(months) // 2]).to_numpy()
day = (a.ts // DAY).to_numpy()

RULES = ["BASE"] + [f"BE_{x}" for x in (0.005, 0.01)] + ["LOCK"] + [f"BACK_{x}" for x in (0.003, 0.005, 0.01)] + \
        [f"EMA_{x}" for x in (0.005, 0.01)] + ["SWING", "WICK"] + [f"GIVE_{x}" for x in (0.0075, 0.01)] + \
        [f"TIME_{x}" for x in (30, 60)]


def simulate(post, pre_c, side, rule):
    """post (n,289,4) o,h,l,c; pre_c (n,72) closes before entry. Returns net % per event."""
    n, B = post.shape[0], 288
    e = post[:, 0, 0].astype(np.float64)
    s = side.astype(np.float64)
    o, h, l, c = (post[:, :B, k].astype(np.float64) for k in range(4))
    fav_h = np.where(s[:, None] > 0, h / e[:, None] - 1, 1 - l / e[:, None])     # best point in the candle
    adv_l = np.where(s[:, None] > 0, 1 - l / e[:, None], h / e[:, None] - 1)     # worst point in the candle
    cl = s[:, None] * (c / e[:, None] - 1)                                        # close, trade direction
    op = s[:, None] * (o / e[:, None] - 1)
    rngp = (h - l) / e[:, None]
    wick = np.where(s[:, None] > 0, h - np.maximum(o, c), np.minimum(o, c) - l) / np.where(h > l, h - l, np.nan)
    # 5m EMA20 seeded on the 72 closes before entry
    k = 2 / 21
    ema = np.full(n, np.nan)
    for j in range(pre_c.shape[1]):
        x = pre_c[:, j].astype(np.float64)
        ema = np.where(np.isnan(ema), x, ema + k * (x - ema))
    net = np.full(n, np.nan)
    open_ = np.isfinite(e)
    mfe = np.zeros(n)
    stop = np.full(n, -S)                       # stop level in trade-direction return terms
    swing = np.full(n, np.nan)                  # last confirmed swing low (trade direction) since entry
    for j in range(B):
        if not open_.any():
            break
        bad = ~np.isfinite(h[:, j])
        # 1) stop / target inside the candle (stop first)
        hit_s = open_ & ~bad & (-adv_l[:, j] <= stop)
        hit_t = open_ & ~bad & ~hit_s & (fav_h[:, j] >= T)
        net[hit_s] = np.where(stop[hit_s] <= -S + 1e-12, -S, stop[hit_s]) - ENTRY - MKT
        net[hit_t] = T - ENTRY - TGT
        open_ &= ~(hit_s | hit_t)
        # 2) rule on the close
        mfe = np.where(open_ & ~bad, np.maximum(mfe, fav_h[:, j]), mfe)
        cj = cl[:, j]
        ema = ema + k * (c[:, j] - ema)
        ema_tr = s * (ema / e - 1)
        ex = np.zeros(n, bool)
        if rule.startswith("BACK_"):
            ex = (mfe >= float(rule[5:])) & (cj <= 0)
        elif rule.startswith("EMA_"):
            ex = (mfe >= float(rule[4:])) & (cj < ema_tr)
        elif rule == "SWING" and j >= 4:
            lo3 = np.where(s > 0, l[:, j - 2] / e - 1, 1 - h[:, j - 2] / e)        # candle j-2 in trade terms (its low)
            nb = np.stack([np.where(s > 0, l[:, i] / e - 1, 1 - h[:, i] / e) for i in range(j - 4, j + 1)], 1)
            is_sw = lo3 <= np.nanmin(nb, 1)
            swing = np.where(is_sw, lo3, swing)
            ex = (mfe >= 0.005) & np.isfinite(swing) & (cj < swing)
        elif rule == "WICK":
            ex = (mfe >= 0.005) & (wick[:, j] >= 0.6) & (rngp[:, j] >= 0.003)
        elif rule.startswith("GIVE_"):
            ex = (mfe >= 0.01) & (cj <= mfe - float(rule[5:]))
        elif rule.startswith("TIME_"):
            ex = ((j + 1) * 5 >= int(rule[5:])) & (mfe < 0.005)
            ex &= ((j + 1) * 5 == int(rule[5:]))
        ex &= open_ & ~bad
        net[ex] = cj[ex] - ENTRY - MKT
        open_ &= ~ex
        # 3) stop moves (take effect from the next candle)
        if rule.startswith("BE_"):
            stop = np.where(mfe >= float(rule[3:]), np.maximum(stop, 0.0), stop)
        elif rule == "LOCK":
            stop = np.where(mfe >= 0.015, np.maximum(stop, 0.005), stop)
    last = cl[:, B - 1]
    net[open_] = last[open_] - ENTRY - MKT        # 24h time exit
    return net


files = sorted(glob.glob(OUT + r"\paths\chunk_*.npy"))
pfiles = sorted(glob.glob(OUT + r"\pre\chunk_*.npy"))
res = {r: np.full(len(a), np.nan) for r in RULES}
start = 0
side = a.side.to_numpy()
for fp, fq in zip(files, pfiles):
    post = np.load(fp)
    pre = np.load(fq)[:, :, 3]
    idx = np.arange(start, start + len(post))
    ok = idx < len(a)
    for r in RULES:
        res[r][idx[ok]] = simulate(post[ok], pre[ok], side[idx[ok]], r)
    start += len(post)
    print(".", end="", flush=True)
print()


def st(v, m):
    x = v[m & np.isfinite(v)]
    if len(x) < 30:
        return None
    d = pd.Series(x).groupby(day[m & np.isfinite(v)]).mean()
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan
    mo = pd.Series(x).groupby(month.to_numpy()[m & np.isfinite(v)]).mean()
    return {"n": len(x), "win": (x > 0).mean(), "loss_full": (x <= -S - ENTRY - MKT + 1e-9).mean(), "net": x.mean(), "t": t, "mpos": (mo > 0).mean()}


for gname, gm in (("ALL GRINDS", np.ones(len(a), bool)), ("NEW RULE (all-day steady)", R3), ("OLD RULE (model top 20%, unseen months)", OLD)):
    print(f"\n=== {gname} ===")
    print(f"{'exit rule':<12} {'first 7 months':>58} | {'last 7 months':>58}")
    for r in RULES:
        d, c2 = st(res[r], gm & DISC), st(res[r], gm & ~DISC)
        f = lambda s: "n<30" if s is None else f"n {s['n']:>6} won {s['win']:.1%} full stops {s['loss_full']:.1%} {s['net']:+.3%} t {s['t']:+.1f} m+ {s['mpos']:.0%}"
        print(f"{r:<12} {f(d):>58} | {f(c2):>58}")
pd.DataFrame({r: res[r] for r in RULES}).assign(symbol=a.symbol, ts=a.ts, side=a.side, R3=R3, OLD=OLD, DISC=DISC) \
    .to_parquet(OUT + r"\exits_results.parquet")
