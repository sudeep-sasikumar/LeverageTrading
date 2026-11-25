"""Historical study: when a coin has been grinding steadily for the last N hours, does it keep going?

Decision points: every hour close, every liquid coin. Features use only data up to that close.
Outcomes: the following 5m path (entry at the next bar's open), with realistic costs.
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, r"D:\Leverage trading")
from lts.config import load_config
from lts.db import DB
from lts import features as F

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
cfg = load_config(r"D:\Leverage trading\config.yaml"); db = DB(cfg.path(cfg.storage.db_path))
H, D = 12, 288
COST = 2 * (0.0008 + 0.0004)                     # market in + market out, Weex VIP0 + slippage
FWD = 24 * H                                     # up to 24h of forward path

def wilder(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()

def rolling_r2(y, n):
    """R^2 of y against time over trailing windows of n points (vectorised)."""
    out = np.full(len(y), np.nan)
    t = np.arange(n, dtype=float); t -= t.mean(); st2 = (t ** 2).sum()
    from numpy.lib.stride_tricks import sliding_window_view as sw
    if len(y) < n:
        return out
    W = sw(y, n)
    yc = W - W.mean(1, keepdims=True)
    cov = (yc * t).sum(1); sy2 = (yc ** 2).sum(1)
    out[n - 1:] = np.where(sy2 > 0, cov ** 2 / (st2 * sy2), np.nan)
    return out

def barrier(side, entry, sl, tp, h, l, c, n):
    if side > 0:
        hs, ht = l[:n] <= entry * (1 - sl), h[:n] >= entry * (1 + tp)
    else:
        hs, ht = h[:n] >= entry * (1 + sl), l[:n] <= entry * (1 - tp)
    i_s = np.argmax(hs) if hs.any() else 10 ** 9
    i_t = np.argmax(ht) if ht.any() else 10 ** 9
    if i_s == i_t == 10 ** 9:
        r = side * (c[n - 1] / entry - 1)
        return "time", (r - COST) / sl
    if i_s <= i_t:
        return "sl", (-sl - COST) / sl
    return "tp", (tp - COST) / sl

t0 = time.time()
syms = [r["symbol"] for r in db.query("SELECT symbol FROM symbols WHERE active=1 AND src='binance'")]
rows = []
for sym in syms:
    k = db.klines(sym)
    if len(k) < 5 * D:
        continue
    k = F.regularize(k)
    ts5 = k.index.to_numpy(); o, h, l, c, qv = (k[x].to_numpy() for x in ("o", "h", "l", "c", "qv"))
    lc = np.log(c)
    # 15m and 1h grids aligned to bar closes
    i15 = np.arange(2, len(c), 3)
    lc15 = lc[i15]
    d15 = np.diff(lc15, prepend=np.nan)
    i1h = np.arange(11, len(c), 12)
    hh = pd.Series(h).rolling(12).max().to_numpy()[i1h]; ll = pd.Series(l).rolling(12).min().to_numpy()[i1h]
    c1h = c[i1h]; prev = np.roll(c1h, 1); prev[0] = np.nan
    tr = np.nanmax(np.vstack([hh - ll, np.abs(hh - prev), np.abs(ll - prev)]), axis=0)
    atr1h = wilder(tr, 14) / c1h
    lc1h = lc[i1h]; d1h = np.diff(lc1h, prepend=np.nan)
    qv1h = pd.Series(qv).rolling(12).sum().to_numpy()[i1h]
    qv24 = pd.Series(qv1h).rolling(24).sum().to_numpy()
    qv6 = pd.Series(qv1h).rolling(6).sum().to_numpy()
    qv6_base = pd.Series(qv1h).rolling(24 * 7, min_periods=48).mean().shift(6).to_numpy() * 6
    oi = db.frame("SELECT ts, oi FROM oi WHERE symbol=? ORDER BY ts", [sym])
    close_ts = ts5[i1h] + F.BAR_MS
    oi_now = F._asof(close_ts, oi, "oi", stale_ms=3 * 3_600_000)
    oi_6h = F._asof(close_ts - 6 * 3_600_000, oi, "oi", stale_ms=3 * 3_600_000)
    # 15m-based stats over trailing 6h (24 x 15m) mapped onto the hourly grid
    r2_15 = rolling_r2(lc15, 24)
    er_15 = np.abs(pd.Series(d15).rolling(24).sum().to_numpy()) / pd.Series(np.abs(d15)).rolling(24).sum().to_numpy()
    j15 = (i1h - 2) // 3                                     # 15m index whose close == the hour close
    r2_6, er_6 = r2_15[j15], er_15[j15]
    r2_12 = rolling_r2(lc15, 48)[j15]
    for n_h, name in ((6, "r6"), (12, "r12"), (24, "r24")):
        pass
    r6 = lc1h - np.roll(lc1h, 6); r6[:6] = np.nan
    r12 = lc1h - np.roll(lc1h, 12); r12[:12] = np.nan
    r24 = lc1h - np.roll(lc1h, 24); r24[:24] = np.nan
    sgn = np.sign(r6)
    W6 = np.lib.stride_tricks.sliding_window_view(np.nan_to_num(d1h), 6)
    with6 = np.full(len(d1h), np.nan); big6 = np.full(len(d1h), np.nan)
    with6[5:] = (np.sign(W6) == sgn[5:, None]).sum(1)
    big6[5:] = (W6 * sgn[5:, None]).max(1) / np.maximum(np.abs(r6[5:]), 1e-9)
    for j in range(24 * 3, len(i1h)):
        i = i1h[j]                                           # 5m bar index of the hour close
        if i + 1 + FWD > len(c) or not np.isfinite(qv24[j]) or qv24[j] < 10e6 or not np.isfinite(r6[j]):
            continue
        s = 1 if r6[j] > 0 else -1
        e = o[i + 1]
        fh, fl, fc = h[i + 1:i + 1 + FWD], l[i + 1:i + 1 + FWD], c[i + 1:i + 1 + FWD]
        row = dict(symbol=sym, ts=int(close_ts[j]), side=s, r6=s * r6[j], r12=s * r12[j], r24=s * r24[j],
                   r2_6=r2_6[j], er_6=er_6[j], r2_12=r2_12[j], with6=with6[j], big6=big6[j], atr1h=atr1h[j],
                   rvol6=qv6[j] / qv6_base[j] if qv6_base[j] else np.nan, qv24=qv24[j],
                   oi6=s * (oi_now[j] / oi_6h[j] - 1),
                   fwd12=s * (fc[12 * H - 1] / e - 1), fwd24=s * (fc[-1] / e - 1),
                   mfe24=(fh.max() / e - 1) if s > 0 else (1 - fl.min() / e),
                   mae24=(1 - fl.min() / e) if s > 0 else (fh.max() / e - 1))
        a = atr1h[j]
        if np.isfinite(a) and a > 0:
            for m in (1.5, 2.5):
                for rr in (1.0, 2.0):
                    sl = m * a
                    row[f"b{m}_{rr}"] = barrier(s, e, sl, rr * sl, fh, fl, fc, FWD)[1]
            # fixed-percent plan: stop 3%, target 3%
            row["fix3"] = barrier(s, e, 0.03, 0.03, fh, fl, fc, FWD)[1]
        rows.append(row)
df = pd.DataFrame(rows)
df.to_parquet(r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad\grind_points.parquet")
print(f"{len(df):,} coin-hours, {df.symbol.nunique()} coins, {(df.ts.max() - df.ts.min()) / 86_400_000:.1f} days, {time.time() - t0:.0f}s")
