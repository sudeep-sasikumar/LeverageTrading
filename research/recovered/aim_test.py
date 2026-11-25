"""The aim test: one grind trade a day, +1-2% target. Discovery selects (fixed grid), last 120 days confirms once."""
import sys, glob, math, time
sys.path.insert(0, r"D:\Leverage trading")
import numpy as np, pandas as pd
from lts import aim as A
from lts import grindlab as G
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
DAY = A.DAY
ev = pd.read_parquet(OUT + r"\aim_events.parquet").reset_index(drop=True)
paths = np.concatenate([np.load(f) for f in sorted(glob.glob(OUT + r"\paths\chunk_*.npy"))])[:len(ev)]
ev = ev.iloc[:len(paths)].copy()
ok = np.isfinite(paths[:, :, :]).all(axis=(1, 2)) | np.isfinite(paths[:, :96, :]).all(axis=(1, 2))
print(f"events {len(ev):,}, with usable 5m paths {ok.sum():,}")
ev, paths = ev[ok].reset_index(drop=True), paths[ok]
order = np.argsort(ev.ts.to_numpy(), kind="stable")
ev, paths = ev.iloc[order].reset_index(drop=True), paths[order]
ts, side = ev.ts.to_numpy(), ev.side.to_numpy()
split = ts.max() - 120 * DAY
disc, conf = ts < split, ts >= split
days_disc, days_conf = (split - ts.min()) / DAY, (ts.max() - split) / DAY
outs = A.outcomes(paths, side)
print(f"discovery {days_disc:.0f} days ({disc.sum():,} candidates), confirmation {days_conf:.0f} days ({conf.sum():,})")

# ---- unconditional base rates: how often does an early grind reach +T before -S? (all candidates)
print("\nAll early grinds (no selection): share reaching target before stop within 24h, discovery")
rows = []
for T in A.TARGETS:
    r = {"target": f"+{T:.1%}"}
    for S in A.STOPS:
        net, _ = outs[(T, S, 24)]
        x = net[disc]
        r[f"stop {S:.1%}"] = f"{np.nanmean(x > 0):.0%} / {np.nanmean(x) * 100:+.2f}%"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False), "   (win rate / avg net % of price per trade)")

# ---- ranking scores (all decided at the moment, using only discovery data to fit anything)
train = ev[disc]
thr_ck = G.learn_thresholds(train)
score = {"first": np.zeros(len(ev)), "catalyst": ev.get("catalyst_short", pd.Series(0, index=ev.index)).fillna(0).to_numpy(),
         "checklist": G.checklist(ev, thr_ck).sum(axis=1).to_numpy().astype(float),
         "smooth": (ev.r2_6h * ev.hours_with6).to_numpy()}
feats = G.usable(train, G.model_features(train))
from sklearn.ensemble import HistGradientBoostingClassifier
model_scores = {}

def fit(X, y):
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=80,
                                       l2_regularization=5.0, random_state=7)
    return m.fit(X, y)

for T in A.TARGETS:
    for S in A.STOPS:
        net, _ = outs[(T, S, 24)]
        y = (net > 0).astype(int)
        sc = np.full(len(ev), np.nan)
        # discovery: out-of-fold scores (3 expanding time folds, purged 1 day)
        d_idx = np.flatnonzero(disc)
        edges = np.quantile(ts[d_idx], [0.4, 0.6, 0.8, 1.0])
        lo_t = ts[d_idx].min()
        for a, b in zip(edges[:-1], edges[1:]):
            tr = d_idx[ts[d_idx] < a - DAY]
            te = d_idx[(ts[d_idx] >= a) & (ts[d_idx] <= b)]
            if len(tr) > 2000 and len(te):
                m = fit(ev.loc[tr, feats].to_numpy(float), y[tr])
                sc[te] = m.predict_proba(ev.loc[te, feats].to_numpy(float))[:, 1]
        # confirmation: model trained on ALL discovery days only
        m = fit(ev.loc[d_idx, feats].to_numpy(float), y[d_idx])
        sc[conf] = m.predict_proba(ev.loc[conf, feats].to_numpy(float))[:, 1]
        model_scores[(T, S)] = sc

# ---- grid on discovery
res = []
for (T, S, H), (net, xbar) in outs.items():
    for rk in A.RANKERS:
        sc = model_scores[(T, S)] if rk == "model" else score[rk]
        qs = (1.0,) if rk == "catalyst" else ((0.0,) if rk == "first" else A.QUANTILES)
        for q in qs:
            sd = sc[disc]
            thr = 0.5 if rk == "catalyst" else (np.nanquantile(sd[np.isfinite(sd)], q) if q > 0 else -np.inf)
            if rk == "model":
                use = disc & np.isfinite(sc)
            else:
                use = disc
            idx = np.flatnonzero(use)
            taken = idx[A.daily_policy(ts[idx], sc[idx], thr, net[idx], xbar[idx])]
            s = A.summarize(ts[taken], net[taken], days_disc)
            res.append({"target": T, "stop": S, "hours": H, "ranker": rk, "q": q, "thr": thr, **s})
R = pd.DataFrame(res)
R.to_parquet(OUT + r"\aim_grid.parquet")
R["daily_pct"] = R.mean_pct * R.per_day
print(f"\nGrid on discovery: {len(R)} combinations")
cols = ["target", "stop", "hours", "ranker", "q", "n", "per_day", "win", "mean_pct", "t", "months_pos", "margin_5x", "margin_10x", "worst_streak", "dd_5x_20pct_margin"]
print("Top 15 by t-stat on discovery:")
print(R[R.n >= 60].sort_values("t", ascending=False)[cols].head(15).round(4).to_string(index=False))
sel = R[(R.t >= 3) & (R.months_pos >= 0.6) & (R.per_day >= 0.5)].sort_values("daily_pct", ascending=False)
print(f"\nPASSED DISCOVERY RULES (t>=3, >=60% months positive, >=1 trade every 2 days): {len(sel)}")
if len(sel):
    print(sel[cols].head(10).round(4).to_string(index=False))
    best = sel.iloc[0]
    T, S, H, rk, q, thr = best.target, best.stop, int(best.hours), best.ranker, best.q, best.thr
    net, xbar = outs[(T, S, H)]
    sc = model_scores[(T, S)] if rk == "model" else score[rk]
    idx = np.flatnonzero(conf)
    taken = idx[A.daily_policy(ts[idx], sc[idx], thr, net[idx], xbar[idx])]
    c = A.summarize(ts[taken], net[taken], days_conf)
    print(f"\nCONFIRMATION (frozen: target +{T:.1%}, stop -{S:.1%}, {H}h, ranker={rk}, threshold q={q}):")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
    conf_rows = []
    for _, b in sel.head(10).iterrows():
        net, xbar = outs[(b.target, b.stop, int(b.hours))]
        sc = model_scores[(b.target, b.stop)] if b.ranker == "model" else score[b.ranker]
        tk = idx[A.daily_policy(ts[idx], sc[idx], b.thr, net[idx], xbar[idx])]
        cc = A.summarize(ts[tk], net[tk], days_conf)
        conf_rows.append({"target": b.target, "stop": b.stop, "hours": b.hours, "ranker": b.ranker, "q": b.q,
                          "disc_mean": b.mean_pct, "disc_t": b.t, **{f"conf_{k}": v for k, v in cc.items()}})
    print("\n(for transparency) the other discovery passes on the confirmation months:")
    print(pd.DataFrame(conf_rows).round(4).to_string(index=False))
