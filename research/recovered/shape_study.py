"""Can pattern recognition separate winning grinds from losers? Monthly walk-forward over the year."""
import sys, glob, math, time
sys.path.insert(0, r"D:\Leverage trading")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from lts import aim as A, grindlab as G
from lts.grindshape import shape_matrix
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
DAY = A.DAY
t0 = time.time()
ev = pd.read_parquet(OUT + r"\aim_events.parquet").reset_index(drop=True)
post = np.concatenate([np.load(f) for f in sorted(glob.glob(OUT + r"\paths\chunk_*.npy"))])[:len(ev)]
pre = np.concatenate([np.load(f) for f in sorted(glob.glob(OUT + r"\pre\chunk_*.npy"))])[:len(ev)]
n = min(len(ev), len(post), len(pre))
ev, post, pre = ev.iloc[:n].copy(), post[:n], pre[:n]
outs = A.outcomes(post, ev.side.to_numpy())
for (T, S, H) in [(0.01, 0.01, 24), (0.015, 0.015, 24), (0.01, 0.02, 24)]:
    net, _ = outs[(T, S, H)]
    ev[f"net_{T:g}_{S:g}"] = net
sh = shape_matrix(pre, ev.side.to_numpy())
print(f"{n:,} events, shape features {sh.shape[1]} ({time.time()-t0:.0f}s)")
ev = pd.concat([ev.reset_index(drop=True), sh], axis=1)
ev = ev[np.isfinite(ev["net_0.01_0.01"]) & ev["sh_ret_6h"].notna()].sort_values("ts").reset_index(drop=True)
ev.to_parquet(OUT + r"\shape_events.parquet")
base_feats = G.usable(ev, G.model_features(ev.drop(columns=[c for c in ev.columns if c.startswith(("sh_", "net_"))])))
shape_feats = [c for c in sh.columns if ev[c].notna().mean() > 0.5]
sets = {"hourly/daily factors": base_feats, "minute shape only": shape_feats, "both": base_feats + shape_feats}
ev["month"] = pd.to_datetime(ev.ts, unit="ms").dt.strftime("%Y-%m")
months = sorted(ev.month.unique())
print(f"months: {months[0]} .. {months[-1]} ({len(months)}); base rate +1%/-1%: {(ev['net_0.01_0.01'] > 0).mean():.1%} win, "
      f"{ev['net_0.01_0.01'].mean() * 100:+.3f}% per trade")


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_depth=4, min_samples_leaf=100,
                                          l2_regularization=5.0, random_state=7)


results = {}
for label in ("net_0.01_0.01", "net_0.015_0.015"):
    y = (ev[label] > 0).astype(int).to_numpy()
    for set_name, feats in sets.items():
        p = np.full(len(ev), np.nan)
        for m in months[3:]:
            m_start = ev.loc[ev.month == m, "ts"].min()
            tr = ev.ts < m_start - DAY
            te = (ev.month == m).to_numpy()
            mdl = model().fit(ev.loc[tr, feats].to_numpy(float), y[tr])
            p[te] = mdl.predict_proba(ev.loc[te, feats].to_numpy(float))[:, 1]
        results[(label, set_name)] = p
        oos = np.isfinite(p)
        print(f"{label} | {set_name:<22} OOS AUC {roc_auc_score(y[oos], p[oos]):.3f}  ({time.time()-t0:.0f}s)", flush=True)

np.save(OUT + r"\shape_preds.npy", np.array([results[k] for k in sorted(results)]))
pd.Series([f"{a}|{b}" for a, b in sorted(results)]).to_csv(OUT + r"\shape_pred_keys.csv", index=False)


def band_table(label, set_name):
    p = results[(label, set_name)]
    oos = np.isfinite(p)
    d = ev[oos].assign(p=p[oos])
    days = (d.ts.max() - d.ts.min()) / DAY
    rows = []
    for thr in (0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
        g = d[d.p >= thr]
        if len(g) < 30:
            continue
        net = g[label]
        daily = net.groupby(g.ts // DAY).mean()
        t = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))) if len(daily) > 5 else np.nan
        mon = net.groupby(g.month).mean()
        rows.append({"p>=": thr, "signals": len(g), "per_day": len(g) / days, "win": (net > 0).mean(),
                     "net_pct": net.mean() * 100, "t_day": t, "months_pos": (mon > 0).mean(), "n_months": len(mon),
                     "10x_margin_pct": net.mean() * 1000, "worst_month_pct": mon.min() * 100})
    return pd.DataFrame(rows)


for label in ("net_0.01_0.01", "net_0.015_0.015"):
    for set_name in sets:
        print(f"\n=== {label}  |  {set_name}: unseen months, every signal above the probability threshold")
        print(band_table(label, set_name).round(3).to_string(index=False))
