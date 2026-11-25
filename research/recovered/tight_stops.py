"""The user's plans: +10% on margin at 5x / 3x / 2x = +2% / +3.33% / +5% moves, with TIGHT stops.
All 63,528 grinds of the year, 5-minute paths (24h), Weex fees, stop counted first if a 5m bar touches both.
Then the minute-shape model as a filter, monthly walk-forward (11 unseen months)."""
import sys, glob, math, time
sys.path.insert(0, r"D:\Leverage trading")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from lts import aim as A
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
DAY = A.DAY
t0 = time.time()
ev = pd.read_parquet(OUT + r"\shape_events.parquet").sort_values("ts").reset_index(drop=True)
raw = pd.read_parquet(OUT + r"\aim_events.parquet")[["symbol", "ts", "side"]].reset_index()
post_all = np.concatenate([np.load(f) for f in sorted(glob.glob(OUT + r"\paths\chunk_*.npy"))])
pos = raw.set_index(["symbol", "ts", "side"])["index"]
post = post_all[pos.loc[list(zip(ev.symbol, ev.ts, ev.side))].to_numpy()]
PLANS = [(0.02, 5), (0.0333, 3), (0.05, 2)]              # (target move, leverage that makes it +10% on margin)
STOPS = (0.003, 0.005, 0.0075, 0.01)
A.TARGETS, A.STOPS, A.HOURS = tuple(t for t, _ in PLANS), STOPS, (24,)
outs = A.outcomes(post, ev.side.to_numpy())
ev["month"] = pd.to_datetime(ev.ts, unit="ms").dt.strftime("%Y-%m")
months = sorted(ev.month.unique())
oos_mask = ev.month.isin(months[3:]).to_numpy()
days_oos = (ev.ts[oos_mask].max() - ev.ts[oos_mask].min()) / DAY
cost_win, cost_loss = A.ENTRY_COST + A.MAKER, A.ENTRY_COST + A.TAKER + A.SLIP


def stats(net, ts, month, lev, days):
    net = net[np.isfinite(net)] if False else net
    s = pd.Series(net)
    daily = s.groupby(ts // DAY).mean()
    t = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))) if len(daily) > 5 and daily.std() > 0 else np.nan
    mon = s.groupby(month).mean()
    return {"signals": len(s), "per_day": len(s) / days, "win": float((s > 0).mean()), "net_pct": float(s.mean() * 100),
            "margin_pct": float(s.mean() * 100 * lev), "t_day": float(t), "months_pos": float((mon > 0).mean())}


print(f"{len(ev):,} grinds, {len(months)} months, unseen months used: {months[3]}..{months[-1]} ({days_oos:.0f} days)")
print("\nALL GRINDS (no filter), unseen months. Breakeven win rate includes fees (taker in, maker target, taker stop).")
rows = []
for T, lev in PLANS:
    for S in STOPS:
        net, xbar = outs[(T, S, 24)]
        g = oos_mask & np.isfinite(net)
        be = (S + cost_loss) / (T - cost_win + S + cost_loss)
        hit_t = np.mean(net[g] > T * 0.5)
        rows.append({"plan": f"{lev}x: +{T:.2%}", "stop": f"-{S:.2%}", "breakeven_win": be,
                     **stats(net[g], ev.ts.to_numpy()[g], ev.month.to_numpy()[g], lev, days_oos),
                     "median_hold_min": float(np.median(xbar[g]) * 5)})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---- model filter (minute shape), monthly walk-forward per plan
feats = [c for c in ev.columns if c.startswith("sh_") and ev[c].notna().mean() > 0.5]
print(f"\nMINUTE-SHAPE MODEL as a filter, trained only on earlier months ({len(feats)} features)")
res = []
for T, lev in PLANS:
    for S in STOPS:
        net, _ = outs[(T, S, 24)]
        y = (net > 0).astype(int)
        p = np.full(len(ev), np.nan)
        for m in months[3:]:
            start = ev.loc[ev.month == m, "ts"].min()
            tr = (ev.ts < start - DAY).to_numpy() & np.isfinite(net)
            te = (ev.month == m).to_numpy()
            mdl = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=100,
                                                 l2_regularization=5.0, random_state=7)
            mdl.fit(ev.loc[tr, feats].to_numpy(float), y[tr])
            p[te] = mdl.predict_proba(ev.loc[te, feats].to_numpy(float))[:, 1]
        g = oos_mask & np.isfinite(net) & np.isfinite(p)
        auc = roc_auc_score(y[g], p[g])
        for top in (1.0, 0.2, 0.1, 0.05, 0.02):
            thr = np.quantile(p[g], 1 - top) if top < 1 else -1
            sel = g & (p >= thr)
            res.append({"plan": f"{lev}x: +{T:.2%}", "stop": f"-{S:.2%}", "AUC": auc, "model_top": f"{top:.0%}",
                        **stats(net[sel], ev.ts.to_numpy()[sel], ev.month.to_numpy()[sel], lev, days_oos)})
        print(f"  {lev}x +{T:.2%} / -{S:.2%}: AUC {auc:.3f} ({time.time() - t0:.0f}s)", flush=True)
R = pd.DataFrame(res)
R.to_parquet(OUT + r"\tight_results.parquet")
print("\nBy model band (unseen months):")
print(R.round(3).to_string(index=False))
ok = R[(R.net_pct > 0) & (R.t_day >= 3) & (R.months_pos >= 0.7) & (R.per_day >= 1)]
print(f"\nPLANS THAT QUALIFY (profitable after fees, t>=3, >=70% of unseen months positive, >=1 signal/day): {len(ok)}")
if len(ok):
    print(ok.round(3).to_string(index=False))
