"""What separates 5x +2% / -1% winners among the year's early grinds?

Label: the 5x plan (target +2%, stop -1%, 24h, Weex fees) from detection, entered at once.
1. Factor scan on the first half (discovery), checked on the second half (confirmation).
2. Monthly walk-forward models with three feature sets: minute shape only (today's model), hourly/daily
   factors only, and both. Out-of-sample bands, day-clustered t, months positive.
3. The aim: per day, the FIRST grind (in time) whose out-of-sample score clears the band threshold.
"""
import glob, sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:\Leverage trading")
from lts import grindsignal as GS

OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
DAY = 86_400_000
ev = pd.read_parquet(OUT + r"\shape_events.parquet").sort_values("ts").reset_index(drop=True)
raw = pd.read_parquet(OUT + r"\aim_events.parquet")[["symbol", "ts", "side"]].reset_index()
post_all = np.concatenate([np.load(f) for f in sorted(glob.glob(OUT + r"\paths\chunk_*.npy"))])
post = post_all[raw.set_index(["symbol", "ts", "side"])["index"].loc[list(zip(ev.symbol, ev.ts, ev.side))].to_numpy()]
PL = {"leverage": 5, "target": 0.02, "stop": 0.01}
ev["net"] = GS.plan_outcomes(post, ev.side.to_numpy(), [PL], 24)[GS.plan_label(PL)]
ev = ev[np.isfinite(ev.net)].reset_index(drop=True)
ev["win"] = (ev.net > 0).astype(int)
ev["month"] = pd.to_datetime(ev.ts, unit="ms").dt.strftime("%Y-%m")
ev["day"] = ev.ts // DAY
months = sorted(ev.month.unique())
disc = ev.month.isin(months[:len(months) // 2])

DROP = {"symbol", "ts", "side", "price", "success", "result", "ret_r", "mfe", "mae", "fwd24", "hours_to_target", "fwd12",
        "x_trail", "net", "win", "month", "day", "oi_chg_6h", "oi_chg_24h", "lsr_glob_chg24", "lsr_top_chg24", "lsr_glob"}
DROP |= {c for c in ev.columns if c.startswith(("x_0", "net_"))}
shape = [c for c in ev.columns if c.startswith("sh_") and ev[c].notna().mean() > 0.5]
hourly = [c for c in ev.columns if c not in DROP and not c.startswith("sh_") and ev[c].dtype.kind in "fiub"
          and ev[c].notna().mean() > 0.5]
BE = (0.01 + 0.0012 + 0.0012) / (0.02 - 0.0012 - 0.0002 + 0.01 + 0.0024)
print(f"grinds {len(ev):,}; 5x +2/-1 won {ev.win.mean():.1%} (break-even about {BE:.1%}), {ev.net.mean():+.3%} per trade")
print(f"shape features {len(shape)}, hourly/daily factors {len(hourly)}")

# ---- 1. factor scan --------------------------------------------------------
print("\n=== factors: win rate in top vs bottom fifth (discovery -> confirmation) ===")
res = []
for c in hourly + shape:
    x = ev[c]
    if x.nunique() < 5:
        continue
    q = x[disc].quantile([0.2, 0.8]).to_numpy()
    out = {"factor": c}
    for part, m in (("d", disc), ("c", ~disc)):
        lo, hi = ev.win[m & (x <= q[0])], ev.win[m & (x >= q[1])]
        nlo, nhi = ev.net[m & (x <= q[0])], ev.net[m & (x >= q[1])]
        out[f"{part}_hi"], out[f"{part}_lo"] = hi.mean(), lo.mean()
        out[f"{part}_nethi"], out[f"{part}_netlo"] = nhi.mean(), nlo.mean()
    res.append(out)
f = pd.DataFrame(res)
f["d_gap"] = f.d_hi - f.d_lo
f["c_gap"] = f.c_hi - f.c_lo
f["same"] = np.sign(f.d_gap) == np.sign(f.c_gap)
f = f.reindex(f.d_gap.abs().sort_values(ascending=False).index)
show = f.head(20)
for r in show.itertuples():
    print(f"{r.factor:<22} top fifth won {r.d_hi:.1%} vs bottom {r.d_lo:.1%} | confirmation {r.c_hi:.1%} vs {r.c_lo:.1%} "
          f"{'held' if r.same and abs(r.c_gap) > 0.02 else 'did not hold'}; best fifth net (conf) "
          f"{max(r.c_nethi, r.c_netlo):+.3%}")


# ---- 2. walk-forward models ------------------------------------------------
def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_depth=4, min_samples_leaf=150,
                                          l2_regularization=5.0, random_state=7)


def walk(feats):
    p = np.full(len(ev), np.nan)
    for m in months[3:]:
        start = ev.loc[ev.month == m, "ts"].min()
        tr = (ev.ts < start - DAY).to_numpy()
        te = (ev.month == m).to_numpy()
        p[te] = model().fit(ev.loc[tr, feats].to_numpy(float), ev.win[tr]).predict_proba(ev.loc[te, feats].to_numpy(float))[:, 1]
    return p


def band(p, top, mask=None):
    ok = np.isfinite(p) if mask is None else (np.isfinite(p) & mask)
    thr = np.nanquantile(p[ok], 1 - top)
    sel = ok & (p >= thr)
    d = ev.loc[sel].groupby("day").net.mean()
    mo = ev.loc[sel].groupby("month").net.mean()
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return {"top": top, "n": int(sel.sum()), "per_day": sel.sum() / ev.loc[ok, "day"].nunique(), "win": ev.win[sel].mean(),
            "net": ev.net[sel].mean(), "t": t, "months_pos": (mo > 0).mean(), "n_months": len(mo)}


from sklearn.metrics import roc_auc_score
scores = {}
for name, feats in (("minute shape (today)", shape), ("hourly/daily factors", hourly), ("both", shape + hourly)):
    p = walk(feats)
    scores[name] = p
    ok = np.isfinite(p)
    print(f"\n=== model: {name}  (unseen months {months[3]}..{months[-1]}, AUC {roc_auc_score(ev.win[ok], p[ok]):.3f}) ===")
    for top in (1.0, 0.2, 0.1, 0.05, 0.02, 0.01):
        b = band(p, top)
        print(f"  top {top:>4.0%}: {b['per_day']:5.1f}/day  won {b['win']:.1%}  {b['net']:+.3%}/trade  ({b['net'] * 5:+.2%} margin)  "
              f"t {b['t']:+.1f}  months+ {b['months_pos']:.0%} of {b['n_months']}")

# ---- 3. the aim: one trade a day -------------------------------------------
print("\n=== the aim: first grind of each day above the threshold (5x +2/-1) ===")
for name, p in scores.items():
    for top in (0.05, 0.02, 0.01):
        ok = np.isfinite(p)
        thr = np.nanquantile(p[ok], 1 - top)
        sel = ev[ok & (p >= thr)].sort_values("ts").groupby("day").head(1)
        days = ev.loc[ok, "day"].nunique()
        mo = sel.groupby("month").net.mean()
        print(f"  {name:<22} top {top:.0%}: trade on {len(sel) / days:.0%} of days, won {sel.win.mean():.1%}, "
              f"{sel.net.mean() * 5:+.2%} margin per trade, months+ {(mo > 0).mean():.0%}")
ev.assign(**{f"p_{k}": v for k, v in scores.items()})[["symbol", "ts", "side", "net", "win"] + [f"p_{k}" for k in scores]] \
    .to_parquet(OUT + r"\five_x_scores.parquet")
