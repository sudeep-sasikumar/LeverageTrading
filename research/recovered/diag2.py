"""Walk-forward comparison: feature subsets, model families, and a cross-sectional (market-neutral) target."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, r"D:\Leverage trading")
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from lts.config import load_config
from lts.db import DB
from lts import learner as L
from lts.features import feature_names, MARKET_LEVEL

cfg = load_config(r"D:\Leverage trading\config.yaml"); db = DB(cfg.path(cfg.storage.db_path))
df = L.load_labelled(db)
H4 = 4 * 3_600_000
df["bucket"] = df.ts // H4
print("rows", len(df), "| base TP", round(df.label.mean(), 3), "net R", round(df.ret_r.mean(), 3))
allf = [c for c in feature_names() if c in df and df[c].notna().mean() > 0.02]
idio = [f for f in allf if f not in MARKET_LEVEL]

def folds(df, n=5):
    tmin, tmax = int(df.ts.min()), int(df.ts.max()) + 1
    edges = np.linspace(tmin + 0.4 * (tmax - tmin), tmax, n + 1).astype(np.int64)
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        yield i, df[(df.ts < lo) & (df.exit_ts < lo)], df[(df.ts >= lo) & (df.ts < hi)]

def dm(s, b):
    return s - s.groupby(b).transform("mean")

def evaluate(name, oos):
    xs = np.corrcoef(dm(oos.score, oos.bucket), dm(oos.label.astype(float), oos.bucket))[0, 1]
    top = oos[oos.score >= oos.score.quantile(0.9)]
    best4 = oos.loc[oos.groupby("bucket").score.idxmax()]                 # best candidate per 4h block
    oos = oos.assign(day=oos.ts // 86_400_000)
    best1 = oos.loc[oos.groupby("day").score.idxmax()]
    # rank within bucket: top 10% of each bucket (needs >=10 candidates)
    r = oos.groupby("bucket").score.rank(pct=True)
    xtop = oos[r >= 0.9]
    print(f"{name:<30} AUC {roc_auc_score(oos.label, oos.score):.3f} | xsec corr {xs:+.3f} | pooled top10% TP {top.label.mean():.1%} R {top.ret_r.mean():+.3f} "
          f"| in-bucket top10% TP {xtop.label.mean():.1%} R {xtop.ret_r.mean():+.3f} (n={len(xtop)}) "
          f"| best/4h TP {best4.label.mean():.1%} R {best4.ret_r.mean():+.3f} (n={len(best4)}) "
          f"| best/day TP {best1.label.mean():.1%} R {best1.ret_r.mean():+.3f} (n={len(best1)})")

def run(name, feats, kind, **kw):
    parts = []
    for i, tr, te in folds(df):
        w = L.sample_weights(tr, cfg, int(te.ts.min()))
        Xtr, Xte = tr[feats].to_numpy(), te[feats].to_numpy()
        if kind == "clf":
            m = HistGradientBoostingClassifier(max_iter=kw.get("it", 200), learning_rate=0.04, max_depth=kw.get("d", 3),
                                               min_samples_leaf=kw.get("leaf", 150), l2_regularization=5, max_bins=64, random_state=7)
            m.fit(Xtr, tr.label, sample_weight=w); s = m.predict_proba(Xte)[:, 1]
        elif kind == "xsec":
            y = dm(tr.label.astype(float), tr.bucket)
            m = HistGradientBoostingRegressor(max_iter=kw.get("it", 200), learning_rate=0.04, max_depth=kw.get("d", 3),
                                              min_samples_leaf=kw.get("leaf", 150), l2_regularization=5, max_bins=64, random_state=7)
            m.fit(Xtr, y, sample_weight=w); s = m.predict(Xte)
        elif kind == "xsec_r":      # cross-sectional target on net R
            y = dm(tr.ret_r, tr.bucket)
            m = HistGradientBoostingRegressor(max_iter=kw.get("it", 200), learning_rate=0.04, max_depth=kw.get("d", 3),
                                              min_samples_leaf=kw.get("leaf", 150), l2_regularization=5, max_bins=64, random_state=7)
            m.fit(Xtr, y, sample_weight=w); s = m.predict(Xte)
        elif kind == "ridge":
            y = dm(tr.label.astype(float), tr.bucket)
            m = make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(n_quantiles=200, output_distribution="normal"),
                              Ridge(alpha=kw.get("alpha", 300)))
            m.fit(Xtr, y, ridge__sample_weight=w); s = m.predict(Xte)
        parts.append(te[["ts", "symbol", "label", "ret_r", "exit_ts", "outcome", "bucket"]].assign(score=s))
    evaluate(name, pd.concat(parts))

rng = np.random.default_rng(0)
evaluate("random pick (reference)", df[df.ts >= df.ts.quantile(0.4)].assign(score=rng.random((df.ts >= df.ts.quantile(0.4)).sum())))
run("clf all features", allf, "clf")
run("clf idio", idio, "clf")
run("xsec-label idio GBM d3", idio, "xsec")
run("xsec-label idio GBM d2", idio, "xsec", d=2, it=150)
run("xsec-label idio GBM d4", idio, "xsec", d=4, it=300, leaf=80)
run("xsec-netR idio GBM d3", idio, "xsec_r")
run("xsec-label idio ridge", idio, "ridge")
run("xsec-label all GBM d3", allf, "xsec")
