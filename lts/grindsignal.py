"""Grind signals: every early grind, scored by a pattern-recognition model, sent when its probability of
reaching the target before the stop clears the threshold (no daily cap - the trader chooses).

Model inputs: the hourly/daily grind-lab factors (lts.grindlab) + the minute-level shape of the last 6 hours
(lts.grindshape). Label: +TARGET before -STOP within 24h on 5-minute bars, net of fees (lts.aim).
Training: nightly on the VPS's own stored data (5m klines are kept 120 days); until there is enough, a
seed model trained on a year of history (shipped in lts/models/) is used. Each bundle carries a table of how
each probability band actually did on unseen months, quoted in every signal.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import aim as A
from .db import DB
from .grindshape import BARS, shape_features

log = logging.getLogger("lts.grindsignal")
DAY, HOUR, BAR = 86_400_000, 3_600_000, 300_000
SEED_PATH = Path(__file__).parent / "models" / "grind_seed.joblib"
BANDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75)


def pre_from_db(db: DB, sym: str, ts: int) -> np.ndarray | None:
    """The 72 five-minute bars before ts: o,h,l,c,qv,tbqv (NaN-padded at the front if short)."""
    k = db.frame("SELECT ts,o,h,l,c,qv,tbqv FROM klines WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts",
                 [sym, ts - BARS * BAR, ts])
    if len(k) < BARS * 0.8:
        return None
    arr = np.full((BARS, 6), np.nan, dtype=np.float32)
    idx = ((k["ts"].to_numpy() - (ts - BARS * BAR)) // BAR).astype(int)
    arr[idx] = k[["o", "h", "l", "c", "qv", "tbqv"]].to_numpy(np.float32)
    return arr


def post_from_db(db: DB, sym: str, ts: int, bars: int = 289) -> np.ndarray | None:
    k = db.frame("SELECT ts,o,h,l,c FROM klines WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts", [sym, ts, ts + bars * BAR])
    if len(k) < bars * 0.98:
        return None
    arr = np.full((bars, 4), np.nan, dtype=np.float32)
    idx = ((k["ts"].to_numpy() - ts) // BAR).astype(int)
    arr[idx] = k[["o", "h", "l", "c"]].to_numpy(np.float32)
    return arr


def label_name(target: float, stop: float) -> str:
    return f"net_{target:g}_{stop:g}"


def add_labels(ev: pd.DataFrame, post: np.ndarray, target: float, stop: float, hours: int = 24) -> pd.DataFrame:
    net, _ = A.outcomes(post, ev["side"].to_numpy())[(target, stop, hours)] if (target in A.TARGETS and stop in A.STOPS) \
        else _custom(post, ev["side"].to_numpy(), target, stop, hours)
    return ev.assign(**{label_name(target, stop): net})


def _custom(post, side, target, stop, hours):
    old = (A.TARGETS, A.STOPS, A.HOURS)
    A.TARGETS, A.STOPS, A.HOURS = (target,), (stop,), (hours,)
    try:
        return A.outcomes(post, side)[(target, stop, hours)]
    finally:
        A.TARGETS, A.STOPS, A.HOURS = old


def feature_frame(ev: pd.DataFrame, pre: np.ndarray) -> pd.DataFrame:
    sh = pd.DataFrame([shape_features(pre[i], int(s)) for i, s in enumerate(ev["side"].to_numpy())], index=ev.index)
    return pd.concat([ev, sh], axis=1)


def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_depth=4, min_samples_leaf=100,
                                          l2_regularization=5.0, random_state=7)


def train(df: pd.DataFrame, feats: list[str], label: str, min_train_months: int = 3) -> dict:
    """Monthly walk-forward (for the honest band table), then a final model on everything."""
    df = df.sort_values("ts").reset_index(drop=True)
    y = (df[label] > 0).astype(int).to_numpy()
    month = pd.to_datetime(df["ts"], unit="ms").dt.strftime("%Y-%m")
    months = sorted(month.unique())
    p = np.full(len(df), np.nan)
    for m in months[min_train_months:]:
        start = df.loc[month == m, "ts"].min()
        tr = (df["ts"] < start - DAY).to_numpy()
        te = (month == m).to_numpy()
        if tr.sum() < 2000 or len(np.unique(y[tr])) < 2:
            continue
        p[te] = _model().fit(df.loc[tr, feats].to_numpy(float), y[tr]).predict_proba(df.loc[te, feats].to_numpy(float))[:, 1]
    bands = band_table(df.assign(p=p, month=month), label)
    final = _model().fit(df[feats].to_numpy(float), y)
    return {"model": final, "features": feats, "label": label, "bands": bands, "trained_ts": int(time.time() * 1000),
            "n": int(len(df)), "base_win": float((df[label] > 0).mean()), "base_net": float(df[label].mean()),
            "oos_months": int(np.isfinite(p).any() and month[np.isfinite(p)].nunique()),
            "data_from": int(df["ts"].min()), "data_to": int(df["ts"].max())}


def band_table(d: pd.DataFrame, label: str) -> list[dict]:
    d = d[np.isfinite(d["p"])]
    if d.empty:
        return []
    days = max((d["ts"].max() - d["ts"].min()) / DAY, 1.0)
    rows = []
    for thr in (0.0,) + BANDS:
        g = d[d["p"] >= thr]
        if len(g) < 30:
            continue
        net = g[label]
        daily = net.groupby(g["ts"] // DAY).mean()
        t = float(daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily)))) if len(daily) > 5 and daily.std() > 0 else 0.0
        mon = net.groupby(g["month"]).mean()
        rows.append({"p_min": thr, "signals": int(len(g)), "per_day": float(len(g) / days), "win": float((net > 0).mean()),
                     "net": float(net.mean()), "t_day": t, "months_pos": float((mon > 0).mean()), "n_months": int(len(mon))})
    return rows


def band_for(bundle: dict, p: float) -> dict | None:
    """The highest band at or below p (how signals like this did on unseen months)."""
    best = None
    for b in bundle.get("bands", []):
        if p >= b["p_min"]:
            best = b
    return best


def qualifies(b: dict | None) -> bool:
    """Pre-registered: an edge only if profitable after fees with t >= 3, >= 70% of unseen months positive."""
    return bool(b and b["net"] > 0 and b["t_day"] >= 3 and b["months_pos"] >= 0.7 and b["n_months"] >= 4)


def score_rows(rows: list[dict], pres: list[np.ndarray | None], bundle: dict) -> list[dict]:
    for r, pre in zip(rows, pres):
        r.update(shape_features(pre, r["side"]) if pre is not None else {})
        x = np.array([[r.get(f, np.nan) if r.get(f) is not None else np.nan for f in bundle["features"]]], dtype=float)
        r["p"] = float(bundle["model"].predict_proba(x)[0, 1])
        r["band"] = band_for(bundle, r["p"])
    return rows


def load_bundle(cfg) -> dict | None:
    import joblib
    live = cfg.path(cfg.storage.model_dir) / "grindsignal.joblib"
    for p in (live, SEED_PATH):
        if p.exists():
            b = joblib.load(p)
            b["source"] = "trained on this server" if p == live else "seed (one year of history)"
            return b
    return None


def nightly(cfg, db: DB) -> dict | None:
    """Retrain on this server's own data once it holds enough 5-minute history (else keep the seed model)."""
    import joblib
    from . import grindlab
    gs = cfg.grindsignal
    first = db.query("SELECT MIN(ts) AS a FROM klines WHERE symbol='BTCUSDT'")[0]["a"]
    if not first or (time.time() * 1000 - first) / DAY < gs.min_days_for_own_model:
        return None
    days = min(cfg.grindlab.days, (time.time() * 1000 - first) / DAY - 2)
    ev, _ = grindlab.build(db, days=days)
    if len(ev) < 3000:
        return None
    pres, posts, keep = [], [], []
    for i, r in ev.iterrows():
        pre, post = pre_from_db(db, r["symbol"], int(r["ts"])), post_from_db(db, r["symbol"], int(r["ts"]))
        if pre is None or post is None:
            continue
        pres.append(pre), posts.append(post), keep.append(i)
    ev = ev.loc[keep].reset_index(drop=True)
    ev = add_labels(ev, np.stack(posts), gs.target, gs.stop)
    df = feature_frame(ev, np.stack(pres))
    label = label_name(gs.target, gs.stop)
    df = df[np.isfinite(df[label])]
    seed = load_bundle(cfg) or {}
    feats = [f for f in (seed.get("features") or []) if f in df] or \
        [c for c in df.columns if c.startswith("sh_") or c in grindlab.model_features(df)]
    feats = grindlab.usable(df, feats)
    bundle = train(df, feats, label)
    path = cfg.path(cfg.storage.model_dir) / "grindsignal.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return bundle
