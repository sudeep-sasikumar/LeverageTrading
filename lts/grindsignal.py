"""Grind signals: every early grind is scored by a pattern-recognition model; the best-scoring share is sent
(no daily cap - the trader chooses), each with the trader's three plans:

    5x  : +2%     target, tight stop    (+10% on margin at target)
    3x  : +3.33%  target, tight stop
    2x  : +5%     target, tight stop

Model inputs: the hourly/daily grind-lab factors are available, but the shipped model uses the minute-level
shape of the last 6 hours (lts.grindshape), the only input with any predictive power on unseen months
(docs/DESIGN.md sections 16-17). Outcomes: 5-minute bars, 24h limit, Weex fees, stop first if a bar touches
both. Every bundle carries, per score band, how EACH plan did on unseen months; signals quote it.
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
TOP_SHARES = (1.0, 0.2, 0.1, 0.05, 0.02)        # score bands: all grinds, top 20%, ..., top 2%


def plan_label(plan: dict) -> str:
    return f"net_{plan['target']:g}_{plan['stop']:g}"


def plan_name(plan: dict) -> str:
    return f"{plan['leverage']}x +{plan['target']:.2%} / -{plan['stop']:.2%}"


# ---- 5-minute data -----------------------------------------------------------
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


def plan_outcomes(post: np.ndarray, side: np.ndarray, plans: list[dict], hours: int = 24) -> dict:
    """{label: net % array} for each plan (entry = first bar's open)."""
    old = (A.TARGETS, A.STOPS, A.HOURS)
    out = {}
    try:
        for p in plans:
            A.TARGETS, A.STOPS, A.HOURS = (p["target"],), (p["stop"],), (hours,)
            out[plan_label(p)] = A.outcomes(post, side)[(p["target"], p["stop"], hours)][0]
    finally:
        A.TARGETS, A.STOPS, A.HOURS = old
    return out


def feature_frame(ev: pd.DataFrame, pre: np.ndarray) -> pd.DataFrame:
    sh = pd.DataFrame([shape_features(pre[i], int(s)) for i, s in enumerate(ev["side"].to_numpy())], index=ev.index)
    return pd.concat([ev, sh], axis=1)


# ---- training -----------------------------------------------------------------
def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=100,
                                          l2_regularization=5.0, random_state=7)


def train(df: pd.DataFrame, feats: list[str], plans: list[dict], primary: int, min_train_months: int = 3) -> dict:
    """Monthly walk-forward for the honest band tables (every plan), then a final model on everything.
    The model learns the PRIMARY plan's outcome; all plans are reported per band."""
    df = df.sort_values("ts").reset_index(drop=True)
    label = plan_label(plans[primary])
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
    bands = band_tables(df.assign(p=p, month=month), plans)
    final = _model().fit(df[feats].to_numpy(float), y)
    return {"model": final, "features": feats, "plans": plans, "primary": primary, "bands": bands,
            "trained_ts": int(time.time() * 1000), "n": int(len(df)),
            "oos_months": int(month[np.isfinite(p)].nunique()) if np.isfinite(p).any() else 0,
            "data_from": int(df["ts"].min()), "data_to": int(df["ts"].max())}


def band_tables(d: pd.DataFrame, plans: list[dict]) -> list[dict]:
    d = d[np.isfinite(d["p"])]
    if d.empty:
        return []
    days = max((d["ts"].max() - d["ts"].min()) / DAY, 1.0)
    out = []
    for share in TOP_SHARES:
        thr = float(d["p"].quantile(1 - share)) if share < 1 else 0.0
        g = d[d["p"] >= thr]
        if len(g) < 30:
            continue
        rows = []
        for pl in plans:
            net = g[plan_label(pl)].dropna()
            daily = net.groupby(g.loc[net.index, "ts"] // DAY).mean()
            t = float(daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily)))) if len(daily) > 5 and daily.std() > 0 else 0.0
            mon = net.groupby(g.loc[net.index, "month"]).mean()
            rows.append({**pl, "win": float((net > 0).mean()), "net": float(net.mean()),
                         "margin": float(net.mean() * pl["leverage"]), "t_day": t,
                         "months_pos": float((mon > 0).mean()), "n_months": int(len(mon))})
        out.append({"top": share, "p_min": thr, "signals": int(len(g)), "per_day": float(len(g) / days), "plans": rows})
    return out


def band_for(bundle: dict, p: float) -> dict | None:
    """The narrowest band this score falls in (how signals like it did on unseen months)."""
    best = None
    for b in bundle.get("bands", []):
        if p >= b["p_min"]:
            best = b
    return best


def threshold(bundle: dict, top_share: float) -> float:
    for b in bundle.get("bands", []):
        if abs(b["top"] - top_share) < 1e-9:
            return b["p_min"]
    return 1.0


def qualifies(plan_stats: dict | None) -> bool:
    """Pre-registered: an edge only if profitable after fees with t >= 3, >= 70% of unseen months positive."""
    return bool(plan_stats and plan_stats["net"] > 0 and plan_stats["t_day"] >= 3 and plan_stats["months_pos"] >= 0.7
                and plan_stats["n_months"] >= 4)


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
            if "plans" not in b:                 # an older single-plan bundle: ignore it
                continue
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
    ev = ev.assign(**plan_outcomes(np.stack(posts), ev["side"].to_numpy(), gs.plans, gs.hours))
    df = feature_frame(ev, np.stack(pres))
    df = df[np.isfinite(df[plan_label(gs.plans[gs.primary])])]
    feats = grindlab.usable(df, [c for c in df.columns if c.startswith("sh_")])
    bundle = train(df, feats, gs.plans, gs.primary)
    path = cfg.path(cfg.storage.model_dir) / "grindsignal.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return bundle


# ---- tracking every plan of every signal --------------------------------------
PLAN_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_plans (
    signal_id INTEGER NOT NULL, plan TEXT NOT NULL, leverage INTEGER, target REAL, stop REAL,
    outcome TEXT, net REAL, margin REAL, done_ts INTEGER,
    PRIMARY KEY (signal_id, plan)
) WITHOUT ROWID;
"""


def label_signal_plans(cfg, db: DB) -> int:
    """After 24h, record how each of the three plans would have done for every grind signal."""
    db.conn.executescript(PLAN_SCHEMA)
    gs = cfg.grindsignal
    now = int(time.time() * 1000)
    rows = db.query("""SELECT s.id, s.symbol, s.side, s.ts, s.entry FROM signals s
                       WHERE s.kind='grindsig' AND s.ts <= ?
                       AND NOT EXISTS (SELECT 1 FROM signal_plans p WHERE p.signal_id = s.id)""",
                    (now - gs.hours * HOUR - 10 * 60_000,))
    n = 0
    for r in rows:
        start = (r["ts"] // BAR + 1) * BAR                    # first full 5m bar after the signal
        post = post_from_db(db, r["symbol"], start, gs.hours * 12 + 1)
        if post is None:
            continue
        post = post.copy()
        post[0, 0] = r["entry"]                               # the signal's own entry price
        res = plan_outcomes(post[None], np.array([r["side"]]), gs.plans, gs.hours)
        for pl in gs.plans:
            net = float(res[plan_label(pl)][0])
            outcome = "target" if net >= pl["target"] - 0.003 else ("stop" if net <= -pl["stop"] else "time")
            db.execute("INSERT OR REPLACE INTO signal_plans VALUES (?,?,?,?,?,?,?,?,?)",
                       (r["id"], plan_name(pl), pl["leverage"], pl["target"], pl["stop"], outcome, net,
                        net * pl["leverage"], now))
        n += 1
    return n
