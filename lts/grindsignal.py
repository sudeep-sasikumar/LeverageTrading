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


# ---- recording every detected grind (sent or not) -------------------------------
def micro_context(db: DB, sym: str, ts: int) -> dict:
    """Liquidations and order book around a detection (tables exist once the collector runs)."""
    out = {}
    try:
        for hrs in (1, 6):
            for side in ("long_liq", "short_liq"):
                r = db.query("SELECT COALESCE(SUM(usd),0) u, COUNT(*) n FROM liquidations WHERE symbol=? AND side=? "
                             "AND ts>? AND ts<=?", (sym, side, ts - hrs * HOUR, ts))[0]
                out[f"{side}_{hrs}h_usd"], out[f"{side}_{hrs}h_n"] = float(r["u"]), int(r["n"])
        b = db.query("SELECT * FROM book_snapshots WHERE symbol=? AND ts<=? AND ts>? ORDER BY ts DESC LIMIT 1",
                     (sym, ts, ts - 15 * 60_000))
        if b:
            out["book"] = {k: b[0][k] for k in b[0].keys() if k not in ("symbol",)}
    except Exception:
        pass
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def record_grinds(db: DB, rows: list[dict], p_min: float) -> None:
    """Store everything known about each detected grind, whether or not it was sent."""
    import json
    from .grindlab import ensure_live_schema
    ensure_live_schema(db)
    for r in rows:
        factors = {k: v for k, v in r.items() if k not in ("band", "checks") and not isinstance(v, (dict, list))}
        db.execute("UPDATE grind_live SET all_factors=?, micro=?, threshold=?, model_p=? WHERE symbol=? AND ts=? AND side=?",
                   (json.dumps(_clean(factors)), json.dumps(_clean(micro_context(db, r["symbol"], r["ts"]))), p_min,
                    r.get("p"), r["symbol"], r["ts"], r["side"]))


def label_grinds(cfg, db: DB) -> int:
    """24h after each detected grind: how every plan and a target/stop grid would have done, plus best/worst moves."""
    import json
    from .grindlab import ensure_live_schema
    ensure_live_schema(db)
    gs = cfg.grindsignal
    now = int(time.time() * 1000)
    rows = db.query("SELECT symbol, ts, side FROM grind_live WHERE outcomes IS NULL AND ts <= ?",
                    (now - gs.hours * HOUR - 10 * 60_000,))
    grid = [{"leverage": 0, "target": t, "stop": st} for t in gs.record_targets for st in gs.record_stops]
    n = 0
    for r in rows:
        post = post_from_db(db, r["symbol"], int(r["ts"]), gs.hours * 12 + 1)
        if post is None:
            if now - r["ts"] > 3 * DAY:                       # data never came: stop retrying
                db.execute("UPDATE grind_live SET outcomes=?, outcome_ts=? WHERE symbol=? AND ts=? AND side=?",
                           ('{"missing": true}', now, r["symbol"], r["ts"], r["side"]))
            continue
        side = np.array([r["side"]])
        res_plans = plan_outcomes(post[None], side, gs.plans, gs.hours)
        res_grid = plan_outcomes(post[None], side, grid, gs.hours)
        entry = float(post[0, 0])
        hi, lo = post[:, 1].astype(float), post[:, 2].astype(float)
        fav = (hi / entry - 1) if r["side"] > 0 else (1 - lo / entry)
        adv = (1 - lo / entry) if r["side"] > 0 else (hi / entry - 1)
        out = {"entry": entry,
               "plans": {plan_name(pl): float(res_plans[plan_label(pl)][0]) for pl in gs.plans},
               "grid": {f"{g['target']:g}|{g['stop']:g}": float(res_grid[plan_label(g)][0]) for g in grid},
               "best": {h: float(np.nanmax(fav[:h * 12])) for h in (1, 4, 24)},
               "worst": {h: float(np.nanmax(adv[:h * 12])) for h in (1, 4, 24)},
               "close_24h": float(r["side"] * (post[-1, 3] / entry - 1))}
        db.execute("UPDATE grind_live SET outcomes=?, outcome_ts=? WHERE symbol=? AND ts=? AND side=?",
                   (json.dumps(_clean(out)), now, r["symbol"], r["ts"], r["side"]))
        n += 1
    return n


def plan_progress(post: np.ndarray, side: int, plans: list[dict]) -> dict:
    """Plans on the bars seen so far: {name: net % once the target or stop is hit, else None (still open)}."""
    from .aim import ENTRY_COST, SLIP, TAKER
    ok = np.isfinite(post[:, 1])
    post = post[ok]
    if not len(post):
        return {plan_name(pl): None for pl in plans}
    e = float(post[0, 0])
    hi, lo = post[:, 1].astype(float), post[:, 2].astype(float)
    fav = hi / e - 1 if side > 0 else 1 - lo / e
    adv = 1 - lo / e if side > 0 else hi / e - 1
    out = {}
    for pl in plans:
        hs, ht = np.flatnonzero(adv >= pl["stop"]), np.flatnonzero(fav >= pl["target"])
        i_s, i_t = (hs[0] if len(hs) else 10 ** 9), (ht[0] if len(ht) else 10 ** 9)
        if i_s <= i_t and i_s < 10 ** 9:
            out[plan_name(pl)] = -pl["stop"] - ENTRY_COST - TAKER - SLIP
        elif i_t < 10 ** 9:
            out[plan_name(pl)] = pl["target"] - ENTRY_COST - TAKER
        else:
            out[plan_name(pl)] = None
    return out


def progress_grinds(cfg, db: DB, now: int | None = None) -> int:
    """For grinds under 24h old: each plan's result as soon as its target or stop is hit, plus the best and
    worst move so far (shown on the dashboard instead of 'pending'). The full 24h record comes later."""
    import json
    from .grindlab import ensure_live_schema
    ensure_live_schema(db)
    gs = cfg.grindsignal
    now = now or int(time.time() * 1000)
    rows = db.query("SELECT symbol, ts, side FROM grind_live WHERE outcomes IS NULL AND ts > ?", (now - gs.hours * HOUR,))
    n = 0
    for r in rows:
        bars = int(min(gs.hours * 12, (now - r["ts"]) // BAR))
        if bars < 1:
            continue
        k = db.frame("SELECT ts,o,h,l,c FROM klines WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts",
                     [r["symbol"], r["ts"], r["ts"] + bars * BAR])
        if not len(k):
            continue
        post = k[["o", "h", "l", "c"]].to_numpy(np.float64)
        e = post[0, 0]
        fav = (post[:, 1] / e - 1) if r["side"] > 0 else (1 - post[:, 2] / e)
        adv = (1 - post[:, 2] / e) if r["side"] > 0 else (post[:, 1] / e - 1)
        prog = {"as_of": int(k["ts"].iloc[-1] + BAR), "entry": float(e), "plans": plan_progress(post, r["side"], gs.plans),
                "best": float(fav.max()), "worst": float(adv.max()),
                "now": float(r["side"] * (post[-1, 3] / e - 1))}
        db.execute("UPDATE grind_live SET progress=? WHERE symbol=? AND ts=? AND side=?",
                   (json.dumps(_clean(prog)), r["symbol"], r["ts"], r["side"]))
        n += 1
    return n


def weex_url(symbol: str) -> str:
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    return f"https://www.weex.com/futures/{base}-USDT"
