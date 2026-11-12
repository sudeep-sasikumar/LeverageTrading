"""Learning loop: purged walk-forward validation, calibration, threshold selection, promotion.

The model is a meta-labeller: given a candidate that already passed the momentum trigger,
estimate P(take-profit is hit before the stop or the time limit).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from .config import Config
from .db import DB
from .features import MARKET_LEVEL, feature_names

log = logging.getLogger("lts.learner")
DAY = 86_400_000
XSEC_BUCKET_MS = 4 * 3_600_000


# ---- data -------------------------------------------------------------------
def load_labelled(db: DB) -> pd.DataFrame:
    rows = db.query("""SELECT id, symbol, ts, side, source, label, outcome, ret_r, exit_ts, rule_score, sl_pct,
                              mfe, mae, features FROM candidates
                       WHERE labeled_ts IS NOT NULL AND label IS NOT NULL ORDER BY ts, id""")
    if not rows:
        return pd.DataFrame()
    # 'side' is also a model feature, so take it from the feature frame only
    meta = pd.DataFrame([{k: r[k] for k in r.keys() if k not in ("features", "side")} for r in rows])
    feats = pd.DataFrame([json.loads(r["features"]) for r in rows]).reindex(columns=feature_names())
    df = pd.concat([meta, feats.astype("float64")], axis=1)
    # a live row and its replayed twin describe the same event: keep the live one
    df["_live"] = (df["source"] == "live").astype(int)
    df = df.sort_values(["ts", "symbol", "side", "_live"]).drop_duplicates(["symbol", "ts", "side"], keep="last")
    return df.drop(columns="_live").sort_values(["ts", "id"]).reset_index(drop=True)


def sample_weights(df: pd.DataFrame, cfg: Config, ref_ts: int) -> np.ndarray:
    age_days = (ref_ts - df["ts"].to_numpy()) / DAY
    w = 0.5 ** (np.maximum(age_days, 0) / cfg.learn.time_decay_halflife_days)
    # overlapping candidates of one symbol are near-duplicates: share their weight
    bucket = df["ts"] // (4 * 3_600_000)
    cnt = df.groupby([df["symbol"], bucket])["ts"].transform("count").to_numpy()
    w = w / np.sqrt(cnt)
    w = w * np.where(df["source"].to_numpy() == "live", cfg.learn.live_weight, 1.0)
    return w / w.mean()


def usable_features(df: pd.DataFrame, cfg: Config) -> list[str]:
    feats = [c for c in feature_names() if c in df and df[c].notna().mean() > 0.02]
    span_days = (df["ts"].max() - df["ts"].min()) / DAY
    if span_days < cfg.learn.market_features_min_days:
        feats = [c for c in feats if c not in MARKET_LEVEL]
    return feats


def fit_model(df: pd.DataFrame, feats: list[str], cfg: Config, ref_ts: int):
    lc = cfg.learn
    params = dict(max_iter=lc.max_iter, learning_rate=lc.learning_rate, max_depth=lc.max_depth,
                  min_samples_leaf=lc.min_samples_leaf, l2_regularization=lc.l2, max_bins=64,
                  early_stopping=False, random_state=7)
    w = sample_weights(df, cfg, ref_ts)
    if lc.objective == "xsec":
        y = df["label"].astype(float)
        y = y - y.groupby(df["ts"] // XSEC_BUCKET_MS).transform("mean")      # remove the "kind of day" effect
        return HistGradientBoostingRegressor(**params).fit(df[feats].to_numpy(), y.to_numpy(), sample_weight=w)
    return HistGradientBoostingClassifier(**params).fit(df[feats].to_numpy(), df["label"].to_numpy(), sample_weight=w)


def raw_score(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x)


# ---- validation -------------------------------------------------------------
def walk_forward(df: pd.DataFrame, feats: list[str], cfg: Config) -> pd.DataFrame:
    """Expanding-window out-of-sample scores. Training rows whose label window reaches into
    the test block are purged."""
    tmin, tmax = int(df["ts"].min()), int(df["ts"].max()) + 1
    edges = np.linspace(tmin + 0.4 * (tmax - tmin), tmax, cfg.learn.n_folds + 1).astype(np.int64)
    parts = []
    for i in range(cfg.learn.n_folds):
        lo, hi = edges[i], edges[i + 1]
        train = df[(df["ts"] < lo) & (df["exit_ts"] < lo)]
        test = df[(df["ts"] >= lo) & (df["ts"] < hi)]
        if len(train) < 300 or len(test) == 0 or train["label"].nunique() < 2:
            continue
        model = fit_model(train, feats, cfg, ref_ts=int(lo))
        out = test[["id", "symbol", "ts", "side", "label", "outcome", "ret_r", "exit_ts", "rule_score"]].copy()
        out["score"] = raw_score(model, test[feats].to_numpy())
        out["fold"] = i
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def simulate_policy(df: pd.DataFrame, col: str, thr: float, cfg: Config) -> dict:
    """Replay the live signalling rules over scored candidates (sorted by ts)."""
    r = cfg.risk
    cool_ms = int(r.symbol_cooldown_hours * 3_600_000)
    open_pos: list[tuple[int, str, float]] = []        # (exit_ts, symbol, ret_r)
    cool_until: dict[str, int] = {}
    day, day_r, day_n, streak = None, 0.0, 0, 0
    trades = []
    ts_a, sym_a, p_a = df["ts"].to_numpy(), df["symbol"].to_numpy(), df[col].to_numpy()
    rr_a, ex_a, out_a = df["ret_r"].to_numpy(), df["exit_ts"].to_numpy(), df["outcome"].to_numpy()
    for i in np.flatnonzero(p_a >= thr):
        ts, sym = int(ts_a[i]), sym_a[i]
        d = ts // DAY
        if d != day:
            day, day_r, day_n, streak = d, 0.0, 0, 0
        still = []
        for ex, s, rr in sorted(open_pos):
            if ex <= ts:                                  # realise closed trades
                if ex // DAY == d:
                    day_r += rr
                streak = streak + 1 if rr < 0 else 0
                if rr < 0:
                    cool_until[s] = ex + cool_ms
            else:
                still.append((ex, s, rr))
        open_pos = still
        if len(open_pos) >= r.max_open or day_n >= r.max_per_day:
            continue
        if day_r <= r.daily_stop_r or streak >= r.max_consecutive_losses:
            continue
        if any(s == sym for _, s, _ in open_pos) or cool_until.get(sym, 0) > ts:
            continue
        open_pos.append((int(ex_a[i]), sym, float(rr_a[i])))
        day_n += 1
        trades.append((ts, int(ex_a[i]), float(rr_a[i]), out_a[i]))
    span_days = max((ts_a.max() - ts_a.min()) / DAY, 1e-9) if len(ts_a) else 1.0
    if not trades:
        return {"thr": float(thr), "n": 0, "per_day": 0.0, "t": 0.0, "win": 0.0, "avg_r": 0.0, "total_r": 0.0,
                "max_dd_r": 0.0, "pf": 0.0, "day_cover": 0.0}
    t = pd.DataFrame(trades, columns=["ts", "exit_ts", "r", "outcome"]).sort_values("exit_ts")
    eq = t["r"].cumsum()
    gains, losses = t.loc[t["r"] > 0, "r"].sum(), -t.loc[t["r"] < 0, "r"].sum()
    sd = float(t["r"].std(ddof=1)) if len(t) > 1 else 0.0
    return {"thr": float(thr), "n": len(t), "per_day": len(t) / span_days,
            "t": float(t["r"].mean() / (sd / np.sqrt(len(t)))) if sd > 0 else 0.0,
            "win": float((t["outcome"] == "tp").mean()), "avg_r": float(t["r"].mean()),
            "total_r": float(t["r"].sum()), "max_dd_r": float((eq.cummax().clip(lower=0) - eq).max()),
            "pf": float(gains / losses) if losses > 0 else float("inf"),
            "day_cover": float((t["ts"] // DAY).nunique() / max(span_days, 1.0))}


def threshold_table(df: pd.DataFrame, col: str, cfg: Config) -> list[dict]:
    qs = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]
    thrs = sorted(set(np.round(np.quantile(df[col].to_numpy(), qs), 5)))
    return [simulate_policy(df, col, t, cfg) for t in thrs]


def pick_threshold(table: list[dict], cfg: Config, min_n: int = 25) -> tuple[dict | None, dict | None]:
    """(main, floor): main maximises expectancy while keeping the signal-frequency target;
    floor is the loosest threshold that still clears the minimum expected value."""
    ok = [t for t in table if t["n"] >= min_n]
    if not ok:
        return None, None
    freq = [t for t in ok if t["per_day"] >= 0.8 * cfg.signal.target_per_day]
    main = max(freq or ok, key=lambda t: t["avg_r"])
    floors = [t for t in ok if t["avg_r"] >= cfg.signal.min_ev_r and t["thr"] <= main["thr"]]
    floor = min(floors, key=lambda t: t["thr"]) if floors else main
    return main, floor


def _auc(y, s) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _fmt(t: dict | None) -> str:
    if not t:
        return "n/a"
    return (f"score>={t['thr']:+.3f}: {t['n']} trades ({t['per_day']:.2f}/day), win {t['win']:.0%}, "
            f"avg {t['avg_r']:+.2f}R (t={t.get('t', 0):+.1f}), total {t['total_r']:+.1f}R, "
            f"maxDD {t['max_dd_r']:.1f}R, PF {t['pf']:.2f}")


# ---- train / promote --------------------------------------------------------
def train(cfg: Config, db: DB) -> dict:
    df = load_labelled(db)
    if len(df) < cfg.learn.min_samples:
        return {"ok": False, "summary": f"Not enough labelled candidates yet ({len(df)} < {cfg.learn.min_samples})."}
    feats = usable_features(df, cfg)
    oos = walk_forward(df, feats, cfg)
    if oos.empty or oos["label"].nunique() < 2:
        return {"ok": False, "summary": "Walk-forward produced no usable out-of-sample predictions."}

    hold_start = int(oos["ts"].max()) - int(cfg.learn.holdout_days * DAY)
    sel, hold = oos[oos["ts"] < hold_start], oos[oos["ts"] >= hold_start]
    if len(sel) < 300 or sel["label"].nunique() < 2:
        sel, hold = oos, oos.iloc[0:0]

    # honest pass: choose the threshold on the selection part, test it on the untouched holdout
    # Thresholds live on the raw ranking score: calibrated probabilities are step-shaped and would
    # make many candidates tie at the threshold.
    main_sel, _ = pick_threshold(threshold_table(sel, "score", cfg), cfg)
    hold_eval = simulate_policy(hold, "score", main_sel["thr"], cfg) if len(hold) and main_sel else None
    # rules benchmark, same protocol
    rules_sel, _ = pick_threshold(threshold_table(sel, "rule_score", cfg), cfg)
    rules_hold = simulate_policy(hold, "rule_score", rules_sel["thr"], cfg) if len(hold) and rules_sel else None

    # production pass: calibrate + threshold on all OOS rows, fit the final model on everything
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oos["score"], oos["label"])
    oos = oos.assign(p=cal.predict(oos["score"]))
    table = threshold_table(oos, "score", cfg)
    main, floor = pick_threshold(table, cfg)
    stricter = [t["thr"] for t in table if main and t["thr"] > main["thr"]]
    ref_ts = int(df["ts"].max())
    model = fit_model(df, feats, cfg, ref_ts)

    top = oos[oos["score"] >= oos["score"].quantile(0.9)]
    bucket = oos["ts"] // XSEC_BUCKET_MS
    s_dm = oos["score"] - oos["score"].groupby(bucket).transform("mean")
    y_dm = oos["label"] - oos["label"].groupby(bucket).transform("mean")
    metrics = {
        "objective": cfg.learn.objective,
        # selection skill with the "kind of day" removed: does it pick the better coin at a given time?
        "xsec_corr": float(np.corrcoef(s_dm, y_dm)[0, 1]) if s_dm.std() > 0 and y_dm.std() > 0 else 0.0,
        "n": int(len(df)), "n_oos": int(len(oos)), "n_live": int((df["source"] == "live").sum()),
        "days": round((df["ts"].max() - df["ts"].min()) / DAY, 1),
        "base_rate": float(df["label"].mean()), "base_avg_r": float(df["ret_r"].mean()),
        "auc_oos": _auc(oos["label"], oos["score"]),
        "auc_holdout": _auc(hold["label"], hold["score"]) if len(hold) else float("nan"),
        "auc_rules": _auc(oos["label"], oos["rule_score"]),
        "top_decile_win": float(top["label"].mean()), "top_decile_avg_r": float(top["ret_r"].mean()),
        "selection_pick": main_sel, "holdout_eval": hold_eval,
        "rules_selection_pick": rules_sel, "rules_holdout_eval": rules_hold,
        "final_pick": main, "final_floor": floor, "table": table,
        "win_r": float(df.loc[df["label"] == 1, "ret_r"].mean()),
        "nontp_avg_r": float(df.loc[df["label"] == 0, "ret_r"].mean()),
        "n_features": len(feats),
    }
    version = datetime.now(timezone.utc).strftime("m%Y%m%d-%H%M%S")
    bundle = {"version": version, "model": model, "features": feats, "calibrator": cal,
              "threshold": main["thr"] if main else float("inf"), "floor": floor["thr"] if floor else float("inf"),
              "threshold_safe": min(stricter) if stricter else (main["thr"] if main else float("inf")),
              "win_r": metrics["win_r"], "nontp_avg_r": metrics["nontp_avg_r"],
              "trained_ts": int(time.time() * 1000), "data_until_ts": ref_ts, "metrics": metrics}
    return {"ok": True, "bundle": bundle, "metrics": metrics, "oos": oos}


def _gates(cfg: Config, m: dict) -> list[str]:
    fails = []
    if cfg.learn.objective == "xsec":
        if not (m["xsec_corr"] >= cfg.learn.min_xsec_corr):
            fails.append(f"selection skill {m['xsec_corr']:+.3f} < {cfg.learn.min_xsec_corr}")
    elif not (m["auc_oos"] >= cfg.learn.min_auc):
        fails.append(f"OOS AUC {m['auc_oos']:.3f} < {cfg.learn.min_auc}")
    pick = m.get("final_pick")
    if not pick or pick["avg_r"] <= 0:
        fails.append("no threshold with positive out-of-sample expectancy")
    elif pick["n"] < cfg.learn.min_oos_trades or pick.get("t", 0.0) < cfg.learn.min_oos_t:
        # the threshold is the best of a dozen tried, so a small positive average is not evidence
        fails.append(f"out-of-sample edge not significant (t={pick.get('t', 0.0):+.1f} over {pick['n']} trades; "
                     f"need t>={cfg.learn.min_oos_t} over >={cfg.learn.min_oos_trades})")
    h = m.get("holdout_eval")
    if not h or h["n"] < 10:
        fails.append("holdout too small to judge")
    elif h["avg_r"] <= 0:
        fails.append(f"holdout expectancy {h['avg_r']:+.2f}R <= 0 over {h['n']} trades")
    p = m.get("paper", {})
    if p.get("n", 0) < cfg.learn.min_paper_signals:
        fails.append(f"forward paper record too short ({p.get('n', 0)}/{cfg.learn.min_paper_signals} closed paper signals)")
    elif p.get("avg_r", 0.0) <= 0:
        fails.append(f"forward paper record negative ({p['avg_r']:+.2f}R over {p['n']} signals)")
    return fails


def load_bundle(cfg: Config, db: DB, champion_only: bool = False) -> dict | None:
    """The validated champion if there is one; otherwise the latest unvalidated 'shadow' model,
    which is only ever used for paper signals."""
    statuses = ("champion",) if champion_only else ("champion", "shadow")
    for status in statuses:
        row = db.query("SELECT path FROM models WHERE status=? AND path IS NOT NULL ORDER BY trained_ts DESC LIMIT 1",
                       (status,))
        if row:
            path = Path(row[0]["path"])
            path = path if path.is_absolute() else cfg.path(str(path))
            if path.exists():
                bundle = joblib.load(path)
                bundle["status"] = status
                return bundle
    return None


def predict(bundle: dict, feat: dict) -> tuple[float, float]:
    """(raw ranking score, calibrated probability that the target is hit first)."""
    x = np.array([[feat.get(n, np.nan) if feat.get(n) is not None else np.nan for n in bundle["features"]]],
                 dtype=np.float64)
    score = float(raw_score(bundle["model"], x)[0])
    return score, float(bundle["calibrator"].predict([score])[0])


def expected_r(bundle: dict | None, p: float, win_r: float, loss_r: float) -> float:
    """Expected R of a trade with TP-probability p. Non-TP outcomes average better than a
    full stop because some trades time out near flat."""
    nontp = bundle["nontp_avg_r"] if bundle else loss_r
    return p * win_r + (1 - p) * max(nontp, loss_r)


def summary_text(m: dict, verdict: str) -> str:
    lines = [
        f"Model training — {verdict}",
        f"Data: {m['n']} labelled candidates over {m['days']} days ({m['n_live']} live), {m['n_features']} features",
        f"Unfiltered base rate: TP-first {m['base_rate']:.1%}, avg {m['base_avg_r']:+.3f}R per candidate",
        f"Out-of-sample AUC: model {m['auc_oos']:.3f} | holdout {m['auc_holdout']:.3f} | rules {m['auc_rules']:.3f}"
        f" | same-moment selection skill {m['xsec_corr']:+.3f}",
        f"Top-decile candidates: TP-first {m['top_decile_win']:.1%}, avg {m['top_decile_avg_r']:+.3f}R",
        f"Threshold picked on earlier folds: {_fmt(m['selection_pick'])}",
        f"  -> same threshold on untouched holdout: {_fmt(m['holdout_eval'])}",
        f"Rules-only benchmark on holdout: {_fmt(m['rules_holdout_eval'])}",
        f"Production threshold (all OOS): {_fmt(m['final_pick'])}",
        f"Grade-B floor: {_fmt(m['final_floor'])}",
        f"Forward record (recent closed signals): {m.get('paper', {}).get('n', 0)} signals, "
        f"avg {m.get('paper', {}).get('avg_r', 0.0):+.2f}R",
    ]
    return "\n".join(lines)


def train_and_maybe_promote(cfg: Config, db: DB, force: bool = False) -> dict:
    res = train(cfg, db)
    if not res["ok"]:
        return res
    bundle, m = res["bundle"], res["metrics"]
    # forward evidence: how the most recent signals (paper or real) actually did
    recent = db.query("SELECT ret_r FROM signals WHERE status IN ('tp','sl','time') AND ret_r IS NOT NULL "
                      "ORDER BY exit_ts DESC LIMIT ?", (max(cfg.learn.min_paper_signals, 1) * 2,))
    m["paper"] = {"n": len(recent), "avg_r": float(np.mean([r["ret_r"] for r in recent])) if recent else 0.0}
    fails = _gates(cfg, m)
    champion = load_bundle(cfg, db, champion_only=True)
    demoted = ""
    if champion and m["paper"]["n"] >= cfg.learn.min_paper_signals and m["paper"]["avg_r"] < cfg.learn.demote_avg_r:
        # the live record is clearly negative: fall back to paper mode until evidence returns
        db.execute("UPDATE models SET status='retired' WHERE status='champion'")
        champion = None
        demoted = (f"\nChampion DEMOTED: recent live record {m['paper']['avg_r']:+.2f}R over {m['paper']['n']} "
                   f"signals. Back to paper signals.")
    if champion and not fails:
        # compare on candidates the champion has never seen (after its training cut-off)
        oos = res["oos"]
        fresh = oos[oos["ts"] > champion.get("data_until_ts", 0)]
        if len(fresh) >= 200 and fresh["label"].nunique() == 2:
            df = load_labelled(db)
            rows = df[df["id"].isin(fresh["id"])]
            cols = [c for c in champion["features"]]
            champ_score = raw_score(champion["model"], rows.reindex(columns=cols).to_numpy())
            champ_auc = _auc(rows["label"], champ_score)
            new_auc = _auc(fresh["label"], fresh["score"])
            m["vs_champion"] = {"n": int(len(fresh)), "champion_auc": champ_auc, "challenger_auc": new_auc}
            if new_auc < champ_auc - 0.005:
                fails.append(f"challenger AUC {new_auc:.3f} below champion {champ_auc:.3f} on {len(fresh)} fresh rows")
    promote = force or not fails
    if promote:
        verdict = "PROMOTED to live signals" + (" (forced; failed: " + "; ".join(fails) + ")" if fails else "")
    elif champion:
        verdict = "NOT PROMOTED, champion kept (" + "; ".join(fails) + ")"
    else:
        verdict = "NOT VALIDATED - paper signals only (" + "; ".join(fails) + ")"
    model_dir = cfg.path(cfg.storage.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    rel = f"{cfg.storage.model_dir}/{bundle['version']}.joblib"
    status = "champion" if promote else "shadow"
    joblib.dump(bundle, cfg.path(rel), compress=3)
    # a new champion replaces everything; a new shadow only replaces the previous shadow
    retire = ("champion", "shadow") if promote else ("shadow",)
    db.execute(f"UPDATE models SET status='retired' WHERE status IN ({','.join('?' * len(retire))})", retire)
    db.execute("INSERT OR REPLACE INTO models(version,trained_ts,n_train,metrics,threshold,floor,path,status) "
               "VALUES(?,?,?,?,?,?,?,?)",
               (bundle["version"], bundle["trained_ts"], m["n"], json.dumps(m, default=float), bundle["threshold"],
                bundle["floor"], rel, status))
    for old in db.query("SELECT version, path FROM models WHERE status='retired' AND path IS NOT NULL "
                        "ORDER BY trained_ts DESC LIMIT -1 OFFSET 5"):
        cfg.path(old["path"]).unlink(missing_ok=True)
        db.execute("UPDATE models SET path=NULL WHERE version=?", (old["version"],))
    res["promoted"] = promote
    res["status"] = status
    res["summary"] = summary_text(m, verdict) + demoted
    log.info("\n%s", res["summary"])
    return res
