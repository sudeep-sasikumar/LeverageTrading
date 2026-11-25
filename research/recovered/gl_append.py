

# ---- checklist: the stable traits of successful grinders ---------------------
# (factor, direction, label). Thresholds are learned nightly from the data (training part only).
CHECKS = [
    ("btc_r168", "pos", "BTC's week is going the same way"),
    ("brk_30d", "brk", "breaking out of / at the edge of its 30-day range"),
    ("week_ret", "pos", "the coin's own week is going the same way"),
    ("rs_mkt24", "pos", "stronger than the market over 24h"),
    ("imb_6h", "low", "no heavy aggressive buying/selling in the last 6h"),
    ("mkt_r6", "low", "the move is coin-specific (market not moving the same way)"),
    ("levels_ahead_10pct", "zero", "no daily support/resistance within 10% ahead"),
    ("sector_r24", "pos", "its sector is moving the same way"),
    ("funding", "nonneg", "funding is not against the trade"),
    ("listing_age_d", "low", "younger listing"),
    ("rvol_6h", "high", "volume above normal"),
    ("catalyst_short", "one", "down-grind after an Upbit listing / Binance futures launch"),
]


def learn_thresholds(train: pd.DataFrame) -> dict:
    return {c: float(train[c].median()) for c, mode, _ in CHECKS if mode in ("low", "high") and c in train}


def checklist(d: pd.DataFrame, thr: dict) -> pd.DataFrame:
    out = {}
    nan = pd.Series(np.nan, index=d.index)
    for col, mode, label in CHECKS:
        x = d[col] if col in d else nan
        if col == "brk_30d":
            ok = (x == 1) | ((d["pos_30d"] if "pos_30d" in d else nan) >= 0.8)
        elif mode == "pos":
            ok = x.fillna(0) > 0
        elif mode == "low":
            ok = x <= thr.get(col, np.inf)
        elif mode == "high":
            ok = x >= thr.get(col, -np.inf)
        elif mode == "zero":
            ok = x == 0
        elif mode == "nonneg":
            ok = x.fillna(0) >= 0
        else:
            ok = x == 1
        out[label] = ok.fillna(False).astype(bool)
    return pd.DataFrame(out, index=d.index)


EXIT_LABELS = {"x_0.06_0.03": "+6% / -3%", "x_0.1_0.05": "+10% / -5%", "x_0.04_0.02": "+4% / -2%", "x_trail": "trailing 3%"}


def _exit_stats(g: pd.DataFrame) -> dict:
    out = {"n": int(len(g)), "success": float(g.success.mean()) if len(g) else None,
           "best_med": float(g.mfe.median()) if len(g) else None, "worst_med": float(g.mae.median()) if len(g) else None}
    for col in EXIT_LABELS:
        if col in g and len(g):
            x = g[col]
            out[col] = float(x.mean())
            out[col + "_t"] = float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else 0.0
    return out


def study(df: pd.DataFrame, ep: pd.DataFrame) -> dict:
    """The nightly report: what successful grinders share, and whether it held on later, unseen days."""
    df = df.sort_values("ts").reset_index(drop=True)
    cut = df.ts.quantile(0.6)
    train, test = df[df.ts < cut - DAY], df[df.ts >= cut]
    thr = learn_thresholds(train)
    ck_tr, ck_te = checklist(train, thr), checklist(test, thr)
    traits = []
    for label in ck_tr.columns:
        a, b = train[ck_tr[label]], train[~ck_tr[label]]
        c, e = test[ck_te[label]], test[~ck_te[label]]
        ok = len(a) and len(b) and len(c) and len(e)
        traits.append({"trait": label,
                       "train_with": float(a.success.mean()) if len(a) else None,
                       "train_without": float(b.success.mean()) if len(b) else None,
                       "test_with": float(c.success.mean()) if len(c) else None,
                       "test_without": float(e.success.mean()) if len(e) else None,
                       "n_test_with": int(len(c)),
                       "held": bool(ok and (a.success.mean() - b.success.mean()) * (c.success.mean() - e.success.mean()) > 0)})
    score_te = ck_te.sum(axis=1)
    by_score = [{"score": int(s_), **_exit_stats(g)} for s_, g in test.groupby(np.minimum(score_te, 9))]
    cat = {}
    for side, name in ((-1, "short"), (1, "long")):
        news = (df["ann_upbit_list_7d"] == 1) | (df["ann_futures_launch_7d"] == 1)
        g = df[news & (df.side == side)]
        cat[name] = {**_exit_stats(g), "coins": int(g.symbol.nunique()) if len(g) else 0}
    ft = factor_table(df, n_folds=4)
    stable = ft[ft.stable & (ft.z.abs() >= 3)].head(25)
    wf = walk_forward(df, model_features(df))
    days = (df.ts.max() - df.ts.min()) / DAY
    return {
        "built_ts": int(time.time() * 1000), "days": float(days), "n_decisions": int(len(df)),
        "decisions_per_day": float(len(df) / max(days, 1)), "base_success": float(df.success.mean()),
        "target": TARGET, "stop": STOP, "horizon_h": HORIZON_H,
        "n_episodes": int(len(ep)), "episodes_per_day": float(len(ep) / max(days, 1)),
        "episodes_up": int((ep.side > 0).sum()) if len(ep) else 0,
        "episodes_down": int((ep.side < 0).sum()) if len(ep) else 0,
        "episode_median_move": float(ep.move.abs().median()) if len(ep) else None,
        "thresholds": thr, "traits": traits, "by_score": by_score, "catalyst": cat,
        "test_from_ts": int(cut), "base_test": _exit_stats(test),
        "stable_factors": stable[["factor", "success_high", "success_low", "lift", "z", "folds_agree",
                                  "median_success", "median_fail"]].to_dict("records"),
        "model": {k: v for k, v in wf.items() if k != "oos"} if wf.get("ok") else {"ok": False},
    }


# ---- live detection ----------------------------------------------------------
def live_scan(db: DB, ctx: dict | None = None, now_ms: int | None = None) -> list[dict]:
    """Early-grind detections on the latest closed hourly bar, with all factors (same code as the study)."""
    now_ms = now_ms or int(time.time() * 1000)
    since = now_ms - 40 * DAY
    ctx = ctx or load_context(db, since)
    last_bar = (now_ms // HOUR) * HOUR - HOUR
    out = []
    for sym in ctx["syms"]:
        k = _hourly(db, sym, since)
        if k is None or k.index[-1] != last_bar:
            continue
        f = hourly_shape(k)
        i = len(k) - 1
        for side in (1, -1):
            if early_grind_mask(f.iloc[[i]], side)[0]:
                out.append(decision_row(sym, side, i, k, f, _daily(db, sym), ctx))
    return out


def score_live(rows: list[dict], bundle: dict | None) -> list[dict]:
    if not rows:
        return rows
    d = pd.DataFrame(rows)
    ck = checklist(d, (bundle or {}).get("thresholds", {}))
    d["score"] = ck.sum(axis=1)
    if bundle and bundle.get("model") is not None:
        d["model_p"] = bundle["model"].predict_proba(d.reindex(columns=bundle["features"]).to_numpy(float))[:, 1]
    else:
        d["model_p"] = np.nan
    for i, r in enumerate(rows):
        r["score"] = int(d["score"].iloc[i])
        r["model_p"] = None if np.isnan(d["model_p"].iloc[i]) else float(d["model_p"].iloc[i])
        r["checks"] = {k: bool(v) for k, v in ck.iloc[i].items()}
    return rows
