

# ---- jobs: nightly study and hourly detection --------------------------------
LIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS grind_live (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL, side INTEGER NOT NULL,
    price REAL, score INTEGER, model_p REAL, catalyst INTEGER,
    checks TEXT, factors TEXT, decision TEXT,
    PRIMARY KEY (symbol, ts, side)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS grind_episodes (
    symbol TEXT NOT NULL, side INTEGER NOT NULL, start INTEGER NOT NULL, end INTEGER, move REAL, hours INTEGER,
    best24 REAL, tags TEXT, PRIMARY KEY (symbol, side, start)
) WITHOUT ROWID;
"""
SHOW_FACTORS = ["r_6h", "r_24h", "r2_6h", "hours_with6", "week_ret", "rs_mkt24", "btc_r168", "dpos_30d", "brk_30d",
                "levels_ahead_10pct", "room_ahead", "imb_6h", "mkt_r6", "rvol_6h", "funding", "oi_chg_24h",
                "listing_age_d", "sector_r24", "atr_1h"]


def bundle_path(cfg) -> str:
    return str(cfg.path(cfg.storage.model_dir) / "grindlab.joblib")


def load_bundle(cfg) -> dict | None:
    import joblib
    from pathlib import Path
    p = Path(bundle_path(cfg))
    return joblib.load(p) if p.exists() else None


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def nightly(cfg, db: DB) -> dict:
    """Refresh history, rebuild the study, refit the model, store the report. Run in a worker thread."""
    import asyncio
    import joblib
    from .grindlab_data import refresh
    db.conn.executescript(LIVE_SCHEMA)
    asyncio.run(refresh(db, hourly_days=cfg.grindlab.days + 5, funding_days=cfg.grindlab.days + 5))
    df, ep = build(db, days=cfg.grindlab.days)
    if len(df) < 500:
        rep = {"built_ts": int(time.time() * 1000), "n_decisions": int(len(df)), "insufficient": True}
        db.set("grindlab_report", rep)
        return rep
    rep = study(df, ep)
    feats = usable(df, model_features(df))
    bundle = {"model": fit_final(df, feats), "features": feats, "thresholds": rep["thresholds"],
              "built_ts": rep["built_ts"], "n": int(len(df))}
    cfg.path(cfg.storage.model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path(cfg), compress=3)
    db.execute("DELETE FROM grind_episodes")
    recent = ep[ep["start"] >= int(time.time() * 1000) - 14 * DAY] if len(ep) else ep
    db.insert_rows("grind_episodes", ("symbol", "side", "start", "end", "move", "hours", "best24", "tags"),
                   [tuple(_clean(r[c]) for c in ("symbol", "side", "start", "end", "move", "hours", "best24", "tags"))
                    for r in recent.to_dict("records")])
    db.set("grindlab_report", _clean(rep))
    return rep


def nightly_text(rep: dict) -> str:
    from .telegram import esc
    if rep.get("insufficient"):
        return f"<b>Grind lab</b>\nNot enough history yet ({rep.get('n_decisions', 0)} early-grind moments)."
    held = [t for t in rep["traits"] if t["held"]]
    cat = rep["catalyst"].get("short", {})
    lines = [f"<b>Grind lab (nightly)</b> — {rep['n_episodes']} steady grinds in {rep['days']:.0f} days "
             f"({rep['episodes_per_day']:.0f}/day); {rep['n_decisions']} early-grind moments, "
             f"{rep['base_success']:.0%} went on to +{rep['target']:.0%} before -{rep['stop']:.0%}.",
             f"Traits of successful grinders that held on unseen days: {len(held)} of {len(rep['traits'])}"]
    for t in sorted(held, key=lambda t: (t["test_with"] or 0) - (t["test_without"] or 0), reverse=True)[:5]:
        lines.append(f"  • {esc(t['trait'])}: {t['test_with']:.0%} vs {t['test_without']:.0%}")
    if cat.get("n"):
        lines.append(f"News catalyst (down-grinds after Upbit listing / futures launch): {cat['n']} cases, "
                     f"{cat['success']:.0%} success, {cat.get('x_0.06_0.03', 0) * 100:+.2f}% per trade")
    m = rep.get("model", {})
    if m.get("ok"):
        lines.append(f"Model on unseen days: top 10% succeed {m['top10']['success']:.0%} vs {m['base_success']:.0%} base, "
                     f"{m['top10']['avg_r']:+.2f}R per trade")
    return "\n".join(lines)


def hourly(cfg, db: DB) -> list[dict]:
    """Refresh the latest hourly bars, detect early grinds, score them, record them. Run in a worker thread."""
    import asyncio
    from .grindlab_data import refresh_recent
    db.conn.executescript(LIVE_SCHEMA)
    syms = [r["symbol"] for r in db.query("SELECT symbol FROM symbols WHERE active=1 AND src='binance'")]
    asyncio.run(refresh_recent(db, syms + ["BTCUSDT"]))
    rows = score_live(live_scan(db), load_bundle(cfg))
    for r in rows:
        r["catalyst"] = int(r.get("catalyst_short") or 0)
        db.execute("INSERT OR REPLACE INTO grind_live(symbol, ts, side, price, score, model_p, catalyst, checks, factors, decision) "
                   "VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (r["symbol"], r["ts"], r["side"], r["price"], r["score"], r["model_p"], r["catalyst"],
                    json.dumps(r["checks"]), json.dumps(_clean({k: r.get(k) for k in SHOW_FACTORS})), None))
    return rows
