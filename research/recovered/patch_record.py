import pathlib, re

ROOT = pathlib.Path(r"D:\Leverage trading")


def sub(path, pairs, count=1):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:80])
        s = s.replace(a, b, count)
    p.write_text(s, encoding="utf-8")


# ---- config: 10x plan, top 20%, research grid ----------------------------------------------------
sub("lts/config.py", [
    ('''    plans: list = field(default_factory=lambda: [
        {"leverage": 5, "target": 0.02, "stop": 0.01},''',
     '''    plans: list = field(default_factory=lambda: [
        {"leverage": 10, "target": 0.01, "stop": 0.0075},
        {"leverage": 5, "target": 0.02, "stop": 0.01},'''),
    ('''    primary: int = 2                     # index of the tracked plan (2x +5% / -0.75% held up best on unseen months)
    top_share: float = 0.05              # send the best-scoring 5% of grinds (~7-8 a day)''',
     '''    primary: int = 3                     # index of the tracked plan (2x +5% / -0.75% held up best on unseen months)
    top_share: float = 0.20              # send the best-scoring 20% of grinds (~30 a day) - trader's choice
    # every detected grind is labelled after 24h on this grid too (targets x stops), for later tuning
    record_targets: list = field(default_factory=lambda: [0.01, 0.015, 0.02, 0.0333, 0.05])
    record_stops: list = field(default_factory=lambda: [0.003, 0.005, 0.0075, 0.01, 0.015, 0.02])'''),
])

# ---- grind_live: richer record + migration -------------------------------------------------------
sub("lts/grindlab.py", [
    ('''    checks TEXT, factors TEXT, decision TEXT,
    PRIMARY KEY (symbol, ts, side)
) WITHOUT ROWID;''',
     '''    checks TEXT, factors TEXT, decision TEXT,
    all_factors TEXT,                  -- every factor + minute-shape feature known at detection (JSON)
    micro TEXT,                        -- liquidations / order book around detection (JSON)
    threshold REAL,                    -- the model score needed to be sent at the time
    outcomes TEXT,                     -- after 24h: every plan + a target/stop grid + best/worst moves (JSON)
    outcome_ts INTEGER,
    PRIMARY KEY (symbol, ts, side)
) WITHOUT ROWID;'''),
    ('''    db.conn.executescript(LIVE_SCHEMA)
    syms = [r["symbol"] for r in db.query("SELECT symbol FROM symbols WHERE active=1 AND src='binance'")]''',
     '''    ensure_live_schema(db)
    syms = [r["symbol"] for r in db.query("SELECT symbol FROM symbols WHERE active=1 AND src='binance'")]'''),
    ('''def bundle_path(cfg) -> str:''',
     '''def ensure_live_schema(db: DB) -> None:
    db.conn.executescript(LIVE_SCHEMA)
    have = {r[1] for r in db.conn.execute("PRAGMA table_info(grind_live)")}
    for col, typ in (("all_factors", "TEXT"), ("micro", "TEXT"), ("threshold", "REAL"), ("outcomes", "TEXT"),
                     ("outcome_ts", "INTEGER")):
        if col not in have:
            db.conn.execute(f"ALTER TABLE grind_live ADD COLUMN {col} {typ}")
    db.conn.commit()


def bundle_path(cfg) -> str:'''),
])

# ---- grindsignal: record every grind + label it after 24h --------------------------------------
p = ROOT / "lts/grindsignal.py"
s = p.read_text(encoding="utf-8")
s += '''

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


def weex_url(symbol: str) -> str:
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    return f"https://www.weex.com/futures/{base}-USDT"
'''
p.write_text(s, encoding="utf-8")

# ---- service: record after scoring, label hourly ------------------------------------------------
sub("lts/service.py", [
    ('''        pres = [GS.pre_from_db(self.db, r["symbol"], r["ts"]) for r in rows]
        GS.score_rows(rows, pres, self.gs_bundle)
        p_min = GS.threshold(self.gs_bundle, g.top_share)''',
     '''        pres = [GS.pre_from_db(self.db, r["symbol"], r["ts"]) for r in rows]
        GS.score_rows(rows, pres, self.gs_bundle)
        p_min = GS.threshold(self.gs_bundle, g.top_share)
        GS.record_grinds(self.db, rows, p_min)'''),
    ('''                    from .grindsignal import label_signal_plans
                    label_signal_plans(self.cfg, self.db)''',
     '''                    from .grindsignal import label_grinds, label_signal_plans
                    label_signal_plans(self.cfg, self.db)
                    label_grinds(self.cfg, self.db)'''),
])

# ---- Telegram: Weex chart link -------------------------------------------------------------------
sub("lts/signals.py", [
    ('''        f = c.get("features", {})
        lines.append(f"6h {f.get('r_6h', 0) * 100:+.1f}% · 24h {f.get('r_24h', 0) * 100:+.1f}% · "
                     f"steadiness R² {f.get('sh_r2_6h', float('nan')):.2f}")''',
     '''        f = c.get("features", {})
        lines.append(f"6h {f.get('r_6h', 0) * 100:+.1f}% · 24h {f.get('r_24h', 0) * 100:+.1f}% · "
                     f"steadiness R² {f.get('sh_r2_6h', float('nan')):.2f}")
        from .grindsignal import weex_url
        lines.append(f'<a href="{weex_url(sig["symbol"])}">📈 Open {esc(sig["symbol"])} on Weex</a>')'''),
])

# ---- dashboard data: every grind of the last 3 days with outcomes --------------------------------
sub("lts/dashboard.py", [
    ('''        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='grind_episodes'"):''',
     '''        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='grind_live'"):
            cols_gl = {r["name"] for r in q("PRAGMA table_info(grind_live)")}
            if "outcomes" in cols_gl:
                allg = q("SELECT symbol, ts, side, model_p, threshold, decision, outcomes FROM grind_live "
                         "WHERE ts>=? ORDER BY ts DESC LIMIT 2500", (now - 3 * DAY,))
                for r in allg:
                    o = json.loads(r.pop("outcomes") or "null") or {}
                    r["plans"] = o.get("plans")
                    r["best24"] = (o.get("best") or {}).get("24")
                    r["worst24"] = (o.get("worst") or {}).get("24")
                out["grind_all"] = allg
                out["grinds_recorded"] = q("SELECT COUNT(*) n, SUM(outcomes IS NOT NULL) labelled FROM grind_live")[0]
        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='grind_episodes'"):'''),
])

# ---- dashboard page -----------------------------------------------------------------------------
p = ROOT / "lts/web/index.html"
s = p.read_text(encoding="utf-8")
s = s.replace('`<b>${esc(s.symbol)}</b>`', 'coin(s.symbol)')
s = s.replace('`<b>${esc(c.symbol)}</b>`', 'coin(c.symbol)')
s = s.replace('`<b>${esc(m.symbol)}</b>`', 'coin(m.symbol)')
s = s.replace('`<b>${esc(r.symbol)}</b>${r.catalyst', '`${coin(r.symbol)}${r.catalyst')
s = s.replace('`<b>${esc(e.symbol)}</b>`', 'coin(e.symbol)')
assert s.count("coin(") >= 7, s.count("coin(")
s = s.replace('''const side = s =>''', '''const weexUrl = sym => `https://www.weex.com/futures/${String(sym).replace(/USDT$/, "")}-USDT`;
const coin = sym => `<a href="${weexUrl(sym)}" target="_blank" rel="noopener noreferrer" title="Open the Weex chart"><b>${esc(sym)}</b> ↗</a>`;
const side = s =>''', 1)
s = s.replace('''    <h2>Grinding right now (last 48h)</h2><div class="panel scroll" id="grindLive"></div>''',
              '''    <h2 id="allTitle">Every grind detected (last 3 days) — sent or not, with how each plan did after 24h</h2>
    <div class="panel scroll" id="grindAll"></div>
    <h2>Grinding right now (last 48h) — traits</h2><div class="panel scroll" id="grindLive"></div>''')
s = s.replace('''  const gsig = (d.signals || []).filter(s => s.kind === "grind" || s.kind === "grindsig");''',
              '''  const gsig = (d.signals || []).filter(s => s.kind === "grind" || s.kind === "grindsig");
  const ga = d.grind_all || [];
  const planNames = ga.find(r => r.plans) ? Object.keys(ga.find(r => r.plans).plans) : ["10x", "5x", "3x", "2x"];
  const planCell = (r, name) => !r.plans ? '<span class="muted">pending</span>' :
    (r.plans[name] > 0 ? `<span class="pos">✓ ${signed(r.plans[name] * 100, 2, "%")}</span>` : `<span class="neg">✗ ${signed(r.plans[name] * 100, 2, "%")}</span>`);
  const gr = d.grinds_recorded || {};
  $("allTitle").textContent = `Every grind detected (last 3 days) — ${ga.length} shown; ${(gr.n || 0).toLocaleString()} recorded in total, ${(gr.labelled || 0).toLocaleString()} with 24h outcomes`;
  table($("grindAll"), [["Time"],["Coin"],["Side"],["Model score","num"],["Needed","num"],["Sent?"]].concat(planNames.map(n => [esc(n), "num"])).concat([["Best 24h","num"],["Worst 24h","num"]]),
    ga.map(r => [dt(r.ts), coin(r.symbol), side(r.side), r.model_p == null ? "–" : num(r.model_p, 3), r.threshold == null ? "–" : num(r.threshold, 3),
      r.decision === "sent" ? '<span class="chip accent">sent</span>' : '<span class="muted">' + esc((r.decision || "–").replace("skipped:", "").split(" (")[0]) + "</span>"]
      .concat(planNames.map(n => planCell(r, n))).concat([r.best24 == null ? "–" : signed(r.best24 * 100, 1, "%"), r.worst24 == null ? "–" : signed(-r.worst24 * 100, 1, "%")])),
    "No grinds recorded yet (the scan runs 3 minutes past every hour).");''')
p.write_text(s, encoding="utf-8")
print("recording, links and dashboard wired")
