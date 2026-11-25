def sub(p, old, new, n=1):
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == n, (p, old[:80], s.count(old))
    open(p, "w", encoding="utf-8").write(s.replace(old, new))


p = "lts/grindlab.py"
sub(p, "    selected INTEGER,                  -- met the sending rule at this record (steady all-day grind, or model top share)\n",
    "    selected INTEGER,                  -- met the sending rule at this record (steady all-day grind, or model top share)\n"
    "    exits TEXT,                        -- tracked plan under each shadow early-exit rule (lts/exits.py)\n")
sub(p, '("selected", "INTEGER")):', '("selected", "INTEGER"), ("exits", "TEXT")):')

p = "lts/grindsignal.py"
sub(p, "def label_after_exit(cfg, db: DB, now: int | None = None) -> int:", '''def label_exits(cfg, db: DB, now: int | None = None, max_rows: int = 600) -> int:
    """24h after each recorded grind: the tracked plan under every shadow early-exit rule (lts/exits.py)."""
    import json
    from .exits import shadow
    from .grindlab import ensure_live_schema
    ensure_live_schema(db)
    gs = cfg.grindsignal
    pl = gs.plans[gs.primary]
    now = now or int(time.time() * 1000)
    rows = db.query("SELECT symbol, ts, side FROM grind_live WHERE exits IS NULL AND ts <= ? AND ts > ? ORDER BY ts LIMIT ?",
                    (now - gs.hours * HOUR - 10 * 60_000, now - 5 * DAY, max_rows))
    n = 0
    for r in rows:
        ts = int(r["ts"])
        k = db.frame("SELECT ts,o,h,l,c,qv,tbqv FROM klines WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts",
                     [r["symbol"], ts - BARS * BAR, ts + gs.hours * HOUR])
        if not len(k):
            continue
        pre = k[k.ts < ts]["c"].to_numpy(float)
        post = np.full((gs.hours * 12, 6), np.nan)
        kp = k[k.ts >= ts]
        idx = ((kp["ts"].to_numpy() - ts) // BAR).astype(int)
        post[idx] = kp[["o", "h", "l", "c", "qv", "tbqv"]].to_numpy(float)
        res = shadow(post, pre, r["side"], pl["target"], pl["stop"], gs.hours)
        db.execute("UPDATE grind_live SET exits=? WHERE symbol=? AND ts=? AND side=?",
                   (json.dumps(_clean(res)) if res else '{"missing": true}', r["symbol"], r["ts"], r["side"]))
        n += 1
    return n


def label_after_exit(cfg, db: DB, now: int | None = None) -> int:''')

p = "lts/service.py"
sub(p, '''                    from .grindsignal import (label_after_exit, label_anatomy, label_grinds, label_signal_plans,
                                              record_signal_paths)''', '''                    from .grindsignal import (label_after_exit, label_anatomy, label_exits, label_grinds,
                                              label_signal_plans, record_signal_paths)''')
sub(p, '''                    try:
                        label_anatomy(self.cfg, self.db)
                    except Exception:
                        log.exception("trade anatomy failed")''', '''                    try:
                        label_anatomy(self.cfg, self.db)
                        label_exits(self.cfg, self.db)
                    except Exception:
                        log.exception("trade anatomy / shadow exits failed")''')

p = "lts/dashboard.py"
sub(p, '''                if "anatomy" in cols_gl:''', '''                if "exits" in cols_gl:
                    out["exits_live"] = exits_summary(q(
                        "SELECT exits, COALESCE(selected, 0) AS sel" +
                        ("".join(f", json_extract(all_factors, '$.{k}') AS f_{k}" for k in STEADY_KEYS) if "all_factors" in cols_gl else "") +
                        " FROM grind_live WHERE exits IS NOT NULL AND ts>=?", (now - 14 * DAY,)))
                if "anatomy" in cols_gl:''')
sub(p, '''def anatomy_summary(rows: list[dict]) -> dict:''', '''def exits_summary(rows: list[dict]) -> list[dict]:
    """Per shadow exit rule: the tracked plan's live record, for steady-rule grinds and for all grinds."""
    from .exits import ENTRY, MKT, RULES
    from .grindsignal import steady
    gs = Config().grindsignal
    full = -gs.plans[gs.primary]["stop"] - ENTRY - MKT + 1e-9
    acc = {r: {"all": [], "steady": []} for r in RULES}
    for r in rows:
        e = json.loads(r["exits"] or "null") or {}
        if not e or e.get("missing"):
            continue
        f = {k: r.get(f"f_{k}") for k in STEADY_KEYS}
        is_steady = bool(r.get("sel")) if gs.selector != "steady" else (bool(r.get("sel")) or steady(f, gs))
        for rule, v in e.items():
            if rule in acc:
                acc[rule]["all"].append(v["net"])
                if is_steady:
                    acc[rule]["steady"].append(v["net"])
    out = []
    for rule, desc in RULES.items():
        row = {"rule": rule, "desc": desc}
        for g in ("all", "steady"):
            v = acc[rule][g]
            row[g] = {"n": len(v), "won": sum(1 for x in v if x > 0), "full": sum(1 for x in v if x <= full),
                      "net": (sum(v) / len(v)) if v else None}
        out.append(row)
    return out


def anatomy_summary(rows: list[dict]) -> dict:''')

p = "lts/web/index.html"
sub(p, '''    <div class="panel scroll" id="anatRev" style="margin-top:12px"></div>''', '''    <div class="panel scroll" id="anatRev" style="margin-top:12px"></div>
    <h2>Early exits, live (shadow only — they never close a signal)</h2>
    <p class="muted">Each grind's 5x plan replayed with one early-exit rule, 24h after it was detected (last 14 days). On the past
    year every rule cut full stops but none improved the average per trade. This table shows whether live trades agree.</p>
    <div class="panel scroll" id="exitsLive"></div>''')
sub(p, '''  const lead = v => v == null''', '''  const exCell = s => !s || !s.n ? "–" : `${s.won}/${s.n} won · ${Math.round(100 * s.full / s.n)}% full stops · ${signed(s.net * 100, 2, "%")}`;
  table($("exitsLive"), [["Exit rule"],["All grinds (5x)"],["Steady all-day grinds (5x)"]],
    (d.exits_live || []).map(r => [esc(r.desc), exCell(r.all), exCell(r.steady)]), "Filled in 24h after each grind.");
  const lead = v => v == null''')
print("ok")
