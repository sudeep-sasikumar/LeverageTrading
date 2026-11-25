def sub(p, old, new, n=1):
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == n, (p, old[:70], s.count(old))
    open(p, "w", encoding="utf-8").write(s.replace(old, new))


# ---- config ------------------------------------------------------------------
sub("lts/config.py", "    scan_delay_s: int = 60               # grind scan runs this long after each 5-minute candle closes\n",
    "    scan_delay_s: int = 60               # grind scan runs this long after each 5-minute candle closes\n"
    "    after_hours: int = 24                # price is followed this long after each plan's exit (target, stop or time)\n"
    "    skip_if_level_blocks: bool = False   # skip a signal when a daily/weekly/monthly level sits between entry and\n"
    "                                         # target. Off: on the year it did not help (DESIGN.md section 23); the\n"
    "                                         # levels are shown on every signal and recorded on every grind\n")

# ---- grindsignal: after-exit recording + signal paths -------------------------
p = "lts/grindsignal.py"
sub(p, '''PLAN_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_plans (
    signal_id INTEGER NOT NULL, plan TEXT NOT NULL, leverage INTEGER, target REAL, stop REAL,
    outcome TEXT, net REAL, margin REAL, done_ts INTEGER,
    PRIMARY KEY (signal_id, plan)
) WITHOUT ROWID;
"""''', '''PLAN_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_plans (
    signal_id INTEGER NOT NULL, plan TEXT NOT NULL, leverage INTEGER, target REAL, stop REAL,
    outcome TEXT, net REAL, margin REAL, done_ts INTEGER,
    PRIMARY KEY (signal_id, plan)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS signal_paths (
    signal_id INTEGER PRIMARY KEY, symbol TEXT, side INTEGER, entry REAL,
    start_ts INTEGER,                  -- first 5m bar stored (6h before the signal)
    bars TEXT,                         -- JSON [[o,h,l,c], ...] 5-minute candles: 6h before to 48h after the signal
    after TEXT,                        -- JSON per plan: exit and how price moved in the 24h after it
    done_ts INTEGER
);
"""
PRE_H = 6


def after_exit(post: np.ndarray, side: int, plans: list[dict], hours: int, after_h: int) -> dict:
    """For each plan: how it exited (target / stop / time, minutes after entry) and what price did in the
    after_h hours that followed, measured from the exit price in the trade's direction.
    post: 5m o,h,l,c from entry, at least hours*12 + after_h*12 bars."""
    out = {}
    e = float(post[0, 0])
    for pl in plans:
        old = (A.TARGETS, A.STOPS, A.HOURS)
        try:
            A.TARGETS, A.STOPS, A.HOURS = (pl["target"],), (pl["stop"],), (hours,)
            net, xb = A.outcomes(post[None, :hours * 12 + 1], np.array([side]))[(pl["target"], pl["stop"], hours)]
        finally:
            A.TARGETS, A.STOPS, A.HOURS = old
        xb = int(xb[0])
        kind = "target" if net[0] >= pl["target"] - 0.003 else ("stop" if net[0] <= -pl["stop"] else "time")
        x_px = e * (1 + side * pl["target"]) if kind == "target" else (e * (1 - side * pl["stop"]) if kind == "stop"
                                                                        else float(post[xb, 3]))
        rest = post[xb + 1: xb + 1 + after_h * 12]
        rest = rest[np.isfinite(rest[:, 1])]
        rec = {"exit": kind, "exit_min": (xb + 1) * 5, "exit_price": x_px}
        if len(rest):
            fav = (rest[:, 1] / x_px - 1) if side > 0 else (1 - rest[:, 2] / x_px)
            adv = (1 - rest[:, 2] / x_px) if side > 0 else (rest[:, 1] / x_px - 1)
            cls = side * (rest[:, 3] / x_px - 1)
            for h in (1, 4, after_h):
                n = min(h * 12, len(rest))
                rec[f"best_{h}h"], rec[f"worst_{h}h"], rec[f"close_{h}h"] = float(fav[:n].max()), float(adv[:n].max()), float(cls[n - 1])
            tgt_px = e * (1 + side * pl["target"])
            hit = (rest[:, 1] >= tgt_px) if side > 0 else (rest[:, 2] <= tgt_px)
            rec["target_after_stop"] = bool(kind == "stop" and hit.any())       # stopped, then went on to the target
            rec["bars_after"] = int(len(rest))
        out[plan_name(pl)] = rec
    return out


def record_signal_paths(cfg, db: DB, now: int | None = None) -> int:
    """For every grind signal 48h old: store its 5-minute path (6h before to 48h after) and each plan's after-exit moves."""
    import json
    db.conn.executescript(PLAN_SCHEMA)
    gs = cfg.grindsignal
    now = now or int(time.time() * 1000)
    span = gs.hours + gs.after_hours
    rows = db.query("""SELECT s.id, s.symbol, s.side, s.ts, s.entry FROM signals s
                       WHERE s.kind='grindsig' AND s.ts <= ?
                       AND NOT EXISTS (SELECT 1 FROM signal_paths p WHERE p.signal_id = s.id)""",
                    (now - span * HOUR - 10 * 60_000,))
    n = 0
    for r in rows:
        start = (r["ts"] // BAR + 1) * BAR
        k = db.frame("SELECT ts,o,h,l,c FROM klines WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts",
                     [r["symbol"], start - PRE_H * HOUR, start + span * HOUR])
        post = post_from_db(db, r["symbol"], start, span * 12 + 1)
        if post is None:
            if now - r["ts"] > (span + 48) * HOUR:              # data never came: store what there is
                db.execute("INSERT OR REPLACE INTO signal_paths VALUES (?,?,?,?,?,?,?,?)",
                           (r["id"], r["symbol"], r["side"], r["entry"], start - PRE_H * HOUR, None, '{"missing": true}', now))
            continue
        post = post.copy()
        post[0, 0] = r["entry"]
        after = after_exit(post, r["side"], gs.plans, gs.hours, gs.after_hours)
        bars = [[round(float(x), 10) for x in row] for row in k[["o", "h", "l", "c"]].to_numpy()]
        db.execute("INSERT OR REPLACE INTO signal_paths VALUES (?,?,?,?,?,?,?,?)",
                   (r["id"], r["symbol"], r["side"], r["entry"], int(k["ts"].iloc[0]) if len(k) else start,
                    json.dumps(bars), json.dumps(_clean(after)), now))
        n += 1
    return n


def label_after_exit(cfg, db: DB, now: int | None = None) -> int:
    """Every recorded grind, (hours + after_hours) after detection: each plan's exit and the 24h that followed."""
    import json
    from .grindlab import ensure_live_schema
    ensure_live_schema(db)
    gs = cfg.grindsignal
    now = now or int(time.time() * 1000)
    span = gs.hours + gs.after_hours
    rows = db.query("SELECT symbol, ts, side FROM grind_live WHERE after_exit IS NULL AND outcomes IS NOT NULL AND ts <= ?",
                    (now - span * HOUR - 10 * 60_000,))
    n = 0
    for r in rows:
        post = post_from_db(db, r["symbol"], int(r["ts"]), span * 12 + 1)
        if post is None:
            if now - r["ts"] > (span + 48) * HOUR:
                db.execute("UPDATE grind_live SET after_exit=? WHERE symbol=? AND ts=? AND side=?",
                           ('{"missing": true}', r["symbol"], r["ts"], r["side"]))
            continue
        out = after_exit(post, r["side"], gs.plans, gs.hours, gs.after_hours)
        db.execute("UPDATE grind_live SET after_exit=? WHERE symbol=? AND ts=? AND side=?",
                   (json.dumps(_clean(out)), r["symbol"], r["ts"], r["side"]))
        n += 1
    return n''')

# ---- grind_live column ----------------------------------------------------------
p = "lts/grindlab.py"
sub(p, "    progress TEXT,                     -- under 24h: each plan once decided, best/worst so far (JSON)\n",
    "    progress TEXT,                     -- under 24h: each plan once decided, best/worst so far (JSON)\n"
    "    after_exit TEXT,                   -- 48h: each plan's exit and how price moved in the 24h after it (JSON)\n"
    "    levels TEXT,                       -- daily/weekly/monthly levels between entry and target / stop (JSON)\n")
sub(p, '("outcome_ts", "INTEGER"), ("progress", "TEXT")):', '("outcome_ts", "INTEGER"), ("progress", "TEXT"),\n                     ("after_exit", "TEXT"), ("levels", "TEXT")):')

# ---- service: run them hourly -----------------------------------------------------
p = "lts/service.py"
sub(p, '''                    from .grindsignal import label_grinds, label_signal_plans
                    label_signal_plans(self.cfg, self.db)
                    label_grinds(self.cfg, self.db)''', '''                    from .grindsignal import label_after_exit, label_grinds, label_signal_plans, record_signal_paths
                    label_signal_plans(self.cfg, self.db)
                    label_grinds(self.cfg, self.db)
                    label_after_exit(self.cfg, self.db)
                    record_signal_paths(self.cfg, self.db)''')
print("ok")
