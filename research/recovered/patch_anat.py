def sub(p, old, new, n=1):
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == n, (p, old[:70], s.count(old))
    open(p, "w", encoding="utf-8").write(s.replace(old, new))


sub("lts/reversal.py", '''            "most_candles_against_in_a_row": int(best_run),
            "trades_rvol": None}''', '''            "most_candles_against_in_a_row": int(best_run)}''')

p = "lts/grindlab.py"
sub(p, "    levels TEXT,                       -- daily/weekly/monthly levels between entry and target / stop (JSON)\n",
    "    levels TEXT,                       -- daily/weekly/monthly levels between entry and target / stop (JSON)\n"
    "    anatomy TEXT,                      -- tracked plan: entry -> peak -> exit, what changed at the turn (lts/reversal.py)\n")
sub(p, '("after_exit", "TEXT"), ("levels", "TEXT")):', '("after_exit", "TEXT"), ("levels", "TEXT"), ("anatomy", "TEXT")):')

p = "lts/grindsignal.py"
sub(p, '''def label_after_exit(cfg, db: DB, now: int | None = None) -> int:''', '''ANATOMY_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_anatomy (
    signal_id INTEGER PRIMARY KEY, symbol TEXT, side INTEGER, exit TEXT, reversal INTEGER, mfe REAL,
    anatomy TEXT, done_ts INTEGER
);
"""


def label_anatomy(cfg, db: DB, now: int | None = None, max_rows: int = 400) -> int:
    """Once the tracked plan has exited: the trade's anatomy (entry -> peak -> exit) for every recorded grind
    and every sent grind signal. Reversals (in profit first, then stopped) are flagged."""
    import json
    from .grindlab import _daily, ensure_live_schema
    from .levels import major_levels
    from .reversal import anatomy
    ensure_live_schema(db)
    db.conn.executescript(ANATOMY_SCHEMA)
    gs = cfg.grindsignal
    pl = gs.plans[gs.primary]
    now = now or int(time.time() * 1000)
    old = now - (gs.hours + 48) * HOUR
    n = 0
    daily = {}

    def lv(sym, t):
        if sym not in daily:
            daily[sym] = _daily(db, sym).reset_index()
        return major_levels(daily[sym], t)
    for r in db.query("SELECT symbol, ts, side, price FROM grind_live WHERE anatomy IS NULL AND ts > ? AND ts <= ? "
                      "ORDER BY ts LIMIT ?", (old, now - 30 * 60_000, max_rows)):
        a = anatomy(db, r["symbol"], r["side"], int(r["ts"]), pl, gs.hours, levels=lv(r["symbol"], int(r["ts"])))
        if a is None:
            continue
        db.execute("UPDATE grind_live SET anatomy=? WHERE symbol=? AND ts=? AND side=?",
                   (json.dumps(_clean(a)), r["symbol"], r["ts"], r["side"]))
        n += 1
    for r in db.query("""SELECT s.id, s.symbol, s.side, s.ts, s.entry FROM signals s WHERE s.kind='grindsig' AND s.ts > ?
                         AND NOT EXISTS (SELECT 1 FROM signal_anatomy a WHERE a.signal_id = s.id)""", (old,)):
        a = anatomy(db, r["symbol"], r["side"], int(r["ts"]), pl, gs.hours, entry_px=r["entry"],
                    levels=lv(r["symbol"], int(r["ts"])))
        if a is None:
            continue
        db.execute("INSERT OR REPLACE INTO signal_anatomy VALUES (?,?,?,?,?,?,?,?)",
                   (r["id"], r["symbol"], r["side"], a["exit"], int(a["reversal"]), a["mfe"], json.dumps(_clean(a)), now))
        n += 1
    return n


def label_after_exit(cfg, db: DB, now: int | None = None) -> int:''')

p = "lts/service.py"
sub(p, '''                    from .grindsignal import label_after_exit, label_grinds, label_signal_plans, record_signal_paths''',
    '''                    from .grindsignal import (label_after_exit, label_anatomy, label_grinds, label_signal_plans,
                                              record_signal_paths)''')
sub(p, '''                    record_signal_paths(self.cfg, self.db)
''', '''                    record_signal_paths(self.cfg, self.db)
                    try:
                        label_anatomy(self.cfg, self.db)
                    except Exception:
                        log.exception("trade anatomy failed")
''')
print("ok")
