import pathlib

ROOT = pathlib.Path(r"D:\Leverage trading")


def sub(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:70])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")


sub("lts/db.py", [
    ("CREATE TABLE IF NOT EXISTS oi (",
     "CREATE INDEX IF NOT EXISTS idx_klines_ts ON klines (ts);\nCREATE TABLE IF NOT EXISTS oi ("),
    ("CREATE TABLE IF NOT EXISTS funding (",
     "CREATE INDEX IF NOT EXISTS idx_oi_ts ON oi (ts);\nCREATE TABLE IF NOT EXISTS funding ("),
    ("CREATE TABLE IF NOT EXISTS lsr (",
     "CREATE INDEX IF NOT EXISTS idx_premium_ts ON premium (ts);\nCREATE TABLE IF NOT EXISTS lsr ("),
])

sub("lts/pipeline.py", [(
    '''def build_context(ds: Dataset, symbols: list[str] | None = None):
    """Market context from every symbol in the dataset that is an active crypto perp."""
    use = symbols if symbols is not None else [
        s for s in ds.klines if ds.meta.get(s, {}).get("active", 1) or s == "BTCUSDT"]
    panels = {s: F.panel_inputs(ds.klines[s]) for s in use}
    return F.market_context(panels, ds.klines.get("BTCUSDT"))''',
    '''def build_context(ds: Dataset, symbols: list[str] | None = None):
    """Market context from every symbol in the dataset that is an active crypto perp.

    Returns (market frame, rank frames, per-symbol panel inputs).
    """
    use = symbols if symbols is not None else [
        s for s in ds.klines if ds.meta.get(s, {}).get("active", 1) or s == "BTCUSDT"]
    panels = {s: F.panel_inputs(ds.klines[s]) for s in use}
    market, ranks = F.market_context(panels, ds.klines.get("BTCUSDT"))
    return market, ranks, panels


class DatasetCache:
    """Keeps the live feature window in memory and only reads new rows each cycle."""

    SLOW = (("oi", ("oi",)), ("lsr", ("top_pos", "global_acc")), ("funding", ("rate",)),
            ("premium", ("mark", "index_px", "funding_rate")))

    def __init__(self, db: DB, window_bars: int):
        self.db, self.window = db, window_bars
        self.ds: Dataset | None = None
        self.until = 0

    def get(self, bar_ts: int) -> Dataset:
        since = bar_ts - self.window * F.BAR_MS
        if self.ds is None or bar_ts <= self.until or bar_ts - self.until > 60 * F.BAR_MS:
            self.ds = load_dataset(self.db, since, until_ts=bar_ts)        # cold start or long gap
            self.until = bar_ts
            return self.ds
        ds = self.ds
        ds.meta = {r["symbol"]: dict(r) for r in self.db.query("SELECT * FROM symbols")}
        for sym, new in self.db.klines_many(self.until + F.BAR_MS, bar_ts).items():
            if sym in ds.klines:
                old = ds.klines[sym].reset_index().rename(columns={"index": "ts"})
                merged = F.regularize(pd.concat([old, new], ignore_index=True))
                ds.klines[sym] = merged[merged.index >= since]
            else:                                                             # newly listed / newly long enough
                full = self.db.klines(sym, since_ts=since, until_ts=bar_ts)
                if len(full) >= 300:
                    ds.klines[sym] = F.regularize(full)
        fresh_from = self.until - F.BAR_MS
        cut = bar_ts + F.BAR_MS
        for table, cols in self.SLOW:
            store = getattr(ds, table)
            keep_from = since - 5 * 86_400_000
            for sym, new in self.db.series_many(table, cols, fresh_from).items():
                old = store.get(sym)
                both = new if old is None else pd.concat([old, new], ignore_index=True)
                both = both.drop_duplicates("ts", keep="last").sort_values("ts")
                store[sym] = both[(both["ts"] >= keep_from) & (both["ts"] <= cut)].reset_index(drop=True)
        self.until = bar_ts
        return ds'''),
])

sub("lts/replay.py", [("    market, ranks = build_context(ds)\n", "    market, ranks, _ = build_context(ds)\n")])

sub("lts/service.py", [
    ('''def scan(cfg: Config, db: DB, bar_ts: int) -> list[dict]:
    """Detect candidates on the bar that opened at bar_ts (already closed and stored)."""
    since = bar_ts - cfg.storage.feature_window_bars * BAR_MS
    ds = load_dataset(db, since, until_ts=bar_ts)
    if "BTCUSDT" not in ds.klines:
        log.warning("no BTC data - scan skipped")
        return []
    market, ranks = build_context(ds)''',
     '''def scan(cfg: Config, db: DB, bar_ts: int, cache: DatasetCache | None = None) -> list[dict]:
    """Detect candidates on the bar that opened at bar_ts (already closed and stored)."""
    since = bar_ts - cfg.storage.feature_window_bars * BAR_MS
    ds = cache.get(bar_ts) if cache else load_dataset(db, since, until_ts=bar_ts)
    if "BTCUSDT" not in ds.klines:
        log.warning("no BTC data - scan skipped")
        return []
    market, ranks, panels = build_context(ds)
    trig = cfg.trigger'''),
    ('''        if len(k) < cfg.trigger.min_history_bars or k["qv"].iloc[-288:].sum() < cfg.universe.min_qvol_24h:
            continue''',
     '''        if len(k) < cfg.trigger.min_history_bars or k["qv"].iloc[-288:].sum() < cfg.universe.min_qvol_24h:
            continue
        p = panels[sym].iloc[-1]          # same definitions as the trigger's momentum test
        if not (abs(p["z_1h"]) >= trig.z_1h or abs(p["z_4h"]) >= trig.z_4h or abs(p["r_24h"]) >= trig.r_24h):
            continue'''),
    ('''from .pipeline import build_context, features_to_json, load_dataset, raw_frame''',
     '''from .pipeline import DatasetCache, build_context, features_to_json, load_dataset, raw_frame'''),
    ('''                     bar_ts: int, dry_run: bool = False) -> list[dict]:
    t0 = time.time()
    cands = scan(cfg, db, bar_ts)''',
     '''                     bar_ts: int, dry_run: bool = False, cache: DatasetCache | None = None) -> list[dict]:
    t0 = time.time()
    cands = scan(cfg, db, bar_ts, cache)'''),
    ('''        self.bundle = load_bundle(cfg, db)
        self.started = time.time()''',
     '''        self.bundle = load_bundle(cfg, db)
        self.cache = DatasetCache(db, cfg.storage.feature_window_bars)
        self.started = time.time()'''),
    ('''                await scan_cycle(self.cfg, self.db, self.collector, self.engine, self.bundle, bar_ts)''',
     '''                await scan_cycle(self.cfg, self.db, self.collector, self.engine, self.bundle, bar_ts,
                                 cache=self.cache)'''),
])
print("patched")
