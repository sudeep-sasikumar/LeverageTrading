import pathlib

ROOT = pathlib.Path(r"D:\Leverage trading")


def sub(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:80])
        s = s.replace(a, b, 1)
    p.write_text(s, encoding="utf-8")


sub("lts/config.py", [
    ('''@dataclass
class Risk:''',
     '''@dataclass
class Micro:
    """Live collection of liquidations and order-book depth (for studying sweeps / stop hunts)."""
    enabled: bool = True
    book_symbols: int = 120              # order-book snapshots for the N most liquid coins (+ anything grinding)
    book_every_minutes: int = 5
    keep_days: int = 180


@dataclass
class Risk:'''),
    ('''    grindlab: GrindLab = field(default_factory=GrindLab)''',
     '''    grindlab: GrindLab = field(default_factory=GrindLab)
    micro: Micro = field(default_factory=Micro)'''),
])

sub("lts/service.py", [
    ('''        self.cache = DatasetCache(db, cfg.storage.feature_window_bars)''',
     '''        self.cache = DatasetCache(db, cfg.storage.feature_window_bars)
        from .microstructure import LiquidationStream
        self.liq = LiquidationStream(db)'''),
    ('''    async def grind_loop(self) -> None:''',
     '''    async def book_loop(self) -> None:
        """Order-book depth snapshots, 40s after every N-minute boundary."""
        from .microstructure import book_symbols, snapshot_books
        mc = self.cfg.micro
        step = mc.book_every_minutes * 60
        while True:
            now = time.time()
            await asyncio.sleep(max(1.0, (math.floor(now / step) + 1) * step + 40 - now))
            try:
                ts = int(time.time() // step * step * 1000)
                n = await snapshot_books(self.db, self.binance, book_symbols(self.db, mc.book_symbols), ts)
                log.debug("order-book snapshots: %d", n)
            except Exception:
                log.exception("order-book snapshot failed")

    async def grind_loop(self) -> None:'''),
    ('''                 (self.bar_loop(), self.track_loop(), self.slow_loop(), self.grind_loop(),
                  self.tg.poll_commands(self.command))]''',
     '''                 (self.bar_loop(), self.track_loop(), self.slow_loop(), self.grind_loop(),
                  self.tg.poll_commands(self.command))]
        if self.cfg.micro.enabled:
            tasks += [asyncio.create_task(self.liq.run()), asyncio.create_task(self.book_loop())]'''),
    ('''                    await self.retrain()
                    self.db.prune(self.cfg.storage.keep_klines_days)''',
     '''                    await self.retrain()
                    self.db.prune(self.cfg.storage.keep_klines_days)
                    if self.cfg.micro.enabled:
                        from .microstructure import prune as micro_prune
                        micro_prune(self.db, self.cfg.micro.keep_days)'''),
])

sub("lts/dashboard.py", [
    ('''        out["movers"] = kv.get("movers_latest")''',
     '''        out["movers"] = kv.get("movers_latest")
        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='liquidations'"):
            liq = q("SELECT COUNT(*) n, COALESCE(SUM(usd), 0) usd, MAX(ts) last FROM liquidations WHERE ts>=?", (now - DAY,))[0]
            books = q("SELECT COUNT(*) n, COUNT(DISTINCT symbol) s, MAX(ts) last FROM book_snapshots WHERE ts>=?", (now - DAY,))[0]
            out["micro"] = {"liq_24h": liq["n"], "liq_usd_24h": liq["usd"], "liq_last": liq["last"],
                            "books_24h": books["n"], "book_symbols": books["s"], "book_last": books["last"]}'''),
])

sub("lts/web/index.html", [
    ('''    item("Last nightly training", esc(d.last_train_day || "not yet")),''',
     '''    item("Liquidations recorded (24h)", d.micro ? `${(d.micro.liq_24h || 0).toLocaleString()} ($${((d.micro.liq_usd_24h || 0) / 1e6).toFixed(1)}M), last ${ago(d.micro.liq_last)}` : "not started"),
    item("Order-book snapshots (24h)", d.micro ? `${(d.micro.books_24h || 0).toLocaleString()} across ${d.micro.book_symbols || 0} coins, last ${ago(d.micro.book_last)}` : "not started"),
    item("Last nightly training", esc(d.last_train_day || "not yet")),'''),
])
print("micro wired")
