import pathlib

ROOT = pathlib.Path(r"D:\Leverage trading")


def sub(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:70])
        s = s.replace(a, b, 1)
    p.write_text(s, encoding="utf-8")


sub("lts/collector.py", [
    ("        self.failures = 0\n", "        self.failures = 0\n        self.last_errors = 0\n        self.last_seconds = 0.0\n"),
    ("        self.failures = self.failures + 1 if errors > len(syms) * 0.2 else 0\n",
     "        self.failures = self.failures + 1 if errors > len(syms) * 0.2 else 0\n"
     "        self.last_errors, self.last_seconds = errors, time.time() - t0\n"),
])

sub("lts/service.py", [
    ('''                bar_ts = await self.collector.collect()
                await scan_cycle(self.cfg, self.db, self.collector, self.engine, self.bundle, bar_ts,
                                 cache=self.cache)
                self.last_bar = bar_ts
''',
     '''                bar_ts = await self.collector.collect()
                t_scan = time.time()
                cands = await scan_cycle(self.cfg, self.db, self.collector, self.engine, self.bundle, bar_ts,
                                         cache=self.cache)
                self.last_bar = bar_ts
                # read by the dashboard to show that the scanner is alive
                self.db.set("heartbeat", {
                    "ts": int(time.time() * 1000), "bar_ts": bar_ts, "candidates": len(cands),
                    "signals": sum(1 for c in cands if c.get("decision") == "sent"),
                    "errors": self.collector.last_errors, "collect_s": round(self.collector.last_seconds, 1),
                    "scan_s": round(time.time() - t_scan, 1), "telegram": self.tg.enabled})
'''),
    ('''        await refresh_universe(self.cfg, self.db, self.binance, self.weex)
        if self.bundle and''',
     '''        await refresh_universe(self.cfg, self.db, self.binance, self.weex)
        self.db.set("phase", "running")
        if self.bundle and'''),
])

sub("lts/cli.py", [
    ('''    sub.add_parser("start", help="container entry point: first-time backfill + training if needed, then run")''',
     '''    sub.add_parser("start", help="container entry point: first-time backfill + training if needed, then run")
    sub.add_parser("dashboard", help="serve only the read-only dashboard (http://127.0.0.1:8020 by default)")'''),
    ('''        elif args.cmd == "run":
            from .service import run_service
            asyncio.run(run_service(cfg, db))''',
     '''        elif args.cmd == "run":
            from .dashboard import start_in_thread
            from .service import run_service
            start_in_thread(cfg)
            asyncio.run(run_service(cfg, db))
        elif args.cmd == "dashboard":
            from .dashboard import serve
            asyncio.run(serve(cfg))'''),
    ('''            log.info("LTS version %s", os.environ.get("APP_VERSION", "dev"))''',
     '''            log.info("LTS version %s", os.environ.get("APP_VERSION", "dev"))
            from .dashboard import start_in_thread
            start_in_thread(cfg)                      # visible during the first-start setup too'''),
    ('''                log.info("first start: downloading %.0f days of history (about 30 minutes)", days)''',
     '''                db.set("phase", "backfill")
                log.info("first start: downloading %.0f days of history (about 30 minutes)", days)'''),
    ('''            if not db.get("bootstrap_replay_done"):
                replay(cfg, db)''',
     '''            if not db.get("bootstrap_replay_done"):
                db.set("phase", "replay")
                replay(cfg, db)'''),
    ('''            if not db.query("SELECT 1 FROM models LIMIT 1"):
                log.info(''',
     '''            if not db.query("SELECT 1 FROM models LIMIT 1"):
                db.set("phase", "train")
                log.info('''),
])

sub("Dockerfile", [
    ("    LTS_LOG_DIR=/app/data/logs\n", "    LTS_LOG_DIR=/app/data/logs \\\n    DASHBOARD_BIND=0.0.0.0 \\\n    DASHBOARD_PORT=8020\n"),
    ('# "start" =', 'EXPOSE 8020\n# "start" ='),
])

sub("docker-compose.yaml", [
    ('''# No ports are opened: the bot only makes outbound calls (Binance, Weex, Telegram).
''',
     '''# HTTPS dashboard: the VPS already runs Traefik (set up for Ignition & Coil). The dashboard is served
# under /lts/ on the same hostname, so it reuses that hostname's certificate: no new domain needed.
# No host ports are opened; the dashboard is reached only through Traefik and is read-only.
'''),
    ('''#   TELEGRAM_CHAT_ID     your chat id (a group id starts with -100)
''',
     '''#   TELEGRAM_CHAT_ID     your chat id (a group id starts with -100)
#   DASHBOARD_PASSWORD   required for the dashboard - pick a long one (without it the dashboard stays off)
#   DOMAIN               optional - defaults to this VPS's Hostinger hostname
'''),
    ('''      BACKFILL_DAYS: ${BACKFILL_DAYS:-}
    volumes:
      - lts_data:/app/data                # database, models, logs - survives redeploys
''',
     '''      BACKFILL_DAYS: ${BACKFILL_DAYS:-}
      DASHBOARD_PASSWORD: ${DASHBOARD_PASSWORD:-}   # required: without it the dashboard stays off
      DASHBOARD_COOKIE_SECURE: "true"               # the login cookie is only sent over HTTPS
    volumes:
      - lts_data:/app/data                # database, models, logs - survives redeploys
    expose:
      - "8020"                            # reached only through Traefik
    labels:
      - traefik.enable=true
      - traefik.http.routers.lts.rule=Host(`${DOMAIN:-srv1479428.hstgr.cloud}`) && PathPrefix(`/lts`)
      - traefik.http.routers.lts.entrypoints=websecure
      - traefik.http.routers.lts.tls=true
      - traefik.http.routers.lts.tls.certresolver=letsencrypt
      - traefik.http.routers.lts.middlewares=lts-slash,lts-strip
      # /lts -> /lts/ (the page loads its data with relative links), then strip /lts for the app
      - traefik.http.middlewares.lts-slash.redirectregex.regex=^(https?://[^/]+/lts)$$
      - traefik.http.middlewares.lts-slash.redirectregex.replacement=$${1}/
      - traefik.http.middlewares.lts-slash.redirectregex.permanent=true
      - traefik.http.middlewares.lts-strip.stripprefix.prefixes=/lts
      - traefik.http.services.lts.loadbalancer.server.port=8020
'''),
])

sub("README.md", [
    ('''   | `ACCOUNT_SIZE_USDT` | optional | quotes each signal's margin in USDT |''',
     '''   | `DASHBOARD_PASSWORD` | for the dashboard | a long password of your choice; without it the dashboard stays off |
   | `ACCOUNT_SIZE_USDT` | optional | quotes each signal's margin in USDT |'''),
    ('''No ports are opened: the bot only makes outbound calls.
''',
     '''5. Dashboard: `https://srv1479428.hstgr.cloud/lts/` - it shares the hostname and certificate
   that Ignition & Coil and Launch Radar already use (set `DOMAIN` to change the hostname). It is
   read-only: signals with their results, every scanned candidate and why it was skipped, the
   model's validation numbers, and a health panel. It is up during the first-start setup too.

No host ports are opened; the dashboard is reached only through Traefik. On this PC:
`python -m lts dashboard`, then http://127.0.0.1:8020 (no login on localhost).
'''),
])
sub(".env.example", [("# Optional\n", "# Only needed when the dashboard is reachable beyond this PC (VPS). Use a long random password.\nDASHBOARD_PASSWORD=\n# Optional\n")])
print("patched")
