import pathlib

ROOT = pathlib.Path(r"D:\Leverage trading")


def sub(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:70])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")


sub("lts/config.py", [(
'''    if os.environ.get("LTS_DB_PATH"):
        cfg.storage.db_path = os.environ["LTS_DB_PATH"]
    return cfg''',
'''    # environment overrides (how the Docker deployment is configured)
    env = os.environ
    if env.get("LTS_DB_PATH"):
        cfg.storage.db_path = env["LTS_DB_PATH"]
    if env.get("LTS_MODEL_DIR"):
        cfg.storage.model_dir = env["LTS_MODEL_DIR"]
    if env.get("LTS_LOG_DIR"):
        cfg.storage.log_dir = env["LTS_LOG_DIR"]
    if env.get("ACCOUNT_SIZE_USDT"):
        cfg.telegram.account_size_usdt = float(env["ACCOUNT_SIZE_USDT"])
    if env.get("RISK_PER_TRADE"):
        risk = float(env["RISK_PER_TRADE"])
        if not 0 < risk <= 0.05:
            raise ValueError("RISK_PER_TRADE must be a fraction between 0 and 0.05 (0.01 = 1% of the account)")
        cfg.risk.risk_per_trade = risk
    if env.get("MAX_SIGNALS_PER_DAY"):
        cfg.risk.max_per_day = int(env["MAX_SIGNALS_PER_DAY"])
    return cfg'''),
])

sub("lts/cli.py", [
('''    sub.add_parser("run", help="run the live service (collector, scanner, tracker, learner, Telegram)")''',
 '''    sub.add_parser("run", help="run the live service (collector, scanner, tracker, learner, Telegram)")
    sub.add_parser("start", help="container entry point: first-time backfill + training if needed, then run")'''),
('''        elif args.cmd == "run":
            from .service import run_service
            asyncio.run(run_service(cfg, db))''',
'''        elif args.cmd == "run":
            from .service import run_service
            asyncio.run(run_service(cfg, db))
        elif args.cmd == "start":
            import os
            from .backfill import backfill
            from .learner import train_and_maybe_promote
            from .replay import replay
            from .service import run_service
            log = logging.getLogger("lts.start")
            log.info("LTS version %s", os.environ.get("APP_VERSION", "dev"))
            # each step is skipped once done, so a restart mid-bootstrap resumes where it stopped
            if not db.get("bootstrap_backfill_done"):
                days = float(os.environ.get("BACKFILL_DAYS") or 30)
                log.info("first start: downloading %.0f days of history (about 30 minutes)", days)
                asyncio.run(backfill(cfg, db, days=days))
                db.set("bootstrap_backfill_done", True)
            if not db.get("bootstrap_replay_done"):
                replay(cfg, db)
                db.set("bootstrap_replay_done", True)
            if not db.query("SELECT 1 FROM models LIMIT 1"):
                log.info("\\n%s", train_and_maybe_promote(cfg, db)["summary"])
            asyncio.run(run_service(cfg, db))'''),
])
print("patched")
