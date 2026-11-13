"""Command line entry point: python -m lts <command>."""
from __future__ import annotations

import argparse
import asyncio
import logging
import logging.handlers
import sys

from .config import load_config
from .db import DB


def setup_logging(cfg, name: str) -> None:
    log_dir = cfg.path(cfg.storage.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    fileh = logging.handlers.RotatingFileHandler(log_dir / f"{name}.log", maxBytes=5_000_000, backupCount=5,
                                                 encoding="utf-8")
    fileh.setFormatter(fmt)
    root.addHandler(fileh)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lts", description="Leverage trading signal system")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("backfill", help="download history from Binance (resumable)")
    p.add_argument("--days", type=float, default=30.0)
    p.add_argument("--oi-period", default="15m", choices=["5m", "15m", "30m", "1h"])
    p.add_argument("--symbols", default="", help="comma separated subset")

    p = sub.add_parser("replay", help="regenerate candidates + labels from stored history")
    p.add_argument("--days", type=float, default=0.0, help="limit to the most recent N days (0 = all)")
    p.add_argument("--keep", action="store_true", help="keep existing replay rows instead of rebuilding")

    p = sub.add_parser("train", help="train/validate a model and promote it if it beats the champion")
    p.add_argument("--force", action="store_true", help="promote even if promotion checks fail")

    sub.add_parser("run", help="run the live service (collector, scanner, tracker, learner, Telegram)")
    sub.add_parser("start", help="container entry point: first-time backfill + training if needed, then run")
    sub.add_parser("dashboard", help="serve only the read-only dashboard (http://127.0.0.1:8020 by default)")
    p = sub.add_parser("scan", help="one collection + scan cycle, print candidates, send nothing")
    p.add_argument("--no-collect", action="store_true")
    sub.add_parser("status", help="print database / model / signal status")
    p = sub.add_parser("report", help="print the performance report")
    p.add_argument("--days", type=float, default=7.0)
    sub.add_parser("telegram-test", help="send a test message to the configured chat")
    p = sub.add_parser("research", help="barrier-grid and base-rate study on labelled candidates")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg, args.cmd)
    db = DB(cfg.path(cfg.storage.db_path))

    try:
        if args.cmd == "backfill":
            from .backfill import backfill
            only = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
            asyncio.run(backfill(cfg, db, days=args.days, oi_period=args.oi_period, only=only))
        elif args.cmd == "replay":
            from .replay import replay
            replay(cfg, db, days=args.days, rebuild=not args.keep)
        elif args.cmd == "train":
            from .learner import train_and_maybe_promote
            print(train_and_maybe_promote(cfg, db, force=args.force)["summary"])
        elif args.cmd == "run":
            from .dashboard import start_in_thread
            from .service import run_service
            start_in_thread(cfg)
            asyncio.run(run_service(cfg, db))
        elif args.cmd == "dashboard":
            from .dashboard import serve
            asyncio.run(serve(cfg))
        elif args.cmd == "start":
            from .bootstrap import main as start_main
            start_main(cfg, db)
        elif args.cmd == "scan":
            from .service import scan_once
            asyncio.run(scan_once(cfg, db, collect=not args.no_collect))
        elif args.cmd == "status":
            from .reports import status_text
            print(status_text(cfg, db))
        elif args.cmd == "report":
            from .reports import performance_text
            print(performance_text(cfg, db, days=args.days))
        elif args.cmd == "telegram-test":
            from .telegram import Telegram
            asyncio.run(Telegram(cfg).send_once("LTS test message: Telegram is wired up correctly."))
        elif args.cmd == "research":
            from .research import research_text
            print(research_text(cfg, db))
    except KeyboardInterrupt:
        pass
    finally:
        db.close()
