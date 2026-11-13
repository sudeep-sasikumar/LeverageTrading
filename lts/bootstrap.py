"""Container entry point (`python -m lts start`).

First-start setup (download history, rebuild candidates, train), then the live service.
A failure is logged with a plain-language explanation, shown on the dashboard, sent to
Telegram once, and retried with back-off - the process stays up, so the container never
crash-loops silently and the dashboard stays reachable to say what is wrong.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback

from .config import Config
from .db import DB

log = logging.getLogger("lts.start")
BACKOFF_S = [60, 120, 300, 600, 900]


def explain(e: BaseException) -> str:
    from .exchanges import ApiError
    if isinstance(e, ApiError) and "binance" in str(e) and e.status in (403, 451):
        return (f"Binance refuses connections from this server's location (HTTP {e.status}). "
                "The VPS must be in a region Binance serves (not the US); check the VPS location in Hostinger.")
    if isinstance(e, ApiError):
        return f"Exchange API error: {e}"[:400]
    if isinstance(e, MemoryError):
        return "Out of memory. The VPS needs about 1 GB free for the first-start setup."
    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
        return "The disk is full. The database needs about 1 GB."
    return f"{type(e).__name__}: {e}"[:400]


def notify(cfg: Config, text: str) -> None:
    from .telegram import Telegram

    async def go():
        tg = Telegram(cfg)
        try:
            await tg.send(text)
        finally:
            await tg.close()

    try:
        asyncio.run(go())
    except Exception:
        log.exception("telegram alert failed")


def bootstrap(cfg: Config, db: DB) -> None:
    """Each step is skipped once done, so a restart mid-setup resumes where it stopped."""
    from .backfill import backfill
    from .learner import train_and_maybe_promote
    from .replay import replay
    if not db.get("bootstrap_backfill_done"):
        days = float(os.environ.get("BACKFILL_DAYS") or 30)
        db.set("phase", "backfill")
        log.info("first start: downloading %.0f days of history (about 30 minutes)", days)
        asyncio.run(backfill(cfg, db, days=days))
        db.set("bootstrap_backfill_done", True)
    if not db.get("bootstrap_replay_done"):
        db.set("phase", "replay")
        replay(cfg, db)
        db.set("bootstrap_replay_done", True)
    if not db.query("SELECT 1 FROM models LIMIT 1"):
        db.set("phase", "train")
        log.info("\n%s", train_and_maybe_promote(cfg, db)["summary"])


def main(cfg: Config, db: DB) -> None:
    from .dashboard import start_in_thread
    from .service import run_service
    from .telegram import esc
    log.info("LTS version %s", os.environ.get("APP_VERSION", "dev"))
    start_in_thread(cfg)                       # reachable during setup and while failing, too
    attempt, last_msg = 0, None
    while True:
        try:
            bootstrap(cfg, db)
            asyncio.run(run_service(cfg, db))
            return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            msg = explain(e)
            wait = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
            attempt += 1
            log.error("start-up failed (attempt %d): %s - retrying in %ds\n%s", attempt, msg, wait,
                      traceback.format_exc())
            db.set("last_error", {"ts": int(time.time() * 1000), "message": msg, "phase": db.get("phase"),
                                  "attempt": attempt, "retry_in_s": wait})
            if msg != last_msg:                # one Telegram alert per distinct problem
                notify(cfg, f"⚠️ <b>LTS could not start</b>\n{esc(msg)}\n"
                            f"Retrying automatically in {max(wait // 60, 1)} min. Details: container logs / dashboard.")
                last_msg = msg
            time.sleep(wait)
