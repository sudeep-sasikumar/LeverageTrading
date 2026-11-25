"""Local preview of the dashboard with a throwaway database (port 8031, no login on localhost)."""
import asyncio, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Leverage trading")
from aiohttp import web
from lts.config import Config
from lts.dashboard import build_app
from lts.db import DB
from lts import journal

D = Path(sys.argv[1])
D.mkdir(parents=True, exist_ok=True)
now = int(time.time() * 1000)
db = DB(D / "d.sqlite")
db.set("phase", "running")
db.set("heartbeat", {"ts": now, "bar_ts": now - 300_000, "candidates": 1, "signals": 0, "errors": 0, "collect_s": 4, "scan_s": 16, "telegram": True})
db.execute("""INSERT INTO signals(symbol, side, ts, grade, entry, sl, tp, sl_pct, tp_pct, leverage, status, kind, expires_ts)
              VALUES ('MEGAUSDT', 1, ?, 'P', 0.05457, 0.05402, 0.05566, 0.01, 0.02, 5, 'tp', 'grindsig', ?)""", (now - 3 * 3600_000, now))
jp = journal.path_for(D / "d.sqlite")
if not jp.exists():
    journal.add(jp, {"symbol": "MEGAUSDT", "side": "long", "leverage": 5, "margin": 100.2104, "entry_price": 0.05457,
                     "tp_price": 0.05566, "sl_price": 0.05402, "open_fee": 0.3998, "open_ts": now - 3 * 3600_000 + 120_000,
                     "exit_price": 0.05566, "close_fee": 0.4078, "realized_pnl": 9.9844, "close_ts": now - 3600_000,
                     "status": "closed", "exit_reason": "tp", "source": "screenshot", "notes": "first LTS signal taken"})
cfg = Config()
cfg.storage.db_path = str(D / "d.sqlite")


async def main():
    r = web.AppRunner(build_app(cfg, "", False, False))
    await r.setup()
    await web.TCPSite(r, "127.0.0.1", 8031).start()
    await asyncio.sleep(3600)

asyncio.run(main())
