"""Local preview: the reversal test trade on the dashboard (port 8032, no login)."""
import asyncio, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Leverage trading")
sys.path.insert(0, r"D:\Leverage trading\tests")
import lts.dashboard as D
from aiohttp import web
from lts import grindsignal as GS
from lts.config import Config
import test_reversal as T

d = Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
for f in d.glob("r.sqlite*"):
    f.unlink()
db = T.make(d)
db.execute("INSERT INTO grind_live(symbol, ts, side, price, score, decision, model_p, threshold) VALUES ('AUSDT', ?, 1, 10.0, 5, 'sent', 0.4, 0.367)", (T.NOW,))
db.set("phase", "running")
cfg = Config(); cfg.storage.db_path = str(d / "r.sqlite")
GS.label_anatomy(cfg, db, now=T.NOW + 26 * T.HOUR)
fake = (T.NOW + 26 * T.HOUR) / 1000
D.time.time = lambda: fake


async def main():
    r = web.AppRunner(D.build_app(cfg, "", False, False)); await r.setup()
    await web.TCPSite(r, "127.0.0.1", 8032).start(); await asyncio.sleep(3600)
asyncio.run(main())
