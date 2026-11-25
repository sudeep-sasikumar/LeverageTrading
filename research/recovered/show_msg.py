import asyncio, sys, tempfile, pathlib
sys.path.insert(0, r"D:\Leverage trading"); sys.path.insert(0, r"D:\Leverage trading\tests")
from test_signal_flow import Notifier, bundle, candidate, NOW
from lts.config import Config
from lts.db import DB
from lts.signals import SignalEngine

tg = Notifier()
db = DB(pathlib.Path(tempfile.mkdtemp()) / "m.sqlite")
asyncio.run(SignalEngine(Config(), db, None, tg).decide([candidate()], bundle("shadow"), NOW))
print(tg.sent[0])
