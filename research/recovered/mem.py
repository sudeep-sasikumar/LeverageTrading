import sys, time, threading
sys.path.insert(0, r"D:\Leverage trading")
import psutil
from lts.config import load_config
from lts.db import DB
from lts.pipeline import load_dataset, build_context, raw_frame
from lts import learner

peak = [0]
def watch():
    p = psutil.Process()
    while True:
        peak[0] = max(peak[0], p.memory_info().rss); time.sleep(0.2)
threading.Thread(target=watch, daemon=True).start()
gb = lambda: "%.2f GB" % (peak[0] / 2**30)
cfg = load_config(r"D:\Leverage trading\config.yaml"); db = DB(cfg.path(cfg.storage.db_path))
t = time.time(); ds = load_dataset(db, 0); print("replay load_dataset peak", gb(), "%.0fs" % (time.time() - t))
market, ranks, panels = build_context(ds); print("after context peak", gb())
for s in list(ds.klines)[:40]: raw_frame(ds, s, market, ranks)
print("after 40 frames peak", gb())
del ds, market, ranks, panels
peak[0] = 0
df = learner.load_labelled(db); print("train load_labelled peak", gb(), len(df))
