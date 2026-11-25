"""Cached incremental scans must produce exactly the same candidates and features as fresh scans."""
import sys, time
import numpy as np
sys.path.insert(0, r"D:\Leverage trading")
from lts import BAR_MS
from lts.config import load_config
from lts.db import DB
from lts.pipeline import DatasetCache
from lts.service import scan

cfg = load_config(r"D:\Leverage trading\config.yaml")
t = time.time(); db = DB(cfg.path(cfg.storage.db_path)); print("open (+ one-off index build) %.1fs" % (time.time() - t))
last = max(db.last_kline_ts().values())
cache = DatasetCache(db, cfg.storage.feature_window_bars)
prev = last - 24 * BAR_MS
t = time.time(); scan(cfg, db, prev, cache); print("cold cached scan %.1fs" % (time.time() - t))
bad = checked = 0
times = []
for bar in range(prev + BAR_MS, last + 1, BAR_MS):
    t = time.time(); a = scan(cfg, db, bar, cache); times.append(time.time() - t)
    step = (bar - prev) // BAR_MS
    if step in (1, 8, 16, 24):
        b = scan(cfg, db, bar)
        ka = {(c["symbol"], c["side"]): c for c in a}; kb = {(c["symbol"], c["side"]): c for c in b}
        same = set(ka) == set(kb) and all(
            np.allclose(ka[k]["aligned"].to_numpy(float), kb[k]["aligned"].to_numpy(float), rtol=1e-9, atol=1e-12,
                        equal_nan=True) for k in ka)
        checked += 1; bad += (not same)
        print("bar +%d: %d candidates, incremental %.1fs, identical to fresh scan: %s" % (step, len(a), times[-1], same))
print("incremental scan median %.1fs max %.1fs | mismatching bars: %d of %d" % (np.median(times), max(times), bad, checked))
