"""New streamed replay must reproduce the old replay rows exactly, with far less memory."""
import sys, time, threading, hashlib
sys.path.insert(0, r"D:\Leverage trading")
import psutil
from lts.config import load_config
from lts.db import DB
from lts.replay import replay

cfg = load_config(r"D:\Leverage trading\config.yaml")
db = DB(cfg.path(cfg.storage.db_path))
COLS = "symbol, ts, side, kind, ref_price, atr_pct, sl_pct, features, rule_score, label, outcome, ret_r, exit_ts, mfe, mae, grid"

def digest():
    rows = db.query(f"SELECT {COLS} FROM candidates WHERE source='replay' ORDER BY symbol, ts, side")
    h = hashlib.sha256()
    for r in rows:
        h.update(repr(tuple(r)).encode())
    return len(rows), h.hexdigest()

before = digest()
print("old replay rows", before[0])
peak = [0]
def watch():
    p = psutil.Process()
    while True:
        peak[0] = max(peak[0], p.memory_info().rss); time.sleep(0.2)
threading.Thread(target=watch, daemon=True).start()
t = time.time()
replay(cfg, db)
print("new replay: %.0fs, peak memory %.2f GB" % (time.time() - t, peak[0] / 2**30))
after = digest()
print("new replay rows", after[0], "| identical to old:", before == after)
