"""Old (committed) replay vs new streamed replay on the SAME data: rows must be identical."""
import sys, time, threading, hashlib, subprocess, types
ROOT = r"D:\Leverage trading"
sys.path.insert(0, ROOT)
import psutil
import lts
from lts.config import load_config
from lts.db import DB
from lts.replay import replay as new_replay

old_src = subprocess.run(["git", "-C", ROOT, "show", "HEAD:lts/replay.py"], capture_output=True, text=True, check=True).stdout
old = types.ModuleType("lts.replay_old"); old.__package__ = "lts"
exec(compile(old_src, "replay_old.py", "exec"), old.__dict__)

cfg = load_config(ROOT + r"\config.yaml")
db = DB(cfg.path(cfg.storage.db_path))
COLS = "symbol, ts, side, kind, ref_price, atr_pct, sl_pct, features, rule_score, label, outcome, ret_r, exit_ts, mfe, mae, grid"

def digest():
    rows = db.query(f"SELECT {COLS} FROM candidates WHERE source='replay' ORDER BY symbol, ts, side")
    return len(rows), hashlib.sha256(b"".join(repr(tuple(r)).encode() for r in rows)).hexdigest(), \
        {(r["symbol"], r["ts"], r["side"]): tuple(r) for r in rows}

peak = [0]
def watch():
    p = psutil.Process()
    while True:
        peak[0] = max(peak[0], p.memory_info().rss); time.sleep(0.2)
threading.Thread(target=watch, daemon=True).start()

peak[0] = 0; t = time.time(); old.replay(cfg, db); a = digest()
print("OLD: %d rows, %.0fs, peak %.2f GB" % (a[0], time.time() - t, peak[0] / 2**30))
peak[0] = 0; t = time.time(); new_replay(cfg, db); b = digest()
print("NEW: %d rows, %.0fs, peak %.2f GB" % (b[0], time.time() - t, peak[0] / 2**30))
print("identical:", a[1] == b[1])
if a[1] != b[1]:
    only_a, only_b = set(a[2]) - set(b[2]), set(b[2]) - set(a[2])
    diff = [k for k in set(a[2]) & set(b[2]) if a[2][k] != b[2][k]]
    print("only old", len(only_a), "only new", len(only_b), "changed", len(diff))
    for k in list(only_a)[:3] + list(only_b)[:3]: print(" ", k)
    if diff:
        k = diff[0]
        for name, x, y in zip(COLS.split(", "), a[2][k], b[2][k]):
            if x != y: print("  ", k, name, str(x)[:200], "|", str(y)[:200])
