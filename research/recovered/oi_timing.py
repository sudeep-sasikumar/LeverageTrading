"""Which moment does an openInterestHist record stamped T describe, and when is it published?"""
import json, time, urllib.request

B = "https://fapi.binance.com"
SYMS = ["AGTUSDT", "JCTUSDT", "ARIAUSDT"]

def get(path):
    with urllib.request.urlopen(B + path, timeout=15) as r:
        return json.loads(r.read())

live = {s: [] for s in SYMS}       # (local_ms, oi)
first_seen = {s: {} for s in SYMS}  # stamped ts -> (first seen local ms, value)
t_end = time.time() + 11 * 60
while time.time() < t_end:
    for s in SYMS:
        try:
            cur = get(f"/fapi/v1/openInterest?symbol={s}")
            live[s].append((int(cur["time"]), float(cur["openInterest"])))
            hist = get(f"/futures/data/openInterestHist?symbol={s}&period=5m&limit=3")
            now = int(time.time() * 1000)
            for h in hist:
                first_seen[s].setdefault(int(h["timestamp"]), (now, float(h["sumOpenInterest"])))
        except Exception as e:
            print("err", e)
    time.sleep(10)

for s in SYMS:
    print("==", s)
    for ts, (seen, val) in sorted(first_seen[s].items()):
        def nearest(target):
            return min(live[s], key=lambda x: abs(x[0] - target))
        a, b = nearest(ts), nearest(ts + 300000)
        print(f"stamp {ts} first_seen +{(seen - ts)/1000:.0f}s val {val:.1f} | live@T {a[1]:.1f} (dt {(a[0]-ts)/1000:.0f}s) "
              f"| live@T+5m {b[1]:.1f} (dt {(b[0]-ts-300000)/1000:.0f}s)")
