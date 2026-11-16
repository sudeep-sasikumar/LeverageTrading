import numpy as np
import pandas as pd

from lts import patterns as P
from lts import sweeps as SW

HOUR = 3_600_000


def frame(c, h=None, l=None, start=1_790_000_000_000 // HOUR * HOUR):
    c = np.asarray(c, float)
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * 1.001 if h is None else h
    l = np.minimum(o, c) * 0.999 if l is None else l
    qv = np.full(len(c), 1e6)
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": qv / c, "qv": qv, "trades": 100.0, "tbqv": qv / 2},
                        index=start + np.arange(len(c)) * HOUR)


def test_sweep_of_the_24h_high_is_detected_and_traded_short():
    c = np.full(300, 100.0) + np.sin(np.arange(300) / 3)          # ranging market around 100 (24h high ~101)
    k = frame(c)
    h = k["h"].to_numpy().copy()
    h[250] = 103.0                                               # wick above the 24h high, close back inside
    k["h"] = h
    evs = [e for e in SW.detect(k, min_qv24=0) if e["i"] == 250]
    assert len(evs) == 1 and evs[0]["side"] == -1 and evs[0]["wick"] == 103.0
    r = SW.simulate(evs[0], k, "market", "wick", "2R")
    assert r is not None and r["risk"] > 0 and r["exit"] in ("stop", "target", "time")
    assert SW.plan(evs[0], 100.0, "wide", "2R")[0] > SW.plan(evs[0], 100.0, "wick", "2R")[0]   # wide stop is further away


def test_pattern_simulator_counts_stop_first_and_costs():
    n = 120
    c = np.full(n, 100.0)
    h, l = c * 1.0005, c * 0.9995
    h[10] = 106.5                                               # target (+6% for a 2% stop at 3R) reached on bar 10
    r = P.simulate(1, 100.0, 98.0, 1, h, l, c, c, P.TAKER + P.SLIP)
    assert r["3R_24h"] > 2.8 and r["3R_24h"] < 3.0              # +3R minus fees
    l2 = l.copy(); l2[5] = 97.5                                  # stop hit before the target
    r2 = P.simulate(1, 100.0, 98.0, 1, h, l2, c, c, P.TAKER + P.SLIP)
    assert r2["3R_24h"] < -1.0 and r2["1R_72h"] < -1.0
    assert P.simulate(1, 100.0, 99.9, 1, h, l, c, c, 0.0) is None    # stop tighter than the minimum: skipped


def test_four_hour_bars_are_aligned_and_complete():
    k = frame(np.arange(1, 50, dtype=float))
    k4 = P.to_4h(k)
    assert (k4.index % (4 * HOUR) == 0).all() and len(k4) >= 10
    first = k4.index[0]
    assert k4.loc[first, "h"] == k.loc[first:first + 3 * HOUR, "h"].max()
