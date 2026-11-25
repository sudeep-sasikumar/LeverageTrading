import asyncio

import numpy as np

from lts.fees import fees
import pytest

from lts import aim as A
from lts.config import Config
from lts.db import DB
from lts.signals import SignalEngine

NOW = 1_794_900_000_000 // 3_600_000 * 3_600_000


class Notifier:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(text)
        return True


def cand(symbol, score, side=1):
    return {"symbol": symbol, "side": side, "ts": NOW, "ref_price": 10.0, "atr_pct": 0.02, "kind": "daily",
            "aim_score": score, "on_weex": True, "src": "binance", "price_precision": 4, "weex_qv24": 5e6,
            "book": {"bid": 9.999, "ask": 10.0, "mid": 9.9995, "spread_bps": 1.0, "imb": 0.0, "depth_usd": 1e5},
            "features": {"r_6h": 0.05, "r_24h": 0.08, "weex_dev_bps": 0.5}}


def test_one_daily_trade_with_the_aim_plan(tmp_path):
    cfg, db, tg = Config(), DB(tmp_path / "a.sqlite"), Notifier()
    eng = SignalEngine(cfg, db, None, tg)
    low, best, other = cand("LOWUSDT", 6), cand("BESTUSDT", 10), cand("OTHERUSDT", 9)
    asyncio.run(eng.decide([low, best, other], None, NOW))
    assert low["decision"] == "skipped:below_threshold"
    assert best["decision"] == "sent" and other["decision"] == "skipped:daily_trade_open"
    s = dict(db.query("SELECT * FROM signals")[0])
    assert s["kind"] == "daily" and s["grade"] == "P" and s["leverage"] == 10
    assert abs(s["tp_pct"] - 0.01) < 1e-12 and abs(s["sl_pct"] - 0.01) < 1e-12 and s["expires_ts"] - s["ts"] == 24 * 3_600_000
    assert "DAILY TRADE" in tg.sent[0] and "+1.0%" in tg.sent[0] and "10x" in tg.sent[0]
    # the day's trade closes; a later candidate the same UTC day is still not taken
    db.execute("UPDATE signals SET status='tp', exit_ts=?, ret_r=0.8 WHERE id=?", (NOW + 3_600_000, s["id"]))
    later = cand("LATERUSDT", 10) | {"ts": NOW + 2 * 3_600_000}
    asyncio.run(eng.decide([later], None, NOW + 2 * 3_600_000))
    assert later["decision"] == "skipped:daily_trade_done"


def test_outcomes_on_5m_paths_stop_first_and_fees():
    bars = 289
    base = np.full((3, bars, 4), 100.0, dtype=np.float32)
    up = base.copy(); up[0, 10, 1] = 101.2                         # +1.2% high at bar 10 -> target hit
    up[1, 5, 2] = 98.9                                             # -1.1% low first -> stop
    up[2, 7, 1], up[2, 7, 2] = 101.5, 98.5                         # both in one bar -> counted as stop
    out = A.outcomes(up, np.array([1, 1, 1]))
    net, xbar, kind = out[(0.01, 0.01, 24)]
    f = fees()
    assert net[0] == pytest.approx(0.01 - f.entry - f.target) and xbar[0] == 10
    assert net[1] == pytest.approx(-0.01 - f.entry - f.stop)
    assert net[2] == pytest.approx(net[1])
    assert list(kind) == [A.TARGET, A.STOP, A.STOP]


def test_outcomes_takes_its_grid_as_arguments_and_labels_time_exits():
    """No module globals are changed (thread-safe), and a time exit on a wide target is labelled 'time' even when
    its net is within 0.3% of the target (the old guess `net >= target - 0.003` called it a target)."""
    bars = 289
    p = np.full((1, bars, 4), 100.0, dtype=np.float32)
    p[0, 200:, 1] = p[0, 200:, 3] = 104.8                          # +4.8% for the last hours, never touches +5%
    before = (A.TARGETS, A.STOPS, A.HOURS)
    o = A.outcomes(p, np.array([1]), targets=(0.05,), stops=(0.0075,), hours=(24,))[(0.05, 0.0075, 24)]
    assert (A.TARGETS, A.STOPS, A.HOURS) == before
    assert int(o.kind[0]) == A.TIME and A.EXIT_NAMES[int(o.kind[0])] == "time"
    assert o.net[0] == pytest.approx(0.048 - fees().entry - fees().time)


def test_daily_policy_takes_first_qualifying_and_waits_for_exit():
    day = A.DAY
    ts = np.array([0, 3_600_000, 7_200_000, day + 60_000, day + 120_000])
    score = np.array([1.0, 9.0, 9.0, 9.0, 9.0])
    net = np.array([0.01] * 5)
    exit_bar = np.array([5, 5, 5, 400, 5])                         # day-2 trade stays open past the next candidate
    taken = A.daily_policy(ts, score, 8.0, net, exit_bar)
    assert list(taken) == [1, 3]                                   # first qualifying per day; one at a time
