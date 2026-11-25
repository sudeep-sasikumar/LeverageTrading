"""Review items 8 and 9 (25 Nov 2026): open-signal tracking starts after the signal's own minute, pages back
through long downtimes, and records stretches it could not fill (docs/DESIGN.md section 28)."""
import asyncio
import time
from pathlib import Path

from lts.config import Config
from lts.db import DB
from lts.tracker import fetch_minutes, first_minute, missing, track_open_signals

ROOT = Path(__file__).resolve().parent.parent
HOUR, MIN = 3_600_000, 60_000
T0 = 1_794_900_000_000 // HOUR * HOUR


# ---- item 8: the tracker starts at the first full minute after the signal ----------------------------------
def test_tracker_starts_at_the_first_full_minute_after_the_signal():
    sig = {"ts": T0 + 20_000, "last_checked_ts": T0 + 20_000}           # sent 20s into a minute
    assert first_minute(sig) == T0 + MIN
    assert first_minute({"ts": T0 + 20_000, "last_checked_ts": T0 + 5 * MIN}) == T0 + 5 * MIN
    assert first_minute({"ts": T0, "last_checked_ts": None}) == T0


class FakeVenue:
    def __init__(self, name, bars, fail=False):
        self.name, self.bars, self.fail, self.calls = name, bars, fail, []

    async def klines_range(self, sym, interval, start, end):
        self.calls.append((start, end))
        if self.fail:
            raise RuntimeError("down")
        return [b for b in self.bars if start <= b[0] < end]


def bar(t, o=100.0, h=100.1, l=99.9, c=100.0):
    return (t, o, h, l, c, 0, 0, 0, 0, 0)


def test_tracker_ignores_the_candle_the_signal_was_sent_in(tmp_path):
    db = DB(tmp_path / "t.sqlite")
    sent = T0 + 30_000
    db.execute("""INSERT INTO signals(symbol, side, ts, grade, entry, sl, tp, sl_pct, tp_pct, leverage, kind, expires_ts,
                  status, last_checked_ts) VALUES ('XUSDT', 1, ?, 'P', 100, 99, 102, 0.01, 0.02, 5, 'grindsig', ?, 'open', ?)""",
               (sent, sent + 24 * HOUR, sent))
    # the minute the signal was sent in dipped to 98.9 BEFORE entry; the minutes after it are calm
    bars = [bar(T0, l=98.9)] + [bar(T0 + i * MIN) for i in range(1, 5)]
    venue = FakeVenue("weex", bars)

    class Tg:
        async def send(self, text):
            return True
    real = time.time
    time.time = lambda: (T0 + 5 * MIN + 1000) / 1000
    try:
        asyncio.run(track_open_signals(Config(), db, venue, None, Tg()))      # not on Weex: tracked on Binance
    finally:
        time.time = real
    s = dict(db.query("SELECT status, last_checked_ts FROM signals")[0])
    assert s["status"] == "open" and s["last_checked_ts"] == T0 + 5 * MIN
    assert venue.calls[0][0] == T0 + MIN


# ---- item 9: long downtime is paged and gaps are recorded -------------------------------------------------
def test_tracker_fills_a_long_gap_from_the_other_venue_and_records_what_is_left():
    start, end = T0, T0 + 20 * HOUR                                     # 20h offline: 1,200 minutes
    full = [bar(start + i * MIN) for i in range(1200)]
    primary = FakeVenue("weex", [b for b in full if not (start + 2 * HOUR <= b[0] < start + 3 * HOUR)])
    fallback = FakeVenue("binance", [b for b in full if not (start + 2 * HOUR <= b[0] < start + 2 * HOUR + 30 * MIN)])
    bars, gaps = asyncio.run(fetch_minutes("XUSDT", start, end, primary, fallback))
    assert len(bars) == 1200 - 30 and gaps == [[start + 2 * HOUR, start + 2 * HOUR + 30 * MIN]]
    bars, gaps = asyncio.run(fetch_minutes("XUSDT", start, end, FakeVenue("weex", [], fail=True), None))
    assert bars == [] and gaps == [[start, end]]
    assert missing([bar(start), bar(start + MIN)], start, start + 2 * MIN) == []


def test_weex_history_klines_pages_backwards(monkeypatch):
    from lts.exchanges import Weex
    w = Weex()
    data = [[T0 + i * MIN, "1", "1", "1", "1", "0", 0, "0", 0, "0", "0"] for i in range(250)]

    async def get(path, params=None, **kw):
        assert path == "/capi/v3/market/historyKlines" and params["limit"] == 100
        rows = [r for r in data if params["startTime"] <= r[0] <= params["endTime"]]
        return list(reversed(rows[-100:]))                              # newest first, at most 100
    monkeypatch.setattr(w.http, "get", get)
    rows = asyncio.run(w.klines_range("XUSDT", "1m", T0, T0 + 250 * MIN))
    assert len(rows) == 250 and rows[0][0] == T0 and rows[-1][0] == T0 + 249 * MIN


