"""Shared paths and helpers for the research scripts (run from the repo root: python -m research.<name>).

Data lives in research_data/ (git-ignored): the event table, 5-minute and 1-minute candles fetched from Binance,
and a research copy of the local database. Every figure in docs/results/ is produced by a script in research/.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RD = ROOT / "research_data"
RESULTS = ROOT / "docs" / "results"
RD.mkdir(exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

LOCAL_DB = ROOT / "data" / "lts.sqlite"          # hourly/daily/funding history (Oct 2025 - 16 Nov 2026)
CANDLES_DB = RD / "candles.sqlite"               # 5m and 1m candles fetched for the studies
EVENTS = RD / "aim_events.parquet"               # the year's early-grind moments (grindlab.build)

# The year studies (DESIGN.md sections 15-27) built their event table at this moment (transcript of 16 Nov 2026).
YEAR_NOW = int(datetime(2026, 11, 16, 23, 35, tzinfo=timezone.utc).timestamp() * 1000)
DAY, HOUR, BAR, MIN = 86_400_000, 3_600_000, 300_000, 60_000


def costs():
    """Fees and slippage from config.trade (the one place they are defined)."""
    from lts.config import load_config
    return load_config().trade


def candles_db() -> sqlite3.Connection:
    c = sqlite3.connect(CANDLES_DB)
    c.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS k5 (symbol TEXT, ts INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL, qv REAL,
                                   trades INTEGER, tbv REAL, tbqv REAL, PRIMARY KEY (symbol, ts)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS k1 (symbol TEXT, ts INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL, qv REAL,
                                   trades INTEGER, tbv REAL, tbqv REAL, PRIMARY KEY (symbol, ts)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS fetched (tbl TEXT, symbol TEXT, start INTEGER, end INTEGER, n INTEGER,
                                        PRIMARY KEY (tbl, symbol, start)) WITHOUT ROWID;
    """)
    return c


def load_paths(ev, post_bars: int = 289, pre_bars: int = 0, table: str = "k5", step: int = BAR):
    """5-minute (or 1-minute) candles around each event (columns symbol, ts) from candles.sqlite.
    Returns (post, pre): post (n, post_bars, 4) o,h,l,c from ts on; pre (n, pre_bars, 6) o,h,l,c,qv,tbqv before ts.
    Missing candles are NaN."""
    import numpy as np
    import pandas as pd
    conn = candles_db()
    post = np.full((len(ev), post_bars, 4), np.nan, np.float32)
    pre = np.full((len(ev), pre_bars, 6), np.nan, np.float32)
    ev = ev.reset_index(drop=True)
    for sym, g in ev.groupby("symbol"):
        lo, hi = int(g.ts.min()) - pre_bars * step, int(g.ts.max()) + post_bars * step
        k = pd.read_sql_query(f"SELECT ts,o,h,l,c,qv,tbqv FROM {table} WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts",
                              conn, params=(sym, lo, hi))
        if not len(k):
            continue
        kts = k.ts.to_numpy()
        vals = k[["o", "h", "l", "c", "qv", "tbqv"]].to_numpy(np.float32)
        for i, t in zip(g.index, g.ts.to_numpy()):
            a = np.searchsorted(kts, t)
            b = np.searchsorted(kts, t + post_bars * step)
            idx = ((kts[a:b] - t) // step).astype(int)
            post[i, idx] = vals[a:b, :4]
            if pre_bars:
                a2 = np.searchsorted(kts, t - pre_bars * step)
                idx2 = ((kts[a2:a] - (t - pre_bars * step)) // step).astype(int)
                pre[i, idx2] = vals[a2:a]
    conn.close()
    return post, pre


def r3_mask(a):
    """DESIGN.md section 25's R3 (all-day steady grind) on the event table's side-aligned factors."""
    return ((a.r_24h >= 0.08) & (a.r2_24h >= 0.8) & (a.hours_with24 >= 15) & (a.r2_6h >= 0.9) & (a.hours_with6 >= 5)
            & (a.big6 <= 0.35)).to_numpy()


def halves(ts):
    """The year studies' split: first half of the calendar months (discovery) / second half (confirmation)."""
    import numpy as np
    import pandas as pd
    month = pd.to_datetime(ts, unit="ms").strftime("%Y-%m").to_numpy()
    months = sorted(set(month))
    return np.isin(month, months[:len(months) // 2]), month


class Tee:
    """Print to the console and to a results file at the same time."""

    def __init__(self, name: str):
        self.f = open(RESULTS / name, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()
