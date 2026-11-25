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


class Tee:
    """Print to the console and to a results file at the same time."""

    def __init__(self, name: str):
        self.f = open(RESULTS / name, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()
