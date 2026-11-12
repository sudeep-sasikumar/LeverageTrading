"""SQLite storage. All timestamps are integer milliseconds UTC; kline ts is bar open time."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT PRIMARY KEY,
    src TEXT NOT NULL,                 -- data source: 'binance' or 'weex'
    on_binance INTEGER NOT NULL DEFAULT 0,
    on_weex INTEGER NOT NULL DEFAULT 0,
    base TEXT,
    underlying_type TEXT,
    onboard_ts INTEGER,
    weex_max_lev INTEGER,
    weex_taker_fee REAL,
    weex_maker_fee REAL,
    price_precision INTEGER,
    active INTEGER NOT NULL DEFAULT 1,
    updated_ts INTEGER
);
CREATE TABLE IF NOT EXISTS klines (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL,
    o REAL, h REAL, l REAL, c REAL,
    v REAL, qv REAL, trades INTEGER, tbv REAL, tbqv REAL,
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_klines_ts ON klines (ts);
CREATE TABLE IF NOT EXISTS oi (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL, oi REAL, oi_value REAL,
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_oi_ts ON oi (ts);
CREATE TABLE IF NOT EXISTS funding (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL, rate REAL,
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS premium (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL,
    mark REAL, index_px REAL, funding_rate REAL, next_funding_ts INTEGER,
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_premium_ts ON premium (ts);
CREATE TABLE IF NOT EXISTS lsr (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL,
    top_pos REAL, global_acc REAL,
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS weex_ticker (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL,
    last REAL, qv24 REAL, chg24 REAL, mark REAL,
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    ts INTEGER NOT NULL,               -- decision time = close time of the trigger bar
    side INTEGER NOT NULL,             -- +1 long, -1 short
    kind TEXT,
    source TEXT NOT NULL,              -- 'live' or 'replay'
    ref_price REAL,
    atr_pct REAL,
    sl_pct REAL,
    features TEXT NOT NULL,            -- JSON {name: value}
    rule_score REAL,
    model_p REAL,
    model_version TEXT,
    decision TEXT,                     -- sent / skipped:<reason>
    label INTEGER,                     -- 1 = TP first (primary config), 0 otherwise
    outcome TEXT,                      -- tp / sl / time
    ret_r REAL,                        -- net R multiple of the primary config
    exit_ts INTEGER,
    mfe REAL, mae REAL,                -- max favourable / adverse excursion (fraction) within horizon
    grid TEXT,                         -- JSON outcomes for alternative barrier configs
    labeled_ts INTEGER,
    UNIQUE (symbol, ts, side, source)
);
CREATE INDEX IF NOT EXISTS idx_cand_ts ON candidates (ts);
CREATE INDEX IF NOT EXISTS idx_cand_unlabeled ON candidates (labeled_ts) WHERE labeled_ts IS NULL;
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER,
    symbol TEXT NOT NULL,
    side INTEGER NOT NULL,
    ts INTEGER NOT NULL,
    grade TEXT,
    entry REAL, sl REAL, tp REAL,
    sl_pct REAL, tp_pct REAL,
    leverage INTEGER,
    margin_frac REAL,
    p REAL, ev_r REAL,
    model_version TEXT,
    expires_ts INTEGER,                -- time stop
    status TEXT NOT NULL DEFAULT 'open',   -- open / tp / sl / time / cancelled
    exit_ts INTEGER, exit_price REAL,
    ret_r REAL, ret_margin REAL,
    mfe REAL, mae REAL,
    last_checked_ts INTEGER,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_sig_status ON signals (status);
CREATE TABLE IF NOT EXISTS models (
    version TEXT PRIMARY KEY,
    trained_ts INTEGER,
    n_train INTEGER,
    metrics TEXT,
    threshold REAL,
    floor REAL,
    path TEXT,
    status TEXT                        -- champion / retired / rejected
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""


class DB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=60)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-200000")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- generic helpers -------------------------------------------------
    def execute(self, sql: str, params: Sequence = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def executemany(self, sql: str, rows: Iterable[Sequence]) -> None:
        self.conn.executemany(sql, rows)
        self.conn.commit()

    def query(self, sql: str, params: Sequence = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def frame(self, sql: str, params: Sequence = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=list(params))

    # ---- key/value state -------------------------------------------------
    def get(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value) -> None:
        self.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, json.dumps(value)))

    # ---- writers ---------------------------------------------------------
    def upsert_symbols(self, rows: list[dict]) -> None:
        now = int(time.time() * 1000)
        self.executemany(
            """INSERT INTO symbols(symbol,src,on_binance,on_weex,base,underlying_type,onboard_ts,
                                   weex_max_lev,weex_taker_fee,weex_maker_fee,price_precision,active,updated_ts)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol) DO UPDATE SET src=excluded.src,on_binance=excluded.on_binance,
                   on_weex=excluded.on_weex,base=excluded.base,underlying_type=excluded.underlying_type,
                   onboard_ts=COALESCE(excluded.onboard_ts,symbols.onboard_ts),
                   weex_max_lev=excluded.weex_max_lev,weex_taker_fee=excluded.weex_taker_fee,
                   weex_maker_fee=excluded.weex_maker_fee,price_precision=excluded.price_precision,
                   active=excluded.active,updated_ts=excluded.updated_ts""",
            [(r["symbol"], r["src"], int(r["on_binance"]), int(r["on_weex"]), r.get("base"),
              r.get("underlying_type"), r.get("onboard_ts"), r.get("weex_max_lev"), r.get("weex_taker_fee"),
              r.get("weex_maker_fee"), r.get("price_precision"), int(r.get("active", 1)), now) for r in rows])

    def insert_klines(self, symbol: str, rows: list[tuple]) -> None:
        """rows: (ts, o, h, l, c, v, qv, trades, tbv, tbqv)"""
        self.executemany(
            "INSERT OR REPLACE INTO klines(symbol,ts,o,h,l,c,v,qv,trades,tbv,tbqv) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            [(symbol, *r) for r in rows])

    def insert_rows(self, table: str, cols: Sequence[str], rows: list[tuple]) -> None:
        if not rows:
            return
        marks = ",".join("?" * len(cols))
        self.executemany(f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES({marks})", rows)

    # ---- readers ---------------------------------------------------------
    def symbols(self, active_only: bool = True) -> pd.DataFrame:
        sql = "SELECT * FROM symbols" + (" WHERE active=1" if active_only else "")
        return self.frame(sql)

    def klines(self, symbol: str, since_ts: int = 0, until_ts: int | None = None) -> pd.DataFrame:
        sql = "SELECT ts,o,h,l,c,v,qv,trades,tbv,tbqv FROM klines WHERE symbol=? AND ts>=?"
        params: list = [symbol, since_ts]
        if until_ts is not None:
            sql += " AND ts<=?"
            params.append(until_ts)
        return self.frame(sql + " ORDER BY ts", params)

    def klines_many(self, since_ts: int, until_ts: int | None = None) -> dict[str, pd.DataFrame]:
        sql = "SELECT symbol,ts,o,h,l,c,v,qv,trades,tbv,tbqv FROM klines WHERE ts>=?"
        params: list = [since_ts]
        if until_ts is not None:
            sql += " AND ts<=?"
            params.append(until_ts)
        df = self.frame(sql + " ORDER BY symbol, ts", params)
        return {s: g.drop(columns="symbol").reset_index(drop=True) for s, g in df.groupby("symbol", sort=False)}

    def series_many(self, table: str, cols: Sequence[str], since_ts: int) -> dict[str, pd.DataFrame]:
        df = self.frame(f"SELECT symbol,ts,{','.join(cols)} FROM {table} WHERE ts>=? ORDER BY symbol, ts", [since_ts])
        return {s: g.drop(columns="symbol").reset_index(drop=True) for s, g in df.groupby("symbol", sort=False)}

    def last_kline_ts(self) -> dict[str, int]:
        return {r["symbol"]: r["m"] for r in self.query("SELECT symbol, MAX(ts) AS m FROM klines GROUP BY symbol")}

    def prune(self, keep_days: int) -> None:
        cutoff = int((time.time() - keep_days * 86400) * 1000)
        for table in ("klines", "oi", "premium", "weex_ticker", "lsr"):
            self.execute(f"DELETE FROM {table} WHERE ts<?", (cutoff,))
