"""Shared data-loading + feature pipeline used by both the live scan and the replay."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import features as F
from .db import DB


@dataclass
class Dataset:
    klines: dict[str, pd.DataFrame] = field(default_factory=dict)     # regularized, indexed by ts
    oi: dict[str, pd.DataFrame] = field(default_factory=dict)
    funding: dict[str, pd.DataFrame] = field(default_factory=dict)
    lsr: dict[str, pd.DataFrame] = field(default_factory=dict)
    premium: dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: dict[str, dict] = field(default_factory=dict)


def load_dataset(db: DB, since_ts: int, until_ts: int | None = None, min_bars: int = 300) -> Dataset:
    ds = Dataset()
    ds.meta = {r["symbol"]: dict(r) for r in db.query("SELECT * FROM symbols")}
    raw = db.klines_many(since_ts, until_ts)
    for sym, k in raw.items():
        if len(k) >= min_bars:
            ds.klines[sym] = F.regularize(k)
    slow_since = since_ts - 2 * 86_400_000
    ds.oi = db.series_many("oi", ("oi",), slow_since)
    ds.lsr = db.series_many("lsr", ("top_pos", "global_acc"), slow_since)
    ds.funding = db.series_many("funding", ("rate",), since_ts - 5 * 86_400_000)
    ds.premium = db.series_many("premium", ("mark", "index_px", "funding_rate"), slow_since)
    if until_ts is not None:
        cut = until_ts + F.BAR_MS
        for store in (ds.oi, ds.lsr, ds.funding, ds.premium):
            for sym in list(store):
                store[sym] = store[sym][store[sym]["ts"] <= cut]
    return ds


PANEL_COLS = ["r_1h", "r_4h", "r_24h", "r_3d", "rvol_1h", "z_1h", "z_4h"]   # what market_context uses


def symbol_dataset(db: DB, sym: str, meta: dict, since_ts: int = 0, until_ts: int | None = None,
                   min_bars: int = 300) -> Dataset | None:
    """A Dataset holding a single symbol: same contents load_dataset would give for it."""
    k = db.klines(sym, since_ts, until_ts)
    if len(k) < min_bars:
        return None
    ds = Dataset(meta=meta)
    ds.klines[sym] = F.regularize(k)
    slow_since = since_ts - 2 * 86_400_000
    cut = until_ts + F.BAR_MS if until_ts is not None else None
    for name, table, cols, start in (("oi", "oi", ("oi",), slow_since),
                                     ("lsr", "lsr", ("top_pos", "global_acc"), slow_since),
                                     ("funding", "funding", ("rate",), since_ts - 5 * 86_400_000),
                                     ("premium", "premium", ("mark", "index_px", "funding_rate"), slow_since)):
        df = db.frame(f"SELECT ts,{','.join(cols)} FROM {table} WHERE symbol=? AND ts>=? ORDER BY ts", [sym, start])
        if cut is not None:
            df = df[df["ts"] <= cut].reset_index(drop=True)
        if len(df):
            getattr(ds, name)[sym] = df
    return ds


def streamed_context(db: DB, meta: dict, symbols: list[str], since_ts: int = 0, until_ts: int | None = None,
                     min_bars: int = 300):
    """Same result as build_context(load_dataset(...)) but reads one symbol at a time, keeping only
    the few columns the cross-sectional context needs (a fraction of the memory)."""
    panels, btc = {}, None
    for sym in symbols:
        if not (meta.get(sym, {}).get("active", 1) or sym == "BTCUSDT"):
            continue
        k = db.klines(sym, since_ts, until_ts)
        if len(k) < min_bars:
            continue
        kr = F.regularize(k)
        panels[sym] = F.panel_inputs(kr)[PANEL_COLS]
        if sym == "BTCUSDT":
            btc = kr
    return F.market_context(panels, btc)


def build_context(ds: Dataset, symbols: list[str] | None = None):
    """Market context from every symbol in the dataset that is an active crypto perp.

    Returns (market frame, rank frames, per-symbol panel inputs).
    """
    use = symbols if symbols is not None else [
        s for s in ds.klines if ds.meta.get(s, {}).get("active", 1) or s == "BTCUSDT"]
    panels = {s: F.panel_inputs(ds.klines[s]) for s in use}
    market, ranks = F.market_context(panels, ds.klines.get("BTCUSDT"))
    return market, ranks, panels


class DatasetCache:
    """Keeps the live feature window in memory and only reads new rows each cycle."""

    SLOW = (("oi", ("oi",)), ("lsr", ("top_pos", "global_acc")), ("funding", ("rate",)),
            ("premium", ("mark", "index_px", "funding_rate")))

    def __init__(self, db: DB, window_bars: int):
        self.db, self.window = db, window_bars
        self.ds: Dataset | None = None
        self.until = 0

    def get(self, bar_ts: int) -> Dataset:
        since = bar_ts - self.window * F.BAR_MS
        if self.ds is None or bar_ts <= self.until or bar_ts - self.until > 60 * F.BAR_MS:
            self.ds = load_dataset(self.db, since, until_ts=bar_ts)        # cold start or long gap
            self.until = bar_ts
            return self.ds
        ds = self.ds
        ds.meta = {r["symbol"]: dict(r) for r in self.db.query("SELECT * FROM symbols")}
        for sym, new in self.db.klines_many(self.until + F.BAR_MS, bar_ts).items():
            if sym in ds.klines:
                old = ds.klines[sym].reset_index().rename(columns={"index": "ts"})
                merged = F.regularize(pd.concat([old, new], ignore_index=True))
                ds.klines[sym] = merged[merged.index >= since]
            else:                                                             # newly listed / newly long enough
                full = self.db.klines(sym, since_ts=since, until_ts=bar_ts)
                if len(full) >= 300:
                    ds.klines[sym] = F.regularize(full)
        fresh_from = self.until - F.BAR_MS
        cut = bar_ts + F.BAR_MS
        for table, cols in self.SLOW:
            store = getattr(ds, table)
            keep_from = since - 5 * 86_400_000
            for sym, new in self.db.series_many(table, cols, fresh_from).items():
                old = store.get(sym)
                both = new if old is None else pd.concat([old, new], ignore_index=True)
                both = both.drop_duplicates("ts", keep="last").sort_values("ts")
                store[sym] = both[(both["ts"] >= keep_from) & (both["ts"] <= cut)].reset_index(drop=True)
        self.until = bar_ts
        return ds


def raw_frame(ds: Dataset, sym: str, market: pd.DataFrame, ranks: dict) -> pd.DataFrame:
    f = F.symbol_frame(ds.klines[sym], ds.oi.get(sym), ds.funding.get(sym), ds.lsr.get(sym),
                       ds.premium.get(sym), onboard_ts=ds.meta.get(sym, {}).get("onboard_ts"))
    return F.add_context(f, sym, market, ranks)


def features_to_json(row: pd.Series) -> str:
    out = {}
    for name, x in row.items():
        x = float(x)
        out[name] = None if (math.isnan(x) or math.isinf(x)) else float(f"{x:.6g}")
    return json.dumps(out, separators=(",", ":"))


def features_from_json(text: str) -> dict:
    return {k: (np.nan if v is None else v) for k, v in json.loads(text).items()}
