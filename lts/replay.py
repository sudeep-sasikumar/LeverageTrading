"""Replay stored history through the live pipeline to (re)build the labelled candidate set."""
from __future__ import annotations

import json
import logging
import time

import numpy as np

from . import BAR_MS
from . import features as F
from .candidates import apply_cooldown, kind_of, trigger_mask
from .config import Config
from .db import DB
from .labels import label_path
from .pipeline import features_to_json, raw_frame, streamed_context, symbol_dataset
from .scoring import rule_score

log = logging.getLogger("lts.replay")

INSERT = """INSERT OR REPLACE INTO candidates
    (symbol, ts, side, kind, source, ref_price, atr_pct, sl_pct, features, rule_score,
     label, outcome, ret_r, exit_ts, mfe, mae, grid, labeled_ts)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def replay(cfg: Config, db: DB, days: float = 0.0, rebuild: bool = True) -> int:
    t0 = time.time()
    last = db.last_kline_ts()
    if not last:
        log.warning("no klines stored - run backfill first")
        return 0
    meta = {r["symbol"]: dict(r) for r in db.query("SELECT * FROM symbols")}
    symbols = sorted(last)
    # one symbol at a time: holding all history at once needs >2 GB, too much for a shared VPS
    market, ranks = streamed_context(db, meta, symbols)
    log.info("context built for %d symbols in %.0fs", len(symbols), time.time() - t0)
    if rebuild:
        db.execute("DELETE FROM candidates WHERE source='replay'")
    last_ts = max(last.values())
    min_ts = last_ts - int(days * 86_400_000) if days > 0 else 0
    tc = cfg.trade
    max_h = max([tc.horizon_bars] + list(tc.grid_horizon))
    now = int(time.time() * 1000)
    total = 0
    for n, sym in enumerate(symbols, 1):
        if not meta.get(sym, {}).get("active", 0):
            continue
        ds = symbol_dataset(db, sym, meta)
        if ds is None:
            continue
        k = ds.klines[sym]
        f = raw_frame(ds, sym, market, ranks)
        o, h, l, c = (k[x].to_numpy() for x in ("o", "h", "l", "c"))
        idx = k.index.to_numpy()
        atr = f["atr_pct"].to_numpy()
        rows = []
        for side in (1, -1):
            hits = np.flatnonzero(apply_cooldown(trigger_mask(f, cfg, side), cfg.trigger.cooldown_bars))
            hits = hits[(idx[hits] >= min_ts) & (hits + 1 + max_h <= len(idx))]
            if len(hits) == 0:
                continue
            aligned = F.align(f.iloc[hits], side)
            for j, i in enumerate(hits):
                s = i + 1
                lab = label_path(tc, side, atr[i], o[s:s + max_h], h[s:s + max_h], l[s:s + max_h], c[s:s + max_h],
                                 ref_price=float(c[i]))
                if lab is None:
                    continue
                arow = aligned.iloc[j]
                feat_json = features_to_json(arow)
                fd = json.loads(feat_json)
                ts = int(idx[i]) + BAR_MS
                rows.append((sym, ts, side, kind_of(fd), "replay", float(c[i]), float(atr[i]), lab["sl_pct"],
                             feat_json, rule_score(fd), lab["label"], lab["outcome"], lab["ret_r"],
                             ts + lab["bars"] * BAR_MS, lab["mfe"], lab["mae"],
                             json.dumps(lab["grid"], separators=(",", ":")), now))
        if rows:
            db.executemany(INSERT, rows)
            total += len(rows)
        if n % 50 == 0:
            log.info("replay %d/%d symbols, %d candidates (%.0fs)", n, len(symbols), total, time.time() - t0)
    log.info("replay done: %d labelled candidates in %.0fs", total, time.time() - t0)
    return total
