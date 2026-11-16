import json
import sqlite3

import numpy as np
import pandas as pd

from lts import BAR_MS
from lts import features as F
from lts.candidates import grind_mask, grind_rank
from lts.config import Config
from lts.db import DB, SCHEMA
from lts.labels import label_grind, plan_grind
from lts.movers import is_grinder, path_stats, review, review_text
from lts.signals import RiskState

START = 1_794_000_000_000 // BAR_MS * BAR_MS


def klines(path: np.ndarray, start=START, seed=0, vol=200.0):
    """5m klines following `path` (closes), with small realistic wicks."""
    rng = np.random.default_rng(seed)
    c = np.asarray(path, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.0008, len(c)))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.0008, len(c)))
    v = rng.uniform(0.5, 1.5, len(c)) * vol
    return pd.DataFrame({"ts": start + np.arange(len(c)) * BAR_MS, "o": o, "h": h, "l": l, "c": c, "v": v,
                         "qv": v * c, "trades": rng.integers(100, 500, len(c)), "tbv": v / 2, "tbqv": v * c / 2})


def noisy_flat(n, seed, level=100.0, sd=0.0015):
    rng = np.random.default_rng(seed)
    return level * np.exp(np.cumsum(rng.normal(0, sd, n)))


def frame_for(path, seed=1):
    k = F.regularize(klines(path, seed=seed))
    btc = F.regularize(klines(noisy_flat(len(path), 99), seed=98))
    market, ranks = F.market_context({"AAA": F.panel_inputs(k), "BTCUSDT": F.panel_inputs(btc)}, btc)
    f = F.symbol_frame(k, onboard_ts=START - 60 * 86_400_000)
    return F.add_context(f, "AAA", market, ranks)


def grind_cfg():
    cfg = Config()
    cfg.universe.min_qvol_24h = 0
    cfg.grind.require_market_aligned = False
    return cfg


def test_grind_trigger_fires_on_a_steady_trend_but_not_on_a_spike():
    n = 2400
    base = noisy_flat(n - 96, 3)
    rng = np.random.default_rng(4)
    steady = base[-1] * np.exp(np.cumsum(0.0008 + rng.normal(0, 0.0012, 96)))       # ~+1%/hour for 8h, small noise
    f = frame_for(np.concatenate([base, steady]))
    cfg = grind_cfg()
    assert grind_mask(f.iloc[[-1]], cfg, 1)[0]
    assert not grind_mask(f.iloc[[-1]], cfg, -1)[0]
    last = F.align(f.iloc[[-1]], 1).iloc[0].to_dict()
    assert last["g_r2_6h"] > 0.8 and last["g_hours_with"] >= 5 and last["g_big1h"] < 0.45 and grind_rank(last) > 0

    spike = np.concatenate([base, np.full(96, base[-1])])
    spike[-30:] *= 1.07                                                             # one +7% jump 2.5h ago, flat after
    f2 = frame_for(spike)
    assert not grind_mask(f2.iloc[[-1]], cfg, 1)[0]
    assert F.align(f2.iloc[[-1]], 1).iloc[0]["g_big1h"] > 0.9


def test_grind_plan_and_label():
    cfg = Config()
    plan = plan_grind(cfg.trade, cfg.grind, 0.016)                 # 1.6% hourly ATR -> 4% stop, 4% target
    assert abs(plan["sl_pct"] - 0.04) < 1e-12 and abs(plan["tp_pct"] - 0.04) < 1e-12
    assert plan["horizon_bars"] == 288 and plan["gain_margin"] >= 0.10 and plan["leverage"] <= 10
    assert plan_grind(cfg.trade, cfg.grind, 0.05) is None          # would need a 12.5% stop: skipped
    n = cfg.grind.horizon_bars
    flat = np.full(n, 100.0)
    h = flat.copy(); h[100] = 104.5
    lab = label_grind(cfg.trade, cfg.grind, 1, 0.016, flat, h, flat, flat)
    assert lab["outcome"] == "tp" and lab["label"] == 1 and set(lab["grid"]) == {"1.5|1.0|288", "1.5|2.0|288",
                                                                                 "2.5|1.0|288", "2.5|2.0|288"}
    assert label_grind(cfg.trade, cfg.grind, 1, 0.016, flat[:50], flat[:50], flat[:50], flat[:50]) is None


def test_path_stats_tell_a_grind_from_a_spike():
    rng = np.random.default_rng(5)
    grind = 100 * np.exp(np.cumsum(-0.0006 + rng.normal(0, 0.0015, 288)))        # steady ~ -16% day
    st = path_stats(100.0, grind)
    assert st["side"] == -1 and st["ret24"] < -0.10 and st["r2"] > 0.8 and is_grinder(st)
    jump = np.full(288, 100.0); jump[200:] = 85.0                                 # one -15% candle
    st2 = path_stats(100.0, jump + rng.normal(0, 0.05, 288))
    assert st2["big1h"] > 0.9 and not is_grinder(st2)


def test_daily_review_finds_the_grinder_and_measures_how_late_it_was_flagged(tmp_path):
    cfg = Config()
    cfg.universe.min_qvol_24h = 0
    db = DB(tmp_path / "m.sqlite")
    n = 8 * 288 + 288
    rng = np.random.default_rng(6)
    down = np.concatenate([noisy_flat(n - 288, 7), np.zeros(288)])
    down[-288:] = down[-289] * np.exp(np.cumsum(-0.0006 + rng.normal(0, 0.0012, 288)))
    series = {"BTCUSDT": noisy_flat(n, 8), "GRDUSDT": down, "FLATUSDT": noisy_flat(n, 9)}
    db.upsert_symbols([{"symbol": s, "src": "binance", "on_binance": 1, "on_weex": 1, "active": 1} for s in series])
    for i, (s, path) in enumerate(series.items()):
        k = klines(path, seed=10 + i)
        db.insert_klines(s, [tuple(r) for r in k[["ts", "o", "h", "l", "c", "v", "qv", "trades", "tbv", "tbqv"]].to_numpy()])
    end = START + n * BAR_MS
    flag_ts = end - 288 * BAR_MS + 96 * BAR_MS                    # flagged 8h into the day
    flag_px = float(down[-288 + 95])
    db.execute("INSERT INTO candidates(symbol, ts, side, kind, source, ref_price, atr_pct, features) "
               "VALUES ('GRDUSDT', ?, -1, 'grind', 'live', ?, 0.01, '{}')", (flag_ts, flag_px))
    res = review(cfg, db)
    syms = [m["symbol"] for m in res["movers"]]
    assert syms == ["GRDUSDT"], syms
    m = res["movers"][0]
    total = np.log(down[-1] / down[-289]) * -1
    assert m["flagged_kind"] == "grind" and abs(m["flagged_left"] - np.log(down[-1] / flag_px) * -1 / total) < 1e-6
    assert db.query("SELECT COUNT(*) c FROM daily_movers")[0]["c"] == 1 and db.get("movers_latest")["day"]
    assert "GRDUSDT" in review_text(res) and "still ahead" in review_text(res)


def test_grind_and_momentum_limits_are_independent(tmp_path):
    cfg = Config()
    db = DB(tmp_path / "r.sqlite")
    now = START + 20 * 3_600_000
    for i, sym in enumerate(("G1USDT", "G2USDT")):
        db.execute("""INSERT INTO signals(symbol, side, ts, grade, entry, sl, tp, sl_pct, tp_pct, leverage, status, kind,
                      expires_ts) VALUES (?, -1, ?, 'P', 1, 1.04, 0.96, 0.04, 0.04, 3, 'open', 'grind', ?)""",
                   (sym, now - (i + 1) * 5 * 3_600_000, now + 86_400_000))
    assert RiskState(cfg, db, now, "grind").block_reason(cfg, "G3USDT") == "max_open"
    assert RiskState(cfg, db, now, "momentum").block_reason(cfg, "M1USDT") is None
    assert RiskState(cfg, db, now, "momentum").block_reason(cfg, "G1USDT") == "already_open"   # never both tracks


def test_old_database_is_migrated_without_losing_data(tmp_path):
    path = tmp_path / "old.sqlite"
    old = SCHEMA.replace("UNIQUE (symbol, ts, side, source, kind)", "UNIQUE (symbol, ts, side, source)") \
                .replace("    kind TEXT,                         -- momentum / grind\n", "")
    conn = sqlite3.connect(path)
    conn.executescript(old)
    conn.execute("INSERT INTO candidates(symbol, ts, side, kind, source, features) VALUES ('X', 1, 1, 'trend', 'live', '{}')")
    conn.execute("INSERT INTO signals(symbol, side, ts, status) VALUES ('X', 1, 1, 'open')")
    conn.commit(); conn.close()
    db = DB(path)
    assert "UNIQUE (symbol, ts, side, source, kind)" in db.query("SELECT sql FROM sqlite_master WHERE name='candidates'")[0][0]
    assert db.query("SELECT COUNT(*) c FROM candidates")[0]["c"] == 1
    db.execute("INSERT INTO candidates(symbol, ts, side, kind, source, features) VALUES ('X', 1, 1, 'grind', 'live', '{}')")
    assert db.query("SELECT COUNT(*) c FROM candidates")[0]["c"] == 2                     # same bar, both kinds kept
    assert db.query("SELECT kind FROM signals")[0]["kind"] == "momentum"
    assert db.query("SELECT 1 FROM sqlite_master WHERE name='idx_cand_ts'")
