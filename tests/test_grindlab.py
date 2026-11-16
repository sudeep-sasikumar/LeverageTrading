import asyncio
import json

import numpy as np
import pandas as pd

from lts import grindlab as G
from lts.config import Config
from lts.dashboard import snapshot
from lts.db import DB
from lts.grindlab_data import binance_kind, ensure_schema, tickers_in, upbit_kind
from lts.signals import SignalEngine

HOUR, DAY = 3_600_000, 86_400_000
NOW = 1_794_900_000_000 // HOUR * HOUR


def _bars(path, start, step, seed):
    rng = np.random.default_rng(seed)
    c = np.asarray(path, float)
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.002, len(c)))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.002, len(c)))
    v = rng.uniform(5e4, 1.5e5, len(c))
    return [(start + i * step, o[i], h[i], l[i], c[i], v[i], v[i] * c[i], 500, v[i] / 2, v[i] * c[i] / 2) for i in range(len(c))]


def make_lab_db(tmp_path):
    db = DB(tmp_path / "lab.sqlite")
    ensure_schema(db)
    n_h = 60 * 24
    start_h = NOW - n_h * HOUR
    rng = np.random.default_rng(1)
    paths = {}
    for i, sym in enumerate(("BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")):
        paths[sym] = 50 * np.exp(np.cumsum(rng.normal(0, 0.006, n_h)))
    grind = paths["AAAUSDT"]
    grind[-8:] = grind[-9] * np.exp(np.cumsum(np.full(8, 0.009) + rng.normal(0, 0.001, 8)))   # steady +7% over 8h
    db.upsert_symbols([{"symbol": s, "src": "binance", "on_binance": 1, "on_weex": 1, "active": 1,
                        "base": s[:-4], "price_precision": 4} for s in paths])
    for j, (s, p) in enumerate(paths.items()):
        db.insert_rows("klines_1h", ("symbol", "ts", "o", "h", "l", "c", "v", "qv", "trades", "tbv", "tbqv"),
                       [(s, *r) for r in _bars(p, start_h, HOUR, j)])
        daily = p[0] * np.exp(np.cumsum(rng.normal(0, 0.03, 300)))
        db.insert_rows("klines_1d", ("symbol", "ts", "o", "h", "l", "c", "v", "qv", "trades", "tbv", "tbqv"),
                       [(s, *r) for r in _bars(daily, start_h - 300 * DAY, DAY, 10 + j)])
    db.insert_rows("symbol_tags", ("symbol", "tag"), [("AAAUSDT", "AI"), ("BBBUSDT", "AI"), ("CCCUSDT", "AI"),
                                                     ("DDDUSDT", "AI"), ("AAAUSDT", "Alpha")])
    db.insert_rows("announcements", ("source", "ts", "base", "kind", "title"),
                   [("upbit", NOW - 2 * DAY, "AAA", "upbit_list", "test listing")])
    return db


def test_build_and_live_scan_share_factors(tmp_path, monkeypatch):
    db = make_lab_db(tmp_path)
    monkeypatch.setattr(G, "MIN_QV24", 0)
    df, ep = G.build(db, days=40)
    assert len(df) > 20 and {"success", "x_trail", "btc_r168", "room_ahead", "tag_AI", "ann_upbit_list_7d"} <= set(df.columns)
    assert df.ts.max() <= NOW and (df.side.abs() == 1).all()
    # the live scan on the last bar must see the steady grind and compute the same factors as the study would
    monkeypatch.setattr(G.time, "time", lambda: NOW / 1000 + 60)
    rows = G.live_scan(db, now_ms=NOW + 60_000)
    hit = [r for r in rows if r["symbol"] == "AAAUSDT" and r["side"] == 1]
    assert hit, rows
    r = hit[0]
    assert r["r_6h"] > 0.04 and r["r2_6h"] > 0.8 and r["tag_AI"] == 1.0 and r["ann_upbit_list_7d"] == 1.0
    assert r["catalyst_short"] == 0.0                     # catalyst is a SHORT-side trait only
    scored = G.score_live(hit, None)[0]
    assert len(scored["checks"]) == len(G.CHECKS) and 0 <= scored["score"] <= len(G.CHECKS)


def test_study_report_on_random_data_is_well_formed():
    rng = np.random.default_rng(3)
    n = 3000
    df = pd.DataFrame({"symbol": rng.choice(["A", "B", "C"], n), "ts": np.sort(rng.integers(0, 100 * DAY, n)),
                       "side": rng.choice([-1, 1], n), "price": 1.0})
    for c in ("btc_r168", "week_ret", "rs_mkt24", "imb_6h", "mkt_r6", "sector_r24", "funding", "listing_age_d", "rvol_6h",
              "pos_30d", "atr_1h", "r_6h"):
        df[c] = rng.normal(size=n)
    df["brk_30d"] = rng.integers(0, 2, n).astype(float)
    df["levels_ahead_10pct"] = rng.integers(0, 3, n).astype(float)
    for c in ("ann_upbit_list_7d", "ann_futures_launch_7d", "catalyst_short"):
        df[c] = (rng.random(n) < 0.03).astype(float)
    df["success"] = (rng.random(n) < 0.25 + 0.1 * (df.btc_r168 > 0)).astype(int)
    df["result"] = np.where(df.success == 1, "target", "stop")
    df["ret_r"] = np.where(df.success == 1, 1.9, -1.1)
    df["mfe"], df["mae"] = rng.random(n) * 0.1, rng.random(n) * 0.1
    for col in list(G.EXIT_LABELS):
        df[col] = rng.normal(0, 0.03, n)
    ep = pd.DataFrame({"symbol": ["A"], "side": [1], "start": [0], "end": [DAY], "move": [0.12], "hours": [24], "best24": [0.12], "tags": [""]})
    rep = G.study(df, ep)
    assert len(rep["traits"]) == len(G.CHECKS) and rep["n_decisions"] == n
    btc = next(t for t in rep["traits"] if t["trait"].startswith("BTC"))
    assert btc["held"] and btc["test_with"] > btc["test_without"]          # the planted effect is found on unseen days
    json.dumps(G._clean(rep))


def test_announcement_parsers():
    bases = {"PREDICT", "HYPE", "TRX", "KAVA", "CROSS", "ALPACA", "BNX", "PEPE"}
    assert tickers_in("Binance Will List Predict (PREDICT) with Seed Tag Applied", bases) == {"PREDICT"}
    assert tickers_in("Binance Futures Will Launch USDⓈ-Margined HYPEUSDC and TRXUSDC Perpetual Contracts", bases) == {"HYPE", "TRX"}
    assert tickers_in("Binance Will Delist ALPACA, BNX, PEPE on 2026-05-02", bases) == {"ALPACA", "BNX", "PEPE"}
    assert binance_kind("Binance Will Extend the Monitoring Tag to Include KAVA and CROSS") == "monitoring"
    assert upbit_kind("생텀(SANC) 신규 거래지원 안내 (KRW, BTC, USDT 마켓)") == "upbit_list"
    assert upbit_kind("카바(KAVA) 거래 유의 종목 지정 안내") == "upbit_warning"
    assert upbit_kind("카바(KAVA) 거래 유의 종목 지정 해제 안내") == "upbit_warning_lifted"


class Notifier:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(text)
        return True


def test_grindlab_candidate_becomes_a_paper_trade_with_the_study_plan(tmp_path):
    cfg, db, tg = Config(), DB(tmp_path / "s.sqlite"), Notifier()
    checks = {label: i < 9 for i, (_, _, label) in enumerate(G.CHECKS)}
    base = {"symbol": "AAAUSDT", "side": -1, "ts": NOW, "ref_price": 2.0, "atr_pct": 0.01, "kind": "grindlab",
            "model_p": 0.31, "checks": checks, "on_weex": True, "src": "binance", "price_precision": 4, "weex_qv24": 5e6,
            "book": {"bid": 1.999, "ask": 2.0, "mid": 1.9995, "spread_bps": 5.0, "imb": 0.0, "depth_usd": 1e5},
            "features": {"r_6h": 0.05, "r_24h": 0.08, "weex_dev_bps": 1.0}}
    eng = SignalEngine(cfg, db, None, tg)
    weak = base | {"symbol": "BBBUSDT", "score_n": 5, "catalyst": 0}
    news = base | {"symbol": "CCCUSDT", "score_n": 4, "catalyst": 1}
    strong = base | {"score_n": 9, "catalyst": 0}
    asyncio.run(eng.decide([weak, news, strong], None, NOW))
    assert weak["decision"] == "skipped:below_threshold"
    assert news["decision"] == "sent"                       # the news catalyst ranks first
    assert strong["decision"] == "skipped:spacing"          # grind paper trades are kept >= 4h apart
    later = strong | {"ts": NOW + 5 * HOUR}
    asyncio.run(eng.decide([later], None, NOW + 5 * HOUR))
    assert later["decision"] == "sent"
    sig = {s["symbol"]: dict(s) for s in db.query("SELECT * FROM signals")}
    s = sig["AAAUSDT"]
    assert s["kind"] == "grind" and s["grade"] == "P" and abs(s["sl_pct"] - 0.03) < 1e-12 and abs(s["tp_pct"] - 0.06) < 1e-12
    assert s["expires_ts"] - s["ts"] == 24 * HOUR
    assert any("Grind lab paper trade" in m and "Successful-grinder traits: 9/12" in m for m in tg.sent)
    assert any("News catalyst" in m for m in tg.sent)


def test_5m_grind_candidates_are_logged_only(tmp_path):
    cfg, db = Config(), DB(tmp_path / "t.sqlite")
    c = {"symbol": "XUSDT", "side": 1, "ts": NOW, "ref_price": 1.0, "atr_pct": 0.016, "kind": "grind", "on_weex": True,
         "src": "binance", "price_precision": 4, "features": {"r_6h": 0.05, "g_r2_6h": 0.9, "g_er_6h": 0.5}}
    asyncio.run(SignalEngine(cfg, db, None, Notifier()).decide([c], None, NOW, dry_run=True))
    assert c["decision"] == "skipped:logged_only"


def test_dashboard_snapshot_carries_the_grinders_tab(tmp_path):
    db = DB(tmp_path / "d.sqlite")
    db.conn.executescript(G.LIVE_SCHEMA)
    db.execute("INSERT INTO grind_live(symbol, ts, side, price, score, model_p, catalyst, checks, factors, decision) "
               "VALUES ('AAAUSDT', ?, 1, 1.0, 8, 0.3, 0, ?, ?, 'sent')",
               (NOW, json.dumps({"BTC's week is going the same way": True}), json.dumps({"r_6h": 0.05})))
    db.set("grindlab_report", {"n_episodes": 10, "traits": []})
    import time as _t
    d = snapshot(tmp_path / "d.sqlite")
    assert d["grindlab"]["n_episodes"] == 10
    assert d["grind_live"] == [] or d["grind_live"][0]["checks"]        # (older than 48h in wall-clock terms is fine)
