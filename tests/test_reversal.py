import json

import numpy as np

from lts import grindsignal as GS
from lts.config import Config
from lts.db import DB
from lts.grindlab import ensure_live_schema
from lts.microstructure import ensure_schema as micro_schema
from lts.reversal import anatomy

HOUR, BAR = 3_600_000, 300_000
NOW = 1_794_900_000_000 // HOUR * HOUR
PLAN = {"leverage": 5, "target": 0.02, "stop": 0.01}


def bars(path, start, buy_share):
    out = []
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        qv = 1e4 if i < 156 else 3e4                                   # volume triples at the turn (entry + 7h)
        out.append((start + i * BAR, a, max(a, b) * 1.0002, min(a, b) * 0.9998, b, qv / b, qv, 50,
                    qv * buy_share[i] / b, qv * buy_share[i]))
    return out


def make(tmp_path):
    db = DB(tmp_path / "r.sqlite")
    micro_schema(db)
    ensure_live_schema(db)
    pre = list(np.linspace(9.5, 10.0, 73))                             # the 6h grind before entry
    up = list(np.linspace(10.0, 10.12, 85))[1:]                        # +1.2% over 7h
    down = list(np.linspace(10.12, 9.85, 25))[1:]                      # then down through the -1% stop in 2h
    path = pre + up + down + [9.85] * 200
    share = [0.6] * 72 + [0.6] * 84 + [0.3] * 300                     # takers flip to selling at the peak
    db.insert_klines("AUSDT", bars(path, NOW - 6 * HOUR, share))
    for s, drift in (("BTCUSDT", -0.00005), ("BUSDT", -0.0001), ("CUSDT", 0.0)):
        p = list(100 * np.exp(np.cumsum(np.full(len(path), drift))))
        db.insert_klines(s, bars(p, NOW - 6 * HOUR, [0.5] * 500))
    peak_t = NOW + 83 * BAR
    db.insert_rows("liquidations", ("symbol", "ts", "side", "price", "qty", "usd"),
                   [("AUSDT", peak_t + 20 * 60_000, "long_liq", 10.0, 1000, 10_000.0)])
    db.insert_rows("oi", ("symbol", "ts", "oi", "oi_value"), [("AUSDT", NOW, 1000.0, None), ("AUSDT", peak_t, 1200.0, None),
                                                            ("AUSDT", peak_t + 2 * HOUR, 1100.0, None)])
    db.insert_rows("book_snapshots", ("symbol", "ts", "mid", "spread_bps", "bid05", "ask05", "bid1", "ask1", "bid2", "ask2",
                                      "big_bid_dist", "big_bid_usd", "big_bid_x", "big_ask_dist", "big_ask_usd", "big_ask_x"),
                   [("AUSDT", peak_t, 10.1, 2.0, 1e4, 3e4, 2e4, 5e4, 4e4, 8e4, 0.01, 1e4, 3.0, 0.003, 5e4, 8.0)])
    return db


def test_a_reversal_to_the_stop_is_recorded_with_what_changed(tmp_path):
    db = make(tmp_path)
    a = anatomy(db, "AUSDT", 1, NOW, PLAN, 24)
    assert a["exit"] == "stop" and a["reversal"] and 0.0115 < a["mfe"] < 0.0125
    assert a["minutes_to_peak"] == 420 and 0 < a["minutes_peak_to_exit"] <= 120
    assert a["run"]["taker_flow"] > 0.15 and a["fall"]["taker_flow"] < -0.3           # buyers, then sellers
    assert a["fall"]["rvol"] > 2.5 and a["run"]["rvol"] < 1.5                         # volume jumped on the way down
    assert a["fall"]["liq_against_usd"] == 10_000.0 and a["fall"]["btc"] < 0
    assert abs(a["oi_chg_run"] - 0.2) < 1e-9 and a["book"]["peak"]["imb_1"] < 0          # asks heavier at the top
    assert a["book"]["peak"]["wall_ahead_x"] == 8.0
    assert a["first_close_back_at_entry_min"] < a["minutes_to_exit"]                      # a breakeven exit came first


def test_anatomy_is_labelled_for_grinds_and_signals(tmp_path):
    cfg, db = Config(), make(tmp_path)
    db.execute("INSERT INTO grind_live(symbol, ts, side, price, score) VALUES ('AUSDT', ?, 1, 10.0, 5)", (NOW,))
    db.execute("""INSERT INTO signals(symbol, side, ts, grade, entry, sl, tp, sl_pct, tp_pct, leverage, status, kind)
                  VALUES ('AUSDT', 1, ?, 'P', 10.0, 9.9, 10.2, 0.01, 0.02, 5, 'sl', 'grindsig')""", (NOW - 30_000,))
    assert GS.label_anatomy(cfg, db, now=NOW + 26 * HOUR) == 2
    g = json.loads(db.query("SELECT anatomy FROM grind_live")[0]["anatomy"])
    s = db.query("SELECT exit, reversal FROM signal_anatomy")[0]
    assert g["reversal"] and s["exit"] == "stop" and s["reversal"] == 1
    assert GS.label_anatomy(cfg, db, now=NOW + 27 * HOUR) == 0                          # done once


def test_dashboard_compares_reversals_with_winners(tmp_path, monkeypatch):
    import lts.dashboard as D
    cfg, db = Config(), make(tmp_path)
    db.execute("INSERT INTO grind_live(symbol, ts, side, price, score) VALUES ('AUSDT', ?, 1, 10.0, 5)", (NOW,))
    GS.label_anatomy(cfg, db, now=NOW + 26 * HOUR)
    monkeypatch.setattr(D.time, "time", lambda: (NOW + 26 * HOUR) / 1000)
    an = D.snapshot(tmp_path / "r.sqlite")["anatomy"]
    assert an["n"]["reversal"] == 1 and an["reversals"][0]["symbol"] == "AUSDT"
    med = {m["key"]: m for m in an["medians"]}
    assert med["flow_fall"]["reversal"] < -0.3 and med["flow_fall"]["target"] is None
    assert an["reversals"][0]["lead_entry"] > 0
