import asyncio

import numpy as np

from lts.config import Config
from lts.db import DB
from lts.signals import SignalEngine, close_signal, exit_message
from lts.tracker import evaluate_bars

NOW = 1_794_500_000_000 // 300_000 * 300_000


class StubModel:
    def predict(self, x):
        return np.array([0.5])


class StubCalibrator:
    def predict(self, x):
        return np.array([0.45])


class Notifier:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(text)
        return True


def bundle(status):
    return {"version": "mtest", "status": status, "model": StubModel(), "calibrator": StubCalibrator(),
            "features": ["er_1h"], "threshold": 0.2, "threshold_safe": 0.3, "floor": 0.1,
            "win_r": 1.9, "nontp_avg_r": -0.9}


def candidate(symbol="XYZUSDT", side=1):
    return {"id": None, "symbol": symbol, "side": side, "ts": NOW, "ref_price": 0.4105, "atr_pct": 0.006,
            "on_weex": True, "src": "binance", "price_precision": 4, "weex_qv24": 5e6,
            "book": {"bid": 0.4104, "ask": 0.4105, "mid": 0.41045, "spread_bps": 2.4, "imb": 0.1, "depth_usd": 2e5},
            "features": {"er_1h": 0.6, "r_1h": 0.031, "r_24h": 0.18, "rvol_1h": 2.4, "oi_chg_1h": 0.02,
                         "btc_r_15m": 0.0, "fund_rate": 0.0001, "weex_dev_bps": 1.0}}


def run(coro):
    return asyncio.run(coro)


def test_unvalidated_model_sends_a_clearly_marked_paper_signal_and_tracks_it(tmp_path):
    cfg, db, tg = Config(), DB(tmp_path / "s.sqlite"), Notifier()
    engine = SignalEngine(cfg, db, weex=None, notifier=tg)
    c = candidate()
    sent = run(engine.decide([c], bundle("shadow"), NOW))
    assert len(sent) == 1 and c["decision"] == "sent" and c["grade"] == "P"
    msg = tg.sent[0]
    assert "PAPER" in msg and "do not trade it with real money" in msg
    assert "LONG XYZUSDT" in msg and "limit order preferred" in msg
    sig = dict(db.query("SELECT * FROM signals")[0])
    assert sig["status"] == "open" and sig["grade"] == "P" and sig["entry"] == 0.4105
    assert sig["sl"] < sig["entry"] < sig["tp"]
    assert abs(sig["sl_pct"] - 0.012) < 1e-9 and abs(sig["tp_pct"] - 0.024) < 1e-9      # 2 x ATR, 2:1
    assert 1 <= sig["leverage"] <= cfg.trade.max_leverage
    assert sig["leverage"] * (sig["tp_pct"] - 0.0014) >= cfg.trade.target_margin_return   # >= +10% on margin

    # price runs to the target: the tracker closes it and the exit message reports it
    bars = [(NOW, 0.4105, 0.4150, 0.4090, 0.4140), (NOW + 60_000, 0.4140, sig["tp"] + 0.001, 0.4130, 0.42)]
    status, px, ts, mfe, mae = evaluate_bars(sig, bars)
    assert status == "tp" and px == sig["tp"]
    closed = close_signal(cfg, db, sig, status, px, ts, mfe, mae)
    assert closed["ret_r"] > 1.8 and closed["ret_margin"] >= 0.10
    text = exit_message(cfg, db, closed)
    assert "Target hit" in text and "1W / 0L" in text


def test_validated_model_sends_grade_a_without_the_paper_warning(tmp_path):
    cfg, db, tg = Config(), DB(tmp_path / "s.sqlite"), Notifier()
    c = candidate(side=-1)
    c["features"]["r_1h"] = 0.03
    run(SignalEngine(cfg, db, None, tg).decide([c], bundle("champion"), NOW))
    assert c["grade"] == "A" and "PAPER" not in tg.sent[0] and "SHORT XYZUSDT" in tg.sent[0]
    sig = dict(db.query("SELECT * FROM signals")[0])
    assert sig["tp"] < sig["entry"] < sig["sl"] and sig["entry"] == 0.4104        # shorts enter at the bid


def test_risk_gates_block_signals(tmp_path):
    cfg, db, tg = Config(), DB(tmp_path / "s.sqlite"), Notifier()
    engine = SignalEngine(cfg, db, None, tg)
    b = bundle("champion")

    def decision(**change):
        c = candidate(symbol=change.pop("symbol", "AAAUSDT"))
        for k, v in change.items():
            if k in c:
                c[k] = v
            else:
                c["features"][k] = v
        run(engine.decide([c], b, NOW, dry_run=True))
        return c["decision"]

    assert decision() == "would_send"
    assert decision(on_weex=False) == "skipped:not_on_weex"
    assert decision(atr_pct=0.02) == "skipped:too_volatile"
    assert decision(btc_r_15m=-0.02) == "skipped:btc_shock"
    assert decision(fund_rate=0.002) == "skipped:funding_crowded"
    assert decision(weex_qv24=1e4) == "skipped:weex_illiquid"
    assert decision(weex_dev_bps=80.0) == "skipped:weex_price_deviation"
    wide = candidate()["book"] | {"spread_bps": 30.0}
    assert decision(book=wide) == "skipped:spread_too_wide"

    # portfolio limits: two open signals -> no third; paused -> nothing
    for sym in ("AAAUSDT", "BBBUSDT"):
        run(engine.decide([candidate(symbol=sym)], b, NOW))
    assert len(db.query("SELECT 1 FROM signals WHERE status='open'")) == 2
    assert decision(symbol="CCCUSDT") == "skipped:max_open"
    db.set("paused", True)
    assert decision(symbol="DDDUSDT") == "skipped:paused"
