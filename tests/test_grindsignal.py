import asyncio

import numpy as np

from lts import grindsignal as GS
from lts.config import Config
from lts.db import DB
from lts.signals import SignalEngine

NOW = 1_794_900_000_000 // 3_600_000 * 3_600_000


class Notifier:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(text)
        return True


def band(cfg):
    return {"top": 0.05, "p_min": 0.18, "signals": 100, "per_day": 7.7,
            "plans": [{**pl, "win": 0.2, "net": -0.001, "margin": -0.001 * pl["leverage"], "t_day": -3.0,
                       "months_pos": 0.3, "n_months": 11} for pl in cfg.grindsignal.plans]}


def cand(cfg, symbol, p, side=1):
    return {"symbol": symbol, "side": side, "ts": NOW, "ref_price": 10.0, "atr_pct": 0.02, "kind": "grindsig",
            "p_model": p, "p_min": 0.18, "band": band(cfg), "on_weex": True, "src": "binance", "price_precision": 4,
            "weex_qv24": 5e6,
            "book": {"bid": 9.999, "ask": 10.0, "mid": 9.9995, "spread_bps": 1.0, "imb": 0.0, "depth_usd": 1e5},
            "features": {"r_6h": 0.05, "r_24h": 0.08, "sh_r2_6h": 0.9, "weex_dev_bps": 0.5}}


def test_every_top_signal_is_sent_with_all_three_plans(tmp_path):
    cfg, db, tg = Config(), DB(tmp_path / "g.sqlite"), Notifier()
    cs = [cand(cfg, f"C{i}USDT", 0.2) for i in range(6)] + [cand(cfg, "LOWUSDT", 0.1)]
    asyncio.run(SignalEngine(cfg, db, None, tg).decide(cs, None, NOW))
    assert [c["decision"] for c in cs[:6]] == ["sent"] * 6                 # no daily cap
    assert cs[6]["decision"] == "skipped:below_threshold"
    rows = [dict(r) for r in db.query("SELECT kind, grade, leverage, tp_pct, sl_pct, expires_ts, ts FROM signals")]
    primary = cfg.grindsignal.plans[cfg.grindsignal.primary]
    assert len(rows) == 6 and all(r["kind"] == "grindsig" and r["grade"] == "P" for r in rows)
    assert rows[0]["leverage"] == primary["leverage"] and abs(rows[0]["tp_pct"] - primary["target"]) < 1e-12
    assert abs(rows[0]["sl_pct"] - primary["stop"]) < 1e-12 and rows[0]["expires_ts"] - rows[0]["ts"] == 24 * 3_600_000
    msg = tg.sent[0]
    for pl in cfg.grindsignal.plans:                                       # each plan, with leverage, levels and history
        assert f"<b>{pl['leverage']}x</b>" in msg and f"+{pl['target']:.2%}" in msg and f"−{pl['stop']:.2%}" in msg
    assert msg.count("history (top 5%") == 3 and "← tracked" in msg and "PAPER" in msg
    # +2% at 5x after fees is close to +10% on margin
    assert "+9.3% /" in msg


def test_plan_outcomes_tight_stop_and_target():
    plans = [{"leverage": 5, "target": 0.02, "stop": 0.005}, {"leverage": 2, "target": 0.05, "stop": 0.0075}]
    post = np.full((2, 289, 4), 100.0, dtype=np.float32)
    post[0, 20, 1] = 102.1                          # +2.1% -> 5x plan wins, 2x plan times out
    post[1, 3, 2] = 99.4                            # -0.6% -> 5x plan stopped, 2x plan not
    out = GS.plan_outcomes(post, np.array([1, 1]), plans)
    a, b = out["net_0.02_0.005"], out["net_0.05_0.0075"]
    assert a[0] > 0.018 and a[1] < -0.005 and b[1] > -0.005


def test_bands_threshold_and_qualification():
    b = {"bands": [{"top": 1.0, "p_min": 0.0}, {"top": 0.05, "p_min": 0.18}, {"top": 0.02, "p_min": 0.21}]}
    assert GS.threshold(b, 0.05) == 0.18 and GS.band_for(b, 0.19)["top"] == 0.05 and GS.band_for(b, 0.3)["top"] == 0.02
    assert not GS.qualifies({"net": -0.001, "t_day": 5, "months_pos": 0.9, "n_months": 6})
    assert GS.qualifies({"net": 0.002, "t_day": 3.2, "months_pos": 0.75, "n_months": 6})


def test_seed_model_has_every_plan_and_scores_a_row():
    cfg = Config()
    bundle = GS.load_bundle(cfg)
    assert bundle and bundle["source"].startswith("seed") and len(bundle["plans"]) == 3
    assert all(len(b["plans"]) == 3 for b in bundle["bands"])
    rng = np.random.default_rng(0)
    c = 10 * np.exp(np.cumsum(0.001 + rng.normal(0, 0.002, 72)))
    o = np.concatenate([[c[0]], c[:-1]])
    pre = np.stack([o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c, np.full(72, 1e5), np.full(72, 6e4)], 1)
    rows = GS.score_rows([{"symbol": "X", "side": 1}], [pre], bundle)
    assert 0 <= rows[0]["p"] <= 1 and rows[0]["band"] is not None


def test_every_plan_is_recorded_after_24h(tmp_path):
    cfg, db = Config(), DB(tmp_path / "t.sqlite")
    start = NOW
    bars = [(start + i * 300_000, 10.0, 10.0 * (1.021 if i == 30 else 1.0005), 9.995, 10.0, 1, 10, 1, 0.5, 5)
            for i in range(300)]
    db.insert_klines("AUSDT", bars)
    db.execute("""INSERT INTO signals(symbol, side, ts, grade, entry, sl, tp, sl_pct, tp_pct, leverage, status, kind)
                  VALUES ('AUSDT', 1, ?, 'P', 10.0, 9.925, 10.5, 0.0075, 0.05, 2, 'open', 'grindsig')""", (start - 60_000,))
    import time as _t
    orig = _t.time
    _t.time = lambda: (start + 25 * 3_600_000) / 1000
    try:
        assert GS.label_signal_plans(cfg, db) == 1
    finally:
        _t.time = orig
    res = {r["plan"]: dict(r) for r in db.query("SELECT * FROM signal_plans")}
    assert len(res) == 3
    five = next(v for k, v in res.items() if k.startswith("5x"))
    assert five["outcome"] == "target" and five["margin"] > 0.09        # +2% hit -> ~+9% on margin at 5x
