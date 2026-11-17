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


def cand(symbol, p, band):
    return {"symbol": symbol, "side": 1, "ts": NOW, "ref_price": 10.0, "atr_pct": 0.02, "kind": "grindsig",
            "p_model": p, "band": band, "on_weex": True, "src": "binance", "price_precision": 4, "weex_qv24": 5e6,
            "book": {"bid": 9.999, "ask": 10.0, "mid": 9.9995, "spread_bps": 1.0, "imb": 0.0, "depth_usd": 1e5},
            "features": {"r_6h": 0.05, "r_24h": 0.08, "sh_r2_6h": 0.9, "weex_dev_bps": 0.5}}


def test_all_signals_above_threshold_are_sent_no_daily_cap(tmp_path):
    cfg, db, tg = Config(), DB(tmp_path / "g.sqlite"), Notifier()
    band = {"p_min": 0.6, "win": 0.519, "net": -0.0015, "per_day": 4.4, "t_day": -4.0, "months_pos": 0.18, "n_months": 11}
    cs = [cand(f"C{i}USDT", 0.62, band) for i in range(6)] + [cand("LOWUSDT", 0.55, band)]
    asyncio.run(SignalEngine(cfg, db, None, tg).decide(cs, None, NOW))
    assert [c["decision"] for c in cs[:6]] == ["sent"] * 6               # no daily cap
    assert cs[6]["decision"] == "skipped:below_threshold"
    rows = db.query("SELECT kind, grade, leverage, tp_pct, sl_pct FROM signals")
    assert len(rows) == 6 and all(r["kind"] == "grindsig" and r["grade"] == "P" and r["leverage"] == 10 for r in rows)
    msg = tg.sent[0]
    assert "GRIND LONG" in msg and "model 62%" in msg and "won 52%" in msg and "-0.15% per trade" in msg
    # the same coin is not signalled twice while its trade is open
    again = cand("C0USDT", 0.7, band)
    asyncio.run(SignalEngine(cfg, db, None, tg).decide([again], None, NOW + 3_600_000))
    assert again["decision"] == "skipped:already_open"


def test_band_lookup_and_qualification():
    b = {"bands": [{"p_min": 0.0, "net": -0.002}, {"p_min": 0.6, "net": -0.0015}, {"p_min": 0.7, "net": 0.001}]}
    assert GS.band_for(b, 0.65)["p_min"] == 0.6 and GS.band_for(b, 0.71)["p_min"] == 0.7
    assert not GS.qualifies({"net": -0.001, "t_day": 5, "months_pos": 0.9, "n_months": 6})
    assert GS.qualifies({"net": 0.002, "t_day": 3.2, "months_pos": 0.75, "n_months": 6})


def test_seed_model_scores_a_live_row():
    bundle = GS.load_bundle(Config())
    assert bundle and bundle["source"].startswith("seed") and len(bundle["bands"]) >= 3
    rng = np.random.default_rng(0)
    c = 10 * np.exp(np.cumsum(0.001 + rng.normal(0, 0.002, 72)))
    o = np.concatenate([[c[0]], c[:-1]])
    pre = np.stack([o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c, np.full(72, 1e5), np.full(72, 6e4)], 1)
    rows = GS.score_rows([{"symbol": "X", "side": 1}], [pre], bundle)
    assert 0 <= rows[0]["p"] <= 1 and rows[0]["band"] is not None


def test_pre_from_db_reads_the_six_hours_before(tmp_path):
    db = DB(tmp_path / "k.sqlite")
    rows = [(NOW - (72 - i) * 300_000, 1, 2, 0.5, 1.5, 10, 15, 7, 4, 6) for i in range(72)]
    db.insert_klines("AUSDT", rows)
    pre = GS.pre_from_db(db, "AUSDT", NOW)
    assert pre.shape == (72, 6) and np.isfinite(pre).all() and pre[0, 4] == 15 and pre[0, 5] == 6
    assert GS.pre_from_db(db, "AUSDT", NOW + 10 * 3_600_000) is None
