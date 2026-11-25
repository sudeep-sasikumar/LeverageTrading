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
    cfg.grindsignal.require_positive_ev = False             # this band loses after costs; see the EV-gate test
    cfg.grindsignal.max_same_direction = 0
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
    assert msg.count("history (top 5%") == len(cfg.grindsignal.plans) == 4 and "← tracked" in msg and "PAPER" in msg
    assert 'href="https://www.weex.com/futures/C0-USDT"' in msg                 # one-click chart
    # +2% at 5x after fees is close to +10% on margin
    assert "+9.0% /" in msg                                  # 5x +2%: taker in and out, 0.04% slippage


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
    assert bundle and bundle["source"].startswith("seed") and len(bundle["plans"]) == 4
    assert all(len(b["plans"]) == 4 for b in bundle["bands"])
    assert bundle["plans"][0] == {"leverage": 10, "target": 0.01, "stop": 0.005}
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
    assert len(res) == 4
    ten = next(v for k, v in res.items() if k.startswith("10x"))
    assert ten["outcome"] == "target" and ten["margin"] > 0.079         # +1% at 10x, taker fee both ways
    five = next(v for k, v in res.items() if k.startswith("5x"))
    assert five["outcome"] == "target" and five["margin"] > 0.09        # +2% hit -> ~+9% on margin at 5x


def test_every_grind_is_recorded_and_labelled_after_24h(tmp_path):
    import json, time as _t
    from lts.grindlab import LIVE_SCHEMA, ensure_live_schema
    cfg, db = Config(), DB(tmp_path / "r.sqlite")
    ensure_live_schema(db)
    ts = NOW
    db.execute("INSERT INTO grind_live(symbol, ts, side, price, score) VALUES ('AUSDT', ?, -1, 10.0, 7)", (ts,))
    rows = [{"symbol": "AUSDT", "ts": ts, "side": -1, "p": 0.12, "band": {"x": 1}, "checks": {"a": True},
             "r_6h": 0.05, "sh_r2_6h": 0.8, "price": 10.0}]
    GS.record_grinds(db, rows, 0.146)
    rec = db.query("SELECT all_factors, micro, threshold, model_p FROM grind_live")[0]
    f = json.loads(rec["all_factors"])
    assert f["sh_r2_6h"] == 0.8 and "band" not in f and rec["threshold"] == 0.146 and rec["model_p"] == 0.12
    bars = [(ts + i * 300_000, 10.0, 10.0005, 10.0 * (0.985 if i == 40 else 0.9995), 10.0, 1, 10, 1, 0.5, 5)
            for i in range(300)]
    db.insert_klines("AUSDT", bars)
    orig = _t.time
    _t.time = lambda: (ts + 25 * 3_600_000) / 1000
    try:
        assert GS.label_grinds(cfg, db) == 1
    finally:
        _t.time = orig
    o = json.loads(db.query("SELECT outcomes FROM grind_live")[0]["outcomes"])
    assert set(o["plans"]) == {GS.plan_name(p) for p in cfg.grindsignal.plans}
    assert o["plans"]["10x +1.00% / -0.50%"] > 0.0079                    # short: the -1.5% dip hits +1%
    assert len(o["grid"]) == len(cfg.grindsignal.record_targets) * len(cfg.grindsignal.record_stops)
    assert o["best"]["24"] > 0.0149


def test_weex_link():
    assert GS.weex_url("HUMAUSDT") == "https://www.weex.com/futures/HUMA-USDT"


def test_five_minute_scans_record_about_once_an_hour_and_when_the_score_crosses(tmp_path, monkeypatch):
    import time as _t
    from types import SimpleNamespace
    from lts.grindlab import ensure_live_schema
    from lts.service import Service
    cfg, db = Config(), DB(tmp_path / "f.sqlite")
    cfg.grindsignal.selector = "model"                      # this test is about the model-score path
    ensure_live_schema(db)
    me = SimpleNamespace(cfg=cfg, db=db, gs_bundle={"bands": [{"top": 0.2, "p_min": 0.15}]}, gs_loaded=_t.time(),
                         grind_levels=lambda rows, db=None: None)
    scores = iter([0.10, 0.11, 0.16, 0.17])
    monkeypatch.setattr(GS, "pre_from_db", lambda *a: None)
    monkeypatch.setattr(GS, "score_rows", lambda rows, pres, b: [r.update(p=next(scores), band=None) for r in rows])
    monkeypatch.setattr(GS, "threshold", lambda b, share: 0.15)

    def scan(end):
        rows = [{"symbol": "AUSDT", "ts": end, "side": 1, "price": 1.0, "score": 3, "model_p": None, "checks": {}}]
        kept = Service.grinds_to_record(me, rows, end)
        GS.record_grinds(db, kept, 0.15)
        return kept
    assert len(scan(NOW)) == 1                          # first detection: recorded (below threshold)
    assert scan(NOW + 300_000) == []                     # 5 minutes later, still below: not recorded again
    assert len(scan(NOW + 600_000)) == 1                 # score crossed the threshold: recorded (and sent)
    assert scan(NOW + 900_000) == []                     # already above at its last record
    assert db.query("SELECT COUNT(*) n FROM grind_live")[0]["n"] == 2


def test_grind_signals_daily_and_streak_stops_are_back_on_and_switchable(tmp_path):
    """Review item 6: grind signals obey the daily-loss and losing-streak stops again (DESIGN.md section 19 had
    turned them off); each can be switched off in config."""
    cfg, db, tg = Config(), DB(tmp_path / "l.sqlite"), Notifier()
    cfg.grindsignal.require_positive_ev = False
    for i in range(3):                                   # three losing grind signals today: -3.9R
        db.execute("""INSERT INTO signals(symbol, side, ts, grade, entry, sl, tp, sl_pct, tp_pct, leverage, status, kind,
                      exit_ts, ret_r) VALUES (?, 1, ?, 'P', 10, 9.9, 10.2, 0.01, 0.02, 5, 'sl', 'grindsig', ?, -1.3)""",
                   (f"L{i}USDT", NOW - 3_600_000, NOW - 600_000))
    cs = [cand(cfg, "NEWUSDT", 0.2), cand(cfg, "L0USDT", 0.2)]
    asyncio.run(SignalEngine(cfg, db, None, tg).decide(cs, None, NOW))
    assert cs[0]["decision"] == "skipped:daily_stop" and cs[1]["decision"] == "skipped:symbol_cooldown"
    cfg.grindsignal.daily_stop = False
    c = cand(cfg, "NEW2USDT", 0.2)
    asyncio.run(SignalEngine(cfg, db, None, tg).decide([c], None, NOW))
    assert c["decision"] == "skipped:loss_streak"
    cfg.grindsignal.streak_stop = False
    c = cand(cfg, "NEW3USDT", 0.2)
    asyncio.run(SignalEngine(cfg, db, None, tg).decide([c], None, NOW))
    assert c["decision"] == "sent"


def test_grind_signals_same_direction_limit_while_the_market_moves_that_way(tmp_path):
    cfg, db, tg = Config(), DB(tmp_path / "c.sqlite"), Notifier()
    cfg.grindsignal.require_positive_ev = False
    assert cfg.grindsignal.max_same_direction == 2
    shorts = [cand(cfg, f"S{i}USDT", 0.2, side=-1) for i in range(3)]
    for c in shorts:
        c["features"]["mkt_r6"] = 0.01                   # the median coin is falling too (aligned with the short)
    asyncio.run(SignalEngine(cfg, db, None, tg).decide(shorts, None, NOW))
    assert [c["decision"] for c in shorts] == ["sent", "sent", "skipped:same_direction_limit"]
    lone = cand(cfg, "S9USDT", 0.2, side=-1)
    lone["features"]["mkt_r6"] = -0.01                   # market moving the other way: an own move, no cap
    long_ = cand(cfg, "L9USDT", 0.2, side=1)
    long_["features"]["mkt_r6"] = 0.01
    asyncio.run(SignalEngine(cfg, db, None, tg).decide([lone, long_], None, NOW))
    assert lone["decision"] == "sent" and long_["decision"] == "sent"


def test_btc_shock_and_funding_vetoes_apply_to_grind_signals(tmp_path):
    from lts.service import market_now
    cfg, db, tg = Config(), DB(tmp_path / "b.sqlite"), Notifier()
    cfg.grindsignal.require_positive_ev = False
    for i, c in enumerate((100.0, 100.0, 100.0, 98.0)):          # BTC -2% over the last 15 minutes
        db.execute("INSERT INTO klines(symbol, ts, o, h, l, c) VALUES ('BTCUSDT', ?, ?, ?, ?, ?)",
                   (NOW - (4 - i) * 300_000, c, c, c, c))
    db.execute("INSERT INTO premium(symbol, ts, funding_rate) VALUES ('FUNDUSDT', ?, 0.002)", (NOW - 60_000,))
    m_long, m_short = market_now(db, "FUNDUSDT", 1, NOW), market_now(db, "FUNDUSDT", -1, NOW)
    assert abs(m_long["btc_r_15m"] + 0.02) < 1e-9 and abs(m_short["btc_r_15m"] - 0.02) < 1e-9
    assert m_long["fund_rate"] == 0.002 and m_short["fund_rate"] == -0.002
    a = cand(cfg, "AUSDT", 0.2)
    a["features"].update(market_now(db, "AUSDT", 1, NOW))
    b = cand(cfg, "FUNDUSDT", 0.2, side=1)
    b["features"].update(m_long)
    b["features"]["btc_r_15m"] = 0.0
    asyncio.run(SignalEngine(cfg, db, None, tg).decide([a, b], None, NOW))
    assert a["decision"] == "skipped:btc_shock" and b["decision"] == "skipped:funding_crowded"



def test_levels_between_entry_target_and_stop():
    import pandas as pd
    from lts.levels import between, major_levels
    day = 86_400_000
    t0 = NOW // day * day - 60 * day
    hs = [10.0] * 60
    ls = [9.0] * 60
    hs[40] = 10.4                                    # a daily swing high 40 days in (known 3 days later)
    ls[50] = 8.95                                    # a daily swing low
    d = pd.DataFrame({"ts": [t0 + i * day for i in range(60)], "h": hs, "l": ls})
    lv = major_levels(d, t0 + 60 * day)
    assert any(x["what"] == "daily swing high" and x["price"] == 10.4 for x in lv)
    assert not any(x["price"] == 10.4 for x in major_levels(d, t0 + 42 * day))   # not yet confirmed then
    b = between(lv, 10.25, 1, 0.02, 0.01)            # long at 10.25: target 10.455, stop 10.1475
    assert {x["price"] for x in b["block"]} == {10.4} and b["block"][0]["dist"] > 0     # also the weekly/monthly high
    assert b["block_tf"] == ["monthly", "weekly", "daily"]
    s = between(lv, 9.05, -1, 0.02, 0.01)            # short at 9.05: target 8.869 -> the 8.95 low and 9.0 lows are in the way
    assert {x["price"] for x in s["block"]} >= {8.95, 9.0} and not s["cushion"]


def test_after_exit_follows_price_for_24h_after_the_stop():
    from lts.grindsignal import after_exit
    plans = [{"leverage": 5, "target": 0.02, "stop": 0.01}]
    post = np.full((577, 4), 100.0)
    post[:, 1], post[:, 2] = 100.05, 99.95
    post[10, 2] = 98.9                               # stopped at bar 10 (-1.1%)
    post[100, 1] = 102.5                             # then ran to +2.5% (the target came after the stop)
    a = after_exit(post, 1, plans, 24, 24)["5x +2.00% / -1.00%"]
    assert a["exit"] == "stop" and a["exit_min"] == 55 and a["target_after_stop"]
    assert abs(a["best_24h"] - (102.5 / 99.0 - 1)) < 1e-9 and a["bars_after"] == 288


def test_signal_shows_levels_and_can_skip_when_a_level_blocks(tmp_path):
    cfg, db, tg = Config(), DB(tmp_path / "v.sqlite"), Notifier()
    cfg.grindsignal.require_positive_ev = False
    assert cfg.grindsignal.skip_if_level_blocks                         # trader's rule is on by default
    cfg.grindsignal.skip_if_level_blocks = False
    lv = {"block": [{"price": 10.15, "tf": "weekly", "what": "weekly swing high", "dist": 0.015}],
          "cushion": [{"price": 9.95, "tf": "daily", "what": "previous day low", "dist": -0.005}],
          "block_tf": ["weekly"], "cushion_tf": ["daily"]}
    c = cand(cfg, "LVLUSDT", 0.2) | {"levels": lv}
    asyncio.run(SignalEngine(cfg, db, None, tg).decide([c], None, NOW))
    assert c["decision"] == "sent" and "weekly swing high" in tg.sent[0] and "previous day low" in tg.sent[0]
    cfg.grindsignal.skip_if_level_blocks = True
    c2 = cand(cfg, "LVL2USDT", 0.2) | {"levels": lv}
    asyncio.run(SignalEngine(cfg, db, None, tg).decide([c2], None, NOW))
    assert c2["decision"] == "skipped:level_in_way"
    clear = {"block": [], "cushion": lv["cushion"], "block_tf": [], "cushion_tf": ["daily"]}
    c3 = cand(cfg, "CLRUSDT", 0.2) | {"levels": clear}
    asyncio.run(SignalEngine(cfg, db, None, tg).decide([c3], None, NOW))
    assert c3["decision"] == "sent" and "No daily/weekly/monthly level before" in tg.sent[-1]


def steady_row(**kw):
    r = {"r_24h": 0.10, "r2_24h": 0.85, "hours_with24": 16, "r2_6h": 0.93, "hours_with6": 5, "big6": 0.30}
    r.update(kw)
    return r


def test_steady_rule_and_why_not():
    gs = Config().grindsignal
    assert gs.selector == "steady" and GS.steady(steady_row(), gs)
    assert not GS.steady(steady_row(r_24h=0.05), gs) and not GS.steady(steady_row(big6=0.5), gs)
    assert not GS.steady(steady_row(r2_24h=None), gs)                     # a missing factor never passes
    assert gs.steady_eval == "hour_close_first"
    msg = GS.why_not(steady_row(r_24h=0.05, hours_with24=10, first=True), gs, None)
    assert msg.startswith("not steady all day") and "24h +5.0%" in msg and "10/24 hours" in msg
    assert GS.why_not(steady_row(first=False), gs, None).startswith("not checked")
    assert GS.selected(steady_row(first=True), gs, None) and not GS.selected(steady_row(first=False), gs, None)
    gs.steady_eval = "rolling_5m"                                          # the old behaviour: any 5-minute record
    assert GS.selected(steady_row(), gs, None)


def history_file(tmp_path, net):
    import json
    cfg = Config()
    plans = {GS.plan_name(pl): {"n": 539, "n_days": 300, "win": 0.38, "net": net, "se_day": 0.001, "t_day": net / 0.001,
                                "months_pos": 0.36, "n_months": 14} for pl in cfg.grindsignal.plans}
    p = tmp_path / "steady_history.json"
    p.write_text(json.dumps({"rules": {"steady/hour_close_first": {"plans": plans, "passed_bar": False,
                                                                  "source": "test record"}}}), encoding="utf-8")
    return p


def test_steady_signal_message_uses_the_rules_year_record(tmp_path, monkeypatch):
    cfg, db, tg = Config(), DB(tmp_path / "s.sqlite"), Notifier()
    monkeypatch.setattr(GS, "HISTORY_PATH", history_file(tmp_path, -0.0008))
    cfg.grindsignal.require_positive_ev = False
    c = cand(cfg, "STDYUSDT", None) | {"selector": "steady", "p_min": -1.0, "steady": steady_row(),
                                       "levels": {"block": [], "cushion": []}}
    asyncio.run(SignalEngine(cfg, db, None, tg).decide([c], None, NOW))   # no model score: still sent
    msg = tg.sent[0]
    assert c["decision"] == "sent" and "all-day steady grind" in msg and "16/24 hours" in msg
    assert "history (steady all-day grinds, past year): won 38%" in msg
    assert "Expected net per trade (5x +2.00% / -1.00%): <b>-0.08%</b> (90% interval -0.24% to +0.08%)" in msg
    sig = dict(db.query("SELECT grade, model_version FROM signals")[0])
    assert sig == {"grade": "P", "model_version": "grind-steady-hour_close_first"}


def test_expected_net_gate_and_go_live_gate(tmp_path, monkeypatch):
    """Part C: nothing is sent unless the rule's tested net after costs is positive; 'live' only takes effect
    after the rule passed its bar AND has enough closed live paper trades under that rule."""
    from lts.gate import live_allowed
    cfg, db, tg = Config(), DB(tmp_path / "e.sqlite"), Notifier()
    assert cfg.grindsignal.require_positive_ev and not cfg.grindsignal.live
    st = {"selector": "steady", "p_min": -1.0, "steady": steady_row(), "levels": {"block": [], "cushion": []}}
    monkeypatch.setattr(GS, "HISTORY_PATH", history_file(tmp_path, -0.0008))
    c = cand(cfg, "NEGUSDT", None) | st
    asyncio.run(SignalEngine(cfg, db, None, tg).decide([c], None, NOW))
    assert c["decision"] == "skipped:expected_net_not_positive" and not tg.sent
    monkeypatch.setattr(GS, "HISTORY_PATH", tmp_path / "missing.json")      # untested rule: nothing is sent
    c = cand(cfg, "NONEUSDT", None) | st
    asyncio.run(SignalEngine(cfg, db, None, tg).decide([c], None, NOW))
    assert c["decision"] == "skipped:expected_net_not_positive"
    monkeypatch.setattr(GS, "HISTORY_PATH", history_file(tmp_path, 0.002))
    cfg.grindsignal.live = True
    c = cand(cfg, "POSUSDT", None) | st
    asyncio.run(SignalEngine(cfg, db, None, tg).decide([c], None, NOW))
    assert c["decision"] == "sent" and c["grade"] == "P"                    # live on, but the gate is not met
    ok, why = live_allowed(cfg, db)
    assert not ok and any("pre-registered bar" in w for w in why) and any("1/30 closed" in w or "0/30 closed" in w for w in why)



def test_dashboard_applies_the_rule_to_grinds_recorded_before_it(tmp_path, monkeypatch):
    import json
    import lts.dashboard as D
    from lts.grindlab import ensure_live_schema
    db = DB(tmp_path / "d.sqlite")
    ensure_live_schema(db)
    out = {"plans": {"5x +2.00% / -1.00%": 0.018}}
    for sym, f in (("GOODUSDT", steady_row()), ("MEHUSDT", steady_row(r2_24h=0.5))):
        db.execute("INSERT INTO grind_live(symbol, ts, side, price, score, all_factors, outcomes) VALUES (?, ?, 1, 1.0, 5, ?, ?)",
                   (sym, NOW, json.dumps(f), json.dumps(out)))
    monkeypatch.setattr(D.time, "time", lambda: (NOW + 3_600_000) / 1000)
    snap = D.snapshot(tmp_path / "d.sqlite")
    sel = {r["symbol"]: r["selected"] for r in snap["grind_all"]}
    assert sel == {"GOODUSDT": 1, "MEHUSDT": 0}
    assert snap["rule_live"]["selected"] == {"n": 1, "won": 1, "net": 0.018}
