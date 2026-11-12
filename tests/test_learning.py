import json

import numpy as np

from lts.config import Config
from lts.db import DB
from lts.features import MARKET_LEVEL, feature_names
from lts.learner import load_bundle, predict, train_and_maybe_promote, usable_features, load_labelled
from lts.signals import RiskState, SignalEngine

HOUR = 3_600_000
T0 = 1_700_000_000_000 // 86_400_000 * 86_400_000


def make_db(tmp_path, n=4000, signal=True, seed=0):
    """Synthetic labelled candidates over 30 days. `er_1h` carries real signal; `btc_r_24h`
    (a market-level feature) is a pure 'kind of day' effect."""
    rng = np.random.default_rng(seed)
    db = DB(tmp_path / "t.sqlite")
    names = feature_names()
    rows = []
    for i in range(n):
        ts = T0 + int(i * 30 * 24 * HOUR / n) // 300_000 * 300_000
        day_effect = np.sin(ts / (86_400_000 * 1.7))
        feat = {k: float(rng.normal()) for k in names}
        feat["btc_r_24h"] = float(day_effect)
        feat["side"] = float(rng.choice([-1, 1]))
        p = 0.34 + 0.15 * day_effect + (0.30 * np.tanh(feat["er_1h"]) if signal else 0.0)
        label = int(rng.random() < np.clip(p, 0.02, 0.98))
        ret_r = 1.9 if label else -1.15
        rows.append((f"S{i % 40}USDT", ts, int(feat["side"]), "trend", "replay", 1.0, 0.005, 0.01, json.dumps(feat), 50.0,
                     label, "tp" if label else "sl", ret_r, ts + 2 * HOUR, 0.02, 0.01, "{}", ts + 9 * HOUR))
    db.executemany("""INSERT INTO candidates(symbol, ts, side, kind, source, ref_price, atr_pct, sl_pct, features,
                      rule_score, label, outcome, ret_r, exit_ts, mfe, mae, grid, labeled_ts)
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    return db


def cfg_for(tmp_path):
    cfg = Config()
    cfg.storage.db_path = str(tmp_path / "t.sqlite")
    cfg.storage.model_dir = str(tmp_path / "models")
    cfg.learn.max_iter = 80
    return cfg


def test_market_level_features_are_held_back_on_short_history(tmp_path):
    db, cfg = make_db(tmp_path, n=600), cfg_for(tmp_path)
    feats = usable_features(load_labelled(db), cfg)
    assert "er_1h" in feats and not set(feats) & set(MARKET_LEVEL)
    cfg.learn.market_features_min_days = 10
    assert "btc_r_24h" in usable_features(load_labelled(db), cfg)


def add_forward_record(db, n, avg_r):
    """n closed signals (paper or real) alternating around avg_r."""
    db.executemany("""INSERT INTO signals(symbol, side, ts, grade, entry, sl, tp, sl_pct, tp_pct, leverage, status,
                      exit_ts, exit_price, ret_r, ret_margin) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   [("XUSDT", 1, T0 + i * HOUR, "P", 1.0, 0.99, 1.02, 0.01, 0.02, 5, "tp" if i % 2 else "sl",
                     T0 + i * HOUR + HOUR, 1.0, avg_r + (1.0 if i % 2 else -1.0), 0.0) for i in range(n)])


def test_strong_backtest_alone_is_not_enough_for_real_signals(tmp_path):
    db, cfg = make_db(tmp_path), cfg_for(tmp_path)
    res = train_and_maybe_promote(cfg, db)
    assert res["ok"] and res["metrics"]["xsec_corr"] > 0.05
    assert not res["promoted"] and "forward paper record too short" in res["summary"]
    assert load_bundle(cfg, db)["status"] == "shadow"


def test_real_signal_with_positive_forward_record_is_promoted_then_demoted_when_it_fails(tmp_path):
    db, cfg = make_db(tmp_path), cfg_for(tmp_path)
    add_forward_record(db, 24, avg_r=0.3)
    res = train_and_maybe_promote(cfg, db)
    assert res["promoted"], res["summary"]
    bundle = load_bundle(cfg, db)
    assert bundle["status"] == "champion"
    assert predict(bundle, {"er_1h": 2.5})[0] > predict(bundle, {"er_1h": -2.5})[0]
    # the live record then turns clearly negative -> champion is demoted, back to paper
    db.execute("DELETE FROM signals")
    add_forward_record(db, 40, avg_r=-0.5)
    res = train_and_maybe_promote(cfg, db)
    assert not res["promoted"] and "DEMOTED" in res["summary"]
    assert load_bundle(cfg, db, champion_only=True) is None
    assert load_bundle(cfg, db)["status"] == "shadow"


def test_noise_is_not_promoted_and_only_yields_paper_signals(tmp_path):
    db, cfg = make_db(tmp_path, signal=False), cfg_for(tmp_path)
    res = train_and_maybe_promote(cfg, db)
    assert res["ok"] and not res["promoted"]
    assert "paper" in res["summary"].lower()
    assert load_bundle(cfg, db, champion_only=True) is None
    shadow = load_bundle(cfg, db)
    assert shadow["status"] == "shadow"
    engine = SignalEngine(cfg, db, weex=None, notifier=None)
    state = RiskState(cfg, db, T0)
    grade = engine._grade({"p": 1.0, "score": 9.9, "rule_score": 0.0}, shadow, state, T0)
    assert grade == "P"                                    # never a real grade from an unvalidated model
