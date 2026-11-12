import numpy as np
import pandas as pd
import pytest

from lts import BAR_MS
from lts import features as F
from lts.candidates import apply_cooldown, trigger_mask
from lts.config import Config
from lts.exchanges import _book_stats, _kline_rows
from lts.labels import barrier, costs, label_path, plan_trade, stop_pct
from lts.learner import pick_threshold, simulate_policy
from lts.tracker import evaluate_bars


def synth_klines(n=3000, seed=1, start=1_700_000_100_000 // BAR_MS * BAR_MS):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.002, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.002, n))
    v = rng.uniform(50, 500, n)
    qv = v * c
    return pd.DataFrame({"ts": start + np.arange(n) * BAR_MS, "o": o, "h": h, "l": l, "c": c, "v": v, "qv": qv,
                         "trades": rng.integers(100, 1000, n), "tbv": v * rng.uniform(0.3, 0.7, n),
                         "tbqv": qv * rng.uniform(0.3, 0.7, n)})


def slow_series(k, seed=2):
    rng = np.random.default_rng(seed)
    ts = k["ts"].to_numpy()[::3] + BAR_MS
    oi = pd.DataFrame({"ts": ts, "oi": 1e6 * np.exp(np.cumsum(rng.normal(0, 0.002, len(ts))))})
    fts = k["ts"].to_numpy()[::96]
    funding = pd.DataFrame({"ts": fts, "rate": rng.normal(1e-4, 1e-4, len(fts))})
    lts_ = k["ts"].to_numpy()[::12] + 3_600_000
    lsr = pd.DataFrame({"ts": lts_, "top_pos": rng.uniform(0.8, 2, len(lts_)), "global_acc": rng.uniform(0.5, 3, len(lts_))})
    return oi, funding, lsr


def full_frame(k, btc, oi, funding, lsr):
    kr, br = F.regularize(k), F.regularize(btc)
    market, ranks = F.market_context({"AAA": F.panel_inputs(kr), "BTCUSDT": F.panel_inputs(br)}, br)
    return F.add_context(F.symbol_frame(kr, oi, funding, lsr, None, onboard_ts=k["ts"].iloc[0] - 30 * 86_400_000),
                         "AAA", market, ranks)


def test_features_have_no_lookahead():
    """Features at bar t must not change when later bars are appended."""
    k, btc = synth_klines(seed=1), synth_klines(seed=5)
    oi, funding, lsr = slow_series(k)
    cut = 2600
    cut_ts = k["ts"].iloc[cut - 1] + BAR_MS
    full = full_frame(k, btc, oi, funding, lsr)
    part = full_frame(k.iloc[:cut], btc.iloc[:cut], oi[oi.ts <= cut_ts], funding[funding.ts <= cut_ts],
                      lsr[lsr.ts <= cut_ts])
    a, b = full.iloc[cut - 1], part.iloc[-1]
    assert full.index[cut - 1] == part.index[-1]
    for col in full.columns:
        x, y = a[col], b[col]
        assert (np.isnan(x) and np.isnan(y)) or x == pytest.approx(y, rel=1e-9, abs=1e-12), col
    # and the future OI/funding rows really exist, so the comparison is meaningful
    assert (oi.ts > cut_ts).any() and (funding.ts > cut_ts).any()


def test_feature_frame_is_mostly_populated_and_aligned():
    k, btc = synth_klines(seed=3), synth_klines(seed=4)
    f = full_frame(k, btc, *slow_series(k))
    last = f.iloc[[-1]]
    longs, shorts = F.align(last, 1).iloc[0], F.align(last, -1).iloc[0]
    assert list(longs.index) == F.feature_names()
    missing = [n for n in F.feature_names() if np.isnan(longs[n]) and n not in F.LIVE_ONLY]
    assert not missing, missing
    assert longs["r_1h"] == pytest.approx(-shorts["r_1h"])
    assert longs["pos_24h"] == pytest.approx(1 - shorts["pos_24h"])
    assert longs["dist_ext"] == pytest.approx(shorts["dist_opp"])
    assert longs["rsi14"] == pytest.approx(100 - shorts["rsi14"])
    assert longs["atr_pct"] == shorts["atr_pct"]


def test_regularize_fills_gaps():
    k = synth_klines(50).drop(index=[10, 11]).reset_index(drop=True)
    r = F.regularize(k)
    assert len(r) == 50
    gap = r.iloc[10]
    assert gap["v"] == 0 and gap["o"] == gap["c"] == r.iloc[9]["c"]


def test_trigger_requires_momentum_volume_and_liquidity():
    cfg = Config()
    base = dict(z_1h=3.0, z_4h=0.0, r_24h=0.02, r_1h=0.03, rvol_1h=2.0, rvol_24h=1.0, pos_1h=0.9, d_ema50=2.0,
                atr_pct=0.005, qv_24h=5e7, bars_seen=2000.0, listing_age_days=100.0)
    def fire(side=1, **kw):
        return bool(trigger_mask(pd.DataFrame([{**base, **kw}]), cfg, side)[0])
    assert fire()
    assert not fire(z_1h=1.0)                    # no momentum
    assert not fire(rvol_1h=1.0)                 # no participation
    assert not fire(pos_1h=0.2)                  # already rolled over
    assert not fire(qv_24h=1e6)                  # illiquid
    assert not fire(atr_pct=0.02)                # too volatile for a tight stop
    assert not fire(listing_age_days=1.0)        # too new
    assert not fire(side=-1)                     # wrong direction
    assert fire(side=-1, z_1h=-3.0, r_1h=-0.03, pos_1h=0.1, d_ema50=-2.0)


def test_cooldown_keeps_first_of_cluster():
    mask = np.zeros(30, dtype=bool)
    mask[[2, 3, 8, 9, 20]] = True
    assert list(np.flatnonzero(apply_cooldown(mask, 6))) == [2, 9, 20]


def test_barrier_outcomes():
    tc = Config().trade
    cst = costs(tc)
    flat = np.full(10, 100.0)
    h, l = flat.copy(), flat.copy()
    h[3] = 102.1
    out = barrier(1, 100.0, 0.01, 0.02, h, l, flat, 10, cst)
    assert out["outcome"] == "tp" and out["bars"] == 4 and out["ret_r"] == pytest.approx((0.02 - cst["tp"]) / 0.01)
    l[2] = 98.9                                   # stop touched earlier
    assert barrier(1, 100.0, 0.01, 0.02, h, l, flat, 10, cst)["outcome"] == "sl"
    both_h, both_l = flat.copy(), flat.copy()
    both_h[1], both_l[1] = 103, 98               # same bar touches both -> stop
    assert barrier(1, 100.0, 0.01, 0.02, both_h, both_l, flat, 10, cst)["outcome"] == "sl"
    short = barrier(-1, 100.0, 0.01, 0.02, flat, np.where(np.arange(10) == 5, 97.9, 100.0), flat, 10, cst)
    assert short["outcome"] == "tp"
    drift = np.linspace(100, 100.5, 10)
    t = barrier(1, 100.0, 0.01, 0.02, drift, drift, drift, 10, cst)
    assert t["outcome"] == "time" and t["ret"] == pytest.approx(0.005)
    assert barrier(1, 100.0, 0.01, 0.02, drift[:5], drift[:5], drift[:5], 10, cst) is None   # path too short


def test_plan_trade_targets_ten_percent_within_leverage_cap():
    tc = Config().trade
    for atr in (0.002, 0.004, 0.006, 0.01, 0.0125):
        plan = plan_trade(tc, atr)
        assert plan is not None
        assert 1 <= plan["leverage"] <= tc.max_leverage
        assert plan["gain_margin"] >= tc.target_margin_return - 1e-9
        assert tc.min_sl_pct <= plan["sl_pct"] <= tc.max_sl_pct
        assert plan["sl_pct"] < 0.6 / plan["leverage"]            # stop well inside liquidation
        assert plan["margin_frac"] * plan["loss_margin"] <= 0.01 + 1e-9
    assert plan_trade(tc, 0.02) is None                            # too volatile -> no trade
    assert stop_pct(tc, 0.001) == tc.min_sl_pct


def test_label_path_needs_full_horizon():
    tc = Config().trade
    n = max([tc.horizon_bars] + tc.grid_horizon)
    flat = np.full(n, 100.0)
    assert label_path(tc, 1, 0.005, flat[:10], flat[:10], flat[:10], flat[:10]) is None
    lab = label_path(tc, 1, 0.005, flat, flat * 1.05, flat, flat)
    assert lab["label"] == 1 and lab["outcome"] == "tp" and lab["grid"]


def test_policy_respects_portfolio_limits():
    cfg = Config()
    hour = 3_600_000
    rows = []
    for i in range(12):   # 12 losing candidates, 20 minutes apart, all above threshold
        ts = 1_700_000_000_000 // 86_400_000 * 86_400_000 + i * hour // 3
        rows.append({"ts": ts, "symbol": f"S{i}", "p": 0.9, "ret_r": -1.2, "exit_ts": ts + hour // 6, "outcome": "sl"})
    res = simulate_policy(pd.DataFrame(rows), "p", 0.5, cfg)
    assert res["n"] == cfg.risk.max_consecutive_losses        # stops for the day after the loss streak
    assert simulate_policy(pd.DataFrame(rows), "p", 0.95, cfg)["n"] == 0


def test_pick_threshold_prefers_expectancy_with_frequency():
    cfg = Config()
    table = [{"thr": 0.3, "n": 200, "per_day": 3.0, "avg_r": 0.05}, {"thr": 0.4, "n": 60, "per_day": 1.2, "avg_r": 0.30},
             {"thr": 0.5, "n": 30, "per_day": 0.3, "avg_r": 0.60}, {"thr": 0.6, "n": 5, "per_day": 0.1, "avg_r": 2.0}]
    main, floor = pick_threshold(table, cfg)
    assert main["thr"] == 0.4 and floor["thr"] == 0.4


def test_tracker_evaluates_minute_bars():
    sig = {"side": 1, "entry": 100.0, "sl": 99.0, "tp": 102.0}
    bars = [(0, 100, 100.5, 99.5, 100.2), (60_000, 100.2, 102.3, 100.1, 102.0)]
    status, px, ts, mfe, mae = evaluate_bars(sig, bars)
    assert (status, px, ts) == ("tp", 102.0, 120_000) and mfe == pytest.approx(0.023) and mae == pytest.approx(0.005)
    short = {"side": -1, "entry": 100.0, "sl": 101.0, "tp": 98.0}
    assert evaluate_bars(short, [(0, 100, 101.2, 97.5, 99)])[0] == "sl"      # both touched -> stop
    assert evaluate_bars(sig, [(0, 100, 100.4, 99.6, 100)])[0] is None


def test_exchange_parsers():
    raw = [[2000, "1", "2", "0.5", "1.5", "10", 2999, "15", 7, "4", "6"], [1000, "1", "1", "1", "1", "1", 1999, "1", 1, "0", "0"]]
    rows = _kline_rows(raw)
    assert [r[0] for r in rows] == [1000, 2000] and rows[1] == (2000, 1.0, 2.0, 0.5, 1.5, 10.0, 15.0, 7, 4.0, 6.0)
    b = _book_stats([["99.9", "10"]], [["100.1", "5"]])
    assert b["spread_bps"] == pytest.approx(20.0) and b["imb"] > 0
    assert _book_stats([], [["1", "1"]]) is None
