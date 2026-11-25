"""lts/validate.py: day-clustered statistics, Holm correction, the locked confirmation split."""
import numpy as np
import pandas as pd
import pytest

from lts import validate as V

DAY = V.DAY


def test_day_clustered_t_uses_daily_means():
    ts = np.repeat(np.arange(100) * DAY, 10)                 # 10 identical trades a day: 100 observations, not 1000
    r = np.repeat(np.where(np.arange(100) % 2 == 0, 0.3, -0.1), 10)
    st = V.stats(r, ts)
    daily = pd.Series(r).groupby(ts // DAY).mean()
    assert st["n"] == 1000 and st["days"] == 100
    assert st["t_day"] == pytest.approx(daily.mean() / (daily.std(ddof=1) / 10))
    assert st["mean"] == pytest.approx(0.1) and 0 < st["months_pos"] <= 1


def test_holm_is_never_more_lenient_and_matches_by_hand():
    t = [4.0, 2.0, 3.0]
    adj = V.holm(t)
    assert all(a <= b + 1e-9 for a, b in zip(adj, t))
    p = sorted(V._norm_sf(x) for x in t)                     # smallest p first: x3, x2, x1 multipliers
    by_hand = [max(3 * p[0], 0), max(3 * p[0], 2 * p[1]), max(3 * p[0], 2 * p[1], p[2])]
    assert V._norm_sf(adj[0]) == pytest.approx(by_hand[0], rel=1e-6)
    assert V._norm_sf(adj[2]) == pytest.approx(by_hand[1], rel=1e-6)
    assert V._norm_sf(adj[1]) == pytest.approx(by_hand[2], rel=1e-6)


def test_pass_bar():
    good = {"n": 200, "span_days": 90, "mean": 0.15, "months_pos": 0.7}
    assert V.passes_discovery(good, 3.2)[0]
    ok, why = V.passes_discovery(good | {"mean": 0.05}, 3.2)
    assert not ok and "mean" in why[0]
    assert not V.passes_discovery(good, 2.9)[0] and not V.passes_discovery(good | {"n": 100}, 5)[0]
    assert V.passes_confirmation({"n": 50, "mean": 0.1, "t_day": 2.1}) and not V.passes_confirmation({"n": 50, "mean": 0.1, "t_day": 1.9})


def test_confirmation_is_locked_until_frozen_and_cannot_change(tmp_path):
    sp = V.Split("demo", conf_start=10 * DAY, results_dir=tmp_path)
    ts = np.arange(20) * DAY
    assert sp.discovery(ts).sum() == 10
    with pytest.raises(RuntimeError):
        sp.confirmation(ts, "rule A")
    sp.freeze({"rule A": {"stop": 0.05}})
    assert sp.confirmation(ts, "rule A").sum() == 10
    with pytest.raises(RuntimeError):
        sp.confirmation(ts, "rule B")                        # not frozen
    with pytest.raises(RuntimeError):
        V.Split("demo", 10 * DAY, tmp_path).freeze({"rule A": {"stop": 0.04}})     # cannot re-freeze differently


def test_alt_regime_uses_only_past_days():
    days = np.arange(120) * DAY
    up = pd.DataFrame({f"C{i}": np.exp(np.linspace(0, 1, 120)) for i in range(40)}, index=days)
    reg = V.alt_regime(up)
    assert reg.iloc[:50].isna().all() and (reg.iloc[60:] == 1).all()
    down = up.copy()
    down.iloc[100:] = down.iloc[100:].to_numpy() * np.exp(-np.linspace(0, 2, 20))[:, None]
    assert (V.alt_regime(down).iloc[-3:] == -1).all()
