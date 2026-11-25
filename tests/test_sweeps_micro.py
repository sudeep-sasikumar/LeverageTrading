"""Study B4 code: refuses to run before enough data, its joins, and the lead-lag measure."""
import numpy as np
import pandas as pd
import pytest

from lts import sweeps_micro as SM
from lts.db import DB
from lts.leadlag import lead_lag
from lts.microstructure import ensure_schema

DAY, MIN = SM.DAY, SM.MIN
T0 = 1_795_000_000_000 // DAY * DAY


def test_refuses_before_sixty_days_and_locks_confirmation(tmp_path):
    db = DB(tmp_path / "m.sqlite")
    ensure_schema(db)
    with pytest.raises(SM.NotEnoughData):
        SM.require(db)
    db.execute("INSERT INTO liquidations VALUES ('XUSDT', ?, 'long_liq', 1, 1, 1000)", (T0,))
    db.execute("INSERT INTO book_snapshots(symbol, ts, mid) VALUES ('XUSDT', ?, 1)", (T0,))
    db.execute("INSERT INTO liquidations VALUES ('XUSDT', ?, 'long_liq', 1, 1, 1000)", (T0 + 30 * DAY,))
    db.execute("INSERT INTO book_snapshots(symbol, ts, mid) VALUES ('XUSDT', ?, 1)", (T0 + 30 * DAY,))
    with pytest.raises(SM.NotEnoughData, match="first run on"):
        SM.require(db)
    for t in (T0 + 61 * DAY,):
        db.execute("INSERT INTO liquidations VALUES ('XUSDT', ?, 'long_liq', 1, 1, 1000)", (t,))
        db.execute("INSERT INTO book_snapshots(symbol, ts, mid) VALUES ('XUSDT', ?, 1)", (t,))
    assert SM.require(db)["can_discover"]
    with pytest.raises(SM.NotEnoughData, match="locked"):
        SM.require(db, "confirmation")
    assert SM.periods(db)["first_run_date"] == pd.to_datetime(T0 + 60 * DAY, unit="ms").strftime("%Y-%m-%d")


def test_liquidation_and_book_context():
    liq = pd.DataFrame({"ts": [T0 - 5 * MIN, T0 + 5 * MIN, T0 - 30 * MIN], "side": ["long_liq", "short_liq", "long_liq"],
                        "usd": [100.0, 40.0, 999.0]})
    c = SM.liq_context(liq, T0, side=1)                     # a low sweep: longs were the swept side
    assert c == {"swept_before": 100.0, "other_before": 0.0, "swept_after": 0.0, "other_after": 40.0}
    books = pd.DataFrame([{"ts": T0 - 2 * MIN, "bid05": 3, "ask05": 1, "bid1": 6, "ask1": 2, "bid2": 9, "ask2": 9}])
    b = SM.book_context(books, T0, side=1)
    assert b["imb1_before"] == pytest.approx(0.5) and b["age_before_min"] == 2.0 and "imb1_after" not in b


def test_lead_lag_finds_binance_leading_by_one_minute():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.001, 600)
    pb = 100 * np.exp(np.cumsum(r))
    pw = np.r_[pb[0], pb[:-1]] * (1 + rng.normal(0, 0.00005, 600))   # Weex follows one minute later
    ts = T0 + np.arange(600) * MIN
    bin_rows = [(int(t), 0, 0, 0, float(x)) for t, x in zip(ts, pb)]
    wx_rows = [(int(t), 0, 0, 0, float(x)) for t, x in zip(ts, pw)]
    res = lead_lag(bin_rows, wx_rows)
    assert res["best_lag"] == 1 and res["corr"][1] > 0.9 and abs(res["corr"][0]) < 0.2
