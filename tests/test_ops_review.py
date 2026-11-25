"""Review items 16 and 19 (25 Nov 2026): cached grind-lab bundle, nightly schedules."""
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lts.config import Config
from lts.db import DB

ROOT = Path(__file__).resolve().parent.parent
HOUR, MIN = 3_600_000, 60_000
T0 = 1_794_900_000_000 // HOUR * HOUR


# ---- item 16: the grind-lab bundle is cached ----------------------------------------------------------------
def test_grindlab_bundle_is_loaded_once_until_the_file_changes(tmp_path, monkeypatch):
    import os
    import joblib
    from lts import grindlab as G
    cfg = Config()
    cfg.storage.model_dir = str(tmp_path)
    joblib.dump({"v": 1}, G.bundle_path(cfg))
    loads = []
    real = G.load_bundle
    monkeypatch.setattr(G, "load_bundle", lambda c: loads.append(1) or real(c))
    sc = G.FiveMinuteScanner(cfg)
    assert sc.bundle() == {"v": 1} and sc.bundle() == {"v": 1} and len(loads) == 1
    joblib.dump({"v": 2}, G.bundle_path(cfg))
    os.utime(G.bundle_path(cfg), (time.time() + 5, time.time() + 5))
    assert sc.bundle() == {"v": 2} and len(loads) == 2


# ---- item 19: nightly jobs that never send can be paused or run weekly ----------------------------------------
def test_nightly_schedules():
    from lts.service import scheduled
    mon, tue = datetime(2026, 11, 23, tzinfo=timezone.utc), datetime(2026, 11, 24, tzinfo=timezone.utc)
    assert scheduled("daily", tue) and scheduled("weekly", mon) and not scheduled("weekly", tue)
    assert not scheduled("off", mon)
    c = Config()
    assert c.learn.retrain_schedule == "weekly" and c.grindlab.study_schedule == "weekly"


def test_nightly_refresh_only_skips_the_study(tmp_path, monkeypatch):
    from lts import grindlab as G
    from lts import grindlab_data
    called = []

    async def refresh(db, **kw):
        called.append("refresh")
    monkeypatch.setattr(grindlab_data, "refresh", refresh)
    monkeypatch.setattr(G, "build", lambda *a, **k: called.append("build"))
    rep = G.nightly(Config(), DB(tmp_path / "n.sqlite"), study=False)
    assert called == ["refresh"] and rep["refresh_only"]


def test_parity_scan_matches_hourly_bars_on_an_hour_close(tmp_path):
    """On an hour close the rolling hours built from 5m candles equal the clock hours the year test used."""
    from lts.grindlab import _rolling_hours
    db = DB(tmp_path / "p.sqlite")
    end = T0
    rng = np.random.default_rng(1)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 30 * 12)))
    rows = [("XUSDT", end - 30 * HOUR + i * 300_000, px[i], px[i] * 1.001, px[i] * 0.999, px[i], 1, 10, 1, 0.5, 5)
            for i in range(30 * 12)]
    db.insert_rows("klines", ("symbol", "ts", "o", "h", "l", "c", "v", "qv", "trades", "tbv", "tbqv"), rows)
    k = _rolling_hours(db, "XUSDT", end)
    assert list(k.index) == [end - 30 * HOUR + i * HOUR for i in range(30)]
    assert abs(k["c"].iloc[-1] - px[-1]) < 1e-9 and abs(k["o"].iloc[0] - px[0]) < 1e-9


def test_dashboard_shows_scan_timings(tmp_path):
    import lts.dashboard as D
    db = DB(tmp_path / "h.sqlite")
    db.set("heartbeat", {"ts": 1, "collect_s": 8.1, "scan_s": 3.0})
    db.set("grind_heartbeat", {"ts": 1, "grind_s": 21.5})
    snap = D.snapshot(tmp_path / "h.sqlite")
    assert snap["heartbeat"]["collect_s"] == 8.1 and snap["grind_heartbeat"]["grind_s"] == 21.5
    html = (ROOT / "lts" / "web" / "index.html").read_text(encoding="utf-8")
    assert "momentum scan" in html and "grind scan" in html


def test_retrained_grind_model_must_beat_the_seed_on_the_same_unseen_months(tmp_path, monkeypatch):
    """Review item 17: a model trained on the server's shorter history replaces the year-trained seed only if
    it does better on months neither model saw."""
    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from lts import grindsignal as GS
    rng = np.random.default_rng(0)
    n = 4000
    f = rng.normal(size=n)
    net = np.where(f > 0, 0.01, -0.01)                         # high f wins
    seed = LogisticRegression().fit(f[:, None], (f < 0).astype(int))     # a seed that prefers LOW f (bad)
    joblib.dump({"model": seed, "features": ["sh_f"], "data_to": 0}, tmp_path / "seed.joblib")
    monkeypatch.setattr(GS, "SEED_PATH", tmp_path / "seed.joblib")
    label = GS.plan_label(Config().grindsignal.plans[1])
    df = pd.DataFrame({"ts": np.arange(n) * 3_600_000 + 1, "sh_f": f, label: net})
    good = {"plans": Config().grindsignal.plans, "primary": 1, "oos": pd.DataFrame({"ts": df.ts, "p": f, "net": net})}
    v = GS.beats_seed(good, df, 0.2)
    assert v["replace"] and v["new_net"] > v["seed_net"]
    bad = dict(good, oos=pd.DataFrame({"ts": df.ts, "p": -f, "net": net}))
    assert not GS.beats_seed(bad, df, 0.2)["replace"]
    few = dict(good, oos=good["oos"].head(100))
    assert not GS.beats_seed(few, df, 0.2)["replace"]


def test_dashboard_shows_the_signal_policy(tmp_path, monkeypatch):
    import json
    import lts.dashboard as D
    from lts import grindsignal as GS
    plans = {GS.plan_name(pl): {"n": 539, "net": -0.0008, "se_day": 0.001, "win": 0.38} for pl in Config().grindsignal.plans}
    h = tmp_path / "h.json"
    h.write_text(json.dumps({"rules": {"steady/hour_close_first": {"plans": plans, "passed_bar": False}}}), encoding="utf-8")
    monkeypatch.setattr(GS, "HISTORY_PATH", h)
    db = DB(tmp_path / "p.sqlite")
    db.set("phase", "running")
    sp = D.snapshot(tmp_path / "p.sqlite")["signal_policy"]
    assert sp["rule"] == "steady/hour_close_first" and not sp["sending"] and abs(sp["expected"]["net"] + 0.0008) < 1e-12
    assert sp["expected"]["lo"] < sp["expected"]["net"] < sp["expected"]["hi"] and sp["paper_trades"] == 0
    html = (ROOT / "lts" / "web" / "index.html").read_text(encoding="utf-8")
    assert "Signal policy (expected net > 0 to send)" in html
