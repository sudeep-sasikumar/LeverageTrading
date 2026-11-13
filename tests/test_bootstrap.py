import pytest

from lts import bootstrap as B
from lts.config import Config
from lts.dashboard import snapshot
from lts.db import DB
from lts.exchanges import ApiError


class Stop(Exception):
    pass


def test_region_block_is_explained_reported_once_and_retried(tmp_path, monkeypatch):
    cfg = Config()
    cfg.storage.db_path = str(tmp_path / "b.sqlite")
    db = DB(cfg.path(cfg.storage.db_path))
    calls, alerts, sleeps = [], [], []

    def failing_bootstrap(cfg, db):
        db.set("phase", "backfill")
        calls.append(1)
        raise ApiError(451, '{"code":0,"msg":"Service unavailable from a restricted location"}',
                       "https://fapi.binance.com/fapi/v1/exchangeInfo")

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:
            raise Stop()

    monkeypatch.setattr(B, "bootstrap", failing_bootstrap)
    monkeypatch.setattr(B, "notify", lambda cfg, text: alerts.append(text))
    monkeypatch.setattr(B.time, "sleep", fake_sleep)
    monkeypatch.setattr("lts.dashboard.start_in_thread", lambda cfg: None)

    with pytest.raises(Stop):
        B.main(cfg, db)

    assert len(calls) == 3                                  # kept retrying instead of exiting
    assert sleeps == [60, 120, 300]                         # with back-off
    assert len(alerts) == 1 and "restricted" not in alerts[0] and "not the US" in alerts[0]
    err = db.get("last_error")
    assert err["attempt"] == 3 and err["phase"] == "backfill" and "HTTP 451" in err["message"]
    d = snapshot(tmp_path / "b.sqlite")
    assert d["error"]["message"] == err["message"]          # shown on the dashboard


def test_dashboard_clears_the_error_after_a_good_scan(tmp_path):
    db = DB(tmp_path / "c.sqlite")
    db.set("last_error", {"ts": 1_000, "message": "x", "attempt": 1})
    db.set("heartbeat", {"ts": 2_000})
    assert snapshot(tmp_path / "c.sqlite")["error"] is None


def test_explanations():
    assert "not the US" in B.explain(ApiError(451, "", "https://fapi.binance.com/x"))
    assert "Out of memory" in B.explain(MemoryError())
    assert B.explain(ValueError("boom")) == "ValueError: boom"
