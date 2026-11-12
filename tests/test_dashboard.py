import asyncio
import json
import time

from aiohttp.test_utils import TestClient, TestServer

from lts.config import Config
from lts.dashboard import build_app, make_token, snapshot, valid_token
from lts.db import DB

NOW = int(time.time() * 1000)


def make_db(tmp_path):
    db = DB(tmp_path / "d.sqlite")
    db.set("phase", "running")
    db.set("heartbeat", {"ts": NOW, "bar_ts": NOW - 300_000, "candidates": 3, "signals": 1, "errors": 0,
                         "collect_s": 12.0, "scan_s": 6.0, "telegram": True})
    db.insert_klines("BTCUSDT", [(NOW // 300_000 * 300_000 - 300_000, 1, 2, 0.5, 1.5, 10, 15, 7, 4, 6)])
    db.execute("INSERT INTO models(version,trained_ts,n_train,metrics,threshold,floor,path,status) VALUES(?,?,?,?,?,?,?,?)",
               ("m1", NOW, 9790, json.dumps({"xsec_corr": 0.042, "final_pick": {"n": 41, "avg_r": 0.33, "t": 1.4}}),
                0.16, 0.11, "x", "shadow"))
    rows = [("AAAUSDT", 1, NOW - 7_200_000, "P", "tp", NOW - 3_600_000, 1.9, 0.11),
            ("BBBUSDT", -1, NOW - 5_000_000, "P", "sl", NOW - 4_000_000, -1.2, -0.07),
            ("CCCUSDT", 1, NOW - 600_000, "P", "open", None, None, None)]
    db.executemany("""INSERT INTO signals(symbol, side, ts, grade, entry, sl, tp, sl_pct, tp_pct, leverage, status,
                      exit_ts, ret_r, ret_margin, expires_ts) VALUES (?,?,?,?,1.0,0.99,1.02,0.01,0.02,5,?,?,?,?,?)""",
                   [(s, sd, ts, g, st, ex, r, m, ts + 28_800_000) for s, sd, ts, g, st, ex, r, m in rows])
    db.execute("""INSERT INTO candidates(symbol, ts, side, kind, source, ref_price, atr_pct, features, rule_score,
                  model_p, decision) VALUES ('AAAUSDT', ?, 1, 'trend', 'live', 1.0, 0.006, '{}', 50, 0.4,
                  'skipped:below_threshold')""", (NOW - 300_000,))
    return db


def cfg_for(tmp_path):
    cfg = Config()
    cfg.storage.db_path = str(tmp_path / "d.sqlite")
    return cfg


def test_snapshot_summarises_signals_and_health(tmp_path):
    make_db(tmp_path)
    d = snapshot(tmp_path / "d.sqlite")
    assert d["phase"] == "running" and d["mode"] == "paper" and d["model"]["version"] == "m1"
    assert len(d["open"]) == 1 and d["open"][0]["symbol"] == "CCCUSDT"
    assert d["stats"]["all"]["n"] == 2 and abs(d["stats"]["all"]["total_r"] - 0.7) < 1e-9
    assert d["stats"]["paper"]["n"] == 2 and d["stats"]["real"]["n"] == 0
    assert d["candidates"][0]["decision"] == "skipped:below_threshold"
    assert d["heartbeat"]["candidates"] == 3 and d["data"]["last_bar_ts"] is not None
    json.dumps(d)                                       # must be serialisable as-is


def test_snapshot_before_the_database_exists(tmp_path):
    assert snapshot(tmp_path / "missing.sqlite")["empty"] is True


def test_tokens_expire_and_cannot_be_forged():
    secret = b"k" * 32
    assert valid_token(secret, make_token(secret))
    assert not valid_token(secret, make_token(secret, days=-1))
    assert not valid_token(b"other" * 8, make_token(secret))
    assert not valid_token(secret, None) and not valid_token(secret, "123.abc")


def test_login_is_required_and_rate_limited(tmp_path):
    make_db(tmp_path)

    async def go():
        app = build_app(cfg_for(tmp_path), password="correct horse", need_login=True, cookie_secure=False)
        async with TestClient(TestServer(app)) as c:
            assert (await c.get("/")).status == 200                       # the page shell itself is public
            assert (await c.get("/api/data")).status == 401               # the data is not
            assert (await c.post("/login", json={"password": "nope"})).status == 401
            r = await c.post("/login", json={"password": "correct horse"})
            assert r.status == 200
            r = await c.get("/api/data")
            assert r.status == 200 and (await r.json())["mode"] == "paper"
            assert r.headers["Cache-Control"] == "no-store"
        app = build_app(cfg_for(tmp_path), password="correct horse", need_login=True, cookie_secure=False)
        async with TestClient(TestServer(app)) as c:
            for _ in range(10):
                assert (await c.post("/login", json={"password": "guess"})).status == 401
            assert (await c.post("/login", json={"password": "correct horse"})).status == 429   # locked out

    asyncio.run(go())


def test_no_login_on_localhost(tmp_path):
    make_db(tmp_path)

    async def go():
        app = build_app(cfg_for(tmp_path), password="", need_login=False, cookie_secure=False)
        async with TestClient(TestServer(app)) as c:
            assert (await c.get("/api/data")).status == 200

    asyncio.run(go())
