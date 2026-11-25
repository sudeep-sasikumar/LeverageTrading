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
            a, b = {"X-Forwarded-For": "203.0.113.7"}, {"X-Forwarded-For": "198.51.100.9"}
            for _ in range(5):                                            # 5 failed attempts from one IP ...
                assert (await c.post("/login", json={"password": "guess"}, headers=a)).status == 401
            assert (await c.post("/login", json={"password": "correct horse"}, headers=a)).status == 429
            assert (await c.post("/login", json={"password": "correct horse"}, headers=b)).status == 200  # other IP
            # a forged leading X-Forwarded-For entry does not dodge the limit: the last entry (Traefik's) counts
            forged = {"X-Forwarded-For": "1.2.3.4, 203.0.113.7"}
            assert (await c.post("/login", json={"password": "correct horse"}, headers=forged)).status == 429

    asyncio.run(go())


def test_no_login_on_localhost(tmp_path):
    make_db(tmp_path)

    async def go():
        app = build_app(cfg_for(tmp_path), password="", need_login=False, cookie_secure=False)
        async with TestClient(TestServer(app)) as c:
            assert (await c.get("/api/data")).status == 200

    asyncio.run(go())


def test_my_trades_log_edit_close_and_delete(tmp_path):
    make_db(tmp_path)

    async def go():
        app = build_app(cfg_for(tmp_path), password="pw", need_login=True, cookie_secure=False)
        async with TestClient(TestServer(app)) as c:
            assert (await c.post("/api/my_trades", json={"symbol": "x", "side": "long", "entry_price": 1})).status == 401
            await c.post("/login", json={"password": "pw"})
            # a plain form post (cross-site style) is refused
            assert (await c.post("/api/my_trades", data={"symbol": "AAAUSDT"})).status == 400
            assert (await c.post("/api/my_trades", json={"symbol": "AAA", "side": "long"})).status == 400   # no entry
            # the MEGA trade: 5x long, $100 margin, entry fee 0.3998, opened 10 minutes after the AAA signal
            r = await c.post("/api/my_trades", json={"symbol": "aaa", "side": "long", "leverage": 5, "margin": 100.2104,
                                                      "entry_price": 0.05457, "tp_price": 0.05566, "sl_price": 0.05402,
                                                      "open_fee": 0.3998, "open_ts": NOW - 7_200_000 + 600_000})
            tid = (await r.json())["id"]
            d = await (await c.get("/api/data")).json()
            t = d["my_trades"]["trades"][0]
            assert t["symbol"] == "AAAUSDT" and t["status"] == "open" and t["net"] is None
            assert t["signal"] and t["signal"]["ts"] == NOW - 7_200_000          # matched to the LTS signal
            # closed at the target: Weex realized PnL 9.9844, close fee 0.4078
            r = await c.post(f"/api/my_trades/{tid}", json={"exit_price": 0.05566, "close_fee": 0.4078, "realized_pnl": 9.9844,
                                                             "status": "closed", "exit_reason": "tp"})
            assert r.status == 200
            d = await (await c.get("/api/data")).json()
            t, tot = d["my_trades"]["trades"][0], d["my_trades"]["totals"]
            assert abs(t["net"] - (9.9844 - 0.3998 - 0.4078)) < 1e-9 and abs(t["net_pct"] - t["net"] / 100.2104) < 1e-9
            assert tot["closed"] == 1 and tot["wins"] == 1
            assert (await c.delete(f"/api/my_trades/{tid}")).status == 200
            d = await (await c.get("/api/data")).json()
            assert d["my_trades"]["trades"] == []

    asyncio.run(go())


def test_trading_database_stays_read_only(tmp_path):
    """Logging a trade writes only the separate journal file."""
    from lts import journal
    make_db(tmp_path)
    jp = journal.path_for(tmp_path / "d.sqlite")
    journal.add(jp, {"symbol": "BBBUSDT", "side": "short", "entry_price": 2.0, "exit_price": 1.96, "leverage": 5,
                     "margin": 50, "open_fee": 0.04, "close_fee": 0.04})
    t = journal.read(jp)["trades"][0]
    assert jp.name == "my_trades.sqlite" and t["status"] == "closed"
    assert abs(t["gross"] - 0.02 * 250) < 1e-9 and abs(t["net"] - (5.0 - 0.08)) < 1e-9     # worked out from prices
