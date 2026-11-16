import pytest

from lts.db import DB
from lts.microstructure import book_metrics, ensure_schema, parse_liquidation, prune


def test_liquidation_message_is_parsed():
    msg = {"e": "forceOrder", "E": 1794900000123,
           "o": {"s": "SOLUSDT", "S": "SELL", "o": "LIMIT", "f": "IOC", "q": "12", "p": "70.10", "ap": "70.00",
                 "X": "FILLED", "l": "12", "z": "12", "T": 1794900000100}}
    sym, ts, side, price, qty, usd = parse_liquidation(msg)
    assert (sym, ts, side, price, qty) == ("SOLUSDT", 1794900000100, "long_liq", 70.0, 12.0)
    assert usd == pytest.approx(840.0)
    msg["o"]["S"] = "BUY"
    assert parse_liquidation(msg)[2] == "short_liq"
    assert parse_liquidation({"e": "other"}) is None


def test_book_metrics_depth_and_walls():
    bids = [[99.95, 10], [99.7, 10], [99.2, 300], [98.5, 10], [97.0, 10]]       # big bid 0.75% below mid
    asks = [[100.05, 10], [100.3, 10], [100.8, 10], [101.5, 10], [103.0, 10]]
    m = book_metrics(bids, asks)
    assert m["mid"] == pytest.approx(100.0) and m["spread_bps"] == pytest.approx(10.0)
    assert m["bid05"] == pytest.approx(99.95 * 10 + 99.7 * 10)
    assert m["bid1"] > m["bid05"] and m["bid2"] >= m["bid1"]
    assert m["big_bid_dist"] == pytest.approx(0.008) and m["big_bid_x"] > 10
    assert book_metrics([], asks) is None


def test_prune_keeps_recent_rows(tmp_path):
    import time
    db = DB(tmp_path / "m.sqlite")
    ensure_schema(db)
    now = int(time.time() * 1000)
    db.insert_rows("liquidations", ("symbol", "ts", "side", "price", "qty", "usd"),
                   [("A", now - 200 * 86_400_000, "long_liq", 1, 1, 1), ("A", now, "long_liq", 1, 1, 1)])
    prune(db, 180)
    assert db.query("SELECT COUNT(*) c FROM liquidations")[0]["c"] == 1
