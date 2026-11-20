import json

import numpy as np

from lts import grindsignal as GS
from lts.config import Config
from lts.db import DB
from lts.exits import RULES, shadow
from lts.grindlab import ensure_live_schema

HOUR, BAR = 3_600_000, 300_000
NOW = 1_794_900_000_000 // HOUR * HOUR


def path(closes, buy=0.6):
    c = np.asarray(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    qv = np.full(len(c), 1e4)
    return np.stack([o, np.maximum(o, c) * 1.0002, np.minimum(o, c) * 0.9998, c, qv, qv * buy], 1)


def test_rules_on_a_winner_that_turns_into_a_loser():
    up = np.linspace(100, 101.2, 25)                     # +1.2% in 2h
    down = np.linspace(101.2, 98.8, 25)[1:]              # then through the -1% stop
    p = path(np.concatenate([up, down, np.full(240, 98.8)]))
    p[25:, 5] = p[25:, 4] * 0.2                          # takers turn to selling at the top
    r = shadow(p, np.full(72, 99.5), 1, 0.02, 0.01)
    assert set(r) == set(RULES)
    assert r["none"]["exit"] == "stop" and r["none"]["net"] < -0.011
    for rule in ("back_0.5", "ema_1", "give_0.75", "flow15", "flow_ema"):
        assert r[rule]["exit"] == "rule" and r[rule]["net"] > r["none"]["net"], rule
        assert r[rule]["min"] < r["none"]["min"]


def test_rules_cost_a_winner_that_dips_first():
    p = path(np.concatenate([np.linspace(100, 100.8, 13), np.linspace(100.8, 99.6, 13)[1:],
                             np.linspace(99.6, 102.5, 40)[1:], np.full(240, 102.5)]))
    r = shadow(p, np.full(72, 99.5), 1, 0.02, 0.01)
    assert r["none"]["exit"] == "target" and r["none"]["net"] > 0.018
    assert r["back_0.5"]["exit"] == "rule" and r["back_0.5"]["net"] < 0      # closed at entry before the run


def test_exits_are_labelled_after_24h(tmp_path):
    cfg, db = Config(), DB(tmp_path / "e.sqlite")
    ensure_live_schema(db)
    db.execute("INSERT INTO grind_live(symbol, ts, side, price, score) VALUES ('AUSDT', ?, 1, 100.0, 5)", (NOW,))
    p = path(np.concatenate([np.full(72, 100.0), np.linspace(100, 102.5, 30), np.full(260, 102.5)]))
    db.insert_klines("AUSDT", [(NOW - 72 * BAR + i * BAR, *row[:4], 1.0, row[4], 10, 0.5, row[5]) for i, row in enumerate(p)])
    assert GS.label_exits(cfg, db, now=NOW + 25 * HOUR) == 1
    e = json.loads(db.query("SELECT exits FROM grind_live")[0]["exits"])
    assert e["none"]["exit"] == "target" and set(e) == set(RULES)
