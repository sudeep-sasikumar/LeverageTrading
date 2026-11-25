"""Review items 1a, 2, 3 and 13 (25 Nov 2026): the steady rule checked like the year test, grind signals that
survive a Weex error and need no model, and the generated rule record (docs/DESIGN.md section 28)."""
import asyncio
import time
from pathlib import Path
from types import SimpleNamespace


from lts import grindsignal as GS
from lts.config import Config
from lts.db import DB
from lts.grindlab import ensure_live_schema

ROOT = Path(__file__).resolve().parent.parent
HOUR, MIN = 3_600_000, 60_000
T0 = 1_794_900_000_000 // HOUR * HOUR


# ---- item 1a: the steady rule is checked like the year test -----------------------------------------------
def test_hour_close_first_checks_only_the_first_detection_per_window(tmp_path):
    from lts.service import Service
    cfg, db = Config(), DB(tmp_path / "f.sqlite")
    ensure_live_schema(db)
    me = SimpleNamespace(cfg=cfg, db=db, gs_bundle=None, gs_loaded=time.time(), grind_levels=lambda rows, db=None: None)
    steady = {"r_24h": 0.10, "r2_24h": 0.85, "hours_with24": 16, "r2_6h": 0.93, "hours_with6": 5, "big6": 0.30}

    def scan(end, sym="AUSDT", **kw):
        rows = [{"symbol": sym, "ts": end, "side": 1, "price": 1.0, "score": 3, "model_p": None, "checks": {},
                 **steady, **kw}]
        return Service.grinds_to_record(me, rows, end)
    db.set("grind_first_seeded", T0 - HOUR)                            # running before this coin appeared
    # first detection at 10:05 (not an hour close): recorded, but the rule is not checked there
    k = scan(T0 + 5 * MIN)
    assert len(k) == 1 and not k[0]["selected"]
    # the year test's first decision for this coin is at the next hour close it is detected; but the coin was
    # already detected at 10:05 and nothing was recorded at an hour close before, so 11:00 opens the window
    k = scan(T0 + HOUR)
    assert len(k) == 1 and k[0]["first"] and k[0]["selected"]
    # within 12h: never selected again (no re-sends), at hour closes or between them
    for t in (T0 + 2 * HOUR, T0 + 5 * HOUR + 5 * MIN, T0 + 12 * HOUR):
        for r in scan(t):
            assert not r["selected"]
    # 12h after the window opened: the next hour-close detection opens a new window
    k = scan(T0 + 13 * HOUR)
    assert k and k[0]["first"] and k[0]["selected"]
    # a first detection that is not steady opens the window anyway (the year test took its first moment)
    k = scan(T0 + 20 * HOUR, sym="BUSDT", r_24h=0.02)
    assert k[0]["first"] and not k[0]["selected"]
    k = scan(T0 + 22 * HOUR, sym="BUSDT")                              # steady now, but inside B's window
    assert k and not k[0]["first"] and not k[0]["selected"]


def test_hour_close_first_after_deploy_treats_already_grinding_coins_as_seen(tmp_path):
    cfg, db = Config(), DB(tmp_path / "d.sqlite")
    ensure_live_schema(db)
    db.execute("INSERT INTO grind_live(symbol, ts, side, price, score) VALUES ('OLDUSDT', ?, -1, 1.0, 3)", (T0 - 3 * HOUR,))
    rows = [{"symbol": "OLDUSDT", "side": -1}, {"symbol": "NEWUSDT", "side": -1}]
    assert GS.mark_first(db, rows, T0, cfg.grindsignal) == {("NEWUSDT", -1)}


def test_rolling_5m_keeps_the_old_behaviour(tmp_path):
    from lts.service import Service
    cfg, db = Config(), DB(tmp_path / "r.sqlite")
    cfg.grindsignal.steady_eval = "rolling_5m"
    ensure_live_schema(db)
    me = SimpleNamespace(cfg=cfg, db=db, gs_bundle=None, gs_loaded=time.time(), grind_levels=lambda rows, db=None: None)
    row = {"symbol": "AUSDT", "ts": T0 + 5 * MIN, "side": 1, "price": 1.0, "score": 3, "model_p": None, "checks": {},
           "r_24h": 0.10, "r2_24h": 0.85, "hours_with24": 16, "r2_6h": 0.93, "hours_with6": 5, "big6": 0.30}
    k = Service.grinds_to_record(me, [row], T0 + 5 * MIN)
    assert k[0]["selected"]                                            # checked at any 5-minute record


# ---- items 2 and 3: a Weex error or a missing model does not lose the cycle -------------------------------
def test_grind_signals_survive_a_weex_error_and_need_no_model(tmp_path, monkeypatch):
    from lts.service import Service
    cfg, db = Config(), DB(tmp_path / "w.sqlite")
    ensure_live_schema(db)
    cfg.grindsignal.require_positive_ev = False
    decided = []

    class Engine:
        async def decide(self, picks, bundle, now):
            decided.extend(picks)
            for p in picks:
                p["decision"] = "skipped:no_weex_book" if p["book"] is None else "sent"

    class BadWeex:
        async def book(self, sym):
            raise RuntimeError("Weex 502")
    me = SimpleNamespace(cfg=cfg, db=db, weex=BadWeex(), engine=Engine(), collector=SimpleNamespace(weex_tickers={}),
                         market_now=lambda sym, side, end: {})
    row = {"symbol": "AUSDT", "ts": T0, "side": 1, "price": 1.0, "score": 3, "selected": True, "r_24h": 0.1}
    db.execute("INSERT INTO grind_live(symbol, ts, side, price, score) VALUES ('AUSDT', ?, 1, 1.0, 3)", (T0,))
    asyncio.run(Service.grind_signals(me, [row], {"AUSDT": {"on_weex": 1}}))      # no gs_bundle at all
    assert len(decided) == 1 and decided[0]["book"] is None
    assert db.query("SELECT decision FROM grind_live")[0]["decision"] == "skipped:no_weex_book"


# ---- item 13: the steady rule's record is generated, not typed in -----------------------------------------
def test_steady_history_is_loaded_from_the_generated_file():
    assert not hasattr(GS, "STEADY_HISTORY")
    h = GS.load_history()
    if h:                                                              # generated by research/reproduce_r3.py
        assert h.get("generated_by") in ("research/reproduce_r3.py", "research/parity.py")
        rec = h["rules"]["steady/hour_close_first"]["plans"]["5x +2.00% / -1.00%"]
        assert {"n", "win", "net", "se_day", "t_day", "months_pos", "n_months"} <= set(rec)


