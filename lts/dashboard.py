"""Web dashboard (aiohttp), served from a background thread.

It opens the trading SQLite file read-only, so it works during the first-start bootstrap as well as
while the service runs, and it can never change trading state. The one thing it writes is the
trader's own trade journal, a separate file (lts/journal.py).

Environment:
  DASHBOARD_PASSWORD       required whenever the dashboard is reachable beyond localhost
  DASHBOARD_BIND           default 127.0.0.1 (the Docker image sets 0.0.0.0)
  DASHBOARD_PORT           default 8020
  DASHBOARD_COOKIE_SECURE  true when served over HTTPS (set in docker-compose.yaml)
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from aiohttp import web

from . import journal
from .config import Config

log = logging.getLogger("lts.dashboard")
DAY = 86_400_000
COOKIE = "lts_session"
SESSION_DAYS = 14
LOCAL = ("127.0.0.1", "localhost", "::1")
INDEX = Path(__file__).parent / "web" / "index.html"


# ---- auth -------------------------------------------------------------------
def _secret(data_dir: Path) -> bytes:
    path = data_dir / "dashboard.secret"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(secrets.token_bytes(32))
    return path.read_bytes()


def make_token(secret: bytes, days: float = SESSION_DAYS) -> str:
    exp = str(int(time.time() + days * 86400))
    return f"{exp}.{hmac.new(secret, exp.encode(), hashlib.sha256).hexdigest()}"


def valid_token(secret: bytes, token: str | None) -> bool:
    if not token or "." not in token:
        return False
    exp, sig = token.split(".", 1)
    good = hmac.new(secret, exp.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, good) and exp.isdigit() and int(exp) > time.time()


# ---- data -------------------------------------------------------------------
def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _stats(rows: list[dict]) -> dict:
    rs = [r["ret_r"] for r in rows if r["ret_r"] is not None]
    if not rs:
        return {"n": 0}
    eq = peak = dd = 0.0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    margins = [r["ret_margin"] for r in rows if r["ret_margin"] is not None]
    return {"n": len(rs), "win": sum(1 for r in rs if r > 0) / len(rs), "avg_r": sum(rs) / len(rs),
            "total_r": sum(rs), "max_dd": dd, "avg_margin": sum(margins) / len(margins) if margins else 0.0,
            "tp": sum(1 for r in rows if r["status"] == "tp"), "sl": sum(1 for r in rows if r["status"] == "sl"),
            "time": sum(1 for r in rows if r["status"] == "time")}


STEADY_KEYS = ("r_24h", "r2_24h", "hours_with24", "r2_6h", "hours_with6", "big6")
ANAT_FACTORS = [  # (key, label, path in the anatomy JSON)
    ("mfe", "Best move before exit", ("mfe",)),
    ("to_peak", "Minutes to the peak", ("minutes_to_peak",)),
    ("peak_to_exit", "Minutes peak to exit", ("minutes_peak_to_exit",)),
    ("flow_run", "Taker flow, entry to peak", ("run", "taker_flow")),
    ("flow_turn", "Taker flow at the turn", ("turn", "taker_flow")),
    ("flow_fall", "Taker flow after the peak", ("fall", "taker_flow")),
    ("rvol_turn", "Volume at the turn (x before entry)", ("turn", "rvol")),
    ("rvol_fall", "Volume after the peak (x before entry)", ("fall", "rvol")),
    ("wick", "Wick against us on the peak candle", ("peak_wick_against",)),
    ("btc_fall", "BTC after the peak", ("fall", "btc")),
    ("mkt_fall", "Market (median coin) after the peak", ("fall", "mkt_med")),
    ("liq_against", "Liquidations against us after the peak ($)", ("fall", "liq_against_usd")),
    ("oi_run", "Open interest change, entry to peak", ("oi_chg_run",)),
    ("oi_fall", "Open interest change, peak to exit", ("oi_chg_fall",)),
    ("book_peak", "Order-book imbalance at the peak (1%)", ("book", "peak", "imb_1")),
    ("wall_peak", "Wall ahead at the peak (x median level)", ("book", "peak", "wall_ahead_x")),
    ("level_peak", "Nearest major level from the peak", ("peak_level", "dist_from_peak")),
]


def _dig(o, path):
    for k in path:
        if not isinstance(o, dict):
            return None
        o = o.get(k)
    return o if isinstance(o, (int, float)) and not isinstance(o, bool) else None


def exits_summary(rows: list[dict]) -> list[dict]:
    """Per shadow exit rule: the tracked plan's live record, for steady-rule grinds and for all grinds."""
    from .exits import RULES
    from .fees import fees
    from .grindsignal import steady
    gs = Config().grindsignal
    fee = fees()
    full = -gs.plans[gs.primary]["stop"] - fee.entry - fee.stop + 1e-9
    acc = {r: {"all": [], "steady": []} for r in RULES}
    for r in rows:
        e = json.loads(r["exits"] or "null") or {}
        if not e or e.get("missing"):
            continue
        f = {k: r.get(f"f_{k}") for k in STEADY_KEYS}
        is_steady = bool(r.get("sel")) if gs.selector != "steady" else (bool(r.get("sel")) or steady(f, gs))
        for rule, v in e.items():
            if rule in acc:
                acc[rule]["all"].append(v["net"])
                if is_steady:
                    acc[rule]["steady"].append(v["net"])
    out = []
    for rule, desc in RULES.items():
        row = {"rule": rule, "desc": desc}
        for g in ("all", "steady"):
            v = acc[rule][g]
            row[g] = {"n": len(v), "won": sum(1 for x in v if x > 0), "full": sum(1 for x in v if x <= full),
                      "net": (sum(v) / len(v)) if v else None}
        out.append(row)
    return out


def anatomy_summary(rows: list[dict]) -> dict:
    """Reversals (in profit, then stopped) vs trades that reached the target vs other stops: medians of each factor,
    and the list of recent reversals with their main clues."""
    groups = {"reversal": [], "target": [], "stop": []}
    rev = []
    for r in rows:
        a = json.loads(r["anatomy"] or "null")
        if not a or a.get("exit") == "time":
            continue
        g = "reversal" if a.get("reversal") else a.get("exit")
        if g in groups:
            groups[g].append(a)
        if g == "reversal":
            x = {k: _dig(a, p) for k, _, p in ANAT_FACTORS}
            ex = a.get("minutes_to_exit")
            for clue, key in (("ema", "first_close_through_ema20_min"), ("entry", "first_close_back_at_entry_min"),
                              ("swing", "first_close_through_swing_min")):
                v = a.get(key)
                x[f"lead_{clue}"] = (ex - v) if (v is not None and ex is not None) else None
            x.update({"symbol": r["symbol"], "ts": r["ts"], "side": r["side"],
                      "level": (a.get("peak_level") or {}).get("what"), "news": len(a.get("announcements") or [])})
            rev.append(x)
    med = []
    for k, label, p in ANAT_FACTORS:
        row = {"factor": label, "key": k}
        for g, items in groups.items():
            v = sorted(x for x in (_dig(a, p) for a in items) if x is not None)
            row[g] = v[len(v) // 2] if v else None
        med.append(row)
    return {"n": {g: len(v) for g, v in groups.items()}, "medians": med, "reversals": rev[:300]}


def snapshot(db_path: Path) -> dict:
    """Everything the page shows, in one read-only pass."""
    now = int(time.time() * 1000)
    out: dict = {"now": now, "version": os.environ.get("APP_VERSION", "dev")[:7]}
    if not db_path.exists():
        return out | {"phase": "starting", "empty": True}
    conn = _connect(db_path)
    try:
        def q(sql: str, params: tuple = ()) -> list[dict]:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

        kv = {r["key"]: json.loads(r["value"]) for r in q("SELECT key, value FROM kv")}
        out["phase"] = kv.get("phase", "unknown")
        out["paused"] = bool(kv.get("paused", False))
        out["heartbeat"] = kv.get("heartbeat")
        err = kv.get("last_error")
        hb_ts = (out["heartbeat"] or {}).get("ts", 0)
        out["error"] = err if err and err.get("ts", 0) > hb_ts else None     # cleared by the next good scan
        out["last_train_day"] = kv.get("last_train_day")

        last_bar = q("SELECT MAX(ts) AS t FROM klines WHERE symbol='BTCUSDT'")[0]["t"]
        out["data"] = {
            "last_bar_ts": last_bar,
            "symbols": q("SELECT COUNT(*) c FROM symbols WHERE active=1")[0]["c"],
            "candidates": q("SELECT COUNT(*) c FROM candidates")[0]["c"],
            "labelled": q("SELECT COUNT(*) c FROM candidates WHERE label IS NOT NULL")[0]["c"],
            "live_candidates": q("SELECT COUNT(*) c FROM candidates WHERE source='live'")[0]["c"],
            "candidates_24h": q("SELECT COUNT(*) c FROM candidates WHERE source='live' AND ts>=?", (now - DAY,))[0]["c"],
        }
        first_bar = q("SELECT MIN(ts) AS t FROM klines WHERE symbol='BTCUSDT'")[0]["t"]
        out["data"]["first_bar_ts"] = first_bar

        models = q("SELECT version, trained_ts, n_train, threshold, floor, status, metrics FROM models "
                   "ORDER BY trained_ts DESC LIMIT 12")
        for m in models:
            met = json.loads(m.pop("metrics") or "{}")
            pick, hold = met.get("final_pick") or {}, met.get("holdout_eval") or {}
            m.update({"xsec_corr": met.get("xsec_corr"), "base_rate": met.get("base_rate"),
                      "base_avg_r": met.get("base_avg_r"), "days": met.get("days"), "n_live": met.get("n_live"),
                      "oos_n": pick.get("n"), "oos_avg_r": pick.get("avg_r"), "oos_t": pick.get("t"),
                      "oos_win": pick.get("win"), "hold_n": hold.get("n"), "hold_avg_r": hold.get("avg_r"),
                      "paper_n": (met.get("paper") or {}).get("n"), "paper_avg_r": (met.get("paper") or {}).get("avg_r")})
        out["models"] = models
        active = next((m for m in models if m["status"] == "champion"), None) or \
            next((m for m in models if m["status"] == "shadow"), None)
        out["model"] = active
        out["mode"] = "live" if active and active["status"] == "champion" else "paper"

        has_kind = "kind" in [r["name"] for r in q("PRAGMA table_info(signals)")]
        kind_col = "COALESCE(kind,'momentum') AS kind" if has_kind else "'momentum' AS kind"
        cols = ("id, symbol, side, ts, grade, entry, sl, tp, sl_pct, tp_pct, leverage, margin_frac, p, ev_r, "
                f"model_version, expires_ts, status, exit_ts, exit_price, ret_r, ret_margin, mfe, mae, {kind_col}")
        out["open"] = q(f"SELECT {cols} FROM signals WHERE status='open' ORDER BY ts DESC")
        out["signals"] = q(f"SELECT {cols} FROM signals ORDER BY ts DESC LIMIT 300")
        closed = q(f"SELECT grade, status, exit_ts, ret_r, ret_margin, {kind_col} FROM signals WHERE status!='open' "
                   "AND ret_r IS NOT NULL ORDER BY exit_ts")
        day0 = now // DAY * DAY
        out["stats"] = {
            "today": _stats([r for r in closed if r["exit_ts"] >= day0]),
            "d7": _stats([r for r in closed if r["exit_ts"] >= now - 7 * DAY]),
            "d30": _stats([r for r in closed if r["exit_ts"] >= now - 30 * DAY]),
            "all": _stats(closed),
            "paper": _stats([r for r in closed if r["grade"] == "P"]),
            "real": _stats([r for r in closed if r["grade"] != "P"]),
            "momentum": _stats([r for r in closed if r["kind"] not in ("grind", "daily", "grindsig")]),
            "grind": _stats([r for r in closed if r["kind"] == "grind"]),
            "daily": _stats([r for r in closed if r["kind"] == "daily"]),
            "grindsig": _stats([r for r in closed if r["kind"] == "grindsig"]),
        }
        out["movers"] = kv.get("movers_latest")
        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='signal_plans'"):
            out["plan_stats"] = q("""SELECT plan, leverage, COUNT(*) n, AVG(net > 0) win, AVG(net) net, AVG(margin) margin,
                                     SUM(margin) total_margin FROM signal_plans GROUP BY plan ORDER BY leverage DESC""")
        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='liquidations'"):
            liq = q("SELECT COUNT(*) n, COALESCE(SUM(usd), 0) usd, MAX(ts) last FROM liquidations WHERE ts>=?", (now - DAY,))[0]
            books = q("SELECT COUNT(*) n, COUNT(DISTINCT symbol) s, MAX(ts) last FROM book_snapshots WHERE ts>=?", (now - DAY,))[0]
            out["micro"] = {"liq_24h": liq["n"], "liq_usd_24h": liq["usd"], "liq_last": liq["last"],
                            "books_24h": books["n"], "book_symbols": books["s"], "book_last": books["last"]}
        out["grindlab"] = kv.get("grindlab_report")
        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='grind_live'"):
            live = q("SELECT symbol, ts, side, price, score, model_p, catalyst, checks, factors, decision FROM grind_live "
                     "WHERE ts>=? ORDER BY ts DESC, score DESC LIMIT 300", (now - 2 * DAY,))
            for r in live:
                r["checks"] = json.loads(r["checks"] or "{}")
                r["factors"] = json.loads(r["factors"] or "{}")
            out["grind_live"] = live
        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='grind_live'"):
            cols_gl = {r["name"] for r in q("PRAGMA table_info(grind_live)")}
            if "outcomes" in cols_gl:
                prog = ", progress" if "progress" in cols_gl else ", NULL AS progress"
                prog += ", levels" if "levels" in cols_gl else ", NULL AS levels"
                prog += ", selected" if "selected" in cols_gl else ", NULL AS selected"
                if "all_factors" in cols_gl:            # the steady rule's inputs, for grinds recorded before it existed
                    prog += "".join(f", json_extract(all_factors, '$.{k}') AS f_{k}" for k in STEADY_KEYS)
                allg = q("SELECT symbol, ts, side, model_p, threshold, decision, outcomes" + prog + " FROM grind_live "
                         "WHERE ts>=? ORDER BY ts DESC LIMIT 2500", (now - 3 * DAY,))
                for r in allg:
                    o = json.loads(r.pop("outcomes") or "null") or {}
                    pr = json.loads(r.pop("progress") or "null") or {}
                    lv = json.loads(r.pop("levels") or "null")
                    r["levels"] = None if lv is None else {"block": [x["tf"] for x in lv.get("block", [])][:3],
                                                           "cushion": [x["tf"] for x in lv.get("cushion", [])][:3]}
                    r["plans"] = o.get("plans")
                    r["best24"] = (o.get("best") or {}).get("24")
                    r["worst24"] = (o.get("worst") or {}).get("24")
                    if not r["plans"] and pr:                 # under 24h: plans decided so far, moves so far
                        r["progress"] = {"plans": pr.get("plans"), "best": pr.get("best"), "worst": pr.get("worst"),
                                         "as_of": pr.get("as_of")}
                from .grindsignal import steady
                gs = Config().grindsignal
                for r in allg:
                    f = {k: r.pop(f"f_{k}", None) for k in STEADY_KEYS}
                    if r.get("selected") is None and gs.selector == "steady" and f.get("r_24h") is not None:
                        r["selected"] = int(steady(f, gs))
                        r["selected_retro"] = True
                out["grind_all"] = allg
                five = "5x +2.00% / -1.00%"
                def rec(rows):
                    v = [(r["plans"] or {}).get(five) if r["plans"] else ((r.get("progress") or {}).get("plans") or {}).get(five)
                         for r in rows]
                    v = [x for x in v if x is not None]
                    return {"n": len(v), "won": sum(1 for x in v if x > 0), "net": (sum(v) / len(v)) if v else None}
                out["rule_live"] = {"selected": rec([r for r in allg if r.get("selected")]),
                                    "others": rec([r for r in allg if r.get("selected") == 0])}
                if "exits" in cols_gl:
                    out["exits_live"] = exits_summary(q(
                        "SELECT exits, COALESCE(selected, 0) AS sel" +
                        ("".join(f", json_extract(all_factors, '$.{k}') AS f_{k}" for k in STEADY_KEYS) if "all_factors" in cols_gl else "") +
                        " FROM grind_live WHERE exits IS NOT NULL AND ts>=?", (now - 14 * DAY,)))
                if "anatomy" in cols_gl:
                    out["anatomy"] = anatomy_summary(q("SELECT symbol, ts, side, anatomy FROM grind_live WHERE anatomy IS NOT NULL "
                                                       "AND ts>=? ORDER BY ts DESC LIMIT 3000", (now - 7 * DAY,)))
                out["grinds_recorded"] = q("SELECT COUNT(*) n, SUM(outcomes IS NOT NULL) labelled FROM grind_live")[0]
        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='grind_episodes'"):
            out["grind_episodes"] = q("SELECT symbol, side, start, end, move, hours, tags FROM grind_episodes "
                                      "WHERE start>=? ORDER BY start DESC LIMIT 400", (now - 7 * DAY,))
        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_movers'"):
            hist = q("SELECT day, symbol, side, ret24, r2, hours_with, rvol24, oi_chg24, flagged_ts, flagged_kind, "
                     "flagged_left, signal_id, signal_status, signal_r FROM daily_movers ORDER BY day DESC, "
                     "ABS(ret24) * r2 DESC LIMIT 300")
            out["movers_history"] = hist
            recent = [h for h in hist if h["day"] >= time.strftime("%Y-%m-%d", time.gmtime(now / 1000 - 30 * 86400))]
            flagged = [h for h in recent if h["flagged_ts"]]
            lefts = sorted(h["flagged_left"] for h in flagged if h["flagged_left"] is not None)
            out["movers_stats"] = {"n": len(recent), "flagged": len(flagged),
                                   "median_left": lefts[len(lefts) // 2] if lefts else None,
                                   "signalled": sum(1 for h in recent if h["signal_id"])}
        out["sent_today"] = q("SELECT COUNT(*) c FROM signals WHERE ts>=?", (day0,))[0]["c"]
        sigs = q("SELECT id, symbol, side, ts, kind, entry, status, leverage FROM signals WHERE ts>=?", (now - 60 * DAY,))
        try:
            out["my_trades"] = journal.read(journal.path_for(db_path), sigs)
        except Exception as e:                                # never let the journal break the page
            log.warning("my trades: %s", e)
            out["my_trades"] = {"trades": [], "totals": {"n": 0}, "error": str(e)}
        out["candidates"] = q("SELECT id, symbol, side, ts, kind, atr_pct, rule_score, model_p, decision, outcome, "
                              "ret_r FROM candidates WHERE source='live' ORDER BY ts DESC, id DESC LIMIT 200")
        out["skips_24h"] = q("SELECT COALESCE(decision,'n/a') AS decision, COUNT(*) n FROM candidates "
                             "WHERE source='live' AND ts>=? GROUP BY decision ORDER BY n DESC", (now - DAY,))
    finally:
        conn.close()
    return out


# ---- server -----------------------------------------------------------------
def build_app(cfg: Config, password: str, need_login: bool, cookie_secure: bool) -> web.Application:
    db_path = cfg.path(cfg.storage.db_path)
    secret = _secret(db_path.parent)
    cache: dict = {"t": 0.0, "data": None}
    fails: list[float] = []

    @web.middleware
    async def auth(request: web.Request, handler):
        open_paths = ("/", "/login", "/healthz")
        if need_login and request.path not in open_paths and not valid_token(secret, request.cookies.get(COOKIE)):
            return web.json_response({"error": "login required"}, status=401)
        resp = await handler(request)
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    async def index(_request: web.Request):
        return web.Response(text=INDEX.read_text(encoding="utf-8"), content_type="text/html")

    async def login(request: web.Request):
        now = time.time()
        fails[:] = [t for t in fails if now - t < 900]
        if len(fails) >= 10:
            return web.json_response({"error": "too many attempts - wait 15 minutes"}, status=429)
        try:
            given = str((await request.json()).get("password", ""))
        except Exception:
            given = ""
        if not (password and hmac.compare_digest(given.encode(), password.encode())):
            fails.append(now)
            return web.json_response({"error": "wrong password"}, status=401)
        resp = web.json_response({"ok": True})
        resp.set_cookie(COOKIE, make_token(secret), max_age=SESSION_DAYS * 86400, httponly=True,
                        samesite="Strict", secure=cookie_secure)
        return resp

    async def data(_request: web.Request):
        if cache["data"] is None or time.time() - cache["t"] > 10:
            cache["data"] = await asyncio.to_thread(snapshot, db_path)
            cache["t"] = time.time()
        return web.json_response(cache["data"])

    async def healthz(_request: web.Request):
        return web.json_response({"ok": True})

    jpath = journal.path_for(db_path)

    async def _body(request: web.Request) -> dict:
        if request.content_type != "application/json":          # blocks plain cross-site form posts
            raise ValueError("send JSON")
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("send a JSON object")
        return body

    async def trade_add(request: web.Request):
        try:
            tid = await asyncio.to_thread(journal.add, jpath, await _body(request))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        cache["data"] = None
        return web.json_response({"ok": True, "id": tid})

    async def trade_update(request: web.Request):
        try:
            ok = await asyncio.to_thread(journal.update, jpath, int(request.match_info["id"]), await _body(request))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        cache["data"] = None
        return web.json_response({"ok": ok}, status=200 if ok else 404)

    async def trade_delete(request: web.Request):
        ok = await asyncio.to_thread(journal.delete, jpath, int(request.match_info["id"]))
        cache["data"] = None
        return web.json_response({"ok": ok}, status=200 if ok else 404)

    app = web.Application(middlewares=[auth])
    app.add_routes([web.get("/", index), web.post("/login", login), web.get("/api/data", data),
                    web.get("/healthz", healthz), web.post("/api/my_trades", trade_add),
                    web.post(r"/api/my_trades/{id:\d+}", trade_update), web.delete(r"/api/my_trades/{id:\d+}", trade_delete)])
    return app


def settings() -> dict:
    env = os.environ
    return {"bind": env.get("DASHBOARD_BIND") or "127.0.0.1", "port": int(env.get("DASHBOARD_PORT") or 8020),
            "password": env.get("DASHBOARD_PASSWORD") or "",
            "cookie_secure": (env.get("DASHBOARD_COOKIE_SECURE") or "").lower() in ("1", "true", "yes")}


async def serve(cfg: Config) -> None:
    s = settings()
    need_login = s["bind"] not in LOCAL
    if need_login and not s["password"]:
        log.warning("dashboard is OFF: set DASHBOARD_PASSWORD to enable it beyond localhost")
        return
    runner = web.AppRunner(build_app(cfg, s["password"], need_login, s["cookie_secure"]), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, s["bind"], s["port"]).start()
    log.info("dashboard on http://%s:%d (login %s)", s["bind"], s["port"], "required" if need_login else "not needed")
    while True:
        await asyncio.sleep(3600)


def start_in_thread(cfg: Config) -> None:
    """Run the dashboard beside whatever the main thread is doing (bootstrap or the service)."""
    def run():
        try:
            asyncio.run(serve(cfg))
        except Exception:
            log.exception("dashboard stopped")
    threading.Thread(target=run, name="dashboard", daemon=True).start()
