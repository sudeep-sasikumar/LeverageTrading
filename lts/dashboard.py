"""Read-only web dashboard (aiohttp), served from a background thread.

It opens the SQLite file read-only, so it works during the first-start bootstrap as well as
while the service runs, and it can never change trading state.

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
                allg = q("SELECT symbol, ts, side, model_p, threshold, decision, outcomes FROM grind_live "
                         "WHERE ts>=? ORDER BY ts DESC LIMIT 2500", (now - 3 * DAY,))
                for r in allg:
                    o = json.loads(r.pop("outcomes") or "null") or {}
                    r["plans"] = o.get("plans")
                    r["best24"] = (o.get("best") or {}).get("24")
                    r["worst24"] = (o.get("worst") or {}).get("24")
                out["grind_all"] = allg
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

    app = web.Application(middlewares=[auth])
    app.add_routes([web.get("/", index), web.post("/login", login), web.get("/api/data", data),
                    web.get("/healthz", healthz)])
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
