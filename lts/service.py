"""The live service: collect -> scan -> score -> signal, plus tracking, labelling and learning."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone

from . import BAR_MS
from . import features as F
from .candidates import kind_of, trigger_mask
from .collector import Collector, last_closed_bar
from .config import Config
from .db import DB
from .exchanges import Binance, Weex
from .learner import load_bundle, train_and_maybe_promote
from .pipeline import DatasetCache, build_context, features_to_json, load_dataset, raw_frame
from .reports import daily_report_text, performance_text, status_text
from .signals import SignalEngine
from .telegram import Telegram, esc
from .tracker import label_matured, track_open_signals
from .universe import refresh_universe

log = logging.getLogger("lts.service")


# ---- scan -------------------------------------------------------------------
def scan(cfg: Config, db: DB, bar_ts: int, cache: DatasetCache | None = None) -> list[dict]:
    """Detect candidates on the bar that opened at bar_ts (already closed and stored)."""
    since = bar_ts - cfg.storage.feature_window_bars * BAR_MS
    ds = cache.get(bar_ts) if cache else load_dataset(db, since, until_ts=bar_ts)
    if "BTCUSDT" not in ds.klines:
        log.warning("no BTC data - scan skipped")
        return []
    market, ranks, panels = build_context(ds)
    trig = cfg.trigger
    close_ts = bar_ts + BAR_MS
    cool_from = close_ts - cfg.trigger.cooldown_bars * BAR_MS
    recent = {(r["symbol"], r["side"]) for r in
              db.query("SELECT symbol, side FROM candidates WHERE source='live' AND ts>=? AND ts<?",
                       (cool_from, close_ts))}
    out = []
    for sym, k in ds.klines.items():
        meta = ds.meta.get(sym, {})
        if not meta.get("active") or k.index[-1] != bar_ts:
            continue
        # cheap necessary conditions before paying for the full feature frame
        if len(k) < cfg.trigger.min_history_bars or k["qv"].iloc[-288:].sum() < cfg.universe.min_qvol_24h:
            continue
        p = panels[sym].iloc[-1]          # same definitions as the trigger's momentum test
        if not (abs(p["z_1h"]) >= trig.z_1h or abs(p["z_4h"]) >= trig.z_4h or abs(p["r_24h"]) >= trig.r_24h):
            continue
        f = raw_frame(ds, sym, market, ranks)
        last = f.iloc[[-1]]
        for side in (1, -1):
            if (sym, side) in recent or not trigger_mask(last, cfg, side)[0]:
                continue
            aligned = F.align(last, side).iloc[0]
            out.append({"symbol": sym, "side": side, "ts": close_ts, "ref_price": float(k["c"].iloc[-1]),
                        "atr_pct": float(last["atr_pct"].iloc[0]), "aligned": aligned,
                        "on_weex": bool(meta.get("on_weex")), "src": meta.get("src"),
                        "price_precision": meta.get("price_precision")})
    return out


async def enrich(cands: list[dict], weex: Weex, weex_tickers: dict) -> None:
    """Attach Weex execution data and the live-only features."""
    async def one(c: dict):
        c["book"] = None
        if c["on_weex"]:
            try:
                c["book"] = await weex.book(c["symbol"])
            except Exception as e:
                log.debug("weex book %s failed: %s", c["symbol"], e)
        a = c["aligned"].copy()
        wt = weex_tickers.get(c["symbol"]) or {}
        c["weex_qv24"] = wt.get("qv24")
        if wt.get("qv24"):
            a["weex_log_qv"] = math.log10(max(wt["qv24"], 1.0))
        if c["book"]:
            b = c["book"]
            a["weex_spread_bps"] = b["spread_bps"]
            a["book_imb"] = b["imb"] * c["side"]
            a["book_depth_log"] = math.log10(max(b["depth_usd"], 1.0))
            if c["src"] == "binance" and c["ref_price"] > 0:
                a["weex_dev_bps"] = (b["mid"] / c["ref_price"] - 1) * 1e4 * c["side"]
        c["features_json"] = features_to_json(a)
        c["features"] = {k: v for k, v in json.loads(c["features_json"]).items() if v is not None}

    await asyncio.gather(*(one(c) for c in cands))


def store_candidates(db: DB, cands: list[dict], bundle: dict | None) -> None:
    for c in cands:
        cur = db.execute(
            """INSERT OR IGNORE INTO candidates(symbol, ts, side, kind, source, ref_price, atr_pct, sl_pct, features,
                                                rule_score, model_p, model_version, decision)
               VALUES(?,?,?,?,'live',?,?,?,?,?,?,?,?)""",
            (c["symbol"], c["ts"], c["side"], kind_of(c["features"]), c["ref_price"], c["atr_pct"],
             (c.get("plan") or {}).get("sl_pct"), c["features_json"], c.get("rule_score"), c.get("p"),
             bundle["version"] if bundle else None, c.get("decision")))
        c["id"] = cur.lastrowid


async def scan_cycle(cfg: Config, db: DB, collector: Collector, engine: SignalEngine, bundle: dict | None,
                     bar_ts: int, dry_run: bool = False, cache: DatasetCache | None = None) -> list[dict]:
    t0 = time.time()
    cands = scan(cfg, db, bar_ts, cache)
    await enrich(cands, collector.weex, collector.weex_tickers)
    now = int(time.time() * 1000)
    # score first so candidates are stored (with ids) before any signal references them
    for c in cands:
        engine.score(c, bundle)
    store_candidates(db, cands, bundle)
    sent = await engine.decide(cands, bundle, now, dry_run=dry_run)
    for c in cands:
        db.execute("UPDATE candidates SET decision=? WHERE id=?", (c.get("decision"), c["id"]))
    log.info("scan bar %d: %d candidates, %d signals, %.1fs", bar_ts, len(cands), len(sent), time.time() - t0)
    return cands


async def scan_once(cfg: Config, db: DB, collect: bool = True) -> None:
    """CLI helper: one cycle, nothing is sent."""
    binance, weex = Binance(), Weex()
    tg = Telegram(cfg)
    tg.enabled = False
    try:
        collector = Collector(cfg, db, binance, weex)
        if collect:
            if not db.query("SELECT 1 FROM symbols LIMIT 1"):
                await refresh_universe(cfg, db, binance, weex)
            bar_ts = await collector.collect()
        else:
            bar_ts = max(db.last_kline_ts().values())
        bundle = load_bundle(cfg, db)
        cands = await scan_cycle(cfg, db, collector, SignalEngine(cfg, db, weex, tg), bundle, bar_ts, dry_run=True)
        print(f"bar {datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc):%Y-%m-%d %H:%M} UTC: "
              f"{len(cands)} candidates (model: {bundle['version'] if bundle else 'rules only'})")
        for c in sorted(cands, key=lambda c: -(c.get("score") if c.get("score") is not None else -9)):
            p = f"score={c['score']:+.3f} p={c['p']:.2f}" if c.get("p") is not None else "score=n/a"
            print(f"  {c['symbol']:<14} {'LONG ' if c['side'] > 0 else 'SHORT'} {p} rules={c['rule_score']:.0f} "
                  f"atr={c['atr_pct']:.2%} -> {c.get('decision')}")
    finally:
        await binance.close()
        await weex.close()
        await tg.close()


# ---- service ----------------------------------------------------------------
class Service:
    def __init__(self, cfg: Config, db: DB):
        self.cfg, self.db = cfg, db
        self.binance, self.weex = Binance(), Weex()
        self.tg = Telegram(cfg)
        self.collector = Collector(cfg, db, self.binance, self.weex)
        self.engine = SignalEngine(cfg, db, self.weex, self.tg)
        self.bundle = load_bundle(cfg, db)
        self.cache = DatasetCache(db, cfg.storage.feature_window_bars)
        self.started = time.time()
        self.last_bar = 0
        self.alerted = False

    async def bar_loop(self) -> None:
        while True:
            now = time.time()
            nxt = (math.floor(now / 300) + 1) * 300 + 4          # 4s after each 5m boundary
            await asyncio.sleep(max(0.5, nxt - now))
            try:
                bar_ts = await self.collector.collect()
                await scan_cycle(self.cfg, self.db, self.collector, self.engine, self.bundle, bar_ts,
                                 cache=self.cache)
                self.last_bar = bar_ts
                if self.collector.failures >= 3 and not self.alerted:
                    self.alerted = True
                    await self.tg.send("⚠️ Data collection is failing for many symbols. Signals may be delayed.")
                elif self.collector.failures == 0:
                    self.alerted = False
            except Exception as e:
                log.exception("bar cycle failed")
                if not self.alerted:
                    self.alerted = True
                    await self.tg.send(f"⚠️ Scan cycle error: {esc(type(e).__name__)}: {esc(str(e)[:200])}")

    async def track_loop(self) -> None:
        while True:
            await asyncio.sleep(60 - time.time() % 60 + 8)
            try:
                await track_open_signals(self.cfg, self.db, self.binance, self.weex, self.tg)
            except Exception:
                log.exception("tracking failed")

    async def slow_loop(self) -> None:
        last_hour, last_train_day, last_report_day = -1, self.db.get("last_train_day"), self.db.get("last_report_day")
        while True:
            await asyncio.sleep(45)
            now = datetime.now(timezone.utc)
            try:
                if now.hour != last_hour and now.minute >= 8:      # hourly ratios are published by then
                    last_hour = now.hour
                    await self.collector.collect_slow()
                    label_matured(self.cfg, self.db)
                    if now.hour % 6 == 0:
                        await refresh_universe(self.cfg, self.db, self.binance, self.weex)
                day = now.strftime("%Y-%m-%d")
                lc, tc = self.cfg.learn, self.cfg.telegram
                if last_train_day != day and (now.hour, now.minute) >= (lc.retrain_utc_hour, lc.retrain_utc_minute):
                    last_train_day = day
                    self.db.set("last_train_day", day)
                    await self.retrain()
                    self.db.prune(self.cfg.storage.keep_klines_days)
                if last_report_day != day and (now.hour, now.minute) >= (tc.report_utc_hour, tc.report_utc_minute):
                    last_report_day = day
                    self.db.set("last_report_day", day)
                    await self.tg.send(daily_report_text(self.cfg, self.db))
            except Exception:
                log.exception("slow loop failed")

    async def retrain(self) -> None:
        def work():
            own = DB(self.cfg.path(self.cfg.storage.db_path))     # sqlite connections are per-thread
            try:
                return train_and_maybe_promote(self.cfg, own)
            finally:
                own.close()
        res = await asyncio.to_thread(work)
        self.bundle = load_bundle(self.cfg, self.db)      # champion if validated, else the shadow model
        await self.tg.send("<b>Nightly learning</b>\n<pre>" + esc(res["summary"]) + "</pre>")

    async def command(self, text: str) -> str:
        cmd = text.split()[0].split("@")[0].lower()
        if cmd == "/status":
            up = int(time.time() - self.started)
            return f"<pre>{esc(status_text(self.cfg, self.db))}\nUptime: {up // 3600}h {up % 3600 // 60}m</pre>"
        if cmd == "/stats":
            return f"<pre>{esc(performance_text(self.cfg, self.db))}</pre>"
        if cmd == "/open":
            rows = self.db.query("SELECT * FROM signals WHERE status='open'")
            if not rows:
                return "No open signals."
            return "\n".join(f"#S{s['id']} {esc(s['symbol'])} {'LONG' if s['side'] > 0 else 'SHORT'} "
                             f"entry {s['entry']:g} · SL {s['sl']:g} · TP {s['tp']:g}" for s in rows)
        if cmd == "/pause":
            self.db.set("paused", True)
            return "Signals paused. Data collection and learning continue. /resume to re-enable."
        if cmd == "/resume":
            self.db.set("paused", False)
            return "Signals resumed."
        if cmd in ("/help", "/start"):
            return "/status · /stats · /open · /pause · /resume"
        return ""

    async def run(self) -> None:
        await refresh_universe(self.cfg, self.db, self.binance, self.weex)
        if self.bundle and self.bundle.get("status") == "champion":
            mode = f"validated model {self.bundle['version']} - live signals"
        elif self.bundle:
            mode = f"model {self.bundle['version']} not yet validated - PAPER signals only"
        else:
            mode = "no model yet - PAPER signals only"
        await self.tg.send(f"🤖 LTS started: {esc(mode)}. Scanning every 5 minutes.")
        tasks = [asyncio.create_task(t) for t in
                 (self.bar_loop(), self.track_loop(), self.slow_loop(), self.tg.poll_commands(self.command))]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            await self.binance.close()
            await self.weex.close()
            await self.tg.close()


async def run_service(cfg: Config, db: DB) -> None:
    await Service(cfg, db).run()
