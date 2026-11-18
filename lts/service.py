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
from .candidates import grind_mask, kind_of, trigger_mask
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
    gc = cfg.grind
    recent = {(r["symbol"], r["side"]) for r in
              db.query("SELECT symbol, side FROM candidates WHERE source='live' AND COALESCE(kind,'')!='grind' "
                       "AND ts>=? AND ts<?", (close_ts - cfg.trigger.cooldown_bars * BAR_MS, close_ts))}
    recent_grind = {(r["symbol"], r["side"]) for r in
                    db.query("SELECT symbol, side FROM candidates WHERE source='live' AND kind='grind' "
                             "AND ts>=? AND ts<?", (close_ts - gc.cooldown_bars * BAR_MS, close_ts))}
    out = []
    for sym, k in ds.klines.items():
        meta = ds.meta.get(sym, {})
        if not meta.get("active") or k.index[-1] != bar_ts:
            continue
        # cheap necessary conditions before paying for the full feature frame
        if len(k) < cfg.trigger.min_history_bars or k["qv"].iloc[-288:].sum() < cfg.universe.min_qvol_24h:
            continue
        p = panels[sym].iloc[-1]          # same definitions as the trigger's momentum test
        momentum_possible = abs(p["z_1h"]) >= trig.z_1h or abs(p["z_4h"]) >= trig.z_4h or abs(p["r_24h"]) >= trig.r_24h
        grind_possible = gc.enabled and abs(p["r_6h"]) >= gc.min_r6
        if not (momentum_possible or grind_possible):
            continue
        f = raw_frame(ds, sym, market, ranks)
        last = f.iloc[[-1]]
        base = {"ts": close_ts, "ref_price": float(k["c"].iloc[-1]), "on_weex": bool(meta.get("on_weex")),
                "src": meta.get("src"), "price_precision": meta.get("price_precision"), "symbol": sym}
        for side in (1, -1):
            if (sym, side) not in recent and trigger_mask(last, cfg, side)[0]:
                out.append(base | {"side": side, "kind": None, "atr_pct": float(last["atr_pct"].iloc[0]),
                                   "aligned": F.align(last, side).iloc[0]})
            if (sym, side) not in recent_grind and grind_mask(last, cfg, side)[0]:
                out.append(base | {"side": side, "kind": "grind", "atr_pct": float(last["atr_1h"].iloc[0]),
                                   "aligned": F.align(last, side).iloc[0]})
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
            (c["symbol"], c["ts"], c["side"], c.get("kind") or kind_of(c["features"]), c["ref_price"], c["atr_pct"],
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
        from .microstructure import LiquidationStream
        self.liq = LiquidationStream(db)
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
                t_scan = time.time()
                cands = await scan_cycle(self.cfg, self.db, self.collector, self.engine, self.bundle, bar_ts,
                                         cache=self.cache)
                self.last_bar = bar_ts
                # read by the dashboard to show that the scanner is alive
                self.db.set("heartbeat", {
                    "ts": int(time.time() * 1000), "bar_ts": bar_ts, "candidates": len(cands),
                    "signals": sum(1 for c in cands if c.get("decision") == "sent"),
                    "errors": self.collector.last_errors, "collect_s": round(self.collector.last_seconds, 1),
                    "scan_s": round(time.time() - t_scan, 1), "telegram": self.tg.enabled})
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
                    from .grindsignal import label_grinds, label_signal_plans
                    label_signal_plans(self.cfg, self.db)
                    label_grinds(self.cfg, self.db)
                    if now.hour % 6 == 0:
                        await refresh_universe(self.cfg, self.db, self.binance, self.weex)
                day = now.strftime("%Y-%m-%d")
                lc, tc = self.cfg.learn, self.cfg.telegram
                if last_train_day != day and (now.hour, now.minute) >= (lc.retrain_utc_hour, lc.retrain_utc_minute):
                    last_train_day = day
                    self.db.set("last_train_day", day)
                    await self.retrain()
                    self.db.prune(self.cfg.storage.keep_klines_days)
                    if self.cfg.micro.enabled:
                        from .microstructure import prune as micro_prune
                        micro_prune(self.db, self.cfg.micro.keep_days)
                gl = self.cfg.grindlab
                if gl.enabled and self.db.get("last_grindlab_day") != day and \
                        (now.hour, now.minute) >= (gl.nightly_utc_hour, gl.nightly_utc_minute):
                    self.db.set("last_grindlab_day", day)
                    await self.grind_nightly()
                if last_report_day != day and (now.hour, now.minute) >= (tc.report_utc_hour, tc.report_utc_minute):
                    last_report_day = day
                    self.db.set("last_report_day", day)
                    await self.daily_report()
            except Exception:
                log.exception("slow loop failed")

    async def book_loop(self) -> None:
        """Order-book depth snapshots, 40s after every N-minute boundary."""
        from .microstructure import book_symbols, snapshot_books
        mc = self.cfg.micro
        step = mc.book_every_minutes * 60
        while True:
            now = time.time()
            await asyncio.sleep(max(1.0, (math.floor(now / step) + 1) * step + 40 - now))
            try:
                ts = int(time.time() // step * step * 1000)
                n = await snapshot_books(self.db, self.binance, book_symbols(self.db, mc.book_symbols), ts)
                log.debug("order-book snapshots: %d", n)
            except Exception:
                log.exception("order-book snapshot failed")

    async def grind_loop(self) -> None:
        """Every 5 minutes (about a minute after each 5-minute candle closes): early-grind detection -> signals."""
        if not self.cfg.grindlab.enabled:
            return
        from .grindlab import FiveMinuteScanner
        self.grind_scanner = FiveMinuteScanner(self.cfg)
        while True:
            now = time.time()
            await asyncio.sleep(max(1.0, (math.floor(now / 300) + 1) * 300 + self.cfg.grindsignal.scan_delay_s - now))
            try:
                await self.grind_hour()
            except Exception:
                log.exception("grind scan failed")

    async def grind_hour(self, end: int | None = None) -> list[dict]:
        """One grind scan as of the latest 5-minute boundary (name kept from the hourly version)."""
        from . import grindlab
        end = end or int(time.time() // 300 * 300 * 1000)

        def work():
            own = DB(self.cfg.path(self.cfg.storage.db_path))
            try:
                return self.grind_scanner.scan(own, end)
            finally:
                own.close()
        if not getattr(self, "grind_scanner", None):
            from .grindlab import FiveMinuteScanner
            self.grind_scanner = FiveMinuteScanner(self.cfg)
        rows = await asyncio.to_thread(work)
        rows = self.grinds_to_record(rows, end)
        meta = {r["symbol"]: dict(r) for r in self.db.query("SELECT symbol, on_weex, price_precision FROM symbols")}
        cands = []
        for r in rows:
            m = meta.get(r["symbol"], {})
            feats = {k: r.get(k) for k in grindlab.SHOW_FACTORS if r.get(k) is not None}
            cands.append({"symbol": r["symbol"], "side": r["side"], "ts": r["ts"], "ref_price": r["price"],
                          "atr_pct": r.get("atr_1h"), "kind": "grindlab", "score_n": r["score"],
                          "model_p": r.get("model_p"), "checks": r.get("checks"), "catalyst": r.get("catalyst", 0),
                          "on_weex": bool(m.get("on_weex")), "src": "binance",
                          "price_precision": m.get("price_precision"), "features": feats})
        worth = [c for c in cands if self.cfg.grindlab.paper_trades and
                 (c["catalyst"] or c["score_n"] >= self.cfg.grindlab.min_score)]
        for c in worth:                                  # Weex book only for the ones that may be sent
            c["book"] = None
            if c["on_weex"]:
                try:
                    c["book"] = await self.weex.book(c["symbol"])
                except Exception:
                    pass
            wt = self.collector.weex_tickers.get(c["symbol"]) or {}
            c["weex_qv24"] = wt.get("qv24")
            if c["book"] and c["ref_price"]:
                c["features"]["weex_dev_bps"] = (c["book"]["mid"] / c["ref_price"] - 1) * 1e4 * c["side"]
        await self.engine.decide(worth, None, int(time.time() * 1000))
        await self.daily_pick(rows, meta)
        if self.cfg.grindlab.paper_trades:
            for c in cands:
                self.db.execute("UPDATE grind_live SET decision=? WHERE symbol=? AND ts=? AND side=?",
                                (c.get("decision") or "skipped:below_score", c["symbol"], c["ts"], c["side"]))
        await self.grind_signals(rows, meta)

        def progress():
            from .grindsignal import progress_grinds
            own = DB(self.cfg.path(self.cfg.storage.db_path))
            try:
                return progress_grinds(self.cfg, own)
            finally:
                own.close()
        try:
            await asyncio.to_thread(progress)
        except Exception:
            log.exception("grind progress update failed")
        log.info("grind scan: %d new early grinds, %d with enough traits, %d sent", len(cands), len(worth),
                 sum(1 for c in worth if c.get("decision") == "sent"))
        return cands

    def grinds_to_record(self, rows: list[dict], end: int) -> list[dict]:
        """Score every detected grind, then keep (record, and consider for sending) only:
        - a grind with no record for that coin and side in the last 55 minutes (about one record an hour), or
        - a grind whose score has just crossed the sending threshold (it was below at its last record)."""
        from . import grindsignal as GS
        from .grindlab import _insert_live
        g = self.cfg.grindsignal
        if not rows:
            return rows
        if not getattr(self, "gs_bundle", None) or time.time() - getattr(self, "gs_loaded", 0) > 6 * 3600:
            self.gs_bundle, self.gs_loaded = GS.load_bundle(self.cfg), time.time()
        if g.enabled and self.gs_bundle:
            pres = [GS.pre_from_db(self.db, r["symbol"], r["ts"]) for r in rows]
            GS.score_rows(rows, pres, self.gs_bundle)
            self.gs_pmin = GS.threshold(self.gs_bundle, g.top_share)
        keep = []
        for r in rows:
            last = self.db.query("SELECT model_p, threshold FROM grind_live WHERE symbol=? AND side=? AND ts>? "
                                 "ORDER BY ts DESC LIMIT 1", (r["symbol"], r["side"], end - 55 * 60_000))
            p, p_min = r.get("p"), getattr(self, "gs_pmin", None)
            crossed = (p is not None and p_min is not None and p >= p_min and last
                       and (last[0]["model_p"] or 0) < (last[0]["threshold"] or 1))
            if not last or crossed:
                keep.append(r)
        _insert_live(self.db, keep)
        return keep

    async def grind_signals(self, rows: list[dict], meta: dict) -> None:
        """Send ALL recorded grinds whose model score is at or above the top-share threshold."""
        from . import grindsignal as GS
        g = self.cfg.grindsignal
        if not g.enabled or not rows or not getattr(self, "gs_bundle", None) or "p" not in rows[0]:
            return
        p_min = self.gs_pmin
        GS.record_grinds(self.db, rows, p_min)
        picks = []
        for r in rows:
            if r["p"] < p_min:
                continue
            m = meta.get(r["symbol"], {})
            picks.append({"symbol": r["symbol"], "side": r["side"], "ts": r["ts"], "ref_price": r["price"],
                          "atr_pct": r.get("atr_1h"), "kind": "grindsig", "p_model": r["p"], "band": r["band"],
                          "p_min": p_min,
                          "on_weex": bool(m.get("on_weex")), "src": "binance", "price_precision": m.get("price_precision"),
                          "features": {k: r.get(k) for k in ("r_6h", "r_24h", "sh_r2_6h") if r.get(k) is not None}})
        for p in picks:
            p["book"] = await self.weex.book(p["symbol"]) if p["on_weex"] else None
            p["weex_qv24"] = (self.collector.weex_tickers.get(p["symbol"]) or {}).get("qv24")
            if p["book"] and p["ref_price"]:
                p["features"]["weex_dev_bps"] = (p["book"]["mid"] / p["ref_price"] - 1) * 1e4 * p["side"]
        for r in rows:
            self.db.execute("UPDATE grind_live SET model_p=? WHERE symbol=? AND ts=? AND side=?",
                            (r["p"], r["symbol"], r["ts"], r["side"]))
        if picks:
            await self.engine.decide(picks, None, int(time.time() * 1000))
        decided = {(p["symbol"], p["side"]): p.get("decision") for p in picks}
        for r in rows:                                   # what happened to each grind, for the Grinders tab
            dec = decided.get((r["symbol"], r["side"])) or f"below model threshold ({r['p']:.3f} < {p_min:.3f})"
            self.db.execute("UPDATE grind_live SET decision=? WHERE symbol=? AND ts=? AND side=?",
                            (dec, r["symbol"], r["ts"], r["side"]))
        log.info("grind signals: %d early grinds scored, %d in the top %.0f%%, %d sent", len(rows), len(picks), g.top_share * 100,
                 sum(1 for p in picks if p.get("decision") == "sent"))

    async def daily_pick(self, rows: list[dict], meta: dict) -> None:
        """The project's aim: the first early grind of the day that clears the daily-trade threshold."""
        from .aim import live_score
        a = self.cfg.aim
        if not a.enabled or not rows:
            return
        picks = []
        for r in rows:
            m = meta.get(r["symbol"], {})
            picks.append({"symbol": r["symbol"], "side": r["side"], "ts": r["ts"], "ref_price": r["price"],
                          "atr_pct": r.get("atr_1h"), "kind": "daily", "aim_score": live_score(r, a.ranker),
                          "on_weex": bool(m.get("on_weex")), "src": "binance", "price_precision": m.get("price_precision"),
                          "features": {k: r.get(k) for k in ("r_6h", "r_24h") if r.get(k) is not None}})
        picks = [p for p in picks if p["aim_score"] >= a.threshold]
        for p in picks:
            p["book"] = await self.weex.book(p["symbol"]) if p["on_weex"] else None
            p["weex_qv24"] = (self.collector.weex_tickers.get(p["symbol"]) or {}).get("qv24")
            if p["book"] and p["ref_price"]:
                p["features"]["weex_dev_bps"] = (p["book"]["mid"] / p["ref_price"] - 1) * 1e4 * p["side"]
        if picks:
            await self.engine.decide(picks, None, int(time.time() * 1000))
            log.info("daily trade: %d candidates cleared the threshold, decisions %s", len(picks),
                     [p.get("decision") for p in picks])

    async def grind_nightly(self) -> None:
        from . import grindlab

        def work():
            own = DB(self.cfg.path(self.cfg.storage.db_path))
            try:
                return grindlab.nightly(self.cfg, own)
            finally:
                own.close()
        rep = await asyncio.to_thread(work)
        await self.tg.send(grindlab.nightly_text(rep))

        def retrain_signals():
            from . import grindsignal as GS
            own = DB(self.cfg.path(self.cfg.storage.db_path))
            try:
                return GS.nightly(self.cfg, own)
            finally:
                own.close()
        try:
            b = await asyncio.to_thread(retrain_signals)
            if b:
                self.gs_bundle = None                     # reload on the next hour
                from .grindsignal import plan_name
                sent = next((x for x in b.get("bands", []) if abs(x["top"] - self.cfg.grindsignal.top_share) < 1e-9), None)
                lines = [f"<b>Grind model retrained</b> on {b['n']:,} grinds from this server's data."]
                for h in (sent or {}).get("plans", []):
                    lines.append(f"  {plan_name(h)}: won {h['win']:.0%}, {h['net'] * 100:+.2f}% per trade (unseen months)")
                await self.tg.send("\n".join(lines))
        except Exception:
            log.exception("grind model retraining failed")

    async def daily_report(self) -> None:
        from .movers import review, review_text

        def work():
            own = DB(self.cfg.path(self.cfg.storage.db_path))     # sqlite connections are per-thread
            try:
                return review(self.cfg, own)
            finally:
                own.close()
        text = daily_report_text(self.cfg, self.db)
        try:
            res = await asyncio.to_thread(work)
            text += "\n\n" + review_text(res, self.cfg.telegram.local_utc_offset_minutes, self.cfg.telegram.local_tz_label)
        except Exception:
            log.exception("daily grinder review failed")
        await self.tg.send(text)

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
        self.db.set("phase", "running")
        if self.bundle and self.bundle.get("status") == "champion":
            mode = f"validated model {self.bundle['version']} - live signals"
        elif self.bundle:
            mode = f"model {self.bundle['version']} not yet validated - PAPER signals only"
        else:
            mode = "no model yet - PAPER signals only"
        await self.tg.send(f"🤖 LTS started: {esc(mode)}. Scanning every 5 minutes.")
        tasks = [asyncio.create_task(t) for t in
                 (self.bar_loop(), self.track_loop(), self.slow_loop(), self.grind_loop(),
                  self.tg.poll_commands(self.command))]
        if self.cfg.micro.enabled:
            tasks += [asyncio.create_task(self.liq.run()), asyncio.create_task(self.book_loop())]
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
