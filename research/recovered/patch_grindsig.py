import pathlib

ROOT = pathlib.Path(r"D:\Leverage trading")


def sub(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:80])
        s = s.replace(a, b, 1)
    p.write_text(s, encoding="utf-8")


sub("lts/config.py", [
    ('''@dataclass
class Aim:''',
     '''@dataclass
class GrindSignal:
    """Every early grind scored by the pattern-recognition model; ALL above min_p are sent (no daily cap)."""
    enabled: bool = True
    target: float = 0.01                 # +1% in the coin (= ~+10% on margin at 10x)
    stop: float = 0.01
    hours: int = 24
    min_p: float = 0.60                  # model probability needed to send a signal
    max_leverage: int = 10
    min_days_for_own_model: float = 60.0 # retrain on this server's data once it holds this much 5m history
    live: bool = False                   # signals stay PAPER unless their band qualifies AND this is switched on


@dataclass
class Aim:'''),
    ('''    enabled: bool = True
    ranker: str = "checklist"            # checklist / smooth / first''',
     '''    enabled: bool = False                # superseded by grindsignal (all signals, no daily cap)
    ranker: str = "checklist"            # checklist / smooth / first'''),
    ('''    aim: Aim = field(default_factory=Aim)''',
     '''    aim: Aim = field(default_factory=Aim)
    grindsignal: GrindSignal = field(default_factory=GrindSignal)'''),
    ('''    min_score: int = 8                   # checklist traits (of 12) needed for a paper trade''',
     '''    min_score: int = 8                   # checklist traits (of 12) needed for a paper trade
    paper_trades: bool = False           # superseded by grindsignal (the scored signals)'''),
])

sub("lts/signals.py", [
    ('''        if cand.get("kind") == "daily":
            a = self.cfg.aim''',
     '''        if cand.get("kind") == "grindsig":
            g = self.cfg.grindsignal
            tc = self.cfg.trade
            cost = tc.taker_fee + tc.slippage + tc.maker_fee
            lev = int(min(g.max_leverage, max(1, math.ceil(0.10 / max(g.target - cost, 1e-6)))))
            cand["plan"] = plan_from_stop(tc, g.stop, g.target / g.stop, self.cfg.risk.risk_per_trade, g.hours * 12)
            cand["plan"]["leverage"] = lev
            cand["plan"]["gain_margin"] = lev * (g.target - cost)
            cand["plan"]["loss_margin"] = lev * (g.stop + 2 * (tc.taker_fee + tc.slippage))
            cand["plan"]["margin_frac"] = min(1.0, self.cfg.risk.risk_per_trade / cand["plan"]["loss_margin"])
            cand["rule_score"] = cand["score"] = float(cand["p_model"])
            cand["p"], cand["ev_r"] = cand["p_model"], None
            return
        if cand.get("kind") == "daily":
            a = self.cfg.aim'''),
    ('''        if cand.get("kind") == "daily":
            return ("D" if self.cfg.aim.live else "P") if cand["score"] >= self.cfg.aim.threshold else None''',
     '''        if cand.get("kind") == "grindsig":
            from .grindsignal import qualifies
            if cand["score"] < self.cfg.grindsignal.min_p:
                return None
            return "G" if (self.cfg.grindsignal.live and qualifies(cand.get("band"))) else "P"
        if cand.get("kind") == "daily":
            return ("D" if self.cfg.aim.live else "P") if cand["score"] >= self.cfg.aim.threshold else None'''),
    ('''            state = RiskState(cfg, self.db, now_ms, {"grind": "grind", "grindlab": "grind", "daily": "daily"}.get(
                c.get("kind"), "momentum"))''',
     '''            state = RiskState(cfg, self.db, now_ms, {"grind": "grind", "grindlab": "grind", "daily": "daily",
                                                     "grindsig": "grindsig"}.get(c.get("kind"), "momentum"))'''),
    ('''        kind = {"grind": "grind", "grindlab": "grind", "daily": "daily"}.get(c.get("kind"), "momentum")
        version = {"grind": "grind-rules", "grindlab": "grind-lab", "daily": "daily-aim"}.get(c.get("kind")) or \\''',
     '''        kind = {"grind": "grind", "grindlab": "grind", "daily": "daily", "grindsig": "grindsig"}.get(c.get("kind"), "momentum")
        version = {"grind": "grind-rules", "grindlab": "grind-lab", "daily": "daily-aim",
                   "grindsig": "grind-model"}.get(c.get("kind")) or \\'''),
    ('''        if self.kind == "daily":
            if self.open:''',
     '''        if self.kind == "grindsig":
            pass                                          # no daily cap: every qualifying signal is sent
        elif self.kind == "daily":
            if self.open:'''),
    ('''        if c.get("kind") == "daily":
            return self._daily_message(sig, c, plan, prec, icon, word)''',
     '''        if c.get("kind") == "daily":
            return self._daily_message(sig, c, plan, prec, icon, word)
        if c.get("kind") == "grindsig":
            return self._grindsig_message(sig, c, plan, prec, icon, word)'''),
    ('''    def _daily_message(self, sig, c, plan, prec, icon, word) -> str:''',
     '''    def _grindsig_message(self, sig, c, plan, prec, icon, word) -> str:
        g = self.cfg.grindsignal
        b = c.get("band") or {}
        lines = [f"{icon} <b>GRIND {word} {esc(sig['symbol'])}</b> · model {c['p_model']:.0%}"
                 + (" · PAPER" if sig["grade"] == "P" else "")]
        if sig["grade"] == "P":
            lines.append("📝 <i>Unproven: signals in this probability band have not shown an edge after fees on unseen "
                         "months. Your call.</i>")
        lines += ["",
                  f"Entry  <code>{fmt_price(sig['entry'], prec)}</code>  (market now)",
                  f"Target <code>{fmt_price(sig['tp'], prec)}</code>  (+{g.target:.1%})   "
                  f"Stop <code>{fmt_price(sig['sl'], prec)}</code>  (−{g.stop:.1%})   max {g.hours}h",
                  f"{plan['leverage']}x → target ≈ <b>+{plan['gain_margin']:.1%}</b> on margin, stop ≈ −{plan['loss_margin']:.1%} "
                  f"(5x: +{plan['gain_margin'] / plan['leverage'] * 5:.1%} / −{plan['loss_margin'] / plan['leverage'] * 5:.1%})"]
        if b:
            lines.append(f"History of signals scoring ≥{b['p_min']:.0%} (unseen months): won {b['win']:.0%}, "
                         f"{b['net'] * 100:+.2f}% per trade after fees, ~{b['per_day']:.1f}/day")
        f = c.get("features", {})
        lines.append(f"6h {f.get('r_6h', 0) * 100:+.1f}% · 24h {f.get('r_24h', 0) * 100:+.1f}% · "
                     f"steadiness R² {f.get('sh_r2_6h', float('nan')):.2f}")
        lines.append(f"#S{sig['id']}")
        return "\\n".join(lines)

    def _daily_message(self, sig, c, plan, prec, icon, word) -> str:'''),
])

sub("lts/service.py", [
    ('''        worth = [c for c in cands if c["catalyst"] or c["score_n"] >= self.cfg.grindlab.min_score]''',
     '''        worth = [c for c in cands if self.cfg.grindlab.paper_trades and
                 (c["catalyst"] or c["score_n"] >= self.cfg.grindlab.min_score)]'''),
    ('''        await self.daily_pick(rows, meta)''',
     '''        await self.daily_pick(rows, meta)
        await self.grind_signals(rows, meta)'''),
    ('''    async def daily_pick(self, rows: list[dict], meta: dict) -> None:''',
     '''    async def grind_signals(self, rows: list[dict], meta: dict) -> None:
        """Score every early grind with the pattern-recognition model; send ALL above min_p."""
        from . import grindsignal as GS
        g = self.cfg.grindsignal
        if not g.enabled or not rows:
            return
        if not getattr(self, "gs_bundle", None) or time.time() - getattr(self, "gs_loaded", 0) > 6 * 3600:
            self.gs_bundle, self.gs_loaded = GS.load_bundle(self.cfg), time.time()
        if not self.gs_bundle:
            return
        pres = [GS.pre_from_db(self.db, r["symbol"], r["ts"]) for r in rows]
        GS.score_rows(rows, pres, self.gs_bundle)
        picks = []
        for r in rows:
            if r["p"] < g.min_p:
                continue
            m = meta.get(r["symbol"], {})
            picks.append({"symbol": r["symbol"], "side": r["side"], "ts": r["ts"], "ref_price": r["price"],
                          "atr_pct": r.get("atr_1h"), "kind": "grindsig", "p_model": r["p"], "band": r["band"],
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
        log.info("grind signals: %d early grinds scored, %d above %.0f%%, %d sent", len(rows), len(picks), g.min_p * 100,
                 sum(1 for p in picks if p.get("decision") == "sent"))

    async def daily_pick(self, rows: list[dict], meta: dict) -> None:'''),
    ('''        rep = await asyncio.to_thread(work)
        await self.tg.send(grindlab.nightly_text(rep))''',
     '''        rep = await asyncio.to_thread(work)
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
                top = (b.get("bands") or [{}])[-1]
                await self.tg.send(f"<b>Grind model retrained</b> on {b['n']:,} grinds from this server's data. "
                                   f"Top band on unseen months: won {top.get('win', 0):.0%}, "
                                   f"{top.get('net', 0) * 100:+.2f}% per trade.")
        except Exception:
            log.exception("grind model retraining failed")'''),
])
print("grind signals wired")
