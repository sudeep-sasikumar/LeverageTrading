import pathlib

ROOT = pathlib.Path(r"D:\Leverage trading")


def sub(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:80])
        s = s.replace(a, b, 1)
    p.write_text(s, encoding="utf-8")


# ---- signal engine ------------------------------------------------------------------------------
sub("lts/signals.py", [
    ("from .labels import costs, plan_grind, plan_trade", "from .labels import costs, plan_from_stop, plan_grind, plan_trade"),
    ('''        if cand.get("kind") == "grind":
            # separate research track: own trade plan, no model yet (the model is trained on momentum candidates)''',
     '''        if cand.get("kind") == "grindlab":
            # grind-lab paper trade: the study's own plan (-3% stop, +6% target, 24h), ranked by the checklist
            gl = self.cfg.grindlab
            cand["plan"] = plan_from_stop(self.cfg.trade, gl.stop, gl.target / gl.stop, self.cfg.risk.risk_per_trade,
                                          gl.horizon_hours * 12)
            cand["rule_score"] = float(cand["score_n"])
            cand["score"] = 100.0 * cand.get("catalyst", 0) + cand["score_n"] + (cand.get("model_p") or 0.0)
            cand["p"] = cand["ev_r"] = None
            return
        if cand.get("kind") == "grind":
            # separate research track: own trade plan, no model yet (the model is trained on momentum candidates)'''),
    ('''        if cand.get("kind") == "grind":
            g = self.cfg.grind
            return ("G" if g.live else "P") if cand["score"] >= g.min_rank else None''',
     '''        if cand.get("kind") == "grindlab":
            ok = cand.get("catalyst") or cand["score_n"] >= self.cfg.grindlab.min_score
            return ("G" if self.cfg.grind.live else "P") if ok else None
        if cand.get("kind") == "grind":
            g = self.cfg.grind
            return ("G" if g.live else "P") if cand["score"] >= g.min_rank else None'''),
    ('''        order.sort(key=lambda c: c.get("kind") == "grind")       # momentum first; stable within each track''',
     '''        order.sort(key=lambda c: c.get("kind") in ("grind", "grindlab"))   # momentum first; stable within a track'''),
    ('''            state = RiskState(cfg, self.db, now_ms, "grind" if c.get("kind") == "grind" else "momentum")''',
     '''            state = RiskState(cfg, self.db, now_ms, "grind" if c.get("kind") in ("grind", "grindlab") else "momentum")'''),
    ('''            elif c["plan"] is None:
                reason = "too_volatile"''',
     '''            elif c["plan"] is None:
                reason = "too_volatile"
            elif c.get("kind") == "grind" and not cfg.grind.signals_from_5m:
                reason = "logged_only"                    # grind paper trades come from the hourly grind lab'''),
    ('''        kind = "grind" if c.get("kind") == "grind" else "momentum"
        version = "grind-rules" if kind == "grind" else (bundle["version"] if bundle else "rules")''',
     '''        kind = "grind" if c.get("kind") in ("grind", "grindlab") else "momentum"
        version = {"grind": "grind-rules", "grindlab": "grind-lab"}.get(c.get("kind")) or \\
            (bundle["version"] if bundle else "rules")'''),
    ('''        grind = c.get("kind") == "grind"
        title = f"{icon} <b>{word} {esc(sig['symbol'])}</b> · {grade_note.get(sig['grade'], sig['grade'])}"
        lines = [title + (" · steady-trend (grind) setup" if grind else "")]
        if grind and sig["grade"] == "P":''',
     '''        grind = c.get("kind") in ("grind", "grindlab")
        title = f"{icon} <b>{word} {esc(sig['symbol'])}</b> · {grade_note.get(sig['grade'], sig['grade'])}"
        lines = [title + (" · steady-trend (grind) setup" if grind else "")]
        if c.get("kind") == "grindlab" and sig["grade"] == "P":
            lines.append("📝 <b>Grind lab paper trade — testing the traits of successful grinders forward. "
                         "No proven edge yet. Track it, do not trade it.</b>")
        elif grind and sig["grade"] == "P":'''),
    ('''        if grind:
            lines.append(f"Grind: 6h {f.get('r_6h', 0) * 100:+.1f}% · straightness R² {f.get('g_r2_6h', 0):.2f} · "''',
     '''        if c.get("kind") == "grindlab":
            ck = c.get("checks") or {}
            if c.get("catalyst"):
                lines.append("⚡ News catalyst: down-grind in the week after an Upbit listing / Binance futures launch "
                             "(the one directional pattern the study found)")
            lines.append(f"Successful-grinder traits: {sum(ck.values())}/{len(ck)}"
                         + (f" · model {c['model_p']:.0%}" if c.get("model_p") is not None else ""))
            lines += [f"  {'✅' if v else '▫️'} {esc(k)}" for k, v in ck.items()]
            lines.append(f"6h {f.get('r_6h', 0) * 100:+.1f}% · 24h {f.get('r_24h', 0) * 100:+.1f}% · "
                         f"hold up to {self.cfg.grindlab.horizon_hours}h")
            lines.append(f"#S{sig['id']}")
            return "\\n".join(lines)
        if grind:
            lines.append(f"Grind: 6h {f.get('r_6h', 0) * 100:+.1f}% · straightness R² {f.get('g_r2_6h', 0):.2f} · "'''),
])

# ---- service: hourly grind lab + nightly study ---------------------------------------------------
sub("lts/service.py", [
    ('''    async def daily_report(self) -> None:''',
     '''    async def grind_loop(self) -> None:
        """Hourly (3 minutes after the hour): latest 1h bars -> early-grind detection -> paper trades."""
        if not self.cfg.grindlab.enabled:
            return
        while True:
            now = time.time()
            await asyncio.sleep(max(1.0, (math.floor(now / 3600) + 1) * 3600 + 180 - now))
            try:
                await self.grind_hour()
            except Exception:
                log.exception("grind lab hourly failed")

    async def grind_hour(self) -> list[dict]:
        from . import grindlab

        def work():
            own = DB(self.cfg.path(self.cfg.storage.db_path))
            try:
                return grindlab.hourly(self.cfg, own)
            finally:
                own.close()
        rows = await asyncio.to_thread(work)
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
        worth = [c for c in cands if c["catalyst"] or c["score_n"] >= self.cfg.grindlab.min_score]
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
        for c in cands:
            self.db.execute("UPDATE grind_live SET decision=? WHERE symbol=? AND ts=? AND side=?",
                            (c.get("decision") or "skipped:below_score", c["symbol"], c["ts"], c["side"]))
        log.info("grind lab hour: %d early grinds, %d with enough traits, %d sent", len(cands), len(worth),
                 sum(1 for c in worth if c.get("decision") == "sent"))
        return cands

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

    async def daily_report(self) -> None:'''),
    ('''                if last_report_day != day and (now.hour, now.minute) >= (tc.report_utc_hour, tc.report_utc_minute):''',
     '''                gl = self.cfg.grindlab
                if gl.enabled and self.db.get("last_grindlab_day") != day and \\
                        (now.hour, now.minute) >= (gl.nightly_utc_hour, gl.nightly_utc_minute):
                    self.db.set("last_grindlab_day", day)
                    await self.grind_nightly()
                if last_report_day != day and (now.hour, now.minute) >= (tc.report_utc_hour, tc.report_utc_minute):'''),
    ('''                 (self.bar_loop(), self.track_loop(), self.slow_loop(), self.tg.poll_commands(self.command))]''',
     '''                 (self.bar_loop(), self.track_loop(), self.slow_loop(), self.grind_loop(),
                  self.tg.poll_commands(self.command))]'''),
])

# ---- first start: grind lab history + first study -----------------------------------------------
sub("lts/bootstrap.py", [
    ('''    if not db.query("SELECT 1 FROM models LIMIT 1"):
        db.set("phase", "train")
        log.info("\\n%s", train_and_maybe_promote(cfg, db)["summary"])''',
     '''    if not db.query("SELECT 1 FROM models LIMIT 1"):
        db.set("phase", "train")
        log.info("\\n%s", train_and_maybe_promote(cfg, db)["summary"])
    if cfg.grindlab.enabled and not db.get("bootstrap_grindlab_done"):
        from .grindlab import nightly
        db.set("phase", "grindlab")
        log.info("grind lab: downloading ~4 months of hourly/daily history and running the first study (~20 min)")
        rep = nightly(cfg, db)
        db.set("last_grindlab_day", time.strftime("%Y-%m-%d", time.gmtime()))
        db.set("bootstrap_grindlab_done", True)
        log.info("grind lab: %s early-grind moments studied", rep.get("n_decisions"))'''),
])
print("signals/service/bootstrap wired")
