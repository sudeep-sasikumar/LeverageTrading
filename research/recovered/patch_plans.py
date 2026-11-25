import pathlib

ROOT = pathlib.Path(r"D:\Leverage trading")


def sub(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:80])
        s = s.replace(a, b, 1)
    p.write_text(s, encoding="utf-8")


sub("lts/signals.py", [
    ('''        if cand.get("kind") == "grindsig":
            g = self.cfg.grindsignal
            tc = self.cfg.trade
            cost = tc.taker_fee + tc.slippage + tc.maker_fee
            lev = int(min(g.max_leverage, max(1, math.ceil(0.10 / max(g.target - cost, 1e-6)))))
            cand["plan"] = plan_from_stop(tc, g.stop, g.target / g.stop, self.cfg.risk.risk_per_trade, g.hours * 12)
            cand["plan"]["leverage"] = lev
            cand["plan"]["gain_margin"] = lev * (g.target - cost)
            cand["plan"]["loss_margin"] = lev * (g.stop + 2 * (tc.taker_fee + tc.slippage))
            cand["plan"]["margin_frac"] = min(1.0, self.cfg.risk.risk_per_trade / cand["plan"]["loss_margin"])''',
     '''        if cand.get("kind") == "grindsig":
            g = self.cfg.grindsignal
            tc = self.cfg.trade
            cost_win = tc.taker_fee + tc.slippage + tc.maker_fee
            cost_loss = 2 * (tc.taker_fee + tc.slippage)
            cand["all_plans"] = [{**pl, "gain_margin": pl["leverage"] * (pl["target"] - cost_win),
                                  "loss_margin": pl["leverage"] * (pl["stop"] + cost_loss)} for pl in g.plans]
            pp = g.plans[g.primary]                                   # the tracked plan
            cand["plan"] = plan_from_stop(tc, pp["stop"], pp["target"] / pp["stop"], self.cfg.risk.risk_per_trade,
                                          g.hours * 12)
            cand["plan"]["leverage"] = pp["leverage"]
            cand["plan"]["gain_margin"] = cand["all_plans"][g.primary]["gain_margin"]
            cand["plan"]["loss_margin"] = cand["all_plans"][g.primary]["loss_margin"]
            cand["plan"]["margin_frac"] = min(1.0, self.cfg.risk.risk_per_trade / cand["plan"]["loss_margin"])'''),
    ('''        if cand.get("kind") == "grindsig":
            from .grindsignal import qualifies
            if cand["score"] < self.cfg.grindsignal.min_p:
                return None
            return "G" if (self.cfg.grindsignal.live and qualifies(cand.get("band"))) else "P"''',
     '''        if cand.get("kind") == "grindsig":
            from .grindsignal import qualifies
            g = self.cfg.grindsignal
            if cand["score"] < cand.get("p_min", 1.0):
                return None
            band_plans = (cand.get("band") or {}).get("plans") or []
            primary = band_plans[g.primary] if len(band_plans) > g.primary else None
            return "G" if (g.live and qualifies(primary)) else "P"'''),
    ('''    def _grindsig_message(self, sig, c, plan, prec, icon, word) -> str:
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
        f = c.get("features", {})''',
     '''    def _grindsig_message(self, sig, c, plan, prec, icon, word) -> str:
        g = self.cfg.grindsignal
        b = c.get("band") or {}
        hist = {(h["leverage"], h["target"], h["stop"]): h for h in b.get("plans", [])}
        top = f"top {b['top']:.0%} of grinds" if b.get("top", 1) < 1 else "all grinds"
        lines = [f"{icon} <b>GRIND {word} {esc(sig['symbol'])}</b> · model score {c['p_model']:.2f} ({top})"
                 + (" · PAPER" if sig["grade"] == "P" else "")]
        if sig["grade"] == "P":
            lines.append("📝 <i>Unproven: no plan has shown an edge after fees on unseen months. Your call.</i>")
        lines += ["", f"Entry <code>{fmt_price(sig['entry'], prec)}</code> (market now) · close after {g.hours}h if no exit", ""]
        s = sig["side"]
        for i, pl in enumerate(c.get("all_plans", [])):
            tp = sig["entry"] * (1 + s * pl["target"])
            sl = sig["entry"] * (1 - s * pl["stop"])
            mark = " ← tracked" if i == g.primary else ""
            lines.append(f"<b>{pl['leverage']}x</b>: target <code>{fmt_price(tp, prec)}</code> (+{pl['target']:.2%}) · "
                         f"stop <code>{fmt_price(sl, prec)}</code> (−{pl['stop']:.2%}) → +{pl['gain_margin']:.1%} / "
                         f"−{pl['loss_margin']:.1%} on margin{mark}")
            h = hist.get((pl["leverage"], pl["target"], pl["stop"]))
            if h:
                lines.append(f"     history ({top}, unseen months): won {h['win']:.0%}, {h['net'] * 100:+.2f}% per trade, "
                             f"{h['margin'] * 100:+.1f}% margin avg")
        f = c.get("features", {})'''),
])

sub("lts/service.py", [
    ('''        picks = []
        for r in rows:
            if r["p"] < g.min_p:
                continue
            m = meta.get(r["symbol"], {})
            picks.append({"symbol": r["symbol"], "side": r["side"], "ts": r["ts"], "ref_price": r["price"],
                          "atr_pct": r.get("atr_1h"), "kind": "grindsig", "p_model": r["p"], "band": r["band"],''',
     '''        p_min = GS.threshold(self.gs_bundle, g.top_share)
        picks = []
        for r in rows:
            if r["p"] < p_min:
                continue
            m = meta.get(r["symbol"], {})
            picks.append({"symbol": r["symbol"], "side": r["side"], "ts": r["ts"], "ref_price": r["price"],
                          "atr_pct": r.get("atr_1h"), "kind": "grindsig", "p_model": r["p"], "band": r["band"],
                          "p_min": p_min,'''),
    ('''        log.info("grind signals: %d early grinds scored, %d above %.0f%%, %d sent", len(rows), len(picks), g.min_p * 100,''',
     '''        log.info("grind signals: %d early grinds scored, %d in the top %.0f%%, %d sent", len(rows), len(picks), g.top_share * 100,'''),
    ('''                top = (b.get("bands") or [{}])[-1]
                await self.tg.send(f"<b>Grind model retrained</b> on {b['n']:,} grinds from this server's data. "
                                   f"Top band on unseen months: won {top.get('win', 0):.0%}, "
                                   f"{top.get('net', 0) * 100:+.2f}% per trade.")''',
     '''                from .grindsignal import plan_name
                sent = next((x for x in b.get("bands", []) if abs(x["top"] - self.cfg.grindsignal.top_share) < 1e-9), None)
                lines = [f"<b>Grind model retrained</b> on {b['n']:,} grinds from this server's data."]
                for h in (sent or {}).get("plans", []):
                    lines.append(f"  {plan_name(h)}: won {h['win']:.0%}, {h['net'] * 100:+.2f}% per trade (unseen months)")
                await self.tg.send("\\n".join(lines))'''),
    ('''                    await self.collector.collect_slow()
                    label_matured(self.cfg, self.db)''',
     '''                    await self.collector.collect_slow()
                    label_matured(self.cfg, self.db)
                    from .grindsignal import label_signal_plans
                    label_signal_plans(self.cfg, self.db)'''),
])

sub("lts/dashboard.py", [
    ('''        out["movers"] = kv.get("movers_latest")''',
     '''        out["movers"] = kv.get("movers_latest")
        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='signal_plans'"):
            out["plan_stats"] = q("""SELECT plan, leverage, COUNT(*) n, AVG(net > 0) win, AVG(net) net, AVG(margin) margin,
                                     SUM(margin) total_margin FROM signal_plans GROUP BY plan ORDER BY leverage DESC""")'''),
])

sub("lts/web/index.html", [
    ('''    <h2>Grind paper trades</h2><div class="panel scroll" id="grindTrades"></div>''',
     '''    <h2>Grind signals: how each plan did (every signal, scored after 24h)</h2><div class="panel scroll" id="planStats"></div>
    <h2>Grind signals</h2><div class="panel scroll" id="grindTrades"></div>'''),
    ('''  const gsig = (d.signals || []).filter(s => s.kind === "grind" || s.kind === "grindsig");''',
     '''  const gsig = (d.signals || []).filter(s => s.kind === "grind" || s.kind === "grindsig");
  table($("planStats"), [["Plan"],["Signals scored","num"],["Won","num"],["Avg per trade","num"],["Avg on margin","num"],["Total on margin","num"]],
    (d.plan_stats || []).map(r => [esc(r.plan), r.n, pct(r.win, 0), signed(r.net * 100, 2, "%"), signed(r.margin * 100, 1, "%"),
      signed(r.total_margin * 100, 0, "%")]), "No grind signal has completed its 24 hours yet.");'''),
])
print("plans wired")
