def sub(p, old, new, n=1):
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == n, (p, old[:70], s.count(old))
    open(p, "w", encoding="utf-8").write(s.replace(old, new))


# ---- service: levels for every recorded grind, carried into the signal --------------
p = "lts/service.py"
sub(p, '''            if not last or crossed:
                keep.append(r)
        _insert_live(self.db, keep)
        return keep''', '''            if not last or crossed:
                keep.append(r)
        _insert_live(self.db, keep)
        self.grind_levels(keep)
        return keep

    def grind_levels(self, rows: list[dict]) -> None:
        """Daily/weekly/monthly levels between entry and the tracked plan's target and stop, for each recorded grind."""
        import json
        from .grindlab import _daily
        from .levels import between, major_levels
        g = self.cfg.grindsignal
        pl = g.plans[g.primary]
        for r in rows:
            try:
                d = _daily(self.db, r["symbol"]).reset_index()
                lv = between(major_levels(d, r["ts"]), r["price"], r["side"], pl["target"], pl["stop"])
            except Exception:
                log.exception("levels for %s", r["symbol"])
                continue
            r["levels"] = lv
            self.db.execute("UPDATE grind_live SET levels=? WHERE symbol=? AND ts=? AND side=?",
                            (json.dumps(lv), r["symbol"], r["ts"], r["side"]))''')
sub(p, '''                          "atr_pct": r.get("atr_1h"), "kind": "grindsig", "p_model": r["p"], "band": r["band"],
                          "p_min": p_min,''', '''                          "atr_pct": r.get("atr_1h"), "kind": "grindsig", "p_model": r["p"], "band": r["band"],
                          "p_min": p_min, "levels": r.get("levels"),''')

# ---- signals: optional skip + show the levels on the message ---------------------
p = "lts/signals.py"
sub(p, '''            elif c.get("kind", "momentum") == "momentum" and not cfg.signal.momentum_send:''',
'''            elif c.get("kind") == "grindsig" and cfg.grindsignal.skip_if_level_blocks and (c.get("levels") or {}).get("block"):
                reason = "level_in_way"                   # a daily/weekly/monthly level before the target
            elif c.get("kind", "momentum") == "momentum" and not cfg.signal.momentum_send:''')
sub(p, '''        f = c.get("features", {})
        lines.append(f"6h {f.get('r_6h', 0) * 100:+.1f}% · 24h {f.get('r_24h', 0) * 100:+.1f}% · "''',
'''        lines += self._level_lines(c, prec)
        f = c.get("features", {})
        lines.append(f"6h {f.get('r_6h', 0) * 100:+.1f}% · 24h {f.get('r_24h', 0) * 100:+.1f}% · "''')
sub(p, '''    def _daily_message(self, sig, c, plan, prec, icon, word) -> str:''', '''    def _level_lines(self, c, prec) -> list[str]:
        """Daily/weekly/monthly levels between entry and the tracked plan's target, and between entry and its stop."""
        lv = c.get("levels")
        if lv is None:
            return []
        from .levels import describe
        g = self.cfg.grindsignal
        pl = g.plans[g.primary]
        word = "resistance" if c["side"] > 0 else "support"
        fmt = lambda p: fmt_price(p, prec)
        out = [""]
        if lv.get("block"):
            out.append(f"⚠️ Level before the {pl['leverage']}x target ({word}): " +
                       "; ".join(esc(describe(x, fmt)) for x in lv["block"][:3]))
        else:
            out.append(f"✅ No daily/weekly/monthly level before the {pl['leverage']}x target")
        if lv.get("cushion"):
            out.append("🛡 Level between entry and stop: " + "; ".join(esc(describe(x, fmt)) for x in lv["cushion"][:3]))
        return out

    def _daily_message(self, sig, c, plan, prec, icon, word) -> str:''')

# ---- dashboard: show levels on every grind row --------------------------------
p = "lts/dashboard.py"
sub(p, '''                prog = ", progress" if "progress" in cols_gl else ", NULL AS progress"''',
'''                prog = ", progress" if "progress" in cols_gl else ", NULL AS progress"
                prog += ", levels" if "levels" in cols_gl else ", NULL AS levels"''')
sub(p, '''                    pr = json.loads(r.pop("progress") or "null") or {}''', '''                    pr = json.loads(r.pop("progress") or "null") or {}
                    lv = json.loads(r.pop("levels") or "null")
                    r["levels"] = None if lv is None else {"block": [x["tf"] for x in lv.get("block", [])][:3],
                                                           "cushion": [x["tf"] for x in lv.get("cushion", [])][:3]}''')
p = "lts/web/index.html"
sub(p, '''  table($("grindAll"), [["Time"],["Coin"],["Side"],["Model score","num"],["Needed","num"],["Sent?"]]''',
'''  const lvCell = r => !r.levels ? '<span class="muted">–</span>' :
    (r.levels.block.length ? `<span class="neg" title="a level sits between entry and the target">⚠ ${esc([...new Set(r.levels.block)].join(", "))}</span>` : '<span class="pos">clear</span>') +
    (r.levels.cushion.length ? ` <span class="muted" title="a level sits between entry and the stop">🛡 ${esc([...new Set(r.levels.cushion)].join(", "))}</span>` : "");
  table($("grindAll"), [["Time"],["Coin"],["Side"],["Model score","num"],["Needed","num"],["Sent?"],["Levels to target / stop"]]''')
sub(p, '''      r.decision === "sent" ? '<span class="chip accent">sent</span>' : '<span class="muted">' + esc((r.decision || "–").replace("skipped:", "").split(" (")[0]) + "</span>"]''',
'''      r.decision === "sent" ? '<span class="chip accent">sent</span>' : '<span class="muted">' + esc((r.decision || "–").replace("skipped:", "").split(" (")[0]) + "</span>", lvCell(r)]''')
print("ok")
