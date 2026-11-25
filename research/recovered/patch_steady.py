def sub(p, old, new, n=1):
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == n, (p, old[:80], s.count(old))
    open(p, "w", encoding="utf-8").write(s.replace(old, new))


# ---- config ------------------------------------------------------------------------
sub("lts/config.py", "    top_share: float = 0.20              # send the best-scoring 20% of grinds (~30 a day) - trader's choice\n",
    "    top_share: float = 0.20              # used only when selector = \"model\"\n"
    "    # what gets sent. \"steady\" (default): only all-day steady grinds (DESIGN.md section 25) - about 1.4 a day,\n"
    "    # 5x won 38% on the year vs 34% for the model's top 20%. \"model\": the minute-shape model's top_share.\n"
    "    selector: str = \"steady\"\n"
    "    steady_min_r24: float = 0.08         # moved >= 8% the trade's way over 24h\n"
    "    steady_min_r2_24: float = 0.80       # straight 24h path (R^2 of hourly closes)\n"
    "    steady_min_hours24: int = 15         # >= 15 of the last 24 hours moved the trade's way\n"
    "    steady_min_r2_6: float = 0.90        # very straight last 6h\n"
    "    steady_min_hours6: int = 5           # >= 5 of the last 6 hours the trade's way\n"
    "    steady_max_big6: float = 0.35        # no single hour did more than 35% of the 6h move\n")

# ---- grindsignal: the rule and its year record ---------------------------------------
p = "lts/grindsignal.py"
sub(p, "def weex_url(symbol: str) -> str:", '''# Year record of the steady rule (63,528 early grinds, Oct 2025 - Nov 2026; 539 passed, 1.4 a day;
# taker fee in and out, 0.04% slippage on entry and stop). DESIGN.md section 25.
STEADY_HISTORY = {
    "10x +1.00% / -0.50%": {"n": 539, "win": 0.343, "net": -0.00211, "months_pos": 0.07},
    "5x +2.00% / -1.00%": {"n": 539, "win": 0.380, "net": -0.00084, "months_pos": 0.36},
    "3x +3.33% / -1.00%": {"n": 539, "win": 0.263, "net": -0.00089, "months_pos": 0.50},
    "2x +5.00% / -0.75%": {"n": 539, "win": 0.152, "net": -0.00113, "months_pos": 0.43},
}


def steady(r: dict, gs) -> bool:
    """All-day steady grind: a straight 24h move with most hours the trade's way, and a very straight last 6h
    in which no single hour did the work. Factors are side-aligned (grindlab.decision_row)."""
    def v(k):
        x = r.get(k)
        return float(x) if x is not None and x == x else float("nan")
    return (v("r_24h") >= gs.steady_min_r24 and v("r2_24h") >= gs.steady_min_r2_24 and
            v("hours_with24") >= gs.steady_min_hours24 and v("r2_6h") >= gs.steady_min_r2_6 and
            v("hours_with6") >= gs.steady_min_hours6 and v("big6") <= gs.steady_max_big6)


def selected(r: dict, gs, p_min: float | None) -> bool:
    if gs.selector == "steady":
        return steady(r, gs)
    return r.get("p") is not None and p_min is not None and r["p"] >= p_min


def why_not(r: dict, gs, p_min: float | None) -> str:
    if gs.selector != "steady":
        return f"below model threshold ({r.get('p', 0):.3f} < {p_min:.3f})"
    f = lambda k: r.get(k) if r.get(k) is not None else float("nan")
    miss = []
    if not f("r_24h") >= gs.steady_min_r24:
        miss.append(f"24h {f('r_24h') * 100:+.1f}%")
    if not f("r2_24h") >= gs.steady_min_r2_24:
        miss.append(f"24h R² {f('r2_24h'):.2f}")
    if not f("hours_with24") >= gs.steady_min_hours24:
        miss.append(f"{f('hours_with24'):.0f}/24 hours")
    if not f("r2_6h") >= gs.steady_min_r2_6:
        miss.append(f"6h R² {f('r2_6h'):.2f}")
    if not f("hours_with6") >= gs.steady_min_hours6:
        miss.append(f"{f('hours_with6'):.0f}/6 hours")
    if not f("big6") <= gs.steady_max_big6:
        miss.append(f"one hour {f('big6'):.0%} of the 6h move")
    return "not steady all day (" + ", ".join(miss) + ")"


def weex_url(symbol: str) -> str:''')

# ---- grind_live column for "met the sending rule" -----------------------------------------
p = "lts/grindlab.py"
sub(p, "    anatomy TEXT,                      -- tracked plan: entry -> peak -> exit, what changed at the turn (lts/reversal.py)\n",
    "    anatomy TEXT,                      -- tracked plan: entry -> peak -> exit, what changed at the turn (lts/reversal.py)\n"
    "    selected INTEGER,                  -- met the sending rule at this record (steady all-day grind, or model top share)\n")
sub(p, '("after_exit", "TEXT"), ("levels", "TEXT"), ("anatomy", "TEXT")):',
    '("after_exit", "TEXT"), ("levels", "TEXT"), ("anatomy", "TEXT"),\n                     ("selected", "INTEGER")):')

# ---- service: select by the rule ---------------------------------------------------------
p = "lts/service.py"
sub(p, '''        """Score every detected grind, then keep (record, and consider for sending) only:
        - a grind with no record for that coin and side in the last 55 minutes (about one record an hour), or
        - a grind whose score has just crossed the sending threshold (it was below at its last record)."""''',
    '''        """Score every detected grind, then keep (record, and consider for sending) only:
        - a grind with no record for that coin and side in the last 55 minutes (about one record an hour), or
        - a grind that has just started to meet the sending rule (it did not at its last record)."""''')
sub(p, '''        keep = []
        for r in rows:
            last = self.db.query("SELECT model_p, threshold FROM grind_live WHERE symbol=? AND side=? AND ts>? "
                                 "ORDER BY ts DESC LIMIT 1", (r["symbol"], r["side"], end - 55 * 60_000))
            p, p_min = r.get("p"), getattr(self, "gs_pmin", None)
            crossed = (p is not None and p_min is not None and p >= p_min and last
                       and (last[0]["model_p"] or 0) < (last[0]["threshold"] or 1))
            if not last or crossed:
                keep.append(r)
        _insert_live(self.db, keep)''', '''        keep = []
        p_min = getattr(self, "gs_pmin", None)
        for r in rows:
            r["selected"] = GS.selected(r, g, p_min)
            last = self.db.query("SELECT selected FROM grind_live WHERE symbol=? AND side=? AND ts>? "
                                 "ORDER BY ts DESC LIMIT 1", (r["symbol"], r["side"], end - 55 * 60_000))
            crossed = r["selected"] and last and not last[0]["selected"]
            if not last or crossed:
                keep.append(r)
        _insert_live(self.db, keep)
        for r in keep:
            self.db.execute("UPDATE grind_live SET selected=? WHERE symbol=? AND ts=? AND side=?",
                            (int(r["selected"]), r["symbol"], r["ts"], r["side"]))''')
sub(p, '''        """Send ALL recorded grinds whose model score is at or above the top-share threshold."""''',
    '''        """Send ALL recorded grinds that meet the sending rule (steady all-day grind, or the model's top share)."""''')
sub(p, '''        for r in rows:
            if r["p"] < p_min:
                continue
            m = meta.get(r["symbol"], {})
            picks.append({"symbol": r["symbol"], "side": r["side"], "ts": r["ts"], "ref_price": r["price"],
                          "atr_pct": r.get("atr_1h"), "kind": "grindsig", "p_model": r["p"], "band": r["band"],
                          "p_min": p_min, "levels": r.get("levels"),''', '''        for r in rows:
            if not r.get("selected"):
                continue
            m = meta.get(r["symbol"], {})
            picks.append({"symbol": r["symbol"], "side": r["side"], "ts": r["ts"], "ref_price": r["price"],
                          "atr_pct": r.get("atr_1h"), "kind": "grindsig", "p_model": r["p"], "band": r["band"],
                          "p_min": p_min if g.selector != "steady" else -1.0, "selector": g.selector,
                          "steady": {k: r.get(k) for k in ("r_24h", "r2_24h", "hours_with24", "r2_6h", "hours_with6", "big6")},
                          "levels": r.get("levels"),''')
sub(p, '''            dec = decided.get((r["symbol"], r["side"])) or f"below model threshold ({r['p']:.3f} < {p_min:.3f})"''',
    '''            dec = decided.get((r["symbol"], r["side"])) or GS.why_not(r, g, p_min)''')
sub(p, '''        log.info("grind signals: %d early grinds scored, %d in the top %.0f%%, %d sent", len(rows), len(picks), g.top_share * 100,''',
    '''        log.info("grind signals: %d early grinds, %d met the %s rule, %d sent", len(rows), len(picks), g.selector,''')

# ---- signals: message for steady picks -------------------------------------------------------
p = "lts/signals.py"
sub(p, '''        b = c.get("band") or {}
        hist = {(h["leverage"], h["target"], h["stop"]): h for h in b.get("plans", [])}
        top = f"top {b['top']:.0%} of grinds" if b.get("top", 1) < 1 else "all grinds"
        lines = [f"{icon} <b>GRIND {word} {esc(sig['symbol'])}</b> · model score {c['p_model']:.2f} ({top})"
                 + (" · PAPER" if sig["grade"] == "P" else "")]''', '''        b = c.get("band") or {}
        from .grindsignal import STEADY_HISTORY, plan_name
        is_steady = c.get("selector") == "steady"
        if is_steady:
            hist = {(pl["leverage"], pl["target"], pl["stop"]): STEADY_HISTORY.get(plan_name(pl)) for pl in g.plans}
            top = "steady all-day grinds, past year"
            st = c.get("steady") or {}
            head = (f" · all-day steady grind: 24h {(st.get('r_24h') or 0) * 100:+.1f}%, "
                    f"{(st.get('hours_with24') or 0):.0f}/24 hours its way, R² 24h {(st.get('r2_24h') or 0):.2f} / 6h {(st.get('r2_6h') or 0):.2f}")
        else:
            hist = {(h["leverage"], h["target"], h["stop"]): h for h in b.get("plans", [])}
            top = f"top {b['top']:.0%} of grinds" if b.get("top", 1) < 1 else "all grinds"
            head = f" · model score {c['p_model']:.2f} ({top})"
        lines = [f"{icon} <b>GRIND {word} {esc(sig['symbol'])}</b>{head}" + (" · PAPER" if sig["grade"] == "P" else "")]''')
sub(p, '''            h = hist.get((pl["leverage"], pl["target"], pl["stop"]))
            if h:
                lines.append(f"     history ({top}, unseen months): won {h['win']:.0%}, {h['net'] * 100:+.2f}% per trade, "
                             f"{h['margin'] * 100:+.1f}% margin avg")''', '''            h = hist.get((pl["leverage"], pl["target"], pl["stop"]))
            if h and is_steady:
                lines.append(f"     history ({top}): won {h['win']:.0%}, {h['net'] * 100:+.2f}% per trade, "
                             f"{h['net'] * pl['leverage'] * 100:+.1f}% margin avg")
            elif h:
                lines.append(f"     history ({top}, unseen months): won {h['win']:.0%}, {h['net'] * 100:+.2f}% per trade, "
                             f"{h['margin'] * 100:+.1f}% margin avg")''')

# ---- dashboard: live record of the rule -------------------------------------------------------
p = "lts/dashboard.py"
sub(p, '''                prog += ", levels" if "levels" in cols_gl else ", NULL AS levels"''',
    '''                prog += ", levels" if "levels" in cols_gl else ", NULL AS levels"
                prog += ", selected" if "selected" in cols_gl else ", NULL AS selected"''')
sub(p, '''                out["grind_all"] = allg''', '''                out["grind_all"] = allg
                five = "5x +2.00% / -1.00%"
                def rec(rows):
                    v = [(r["plans"] or {}).get(five) if r["plans"] else ((r.get("progress") or {}).get("plans") or {}).get(five)
                         for r in rows]
                    v = [x for x in v if x is not None]
                    return {"n": len(v), "won": sum(1 for x in v if x > 0), "net": (sum(v) / len(v)) if v else None}
                out["rule_live"] = {"selected": rec([r for r in allg if r.get("selected")]),
                                    "others": rec([r for r in allg if r.get("selected") == 0])}''')

p = "lts/web/index.html"
sub(p, '''    statTile("Grind signals (tracked plan: 5x +2% / −1%)", st.grindsig),''',
    '''    statTile("Grind signals (tracked plan: 5x +2% / −1%)", st.grindsig),
    (() => { const rl = (d.rule_live || {}).selected || {}, ot = (d.rule_live || {}).others || {};
      return tile("Steady-grind rule, live (5x)", rl.n ? `${rl.won}/${rl.n} won` : "–",
        rl.n ? `${signed(rl.net * 100, 2, "%")}/trade · other grinds ${ot.n ? Math.round(100 * ot.won / ot.n) + "% won" : "–"}` : "decided grinds that met the rule, last 3 days"); })(),''')
print("ok")
