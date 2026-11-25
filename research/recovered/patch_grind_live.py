import pathlib

ROOT = pathlib.Path(r"D:\Leverage trading")


def sub(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:70])
        s = s.replace(a, b, 1)
    p.write_text(s, encoding="utf-8")


sub("lts/config.py", [
    ('''    min_rank: float = 0.025              # rank = R^2 x efficiency x min(6h move, 15%); calibrated by replay''',
     '''    min_rank: float = 0.0                # rank = R^2 x efficiency x min(6h move, 15%). The 31-day replay showed
                                         # higher ranks did NOT do better, so no cut-off: rank only orders them
    live: bool = False                   # paper-only until the grind track has a proven forward record'''),
])

sub("lts/service.py", [
    ("from .candidates import kind_of, trigger_mask", "from .candidates import grind_mask, kind_of, trigger_mask"),
    ('''    cool_from = close_ts - cfg.trigger.cooldown_bars * BAR_MS
    recent = {(r["symbol"], r["side"]) for r in
              db.query("SELECT symbol, side FROM candidates WHERE source='live' AND ts>=? AND ts<?",
                       (cool_from, close_ts))}''',
     '''    gc = cfg.grind
    recent = {(r["symbol"], r["side"]) for r in
              db.query("SELECT symbol, side FROM candidates WHERE source='live' AND COALESCE(kind,'')!='grind' "
                       "AND ts>=? AND ts<?", (close_ts - cfg.trigger.cooldown_bars * BAR_MS, close_ts))}
    recent_grind = {(r["symbol"], r["side"]) for r in
                    db.query("SELECT symbol, side FROM candidates WHERE source='live' AND kind='grind' "
                             "AND ts>=? AND ts<?", (close_ts - gc.cooldown_bars * BAR_MS, close_ts))}'''),
    ('''        if not (abs(p["z_1h"]) >= trig.z_1h or abs(p["z_4h"]) >= trig.z_4h or abs(p["r_24h"]) >= trig.r_24h):
            continue''',
     '''        momentum_possible = abs(p["z_1h"]) >= trig.z_1h or abs(p["z_4h"]) >= trig.z_4h or abs(p["r_24h"]) >= trig.r_24h
        grind_possible = gc.enabled and abs(p["r_6h"]) >= gc.min_r6
        if not (momentum_possible or grind_possible):
            continue'''),
    ('''        for side in (1, -1):
            if (sym, side) in recent or not trigger_mask(last, cfg, side)[0]:
                continue
            aligned = F.align(last, side).iloc[0]
            out.append({"symbol": sym, "side": side, "ts": close_ts, "ref_price": float(k["c"].iloc[-1]),
                        "atr_pct": float(last["atr_pct"].iloc[0]), "aligned": aligned,
                        "on_weex": bool(meta.get("on_weex")), "src": meta.get("src"),
                        "price_precision": meta.get("price_precision")})''',
     '''        base = {"ts": close_ts, "ref_price": float(k["c"].iloc[-1]), "on_weex": bool(meta.get("on_weex")),
                "src": meta.get("src"), "price_precision": meta.get("price_precision"), "symbol": sym}
        for side in (1, -1):
            if (sym, side) not in recent and trigger_mask(last, cfg, side)[0]:
                out.append(base | {"side": side, "kind": None, "atr_pct": float(last["atr_pct"].iloc[0]),
                                   "aligned": F.align(last, side).iloc[0]})
            if (sym, side) not in recent_grind and grind_mask(last, cfg, side)[0]:
                out.append(base | {"side": side, "kind": "grind", "atr_pct": float(last["atr_1h"].iloc[0]),
                                   "aligned": F.align(last, side).iloc[0]})'''),
    ('''            (c["symbol"], c["ts"], c["side"], kind_of(c["features"]), c["ref_price"], c["atr_pct"],''',
     '''            (c["symbol"], c["ts"], c["side"], c.get("kind") or kind_of(c["features"]), c["ref_price"], c["atr_pct"],'''),
])

sub("lts/signals.py", [
    ("from .labels import costs, plan_trade", "from .candidates import grind_rank\nfrom .labels import costs, plan_grind, plan_trade"),
    ('''    def __init__(self, cfg: Config, db: DB, now_ms: int):
        day0 = now_ms // DAY * DAY
        self.open = [dict(r) for r in db.query("SELECT * FROM signals WHERE status='open'")]
        today = [dict(r) for r in db.query("SELECT * FROM signals WHERE ts>=? ORDER BY ts", (day0,))]''',
     '''    def __init__(self, cfg: Config, db: DB, now_ms: int, kind: str = "momentum"):
        """Limits are per track: grind (paper research) signals never use up momentum slots."""
        day0 = now_ms // DAY * DAY
        self.kind, self.now = kind, now_ms
        self.all_open = [dict(r) for r in db.query("SELECT * FROM signals WHERE status='open'")]
        same = "COALESCE(kind,'momentum')=?"
        self.open = [s for s in self.all_open if (s.get("kind") or "momentum") == kind]
        today = [dict(r) for r in db.query(f"SELECT * FROM signals WHERE ts>=? AND {same} ORDER BY ts", (day0, kind))]
        self.last_ts = max((s["ts"] for s in today), default=0)'''),
    ('''        week = db.query("SELECT COALESCE(SUM(ret_r),0) AS r FROM signals WHERE status!='open' AND exit_ts>=?",
                        (now_ms - 7 * DAY,))''',
     '''        week = db.query(f"SELECT COALESCE(SUM(ret_r),0) AS r FROM signals WHERE status!='open' AND exit_ts>=? "
                        f"AND {same}", (now_ms - 7 * DAY, kind))'''),
    ('''    def block_reason(self, cfg: Config, symbol: str) -> str | None:
        r = cfg.risk
        if len(self.open) >= r.max_open:
            return "max_open"
        if self.sent_today >= r.max_per_day:
            return "max_per_day"''',
     '''    def block_reason(self, cfg: Config, symbol: str) -> str | None:
        r = cfg.risk
        if self.kind == "grind":
            g = cfg.grind
            if len(self.open) >= g.max_open:
                return "max_open"
            if self.sent_today >= g.max_per_day:
                return "max_per_day"
            if self.now - self.last_ts < g.min_gap_hours * 3_600_000:
                return "spacing"
        else:
            if len(self.open) >= r.max_open:
                return "max_open"
            if self.sent_today >= r.max_per_day:
                return "max_per_day"'''),
    ('''        if any(s["symbol"] == symbol for s in self.open):
            return "already_open"''',
     '''        if any(s["symbol"] == symbol for s in self.all_open):
            return "already_open"'''),
    ('''    def score(self, cand: dict, bundle: dict | None) -> None:
        feat = cand["features"]''',
     '''    def score(self, cand: dict, bundle: dict | None) -> None:
        feat = cand["features"]
        if cand.get("kind") == "grind":
            # separate research track: own trade plan, no model yet (the model is trained on momentum candidates)
            cand["rule_score"] = cand["score"] = grind_rank(feat)
            cand["plan"] = plan_grind(self.cfg.trade, self.cfg.grind, cand["atr_pct"], self.cfg.risk.risk_per_trade)
            cand["p"] = cand["ev_r"] = None
            return'''),
    ('''        """A / B = validated model (real signals). P = paper: no validated model yet."""
        sc = self.cfg.signal''',
     '''        """A / B = validated model (real signals). P = paper: no validated model yet."""
        sc = self.cfg.signal
        if cand.get("kind") == "grind":
            g = self.cfg.grind
            return ("G" if g.live else "P") if cand["score"] >= g.min_rank else None'''),
    ('''        order = sorted(cands, key=lambda c: (c["score"] if c["score"] is not None else c["rule_score"]), reverse=True)''',
     '''        order = sorted(cands, key=lambda c: (c["score"] if c["score"] is not None else c["rule_score"]), reverse=True)
        order.sort(key=lambda c: c.get("kind") == "grind")       # momentum first; stable within each track'''),
    ('''            state = RiskState(cfg, self.db, now_ms)       # re-read: a signal sent this cycle changes it''',
     '''            # re-read every time: a signal sent this cycle changes it
            state = RiskState(cfg, self.db, now_ms, "grind" if c.get("kind") == "grind" else "momentum")'''),
    ('''        expires = now_ms + self.cfg.trade.horizon_bars * BAR_MS
        cur = self.db.execute(
            """INSERT INTO signals(candidate_id,symbol,side,ts,grade,entry,sl,tp,sl_pct,tp_pct,leverage,margin_frac,
                                   p,ev_r,model_version,expires_ts,status,last_checked_ts)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',?)""",
            (c.get("id"), c["symbol"], side, now_ms, c["grade"], entry, sl, tp, plan["sl_pct"], plan["tp_pct"],
             plan["leverage"], plan["margin_frac"], c["p"], c["ev_r"], bundle["version"] if bundle else "rules",
             expires, now_ms))''',
     '''        expires = now_ms + plan.get("horizon_bars", self.cfg.trade.horizon_bars) * BAR_MS
        kind = "grind" if c.get("kind") == "grind" else "momentum"
        version = "grind-rules" if kind == "grind" else (bundle["version"] if bundle else "rules")
        cur = self.db.execute(
            """INSERT INTO signals(candidate_id,symbol,side,ts,grade,entry,sl,tp,sl_pct,tp_pct,leverage,margin_frac,
                                   p,ev_r,model_version,kind,expires_ts,status,last_checked_ts)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',?)""",
            (c.get("id"), c["symbol"], side, now_ms, c["grade"], entry, sl, tp, plan["sl_pct"], plan["tp_pct"],
             plan["leverage"], plan["margin_frac"], c["p"], c["ev_r"], version, kind, expires, now_ms))'''),
    ('''        grade_note = {"A": "Grade A", "B": "Grade B (best of day, lower conviction)", "P": "PAPER"}
        lines = [f"{icon} <b>{word} {esc(sig['symbol'])}</b> · {grade_note.get(sig['grade'], sig['grade'])}"]
        if sig["grade"] == "P":''',
     '''        grade_note = {"A": "Grade A", "B": "Grade B (best of day, lower conviction)", "P": "PAPER", "G": "Grind"}
        grind = c.get("kind") == "grind"
        title = f"{icon} <b>{word} {esc(sig['symbol'])}</b> · {grade_note.get(sig['grade'], sig['grade'])}"
        lines = [title + (" · steady-trend (grind) setup" if grind else "")]
        if grind and sig["grade"] == "P":
            lines.append("📝 <b>Paper research signal — the grind setup has no proven edge yet "
                         "(break-even over the last month after fees). Track it, do not trade it.</b>")
        elif sig["grade"] == "P":'''),
    ('''        if sig["p"] is not None:''',
     '''        if grind:
            lines.append(f"Grind: 6h {f.get('r_6h', 0) * 100:+.1f}% · straightness R² {f.get('g_r2_6h', 0):.2f} · "
                         f"{int(f.get('g_hours_with') or 0)}/6 hours with the trend · biggest hour "
                         f"{(f.get('g_big1h') or 0):.0%} of the move · hold up to "
                         f"{plan.get('horizon_bars', 288) * 5 // 60}h")
        elif sig["p"] is not None:'''),
])
print("service/signals patched")
