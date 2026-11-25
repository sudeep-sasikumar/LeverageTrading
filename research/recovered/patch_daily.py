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
class Micro:''',
     '''@dataclass
class Aim:
    """The project's aim: ONE trade a day worth ~10% on margin (a +1% move at 10x, +2% at 5x).
    Each hour, the first early grind of the UTC day whose score clears `threshold` becomes the day's trade.
    Settings come from the one-year 5-minute test (docs/DESIGN.md section 15)."""
    enabled: bool = True
    ranker: str = "checklist"            # checklist / smooth / first
    threshold: float = 8.0
    target: float = 0.01                 # +1% in the coin
    stop: float = 0.01
    hours: int = 24                      # time limit
    max_leverage: int = 10
    live: bool = False                   # paper until the forward record proves it


@dataclass
class Micro:'''),
    ('''    micro: Micro = field(default_factory=Micro)''',
     '''    micro: Micro = field(default_factory=Micro)
    aim: Aim = field(default_factory=Aim)'''),
])

sub("lts/signals.py", [
    ('''        if cand.get("kind") == "grindlab":
            # grind-lab paper trade''',
     '''        if cand.get("kind") == "daily":
            a = self.cfg.aim
            tc = self.cfg.trade
            cost = tc.taker_fee + tc.slippage + tc.maker_fee
            lev = int(min(a.max_leverage, max(1, math.ceil(0.10 / max(a.target - cost, 1e-6)))))
            cand["plan"] = plan_from_stop(tc, a.stop, a.target / a.stop, self.cfg.risk.risk_per_trade, a.hours * 12)
            cand["plan"]["leverage"] = lev
            cand["plan"]["gain_margin"] = lev * (a.target - cost)
            cand["plan"]["loss_margin"] = lev * (a.stop + 2 * (tc.taker_fee + tc.slippage))
            cand["plan"]["margin_frac"] = min(1.0, self.cfg.risk.risk_per_trade / cand["plan"]["loss_margin"])
            cand["rule_score"] = cand["score"] = float(cand["aim_score"])
            cand["p"] = cand["ev_r"] = None
            return
        if cand.get("kind") == "grindlab":
            # grind-lab paper trade'''),
    ('''        if cand.get("kind") == "grindlab":
            ok = cand.get("catalyst") or cand["score_n"] >= self.cfg.grindlab.min_score''',
     '''        if cand.get("kind") == "daily":
            return ("D" if self.cfg.aim.live else "P") if cand["score"] >= self.cfg.aim.threshold else None
        if cand.get("kind") == "grindlab":
            ok = cand.get("catalyst") or cand["score_n"] >= self.cfg.grindlab.min_score'''),
    ('''            state = RiskState(cfg, self.db, now_ms, "grind" if c.get("kind") in ("grind", "grindlab") else "momentum")''',
     '''            state = RiskState(cfg, self.db, now_ms, {"grind": "grind", "grindlab": "grind", "daily": "daily"}.get(
                c.get("kind"), "momentum"))'''),
    ('''        kind = "grind" if c.get("kind") in ("grind", "grindlab") else "momentum"
        version = {"grind": "grind-rules", "grindlab": "grind-lab"}.get(c.get("kind")) or \\''',
     '''        kind = {"grind": "grind", "grindlab": "grind", "daily": "daily"}.get(c.get("kind"), "momentum")
        version = {"grind": "grind-rules", "grindlab": "grind-lab", "daily": "daily-aim"}.get(c.get("kind")) or \\'''),
    ('''        grade_note = {"A": "Grade A", "B": "Grade B (best of day, lower conviction)", "P": "PAPER", "G": "Grind"}''',
     '''        grade_note = {"A": "Grade A", "B": "Grade B (best of day, lower conviction)", "P": "PAPER", "G": "Grind",
                      "D": "Daily trade"}
        if c.get("kind") == "daily":
            return self._daily_message(sig, c, plan, prec, icon, word)'''),
    ('''    # ---- message -------------------------------------------------------''',
     '''    def _daily_message(self, sig, c, plan, prec, icon, word) -> str:
        a = self.cfg.aim
        lines = [f"🎯 <b>DAILY TRADE — {word} {esc(sig['symbol'])}</b>" + (" · PAPER" if sig["grade"] == "P" else "")]
        if sig["grade"] == "P":
            lines.append("📝 <b>Paper: forward-testing the daily-trade rule. Track it, do not trade it with real money yet.</b>")
        lines += ["",
                  f"Entry  <code>{fmt_price(sig['entry'], prec)}</code>  (market now)",
                  f"Target <code>{fmt_price(sig['tp'], prec)}</code>  (+{a.target:.1%})",
                  f"Stop   <code>{fmt_price(sig['sl'], prec)}</code>  (−{a.stop:.1%})",
                  f"Leverage <b>{plan['leverage']}x isolated</b> → at target ≈ <b>+{plan['gain_margin']:.1%}</b> on margin, "
                  f"at stop ≈ −{plan['loss_margin']:.1%} (after fees)",
                  f"Size: margin ≤ {plan['margin_frac']:.0%} of account (risks {self.cfg.risk.risk_per_trade:.1%})",
                  f"Close at market after {a.hours}h if neither level is hit",
                  "",
                  f"Why this coin: steady grind, {esc(a.ranker)} score {c['score']:.2f} (needs ≥ {a.threshold:g}); "
                  f"6h {c['features'].get('r_6h', 0) * 100:+.1f}%, 24h {c['features'].get('r_24h', 0) * 100:+.1f}%",
                  f"#S{sig['id']}"]
        return "\\n".join(lines)

    # ---- message -------------------------------------------------------'''),
])

sub("lts/signals.py", [
    ('''        if self.kind == "grind":
            g = cfg.grind''',
     '''        if self.kind == "daily":
            if self.open:
                return "daily_trade_open"
            if self.sent_today >= 1:
                return "daily_trade_done"
        elif self.kind == "grind":
            g = cfg.grind'''),
])

sub("lts/service.py", [
    ('''        await self.engine.decide(worth, None, int(time.time() * 1000))
        for c in cands:''',
     '''        await self.engine.decide(worth, None, int(time.time() * 1000))
        await self.daily_pick(rows, meta)
        for c in cands:'''),
    ('''    async def grind_nightly(self) -> None:''',
     '''    async def daily_pick(self, rows: list[dict], meta: dict) -> None:
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

    async def grind_nightly(self) -> None:'''),
])

p = ROOT / "lts/aim.py"
s = p.read_text(encoding="utf-8")
s += '''

def live_score(row: dict, ranker: str) -> float:
    """The daily-trade ranking score for a live early-grind row (same definitions as the study)."""
    if ranker == "checklist":
        return float(row.get("score") or 0)
    if ranker == "smooth":
        return float((row.get("r2_6h") or 0) * (row.get("hours_with6") or 0))
    if ranker == "catalyst":
        return float(row.get("catalyst_short") or 0)
    return 1.0                                      # "first": any early grind
'''
p.write_text(s, encoding="utf-8")
print("daily-trade track wired")
