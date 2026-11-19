"""Turn scored candidates into signals: thresholds, risk gates, Weex execution checks."""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone

from . import BAR_MS
from .config import Config
from .db import DB
from .candidates import grind_rank
from .labels import costs, plan_from_stop, plan_grind, plan_trade
from .learner import expected_r, predict
from .scoring import reasons, rule_score
from .telegram import esc

log = logging.getLogger("lts.signals")
DAY = 86_400_000


def fmt_price(x: float, precision: int | None) -> str:
    if precision is None or precision < 0:
        precision = max(0, 5 - int(math.floor(math.log10(abs(x)))) - 1) if x else 2
    return f"{x:.{int(precision)}f}"


def utc(ts_ms: int, fmt: str = "%H:%M UTC") -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime(fmt)


class RiskState:
    """Portfolio-level state derived from the signals table."""

    def __init__(self, cfg: Config, db: DB, now_ms: int, kind: str = "momentum"):
        """Limits are per track: grind (paper research) signals never use up momentum slots."""
        day0 = now_ms // DAY * DAY
        self.kind, self.now = kind, now_ms
        self.all_open = [dict(r) for r in db.query("SELECT * FROM signals WHERE status='open'")]
        same = "COALESCE(kind,'momentum')=?"
        self.open = [s for s in self.all_open if (s.get("kind") or "momentum") == kind]
        today = [dict(r) for r in db.query(f"SELECT * FROM signals WHERE ts>=? AND {same} ORDER BY ts", (day0, kind))]
        self.last_ts = max((s["ts"] for s in today), default=0)
        self.sent_today = len(today)
        closed = [s for s in today if s["status"] != "open" and s["ret_r"] is not None]
        self.day_r = sum(s["ret_r"] for s in closed)
        self.streak = 0
        for s in sorted(closed, key=lambda s: s["exit_ts"] or 0, reverse=True):
            if s["ret_r"] < 0:
                self.streak += 1
            else:
                break
        week = db.query(f"SELECT COALESCE(SUM(ret_r),0) AS r FROM signals WHERE status!='open' AND exit_ts>=? "
                        f"AND {same}", (now_ms - 7 * DAY, kind))
        self.week_r = float(week[0]["r"])
        self.safe_mode = self.week_r <= cfg.risk.safe_mode_7d_r
        cool = db.query("SELECT symbol, MAX(exit_ts) AS t FROM signals WHERE ret_r<0 AND exit_ts>=? GROUP BY symbol",
                        (now_ms - int(cfg.risk.symbol_cooldown_hours * 3_600_000),))
        self.cooling = {r["symbol"] for r in cool}

    def block_reason(self, cfg: Config, symbol: str) -> str | None:
        r = cfg.risk
        if self.kind == "grindsig":                       # every qualifying grind is sent; the trader picks.
            if any(s["symbol"] == symbol for s in self.all_open):    # no daily cap, daily stop or streak stop
                return "already_open"
            return "symbol_cooldown" if symbol in self.cooling else None
        if self.kind == "daily":
            if self.open:
                return "daily_trade_open"
            if self.sent_today >= 1:
                return "daily_trade_done"
        elif self.kind == "grind":
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
                return "max_per_day"
        if self.day_r <= r.daily_stop_r:
            return "daily_stop"
        if self.streak >= r.max_consecutive_losses:
            return "loss_streak"
        if any(s["symbol"] == symbol for s in self.all_open):
            return "already_open"
        if symbol in self.cooling:
            return "symbol_cooldown"
        return None


class SignalEngine:
    def __init__(self, cfg: Config, db: DB, weex, notifier):
        self.cfg, self.db, self.weex, self.tg = cfg, db, weex, notifier

    # ---- scoring -------------------------------------------------------
    def score(self, cand: dict, bundle: dict | None) -> None:
        feat = cand["features"]
        if cand.get("kind") == "grindsig":
            g = self.cfg.grindsignal
            tc = self.cfg.trade
            cost_win = 2 * tc.taker_fee + tc.slippage             # Weex TP fills as a market order
            cost_loss = 2 * (tc.taker_fee + tc.slippage)
            cand["all_plans"] = [{**pl, "gain_margin": pl["leverage"] * (pl["target"] - cost_win),
                                  "loss_margin": pl["leverage"] * (pl["stop"] + cost_loss)} for pl in g.plans]
            pp = g.plans[g.primary]                                   # the tracked plan
            cand["plan"] = plan_from_stop(tc, pp["stop"], pp["target"] / pp["stop"], self.cfg.risk.risk_per_trade,
                                          g.hours * 12)
            cand["plan"]["leverage"] = pp["leverage"]
            cand["plan"]["gain_margin"] = cand["all_plans"][g.primary]["gain_margin"]
            cand["plan"]["loss_margin"] = cand["all_plans"][g.primary]["loss_margin"]
            cand["plan"]["margin_frac"] = min(1.0, self.cfg.risk.risk_per_trade / cand["plan"]["loss_margin"])
            cand["rule_score"] = cand["score"] = float(cand["p_model"])
            cand["p"], cand["ev_r"] = cand["p_model"], None
            return
        if cand.get("kind") == "daily":
            a = self.cfg.aim
            tc = self.cfg.trade
            cost = 2 * tc.taker_fee + tc.slippage
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
            # grind-lab paper trade: the study's own plan (-3% stop, +6% target, 24h), ranked by the checklist
            gl = self.cfg.grindlab
            cand["plan"] = plan_from_stop(self.cfg.trade, gl.stop, gl.target / gl.stop, self.cfg.risk.risk_per_trade,
                                          gl.horizon_hours * 12)
            cand["rule_score"] = float(cand["score_n"])
            cand["score"] = 100.0 * cand.get("catalyst", 0) + cand["score_n"] + (cand.get("model_p") or 0.0)
            cand["p"] = cand["ev_r"] = None
            return
        if cand.get("kind") == "grind":
            # separate research track: own trade plan, no model yet (the model is trained on momentum candidates)
            cand["rule_score"] = cand["score"] = grind_rank(feat)
            cand["plan"] = plan_grind(self.cfg.trade, self.cfg.grind, cand["atr_pct"], self.cfg.risk.risk_per_trade)
            cand["p"] = cand["ev_r"] = None
            return
        cand["rule_score"] = rule_score(feat)
        cand["plan"] = plan_trade(self.cfg.trade, cand["atr_pct"], self.cfg.risk.risk_per_trade)
        cand["score"], cand["p"] = predict(bundle, feat) if bundle else (None, None)
        cand["ev_r"] = None
        if cand["plan"] and cand["p"] is not None:
            cand["ev_r"] = expected_r(bundle, cand["p"], cand["plan"]["win_r"], cand["plan"]["loss_r"])

    def _grade(self, cand: dict, bundle: dict | None, state: RiskState, now_ms: int) -> str | None:
        """A / B = validated model (real signals). P = paper: no validated model yet."""
        sc = self.cfg.signal
        if cand.get("kind") == "grindsig":
            from .grindsignal import qualifies
            g = self.cfg.grindsignal
            if cand["score"] < cand.get("p_min", 1.0):
                return None
            band_plans = (cand.get("band") or {}).get("plans") or []
            primary = band_plans[g.primary] if len(band_plans) > g.primary else None
            return "G" if (g.live and qualifies(primary)) else "P"
        if cand.get("kind") == "daily":
            return ("D" if self.cfg.aim.live else "P") if cand["score"] >= self.cfg.aim.threshold else None
        if cand.get("kind") == "grindlab":
            ok = cand.get("catalyst") or cand["score_n"] >= self.cfg.grindlab.min_score
            return ("G" if self.cfg.grind.live else "P") if ok else None
        if cand.get("kind") == "grind":
            g = self.cfg.grind
            return ("G" if g.live else "P") if cand["score"] >= g.min_rank else None
        if bundle is None:
            return "P" if cand["rule_score"] >= sc.rules_min_score else None
        paper = bundle.get("status") != "champion"
        thr = bundle["threshold_safe"] if state.safe_mode else bundle["threshold"]
        if cand["score"] >= thr:
            return "P" if paper else "A"
        hour = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).hour
        if state.sent_today == 0 and not state.safe_mode and hour >= sc.fallback_after_utc_hour \
                and cand["score"] >= bundle["floor"]:
            return "P" if paper else "B"
        return None

    # ---- decision ------------------------------------------------------
    async def decide(self, cands: list[dict], bundle: dict | None, now_ms: int, dry_run: bool = False) -> list[dict]:
        """Score, gate and (unless dry_run) send. Sets cand['decision'] on every candidate."""
        cfg = self.cfg
        for c in cands:
            self.score(c, bundle)
        order = sorted(cands, key=lambda c: (c["score"] if c["score"] is not None else c["rule_score"]), reverse=True)
        order.sort(key=lambda c: c.get("kind") in ("grind", "grindlab"))   # momentum first; stable within a track
        paused = bool(self.db.get("paused", False))
        sent = []
        for c in order:
            # re-read every time: a signal sent this cycle changes it
            state = RiskState(cfg, self.db, now_ms, {"grind": "grind", "grindlab": "grind", "daily": "daily",
                                                     "grindsig": "grindsig"}.get(c.get("kind"), "momentum"))
            feat = c["features"]
            reason = None
            if paused:
                reason = "paused"
            elif cfg.universe.require_weex and not c["on_weex"]:
                reason = "not_on_weex"
            elif c["plan"] is None:
                reason = "too_volatile"
            elif c.get("kind") == "grind" and not cfg.grind.signals_from_5m:
                reason = "logged_only"                    # grind signals come from the grind scanner
            elif c.get("kind") == "grindsig" and cfg.grindsignal.skip_if_level_blocks and (c.get("levels") or {}).get("block"):
                reason = "level_in_way"                   # a daily/weekly/monthly level before the target
            elif c.get("kind", "momentum") == "momentum" and not cfg.signal.momentum_send:
                reason = "logged_only"                    # momentum setups: recorded for learning, not sent
            else:
                grade = self._grade(c, bundle, state, now_ms)
                if grade is None:
                    reason = "below_threshold"
                elif grade != "P" and c["ev_r"] is not None and c["ev_r"] < cfg.signal.min_ev_r:
                    reason = "low_ev"                     # paper signals are tracked regardless of EV
                elif (feat.get("btc_r_15m") or 0.0) <= -cfg.risk.btc_shock_15m:
                    reason = "btc_shock"
                elif (feat.get("fund_rate") or 0.0) > cfg.risk.max_aligned_funding:
                    reason = "funding_crowded"
                else:
                    reason = state.block_reason(cfg, c["symbol"])
                if reason is None:
                    reason = self._execution_check(c)
                if reason is None:
                    c["grade"] = grade
                    if dry_run:
                        c["decision"] = "would_send"
                        sent.append(c)
                        continue
                    sig = self._create_signal(c, bundle, state, now_ms)
                    c["decision"] = "sent"
                    sent.append(c)
                    await self.tg.send(self.signal_message(sig, c, state))
                    continue
            c["decision"] = f"skipped:{reason}"
        return sent

    def _execution_check(self, c: dict) -> str | None:
        """Weex-side tradability using the book snapshot attached by the service."""
        r, u = self.cfg.risk, self.cfg.universe
        feat, plan = c["features"], c["plan"]
        book = c.get("book")
        if book is None:
            return "no_weex_book" if u.require_weex else None
        if (c.get("weex_qv24") or 0.0) < u.min_weex_qvol_24h:
            return "weex_illiquid"
        spread = book["spread_bps"] / 1e4
        if book["spread_bps"] > r.max_spread_bps or spread > r.max_spread_frac_of_sl * plan["sl_pct"]:
            return "spread_too_wide"
        dev = feat.get("weex_dev_bps")
        if dev is not None and not math.isnan(dev) and abs(dev) > r.max_weex_dev_bps:
            return "weex_price_deviation"
        return None

    def _create_signal(self, c: dict, bundle: dict | None, state: RiskState, now_ms: int) -> dict:
        plan, side, book = c["plan"], c["side"], c.get("book")
        entry = (book["ask"] if side > 0 else book["bid"]) if book else c["ref_price"]
        prec = c.get("price_precision")
        sl = entry * (1 - side * plan["sl_pct"])
        tp = entry * (1 + side * plan["tp_pct"])
        if prec is not None:
            sl, tp = round(sl, int(prec)), round(tp, int(prec))
        expires = now_ms + plan.get("horizon_bars", self.cfg.trade.horizon_bars) * BAR_MS
        kind = {"grind": "grind", "grindlab": "grind", "daily": "daily", "grindsig": "grindsig"}.get(c.get("kind"), "momentum")
        version = {"grind": "grind-rules", "grindlab": "grind-lab", "daily": "daily-aim",
                   "grindsig": "grind-model"}.get(c.get("kind")) or \
            (bundle["version"] if bundle else "rules")
        cur = self.db.execute(
            """INSERT INTO signals(candidate_id,symbol,side,ts,grade,entry,sl,tp,sl_pct,tp_pct,leverage,margin_frac,
                                   p,ev_r,model_version,kind,expires_ts,status,last_checked_ts)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',?)""",
            (c.get("id"), c["symbol"], side, now_ms, c["grade"], entry, sl, tp, plan["sl_pct"], plan["tp_pct"],
             plan["leverage"], plan["margin_frac"], c["p"], c["ev_r"], version, kind, expires, now_ms))
        return dict(self.db.query("SELECT * FROM signals WHERE id=?", (cur.lastrowid,))[0]) | {"precision": prec}

    def _grindsig_message(self, sig, c, plan, prec, icon, word) -> str:
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
        lines += self._level_lines(c, prec)
        f = c.get("features", {})
        lines.append(f"6h {f.get('r_6h', 0) * 100:+.1f}% · 24h {f.get('r_24h', 0) * 100:+.1f}% · "
                     f"steadiness R² {f.get('sh_r2_6h', float('nan')):.2f}")
        from .grindsignal import weex_url
        lines.append(f'<a href="{weex_url(sig["symbol"])}">📈 Open {esc(sig["symbol"])} on Weex</a>')
        lines.append(f"#S{sig['id']}")
        return "\n".join(lines)

    def _level_lines(self, c, prec) -> list[str]:
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

    def _daily_message(self, sig, c, plan, prec, icon, word) -> str:
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
        return "\n".join(lines)

    # ---- message -------------------------------------------------------
    def signal_message(self, sig: dict, c: dict, state: RiskState) -> str:
        cfg, plan, side = self.cfg, c["plan"], sig["side"]
        prec = sig.get("precision")
        icon, word = ("🟢", "LONG") if side > 0 else ("🔴", "SHORT")
        chase = sig["entry"] * (1 + side * 0.25 * plan["tp_pct"])
        pos, neg = reasons(c["features"])
        f = c["features"]
        grade_note = {"A": "Grade A", "B": "Grade B (best of day, lower conviction)", "P": "PAPER", "G": "Grind",
                      "D": "Daily trade"}
        if c.get("kind") == "daily":
            return self._daily_message(sig, c, plan, prec, icon, word)
        if c.get("kind") == "grindsig":
            return self._grindsig_message(sig, c, plan, prec, icon, word)
        grind = c.get("kind") in ("grind", "grindlab")
        title = f"{icon} <b>{word} {esc(sig['symbol'])}</b> · {grade_note.get(sig['grade'], sig['grade'])}"
        lines = [title + (" · steady-trend (grind) setup" if grind else "")]
        if c.get("kind") == "grindlab" and sig["grade"] == "P":
            lines.append("📝 <b>Grind lab paper trade — testing the traits of successful grinders forward. "
                         "No proven edge yet. Track it, do not trade it.</b>")
        elif grind and sig["grade"] == "P":
            lines.append("📝 <b>Paper research signal — the grind setup has no proven edge yet "
                         "(break-even over the last month after fees). Track it, do not trade it.</b>")
        elif sig["grade"] == "P":
            lines.append("📝 <b>Paper signal — the model has not passed validation yet. "
                         "Track it, do not trade it with real money.</b>")
        lines += [
            "",
            f"Entry  <code>{fmt_price(sig['entry'], prec)}</code>  (limit order preferred — about half the fee "
            f"cost of a market entry; valid {cfg.trade.entry_valid_minutes} min)",
            f"Stop   <code>{fmt_price(sig['sl'], prec)}</code>  (−{plan['sl_pct']:.2%})",
            f"Target <code>{fmt_price(sig['tp'], prec)}</code>  (+{plan['tp_pct']:.2%})",
            f"Leverage <b>{plan['leverage']}x isolated</b>",
            f"At target ≈ <b>+{plan['gain_margin']:.1%}</b> on margin · at stop ≈ −{plan['loss_margin']:.1%} (after fees)",
        ]
        size = f"Size: margin ≤ {plan['margin_frac']:.0%} of account (risks {cfg.risk.risk_per_trade:.1%})"
        if cfg.telegram.account_size_usdt > 0:
            size += f" ≈ {plan['margin_frac'] * cfg.telegram.account_size_usdt:.0f} USDT"
        lines += [size,
                  f"Time stop: close at market by {utc(sig['expires_ts'])} "
                  f"({utc(sig['expires_ts'] + cfg.telegram.local_utc_offset_minutes * 60_000, '%H:%M')} "
                  f"{esc(cfg.telegram.local_tz_label)}) if neither level is hit",
                  f"Skip if price is already beyond <code>{fmt_price(chase, prec)}</code>", ""]
        if c.get("kind") == "grindlab":
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
            return "\n".join(lines)
        if grind:
            lines.append(f"Grind: 6h {f.get('r_6h', 0) * 100:+.1f}% · straightness R² {f.get('g_r2_6h', 0):.2f} · "
                         f"{int(f.get('g_hours_with') or 0)}/6 hours with the trend · biggest hour "
                         f"{(f.get('g_big1h') or 0):.0%} of the move · hold up to "
                         f"{plan.get('horizon_bars', 288) * 5 // 60}h")
        elif sig["p"] is not None:
            label = "Unvalidated model estimate" if sig["grade"] == "P" else "Model"
            lines.append(f"{label}: P(target first) {sig['p']:.0%} · EV {sig['ev_r']:+.2f}R · "
                         f"{esc(sig['model_version'])}")
        else:
            lines.append(f"Rule score {c['rule_score']:.0f}/100")
        ctx = [f"1h {f.get('r_1h', 0) * 100:+.1f}%", f"24h {f.get('r_24h', 0) * 100:+.1f}%",
               f"vol {f.get('rvol_1h', 0):.1f}x"]
        if f.get("oi_chg_1h") is not None and not math.isnan(f.get("oi_chg_1h")):
            ctx.append(f"OI 1h {f['oi_chg_1h'] * 100:+.1f}%")
        lines.append("With-trend move: " + ", ".join(ctx))
        if pos:
            lines.append("Context: " + "; ".join(esc(p) for p in pos))
        if neg:
            lines.append("Watch: " + "; ".join(esc(n) for n in neg))
        if state.safe_mode:
            lines.append("Safe mode is on (7-day result is negative): threshold raised.")
        lines.append(f"#S{sig['id']}")
        return "\n".join(lines)


def exit_message(cfg: Config, db: DB, sig: dict) -> str:
    icon = {"tp": "✅ Target hit", "sl": "🛑 Stopped out", "time": "⏱ Time stop", "cancelled": "✖️ Cancelled"}[sig["status"]]
    word = "LONG" if sig["side"] > 0 else "SHORT"
    held = max(0, (sig["exit_ts"] - sig["ts"]) // 60_000)
    day0 = sig["exit_ts"] // DAY * DAY
    rows = db.query("SELECT ret_r FROM signals WHERE status!='open' AND exit_ts>=? AND ret_r IS NOT NULL", (day0,))
    wins, losses = sum(1 for r in rows if r["ret_r"] > 0), sum(1 for r in rows if r["ret_r"] <= 0)
    total = sum(r["ret_r"] for r in rows)
    move = sig["side"] * (sig["exit_price"] / sig["entry"] - 1)
    return "\n".join([
        f"{icon} · <b>{esc(sig['symbol'])} {word}</b> #S{sig['id']}",
        f"Move {move:+.2%} · {sig['ret_margin']:+.1%} on margin at {sig['leverage']}x · {sig['ret_r']:+.2f}R · "
        f"held {held // 60}h {held % 60}m",
        f"Today: {wins}W / {losses}L, {total:+.2f}R",
    ])


def close_signal(cfg: Config, db: DB, sig: dict, status: str, exit_price: float, exit_ts: int,
                 mfe: float, mae: float) -> dict:
    cst = costs(cfg.trade)[status if status in ("tp", "sl") else "time"]
    ret = sig["side"] * (exit_price / sig["entry"] - 1)
    ret_r = (ret - cst) / sig["sl_pct"]
    ret_margin = sig["leverage"] * (ret - cst)
    db.execute("UPDATE signals SET status=?, exit_ts=?, exit_price=?, ret_r=?, ret_margin=?, mfe=?, mae=? WHERE id=?",
               (status, exit_ts, exit_price, ret_r, ret_margin, mfe, mae, sig["id"]))
    return dict(db.query("SELECT * FROM signals WHERE id=?", (sig["id"],))[0])


def now_ms() -> int:
    return int(time.time() * 1000)
