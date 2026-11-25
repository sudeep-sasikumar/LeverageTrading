"""Signal policy (docs/DESIGN.md section 28, Part C):

1. Expected net: a grind signal carries the selecting rule's tested net return per trade after costs, with a 90%
   confidence interval from day-clustered statistics. With grindsignal.require_positive_ev on, nothing is sent
   unless that expected net is positive. No-trade days are fine.
2. Go-live gate: grindsignal.live only takes effect once the rule (a) passed the pre-registered bar of the study
   that tested it (recorded in lts/models/steady_history.json, or the model's band table), AND (b) has at least
   grindsignal.min_live_paper_trades closed live PAPER trades sent under exactly that rule.
"""
from __future__ import annotations

from .grindsignal import plan_name, qualifies, rule_key, rule_record

Z90 = 1.6449


def rule_version(gs) -> str:
    """Stored as signals.model_version, so live paper trades can be matched to the rule that sent them."""
    return "grind-" + rule_key(gs).replace("/", "-")


def expected_net(gs, cand: dict | None = None) -> dict | None:
    """{"net", "lo", "hi", "n", "source"} for the tracked plan under the configured rule, or None if untested."""
    pl = gs.plans[gs.primary]
    if gs.selector == "steady":
        rec = rule_record(gs) or {}
        st = (rec.get("plans") or {}).get(plan_name(pl))
        if not st or st.get("net") is None:
            return None
        se = st.get("se_day")
        src = rec.get("source", "steady rule, past year")
    else:
        bp = ((cand or {}).get("band") or {}).get("plans") or []
        st = bp[gs.primary] if len(bp) > gs.primary else None
        if not st:
            return None
        t = st.get("t_day") or 0.0
        se = abs(st["net"] / t) if t else None
        src = "model score band, unseen months"
    net = float(st["net"])
    lo, hi = (net - Z90 * se, net + Z90 * se) if se else (None, None)
    return {"net": net, "lo": lo, "hi": hi, "n": st.get("n"), "source": src}


def passed_bar(gs, cand: dict | None = None) -> bool:
    if gs.selector == "steady":
        return bool((rule_record(gs) or {}).get("passed_bar"))
    bp = ((cand or {}).get("band") or {}).get("plans") or []
    return qualifies(bp[gs.primary] if len(bp) > gs.primary else None)


def live_paper_trades(db, gs) -> int:
    r = db.query("SELECT COUNT(*) AS n FROM signals WHERE kind='grindsig' AND model_version=? AND grade='P' "
                 "AND status IN ('tp','sl','time')", (rule_version(gs),))
    return int(r[0]["n"]) if r else 0


def live_allowed(cfg, db, cand: dict | None = None) -> tuple[bool, list[str]]:
    """Whether a grind signal may be graded real-money ('G'), and every reason it may not."""
    gs = cfg.grindsignal
    why = []
    if not gs.live:
        why.append("grindsignal.live is off")
    if not passed_bar(gs, cand):
        why.append(f"rule {rule_key(gs)} has not passed the pre-registered bar")
    n = live_paper_trades(db, gs)
    if n < gs.min_live_paper_trades:
        why.append(f"{n}/{gs.min_live_paper_trades} closed live paper trades under {rule_key(gs)}")
    return (not why), why
