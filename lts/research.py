"""Research report: base rates and the stop/target/horizon grid over labelled candidates."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .config import Config
from .db import DB

DAY = 86_400_000


def research_text(cfg: Config, db: DB) -> str:
    rows = db.query("SELECT ts, symbol, side, kind, source, label, outcome, ret_r, sl_pct, atr_pct, rule_score, "
                    "mfe, mae, grid FROM candidates WHERE labeled_ts IS NOT NULL AND label IS NOT NULL")
    if not rows:
        return "No labelled candidates yet."
    n = len(rows)
    days = (max(r["ts"] for r in rows) - min(r["ts"] for r in rows)) / DAY
    out = [f"Labelled candidates: {n:,} over {days:.1f} days ({n / max(days, 1e-9):.0f}/day), "
           f"{len({r['symbol'] for r in rows})} symbols"]

    def block(title: str, key):
        groups = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        out.append(f"\n{title}")
        for g in sorted(groups, key=str):
            rs = groups[g]
            out.append(f"  {str(g):<12} n={len(rs):>6}  TP-first {np.mean([r['label'] for r in rs]):6.1%}  "
                       f"avg {np.mean([r['ret_r'] for r in rs]):+.3f}R  "
                       f"SL {np.mean([r['outcome'] == 'sl' for r in rs]):5.1%}  "
                       f"time {np.mean([r['outcome'] == 'time' for r in rs]):5.1%}")

    block("Primary config, all candidates (unfiltered base rate)", lambda r: "all")
    block("By side", lambda r: "long" if r["side"] > 0 else "short")
    block("By setup kind", lambda r: r["kind"])
    block("By rule-score band", lambda r: f"{int(min(r['rule_score'], 99) // 20) * 20:02d}-{int(min(r['rule_score'], 99) // 20) * 20 + 19}")
    sl = np.array([r["sl_pct"] for r in rows])
    out.append(f"\nStop distance: median {np.median(sl):.2%}, p10 {np.quantile(sl, .1):.2%}, p90 {np.quantile(sl, .9):.2%}")
    mfe = np.array([r["mfe"] for r in rows])
    mae = np.array([r["mae"] for r in rows])
    out.append(f"Within the horizon: median best move {np.median(mfe):.2%}, median worst move {np.median(mae):.2%}; "
               f"share reaching +1% {np.mean(mfe >= 0.01):.0%}, +2% {np.mean(mfe >= 0.02):.0%}, +3% {np.mean(mfe >= 0.03):.0%}")

    grid = defaultdict(list)
    limit = []
    for r in rows:
        for key, (outcome, ret_r, _bars) in json.loads(r["grid"] or "{}").items():
            if key == "limit":
                limit.append((outcome, ret_r, r["ret_r"]))
            else:
                grid[key].append((outcome, ret_r))
    if limit:
        fills = [x for x in limit if x[0] != "nofill"]
        missed = [x[2] for x in limit if x[0] == "nofill"]
        out.append(f"\nLimit entry at the trigger close (maker fee) instead of market: fills {len(fills) / len(limit):.0%}; "
                   f"filled trades TP-first {np.mean([x[0] == 'tp' for x in fills]):.1%}, avg {np.mean([x[1] for x in fills]):+.3f}R; "
                   f"per signal {np.sum([x[1] for x in fills]) / len(limit):+.3f}R "
                   f"(market entry: {np.mean([x[2] for x in limit]):+.3f}R). "
                   f"The missed (unfilled) trades would have made {np.mean(missed) if missed else 0:+.3f}R at market.")
    out.append("\nBarrier grid (stop = mult x ATR | reward:risk | horizon bars), unfiltered:")
    out.append("  sl_mult  rr   horizon      n   TP-first   avg R   breakeven win")
    for key in sorted(grid, key=lambda k: -np.mean([x[1] for x in grid[k]])):
        m, rr, hz = key.split("|")
        vals = grid[key]
        out.append(f"  {m:>6} {rr:>4} {hz:>8} {len(vals):>7}   {np.mean([v[0] == 'tp' for v in vals]):6.1%}  "
                   f"{np.mean([v[1] for v in vals]):+.3f}   ~{1 / (1 + float(rr)):.0%}")
    return "\n".join(out)
