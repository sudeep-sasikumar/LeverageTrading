"""Text reports for the CLI and Telegram."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from .config import Config
from .db import DB

DAY = 86_400_000


def _stats(rows) -> dict:
    rs = [r["ret_r"] for r in rows if r["ret_r"] is not None]
    if not rs:
        return {"n": 0}
    wins = [r for r in rs if r > 0]
    eq, peak, dd = 0.0, 0.0, 0.0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    margin = [r["ret_margin"] for r in rows if r["ret_margin"] is not None]
    return {"n": len(rs), "win": len(wins) / len(rs), "avg_r": sum(rs) / len(rs), "total_r": sum(rs),
            "max_dd": dd, "avg_margin": sum(margin) / len(margin) if margin else 0.0,
            "tp": sum(1 for r in rows if r["status"] == "tp"), "sl": sum(1 for r in rows if r["status"] == "sl"),
            "time": sum(1 for r in rows if r["status"] == "time")}


def _line(label: str, s: dict) -> str:
    if not s["n"]:
        return f"{label}: no closed signals"
    return (f"{label}: {s['n']} closed ({s['tp']} TP / {s['sl']} SL / {s['time']} time), win {s['win']:.0%}, "
            f"avg {s['avg_r']:+.2f}R ({s['avg_margin']:+.1%} margin), total {s['total_r']:+.1f}R, "
            f"max drawdown {s['max_dd']:.1f}R")


def champion_row(db: DB):
    """The validated champion, or failing that the unvalidated shadow model (paper signals)."""
    for status in ("champion", "shadow"):
        rows = db.query("SELECT * FROM models WHERE status=? ORDER BY trained_ts DESC LIMIT 1", (status,))
        if rows:
            return rows[0]
    return None


def _mode(row) -> str:
    return "validated, live signals" if row["status"] == "champion" else "NOT validated, paper signals only"


def performance_text(cfg: Config, db: DB, days: float = 7.0) -> str:
    now = int(time.time() * 1000)
    out = []
    for label, d in (("Today", 0), (f"Last {days:g}d", days), ("Last 30d", 30), ("All time", 10_000)):
        since = now // DAY * DAY if d == 0 else now - int(d * DAY)
        rows = db.query("SELECT * FROM signals WHERE status!='open' AND exit_ts>=? ORDER BY exit_ts", (since,))
        out.append(_line(label, _stats(rows)))
    by_grade = db.query("SELECT grade, COUNT(*) n, AVG(ret_r) r, AVG(ret_r>0) w FROM signals "
                        "WHERE status!='open' GROUP BY grade")
    for g in by_grade:
        out.append(f"  grade {g['grade']}: {g['n']} signals, win {g['w']:.0%}, avg {g['r']:+.2f}R")
    return "\n".join(out)


def status_text(cfg: Config, db: DB) -> str:
    now = int(time.time() * 1000)
    n_sym = db.query("SELECT COUNT(*) c FROM symbols WHERE active=1")[0]["c"]
    k = db.query("SELECT COUNT(*) c, MIN(ts) a, MAX(ts) b FROM klines")[0]
    c = db.query("SELECT COUNT(*) c, SUM(labeled_ts IS NOT NULL AND label IS NOT NULL) l, "
                 "SUM(source='live') live FROM candidates")[0]
    today = db.query("SELECT COUNT(*) c FROM candidates WHERE source='live' AND ts>=?", (now // DAY * DAY,))[0]["c"]
    open_sigs = db.query("SELECT * FROM signals WHERE status='open'")
    champ = champion_row(db)
    fmt = lambda ts: datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if ts else "-"
    lines = [
        f"Universe: {n_sym} active symbols",
        f"Klines: {k['c']:,} bars, {fmt(k['a'])} -> {fmt(k['b'])} UTC",
        f"Candidates: {c['c'] or 0:,} total, {c['l'] or 0:,} labelled, {c['live'] or 0:,} live ({today} today)",
        f"Open signals: {len(open_sigs)}" + "".join(
            f"\n  #S{s['id']} {s['symbol']} {'LONG' if s['side'] > 0 else 'SHORT'} entry {s['entry']:g} "
            f"SL {s['sl']:g} TP {s['tp']:g}" for s in open_sigs),
        f"Paused: {'yes' if db.get('paused', False) else 'no'}",
    ]
    if champ:
        m = json.loads(champ["metrics"])
        lines.append(f"Model: {champ['version']} [{_mode(champ)}] (trained {fmt(champ['trained_ts'])} UTC on "
                     f"{champ['n_train']:,} rows, selection skill {m.get('xsec_corr', float('nan')):+.3f}, "
                     f"score threshold {champ['threshold']:+.3f}, floor {champ['floor']:+.3f})")
    else:
        lines.append("Model: none yet (paper signals from rules only)")
    return "\n".join(lines)


def daily_report_text(cfg: Config, db: DB) -> str:
    now = int(time.time() * 1000)
    day0 = now - DAY
    sigs = db.query("SELECT * FROM signals WHERE ts>=? ORDER BY ts", (day0,))
    cands = db.query("SELECT COUNT(*) c FROM candidates WHERE source='live' AND ts>=?", (day0,))[0]["c"]
    skipped = db.query("SELECT decision, COUNT(*) n FROM candidates WHERE source='live' AND ts>=? "
                       "GROUP BY decision ORDER BY n DESC LIMIT 6", (day0,))
    lines = ["<b>Daily report (last 24h)</b>", f"Candidates scanned: {cands} · signals sent: {len(sigs)}"]
    if not sigs:
        lines.append("No-trade day: nothing cleared the quality bar. Capital preserved.")
    for s in sigs:
        res = "open" if s["status"] == "open" else f"{s['status'].upper()} {s['ret_r']:+.2f}R"
        lines.append(f"  #S{s['id']} {s['symbol']} {'LONG' if s['side'] > 0 else 'SHORT'} [{s['grade']}] → {res}")
    lines.append(performance_text(cfg, db, days=7))
    if skipped:
        lines.append("Skip reasons: " + ", ".join(f"{(r['decision'] or 'n/a').replace('skipped:', '')} {r['n']}"
                                                  for r in skipped))
    champ = champion_row(db)
    lines.append(f"Model: {champ['version']} [{_mode(champ)}]" if champ else "Model: none yet (paper signals)")
    return "\n".join(lines)
