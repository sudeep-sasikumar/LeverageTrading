"""Daily grinder review.

Once a day, look back over the last 24 hours and find the coins that moved a lot *steadily*
(the "grind" shape: straight path, most hours moving the same way, no single candle doing the
work). For each, record what the data looked like (volume, open interest, taker flow, funding,
positioning) and whether LTS flagged it in time - and how much of the move was still ahead when
it did. The records accumulate in the `daily_movers` table: a growing, honest library of the
moves the system is trying to catch, used to judge and improve it.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import BAR_MS
from . import features as F
from .config import Config
from .db import DB

log = logging.getLogger("lts.movers")
H, D = 12, 288
DAY = 86_400_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_movers (
    day TEXT NOT NULL,                  -- UTC date the 24h window ends on
    end_ts INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side INTEGER NOT NULL,              -- +1 rose, -1 fell
    ret24 REAL, r2 REAL, er15 REAL, hours_with REAL, big1h REAL, worst_pull REAL,
    rvol24 REAL, oi_chg24 REAL, taker_imb REAL, fund_avg REAL, glob_lsr_chg REAL, top_lsr_chg REAL,
    qv24 REAL,
    flagged_ts INTEGER, flagged_kind TEXT, flagged_left REAL,  -- share of the 24h move still ahead when flagged
    signal_id INTEGER, signal_status TEXT, signal_r REAL,
    PRIMARY KEY (day, symbol)
);
"""

MIN_MOVE, MIN_R2, MAX_BIG1H, MIN_HOURS = 0.10, 0.60, 0.45, 14     # >=14 of 24 hours moving the same way
TOP_N = 5


def path_stats(o0: float, c: np.ndarray) -> dict:
    """Shape of a 24h path: c = the 288 5m closes, o0 = the window's opening price."""
    lp = np.log(np.concatenate([[o0], c]))
    ret = c[-1] / o0 - 1
    s = 1.0 if ret >= 0 else -1.0
    p15 = lp[::3]                                          # 15m steps, 97 points
    d15 = np.diff(p15)
    er15 = abs(d15.sum()) / max(np.abs(d15).sum(), 1e-12)
    t = np.arange(len(p15))
    r2 = float(np.corrcoef(t, p15)[0, 1] ** 2) if np.std(p15) > 0 else 0.0
    d1h = np.diff(lp[::12])                                # 24 hourly steps
    move = max(abs(lp[-1] - lp[0]), 1e-12)
    sp = lp * s
    run_max = np.maximum.accumulate(sp)
    progress = run_max - sp[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        pull = np.where(progress > 0.03, (run_max - sp) / progress, 0.0)
    return {"ret24": float(ret), "side": int(s), "r2": r2, "er15": float(er15),
            "hours_with": float(np.sum(np.sign(d1h) == s)), "big1h": float(np.max(s * d1h) / move),
            "worst_pull": float(np.max(pull))}


def is_grinder(st: dict) -> bool:
    return (abs(st["ret24"]) >= MIN_MOVE and st["r2"] >= MIN_R2 and st["big1h"] <= MAX_BIG1H
            and st["hours_with"] >= MIN_HOURS)


def _flows(db: DB, sym: str, k: pd.DataFrame, end: int, s: int) -> dict:
    w = k.iloc[-D:]
    qv24 = float(w["qv"].sum())
    prev = k["qv"].iloc[-8 * D:-D]
    base = float(prev.sum()) / max(len(prev) / D, 1e-9) if len(prev) >= D else np.nan
    imb = (2 * float(w["tbqv"].sum()) - qv24) / qv24 if qv24 > 0 else np.nan
    oi = db.frame("SELECT ts, oi FROM oi WHERE symbol=? AND ts>=? AND ts<=? ORDER BY ts", [sym, end - DAY - 3_600_000, end])
    oi_chg = np.nan
    if len(oi) >= 2 and oi["ts"].iloc[0] <= end - DAY + 3_600_000:
        oi_chg = float(oi["oi"].iloc[-1] / oi["oi"].iloc[0] - 1)
    fund = db.frame("SELECT rate FROM funding WHERE symbol=? AND ts>? AND ts<=?", [sym, end - DAY, end])
    lsr = db.frame("SELECT ts, top_pos, global_acc FROM lsr WHERE symbol=? AND ts>=? AND ts<=? ORDER BY ts",
                   [sym, end - DAY - 3_600_000, end])

    def chg(col):
        x = lsr[col].dropna()
        return float(np.log(x.iloc[-1] / x.iloc[0])) * s if len(x) >= 2 and x.iloc[0] > 0 else np.nan
    return {"rvol24": qv24 / base if base and base > 0 else np.nan, "oi_chg24": oi_chg * s if np.isfinite(oi_chg) else np.nan,
            "taker_imb": imb * s if np.isfinite(imb) else np.nan,
            "fund_avg": float(fund["rate"].mean()) * s if len(fund) else np.nan,
            "glob_lsr_chg": chg("global_acc"), "top_lsr_chg": chg("top_pos"), "qv24": qv24}


def _caught(db: DB, sym: str, side: int, start: int, end: int, o0: float, last: float) -> dict:
    """Did LTS flag this move, when, and how much of it was still ahead at that point?"""
    row = db.query("SELECT ts, kind, ref_price FROM candidates WHERE source='live' AND symbol=? AND side=? "
                   "AND ts>? AND ts<=? ORDER BY ts LIMIT 1", (sym, side, start, end))
    out = {"flagged_ts": None, "flagged_kind": None, "flagged_left": None,
           "signal_id": None, "signal_status": None, "signal_r": None}
    if row:
        r = row[0]
        total = np.log(last / o0) * side
        left = np.log(last / r["ref_price"]) * side
        out.update(flagged_ts=r["ts"], flagged_kind="grind" if r["kind"] == "grind" else "momentum",
                   flagged_left=float(left / total) if total > 0 else None)
    sig = db.query("SELECT id, status, ret_r FROM signals WHERE symbol=? AND side=? AND ts>? AND ts<=? ORDER BY ts LIMIT 1",
                   (sym, side, start, end))
    if sig:
        out.update(signal_id=sig[0]["id"], signal_status=sig[0]["status"], signal_r=sig[0]["ret_r"])
    return out


def review(cfg: Config, db: DB, end_ts: int | None = None, store: bool = True) -> dict:
    """Find the last 24h's steady movers, profile them, and record whether LTS caught them."""
    db.conn.executescript(SCHEMA)
    last = db.last_kline_ts()
    if not last:
        return {"end_ts": None, "movers": [], "n_coins": 0}
    end = end_ts or (last.get("BTCUSDT") or max(last.values())) + BAR_MS
    start = end - DAY
    since = end - 8 * DAY
    movers, n = [], 0
    for r in db.query("SELECT symbol FROM symbols WHERE active=1"):
        sym = r["symbol"]
        if last.get(sym, 0) + BAR_MS < end:
            continue
        k = db.klines(sym, since, end - BAR_MS)
        if len(k) < D + 1:
            continue
        k = F.regularize(k)
        w = k.loc[start:end - BAR_MS]
        if len(w) < D or w["qv"].sum() < cfg.universe.min_qvol_24h / 2:
            continue
        n += 1
        st = path_stats(float(w["o"].iloc[0]), w["c"].to_numpy())
        if not is_grinder(st):
            continue
        st |= _flows(db, sym, k.loc[:end - BAR_MS], end, st["side"])
        st |= _caught(db, sym, st["side"], start, end, float(w["o"].iloc[0]), float(w["c"].iloc[-1]))
        movers.append({"symbol": sym, **st})
    movers.sort(key=lambda m: abs(m["ret24"]) * m["r2"], reverse=True)
    top = [m for m in movers if m["side"] > 0][:TOP_N] + [m for m in movers if m["side"] < 0][:TOP_N]
    day = datetime.fromtimestamp(end / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    result = {"day": day, "end_ts": end, "n_coins": n, "n_grinders": len(movers), "movers": top}
    if store:
        cols = ["day", "end_ts", "symbol", "side", "ret24", "r2", "er15", "hours_with", "big1h", "worst_pull",
                "rvol24", "oi_chg24", "taker_imb", "fund_avg", "glob_lsr_chg", "top_lsr_chg", "qv24",
                "flagged_ts", "flagged_kind", "flagged_left", "signal_id", "signal_status", "signal_r"]
        db.insert_rows("daily_movers", cols, [tuple(_clean((m | {"day": day, "end_ts": end}).get(c)) for c in cols)
                                              for m in movers])
        db.set("movers_latest", json.loads(json.dumps(result, default=_clean)))
    return result


def _clean(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def review_text(res: dict, local_offset_min: int = 330, tz_label: str = "IST") -> str:
    from .telegram import esc
    if not res.get("movers"):
        return "<b>Steady movers (last 24h)</b>\nNo coin moved 10%+ in a steady line today."
    lines = [f"<b>Steady movers, last 24h</b> ({res['n_grinders']} of {res['n_coins']} coins moved 10%+ in a steady line)"]
    for m in res["movers"]:
        icon = "📈" if m["side"] > 0 else "📉"
        bits = [f"R² {m['r2']:.2f}", f"{int(m['hours_with'])}/24h {'up' if m['side'] > 0 else 'down'}"]
        if m.get("oi_chg24") is not None:
            bits.append(f"OI {m['oi_chg24'] * m['side'] * 100:+.0f}%")
        if m.get("rvol24") is not None:
            bits.append(f"vol {m['rvol24']:.1f}x")
        if m.get("flagged_ts"):
            t = datetime.fromtimestamp((m["flagged_ts"] + local_offset_min * 60_000) / 1000, tz=timezone.utc)
            left = f", {m['flagged_left']:.0%} of the move still ahead" if m.get("flagged_left") is not None else ""
            caught = f"flagged {t:%H:%M} {tz_label} ({m['flagged_kind']}{left})"
            if m.get("signal_id"):
                caught += f", signal #S{m['signal_id']} {m['signal_status']}"
        else:
            caught = "not flagged"
        lines.append(f"{icon} <b>{esc(m['symbol'])}</b> {m['ret24'] * 100:+.1f}% · " + " · ".join(bits) + f" — {caught}")
    return "\n".join(lines)


def history_stats(db: DB, days: int = 30) -> dict:
    """How well LTS has been catching steady movers."""
    db.conn.executescript(SCHEMA)
    rows = db.query("SELECT flagged_ts, flagged_left, signal_id FROM daily_movers WHERE end_ts>=?",
                    (int(time.time() * 1000) - days * DAY,))
    if not rows:
        return {"n": 0}
    flagged = [r for r in rows if r["flagged_ts"]]
    left = [r["flagged_left"] for r in flagged if r["flagged_left"] is not None]
    return {"n": len(rows), "flagged": len(flagged) / len(rows),
            "median_left": float(np.median(left)) if left else None,
            "signalled": sum(1 for r in rows if r["signal_id"]) / len(rows)}
