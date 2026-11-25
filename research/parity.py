"""P1 (docs/HYPOTHESES.md): the year under the LIVE behaviour of the steady rule (until 25 Nov 2026).

Phase 1, per coin: for every 5-minute end, the 30 rolling one-hour bars ending there (built from 5-minute candles,
as grindlab._rolling_hours), grindlab.hourly_shape on them, the early-grind mask and R3 (grindsignal.steady on
decision_row's side-aligned factors). R3 and the mask need only the last 25 bars, so the hourly-table history
that live puts before the 30 rolling bars cannot change them. A rolling window counts only when all 30 hour bins
hold candles and >= 90% of its 360 five-minute candles exist (as _rolling_hours requires).
Every R3 detection also gets its 5x outcome (entry = open of the 5-minute candle at that end) and its level-in-
way flag (levels.between on daily candles).
Phase 2: the record / send state machine of service.grinds_to_record + RiskState (grindsig, as it was), in time
order over all coins.
Variant (c): the hour_close_first rule = P0's 539 R3 moments, plain (the year result) and with today's limits.
Run: python -m research.parity  ->  docs/results/parity.txt
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.common import BAR, DAY, EVENTS, HOUR, LOCAL_DB, MIN, RESULTS, YEAR_NOW, Tee, candles_db, load_paths, r3_mask

from lts import grindlab as G
from lts import grindsignal as GS
from lts import validate as V
from lts.config import Config
from lts.db import DB
from lts.fees import fees
from lts.levels import between, major_levels

out = Tee("parity.txt")
cfg = Config()
gs = cfg.grindsignal
PL = gs.plans[gs.primary]
FEE = fees()
START = YEAR_NOW - 390 * DAY


def rule_shape(k: pd.DataFrame) -> pd.DataFrame:
    """The columns of grindlab.hourly_shape that the early-grind mask and R3 use, with the same formulas
    (tests/test_research_parity.py checks they are identical); hourly_shape's 30-day features are skipped."""
    from lts.features import strided_r2
    c, qv = k["c"], k["qv"]
    y = np.log(c.to_numpy())
    d1 = np.diff(y, prepend=np.nan)
    f = pd.DataFrame(index=k.index)
    for n in (6, 24):
        f[f"r_{n}h"] = c / c.shift(n) - 1
    f["r2_6h"] = strided_r2(y, 7, 1)
    f["r2_24h"] = strided_r2(y, 25, 1)
    up, dn = pd.Series(d1 > 0, index=k.index).astype(float), pd.Series(d1 < 0, index=k.index).astype(float)
    f["up6"], f["dn6"] = up.rolling(6).sum(), dn.rolling(6).sum()
    f["up24"], f["dn24"] = up.rolling(24).sum(), dn.rolling(24).sum()
    r6 = (np.log(c) - np.log(c.shift(6))).abs()
    f["big_up6"] = pd.Series(d1, index=k.index).rolling(6).max() / r6
    f["big_dn6"] = (-pd.Series(d1, index=k.index)).rolling(6).max() / r6
    f["qv_24h"] = qv.rolling(24).sum()
    return f


def outcome_5x(o, h, l, c, j, side):
    """5x plan from 5-minute candle j (entry = its open), 24h, stop first inside a candle. (net, exit bar) or None."""
    nb = gs.hours * 12
    if j + nb > len(o) or not np.isfinite(o[j]) or np.isfinite(h[j:j + nb]).mean() < 0.98:
        return None
    e = o[j]
    hi, lo = h[j:j + nb], l[j:j + nb]
    fav = np.nan_to_num((hi / e - 1) if side > 0 else (1 - lo / e), nan=-9)
    adv = np.nan_to_num((1 - lo / e) if side > 0 else (hi / e - 1), nan=-9)
    hs, ht = np.flatnonzero(adv >= PL["stop"]), np.flatnonzero(fav >= PL["target"])
    i_s = hs[0] if len(hs) else 10 ** 9
    i_t = ht[0] if len(ht) else 10 ** 9
    if i_s <= i_t and i_s < 10 ** 9:
        return -PL["stop"] - FEE.entry - FEE.stop, int(i_s)
    if i_t < 10 ** 9:
        return PL["target"] - FEE.entry - FEE.target, int(i_t)
    cc = c[j:j + nb]
    last = cc[np.isfinite(cc)][-1]
    return side * (last / e - 1) - FEE.entry - FEE.time, nb - 1


def detections(k5: pd.DataFrame, sym: str, daily: pd.DataFrame | None, with_outcomes: bool = True) -> list[dict]:
    """Every (5-minute end, side) where the early-grind mask fires on rolling hours, with R3 and (for R3) outcome."""
    k5 = k5.drop_duplicates("ts").sort_values("ts")
    t0 = int(k5.ts.min()) // BAR * BAR
    grid = np.arange(t0, int(k5.ts.max()) + BAR, BAR, dtype=np.int64)
    g = k5.set_index("ts").reindex(grid)
    have = g["c"].notna().to_numpy()
    o, h, l, c = (g[x].to_numpy(float) for x in ("o", "h", "l", "c"))
    cnt360 = pd.Series(have.astype(float)).rolling(360).sum().to_numpy()      # candles in the 30h before end
    out_rows = []
    for m in range(12):
        off = m * BAR
        bins = (grid - off) // HOUR
        df = g.assign(bin=bins, have=have)
        agg = df[df["have"]].groupby("bin").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"),
                                                v=("v", "sum"), qv=("qv", "sum"), trades=("trades", "sum"),
                                                tbqv=("tbqv", "sum"))
        if len(agg) < 30:
            continue
        full = np.arange(agg.index.min(), agg.index.max() + 1)
        hb = agg.reindex(full)
        nonempty = hb["c"].notna().astype(float).rolling(30).sum().to_numpy()
        hb.index = full * HOUR + off                                   # bin start; the bin ends 1h later
        hb_f = hb.copy()
        hb_f["c"] = hb_f["c"].ffill()
        for col in ("o", "h", "l"):
            hb_f[col] = hb_f[col].fillna(hb_f["c"])
        for col in ("v", "qv", "trades", "tbqv"):
            hb_f[col] = hb_f[col].fillna(0.0)
        f = rule_shape(hb_f)
        ends = hb.index.to_numpy() + HOUR
        gi = ((ends - t0) // BAR).astype(int) - 1                         # grid index of the last candle before end
        okw = (nonempty == 30) & (gi >= 0) & (gi < len(grid))
        okw[okw] &= cnt360[gi[okw]] >= 324
        for side in (1, -1):
            mask = G.early_grind_mask(f, side) & okw & (ends >= START) & (ends <= YEAR_NOW)
            for i in np.flatnonzero(mask):
                fr = f.iloc[i]
                r = {"r_24h": side * fr["r_24h"], "r2_24h": fr["r2_24h"], "hours_with24": fr["up24"] if side > 0 else fr["dn24"],
                     "r2_6h": fr["r2_6h"], "hours_with6": fr["up6"] if side > 0 else fr["dn6"],
                     "big6": fr["big_up6"] if side > 0 else fr["big_dn6"]}
                st = bool(GS.steady(r, gs))
                e = int(ends[i])
                row = {"symbol": sym, "ts": e, "side": side, "steady": st}
                if st and with_outcomes:
                    j = int((e - t0) // BAR)
                    res = outcome_5x(o, h, l, c, j, side)
                    row["net"], row["exit_bar"] = (res if res else (np.nan, -1))
                    row["entry"] = float(o[j]) if j < len(o) else np.nan
                    if daily is not None and res and np.isfinite(row["entry"]):
                        try:
                            lv = between(major_levels(daily, e), row["entry"], side, PL["target"], PL["stop"])
                            row["level_block"] = bool(lv.get("block"))
                        except Exception:
                            row["level_block"] = False
                out_rows.append(row)
    return out_rows


def replay(det: pd.DataFrame, level_skip: bool) -> pd.DataFrame:
    """service.grinds_to_record (rolling_5m) + RiskState for grindsig as it was until 25 Nov 2026."""
    det = det.sort_values(["ts", "symbol", "side"])
    rec = {}                  # (sym, side) -> (last record ts, selected)
    open_until = {}           # sym -> exit time of its open signal
    last_loss = {}            # sym -> exit time of its last losing signal
    sent = []
    for r in det.itertuples(index=False):
        key = (r.symbol, r.side)
        last = rec.get(key)
        recent = last if last and last[0] > r.ts - 55 * MIN else None
        if recent is None or (r.steady and not recent[1]):
            rec[key] = (r.ts, r.steady)
            if not r.steady:
                continue
            if open_until.get(r.symbol, 0) > r.ts:
                continue                                               # already_open (either side)
            if last_loss.get(r.symbol, -10 ** 18) >= r.ts - int(cfg.risk.symbol_cooldown_hours * HOUR):
                continue                                               # symbol_cooldown
            lb = getattr(r, "level_block", False)
            if level_skip and isinstance(lb, (bool, np.bool_)) and lb:
                continue
            if not np.isfinite(r.net):
                continue
            exit_ts = r.ts + (int(r.exit_bar) + 1) * BAR
            open_until[r.symbol] = exit_ts
            if r.net < 0:
                last_loss[r.symbol] = exit_ts
            sent.append(r._asdict() | {"exit_ts": exit_ts})
    return pd.DataFrame(sent)


def report(lab: str, d: pd.DataFrame) -> dict:
    if not len(d):
        out(f"  {lab:<52} n 0")
        return {}
    st = V.stats(d["net"].to_numpy(), d["ts"].to_numpy())
    d = d.sort_values("ts")
    prev = d.groupby(["symbol", "side"])["ts"].shift(1)
    rep = float(((d["ts"] - prev) < DAY).mean())
    out(f"  {lab:<52} n {st['n']:>5} ({st['per_day']:.1f}/day) won {st['win']:.1%} net {st['mean']:+.3%} "
        f"t {st['t_day']:+.2f} months+ {st['months_pos']:.0%} repeats<24h {rep:.0%}")
    return st | {"repeat_share": rep}


if __name__ == "__main__":
    t_start = time.time()
    out(f"P1 live-parity replay - run {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    ldb = DB(LOCAL_DB)
    conn = candles_db()
    syms = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM k5")]
    rows = []
    for n, sym in enumerate(syms):
        k5 = pd.read_sql_query("SELECT ts,o,h,l,c,v,qv,trades,tbqv FROM k5 WHERE symbol=? ORDER BY ts", conn, params=(sym,))
        if len(k5) < 400:
            continue
        try:
            daily = G._daily(ldb, sym).reset_index()
        except Exception:
            daily = None
        rows += detections(k5, sym, daily)
        if n % 50 == 0:
            out(f"  phase 1: {n}/{len(syms)} coins, {len(rows):,} detections, {time.time() - t_start:.0f}s")
    det = pd.DataFrame(rows)
    det.to_parquet(RESULTS.parent.parent / "research_data" / "parity_detections.parquet")
    out(f"{len(det):,} early-grind detections at 5-minute ends ({int(det.steady.sum()):,} meeting R3) in fetched windows")

    out("\nYear, 5x +2% / -1%, 24h (live behaviour vs the year test):")
    a = replay(det, level_skip=False)
    sa = report("(a) rolling 5m + re-sends (live until 25 Nov)", a)
    b = replay(det, level_skip=True)
    sb = report("(b) (a) + level-in-way skip (live since 19 Nov)", b)

    ev = pd.read_parquet(EVENTS).reset_index(drop=True)
    R3 = r3_mask(ev)
    e3 = ev[R3].reset_index(drop=True)
    post, _ = load_paths(e3[["symbol", "ts"]], 289)
    res = GS.plan_outcomes(post, e3.side.to_numpy(), [PL], gs.hours, with_kind=True)[GS.plan_label(PL)]
    e3["net"], e3["exit_bar"] = res.net, res.exit_bar
    sc = report("(c1) hour_close_first, no limits (= the year R3 test)", e3[np.isfinite(e3.net)])
    # (c2) with today's limits: open signal, 4h cooldown, daily stop, streak stop, same-direction cap
    e3 = e3[np.isfinite(e3.net)].sort_values("ts")
    keep, open_until, last_loss, closed = [], {}, {}, []
    for r in e3.itertuples(index=False):
        day0 = r.ts // DAY * DAY
        today = [x for x in closed if x["exit_ts"] >= day0 and x["exit_ts"] <= r.ts and x["ts"] >= day0]
        day_r = sum(x["net"] / PL["stop"] for x in today)
        streak = 0
        for x in sorted(today, key=lambda x: x["exit_ts"], reverse=True):
            if x["net"] < 0:
                streak += 1
            else:
                break
        same = [x for x in keep if x["side"] == r.side and (x["exit_ts"] > r.ts or x["ts"] >= day0)]
        if open_until.get(r.symbol, 0) > r.ts or last_loss.get(r.symbol, -10 ** 18) >= r.ts - 4 * HOUR:
            continue
        if day_r <= cfg.risk.daily_stop_r or streak >= cfg.risk.max_consecutive_losses:
            continue
        if (r.mkt_r6 if np.isfinite(r.mkt_r6) else 0) > 0 and len(same) >= gs.max_same_direction:
            continue
        x = {"symbol": r.symbol, "ts": r.ts, "side": r.side, "net": r.net,
             "exit_ts": r.ts + (int(r.exit_bar) + 1) * BAR}
        keep.append(x)
        closed.append(x)
        open_until[r.symbol] = x["exit_ts"]
        if r.net < 0:
            last_loss[r.symbol] = x["exit_ts"]
    sc2 = report("(c2) hour_close_first + today's limits", pd.DataFrame(keep))

    # ---- coverage check on the 30 days where the local database holds every coin's 5-minute candles
    out("\nCoverage check of the 5-minute fetch (13 Oct - 16 Nov 2026, every coin, no prefilter):")
    lc = sqlite3.connect(f"file:{LOCAL_DB}?mode=ro", uri=True)
    fetched = pd.read_sql_query("SELECT symbol, start, end FROM fetched WHERE tbl='k5'", conn)
    fw = {s: g[["start", "end"]].to_numpy() for s, g in fetched.groupby("symbol")}
    total = missed = 0
    for sym in [r[0] for r in lc.execute("SELECT DISTINCT symbol FROM klines")]:
        k5 = pd.read_sql_query("SELECT ts,o,h,l,c,v,qv,trades,tbqv FROM klines WHERE symbol=? ORDER BY ts", lc, params=(sym,))
        if len(k5) < 400:
            continue
        d = [x for x in detections(k5, sym, None, with_outcomes=False) if x["steady"]]
        total += len(d)
        w = fw.get(sym)
        for x in d:
            inside = w is not None and ((w[:, 0] <= x["ts"] - 30 * HOUR) & (w[:, 1] >= x["ts"] + 24 * HOUR)).any()
            missed += not inside
    out(f"  R3 detections at 5-minute ends: {total:,}; outside the fetched windows: {missed:,} "
        f"({missed / max(total, 1):.2%})")

    # ---- same answer as the live code? 300 random detections recomputed with scan_now's own steps
    rng = np.random.default_rng(25)
    loc_rows = []
    for sym in rng.choice([r[0] for r in lc.execute("SELECT DISTINCT symbol FROM klines")], 60, replace=False):
        k5 = pd.read_sql_query("SELECT ts,o,h,l,c,v,qv,trades,tbqv FROM klines WHERE symbol=? ORDER BY ts", lc, params=(sym,))
        if len(k5) >= 400:
            loc_rows += detections(k5, sym, None, with_outcomes=False)
    pick = [loc_rows[i] for i in rng.choice(len(loc_rows), min(300, len(loc_rows)), replace=False)] if loc_rows else []
    agree = 0
    for x in pick:
        old = G._hourly(ldb, x["symbol"], x["ts"] - 40 * DAY)
        recent = G._rolling_hours(ldb, x["symbol"], x["ts"])
        if old is None or recent is None:
            continue
        old = old[old.index + HOUR <= recent.index[0]]
        kk = pd.concat([old, recent])
        f = G.hourly_shape(kk)
        i, side = len(kk) - 1, x["side"]
        if not G.early_grind_mask(f.iloc[[i]], side)[0]:
            continue
        fr = f.iloc[i]
        r = {"r_24h": side * fr["r_24h"], "r2_24h": fr["r2_24h"], "hours_with24": fr["up24"] if side > 0 else fr["dn24"],
             "r2_6h": fr["r2_6h"], "hours_with6": fr["up6"] if side > 0 else fr["dn6"],
             "big6": fr["big_up6"] if side > 0 else fr["big_dn6"]}
        agree += bool(GS.steady(r, gs)) == x["steady"]
    out(f"  live-code check: {len(pick)} random detections recomputed with _rolling_hours + hourly_shape (as scan_now): "
        f"{agree} give the same early-grind and R3 answer")

    # ---- the record of each rule for the signal policy (lts/models/steady_history.json)
    h = GS.load_history() or {"rules": {}}
    if sa:
        h["rules"]["steady/rolling_5m"] = {
            "source": "steady rule checked every 5 minutes with re-sends, past year replay (DESIGN.md section 28)",
            "data": "research/parity.py variant (a)", "script": "research/parity.py",
            "plans": {GS.plan_name(PL): {"n": sa["n"], "n_days": sa["days"], "win": sa["win"], "net": sa["mean"],
                                         "se_day": sa["se_day"], "t_day": sa["t_day"], "months_pos": sa["months_pos"],
                                         "n_months": sa["n_months"], "per_day": sa["per_day"]}},
            "passed_bar": False, "bar_failures": ["not tested against the bar (descriptive replay)"]}
        GS.HISTORY_PATH.write_text(json.dumps(h, indent=1), encoding="utf-8")
        out("wrote steady_history.json (rule steady/rolling_5m)")
    out(f"done in {(time.time() - t_start) / 60:.0f} min")
