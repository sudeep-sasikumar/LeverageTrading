def sub(p, old, new, n=1):
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == n, (p, old[:70], s.count(old))
    open(p, "w", encoding="utf-8").write(s.replace(old, new))


p = "lts/dashboard.py"
sub(p, '''def snapshot(db_path: Path) -> dict:''', '''ANAT_FACTORS = [  # (key, label, path in the anatomy JSON)
    ("mfe", "Best move before exit", ("mfe",)),
    ("to_peak", "Minutes to the peak", ("minutes_to_peak",)),
    ("peak_to_exit", "Minutes peak to exit", ("minutes_peak_to_exit",)),
    ("flow_run", "Taker flow, entry to peak", ("run", "taker_flow")),
    ("flow_turn", "Taker flow at the turn", ("turn", "taker_flow")),
    ("flow_fall", "Taker flow after the peak", ("fall", "taker_flow")),
    ("rvol_turn", "Volume at the turn (x before entry)", ("turn", "rvol")),
    ("rvol_fall", "Volume after the peak (x before entry)", ("fall", "rvol")),
    ("wick", "Wick against us on the peak candle", ("peak_wick_against",)),
    ("btc_fall", "BTC after the peak", ("fall", "btc")),
    ("mkt_fall", "Market (median coin) after the peak", ("fall", "mkt_med")),
    ("liq_against", "Liquidations against us after the peak ($)", ("fall", "liq_against_usd")),
    ("oi_run", "Open interest change, entry to peak", ("oi_chg_run",)),
    ("oi_fall", "Open interest change, peak to exit", ("oi_chg_fall",)),
    ("book_peak", "Order-book imbalance at the peak (1%)", ("book", "peak", "imb_1")),
    ("wall_peak", "Wall ahead at the peak (x median level)", ("book", "peak", "wall_ahead_x")),
    ("level_peak", "Nearest major level from the peak", ("peak_level", "dist_from_peak")),
]


def _dig(o, path):
    for k in path:
        if not isinstance(o, dict):
            return None
        o = o.get(k)
    return o if isinstance(o, (int, float)) and not isinstance(o, bool) else None


def anatomy_summary(rows: list[dict]) -> dict:
    """Reversals (in profit, then stopped) vs trades that reached the target vs other stops: medians of each factor,
    and the list of recent reversals with their main clues."""
    groups = {"reversal": [], "target": [], "stop": []}
    rev = []
    for r in rows:
        a = json.loads(r["anatomy"] or "null")
        if not a or a.get("exit") == "time":
            continue
        g = "reversal" if a.get("reversal") else a.get("exit")
        if g in groups:
            groups[g].append(a)
        if g == "reversal":
            x = {k: _dig(a, p) for k, _, p in ANAT_FACTORS}
            ex = a.get("minutes_to_exit")
            for clue, key in (("ema", "first_close_through_ema20_min"), ("entry", "first_close_back_at_entry_min"),
                              ("swing", "first_close_through_swing_min")):
                v = a.get(key)
                x[f"lead_{clue}"] = (ex - v) if (v is not None and ex is not None) else None
            x.update({"symbol": r["symbol"], "ts": r["ts"], "side": r["side"],
                      "level": (a.get("peak_level") or {}).get("what"), "news": len(a.get("announcements") or [])})
            rev.append(x)
    med = []
    for k, label, p in ANAT_FACTORS:
        row = {"factor": label, "key": k}
        for g, items in groups.items():
            v = sorted(x for x in (_dig(a, p) for a in items) if x is not None)
            row[g] = v[len(v) // 2] if v else None
        med.append(row)
    return {"n": {g: len(v) for g, v in groups.items()}, "medians": med, "reversals": rev[:300]}


def snapshot(db_path: Path) -> dict:''')
sub(p, '''                out["grind_all"] = allg''', '''                out["grind_all"] = allg
                if "anatomy" in cols_gl:
                    out["anatomy"] = anatomy_summary(q("SELECT symbol, ts, side, anatomy FROM grind_live WHERE anatomy IS NOT NULL "
                                                       "AND ts>=? ORDER BY ts DESC LIMIT 3000", (now - 7 * DAY,)))''')

p = "lts/web/index.html"
sub(p, '''    <h2>Grinding right now (last 48h) — traits</h2>''', '''    <h2 id="anatTitle">Went our way, then reversed to the stop — what changed (5x plan, last 7 days)</h2>
    <p class="muted">Every grind is followed from entry to its best point (the peak) to the exit. A reversal = at least +0.5% in profit, then the
    −1% stop. Medians compare reversals with trades that reached the target. Taker flow: +1 all takers pushing our way, −1 all against.
    Lead = minutes before the stop that the clue appeared (none = it did not appear before the stop).</p>
    <div class="panel scroll" id="anatMed"></div>
    <div class="panel scroll" id="anatRev" style="margin-top:12px"></div>
    <h2>Grinding right now (last 48h) — traits</h2>''')
sub(p, '''  table($("planStats"),''', '''  const an = d.anatomy || {n: {}, medians: [], reversals: []};
  const fmtF = (k, v) => v == null ? "–" : (["mfe", "btc_fall", "mkt_fall", "oi_run", "oi_fall", "level_peak"].includes(k) ? signed(v * 100, 2, "%") :
    ["to_peak", "peak_to_exit"].includes(k) ? num(v, 0) + " min" : k === "liq_against" ? "$" + Math.round(v).toLocaleString() :
    k.startsWith("rvol") || k === "wall_peak" ? num(v, 1) + "x" : signed(v, 2, ""));
  $("anatTitle").textContent = `Went our way, then reversed to the stop — what changed (5x plan, last 7 days: ${an.n.reversal || 0} reversals, ${an.n.target || 0} targets, ${an.n.stop || 0} other stops)`;
  table($("anatMed"), [["Factor (median)"],["Reversals","num"],["Reached target","num"],["Stopped without a run","num"]],
    an.medians.map(m => [esc(m.factor), fmtF(m.key, m.reversal), fmtF(m.key, m.target), fmtF(m.key, m.stop)]),
    "Filled in as grinds exit (needs the trade to reach its target or stop).");
  const lead = v => v == null ? '<span class="muted">none</span>' : num(v, 0) + " min";
  table($("anatRev"), [["Detected"],["Coin"],["Side"],["Best","num"],["To peak","num"],["Peak→stop","num"],["Flow run→fall"],["Volume after peak","num"],
                       ["BTC after peak","num"],["Market after peak","num"],["Liq. against","num"],["OI run / fall"],["Book at peak","num"],["Level near peak"],
                       ["Close back at entry"],["Close through EMA20"],["Broke run's swing"]],
    an.reversals.map(x => [dt(x.ts), coin(x.symbol), side(x.side), fmtF("mfe", x.mfe), fmtF("to_peak", x.to_peak), fmtF("peak_to_exit", x.peak_to_exit),
      `${fmtF("flow_run", x.flow_run)} → ${fmtF("flow_fall", x.flow_fall)}`, fmtF("rvol_fall", x.rvol_fall), fmtF("btc_fall", x.btc_fall), fmtF("mkt_fall", x.mkt_fall),
      fmtF("liq_against", x.liq_against), `${fmtF("oi_run", x.oi_run)} / ${fmtF("oi_fall", x.oi_fall)}`, fmtF("book_peak", x.book_peak),
      x.level ? `${esc(x.level)} (${fmtF("level_peak", x.level_peak)})` : "–", lead(x.lead_entry), lead(x.lead_ema), lead(x.lead_swing)]),
    "No reversals recorded yet.");
  table($("planStats"),''')
print("ok")
