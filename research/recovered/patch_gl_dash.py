import pathlib, re

ROOT = pathlib.Path(r"D:\Leverage trading")


def sub(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:80])
        s = s.replace(a, b, 1)
    p.write_text(s, encoding="utf-8")


sub("lts/dashboard.py", [
    ('''        out["movers"] = kv.get("movers_latest")''',
     '''        out["movers"] = kv.get("movers_latest")
        out["grindlab"] = kv.get("grindlab_report")
        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='grind_live'"):
            live = q("SELECT symbol, ts, side, price, score, model_p, catalyst, checks, factors, decision FROM grind_live "
                     "WHERE ts>=? ORDER BY ts DESC, score DESC LIMIT 300", (now - 2 * DAY,))
            for r in live:
                r["checks"] = json.loads(r["checks"] or "{}")
                r["factors"] = json.loads(r["factors"] or "{}")
            out["grind_live"] = live
        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='grind_episodes'"):
            out["grind_episodes"] = q("SELECT symbol, side, start, end, move, hours, tags FROM grind_episodes "
                                      "WHERE start>=? ORDER BY start DESC LIMIT 400", (now - 7 * DAY,))'''),
])

p = ROOT / "lts/web/index.html"
s = p.read_text(encoding="utf-8")
s = s.replace('''    <button data-tab="movers">Daily movers</button>''', '''    <button data-tab="grinders">Grinders</button>''')
start = s.index('  <section id="tab-movers" hidden>')
end = s.index('  <section id="tab-system" hidden>')
s = s[:start] + '''  <section id="tab-grinders" hidden>
    <p class="muted">Grinders are coins that move steadily one way for hours (straight path, most hours the same way, no single
    candle doing the work). Every night LTS re-studies the last ~4 months: it finds every steady grind, marks the moments you
    could have entered, and compares the ones that kept going with lookalikes that fizzled. Every hour it checks which coins
    are grinding now and how many of those traits they have. The best are paper-traded (−3% stop, +6% target, 24h).</p>
    <div class="tiles" id="grindTiles"></div>
    <h2>Grinding right now (last 48h)</h2><div class="panel scroll" id="grindLive"></div>
    <h2>Grind paper trades</h2><div class="panel scroll" id="grindTrades"></div>
    <h2 id="labTitle">What successful grinders have in common</h2>
    <div id="labSummary" class="banner"></div>
    <div class="panel scroll" id="labTraits"></div>
    <h2>More traits = bigger moves? (unseen days, by number of traits)</h2><div class="panel scroll" id="labScore"></div>
    <h2>Other factors that separated winners from lookalikes in every period</h2><div class="panel scroll" id="labFactors"></div>
    <h2>Steady grinds, last 7 days</h2><div class="panel scroll" id="grindEpisodes"></div>
    <h2 id="moverTitle">Daily review: did LTS catch the day's steady movers?</h2>
    <div class="tiles" id="moverTiles"></div>
    <div class="panel scroll" id="moverLatest"></div>
    <h2>Review history</h2><div class="panel scroll" id="moverHistory"></div>
  </section>
''' + s[end:]
s = s.replace('''  ["signals", "candidates", "learning", "movers", "system"].forEach(''',
              '''  ["signals", "candidates", "learning", "grinders", "system"].forEach(''')
s = s.replace('''  const phaseNames = {backfill:"Downloading history", replay:"Rebuilding candidates", train:"Training first model", running:"Scanning"};''',
              '''  const phaseNames = {backfill:"Downloading history", replay:"Rebuilding candidates", train:"Training first model", grindlab:"Grind lab: first study", running:"Scanning"};''')
s = s.replace('''  $("moverTitle").textContent = mv && mv.day ? `Latest review: 24h to ${dt(mv.end_ts)} (${mv.n_grinders} of ${mv.n_coins} coins)` : "Latest review";''',
              '''  $("moverTitle").textContent = mv && mv.day ? `Daily review: 24h to ${dt(mv.end_ts)} — did LTS catch the ${mv.n_grinders} steady movers?` : "Daily review: did LTS catch the day's steady movers?";''')
anchor = '''  const item = (k, v) => `<div><b>${k}</b>${v}</div>`;
  const dd = d.data || {};'''
grind_js = '''  // ---- grinders tab ----
  const gl = d.grindlab || {}, gs = (d.stats || {}).grind || {n: 0};
  const live = d.grind_live || [];
  const now12 = live.filter(r => d.now - r.ts <= 12 * 3600000);
  $("grindTiles").innerHTML = [
    statTile("Grind paper trades (all)", gs),
    tile("Grinding now (12h)", now12.length, `${now12.filter(r => r.decision === "sent").length} paper-traded`),
    tile("Steady grinds per day", gl.episodes_per_day == null ? "–" : num(gl.episodes_per_day, 0), gl.days ? `over the last ${num(gl.days, 0)} days` : "study pending"),
    tile("Early grinds that kept going", gl.base_success == null ? "–" : pct(gl.base_success, 0), gl.target ? `+${pct(gl.target, 0)} before −${pct(gl.stop, 0)} within ${gl.horizon_h}h` : ""),
  ].join("");
  const traitChips = ck => Object.entries(ck || {}).map(([k, v]) => `<span class="chip ${v ? "good" : ""}" title="${esc(k)}">${v ? "✓" : "·"} ${esc(k.split(" ").slice(0, 4).join(" "))}</span>`).join(" ");
  table($("grindLive"), [["Time"],["Coin"],["Side"],["6h","num"],["24h","num"],["Traits","num"],["Model","num"],["Decision"],["Traits present"]],
    live.map(r => [dt(r.ts), `<b>${esc(r.symbol)}</b>${r.catalyst ? ' <span class="chip warn">⚡ news</span>' : ""}`, side(r.side),
      signed((r.factors.r_6h || 0) * 100, 1, "%"), signed((r.factors.r_24h || 0) * 100, 1, "%"),
      `<b>${r.score}</b>/${Object.keys(r.checks || {}).length}`, r.model_p == null ? "–" : pct(r.model_p, 0),
      r.decision === "sent" ? '<span class="chip accent">paper trade</span>' : esc((r.decision || "").replace("skipped:", "").replaceAll("_", " ")),
      `<div style="white-space:normal;min-width:420px">${traitChips(r.checks)}</div>`]), "No early grinds detected in the last 48 hours yet (checked every hour).");
  const gsig = (d.signals || []).filter(s => s.kind === "grind");
  table($("grindTrades"), sigHead, gsig.map(sigRow), "No grind paper trades yet.");
  if (gl.traits) {
    const held = gl.traits.filter(t => t.held).length;
    const cat = (gl.catalyst || {}).short || {};
    $("labTitle").textContent = `What successful grinders have in common (study of ${num(gl.days, 0)} days, rebuilt ${dt(gl.built_ts)})`;
    $("labSummary").innerHTML = `${(gl.n_episodes || 0).toLocaleString()} steady grinds (${gl.episodes_up} up, ${gl.episodes_down} down) and
      ${(gl.n_decisions || 0).toLocaleString()} moments where one was grinding early enough to enter. ${pct(gl.base_success, 0)} of those kept going.
      <b>${held} of ${gl.traits.length} traits held up on later days the study had not seen.</b>
      Honest caveat: these traits mostly predict that a coin will move <i>a lot</i> — the typical worst move grows as much as the best move — so on their own they
      have not been profitable after fees. The one directional pattern: down-grinds in the week after an Upbit listing or Binance futures launch
      (${cat.n || 0} cases on ${cat.coins || 0} coins, ${pct(cat.success, 0)} kept going, ${cat["x_0.06_0.03"] == null ? "–" : signed(cat["x_0.06_0.03"] * 100, 2, "%")} per trade) — promising but still a small sample.`;
    const p0 = v => v == null ? "–" : pct(v, 0);
    table($("labTraits"), [["Trait"],["Kept going: with","num"],["without","num"],["Unseen days: with","num"],["without","num"],["Cases (unseen)","num"],["Held up?"]],
      gl.traits.map(t => [esc(t.trait), p0(t.train_with), p0(t.train_without), `<b>${p0(t.test_with)}</b>`, p0(t.test_without), t.n_test_with,
        t.held ? '<span class="chip good">held</span>' : '<span class="chip">faded</span>']), "");
    table($("labScore"), [["Traits","num"],["Cases","num"],["Kept going","num"],["Typical best move","num"],["Typical worst move","num"],["+6%/−3%","num"],["+10%/−5%","num"],["Trailing","num"]],
      (gl.by_score || []).map(r => [r.score, r.n, p0(r.success), p0(r.best_med), p0(r.worst_med),
        r["x_0.06_0.03"] == null ? "–" : signed(r["x_0.06_0.03"] * 100, 2, "%"), r["x_0.1_0.05"] == null ? "–" : signed(r["x_0.1_0.05"] * 100, 2, "%"),
        r.x_trail == null ? "–" : signed(r.x_trail * 100, 2, "%")]), "");
    table($("labFactors"), [["Factor"],["Kept going: top fifth","num"],["bottom fifth","num"],["Periods agreeing"],["Typical value: winners","num"],["lookalikes","num"]],
      (gl.stable_factors || []).map(r => [esc(r.factor), p0(r.success_high), p0(r.success_low), esc(r.folds_agree), num(r.median_success, 3), num(r.median_fail, 3)]), "None yet.");
  } else {
    $("labSummary").textContent = "The first study runs during set-up and then every night.";
  }
  table($("grindEpisodes"), [["Started"],["Coin"],["Direction"],["Move","num"],["Hours","num"],["Sector"]],
    (d.grind_episodes || []).map(e => [dt(e.start), `<b>${esc(e.symbol)}</b>`, side(e.side), signed(e.side * e.move * 100, 1, "%"), e.hours, esc(e.tags || "")]),
    "The catalogue is rebuilt every night.");

'''
assert anchor in s
s = s.replace(anchor, grind_js + anchor, 1)
p.write_text(s, encoding="utf-8")
print("dashboard grinders tab added")
