import pathlib

p = pathlib.Path(r"D:\Leverage trading\lts\web\index.html")
s = p.read_text(encoding="utf-8")
pairs = [
    ('''    <button data-tab="system">System</button>''',
     '''    <button data-tab="movers">Daily movers</button>
    <button data-tab="system">System</button>'''),
    ('''  <section id="tab-system" hidden>''',
     '''  <section id="tab-movers" hidden>
    <p class="muted">Each day LTS looks back at the coins that moved 10%+ in a steady line (straight path, most
    hours moving the same way, no single candle doing the work), records what their data looked like, and checks
    whether it flagged them in time. This is the record used to judge and improve the grind setup.</p>
    <div class="tiles" id="moverTiles"></div>
    <h2 id="moverTitle">Latest review</h2><div class="panel scroll" id="moverLatest"></div>
    <h2>History</h2><div class="panel scroll" id="moverHistory"></div>
  </section>
  <section id="tab-system" hidden>'''),
    ('''    statTile("Paper signals (all)", st.paper), statTile("Real signals (all)", st.real),''',
     '''    statTile("Paper signals (all)", st.paper), statTile("Real signals (all)", st.real),
    statTile("Momentum setup (all)", st.momentum), statTile("Grind setup (all, paper)", st.grind),'''),
    ('''  const sigRow = s => [dt(s.ts), `<b>${esc(s.symbol)}</b>`, side(s.side), grade(s.grade),''',
     '''  const sigRow = s => [dt(s.ts), `<b>${esc(s.symbol)}</b>`, side(s.side), grade(s.grade) + (s.kind === "grind" ? ' <span class="chip">grind</span>' : ""),'''),
    ('''  ["signals", "candidates", "learning", "system"].forEach(t => $("tab-" + t).hidden = t !== b.dataset.tab);''',
     '''  ["signals", "candidates", "learning", "movers", "system"].forEach(t => $("tab-" + t).hidden = t !== b.dataset.tab);'''),
    ('''  const item = (k, v) => `<div><b>${k}</b>${v}</div>`;
  const dd = d.data || {};''',
     '''  const ms = d.movers_stats || {n: 0};
  $("moverTiles").innerHTML = [
    tile("Steady movers recorded (30d)", ms.n, "10%+ in a steady line"),
    tile("Flagged by LTS in time", ms.n ? pct(ms.flagged / ms.n, 0) : "–", `${ms.flagged || 0} of ${ms.n}`),
    tile("Move still ahead when flagged", ms.median_left == null ? "–" : pct(ms.median_left, 0), "median"),
    tile("Got a signal", ms.n ? pct(ms.signalled / ms.n, 0) : "–", `${ms.signalled || 0} of ${ms.n}`),
  ].join("");
  const mv = d.movers;
  $("moverTitle").textContent = mv && mv.day ? `Latest review: 24h to ${dt(mv.end_ts)} (${mv.n_grinders} of ${mv.n_coins} coins)` : "Latest review";
  const flagCell = m => m.flagged_ts ? `${dt(m.flagged_ts)} <span class="muted">(${esc(m.flagged_kind)})</span>` : '<span class="muted">not flagged</span>';
  const leftCell = m => m.flagged_left == null ? "–" : pct(m.flagged_left, 0);
  const sigCell = m => m.signal_id ? `#S${m.signal_id} ${statusChip(m.signal_status)}` : "–";
  table($("moverLatest"), [["Coin"],["Move 24h","num"],["Straightness R²","num"],["Hours with trend","num"],["Volume vs usual","num"],["Open interest 24h","num"],["Flagged"],["Move left","num"],["Signal"]],
    (mv && mv.movers || []).map(m => [`<b>${esc(m.symbol)}</b>`, signed(m.ret24 * 100, 1, "%"), num(m.r2, 2), `${m.hours_with}/24`,
      m.rvol24 == null ? "–" : num(m.rvol24, 1) + "x", m.oi_chg24 == null ? "–" : signed(m.oi_chg24 * m.side * 100, 0, "%"),
      flagCell(m), leftCell(m), sigCell(m)]), "No review yet. It runs with the daily report.");
  table($("moverHistory"), [["Day"],["Coin"],["Move 24h","num"],["R²","num"],["Flagged"],["Move left","num"],["Signal"]],
    (d.movers_history || []).map(m => [esc(m.day), `<b>${esc(m.symbol)}</b>`, signed(m.ret24 * 100, 1, "%"), num(m.r2, 2),
      flagCell(m), leftCell(m), sigCell(m)]), "No history yet.");

  const item = (k, v) => `<div><b>${k}</b>${v}</div>`;
  const dd = d.data || {};'''),
]
for a, b in pairs:
    assert a in s, a[:60]
    s = s.replace(a, b, 1)
p.write_text(s, encoding="utf-8")
print("html patched")
