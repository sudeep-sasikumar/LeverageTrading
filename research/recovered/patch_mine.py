def sub(p, old, new, n=1):
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == n, (old[:70], s.count(old))
    open(p, "w", encoding="utf-8").write(s.replace(old, new))


p = "lts/web/index.html"
sub(p, '''[hidden]{display:none!important}''', '''[hidden]{display:none!important}
.form{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px 12px;padding:14px}
.form label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:4px}
.form select,.form textarea{font:inherit;padding:9px 11px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--text);width:100%}
.form .wide{grid-column:1/-1}
.form .actions{grid-column:1/-1;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.form .actions button{width:auto;margin-top:0;padding:9px 16px}
button.link{font:inherit;font-size:12px;background:none;border:1px solid var(--line);border-radius:6px;padding:3px 8px;color:var(--text);cursor:pointer}
button.link:hover{border-color:var(--accent)}''')
sub(p, '''    <button data-tab="signals" class="on">Signals</button>''', '''    <button data-tab="signals" class="on">Signals</button>
    <button data-tab="mine">My trades</button>''')
sub(p, '''  <section id="tab-candidates" hidden>''', '''  <section id="tab-mine" hidden>
    <div class="tiles" id="mineTiles"></div>
    <h2 id="mineFormTitle">Log a trade</h2>
    <form class="panel form" id="tradeForm" autocomplete="off">
      <input type="hidden" id="tf_id">
      <label>Coin<input id="tf_symbol" placeholder="MEGAUSDT" required></label>
      <label>Side<select id="tf_side"><option value="long">Long</option><option value="short">Short</option></select></label>
      <label>Leverage<input id="tf_leverage" type="number" step="any" min="1" placeholder="5"></label>
      <label>Margin (USDT)<input id="tf_margin" type="number" step="any" min="0" placeholder="100"></label>
      <label>Entry price<input id="tf_entry_price" type="number" step="any" min="0" required></label>
      <label>Take profit<input id="tf_tp_price" type="number" step="any" min="0"></label>
      <label>Stop loss<input id="tf_sl_price" type="number" step="any" min="0"></label>
      <label>Opened at<input id="tf_open_ts" type="datetime-local"></label>
      <label>Opening fee (USDT)<input id="tf_open_fee" type="number" step="any" min="0"></label>
      <label>Status<select id="tf_status"><option value="open">Open</option><option value="closed">Closed</option></select></label>
      <label>Exit price<input id="tf_exit_price" type="number" step="any" min="0"></label>
      <label>Closed at<input id="tf_close_ts" type="datetime-local"></label>
      <label>Closing fee (USDT)<input id="tf_close_fee" type="number" step="any" min="0"></label>
      <label>Realized PnL before fees (USDT)<input id="tf_realized_pnl" type="number" step="any" placeholder="worked out if blank"></label>
      <label>Funding (USDT, + received)<input id="tf_funding" type="number" step="any"></label>
      <label>Exit reason<select id="tf_exit_reason"><option value="">–</option><option value="tp">Target</option><option value="sl">Stop</option><option value="manual">Closed by me</option><option value="liquidated">Liquidated</option></select></label>
      <label class="wide">Notes<textarea id="tf_notes" rows="2" maxlength="1000"></textarea></label>
      <div class="actions"><button class="primary" type="submit" id="tf_save">Save trade</button>
        <button class="link" type="button" id="tf_reset">Clear</button><span class="err" id="tf_err"></span></div>
    </form>
    <h2>My trades</h2><div class="panel scroll" id="mine"></div>
  </section>
  <section id="tab-candidates" hidden>''')
sub(p, '''  <p class="muted" style="margin-top:22px">Read-only. Refreshes''', '''  <p class="muted" style="margin-top:22px">Read-only except My trades. Refreshes''')
sub(p, '''  const item = (k, v) => `<div><b>${k}</b>${v}</div>`;''', '''  renderMine(d.my_trades || {trades: [], totals: {}});

  const item = (k, v) => `<div><b>${k}</b>${v}</div>`;''')
sub(p, '''async function load() {''', '''// ---- my trades ---------------------------------------------------------------
let MINE = [];
const usd = (x, d = 2) => x == null ? "–" : signed(x, d, " USDT");
const toLocalInput = ts => { if (!ts) return ""; const t = new Date(ts); t.setMinutes(t.getMinutes() - t.getTimezoneOffset()); return t.toISOString().slice(0, 16); };
const TF = ["symbol", "leverage", "margin", "entry_price", "tp_price", "sl_price", "open_fee", "status", "exit_price", "close_fee",
            "realized_pnl", "funding", "exit_reason", "notes"];
function renderMine(m) {
  MINE = m.trades || [];
  const t = m.totals || {};
  $("mineTiles").innerHTML = [
    tile("Net result", t.closed ? usd(t.net) : "–", t.closed ? `${t.closed} closed · ${t.open || 0} open` : "no closed trades yet"),
    tile("Won", t.closed ? `${t.wins}/${t.closed}` : "–", t.closed ? pct(t.wins / t.closed, 0) + " of closed trades" : ""),
    tile("Average on margin", t.margin_pct_avg == null ? "–" : signed(t.margin_pct_avg * 100, 1, "%"), "after fees and funding"),
    tile("Fees paid", t.n ? (t.fees || 0).toFixed(2) + " USDT" : "–", "opening + closing"),
  ].join("");
  table($("mine"), [["Opened"],["Coin"],["Side"],["Lev","num"],["Margin","num"],["Entry","num"],["Exit","num"],["Move","num"],
                    ["Gross","num"],["Fees","num"],["Net","num"],["On margin","num"],["Status"],["Signal"],["Notes"],[""]],
    MINE.map(r => [dt(r.open_ts), coin(r.symbol), side(r.side), r.leverage ? num(r.leverage, 0) + "x" : "–", r.margin == null ? "–" : num(r.margin, 2),
      price(r.entry_price), price(r.exit_price), r.move == null ? "–" : signed(r.move * 100, 2, "%"), usd(r.gross), (r.fees || 0).toFixed(4),
      usd(r.net), r.net_pct == null ? "–" : signed(r.net_pct * 100, 2, "%"),
      r.status === "open" ? '<span class="chip accent">open</span>' : ({tp: '<span class="chip good">target</span>', sl: '<span class="chip bad">stopped</span>'}[r.exit_reason] || '<span class="chip">closed</span>') + ` <span class="muted">${dt(r.close_ts)}</span>`,
      r.signal ? `<span class="chip accent">LTS ${dt(r.signal.ts)}</span>` : '<span class="muted">own pick</span>',
      `<span title="${esc(r.notes || "")}">${esc((r.notes || "").slice(0, 40))}</span>${r.source === "screenshot" ? ' <span class="muted">(from screenshot)</span>' : ""}`,
      `<button class="link" data-edit="${r.id}">Edit</button> <button class="link" data-del="${r.id}">Delete</button>`]),
    "No trades logged yet. Use the form above, or send Claude a screenshot of the exchange.");
}
function fillForm(r) {
  $("tf_id").value = r ? r.id : "";
  TF.forEach(k => { $("tf_" + k).value = r && r[k] != null ? r[k] : ""; });
  $("tf_side").value = r && r.side < 0 ? "short" : "long";
  $("tf_status").value = r ? r.status : "open";
  $("tf_open_ts").value = toLocalInput(r && r.open_ts); $("tf_close_ts").value = toLocalInput(r && r.close_ts);
  $("mineFormTitle").textContent = r ? `Edit trade #${r.id} (${r.symbol})` : "Log a trade";
  $("tf_err").textContent = "";
}
$("mine").addEventListener("click", async e => {
  const ed = e.target.dataset.edit, del = e.target.dataset.del;
  if (ed) { fillForm(MINE.find(r => String(r.id) === ed)); $("tradeForm").scrollIntoView({behavior: "smooth"}); }
  if (del) {
    const r = MINE.find(x => String(x.id) === del);
    if (!confirm(`Delete trade #${del} (${r ? r.symbol : ""})? This cannot be undone.`)) return;
    await fetch("api/my_trades/" + del, {method: "DELETE", credentials: "same-origin"}); load();
  }
});
$("tf_reset").addEventListener("click", () => fillForm(null));
$("tradeForm").addEventListener("submit", async e => {
  e.preventDefault();
  const body = {side: $("tf_side").value};
  TF.forEach(k => { body[k] = $("tf_" + k).value; });
  for (const k of ["open_ts", "close_ts"]) { const v = $("tf_" + k).value; body[k] = v ? new Date(v).getTime() : ""; }
  if (!body.open_ts && !$("tf_id").value) body.open_ts = Date.now();
  if (body.exit_price && body.status === "open") body.status = "closed";
  const id = $("tf_id").value;
  const r = await fetch("api/my_trades" + (id ? "/" + id : ""), {method: "POST", credentials: "same-origin",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const res = await r.json().catch(() => ({}));
  if (!r.ok) { $("tf_err").textContent = res.error || "Could not save"; return; }
  fillForm(null); load();
});

async function load() {''')
sub(p, '''  ["signals", "candidates", "learning", "grinders", "system"].forEach''', '''  ["signals", "mine", "candidates", "learning", "grinders", "system"].forEach''')
print("ok")
