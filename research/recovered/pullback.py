"""Pullback entry on the year's early grinds (63.5k events, Oct 2025 - Nov 2026).

At detection, instead of buying/selling at once, rest a limit order `dip` against the move, valid `wait` hours.
If filled: the trader's plan (target / stop) from the fill price, until 24h after detection.
Conservative fills: the fill bar can stop you out but cannot pay the target; a stop and target in the same later
bar counts as the stop. Fees: limit entry and target = maker 0.02%; stop and time exit = taker 0.08% + 0.04% slip.

Pre-registered: variants chosen on the first half of the months (discovery), judged only on the second half
(confirmation). A variant counts only if it is positive after fees in confirmation, day-clustered t >= 3, and
positive in >= 70% of confirmation months.
"""
import glob, itertools
import numpy as np, pandas as pd

OUT = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
MAKER, TAKER, SLIP = 0.0002, 0.0008, 0.0004
ev = pd.read_parquet(OUT + r"\aim_events.parquet")[["symbol", "ts", "side"]]
post = np.concatenate([np.load(f) for f in sorted(glob.glob(OUT + r"\paths\chunk_*.npy"))]).astype(np.float64)
assert len(post) == len(ev)
ok = np.isfinite(post[:, :, :4]).all((1, 2))
ev, post = ev[ok].reset_index(drop=True), post[ok]
side = ev.side.to_numpy()[:, None]
e0 = post[:, 0, 0][:, None]
# side-aligned moves relative to the detection price
fav = np.where(side > 0, post[:, :, 1] / e0 - 1, 1 - post[:, :, 2] / e0)    # best point in the bar
adv = np.where(side > 0, 1 - post[:, :, 2] / e0, post[:, :, 1] / e0 - 1)    # worst point in the bar
cls = np.where(side > 0, post[:, :, 3] / e0 - 1, 1 - post[:, :, 3] / e0)
N, B = adv.shape
month = pd.to_datetime(ev.ts, unit="ms").dt.strftime("%Y-%m").to_numpy()
day = (ev.ts // 86_400_000).to_numpy()
months = sorted(set(month))
disc = np.isin(month, months[:len(months) // 2])
conf = ~disc
PLANS = {"10x +1/-0.5": (0.01, 0.005), "5x +2/-1": (0.02, 0.01), "3x +3.33/-1": (0.0333, 0.01), "2x +5/-0.75": (0.05, 0.0075)}
LEV = {"10x +1/-0.5": 10, "5x +2/-1": 5, "3x +3.33/-1": 3, "2x +5/-0.75": 2}


def run(dip, wait_h, target, stop):
    """Net % per event (NaN = not filled) on price returns relative to the FILL price."""
    w = int(wait_h * 12)
    if dip == 0:
        fill_i = np.zeros(N, int); filled = np.ones(N, bool); entry_cost = TAKER + SLIP
    else:
        hit = adv[:, :w] >= dip
        filled = hit.any(1); fill_i = hit.argmax(1); entry_cost = MAKER
    f = 1 - dip                                           # fill price relative to detection (side-aligned)
    # moves from the fill price: (1+fav)/(1-dip) - 1 for the favourable side, etc.
    fav_f = (1 + fav) / f - 1
    adv_f = 1 - (1 - adv) / f
    cls_f = (1 + cls) / f - 1
    idx = np.arange(B)[None, :]
    after = idx > fill_i[:, None]                          # bars after the fill bar
    in_fill = idx == fill_i[:, None]
    # in the fill bar only the stop can trigger (if the bar went that far past the limit)
    stop_hit = ((adv_f >= stop) & (after | in_fill))
    tgt_hit = (fav_f >= target) & after
    i_s = np.where(stop_hit.any(1), stop_hit.argmax(1), 10 ** 6)
    i_t = np.where(tgt_hit.any(1), tgt_hit.argmax(1), 10 ** 6)
    net = np.where((i_s <= i_t) & (i_s < 10 ** 6), -stop - entry_cost - TAKER - SLIP,
                   np.where(i_t < 10 ** 6, target - entry_cost - MAKER, cls_f[:, -1] - entry_cost - TAKER - SLIP))
    if dip > 0:
        net = np.where(filled, net, np.nan)
    return net


def stats(net, mask):
    v = net[mask]; fin = np.isfinite(v)
    if fin.sum() < 50:
        return None
    d = pd.Series(v[fin]).groupby(day[mask][fin]).mean()
    m = pd.Series(v[fin]).groupby(month[mask][fin]).mean()
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan
    return {"trades": int(fin.sum()), "fill": fin.mean(), "win": (v[fin] > 0).mean(), "net": v[fin].mean(),
            "t_day": t, "months_pos": (m > 0).mean(), "per_day": fin.sum() / len(np.unique(day[mask]))}


rows = []
for (pname, (tg, st)), dip, wait in itertools.product(PLANS.items(), [0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02],
                                                         [1, 2, 4, 8]):
    if dip == 0 and wait != 1:
        continue
    net = run(dip, wait, tg, st)
    for part, mask in (("disc", disc), ("conf", conf)):
        s = stats(net, mask)
        if s:
            rows.append({"plan": pname, "dip": dip, "wait_h": wait if dip else 0, "part": part, **s})
r = pd.DataFrame(rows)
r["margin"] = r.net * r.plan.map(LEV)
pd.set_option("display.width", 220, "display.max_rows", 400)
fmt = {"fill": "{:.0%}", "win": "{:.1%}", "net": "{:+.3%}", "t_day": "{:+.1f}", "months_pos": "{:.0%}", "per_day": "{:.1f}", "margin": "{:+.2%}"}
print(f"events {N:,}, months {months[0]}..{months[-1]}; discovery {months[:len(months)//2][-1]} and before")
best = []
for pname in PLANS:
    d = r[(r.plan == pname) & (r.part == "disc")].sort_values("net", ascending=False)
    base = r[(r.plan == pname) & (r.dip == 0)]
    top = d.head(3)
    print(f"\n=== {pname} ===")
    show = pd.concat([base, r[(r.plan == pname) & r.set_index(["dip", "wait_h"]).index.isin(list(zip(top.dip, top.wait_h)))]])
    show = show.drop_duplicates()
    out = show[["dip", "wait_h", "part", "trades", "per_day", "fill", "win", "net", "margin", "t_day", "months_pos"]].copy()
    for c, f in fmt.items():
        if c in out:
            out[c] = out[c].map(f.format)
    print(out.to_string(index=False))
    for _, b in top.iterrows():
        c = r[(r.plan == pname) & (r.dip == b.dip) & (r.wait_h == b.wait_h) & (r.part == "conf")].iloc[0]
        best.append((pname, b.dip, b.wait_h, c.net, c.t_day, c.months_pos, c.net > 0 and c.t_day >= 3 and c.months_pos >= 0.7))
print("\n=== pre-registered verdict (top 3 per plan from discovery, judged on confirmation) ===")
for b in best:
    print(f"{b[0]:<12} dip {b[1]:.2%} wait {b[2]}h: conf {b[3]:+.3%}/trade, t {b[4]:+.1f}, months+ {b[5]:.0%} -> {'PASS' if b[6] else 'fail'}")
r.to_csv(OUT + r"\pullback_grid.csv", index=False)
# side split for the best 5x variant
