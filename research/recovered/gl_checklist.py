import sys, numpy as np, pandas as pd
sys.path.insert(0, r"D:\Leverage trading")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
S = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
df = pd.read_parquet(S + r"\gl_decisions.parquet").sort_values("ts")
DAY = 86_400_000
cut = df.ts.quantile(0.6)
train, test = df[df.ts < cut - DAY], df[df.ts >= cut]
med = lambda c: train[c].median()

def checklist(d):
    items = {
        "BTC week same way": d.btc_r168 > 0,
        "broke/top of 30d range": (d.brk_30d == 1) | (d.pos_30d >= 0.8),
        "coin's week same way": d.week_ret > 0,
        "stronger than market 24h": d.rs_mkt24 > 0,
        "no heavy aggressive flow": d.imb_6h <= med("imb_6h"),
        "market NOT moved same way 6h": d.mkt_r6 <= med("mkt_r6"),
        "no daily level within 10%": d.levels_ahead_10pct == 0,
        "sector same way": d.sector_r24.fillna(0) > 0,
        "funding not against": d.funding.fillna(0) >= 0,
        "younger listing": d.listing_age_d.fillna(9999) <= med("listing_age_d"),
        "volume surge 6h": d.rvol_6h >= med("rvol_6h"),
        "Upbit listing <7d": d.ann_upbit_list_7d == 1,
    }
    return pd.DataFrame(items)

exits = [c for c in df.columns if c.startswith("x_")]
def summarise(g):
    r = {"n": len(g), "per_day": len(g) / max((test.ts.max() - test.ts.min()) / DAY, 1), "success_6_3": g.success.mean(),
         "fwd24_med": g.fwd24.median(), "mfe_ge6": (g.mfe >= 0.06).mean(), "mfe_ge10": (g.mfe >= 0.10).mean()}
    for e in exits:
        r[e] = g[e].mean()
    return pd.Series(r)

ck_test = checklist(test)
test = test.assign(score=ck_test.sum(axis=1))
print("Each trait on the HELD-OUT 40% (never used to set thresholds): success with vs without")
for name in ck_test.columns:
    a, b = test[ck_test[name]], test[~ck_test[name]]
    print(f"  {name:<32} {a.success.mean():5.1%} vs {b.success.mean():5.1%}  (with: n={len(a)})")
print(f"\nheld-out decisions {len(test)}, base success {test.success.mean():.1%}")
tab = test.groupby(np.minimum(test.score, 9)).apply(summarise)
fmt = {c: "{:.1%}".format for c in ("success_6_3", "fwd24_med", "mfe_ge6", "mfe_ge10")}
fmt |= {e: (lambda v: f"{v * 100:+.2f}%") for e in exits}
fmt |= {"n": "{:.0f}".format, "per_day": "{:.1f}".format}
print("\nBy checklist score (avg net return per trade, % of price, for each exit):")
print(tab.to_string(formatters=fmt))
for side, g in test.groupby("side"):
    hi = g[g.score >= 7]
    print(f"\nscore>=7 {'longs' if side > 0 else 'shorts'}: n={len(hi)} success {hi.success.mean():.1%}  " +
          "  ".join(f"{e}: {hi[e].mean() * 100:+.2f}%" for e in exits))
# per month stability for score>=7
test["month"] = pd.to_datetime(test.ts, unit="ms").dt.strftime("%Y-%m")
print("\nscore>=7 by month:")
print(test[test.score >= 7].groupby("month").apply(lambda g: pd.Series({"n": len(g), "success": g.success.mean(),
      "x_trail": g.x_trail.mean() * 100, "x_0.1_0.05": g["x_0.1_0.05"].mean() * 100, "x_0.06_0.03": g["x_0.06_0.03"].mean() * 100})).round(2).to_string())
best = test.sort_values(["score", "r_6h"], ascending=False).groupby(test.ts // DAY).head(1)
print(f"\nbest checklist pick per day (held-out): n={len(best)}, success {best.success.mean():.1%}, " +
      "  ".join(f"{e}: {best[e].mean() * 100:+.2f}%" for e in exits))
