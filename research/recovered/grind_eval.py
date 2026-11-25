import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
df = pd.read_parquet(r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad\grind_points.parquet")
DAY = 86_400_000
df["day"] = df.ts // DAY
split = df.ts.quantile(2 / 3)
df["half"] = np.where(df.ts < split, "first 20d", "last 10d")

def dedupe(d, hours=6):
    """one decision per coin per `hours` (keep the first) so overlapping windows don't count many times"""
    d = d.sort_values("ts"); keep = []; last = {}
    for i, (s, t) in enumerate(zip(d.symbol.to_numpy(), d.ts.to_numpy())):
        if t - last.get(s, -10 ** 18) >= hours * 3_600_000:
            keep.append(i); last[s] = t
    return d.iloc[keep]

def summary(d, name):
    if len(d) < 15:
        return dict(rule=name, n=len(d))
    t = lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0
    per_day = d.groupby("day")["b2.5_1.0"].mean()
    return dict(rule=name, n=len(d), days=d.day.nunique(),
                fwd12=d.fwd12.median(), fwd24=d.fwd24.median(), cont24=(d.fwd24 > 0).mean(),
                mfe_ge3=(d.mfe24 >= 0.03).mean(), mfe_ge5=(d.mfe24 >= 0.05).mean(),
                R_25x1=d["b2.5_1.0"].mean(), t_25x1=t(d["b2.5_1.0"]), R_25x2=d["b2.5_2.0"].mean(),
                R_15x2=d["b1.5_2.0"].mean(), R_fix3=d["fix3"].mean(), days_pos=(per_day > 0).mean())

GRIND = lambda d, r6=0.04, r2=0.6, er=0.25, w=4, big=0.45: d[(d.r6 >= r6) & (d.r2_6 >= r2) & (d.er_6 >= er) & (d.with6 >= w) & (d.big6 <= big)]
rules = {
    "all liquid coin-hours (baseline)": df,
    "moved >=4% in 6h (any shape)": df[df.r6 >= 0.04],
    "GRIND 6h (>=4%, R2>=.6, ER>=.25, 4/6 hrs, no 1h >45%)": GRIND(df),
    "GRIND strict (>=6%, R2>=.75, ER>=.3, 5/6 hrs)": GRIND(df, 0.06, 0.75, 0.30, 5, 0.40),
    "GRIND + also up over 12h & 24h": GRIND(df)[lambda d: (d.r12 > d.r6) & (d.r24 > 0)],
    "GRIND + OI rising with trend": GRIND(df)[lambda d: d.oi6 > 0.02],
    "GRIND + OI falling": GRIND(df)[lambda d: d.oi6 < -0.02],
    "GRIND + volume surge (rvol6>=2)": GRIND(df)[lambda d: d.rvol6 >= 2],
    "GRIND + quiet volume (rvol6<1.2)": GRIND(df)[lambda d: d.rvol6 < 1.2],
    "spiky mover (>=4%, biggest 1h >60%)": df[(df.r6 >= 0.04) & (df.big6 > 0.6)],
}
out = []
for name, d in rules.items():
    dd = dedupe(d)
    out.append(summary(dd, name))
    for side, g in dd.groupby("side"):
        out.append(summary(g, f"   {'long' if side > 0 else 'short'}"))
    for half, g in dd.groupby("half"):
        out.append(summary(g, f"   {half}"))
res = pd.DataFrame(out).set_index("rule")
fmt = {c: "{:.1%}".format for c in ("fwd12", "fwd24", "cont24", "mfe_ge3", "mfe_ge5", "days_pos")}
print(res.to_string(formatters=fmt, float_format=lambda x: f"{x:+.3f}"))
