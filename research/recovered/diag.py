"""Feature stability study on the TP-first LABEL (not net R, which is mechanically tied to stop size)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, r"D:\Leverage trading")
from lts.config import load_config
from lts.db import DB
from lts.learner import load_labelled
from lts.features import feature_names

cfg = load_config(r"D:\Leverage trading\config.yaml"); db = DB(cfg.path(cfg.storage.db_path))
df = load_labelled(db)
DAY = 86_400_000
df["day"] = df["ts"] // DAY
df["bucket"] = df["ts"] // (4 * 3_600_000)
df["gross_r"] = np.where(df.outcome == "tp", 2.0, np.where(df.outcome == "sl", -1.0, np.nan))
df["y_dm"] = df["label"] - df.groupby("bucket")["label"].transform("mean")
pd.set_option("display.width", 220)
print("rows", len(df), "days", df["day"].nunique(), "symbols", df["symbol"].nunique(),
      "| TP", round(df.label.mean(), 3), "net R", round(df.ret_r.mean(), 3))

def daily_ic(col, target):
    out = []
    for _, g in df.groupby("day"):
        x = g[[col, target]].dropna()
        if len(x) >= 30 and x[col].nunique() > 5 and x[target].nunique() > 1:
            out.append(x[col].rank().corr(x[target].rank()))
    out = np.array(out)
    if len(out) < 8:
        return None
    return out.mean(), out.mean() / (out.std(ddof=1) / np.sqrt(len(out))), (out > 0).mean(), len(out)

rows = []
for col in feature_names():
    a, b = daily_ic(col, "label"), daily_ic(col, "y_dm")
    if a and b:
        rows.append((col, a[0], a[1], a[2], b[0], b[1], b[2], a[3]))
res = pd.DataFrame(rows, columns=["feature", "ic", "t", "pos_days", "ic_neutral", "t_neutral", "pos_days_n", "days"])
print("\nTop features by |t| of market-neutral daily IC on the LABEL:")
print(res.reindex(res["t_neutral"].abs().sort_values(ascending=False).index).head(28).round(3).to_string(index=False))

def table(title, key):
    g = df.groupby(key).agg(n=("label", "size"), tp=("label", "mean"), net_r=("ret_r", "mean"),
                            days=("day", "nunique")).round(3)
    print(f"\n{title}\n{g.to_string()}")

df["side_s"] = np.where(df.side > 0, "long", "short")
for col, name in (("btc_r_3d", "side x BTC 3d trend aligned"), ("mkt_r_3d", "side x median-alt 3d aligned"),
                  ("btc_r_24h", "side x BTC 24h aligned"), ("mkt_r_24h", "side x median-alt 24h aligned")):
    if col in df:
        df["_a"] = np.where(df[col] > 0, "with", "against")
        table(name, ["_a"])
if "impulse_same_1h" in df:
    df["_q"] = pd.qcut(df["impulse_same_1h"], [0, .5, .8, .95, 1], labels=["low", "mid", "high", "extreme"], duplicates="drop")
    table("market impulse (share of coins with |z_1h|>2, same side)", ["_q"])
    df["_q"] = pd.qcut(df["impulse_opp_1h"], [0, .5, .8, .95, 1], labels=["low", "mid", "high", "extreme"], duplicates="drop")
    table("market impulse, opposite side", ["_q"])
df["_q"] = pd.qcut(df["sl_pct"], 5, duplicates="drop")
table("by stop distance (cost drag shows in net R)", ["_q"])
df["_q"] = pd.qcut(df["log_qv_24h"], 5, duplicates="drop")
table("by 24h dollar volume (log10)", ["_q"])
table("by side", ["side_s"])
