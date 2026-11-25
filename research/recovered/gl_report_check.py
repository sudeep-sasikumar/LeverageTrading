import sys, json
sys.path.insert(0, r"D:\Leverage trading")
import pandas as pd
from lts import grindlab as G
S = r"C:\Users\royro\AppData\Local\Temp\claude\4b34ed9f761a19f1\218d1ecf-89cc-4e48-923b-cb405714705a\scratchpad"
df, ep = pd.read_parquet(S + r"\gl_decisions.parquet"), pd.read_parquet(S + r"\gl_episodes.parquet")
rep = G.study(df, ep)
print(f"{rep['n_decisions']} decisions over {rep['days']:.0f}d, base success {rep['base_success']:.1%}; "
      f"episodes {rep['n_episodes']} ({rep['episodes_per_day']:.0f}/day)")
print("traits that held on unseen days:", sum(t["held"] for t in rep["traits"]), "of", len(rep["traits"]))
for t in rep["traits"]:
    mark = "held " if t["held"] else "faded"
    print(f"  {mark} {t['trait']:<62} train {t['train_with']:.0%} vs {t['train_without']:.0%} | "
          f"unseen {t['test_with']:.0%} vs {t['test_without']:.0%} (n={t['n_test_with']})")
for k, v in rep["catalyst"].items():
    print(f"catalyst {k}: n={v['n']} coins={v['coins']} success {v['success']:.1%} "
          f"+6/-3 net {v['x_0.06_0.03'] * 100:+.2f}% (t={v['x_0.06_0.03_t']:+.1f})")
print("model:", round(rep["model"].get("auc", 0), 3), rep["model"].get("top10"))
json.dumps(rep, default=float)
print("report serialisable")
