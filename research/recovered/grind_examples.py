"""What did the user's example 'gradual trend' coins look like over the last 24h, vs everything else?"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, r"D:\Leverage trading")
from lts.config import load_config
from lts.db import DB
from lts import features as F

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
cfg = load_config(r"D:\Leverage trading\config.yaml"); db = DB(cfg.path(cfg.storage.db_path))
H, D = 12, 288
last = max(db.last_kline_ts().values())
end = last + F.BAR_MS
since = end - 9 * 86_400_000
EX = {"AIOTUSDT": -1, "SOPHUSDT": -1, "BTWUSDT": -1, "ZILUSDT": -1,
      "BLESSUSDT": 1, "SPACEUSDT": 1, "CROSSUSDT": 1, "PIEVERSEUSDT": 1}

def path_stats(k: pd.DataFrame) -> dict:
    """k: regularized 5m klines; window = last 24h."""
    w = k.iloc[-D:]
    c = w["c"].to_numpy(); o0 = w["o"].iloc[0]
    ret = c[-1] / o0 - 1
    s = np.sign(ret) or 1
    c15 = w["c"].to_numpy()[2::3]                       # 15m closes
    d15 = np.diff(np.log(np.concatenate([[o0], c15])))
    er15 = abs(d15.sum()) / np.abs(d15).sum()
    c1h = np.concatenate([[o0], w["c"].to_numpy()[11::12]])
    d1h = np.diff(np.log(c1h))
    with_trend_1h = float(np.mean(np.sign(d1h) == s))
    biggest_1h_share = float(np.max(s * d1h) / max(abs(np.log(c[-1] / o0)), 1e-9))
    # deepest counter-trend pullback, as a fraction of the move achieved at that time
    lp = np.log(np.concatenate([[o0], c])) * s
    run_max = np.maximum.accumulate(lp)
    pull = (run_max - lp)
    progress = run_max - lp[0]
    worst_pull = float(np.max(np.where(progress > 0.03, pull / np.maximum(progress, 1e-9), 0)))
    t = np.arange(len(lp)); r2 = np.corrcoef(t, lp)[0, 1] ** 2
    # how much of the 24h move was still ahead after the first 6h / 12h
    left_6 = 1 - (lp[6 * H] - lp[0]) / max(lp[-1] - lp[0], 1e-9)
    left_12 = 1 - (lp[12 * H] - lp[0]) / max(lp[-1] - lp[0], 1e-9)
    return dict(ret24=ret, er15=er15, r2=r2, hours_with=with_trend_1h, biggest_1h=biggest_1h_share,
                worst_pull=worst_pull, left_after_6h=left_6, left_after_12h=left_12)

def flow_stats(sym: str, k: pd.DataFrame, s: int) -> dict:
    w = k.iloc[-D:]
    qv24 = w["qv"].sum(); qv_prev = k["qv"].iloc[-8 * D:-D].sum() / 7
    imb = (2 * w["tbqv"].sum() - qv24) / qv24
    vol_share_top3h = np.sort(w["qv"].to_numpy().reshape(24, 12).sum(1))[-3:].sum() / qv24
    oi = db.frame("SELECT ts, oi FROM oi WHERE symbol=? AND ts>=? ORDER BY ts", [sym, end - 2 * 86_400_000])
    def oi_at(ts):
        x = oi[oi.ts <= ts]
        return x.oi.iloc[-1] if len(x) else np.nan
    oi_chg = oi_at(end) / oi_at(end - 86_400_000) - 1
    fund = db.frame("SELECT ts, rate FROM funding WHERE symbol=? AND ts>=? ORDER BY ts", [sym, end - 86_400_000])
    lsr = db.frame("SELECT ts, top_pos, global_acc FROM lsr WHERE symbol=? AND ts>=? ORDER BY ts", [sym, end - 86_400_000 - 3_600_000])
    return dict(qv24_musd=qv24 / 1e6, rvol24=qv24 / qv_prev if qv_prev else np.nan, taker_imb=imb * s,
                vol_top3h_share=vol_share_top3h, oi_chg24=oi_chg * s, fund_avg=fund.rate.mean() * s if len(fund) else np.nan,
                glob_lsr_chg=(np.log(lsr.global_acc.iloc[-1]) - np.log(lsr.global_acc.iloc[0])) * s if len(lsr) > 1 else np.nan,
                top_lsr_chg=(np.log(lsr.top_pos.iloc[-1]) - np.log(lsr.top_pos.iloc[0])) * s if len(lsr) > 1 else np.nan)

rows = []
active = [r["symbol"] for r in db.query("SELECT symbol FROM symbols WHERE active=1 AND src='binance'")]
for sym in active:
    k = db.klines(sym, since)
    if len(k) < 3 * D:
        continue
    k = F.regularize(k)
    if k.index[-1] != last:
        continue
    ps = path_stats(k)
    rows.append(dict(symbol=sym, **ps))
u = pd.DataFrame(rows).set_index("symbol")
u["abs24"] = u.ret24.abs()
print(f"window: 24h to {pd.to_datetime(end, unit='ms')} UTC, {len(u)} coins")
btc = u.loc["BTCUSDT"]; print(f"BTC 24h {btc.ret24:+.2%}, ETH {u.loc['ETHUSDT'].ret24:+.2%}, median coin {u.ret24.median():+.2%}")
print(f"coins moving >10%: {(u.abs24 > .10).sum()}, >20%: {(u.abs24 > .20).sum()}")

ex = []
for sym, side in EX.items():
    k = F.regularize(db.klines(sym, since))
    ex.append(dict(symbol=sym, side=side, **path_stats(k), **flow_stats(sym, k, side)))
ex = pd.DataFrame(ex).set_index("symbol")
print("\nYOUR EXAMPLES (directional numbers are in the trend's favour: + = with the trend)")
print(ex.round(3).to_string())

big = u[u.abs24 > 0.10]
print("\nALL COINS MOVING >10% IN 24H: path shape")
cols = ["ret24", "er15", "r2", "hours_with", "biggest_1h", "worst_pull", "left_after_6h", "left_after_12h"]
print(big.sort_values("er15", ascending=False)[cols].round(3).to_string())
