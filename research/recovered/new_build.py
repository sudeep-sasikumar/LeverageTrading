def load_context(db: DB, since: int) -> dict:
    """Everything shared across coins: market/sector medians per hour, tags, announcements, funding, OI.
    Built in a narrow first pass (3 columns per coin) so the nightly study fits a small VPS."""
    syms = [r["symbol"] for r in db.query("SELECT DISTINCT symbol FROM klines_1h")]
    tags = {}
    for r in db.query("SELECT symbol, tag FROM symbol_tags"):
        tags.setdefault(r["symbol"], set()).add(r["tag"])
    r24, r6 = {}, {}
    btc = None
    for sym in syms:
        k = _hourly(db, sym, since)
        if k is None:
            continue
        c, qv = k["c"], k["qv"]
        liquid = qv.rolling(24).sum() >= MIN_QV24
        r24[sym] = (c / c.shift(24) - 1).where(liquid)
        r6[sym] = (c / c.shift(6) - 1).where(liquid)
        if sym == "BTCUSDT":
            btc = pd.DataFrame({"r_6h": c / c.shift(6) - 1, "r_24h": c / c.shift(24) - 1, "r_168h": c / c.shift(168) - 1})
    w24, w6 = pd.DataFrame(r24), pd.DataFrame(r6)
    sector24 = {}
    for tag in MAIN_TAGS:
        members = [s for s in w24.columns if tag in tags.get(s, set())]
        if len(members) >= 4:
            sector24[tag] = w24[members]
    ann = db.frame("SELECT source, ts, base, kind FROM announcements")
    oi = db.frame("SELECT symbol, ts, oi FROM oi WHERE ts>=?", [since])
    lsr = db.frame("SELECT symbol, ts, top_pos, global_acc FROM lsr WHERE ts>=?", [since])
    fund = db.frame("SELECT symbol, ts, rate FROM funding WHERE ts>=?", [since - 3 * DAY])
    return {"syms": list(w24.columns), "tags": tags, "mkt24": w24.median(axis=1), "mkt6": w6.median(axis=1),
            "breadth": (w24 > 0).sum(axis=1) / w24.notna().sum(axis=1).replace(0, np.nan),
            "sector24": sector24, "btc": btc,
            "meta": {r["symbol"]: dict(r) for r in db.query("SELECT symbol, base FROM symbols")},
            "ann": {b: g.sort_values("ts") for b, g in ann.groupby("base")} if len(ann) else {},
            "oi": {s: g.sort_values("ts") for s, g in oi.groupby("symbol")},
            "lsr": {s: g.sort_values("ts") for s, g in lsr.groupby("symbol")},
            "fund": {s: g.sort_values("ts") for s, g in fund.groupby("symbol")}}


ANN_KINDS = ("list", "futures_launch", "delist", "monitoring", "alpha", "airdrop", "upbit_list", "upbit_warning",
             "upbit_delist")


def decision_row(sym: str, side: int, i: int, k: pd.DataFrame, f: pd.DataFrame, d: pd.DataFrame, ctx: dict) -> dict:
    """All factors for an early-grind decision at hourly bar i (values known at that bar's close).
    The nightly study and the live detector both use this function."""
    idx = k.index.to_numpy()
    t_close = int(idx[i] + HOUR)
    price = float(k["c"].iloc[i])
    fr = f.iloc[i]
    row = {"symbol": sym, "ts": t_close, "side": side, "price": price}
    for col in ("r_1h", "r_3h", "r_6h", "r_12h", "r_24h", "r_72h", "r_168h", "r_720h", "imb_6h", "imb_24h", "d_vwap24"):
        row[col] = side * fr[col]
    for col in ("r2_6h", "r2_24h", "atr_1h", "vol_exp", "rv_24h", "coil_before", "bbw_pct", "rvol_6h", "rvol_24h",
                "vol_trend_3d", "trades_rvol_6h", "ats_ratio_6h", "hour", "dow"):
        row[col] = fr[col]
    row["hours_with6"] = fr["up6"] if side > 0 else fr["dn6"]
    row["hours_with24"] = fr["up24"] if side > 0 else fr["dn24"]
    row["big6"] = fr["big_up6"] if side > 0 else fr["big_dn6"]
    row["upvol_share_24h"] = fr["upvol_share_24h"] if side > 0 else 1 - fr["upvol_share_24h"]
    for name in ("2d", "7d", "30d"):
        p = fr[f"pos_{name}"]
        row[f"pos_{name}"] = p if side > 0 else 1 - p
    hi7, lo7, hi2, lo2 = fr["hi_7d_prior"], fr["lo_7d_prior"], fr["hi_2d_prior"], fr["lo_2d_prior"]
    row["brk_2d_4h"] = float(price > hi2) if side > 0 else float(price < lo2)
    row["brk_7d_4h"] = float(price > hi7) if side > 0 else float(price < lo7)
    row["log_qv24"] = math.log10(max(fr["qv_24h"], 1.0)) if np.isfinite(fr["qv_24h"]) else np.nan
    row["log_price"] = math.log10(max(price, 1e-12))
    atr_abs = fr["atr_1h"] * price
    if np.isfinite(atr_abs) and atr_abs > 0:
        row.update(daily_levels(d, t_close, price, side, atr_abs))
    ts_i = idx[i]
    row["mkt_r24"] = side * ctx["mkt24"].get(ts_i, np.nan)
    row["mkt_r6"] = side * ctx["mkt6"].get(ts_i, np.nan)
    b = ctx["breadth"].get(ts_i, np.nan)
    row["breadth"] = b if side > 0 else 1 - b
    btc = ctx["btc"]
    if btc is not None and ts_i in btc.index:
        row["btc_r6"], row["btc_r24"], row["btc_r168"] = (side * btc.at[ts_i, c] for c in ("r_6h", "r_24h", "r_168h"))
    row["rs_mkt24"] = row["r_24h"] - row["mkt_r24"] if np.isfinite(row["mkt_r24"]) else np.nan
    my_tags = ctx["tags"].get(sym, set())
    sec_vals, sec_same = [], []
    for tag in my_tags & set(ctx["sector24"]):
        w = ctx["sector24"][tag]
        if ts_i in w.index:
            peers = w.loc[ts_i].drop(labels=[sym], errors="ignore").dropna()
            if len(peers) >= 3:
                sec_vals.append(side * peers.median())
                sec_same.append(float((side * peers > 0.03).mean()))
    row["sector_r24"] = float(np.mean(sec_vals)) if sec_vals else np.nan
    row["sector_peers_moving"] = float(np.mean(sec_same)) if sec_same else np.nan
    for tag in MAIN_TAGS:
        row[f"tag_{tag}"] = float(tag in my_tags)
    fz = ctx["fund"].get(sym)
    if fz is not None:
        fz = fz[fz["ts"] <= t_close]
        if len(fz):
            row["funding"] = -side * fz["rate"].iloc[-1]
            row["funding_3d"] = -side * fz["rate"].iloc[-9:].mean()
            row["funding_chg"] = -side * (fz["rate"].iloc[-1] - fz["rate"].iloc[-9:].mean())
    osym = ctx["oi"].get(sym)
    if osym is not None and len(osym):
        o_now = osym[osym["ts"] <= t_close]
        if len(o_now) and t_close - o_now["ts"].iloc[-1] <= 2 * HOUR:
            for hrs in (6, 24):
                o_then = osym[osym["ts"] <= t_close - hrs * HOUR]
                if len(o_then) and t_close - hrs * HOUR - o_then["ts"].iloc[-1] <= 2 * HOUR:
                    row[f"oi_chg_{hrs}h"] = side * (o_now["oi"].iloc[-1] / o_then["oi"].iloc[-1] - 1)
    lsym = ctx["lsr"].get(sym)
    if lsym is not None and len(lsym):
        z = lsym[lsym["ts"] <= t_close]
        if len(z) >= 25 and t_close - z["ts"].iloc[-1] <= 2 * HOUR:
            row["lsr_glob_chg24"] = side * math.log(z["global_acc"].iloc[-1] / z["global_acc"].iloc[-25])
            row["lsr_top_chg24"] = side * math.log(z["top_pos"].iloc[-1] / z["top_pos"].iloc[-25])
            row["lsr_glob"] = side * math.log(z["global_acc"].iloc[-1])
    base = clean_base(ctx["meta"].get(sym, {}).get("base") or sym.replace("USDT", ""))
    for kind_ in ANN_KINDS:
        row[f"ann_{kind_}_7d"] = 0.0
    row["ann_any_48h"] = 0.0
    a_sym = ctx["ann"].get(base)
    if a_sym is not None:
        recent = a_sym[(a_sym["ts"] <= t_close) & (a_sym["ts"] > t_close - 7 * DAY)]
        for kind_ in set(recent["kind"]):
            if f"ann_{kind_}_7d" in row:
                row[f"ann_{kind_}_7d"] = 1.0
        row["ann_any_48h"] = float(((a_sym["ts"] <= t_close) & (a_sym["ts"] > t_close - 2 * DAY)).any())
    # catalyst found in the study: steady DOWN-grinds in the week after an Upbit listing / Binance futures launch
    row["catalyst_short"] = float(side < 0 and (row["ann_upbit_list_7d"] or row["ann_futures_launch_7d"]))
    return row


def build(db: DB, days: float = 120.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(early-grind decisions with factors and outcomes, hindsight grind episodes). One coin at a time."""
    ensure_schema(db)
    t0 = time.time()
    now = int(time.time() * 1000)
    since = now - int((days + 31) * DAY)
    start_ts = now - int(days * DAY)
    ctx = load_context(db, since)
    log.info("grindlab: context for %d symbols (%.0fs)", len(ctx["syms"]), time.time() - t0)
    rows, episodes = [], []
    for sym in ctx["syms"]:
        k = _hourly(db, sym, since)
        if k is None:
            continue
        f = hourly_shape(k)
        d = _daily(db, sym)
        idx = k.index.to_numpy()
        o, h, l, c = (k[x].to_numpy() for x in ("o", "h", "l", "c"))
        tag_str = ",".join(sorted(ctx["tags"].get(sym, set()) - {"Crypto"}))
        for side in (1, -1):
            hits = np.flatnonzero(hindsight_mask(f, side))
            if len(hits):
                for g_ in np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1):
                    a_, b_ = g_[0], g_[-1]
                    if idx[b_] < start_ts:
                        continue
                    s0 = max(a_ - 23, 0)
                    episodes.append({"symbol": sym, "side": side, "start": int(idx[s0]), "end": int(idx[b_] + HOUR),
                                     "move": float(side * (c[b_] / o[s0] - 1)), "hours": int(b_ - s0 + 1),
                                     "best24": float(np.nanmax(side * f["r_24h"].to_numpy()[g_])), "tags": tag_str})
            last_t = -10 ** 18
            for i in np.flatnonzero(early_grind_mask(f, side)):
                t_close = int(idx[i] + HOUR)
                if t_close < start_ts or i + 1 + HORIZON_H > len(c) or t_close - last_t < DEDUPE_H * HOUR:
                    continue
                last_t = t_close
                row = decision_row(sym, side, i, k, f, d, ctx)
                row.update(outcome(side, o[i + 1], h[i + 1:i + 1 + HORIZON_H], l[i + 1:i + 1 + HORIZON_H],
                                   c[i + 1:i + 1 + HORIZON_H]))
                rows.append(row)
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True) if rows else pd.DataFrame()
    log.info("grindlab: %d early-grind decisions, %d hindsight episodes (%.0fs)", len(df), len(episodes), time.time() - t0)
    return df, pd.DataFrame(episodes)


