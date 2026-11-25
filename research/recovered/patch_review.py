import pathlib

ROOT = pathlib.Path(r"D:\Leverage trading")


def sub(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:70])
        s = s.replace(a, b, 1)
    p.write_text(s, encoding="utf-8")


# ---- service: daily report carries the grinder review ------------------------------------------
sub("lts/service.py", [
    ('''                    await self.tg.send(daily_report_text(self.cfg, self.db))''',
     '''                    await self.daily_report()'''),
    ('''    async def retrain(self) -> None:''',
     '''    async def daily_report(self) -> None:
        from .movers import review, review_text

        def work():
            own = DB(self.cfg.path(self.cfg.storage.db_path))     # sqlite connections are per-thread
            try:
                return review(self.cfg, own)
            finally:
                own.close()
        text = daily_report_text(self.cfg, self.db)
        try:
            res = await asyncio.to_thread(work)
            text += "\\n\\n" + review_text(res, self.cfg.telegram.local_utc_offset_minutes, self.cfg.telegram.local_tz_label)
        except Exception:
            log.exception("daily grinder review failed")
        await self.tg.send(text)

    async def retrain(self) -> None:'''),
])

# ---- CLI ----------------------------------------------------------------------------------------
sub("lts/cli.py", [
    ('''    p = sub.add_parser("research", help="barrier-grid and base-rate study on labelled candidates")''',
     '''    p = sub.add_parser("research", help="barrier-grid and base-rate study on labelled candidates")
    p = sub.add_parser("movers", help="review the last 24h's steady movers and whether LTS caught them")
    p.add_argument("--no-store", action="store_true", help="print only, do not record")'''),
    ('''        elif args.cmd == "research":''',
     '''        elif args.cmd == "movers":
            import re
            from .movers import review, review_text
            text = review_text(review(cfg, db, store=not args.no_store), cfg.telegram.local_utc_offset_minutes,
                               cfg.telegram.local_tz_label)
            print(re.sub(r"</?[a-z]+>", "", text))
        elif args.cmd == "research":'''),
])

# ---- dashboard snapshot -------------------------------------------------------------------------
sub("lts/dashboard.py", [
    ('''        cols = ("id, symbol, side, ts, grade, entry, sl, tp, sl_pct, tp_pct, leverage, margin_frac, p, ev_r, "
                "model_version, expires_ts, status, exit_ts, exit_price, ret_r, ret_margin, mfe, mae")''',
     '''        has_kind = "kind" in [r["name"] for r in q("PRAGMA table_info(signals)")]
        kind_col = "COALESCE(kind,'momentum') AS kind" if has_kind else "'momentum' AS kind"
        cols = ("id, symbol, side, ts, grade, entry, sl, tp, sl_pct, tp_pct, leverage, margin_frac, p, ev_r, "
                f"model_version, expires_ts, status, exit_ts, exit_price, ret_r, ret_margin, mfe, mae, {kind_col}")'''),
    ('''        closed = q("SELECT grade, status, exit_ts, ret_r, ret_margin FROM signals WHERE status!='open' "
                   "AND ret_r IS NOT NULL ORDER BY exit_ts")''',
     '''        closed = q(f"SELECT grade, status, exit_ts, ret_r, ret_margin, {kind_col} FROM signals WHERE status!='open' "
                   "AND ret_r IS NOT NULL ORDER BY exit_ts")'''),
    ('''            "real": _stats([r for r in closed if r["grade"] != "P"]),
        }''',
     '''            "real": _stats([r for r in closed if r["grade"] != "P"]),
            "momentum": _stats([r for r in closed if r["kind"] != "grind"]),
            "grind": _stats([r for r in closed if r["kind"] == "grind"]),
        }
        out["movers"] = kv.get("movers_latest")
        if q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_movers'"):
            hist = q("SELECT day, symbol, side, ret24, r2, hours_with, rvol24, oi_chg24, flagged_ts, flagged_kind, "
                     "flagged_left, signal_id, signal_status, signal_r FROM daily_movers ORDER BY day DESC, "
                     "ABS(ret24) * r2 DESC LIMIT 300")
            out["movers_history"] = hist
            recent = [h for h in hist if h["day"] >= time.strftime("%Y-%m-%d", time.gmtime(now / 1000 - 30 * 86400))]
            flagged = [h for h in recent if h["flagged_ts"]]
            lefts = sorted(h["flagged_left"] for h in flagged if h["flagged_left"] is not None)
            out["movers_stats"] = {"n": len(recent), "flagged": len(flagged),
                                   "median_left": lefts[len(lefts) // 2] if lefts else None,
                                   "signalled": sum(1 for h in recent if h["signal_id"])}'''),
])
print("service/cli/dashboard patched")
