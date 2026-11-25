"""Rebuild the year's early-grind moments exactly as the year studies did: grindlab.build(days=390) on the local
hourly history, with the window ending when the original table was built (16 Nov 2026 23:35 UTC).
Original result (DESIGN.md section 15): 63,528 moments over 389 days, 519 coins."""
from research.common import EVENTS, LOCAL_DB, YEAR_NOW, Tee

from lts import grindlab as G
from lts.db import DB

out = Tee("build_events.txt")
db = DB(LOCAL_DB)
df, ep = G.build(db, days=390, now=YEAR_NOW)
df.to_parquet(EVENTS)
out(f"{len(df):,} early-grind moments over {(df.ts.max() - df.ts.min()) / 86_400_000:.0f} days, {df.symbol.nunique()} coins, "
    f"{len(ep):,} steady grinds (published: 63,528 / 389 days / 519 coins / 27,401)")
