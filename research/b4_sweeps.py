"""B4 (docs/HYPOTHESES.md): sweeps + liquidations + order books. Run ON THE SERVER'S database (it holds the
liquidations and book snapshots): python -m research.b4_sweeps /data/lts.sqlite
Before 60 days of that data exist it only prints when it can first run."""
import sys

from research.common import Tee

from lts import sweeps_micro as SM
from lts.db import DB

out = Tee("b4_status.txt")
db = DB(sys.argv[1] if len(sys.argv) > 1 else "data/lts.sqlite")
p = SM.periods(db)
out(f"B4 data limits: {SM.LIMITS_NOTE}")
if not p.get("start"):
    out("No liquidation + order-book data in this database (collection runs on the server since 16-17 Nov 2026).")
    out("Earliest run: about 16 Jan 2027 (60 days after collection started); confirmation from about 15 Feb 2027.")
    raise SystemExit(0)
out(f"Data from {p['start']} to {p['end']}; discovery possible from {p['first_run_date']}, confirmation from {p['confirm_date']}")
try:
    df = SM.evaluate(db, "discovery")
except SM.NotEnoughData as e:
    out(f"Not run: {e}")
    raise SystemExit(0)
out(f"{len(df):,} sweep trades in discovery")
