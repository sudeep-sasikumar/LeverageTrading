"""Review item 12 (25 Nov 2026): fees, slippage come from config.trade only (lts/fees.py)."""
import re
from pathlib import Path

from lts.config import Config

ROOT = Path(__file__).resolve().parent.parent
HOUR, MIN = 3_600_000, 60_000
T0 = 1_794_900_000_000 // HOUR * HOUR


# ---- item 12: fees come from config.trade only ------------------------------------------------------------
FEE_LITERAL = re.compile(r"(?<![\d.])0\.000(8|2|4)(?!\d)|(?<![\d.])0\.0012(?!\d)")
FEE_NAME = re.compile(r"^\s*(TAKER|MAKER|SLIP|COST|ENTRY|MKT|TGT|ENTRY_COST|taker_fee|maker_fee|slippage)\s*(,\s*\w+\s*)*=\s*[\d(]",
                      re.M)


def test_no_hard_coded_fees_outside_config():
    bad = []
    for p in list((ROOT / "lts").glob("*.py")) + list((ROOT / "research").glob("*.py")):
        if p.name == "config.py":
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if FEE_LITERAL.search(code) or FEE_NAME.search(code):
                bad.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not bad, "fees must come from config.trade (lts.fees):\n" + "\n".join(bad)


def test_fees_follow_config():
    from lts.fees import fees
    t = Config().trade
    t.taker_fee, t.slippage, t.maker_fee = 0.001, 0.0, 0.0
    f = fees(t)
    assert (f.entry, f.stop, f.target, f.target_maker) == (0.001, 0.001, 0.001, 0.0)


