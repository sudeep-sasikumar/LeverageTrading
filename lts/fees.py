"""Trading costs. The ONLY source is config.trade (taker_fee, maker_fee, slippage); every module and research script
takes its fees from here, and tests/test_fees.py fails if a fee number is written anywhere else.

Weex, checked on real trades (docs/DESIGN.md section 22): a market entry pays taker + slippage, a stop or time exit
pays taker + slippage, and a position take-profit also fills as a TAKER order (no slippage assumed: it fills at
the trigger). `target_maker` is for studies of resting reduce-only limit take-profits (study B3)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Fees:
    taker: float
    maker: float
    slip: float

    @property
    def entry(self) -> float:            # market entry
        return self.taker + self.slip

    @property
    def entry_maker(self) -> float:      # resting limit entry
        return self.maker

    @property
    def stop(self) -> float:             # stop-market exit
        return self.taker + self.slip

    @property
    def time(self) -> float:             # market close at the time limit
        return self.taker + self.slip

    @property
    def target(self) -> float:           # Weex position TP: taker, at the trigger price
        return self.taker

    @property
    def target_maker(self) -> float:     # resting reduce-only limit TP
        return self.maker


@lru_cache(maxsize=1)
def _from_config() -> Fees:
    from .config import load_config
    t = load_config().trade
    return Fees(float(t.taker_fee), float(t.maker_fee), float(t.slippage))


def fees(trade=None) -> Fees:
    """Fees from a config.trade object, or from the loaded configuration."""
    if trade is None:
        return _from_config()
    return Fees(float(trade.taker_fee), float(trade.maker_fee), float(trade.slippage))
