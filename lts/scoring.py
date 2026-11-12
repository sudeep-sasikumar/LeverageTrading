"""Rule-based score (research priors) — the cold-start scorer and a sanity benchmark for the model."""
from __future__ import annotations

import math


def _v(feat: dict, name: str, default: float = float("nan")) -> float:
    x = feat.get(name)
    return default if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


# (points, description, predicate over aligned features)
RULES = [
    (14, "clean 1h trend (efficiency > 0.45)", lambda f: _v(f, "er_1h", 0) > 0.45),
    (8, "clean 4h trend (efficiency > 0.35)", lambda f: _v(f, "er_4h", 0) > 0.35),
    (12, "open interest rising with the move", lambda f: _v(f, "oi_chg_1h", 0) > 0.01),
    (6, "OI building over 4h", lambda f: _v(f, "oi_chg_4h", 0) > 0.03),
    (10, "taker flow aligned over 1h", lambda f: _v(f, "imb_1h", 0) > 0.02),
    (6, "taker flow aligned over 4h", lambda f: _v(f, "imb_4h", 0) > 0.01),
    (10, "volume 2x+ normal", lambda f: _v(f, "rvol_1h", 0) >= 2.0),
    (8, "outperforming the market", lambda f: _v(f, "rs_mkt_1h", 0) > 0.005),
    (8, "BTC not against the trade", lambda f: _v(f, "btc_r_1h", 0) > -0.003),
    (6, "EMAs stacked with the trade", lambda f: _v(f, "ema_stack", 0) >= 2),
    (6, "liquid (>$50M 24h)", lambda f: _v(f, "log_qv_24h", 0) >= 7.7),
    (6, "closing strong in the bar", lambda f: _v(f, "clv", 0) > 0.3),
]
PENALTIES = [
    (15, "over-extended from EMA20", lambda f: _v(f, "d_ema20", 0) > 5.0),
    (10, "RSI exhausted", lambda f: _v(f, "rsi14", 50) > 88),
    (10, "rejection wicks against the trade", lambda f: _v(f, "adverse_wick", 0) > 0.45),
    (10, "crowded funding on the trade side", lambda f: _v(f, "fund_rate", 0) > 0.0005),
    (8, "OI falling (covering, not new positions)", lambda f: _v(f, "oi_chg_1h", 0) < -0.02),
    (8, "taker flow against the trade", lambda f: _v(f, "imb_1h", 0) < -0.03),
    (8, "BTC moving against the trade", lambda f: _v(f, "btc_r_1h", 0) < -0.008),
]


def rule_score(feat: dict) -> float:
    score = sum(p for p, _, pred in RULES if pred(feat)) - sum(p for p, _, pred in PENALTIES if pred(feat))
    return float(max(0.0, min(100.0, score)))


def reasons(feat: dict, limit: int = 5) -> tuple[list[str], list[str]]:
    """(supporting, warning) bullet texts for the Telegram message."""
    pos = [d for _, d, pred in RULES if pred(feat)][:limit]
    neg = [d for _, d, pred in PENALTIES if pred(feat)][:3]
    return pos, neg
