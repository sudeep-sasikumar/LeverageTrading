"""Shared validation for every study (docs/HYPOTHESES.md).

- Statistics are clustered by UTC day: trades on the same day are not independent (they share the market), so
  the t-statistic is computed on daily means.
- Share of calendar months with a positive mean.
- A correction for the number of variants tried: Holm's step-down method on one-sided p-values from the
  day-clustered t (normal approximation), reported back as an equivalent t.
- A discovery / confirmation split with the confirmation period LOCKED: confirmation results can only be
  computed for rules that were frozen (written to docs/results/<study>_frozen.json) beforehand, and the frozen
  set cannot be changed afterwards.

Pass bar (docs/HYPOTHESES.md, "Pass bar"): >= 150 trades over >= 60 days, >= +0.10R per trade after all costs,
corrected t >= 3, >= 60% of months positive in discovery; then in confirmation positive with t >= 2.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86_400_000
BAR = {"min_trades": 150, "min_days": 60, "min_r": 0.10, "min_t_corrected": 3.0, "min_months_pos": 0.60,
       "conf_min_t": 2.0}


def _norm_sf(t: float) -> float:
    return 0.5 * math.erfc(t / math.sqrt(2.0))


def _norm_isf(p: float) -> float:
    """Inverse survival function of the standard normal."""
    p = min(max(p, 1e-300), 1 - 1e-16)
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - p)


def stats(r: np.ndarray, ts: np.ndarray) -> dict:
    """r: per-trade result (R or % - whatever the study reports in); ts: entry times (ms)."""
    r, ts = np.asarray(r, float), np.asarray(ts, np.int64)
    ok = np.isfinite(r)
    r, ts = r[ok], ts[ok]
    n = len(r)
    if n == 0:
        return {"n": 0}
    s = pd.Series(r)
    daily = s.groupby(ts // DAY).mean()
    nd = len(daily)
    t = float(daily.mean() / (daily.std(ddof=1) / math.sqrt(nd))) if nd > 2 and daily.std(ddof=1) > 0 else 0.0
    month = pd.to_datetime(ts, unit="ms").strftime("%Y-%m")
    mon = s.groupby(np.asarray(month)).mean()
    span = (ts.max() - ts.min()) / DAY if n > 1 else 0.0
    se = float(daily.std(ddof=1) / math.sqrt(nd)) if nd > 2 else float("nan")
    return {"n": int(n), "days": int(nd), "span_days": float(span), "mean": float(r.mean()), "win": float((r > 0).mean()),
            "t_day": t, "se_day": se, "months_pos": float((mon > 0).mean()), "n_months": int(len(mon)),
            "per_day": float(n / max(span, 1.0))}


def holm(ts_: list[float]) -> list[float]:
    """Holm-corrected equivalent t for a family of one-sided t-statistics (same order as given)."""
    m = len(ts_)
    if m == 0:
        return []
    p = np.array([_norm_sf(t) for t in ts_])
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return [float(_norm_isf(x)) if x < 1 else float("-inf") for x in adj]


def passes_discovery(st: dict, t_corrected: float, bar: dict = BAR) -> tuple[bool, list[str]]:
    why = []
    if st.get("n", 0) < bar["min_trades"]:
        why.append(f"n {st.get('n', 0)} < {bar['min_trades']}")
    if st.get("span_days", 0) < bar["min_days"]:
        why.append(f"{st.get('span_days', 0):.0f} days < {bar['min_days']}")
    if not st.get("mean", -1) >= bar["min_r"]:
        why.append(f"mean {st.get('mean', float('nan')):+.3f} < +{bar['min_r']}")
    if not t_corrected >= bar["min_t_corrected"]:
        why.append(f"corrected t {t_corrected:.2f} < {bar['min_t_corrected']}")
    if not st.get("months_pos", 0) >= bar["min_months_pos"]:
        why.append(f"months positive {st.get('months_pos', 0):.0%} < {bar['min_months_pos']:.0%}")
    return (not why), why


def passes_confirmation(st: dict, bar: dict = BAR) -> bool:
    return st.get("n", 0) > 0 and st["mean"] > 0 and st["t_day"] >= bar["conf_min_t"]


@dataclass
class Split:
    """Discovery = entries before conf_start; confirmation = from conf_start. Confirmation is locked."""
    study: str
    conf_start: int
    results_dir: Path
    frozen: dict = field(default_factory=dict)

    @property
    def frozen_path(self) -> Path:
        return self.results_dir / f"{self.study}_frozen.json"

    def discovery(self, ts: np.ndarray) -> np.ndarray:
        return np.asarray(ts) < self.conf_start

    def freeze(self, rules: dict) -> str:
        """Freeze the rules that go to confirmation (a dict name -> spec). Written once; never changed."""
        blob = json.dumps(rules, sort_keys=True, default=str)
        h = hashlib.sha256(blob.encode()).hexdigest()[:16]
        if self.frozen_path.exists():
            old = json.loads(self.frozen_path.read_text(encoding="utf-8"))
            if old["hash"] != h:
                raise RuntimeError(f"{self.frozen_path.name} already holds a different frozen rule set "
                                   f"({old['hash']}); the confirmation period is locked")
        else:
            self.frozen_path.write_text(json.dumps({"study": self.study, "hash": h, "rules": rules}, indent=1,
                                                   default=str), encoding="utf-8")
        self.frozen = rules
        return h

    def confirmation(self, ts: np.ndarray, rule_name: str) -> np.ndarray:
        if not self.frozen_path.exists():
            raise RuntimeError("confirmation is locked until the rules are frozen (Split.freeze)")
        frozen = json.loads(self.frozen_path.read_text(encoding="utf-8"))["rules"]
        if rule_name not in frozen:
            raise RuntimeError(f"rule {rule_name!r} was not frozen; the confirmation period is locked for it")
        return np.asarray(ts) >= self.conf_start


def by_group(r: np.ndarray, ts: np.ndarray, groups: dict[str, np.ndarray]) -> dict:
    """stats() per named boolean mask (e.g. side, market regime)."""
    return {k: stats(np.asarray(r)[m], np.asarray(ts)[m]) for k, m in groups.items()}


def alt_regime(daily_close: pd.DataFrame, min_coins: int = 30) -> pd.Series:
    """Market regime per UTC day: +1 when an equal-weight altcoin index is above its 50-day mean, else -1.
    daily_close: index = day start (ms), columns = coins, values = daily closes. The index is the cumulative
    median daily return across coins (robust to listings); the regime for day d uses closes up to day d-1."""
    ret = daily_close.pct_change(fill_method=None)
    alts = ret.drop(columns=[c for c in ("BTCUSDT", "ETHUSDT") if c in ret.columns])
    med = alts.median(axis=1, skipna=True).where(alts.notna().sum(axis=1) >= min_coins)
    idx = (1 + med.fillna(0)).cumprod()
    reg = np.sign(idx - idx.rolling(50, min_periods=50).mean()).shift(1)
    return reg.replace(0, -1)
