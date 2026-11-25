"""research/parity.py computes the steady rule with a lean copy of grindlab.hourly_shape: it must be identical."""
import numpy as np
import pandas as pd

from lts import grindlab as G
from research.parity import rule_shape


def test_rule_shape_equals_hourly_shape():
    rng = np.random.default_rng(5)
    n = 900
    c = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    k = pd.DataFrame({"o": c * 0.999, "h": c * 1.004, "l": c * 0.996, "c": c, "v": 1.0, "qv": rng.uniform(1e5, 1e6, n),
                      "trades": 10.0, "tbqv": 5e4}, index=np.arange(n) * 3_600_000)
    a, b = G.hourly_shape(k), rule_shape(k)
    for col in b.columns:
        pd.testing.assert_series_equal(a[col], b[col], check_names=False)
    for side in (1, -1):
        assert (G.early_grind_mask(a, side) == G.early_grind_mask(b, side)).all()
