from typing import Iterable

import numpy as np
from scipy import stats


def compare_conditions(a: Iterable[float], b: Iterable[float]) -> dict:
    x = np.array(list(a), dtype=float)
    y = np.array(list(b), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return {"test": "n/a", "p_value": 1.0, "effect_size_d": 0.0}
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    _, p_normal = stats.shapiro(x - y)
    if p_normal > 0.05:
        _, p = stats.ttest_rel(x, y)
        test = "paired_t_test"
    else:
        _, p = stats.wilcoxon(x, y)
        test = "wilcoxon"
    pooled_std = np.std(np.concatenate([x, y])) + 1e-9
    d = (np.mean(x) - np.mean(y)) / pooled_std
    return {"test": test, "p_value": float(p), "effect_size_d": float(d)}
