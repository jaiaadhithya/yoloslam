from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class LandingMetrics:
    mean_error: float
    median_error: float
    std_error: float
    success_rate: float
    mean_time: float


def summarize_trials(errors_m: Iterable[float], times_s: Iterable[float], success_flags: Iterable[int]) -> LandingMetrics:
    e = np.array(list(errors_m), dtype=float)
    t = np.array(list(times_s), dtype=float)
    s = np.array(list(success_flags), dtype=float)
    if len(e) == 0:
        return LandingMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    return LandingMetrics(
        mean_error=float(np.mean(e)),
        median_error=float(np.median(e)),
        std_error=float(np.std(e)),
        success_rate=float(np.mean(s)),
        mean_time=float(np.mean(t)) if len(t) else 0.0,
    )
