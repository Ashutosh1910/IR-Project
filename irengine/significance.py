"""Paired randomization (sign-flip) test for per-query score differences.

Under the null hypothesis the two systems are exchangeable on every query, so each
per-query difference is equally likely to have either sign. The p-value is the
share of random sign assignments whose |mean difference| is at least the observed
one, with the observed assignment counted once: p = (1 + #extreme) / (1 + resamples).
"""

import random
from typing import Sequence

RESAMPLES = 10_000
SEED = 15


def paired_randomization_test(
    a: Sequence[float], b: Sequence[float], resamples: int = RESAMPLES, seed: int = SEED
) -> float:
    """Two-sided p-value for mean(a - b) != 0."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    if not a:
        raise ValueError("no paired observations")
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(sum(diffs))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        total = sum(d if rng.random() < 0.5 else -d for d in diffs)
        # small tolerance so float noise in an equal-magnitude sum still counts
        if abs(total) >= observed - 1e-12:
            extreme += 1
    return (1 + extreme) / (1 + resamples)
