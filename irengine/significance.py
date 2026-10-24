import random
from typing import Sequence

RESAMPLES = 10_000
SEED = 15


def paired_randomization_test(
    a: Sequence[float], b: Sequence[float], resamples: int = RESAMPLES, seed: int = SEED
) -> float:
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    if not a:
        raise ValueError("no paired observations")
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(sum(diffs))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        # Flip the sign of each difference at random
        total = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(total) >= observed - 1e-12:
            extreme += 1
    # The observed assignment is counted once
    return (1 + extreme) / (1 + resamples)
