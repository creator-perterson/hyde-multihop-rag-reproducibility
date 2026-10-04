"""Paired inference helpers used by the acceptance-revision audit."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Literal


TransferDecision = Literal["supported", "rejected", "unresolved"]


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the caller's original order."""

    values = [float(value) for value in p_values]
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("p-values must be in [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def paired_sign_flip_p(
    deltas: Sequence[float], *, seed: int, n_resamples: int
) -> float:
    """Estimate a two-sided paired sign-flip p-value with a fixed RNG seed."""

    values = [float(value) for value in deltas]
    if not values:
        raise ValueError("at least one paired delta is required")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    observed = abs(sum(values) / len(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_resamples):
        flipped = sum(value if rng.getrandbits(1) else -value for value in values)
        if abs(flipped / len(values)) >= observed:
            extreme += 1
    return (extreme + 1.0) / (n_resamples + 1.0)


def _is_favorable(effect: Mapping[str, object]) -> bool:
    estimate = float(effect["estimate"])
    direction = str(effect["direction"])
    if direction == "higher":
        return estimate > 0.0
    if direction == "lower":
        return estimate < 0.0
    raise ValueError("direction must be 'higher' or 'lower'")


def _is_primary(effect: Mapping[str, object]) -> bool:
    return bool(effect.get("primary_endpoint", True))


def _is_significant(effect: Mapping[str, object]) -> bool:
    p_value = effect.get("holm_p")
    return p_value is not None and float(p_value) < 0.05


def classify_adjusted_transfer(
    source_effect: Mapping[str, object], target_effects: Sequence[Mapping[str, object]]
) -> TransferDecision:
    """Apply the registered adjusted-p three-state transport decision rule."""

    targets = tuple(target_effects)
    if not targets or not _is_primary(source_effect) or not _is_significant(source_effect):
        return "unresolved"
    if not _is_favorable(source_effect):
        return "unresolved"

    for target in targets:
        if str(target.get("direction")) != str(source_effect.get("direction")):
            raise ValueError("source and target effects must use the same direction")
        if not _is_primary(target) or not _is_significant(target):
            return "unresolved"
        if not _is_favorable(target):
            return "rejected"
    return "supported"
