"""Empirical rejection for families missing from a reference bank."""
from __future__ import annotations

from collections import Counter


def relative_gap(first: float, second: float) -> float:
    return (second - first) / second if second > 0 else 0.0


def two_neighbors(neighbors, labels, excluded_models=(), excluded_families=()):
    """Return the first two distinct families remaining after a fold's exclusions."""
    first = None
    for distance, model in neighbors:
        family = labels[model]
        if model in excluded_models or family in excluded_families:
            continue
        if first is None:
            first = family, distance
        elif family != first[0]:
            return first[0], relative_gap(first[1], distance)
    return None, 0.0


def threshold(neighbors, labels, excluded_models=(), excluded_families=()):
    """90th percentile of simulated unknown gaps, giving every family equal mass."""
    examples = []
    for model, ranking in neighbors.items():
        lab = labels[model]
        if model in excluded_models or lab in excluded_families:
            continue
        predicted, gap = two_neighbors(ranking, labels, excluded_models, {*excluded_families, lab})
        if predicted is not None:
            examples.append((gap, lab))
    counts = Counter(f for _, f in examples)
    if len(counts) < 3:
        return None
    mass = 0.0
    for gap, family in sorted(examples):
        mass += 1 / (counts[family] * len(counts))
        if mass >= 0.9 - 1e-12:
            return gap
    return None


def decision(ranked: list[dict], cutoff) -> dict:
    if len(ranked) < 2 or cutoff is None:
        return {"verdict": "insufficient calibration", "gap": None, "cutoff": cutoff}
    gap = relative_gap(ranked[0]["distance"], ranked[1]["distance"])
    return {"verdict": "family match" if gap > cutoff + 1e-12 else "abstain",
            "gap": gap, "cutoff": cutoff}
