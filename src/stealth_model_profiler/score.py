"""Score measured tokenizer deltas against reference vocabularies."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .vocab import Vocab


@dataclass
class VocabScore:
    key: str
    lab: str
    family: str
    score: float
    exact: int
    n: int
    mae: float
    misses: Dict[str, int]
    profile: Tuple[int, ...] = ()


def score_vocab(measured: Dict[str, int], vocab: Vocab, base: str,
                probes: Dict[str, str]) -> Optional[VocabScore]:
    errors: Dict[str, int] = {}
    exact = 0
    n = 0
    total_abs = 0
    for name, predicted_probe in probes.items():
        if name not in measured:
            continue
        predicted = vocab.delta(base, predicted_probe)
        err = measured[name] - predicted
        errors[name] = err
        n += 1
        total_abs += abs(err)
        if err == 0:
            exact += 1
    if not n:
        return None
    mae = total_abs / n
    score = sum(math.exp(-abs(e) / 1.5) for e in errors.values()) / n
    profile = tuple(errors.get(name, 0) for name in sorted(probes))
    return VocabScore(key=vocab.spec.key, lab=vocab.spec.lab, family=vocab.spec.family,
                      score=score, exact=exact, n=n, mae=mae,
                      misses={k: v for k, v in errors.items() if v}, profile=profile)


def rank_vocabularies(measured: Dict[str, int], vocabs: Iterable[Vocab], base: str,
                      probes: Dict[str, str]) -> List[VocabScore]:
    scores = [score for vocab in vocabs if (score := score_vocab(measured, vocab, base, probes)) is not None]
    return sorted(scores, key=lambda s: (-s.score, s.mae))


def group_indistinguishable(scores: Sequence[VocabScore]) -> Dict[Tuple[int, ...], List[str]]:
    groups: Dict[Tuple[int, ...], List[str]] = defaultdict(list)
    for s in scores:
        groups[s.profile].append(s.key)
    return {k: v for k, v in groups.items() if len(v) > 1}
