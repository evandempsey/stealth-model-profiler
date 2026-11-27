"""Ground-truth regression: published measurements of the Ox Alpha stealth model.

The published probe battery reported GLM-5.2 at 24/24 exact and Llama-3 well
behind. Replaying those *recorded* deltas through this project's scoring code
checks the differential maths, the vocabulary loader and the ranking together,
without spending a token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stealth_model_profiler.probes import BASE, TOKENIZER_PROBES
from stealth_model_profiler.score import score_vocab
from stealth_model_profiler.vocab import load_all

FIXTURE = Path(__file__).parent / "fixtures" / "ox_alpha_measured.json"


@pytest.fixture(scope="module")
def measured() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scores(measured):
    required = ["glm-5.2", "llama3", "qwen3"]
    vocabs = load_all(required)
    assert set(vocabs) == set(required), "integration test requires the three pinned tokenizers"
    out = [s for v in vocabs.values()
           if (s := score_vocab(measured, v, BASE, TOKENIZER_PROBES)) is not None]
    out.sort(key=lambda s: (-s.score, s.mae))
    return out


def test_glm_family_is_exact(scores):
    top = scores[0]
    assert top.lab == "zai", f"expected Z.ai on top, got {top.lab} ({top.key})"
    assert top.n == 24
    assert top.exact == top.n, f"expected all probes exact, got {top.exact}/{top.n}"
    assert top.mae == 0.0


def test_glm_beats_unrelated_vocabularies(scores):
    by_key = {s.key: s for s in scores}
    glm = by_key["glm-5.2"]
    for other in ("llama3", "qwen3"):
        if other in by_key:
            assert glm.score > by_key[other].score + 0.3, (
                f"{other} should be clearly separated from GLM, "
                f"got {by_key[other].score:.3f} vs {glm.score:.3f}")


def test_battery_covers_published_probes(measured):
    """Every probe the publication measured must still exist in the battery.

    If a probe is renamed or dropped, the comparison silently narrows, so this
    is asserted rather than assumed.
    """
    missing = set(measured) - set(TOKENIZER_PROBES)
    assert not missing, f"probes dropped from the battery: {sorted(missing)}"
