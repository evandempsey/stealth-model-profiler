"""Validate the supplemental inputs to the published family index."""
import json
from collections import Counter
from importlib.resources import files

import pytest

from stealth_model_profiler.probes import tokenizer_probes


@pytest.fixture(scope="module")
def enrollment():
    return json.loads(files("stealth_model_profiler.family").joinpath("enrollment.json").read_text(encoding="utf-8"))


def test_behavioral_fingerprints_and_calibration_draws(enrollment):
    models = enrollment["models"]
    assert len(models) == 39
    battery = models[next(iter(models))]["cells"].keys()
    assert len(battery) == 40
    for model, sample in models.items():
        cells = sample["cells"]
        assert sample["label"] and cells.keys() == sample["first"].keys() == battery, model
        for key, cell in cells.items():
            first = sample["first"][key]
            assert cell["prompt"] > 0 and first["prompt"] > 0
            assert sum(first["answers"].values()) <= 1
            assert Counter(first["answers"]) <= Counter(cell["answers"])
        if "eligible_cells" in sample:
            eligible = sum(sum(c["answers"].values()) >= 10 for c in cells.values())
            assert sample["eligible_cells"] == eligible
            if sample.get("mode") == "counts_only":
                assert all(not c["answers"] for c in cells.values())
            else:
                assert eligible >= 20


def test_tokenizer_signatures_match_the_probe_panel(enrollment):
    panel = enrollment["tokenizer"]
    assert panel["probes"] == [p.id for p in tokenizer_probes()[1:]]
    assert len(panel["signatures"]) == 151
    for signature in panel["signatures"].values():
        assert len(signature) == len(panel["probes"]) and all(type(n) is int for n in signature)


def test_product_labels_and_reference_evidence(enrollment):
    models = enrollment["models"]
    assert Counter(enrollment["labels"].values()) == {
        "gemma": 7, "seed": 4, "ling": 2, "lfm": 1, "mimo": 1, "hermes": 4,
        "solar": 1, "reka": 1, "perceptron": 1, "morph": 1,
    }
    assert "meituan/longcat-2.0" in models and "baidu/ERNIE-4.5-0.3B-PT" not in models
    assert models["thinkingmachines/inkling-2-small"]["mode"] == "counts_only"
    assert models["thinkingmachines/inkling-small"]["mode"] == "answers_and_counts"
    assert models["thinkingmachines/inkling-2-small"]["label"] == models["thinkingmachines/inkling-small"]["label"] == "inkling"
    for model, label in {
        "sakana/sakana-namazu": "namazu", "scaledown/summarize": "summarize",
        "unbiased/pareto-26.9": "pareto", "thedrummer/unslopnemo-12b": "unslopnemo", "relace/relace-search": "relace",
    }.items():
        assert model in enrollment["tokenizer"]["signatures"]
        assert models[model]["label"] == label
        assert all(10 <= sum(c["answers"].values()) <= 30 for c in models[model]["cells"].values())
    assert "unbiased/pareto-26.10" not in models
