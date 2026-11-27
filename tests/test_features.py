"""Tokenizer feature extraction."""

from __future__ import annotations

from stealth_model_profiler.features import tokenizer_features
from stealth_model_profiler.store import Record


def rec(status=200, body=None, channel="tokenizer", tag="tok:base"):
    return Record(channel=channel, tag=tag, endpoint="openrouter", model="x/y",
                  request={}, status=status, body=body, text=None,
                  latency_ms=1.0, ts=0.0)


def test_tokenizer_features_difference_against_base():
    def usage(n):
        return {"usage": {"prompt_tokens": n, "completion_tokens": 1, "total_tokens": n + 1}}

    records = [
        rec(tag="tok:base", body=usage(100)),
        rec(tag="tok:en_pangram", body=usage(123)),
        rec(tag="tok:zh_simp", body=usage(101)),
    ]
    feats = tokenizer_features(records)
    assert feats["base_prompt_tokens"] == 100
    assert feats["deltas"] == {"en_pangram": 23, "zh_simp": 1}


def test_failed_probe_does_not_override_successful_one():
    def usage(n):
        return {"usage": {"prompt_tokens": n}}

    records = [
        rec(tag="tok:base", body=usage(10)),
        rec(tag="tok:ja", body=usage(20)),
        rec(tag="tok:ja", status=500, body=None),
    ]
    feats = tokenizer_features(records)
    assert feats["deltas"]["ja"] == 10
