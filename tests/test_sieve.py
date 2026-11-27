"""Tests for the 3-layer tokenizer + reasoning sieve."""

from __future__ import annotations

import json

import httpx

from stealth_model_profiler import cli
from stealth_model_profiler.client import Endpoint
from stealth_model_profiler.config import get_provider
from stealth_model_profiler.family.sieve import (
    STAGE2_PROMPTS, load_stage2_references, sieve_identify, stage2_probes,
)
from stealth_model_profiler.probes import tokenizer_probes
from stealth_model_profiler.store import RecordStore, new_record, write_json


def _tok_records(deltas: list[int], base: int = 10):
    probes = tokenizer_probes()
    recs = [new_record("tokenizer", "tok:base", "openrouter", "blind", {}, 200,
                       {"usage": {"prompt_tokens": base}}, "", 1.0)]
    for probe, d in zip(probes[1:], deltas):
        recs.append(new_record("tokenizer", probe.id, "openrouter", "blind", {}, 200,
                               {"usage": {"prompt_tokens": base + d}}, "", 1.0))
    return recs


def _s2_records(ptok: int, reas: int, answer_prefix: str):
    recs = []
    for pid in STAGE2_PROMPTS:
        body = {
            "choices": [{"message": {"content": f"{answer_prefix} response for {pid}"}}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": reas + 10,
                      "completion_tokens_details": {"reasoning_tokens": reas}},
        }
        recs.append(new_record("stage2", f"s2:{pid}", "openrouter", "blind", {}, 200, body, "", 1.0))
    return recs


def _synthetic_bank():
    pids = [p.id for p in tokenizer_probes()[1:]]
    n = len(pids)
    return {
        "schema": 1,
        "taxonomy": cli.references.INDEX_VERSION,
        "battery": {},
        "models": {},
        "tokenizer_probe_ids": pids,
        "tokenizer_signatures": {
            "u/v1": [1] * n,
            "a/v1": [2] * n,
            "b/v1": [2] * n,
        },
        "labels": {"u/v1": "unique_fam", "a/v1": "fam_a", "b/v1": "fam_b"},
    }


def _synthetic_s2_refs():
    from stealth_model_profiler.family.sieve import extract_stage2
    return [
        ("fam_a", "a/v1", extract_stage2(_s2_records(20, 0, "alpha style concise"))),
        ("fam_b", "b/v1", extract_stage2(_s2_records(34, 150, "beta verbose reasoning"))),
    ]


def test_packaged_stage2_references_load():
    refs = load_stage2_references()
    assert len(refs) == 23
    fams = {f for f, _, _ in refs}
    assert {"gemini", "gemma", "deepseek", "step", "mistral", "nemotron", "qwen", "mimo"} <= fams


def test_layer1_unique_and_near_exact_match_and_unseen_rejection():
    bank = _synthetic_bank()
    n = len(bank["tokenizer_probe_ids"])
    # Unique exact match
    res = sieve_identify(_tok_records([1] * n), bank, "blind")
    assert res["verdict"] == "family match" and res["family"] == "unique_fam"
    assert res["layer2"] is None

    # Near-exact match (2 probes differ by 1 -> MAE = 2/76 = 0.026 <= 0.15)
    tweaked = [1] * n
    tweaked[0] = 2
    tweaked[1] = 2
    res2 = sieve_identify(_tok_records(tweaked), bank, "blind")
    assert res2["verdict"] == "family match" and res2["family"] == "unique_fam"

    # Unseen tokenizer (all probes off by 5 -> MAE = 5.0 > 0.15)
    res3 = sieve_identify(_tok_records([6] * n), bank, "blind")
    assert res3["verdict"] == "abstain" and res3["evidence"] == "unseen tokenizer"


def test_collision_agreement_and_system_prompt_shift():
    bank = _synthetic_bank()
    refs = _synthetic_s2_refs()
    n = len(bank["tokenizer_probe_ids"])
    tok = _tok_records([2] * n)

    # Both template (ptoks=34) and behavior (reas=150, beta text) match fam_b -> strict agrees
    recs_agree = tok + _s2_records(34, 150, "beta verbose reasoning")
    res = sieve_identify(recs_agree, bank, "blind", mode="strict", stage2_refs=refs)
    assert res["verdict"] == "family match" and res["family"] == "fam_b"

    # Hidden system prompt shifts ptoks from 20 -> 34 (matches fam_b template) while behavior is fam_a
    recs_shifted = tok + _s2_records(34, 0, "alpha style concise")
    strict = sieve_identify(recs_shifted, bank, "blind", mode="strict", stage2_refs=refs)
    behav = sieve_identify(recs_shifted, bank, "blind", mode="behavioral", stage2_refs=refs)
    assert strict["verdict"] == "abstain" and "disagree" in strict["evidence"]
    assert behav["verdict"] == "family match" and behav["family"] == "fam_a"


def test_cli_sieve_live_skips_stage2_on_unique_and_replays(tmp_path, monkeypatch, capsys, family_index):
    bank = _synthetic_bank()
    write_json(family_index, bank)
    probes = tokenizer_probes()
    counts = {p.body["messages"][0]["content"]: 10 + (0 if p.id == "tok:base" else 1) for p in probes}
    s2_calls = 0

    def respond(request):
        nonlocal s2_calls
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if prompt not in counts:
            s2_calls += 1
        return httpx.Response(200, json={
            "provider": "Host",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": counts.get(prompt, 20), "completion_tokens": 1},
        })

    def open_fixture(args, model, store, pin):
        return Endpoint(get_provider("openrouter"), model, RecordStore(store),
                        _client=httpx.Client(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(cli, "_open_endpoint", open_fixture)
    store = tmp_path / "sieve.jsonl"
    assert cli.main(["sieve", "--model", "opaque", "--store", str(store), "--json", "-"]) == 0
    live = json.loads(capsys.readouterr().out)
    assert live["verdict"] == "family match" and live["family"] == "unique_fam"
    assert s2_calls == 0  # Stage 2 skipped on unique tokenizer!

    assert cli.main(["sieve", "--from-run", str(store), "--json", "-"]) == 0
    offline = json.loads(capsys.readouterr().out)
    assert offline["family"] == live["family"] and offline["verdict"] == live["verdict"]
    assert len(stage2_probes()) == 16
