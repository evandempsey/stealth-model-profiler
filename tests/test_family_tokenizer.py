from dataclasses import replace

from stealth_model_profiler.family.tokenizer import REPEAT_BASELINE, recorded_signature, resolve_ambiguity
from stealth_model_profiler.probes import tokenizer_probes
from stealth_model_profiler.store import new_record


def test_fingerprint_requires_complete_consistent_measurement():
    probes = tokenizer_probes()
    records = [new_record("tokenizer", p.id, "openrouter", "alias", p.body, 200,
                          {"provider": "host", "usage": {"prompt_tokens": 10 + i}}, None, 0)
               for i, p in enumerate(probes)]
    records.append(replace(records[0], tag=REPEAT_BASELINE))
    ids = [p.id for p in probes[1:]]
    assert recorded_signature(records, ids) == list(range(1, len(probes)))
    assert recorded_signature(records[:-1], ids) is None
    assert recorded_signature([*records[:1], *records[2:]], ids) is None
    assert recorded_signature([*records[:-1], replace(records[-1], body={
        "provider": "host", "usage": {"prompt_tokens": 11}})], ids) is None
    assert recorded_signature([replace(records[0], body={**records[0].body, "provider": "other"}), *records[1:]], ids) is None
    assert recorded_signature([replace(records[0], request={**records[0].request, "temperature": 1}), *records[1:]], ids) is None


def test_exact_tokenizer_must_disambiguate_all_nearby_families():
    ranked = [{"distance": .1, "family": "a"}, {"distance": .101, "family": "b"}]
    refs = {"labels": {"A1": "a", "B1": "b", "A2": "a"}, "tokenizer_signatures": {"A1": [4], "B1": [2]}}
    assert resolve_ambiguity(ranked, .05, [2], refs, "query") == "b"
    assert resolve_ambiguity(ranked, .05, [2], refs, "A1") is None
    assert resolve_ambiguity(ranked, .05, None, refs, "query") is None
    refs["tokenizer_signatures"]["A2"] = None
    assert resolve_ambiguity(ranked, .05, [2], refs, "A1") is None
    refs["tokenizer_signatures"]["A2"] = [2]
    assert resolve_ambiguity(ranked, .05, [2], refs, "query") is None


def test_family_live_fallback_and_replay(tmp_path, monkeypatch, family_index):
    import json
    import httpx

    from stealth_model_profiler import cli
    from stealth_model_profiler.client import Endpoint
    from stealth_model_profiler.config import get_provider
    from stealth_model_profiler.store import RecordStore, write_json

    battery = {f"en/{i}": [{"role": "user", "content": f"Question {i}"}] for i in range(40)}
    cells = {key: {"answers": {"yes": 15}, "prompt": 9} for key in battery}
    probes = tokenizer_probes()
    counts = {p.body["messages"][0]["content"]: 9 + i for i, p in enumerate(probes)}
    vector = list(range(1, len(probes)))
    write_json(family_index, {"schema": 1, "taxonomy": cli.references.INDEX_VERSION, "battery": battery,
                              "models": {"A1": cells, "B1": cells}, "labels": {"A1": "a", "B1": "b"},
                              "tokenizer_probe_ids": [p.id for p in probes[1:]],
                              "tokenizer_signatures": {"A1": [0] * len(vector), "B1": vector}})

    def respond(request):
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if prompt in counts and "temperature" in body:
            return httpx.Response(400, json={"error": {"code": 400}})
        return httpx.Response(200, json={"provider": "Host", "choices": [{"message": {"content": "several words"}}],
                                        "usage": {"prompt_tokens": counts.get(prompt, 9)}})

    def endpoint(args, model, store, pin):
        return Endpoint(get_provider("openrouter"), model, RecordStore(store),
                        _client=httpx.Client(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(cli, "_open_endpoint", endpoint)
    direct = tmp_path / "direct.json"
    assert cli.main(["family", "--model", "blind", "--runs", str(tmp_path), "--json", str(direct)]) == 0
    result = json.loads(direct.read_text())
    assert result["method"] == "counts"
    assert result["decision"]["family"] == "b" and result["decision"]["verdict"] == "family match"
    replay = tmp_path / "replay.json"
    assert cli.main(["family", "--from-run", result["records"], "--json", str(replay)]) == 0
    assert json.loads(replay.read_text())["decision"] == result["decision"]
