import json

import httpx
import pytest

from stealth_model_profiler import cli
from stealth_model_profiler.client import Budget, BudgetExceeded, Endpoint
from stealth_model_profiler.config import get_provider
from stealth_model_profiler.family import collect_family, distances, rank_families, recorded_cells
from stealth_model_profiler.family import detector as family_detector
from stealth_model_profiler.family.novelty import decision, two_neighbors
from stealth_model_profiler.features import tokenizer_features
from stealth_model_profiler.probes import Probe
from stealth_model_profiler.report import render_unavailable
from stealth_model_profiler.store import RecordStore, new_record, write_json


def test_retry_capture_spend_and_stop(tmp_path, monkeypatch):
    monkeypatch.setattr("stealth_model_profiler.client.time.sleep", lambda _: None)
    responses = iter([httpx.Response(503), httpx.Response(200, json={"usage": {"cost": 0.3}})])
    with httpx.Client(transport=httpx.MockTransport(lambda _: next(responses))) as client:
        ep = Endpoint(get_provider("openrouter"), "x", RecordStore(tmp_path / "r.jsonl"),
                      Budget(10, 0.2), _client=client)
        with pytest.raises(BudgetExceeded):
            ep.call("family", "fam:0", {})
        with pytest.raises(BudgetExceeded):
            ep.budget.charge()
        assert ep.budget.spend_usd == pytest.approx(0.3)
        assert [r.status for r in ep.store.read()] == [503, 200]


def test_stream_uses_final_usage_and_first_choice(tmp_path):
    stream = ('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
              'data: {"choices":[],"usage":{"cost":0.02,"completion_tokens":2}}\n\n'
              'data: [DONE]\n\n')
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=stream))) as client:
        ep = Endpoint(get_provider("openrouter"), "x", RecordStore(tmp_path / "s"), _client=client)
        result = ep.call("stream", "str:basic", {}, collect_stream=True)
        assert result.ok and result.body["choices"]
        assert ep.budget.spend_usd == pytest.approx(.02)


def test_outages_produce_no_tokenizer_features():
    assert tokenizer_features([]) == {}
    records = [new_record("tokenizer", "tok:base", "openrouter", "m", {}, status, {"error": {}}, "", 1.0)
               for status in (401, 402, 404, 429, 500, 503)]
    assert tokenizer_features(records) == {}


def test_count_offsets_and_sparse_sampling():
    def cells(token_offset, answer, n):
        return {f"en/p{i}": {"answers": {answer: n}, "prompt": i + token_offset} for i in range(40)}
    target, same, different = cells(8, "red", 1), cells(3, "red", 30), cells(3, "blue", 30)
    assert distances(target, same)["combined"] is None
    assert distances(target, same)["predictive"] == 0
    assert distances(target, different)["predictive"] == pytest.approx(.5)
    assert distances(dict(list(target.items())[:2]), same)["predictive"] is None
    ranking = rank_families(target, {"a": same, "b": different}, {"a": "x", "b": "y"}, "predictive")
    assert ranking[0]["family"] == "x"


def test_live_profile_and_offline_replay_report_same_family(tmp_path, monkeypatch, capsys, family_index):
    battery = {f"en/p{i}": [{"role": "user", "content": f"question {i}"}] for i in range(40)}
    def cells(ans):
        return {k: {"answers": {ans: 30}, "prompt": 8} for k in battery}
    write_json(family_index, {"schema": 1, "taxonomy": cli.references.INDEX_VERSION, "battery": battery,
                              "models": {"m1": cells("red"), "m2": cells("blue")},
                              "labels": {"m1": "a", "m2": "b"}, "cutoffs": {"predictive": .25}})
    body = {"provider": "Test host", "choices": [{"message": {"content": "red"}}],
            "usage": {"prompt_tokens": 8, "cost": 0.001}}
    monkeypatch.setattr(cli, "tokenizer_probes", lambda: [Probe("tok:base", "tokenizer", {"messages": []})])
    monkeypatch.setattr(cli, "load_all", lambda keys: {})

    def open_fixture(args, model, store, pin):
        return Endpoint(get_provider("openrouter"), model, RecordStore(store),
                        _client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))))

    monkeypatch.setattr(cli, "_open_endpoint", open_fixture)
    store = tmp_path / "run.jsonl"
    report = tmp_path / "profile.json"
    replay = tmp_path / "replay.json"
    assert cli.main(["profile", "--model", "opaque", "--no-catalog",
                     "--store", str(store), "--json", str(report)]) == 0
    assert cli.main(["replay", str(store), "--json", str(replay)]) == 0
    live, offline = (json.loads(path.read_text()) for path in (report, replay))
    assert live["family"] == offline["family"]
    assert live["family"]["attempts"] == 40 and live["family"]["leading_families"] == ["a"]
    assert live["family"]["decision"]["verdict"] == "family match"
    assert live["family"]["decision"]["cutoff"] == .05
    assert live["gateway_reported_providers"] == offline["gateway_reported_providers"] == {"Test host": 41}
    assert {r.channel for r in RecordStore(store).read()} == {"family", "tokenizer"}
    text = capsys.readouterr().out
    assert "nearest reference for a: m1" in text
    assert "relative gap 100.00%, cutoff 5.00%" in text
    for result in (live, offline):
        assert result["manifest"]["spend_usd"] == pytest.approx(0.041)
    assert cli.main(["profile", "--model", "opaque", "--no-catalog",
                     "--store", str(tmp_path / "stream.jsonl"), "--json", "-"]) == 0
    streamed = json.loads(capsys.readouterr().out)
    assert streamed["family"]["identity"] == offline["family"]["identity"]


def test_mandatory_reasoning_only_supplies_counts(tmp_path):
    battery = {"en/p": [{"role": "user", "content": "name a color"}]}
    responses = iter([
        httpx.Response(400, text="reasoning_effort none is not supported"),
        httpx.Response(200, json={"choices": [{"message": {"content": "blue"}}], "usage": {"prompt_tokens": 8}})
    ])
    with httpx.Client(transport=httpx.MockTransport(lambda _: next(responses))) as client:
        ep = Endpoint(get_provider("openrouter"), "x", RecordStore(tmp_path / "run"), _client=client)
        records = collect_family(ep, battery, 1, "predictive", 1)
    assert [r.status for r in records] == [400, 200]
    assert recorded_cells(records, battery) == {"en/p": {"answers": {}, "prompt": 8}}


@pytest.mark.parametrize("requested,fallback,leader,first,second,method,cutoff,verdict", [
    ("predictive", False, "a", .95, 1, "predictive", .05, "abstain"),
    ("predictive", False, "a", .94, 1, "predictive", .05, "family match"),
    ("predictive", True, "a", .80, 1, "counts", .20, "abstain"),
    ("counts", False, "a", .79, 1, "counts", .20, "family match"),
    ("counts", False, "other", .10, 1, "counts", .20, "abstain"),
    ("counts", False, "a", 0, 1e-14, "counts", .20, "abstain"),
    ("combined", False, "a", .60, 1, "combined", .50, "abstain"),
])
def test_family_decisions_on_an_enrolled_target(monkeypatch, requested, fallback, leader,
                                                first, second, method, cutoff, verdict):
    request = {"messages": [], "temperature": 1, "max_tokens": 16}
    if not fallback:
        request["reasoning"] = {"enabled": False}
    body = {"choices": [{"message": {"content": "red"}}], "usage": {"prompt_tokens": 8}}
    record = new_record("family", "family:en/p:0", "openrouter", "target", request, 200, body, "", 1)
    bank = {"models": {"target": {}, "x": {}, "y": {}}, "battery": {"en/p": []},
            "labels": {"x": leader, "y": "b", "target": "a"},
            "held_out_cutoffs": {"target": {"predictive": .99, "counts": 1, "combined": .50}}}

    def ranking(cells, refs, labels, selected):
        assert selected == method and "target" not in refs
        return [{"family": leader, "distance": first}, {"family": "b", "distance": second}]

    monkeypatch.setattr(family_detector, "rank_families", ranking)
    result = family_detector.family_result([record], bank, "target", requested)
    assert result["decision"]["cutoff"] == cutoff
    assert result["decision"]["verdict"] == verdict


def test_novelty_exclusions_and_ties():
    labels = {"a": "X", "aa": "X", "b": "Y", "c": "Z"}
    rows = [(.1, "a"), (.12, "aa"), (.2, "b"), (.4, "c")]
    assert two_neighbors(rows, labels, excluded_models={"a"}) == ("X", pytest.approx(.4))
    assert two_neighbors(rows, labels, excluded_families={"X"}) == ("Y", pytest.approx(.5))
    assert decision([{"distance": .1}, {"distance": .1}], 0)["verdict"] == "abstain"
    assert decision([{"distance": .1}], .2)["verdict"] == "insufficient calibration"


def test_model_response_failure_distinguishes_authentication():
    r = new_record("family", "t", "openrouter", "m", {}, 401, {"error": "unauthorized"}, "unauthorized", 1)
    report = render_unavailable(target="m", model="m", endpoint="openrouter", records=[r], calls=1)
    assert "NO MODEL RESPONSE" in report and "LISTING UNAVAILABLE" not in report


def test_store_reads_torn_tail_and_preserves_committed_records(tmp_path):
    path = tmp_path / "record.jsonl"
    store = RecordStore(path)
    rec = new_record("family", "a", "openrouter", "x", {}, 200, {"ok": True}, "", 1)
    store.append(rec)
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    store.append(rec)
    complete = path.read_bytes()
    path.write_bytes(complete + b'{"text":"\xe2')
    assert store.read() == [rec, rec]
    with pytest.raises(ValueError, match="incomplete"):
        store.append(rec)
    path.write_bytes(complete + b"invalid\n" + complete)
    with pytest.raises(ValueError, match="invalid JSON"):
        store.read()
