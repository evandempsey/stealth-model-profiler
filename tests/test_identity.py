import json
from dataclasses import replace

import pytest

from stealth_model_profiler import cli
from stealth_model_profiler.family import references
from stealth_model_profiler.family import family_probes, family_result
from stealth_model_profiler.store import RecordStore, new_record, write_json


def observations():
    battery = {f"en/{i}": [{"role": "user", "content": str(i)}] for i in range(20)}
    def reference(word):
        return {key: {"answers": {word: 30}, "prompt": 10} for key in battery}
    models = {"one/v1": reference("yes"), "one/v2": reference("maybe"), "two/v1": reference("no")}
    bank = dict(schema=1, taxonomy=references.INDEX_VERSION, models=models, battery=battery,
                labels={"one/v1": "one", "one/v2": "one", "two/v1": "two"})
    response = {"choices": [{"message": {"content": "yes"}}], "usage": {"prompt_tokens": 10}}
    records = [new_record("family", p.id, "openrouter", "blind", p.body, 200, response, None, 0)
               for p in family_probes(battery, 1)]
    return bank, records


def test_model_ids_are_supporting_candidates_only():
    bank, records = observations()
    def identify():
        return family_result(records, bank, "blind", "predictive")["identity"]
    assert identify()["product_family"]["family"] == "one"
    assert [r["model"] for r in identify()["model_candidates"]] == ["one/v1"]
    assert set(identify()) == {"product_family", "model_candidates", "tokenizer"}
    named = family_result(records, bank, "two/v1", "predictive", exclude_target=False)
    assert named["identity"] == identify()
    bank["models"]["one/v2"] = bank["models"]["one/v1"]
    assert [r["model"] for r in identify()["model_candidates"]] == ["one/v1", "one/v2"]
    bank["models"]["two/v1"] = bank["models"]["one/v1"]
    assert identify()["product_family"]["verdict"] == "ambiguous"
    assert identify()["product_family"]["family"] is None


def test_tokenizer_candidates_survive_missing_behavioral_measurements():
    from stealth_model_profiler.family.identity import identify
    bank, _ = observations()
    bank["tokenizer_signatures"] = {"one/v1": [3], "two/v1": [3]}
    report = dict(model="blind", method="predictive", ranked=[], leading_families=[],
                  decision=dict(verdict="insufficient calibration", cutoff=.05, gap=None))
    result = identify(report, {}, bank["models"], bank, [3])
    assert result["product_family"]["candidates"] == ["one", "two"]
    assert result["product_family"]["family"] is None
    assert all(r["exact_tokenizer"] for r in result["model_candidates"])


def test_replay_and_family_json_stdout_match(tmp_path, capsys, family_index):
    bank, records = observations()
    run = tmp_path / "run.jsonl"
    write_json(family_index, bank)
    store = RecordStore(run)
    for record in records:
        store.append(record)
    options = ["--json", "-"]
    assert cli.main(["family", "--from-run", str(run), *options]) == 0
    family = json.loads(capsys.readouterr().out)
    assert cli.main(["replay", str(run), *options]) == 0
    assert json.loads(capsys.readouterr().out)["family"]["identity"] == family["identity"]
    assert family["identity"]["model_candidates"][0]["model"] == "one/v1"
    assert "model_claim" not in family


def test_supplied_claim_is_independent_of_family_and_replays(tmp_path, capsys, family_index):
    bank, first = observations()
    bank["claim_verification"] = {"cutoff": .25, "references": bank["models"]}
    run = tmp_path / "recording.jsonl"
    write_json(family_index, bank)
    store = RecordStore(run)
    for i in range(15):
        for record in first:
            store.append(replace(record, tag=record.tag[:-1] + str(i)))
    family = cli._family_report(first, bank, "blind", "predictive", None)
    assert cli._family_report(store.read(), bank, "blind", "predictive", None)["identity"] == family["identity"]
    for claim, expected in [("one/v1", "accepted"), ("one/v2", "rejected"), ("unknown", "not enrolled")]:
        options = ["--verify-model", claim, "--json", "-"]
        assert cli.main(["family", "--from-run", str(run), *options]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["identity"] == family["identity"]
        assert report["model_claim"]["verdict"] == expected
        assert report["model_claim"]["claimed_model"] == claim
        assert cli.main(["replay", str(run), *options]) == 0
        assert json.loads(capsys.readouterr().out)["family"]["model_claim"] == report["model_claim"]
    assert cli._family_report(first, bank, "blind", "predictive", "one/v1")["model_claim"]["verdict"] == "insufficient evidence"
    del bank["claim_verification"]
    assert cli._family_report(first, bank, "blind", "predictive", "one/v1")["model_claim"]["verdict"] == "insufficient calibration"


def test_explicit_claim_collects_repetitions_only_for_enrolled_claim(family_index, capsys, monkeypatch):
    from types import SimpleNamespace
    bank, first = observations()
    bank["claim_verification"] = {"cutoff": .25, "references": bank["models"]}
    write_json(family_index, bank)
    requested = []
    def collect(endpoint, battery, repetitions, method, concurrency):
        requested.append(repetitions)
        return [replace(r, tag=r.tag[:-1] + str(i)) for i in range(repetitions) for r in first]
    monkeypatch.setattr(cli, "_open_endpoint", lambda *args: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli, "collect_family", collect)
    options = ["family", "--model", "opaque", "--json"]
    for claim, expected in [(None, None), ("one/v1", "accepted"), ("other", "not enrolled")]:
        assert cli.main(options + (["--verify-model", claim] if claim else [])) == 0
        report = json.loads(capsys.readouterr().out)
        assert (report.get("model_claim") or {}).get("verdict") == expected
        assert report["identity"]["product_family"]["family"] == "one"
    assert requested == [1, 15, 1]


def test_cli_has_no_family_index_selector():
    parser = cli.build_parser()
    for command in (["family-refs"], ["family", "--model", "blind"], ["profile", "--model", "blind"],
                    ["replay", "run"]):
        parser.parse_args(command)
        for option in (["--compact-coverage"], ["--extended-coverage"], ["--published"], ["--reference-file", "other"]):
            with pytest.raises(SystemExit) as exc:
                parser.parse_args(command + option)
            assert exc.value.code == 2


def test_family_refs_upgrades_the_only_index(family_index, monkeypatch, capsys):
    bank, _ = observations()
    old = {**bank, "taxonomy": "extended-coverage-2026-11"}
    write_json(family_index.with_name("reference-expanded.json"), old)
    assert cli.main(["family", "--model", "blind"]) == 2
    assert "install the family index" in capsys.readouterr().err
    write_json(family_index, old)
    assert cli.main(["family", "--model", "blind"]) == 2
    assert "update the installed family index" in capsys.readouterr().err

    def build(destination):
        assert destination == family_index
        write_json(destination, bank)
        return {**bank, "credit": "test"}
    monkeypatch.setattr(references, "build_index", build)
    assert cli.main(["family-refs"]) == 0
    assert references.load_index(family_index) == bank
