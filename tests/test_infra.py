"""Infrastructure invariants: boundary maths, budget accounting, probe hygiene."""

from __future__ import annotations

import json

import pytest

from stealth_model_profiler.client import Budget, BudgetExceeded
from stealth_model_profiler.probes import BASE, tokenizer_probes
from stealth_model_profiler.report import render_unavailable
from stealth_model_profiler.stealth import Slot, interpret_message, new_slugs, parse_slots
from stealth_model_profiler.store import Record
from stealth_model_profiler.vocab import Vocab, VocabSpec


def test_delta_is_measured_on_the_concatenation():
    spec = VocabSpec("joiner", "test", "Joiner", "tiktoken", "joiner")
    vocab = Vocab(spec, lambda s: 1 if s == "abcd" else len(s))
    assert vocab.count("ab") == 2
    assert vocab.count("cd") == 2
    assert vocab.delta("ab", "cd") == -1
    assert vocab.count("cd") != vocab.delta("ab", "cd")


def test_budget_counts_calls_and_spend_independently():
    budget = Budget(max_calls=2, max_spend_usd=1.0)
    budget.charge()
    budget.add_spend(0.25)
    assert budget.calls == 1
    assert budget.spend_usd == pytest.approx(0.25)
    budget.charge()
    assert budget.calls == 2
    assert budget.spend_usd == pytest.approx(0.25)
    with pytest.raises(BudgetExceeded):
        budget.charge()


def test_spend_limit_is_enforced_separately():
    budget = Budget(max_calls=10, max_spend_usd=0.10)
    budget.add_spend(0.05)
    with pytest.raises(BudgetExceeded):
        budget.add_spend(0.06)
    assert budget.calls == 0


def test_tokenizer_probes_are_unique_and_share_prefix():
    probes = tokenizer_probes()
    ids = [p.id for p in probes]
    assert len(ids) == len(set(ids))
    assert probes[0].id == "tok:base"
    for probe in probes[1:]:
        assert probe.body["messages"][0]["content"].startswith(BASE), probe.id


def test_unavailable_run_reports_the_reason_and_does_not_rank():
    body = {"error": {"message": "This model was ZAI's GLM-5.3 Flash.", "code": 404}}
    records = [Record(channel="tokenizer", tag=f"tok:{i}", endpoint="openrouter",
                      model="stealth/x", request={}, status=404, body=body,
                      text='{"error":{"message":"This model was ZAI\'s GLM-5.3 Flash."}}',
                      latency_ms=1.0, ts=0.0) for i in range(3)]
    out = render_unavailable(target="stealth/x", endpoint="openrouter",
                             model="stealth/x", records=records, calls=3)
    assert "LISTING UNAVAILABLE" in out
    assert "0 of 3" in out
    assert "GLM-5.3 Flash" in out
    assert "→" not in out


def test_interpret_retired_notice_with_dotted_model_name():
    revealed, successor, note = interpret_message(
        "Thank you for participating in the Stealth Ox Alpha testing period. "
        "This model was ZAI's GLM-5.3 Flash. "
        "Use it now: https://openrouter.ai/z-ai/glm-5.3-flash")
    assert revealed == "ZAI's GLM-5.3 Flash"
    assert successor == "https://openrouter.ai/z-ai/glm-5.3-flash"
    assert note == ""


def test_interpret_notice_that_names_no_successor():
    revealed, successor, note = interpret_message(
        "stealth/space-bunny-alpha is no longer available on OpenRouter: "
        "no provider serves this model anymore.")
    assert revealed == "" and successor == ""
    assert "no provider serves" in note


def test_parse_slots_reads_the_embedded_item_list():
    item_list = {"@context": "https://schema.org", "@type": "ItemList",
                 "itemListElement": [
                     {"@type": "ListItem", "position": 1, "name": "Gestalt Alpha",
                      "item": "https://openrouter.ai/stealth/gestalt-alpha"},
                     {"@type": "ListItem", "position": 2, "name": "Ox Alpha",
                      "item": "https://openrouter.ai/stealth/ox-alpha"}]}
    html = f"<script>self.__next_f.push([1,{json.dumps(json.dumps(item_list))}])</script>"
    slots = parse_slots(html)
    assert [s.slug for s in slots] == ["stealth/gestalt-alpha", "stealth/ox-alpha"]
    assert slots[0].name == "Gestalt Alpha"


def test_new_slugs_treats_first_run_as_baseline():
    current = [Slot("stealth/a"), Slot("stealth/b")]
    assert new_slugs({}, current) == []
    previous = {"slots": {"stealth/a": {}}}
    assert new_slugs(previous, current) == ["stealth/b"]
