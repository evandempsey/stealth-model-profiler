"""Measured tokenizer evidence for resolving ambiguous product-family rankings."""
from __future__ import annotations

from dataclasses import replace

from .novelty import relative_gap
from ..probes import tokenizer_probes
from ..runner import run_battery

REPEAT_BASELINE = "repeat-baseline"


def _panel(compatible):
    probes = tokenizer_probes()
    if compatible:
        for probe in probes:
            probe.body.pop("temperature")
            probe.body["max_tokens"] = 16
    return probes


def collect_panel(endpoint, concurrency):
    probes = _panel(False)
    first = probes[0]
    answer = endpoint.call(first.channel, first.id, dict(first.body, usage={"include": True}))
    error = (answer.body or {}).get("error") or {}
    if not answer.ok and (answer.status == 400 or isinstance(error, dict) and error.get("code") == 400):
        probes = _panel(True)
        first = probes[0]
        answer = endpoint.call(first.channel, first.id, dict(first.body, usage={"include": True}))
    if answer.ok:
        run_battery(endpoint, [replace(p, body=dict(p.body, usage={"include": True})) for p in probes[1:]],
                    concurrency=concurrency)
        endpoint.call(first.channel, REPEAT_BASELINE, dict(first.body, usage={"include": True}))


def recorded_signature(records, probe_ids):
    """Require the complete ordered panel, a stable baseline, and a consistent route."""
    readings = [r for r in records if r.channel == "tokenizer" and r.ok]
    if not readings or readings[0].tag != "tok:base" or readings[-1].tag != REPEAT_BASELINE:
        return None
    probes = _panel("temperature" not in readings[0].request)
    if probe_ids != [p.id for p in probes[1:]]:
        return None
    expected = {p.id: p.body for p in probes}
    expected[REPEAT_BASELINE] = probes[0].body
    if len(readings) != len(expected):
        return None
    route = set()
    for r in records:
        if r.ok and r.channel in {"tokenizer", "family"}:
            provider = r.body.get("provider")
            if r.endpoint != "openrouter" or not isinstance(provider, str) or not provider:
                return None
            route.add((r.model, provider))
    if len(route) != 1:
        return None
    counts = {}
    for r in readings:
        body = {k: v for k, v in r.request.items() if k not in {"model", "provider", "usage"}}
        value = (r.body.get("usage") or {}).get("prompt_tokens")
        if body != expected.get(r.tag) or r.tag in counts or type(value) is not int or value <= 0:
            return None
        counts[r.tag] = value
    if counts["tok:base"] != counts[REPEAT_BASELINE]:
        return None
    return [counts[tag] - counts["tok:base"] for tag in probe_ids]


def compare_signatures(signature, references, exclude):
    if signature is None:
        return
    for model, vector in references.get("tokenizer_signatures", {}).items():
        if (model == exclude or not isinstance(vector, list) or len(vector) != len(signature)
                or any(type(value) is not int for value in vector)):
            continue
        yield model, references["labels"][model], vector == signature


def resolve_ambiguity(ranked, cutoff, signature, references, exclude):
    if signature is None:
        return None
    first = ranked[0]["distance"]
    candidates = {r["family"] for r in ranked if abs(r["distance"] - first) < 1e-12
                  or relative_gap(first, r["distance"]) <= cutoff + 1e-12}
    if len(candidates) < 2 or "other" in candidates:
        return None
    measured, matches = set(), set()
    for _, family, exact in compare_signatures(signature, references, exclude):
        measured.add(family)
        if exact and family in candidates:
            matches.add(family)
    if candidates <= measured and len(matches) == 1:
        return matches.pop()
    return None
