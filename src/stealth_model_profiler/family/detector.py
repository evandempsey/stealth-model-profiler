"""Model-family retrieval using sampled answers and input-token accounting."""

from __future__ import annotations

import math
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

from .tokenizer import recorded_signature, resolve_ambiguity
from ..probes import Probe
from ..runner import run_battery


DEFAULT_GAP_CUTOFFS = {"predictive": 0.05, "counts": 0.20}


@dataclass
class Cell:
    answers: Counter
    prompt_tokens: list[int]

    def as_dict(self) -> dict:
        return {"answers": dict(self.answers), "prompt": statistics.median(self.prompt_tokens)
                if self.prompt_tokens else None}


def add_observation(cells: dict, key: str, answer: str, tokens: int | None) -> None:
    """Accumulate the two signals independently, including usage for empty answers."""
    if key not in cells:
        cells[key] = Cell(Counter(), [])
    cell = cells[key]
    text = unicodedata.normalize("NFKC", answer).strip().casefold()
    text = text.strip(" .,!?:;。，！？؟؛،\"'“”‘’")
    if text and len(text) <= 100 and len(text.split()) == 1:
        cell.answers[text] += 1
    if type(tokens) is int and tokens >= 0:
        cell.prompt_tokens.append(tokens)


def family_probes(battery: dict, repetitions: int) -> list[Probe]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    probes = []
    for rep in range(repetitions):
        for key, messages in battery.items():
            body = {"messages": messages, "temperature": 1, "max_tokens": 16,
                    "reasoning": {"enabled": False}, "usage": {"include": True}}
            probes.append(Probe(f"family:{key}:{rep}", "family", body))
    return probes


def collect_family(endpoint, battery, repetitions, method, concurrency):
    """Check compatibility before dispatch. Mandatory reasoning permits counts only."""
    probes = family_probes(battery, repetitions)
    first = probes[0]
    outcome = endpoint.call(first.channel, first.id, first.body)
    if not outcome.ok:
        message = (outcome.text or "").casefold()
        mandatory = ("reasoning" in message and any(word in message for word in
                     ("mandatory", "cannot be disabled", "cannot disable", "required", "none is not supported")))
        if outcome.status != 400 or not mandatory or method not in {"counts", "predictive"}:
            return endpoint.store.read()
        probes = family_probes(battery, 1)
        for probe in probes:
            del probe.body["reasoning"]
    else:
        probes = probes[1:]
    run_battery(endpoint, probes, concurrency=concurrency)
    return endpoint.store.read()


def recorded_cells(records, battery: dict) -> dict:
    seen = set()
    cells = {}
    for rec in records:
        if rec.channel != "family" or not rec.ok:
            continue
        prefix, key, rep = rec.tag.split(":", 2)
        payload = rec.request
        if (prefix != "family" or key not in battery or payload.get("messages") != battery[key]
                or payload.get("temperature") != 1 or payload.get("max_tokens") != 16
                or ("reasoning" in payload and payload["reasoning"] != {"enabled": False})):
            raise ValueError("recorded sampling protocol differs from the reference battery")
        if (key, rep) in seen:
            continue
        seen.add((key, rep))
        choices = rec.body.get("choices") or []
        choice = choices[0] if choices else {}
        content = (choice.get("message") or {}).get("content")
        usable_answer = isinstance(content, str) and payload.get("reasoning") == {"enabled": False}
        add_observation(cells, key, content if usable_answer else "",
                        (rec.body.get("usage") or {}).get("prompt_tokens"))
    return {key: cell.as_dict() for key, cell in cells.items()}


@lru_cache(maxsize=32768)
def _js_term(a: int, n: int, b: int, m: int) -> float:
    p, q = a / n, b / m
    mid = (p + q) / 2
    return 0.5 * ((p * math.log2(p / mid) if p else 0.0)
                  + (q * math.log2(q / mid) if q else 0.0))


def distances(a: dict, b: dict) -> dict:
    """Compare shared cells. A language needs multiple prompts to estimate its offset."""
    answer_distances = []
    sparse_js = []
    brier = []
    offsets = defaultdict(list)
    for key in sorted(a.keys() & b.keys()):
        left, right = a[key], b[key]
        if left["prompt"] is not None and right["prompt"] is not None:
            offsets[key.split("/", 1)[0]].append(left["prompt"] - right["prompt"])
        p, q = left["answers"], right["answers"]
        n, m = sum(p.values()), sum(q.values())
        if n and m >= 10:
            js = sum(_js_term(p.get(k, 0), n, q.get(k, 0), m)
                     for k in sorted(p.keys() | q.keys()))
            sparse_js.append(js)
            if n >= 10:
                answer_distances.append(js)
            # Expected half-Brier loss of the enrollment distribution q on the
            # target outcomes p. Distinct pairs remove finite-enrollment bias.
            collision = sum(x * (x - 1) for x in q.values()) / (m * (m - 1))
            overlap = sum(x * q.get(k, 0) for k, x in p.items()) / (n * m)
            brier.append((1 + collision - 2 * overlap) / 2)
    discrepancies = []
    for delta in offsets.values():
        if len(delta) < 2:
            continue
        center = statistics.median(delta)
        discrepancies.extend(1 - math.exp(-abs(v - center) / 2) for v in delta)
    # Require half the forty-cell battery for each distance. A single shared
    # condition can otherwise win simply because its prompt count happened to tie.
    js = statistics.mean(answer_distances) if len(answer_distances) >= 20 else None
    counts = statistics.mean(discrepancies) if len(discrepancies) >= 20 else None
    combined = (js + counts) / 2 if js is not None and counts is not None else None
    predictive = (statistics.mean(brier) + counts) / 2 if len(brier) >= 20 and counts is not None else None
    return {"predictive": predictive, "one_answer_js": statistics.mean(sparse_js) if len(sparse_js) >= 20 else None,
            "predictive_cells": len(brier), "answers": js, "counts": counts, "combined": combined,
            "answer_cells": len(answer_distances), "count_cells": len(discrepancies)}


def rank_families(target: dict, references: dict, labels: dict,
                  method: str = "combined") -> list[dict]:
    """Nearest enrolled example per family. Equal-distance families share the lead."""
    best = {}
    for model, reference in sorted(references.items()):
        scores = distances(target, reference)
        distance = scores[method]
        if distance is None:
            continue
        family = labels[model]
        if family not in best or distance < best[family]["distance"]:
            best[family] = dict(family=family, neighbor=model, distance=distance,
                                answer_cells=scores["predictive_cells" if method == "predictive" else "answer_cells"],
                                count_cells=scores["count_cells"])
    return sorted(best.values(), key=lambda item: item["distance"])


def family_result(records, references: dict, model: str, method: str, *,
                  historical_calibration: bool = False, exclude_target: bool = True) -> dict:
    """Score a named family, or reproduce the original bank-calibrated benchmark."""
    from .novelty import decision

    records = list(records)
    observations = [r for r in records if r.channel == "family"]
    cells = recorded_cells(observations, references["battery"])
    repetitions = {r.tag.rsplit(":", 1)[-1] for r in observations if r.ok}
    extended = (not historical_calibration and method == "predictive" and len(repetitions) == 1
                and "tokenizer_signatures" in references)
    if method == "predictive" and any(r.ok and "reasoning" not in r.request for r in observations):
        method = "counts"
    excluded_model = model if exclude_target else None
    enrolled = {k: v for k, v in references["models"].items() if k != excluded_model}
    ranked = rank_families(cells, enrolled, references["labels"], method)
    if extended and method == "predictive" and not ranked and sum(bool(c["answers"]) for c in cells.values()) < 20:
        method = "counts"
        ranked = rank_families(cells, enrolled, references["labels"], method)
    best = [r["family"] for r in ranked if abs(r["distance"] - ranked[0]["distance"]) < 1e-12]
    if not historical_calibration and method in DEFAULT_GAP_CUTOFFS:
        cutoff = DEFAULT_GAP_CUTOFFS[method]
    else:
        cutoffs = (references.get("held_out_cutoffs", {}).get(model, {}) if excluded_model in references["models"]
                   else references.get("cutoffs", {}))
        cutoff = cutoffs.get(method)
    if method == "predictive" and len(repetitions) > 1:
        cutoff = None  # The predictive cutoffs were evaluated on one draw per prompt.
    result = decision(ranked, cutoff)
    if not historical_calibration and result["verdict"] == "family match" and (len(best) != 1 or best[0] == "other"):
        result["verdict"] = "abstain"
    signature = None
    if not historical_calibration and "tokenizer_probe_ids" in references:
        signature = recorded_signature(records, references["tokenizer_probe_ids"])
    if extended and result["verdict"] == "abstain":
        match = resolve_ambiguity(ranked, cutoff, signature, references, excluded_model)
        if match is not None:
            result.update(verdict="family match", family=match, evidence="unique tokenizer match among candidates")
    report = {"task": "nearest enrolled model family", "method": method, "model": model,
              "attempts": len(observations), "leading_families": best, "ranked": ranked,
              "decision": result}
    if not historical_calibration:
        from .identity import identify

        report["identity"] = identify(report, cells, enrolled, references, signature, excluded_model)
    return report
