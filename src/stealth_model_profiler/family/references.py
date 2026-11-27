"""Build the family index from the pinned PAMELA corpus and the packaged enrollment."""
from __future__ import annotations

import hashlib
import json
import tempfile
from importlib.resources import files
from pathlib import Path
from zipfile import ZipFile

import httpx

from .detector import add_observation, distances
from ..store import write_json
from .novelty import threshold
from .verification import calibrate_claims

DATA = ("21278557", "pamela-publish-data.zip", "160104321694472ba328d48de6f6b93ee962d0e97fb4d212a023efa5a2de9f7c")
CODE = ("21278793", "pamela-publish-code.zip", "8a9c8db47609fd0682a44398e55a4e0b322cf3ae479c3189f0874aae928044ef")
CREDIT = "Tomáš Bruckner (2026), Zenodo 10.5281/zenodo.21278557 (CC BY 4.0) and 10.5281/zenodo.21278793 (MIT). Aggregated from the published observations."


def _archive(spec, directory: Path) -> Path:
    record, filename, digest = spec
    dest = directory / filename
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest:
        return dest
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as fh:
        tmp = Path(fh.name)
    try:
        url = f"https://zenodo.org/records/{record}/files/{filename}"
        with httpx.stream("GET", url, timeout=90, follow_redirects=True) as response:
            response.raise_for_status()
            checksum = hashlib.sha256()
            with tmp.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
                    checksum.update(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError(f"unexpected checksum for {filename}")
        tmp.replace(dest)
        return dest
    finally:
        tmp.unlink(missing_ok=True)


INDEX_VERSION = "extended-coverage-56-2026-11"


def load_index(path: Path) -> dict:
    if not path.is_file():
        raise ValueError("install the family index with 'stealth-model-profiler family-refs'")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 1 or data.get("taxonomy") != INDEX_VERSION:
        raise ValueError("update the installed family index with 'stealth-model-profiler family-refs'")
    return data


def _add_enrollment(index: dict, first: dict) -> None:
    enrollment = json.loads(files(__package__).joinpath("enrollment.json").read_text(encoding="utf-8"))
    models = enrollment["models"]
    signatures = enrollment["tokenizer"]["signatures"]
    if index["models"].keys() & models.keys():
        raise ValueError("duplicate model in family enrollment")
    if enrollment["labels"].keys() - (index["models"].keys() | signatures.keys()):
        raise ValueError("family label has no reference fingerprint")
    index["labels"].update(enrollment["labels"])
    for model, sample in models.items():
        index["models"][model] = sample["cells"]
        index["labels"][model] = sample["label"]
        first[model] = sample["first"]
    if signatures.keys() - index["labels"].keys():
        raise ValueError("tokenizer fingerprint has no family label")
    index["tokenizer_signatures"] = signatures
    index["tokenizer_probe_ids"] = enrollment["tokenizer"]["probes"]


def build_index(output: Path) -> dict:
    """Compile the installed index, including family and model-claim calibration."""
    directory = output.parent
    with ZipFile(_archive(CODE, directory)) as archive:
        prompts = json.loads(archive.read("config/prompts.json"))
        models = json.loads(archive.read("config/models.selected.json"))["models"]
    labels = {m["id"]: m.get("family") or m["family_guess"] for m in models}
    battery = {f"{lang}/{task['id']}": [{"role": "system", "content": prompts["system_prompts"][lang]},
                                        {"role": "user", "content": text}]
               for task in prompts["tasks"] if task["paper"] == 1
               for lang, text in task["prompts"].items()}
    cells = {}
    first_draw = {}
    halves = ({}, {})
    seen = set()
    with ZipFile(_archive(DATA, directory)) as archive:
        cohort = set(json.loads(archive.read("results/divergence.json"))["models"])
        # The normalized study corpus identifies which raw repetitions survived
        # study-wide deduplication. Several pilot runs used different settings.
        with archive.open("data/derived/normalized.jsonl") as stream:
            included = {r["key"] for line in stream if (r := json.loads(line))["temperature"] == 1 and r["run_id"] == "main-02"}
        with archive.open("data/runs/main-02/responses.jsonl") as stream:
            for line in stream:
                r = json.loads(line)
                model = r["model"]
                key = f"{r['lang']}/{r['task_id']}"
                if (r["key"] not in included or r["key"] in seen or model not in cohort
                        or key not in battery or r["temperature"] != 1):
                    continue
                seen.add(r["key"])
                obs = (key, r.get("raw") or "", (r.get("usage") or {}).get("prompt_tokens"))
                add_observation(cells.setdefault(model, {}), *obs)
                add_observation(halves[r["rep"] % 2].setdefault(model, {}), *obs)
                if r["rep"] == 0:
                    add_observation(first_draw.setdefault(model, {}), *obs)
    result = {"schema": 1, "credit": CREDIT, "battery": battery,
              "labels": {m: labels.get(m, m.split("/")[0]) for m in sorted(cohort)},
              "models": {m: {k: v.as_dict() for k, v in f.items()} for m, f in cells.items()}}
    first = {m: {k: v.as_dict() for k, v in cs.items()} for m, cs in first_draw.items()}
    _add_enrollment(result, first)
    result["taxonomy"] = INDEX_VERSION
    result["credit"] += " November product-family labels and supplemental observations by the stealth-model-profiler project."
    result["claim_verification"] = calibrate_claims(*(
        {m: {k: cell.as_dict() for k, cell in data.items()} for m, data in half.items()}
        for half in halves))
    ordered = neighbor_tables(result["models"], first)
    result["cutoffs"] = {method: threshold(table, result["labels"]) for method, table in ordered.items()}
    result["held_out_cutoffs"] = {
        model: {method: threshold(table, result["labels"], excluded_models={model})
                for method, table in ordered.items()}
        for model in result["models"]
    }
    write_json(output, result)
    return result


def neighbor_tables(full, first):
    methods = ("predictive", "counts", "combined", "answers")
    ordered = {method: {} for method in methods}
    for model, enrollment in full.items():
        rows = {method: [] for method in methods}
        for other, reference in full.items():
            if model == other:
                continue
            sparse = distances(first.get(model, {}), reference)
            dense = distances(enrollment, reference)
            for method in methods:
                score = (sparse if method in {"predictive", "counts"} else dense)[method]
                if score is not None:
                    rows[method].append((score, other))
        for method in methods:
            ordered[method][model] = sorted(rows[method])
    return ordered
