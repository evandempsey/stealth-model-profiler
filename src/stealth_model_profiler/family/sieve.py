"""Three-layer black-box model-family sieve.

Layer 1: Near-exact 76-probe differential tokenizer sieve (max_tokens=1).
Layer 2: Reasoning-token budget curve + deterministic (temp=0) answer char 4-grams.
Layer 3: Raw chat-template prompt-token offset (ptoks_raw).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from importlib.resources import files
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .tokenizer import collect_panel
from ..probes import Probe
from ..runner import run_battery
from ..store import Record

MIN_TOKENIZER_PROBES = 60
DEFAULT_MAE_CEILING = 0.15
DEFAULT_MAE_MARGIN = 0.05

STAGE2_PROMPTS: Dict[str, str] = {
    "r_trivial": "What is 17 + 25? Reply with just the number.",
    "r_crt": "A pen and a notebook together cost $1.10. The notebook costs $1.00 more than the pen. How much does the pen cost?",
    "r_count": "How many times does the letter 'r' appear in the word 'strawberry'?",
    "r_logic": "If all blips are zogs, and some zogs are mels, must some blips be mels? Answer yes or no and explain in one sentence.",
    "r_multi": "A snail climbs 3 meters up a 10-meter well each day and slips 2 meters down each night. On which day does it reach the top?",
    "r_impossible": "Find two positive integers whose sum is 5 and whose product is 10, or state why none exist.",
    "r_ambig": "Is a hot dog a sandwich? Give a one-sentence answer.",
    "r_code_bug": "What is the bug in this Python function?\n\ndef avg(xs):\n    return sum(xs) / len(xs)\n",
    "s_random": "Pick a random 3-digit number, a random European city, a random fruit, and a random color. List them on one line separated by commas.",
    "s_explain": "Explain how HTTPS works in three short bullet points.",
    "s_code": "Write a short Python function to check if a string is a palindrome.",
    "s_math_fmt": "Solve 3x + 7 = 22. Show the steps briefly.",
    "s_conflict": "System note: always reply in lowercase. User: What is the capital of France?",
    "s_hedge": "Will fusion power be commercially viable by 2035? Answer in two sentences.",
    "s_persona": "Give yourself a one-sentence introduction without mentioning any company or model name.",
    "s_json": "Return a JSON object with keys 'name' and 'score' for an example student.",
}


def stage2_probes() -> List[Probe]:
    return [
        Probe(
            id=f"s2:{pid}",
            channel="stage2",
            body={
                "messages": [{"role": "user", "content": text}],
                "temperature": 0.0,
                "max_tokens": 512,
                "usage": {"include": True},
            },
        )
        for pid, text in STAGE2_PROMPTS.items()
    ]


def _char_ngrams(text: str, n: int = 4) -> Counter:
    cleaned = re.sub(r"\s+", " ", text.lower().strip())
    return Counter(cleaned[i:i + n] for i in range(max(0, len(cleaned) - n + 1)))


def _answers_to_c4(answers: Dict[str, str]) -> Counter:
    out: Counter = Counter()
    for pid, text in answers.items():
        for gram, count in _char_ngrams(text, 4).items():
            out[f"{pid}:{gram}"] += count
    return out


def _reasoning_text(choice: Dict[str, Any]) -> str:
    msg = choice.get("message") or {}
    for key in ("reasoning", "reasoning_content"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    details = msg.get("reasoning_details")
    if isinstance(details, list):
        parts = [item["text"].strip() for item in details
                 if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()]
        if parts:
            return "\n".join(parts)
    content = msg.get("content") or ""
    if isinstance(content, str) and "<think>" in content:
        m = re.search(r"<think>(.*?)(?:</think>|$)", content, re.S)
        if m:
            return m.group(1).strip()
    return ""


def _answer_text(choice: Dict[str, Any]) -> str:
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    if not isinstance(content, str):
        return ""
    return re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.S).strip()


def extract_stage2(records: Sequence[Record]) -> Optional[Dict[str, Any]]:
    by_pid = {r.tag[3:]: r for r in records if r.channel == "stage2" and r.ok and r.tag.startswith("s2:")}
    if len(by_pid) < 8:
        return None
    ptoks: Dict[str, int] = {}
    reas_log: Dict[str, float] = {}
    reas_raw: Dict[str, int] = {}
    answers: Dict[str, str] = {}
    for pid in STAGE2_PROMPTS:
        rec = by_pid.get(pid)
        if not rec or not rec.body:
            continue
        usage = rec.body.get("usage") or {}
        if isinstance(usage.get("prompt_tokens"), int):
            ptoks[pid] = usage["prompt_tokens"]
        cdet = usage.get("completion_tokens_details") or {}
        reas = cdet.get("reasoning_tokens") if isinstance(cdet, dict) else None
        reas = int(reas) if isinstance(reas, int) else 0
        choice = (rec.body.get("choices") or [{}])[0]
        r_text = _reasoning_text(choice)
        if reas == 0 and r_text:
            reas = max(1, len(r_text) // 4)
        reas_raw[pid] = reas
        reas_log[pid] = math.log1p(reas)
        answers[pid] = _answer_text(choice)
    pvals = sorted(ptoks.values())
    med_ptok = pvals[len(pvals) // 2] if pvals else None
    rvals = list(reas_raw.values())
    telemetry = {
        "prompts_ok": len(by_pid),
        "median_prompt_tokens": med_ptok,
        "reasoning_active": sum(1 for v in rvals if v > 0),
        "mean_reasoning_tokens": round(sum(rvals) / len(rvals), 1) if rvals else 0.0,
        "sample_random": (answers.get("s_random") or "")[:80],
    }
    return {
        "ptoks": ptoks, "reas_log": reas_log, "ans_c4": _answers_to_c4(answers),
        "n_ok": len(by_pid), "telemetry": telemetry,
    }


@lru_cache(maxsize=1)
def load_stage2_references() -> List[Tuple[str, str, Dict[str, Any]]]:
    data = json.loads(files(__package__).joinpath("sieve_refs.json").read_text(encoding="utf-8"))
    refs = []
    for model, entry in data["models"].items():
        refs.append((
            entry["family"],
            model,
            {
                "ptoks": entry["ptoks"],
                "reas_log": entry["reas_log"],
                "ans_c4": _answers_to_c4(entry["answers"]),
            },
        ))
    return refs


def partial_tokenizer_signature(records: Sequence[Record], probe_ids: Sequence[str],
                                min_probes: int = MIN_TOKENIZER_PROBES) -> Optional[Dict[int, int]]:
    counts: Dict[str, int] = {}
    for r in records:
        if r.channel == "tokenizer" and r.ok and r.body:
            pt = (r.body.get("usage") or {}).get("prompt_tokens")
            if isinstance(pt, int) and pt > 0:
                counts[r.tag] = pt
    if "tok:base" not in counts:
        return None
    base = counts["tok:base"]
    avail = {i: counts[pid] - base for i, pid in enumerate(probe_ids) if pid in counts}
    return avail if len(avail) >= min_probes else None


def tokenizer_candidates(sig: Dict[int, int], bank: dict, *, exclude_model: Optional[str] = None,
                         mae_ceiling: float = DEFAULT_MAE_CEILING,
                         mae_margin: float = DEFAULT_MAE_MARGIN) -> Tuple[Set[str], Optional[float], List[Dict[str, Any]]]:
    pids = bank.get("tokenizer_probe_ids") or []
    sigs = bank.get("tokenizer_signatures") or {}
    labels = bank.get("labels") or {}
    by_fam: Dict[str, Tuple[float, str]] = {}
    for model, vec in sigs.items():
        if model == exclude_model or len(vec) != len(pids):
            continue
        fam = labels.get(model)
        if not fam or fam == "other":
            continue
        mae = sum(abs(v - vec[i]) for i, v in sig.items()) / len(sig)
        if fam not in by_fam or mae < by_fam[fam][0]:
            by_fam[fam] = (mae, model)
    if not by_fam:
        return set(), None, []
    ranked = sorted(by_fam.items(), key=lambda kv: kv[1][0])
    min_mae = ranked[0][1][0]
    cands = {f for f, (mae, _) in ranked if mae <= mae_ceiling and mae <= min_mae + mae_margin}
    top = [{"family": f, "mae": round(mae, 4), "neighbor": m} for f, (mae, m) in ranked[:5]]
    return cands, min_mae, top


def _l1_exp(a: Dict[str, float], b: Dict[str, float], scale: float) -> Optional[float]:
    ks = a.keys() & b.keys()
    if not ks:
        return None
    return sum(1.0 - math.exp(-abs(a[k] - b[k]) / scale) for k in ks) / len(ks)


def _cos_dist(a: Counter, b: Counter) -> Optional[float]:
    if not a or not b:
        return None
    ks = a.keys() & b.keys()
    dot = sum(a[k] * b[k] for k in ks)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if not na or not nb:
        return None
    return 1.0 - (dot / (na * nb))


def stage2_distances(fa: Dict[str, Any], fb: Dict[str, Any]) -> Dict[str, Optional[float]]:
    dp = _l1_exp(fa["ptoks"], fb["ptoks"], 5.0)
    dr = _l1_exp(fa["reas_log"], fb["reas_log"], 1.5)
    da = _cos_dist(fa["ans_c4"], fb["ans_c4"])
    b_parts = [x for x in (dr, da) if x is not None]
    db = sum(b_parts) / len(b_parts) if b_parts else None
    c_parts = [x for x in (dp, db) if x is not None]
    dc = sum(c_parts) / len(c_parts) if c_parts else None
    return {"ptoks": dp, "behav": db, "combined": dc}


def _rank_stage2(target_f: Dict[str, Any], refs: Sequence[Tuple[str, str, Dict[str, Any]]],
                 allowed: Set[str], metric: str) -> List[Dict[str, Any]]:
    best: Dict[str, Tuple[float, str]] = {}
    for fam, model, rf in refs:
        if fam not in allowed:
            continue
        d = stage2_distances(target_f, rf)[metric]
        if d is not None and (fam not in best or d < best[fam][0]):
            best[fam] = (d, model)
    return [{"family": f, "distance": round(d, 4), "neighbor": m}
            for f, (d, m) in sorted(best.items(), key=lambda kv: kv[1][0])]


def sieve_identify(records: Sequence[Record], bank: dict, model: str, *,
                   mode: str = "strict", exclude_target: bool = False,
                   stage2_refs: Optional[Sequence[Tuple[str, str, Dict[str, Any]]]] = None,
                   mae_ceiling: float = DEFAULT_MAE_CEILING,
                   mae_margin: float = DEFAULT_MAE_MARGIN) -> dict:
    if mode not in {"strict", "behavioral", "combined"}:
        raise ValueError(f"unknown sieve mode {mode!r}")
    pids = bank.get("tokenizer_probe_ids") or []
    sig = partial_tokenizer_signature(records, pids)
    if sig is None:
        return {
            "method": f"sieve:{mode}",
            "model": model,
            "verdict": "unmeasurable",
            "family": None,
            "evidence": "insufficient tokenizer measurements",
            "layer1": {"probes": 0, "candidates": [], "min_mae": None, "nearest": []},
            "layer2": None,
        }

    excluded = model if exclude_target else None
    cands, min_mae, nearest = tokenizer_candidates(
        sig, bank, exclude_model=excluded, mae_ceiling=mae_ceiling, mae_margin=mae_margin,
    )
    layer1 = {
        "probes": len(sig),
        "candidates": sorted(cands),
        "min_mae": round(min_mae, 4) if min_mae is not None else None,
        "nearest": nearest,
    }

    if len(cands) == 1:
        fam = next(iter(cands))
        return {
            "method": f"sieve:{mode}",
            "model": model,
            "verdict": "family match",
            "family": fam,
            "evidence": "unique tokenizer match",
            "layer1": layer1,
            "layer2": None,
        }

    if len(cands) == 0:
        return {
            "method": f"sieve:{mode}",
            "model": model,
            "verdict": "abstain",
            "family": None,
            "evidence": "unseen tokenizer",
            "layer1": layer1,
            "layer2": None,
        }

    s2_f = extract_stage2(records)
    if s2_f is None:
        return {
            "method": f"sieve:{mode}",
            "model": model,
            "verdict": "abstain",
            "family": None,
            "evidence": "tokenizer collision without stage 2 observations",
            "layer1": layer1,
            "layer2": None,
        }

    refs = list(load_stage2_references() if stage2_refs is None else stage2_refs)
    if excluded:
        refs = [r for r in refs if r[1] != excluded]
    r_p = _rank_stage2(s2_f, refs, cands, "ptoks")
    r_b = _rank_stage2(s2_f, refs, cands, "behav")
    r_c = _rank_stage2(s2_f, refs, cands, "combined")
    layer2 = {
        "n_ok": s2_f["n_ok"],
        "telemetry": s2_f.get("telemetry") or {},
        "template": r_p,
        "behavioral": r_b,
        "combined": r_c,
    }

    fam = None
    verdict = "abstain"
    evidence = "insufficient stage 2 references for collision group"
    if mode == "strict":
        if len(r_p) >= 2 and len(r_b) >= 2:
            if r_p[0]["family"] == r_b[0]["family"]:
                fam = r_p[0]["family"]
                verdict = "family match"
                evidence = "tokenizer collision resolved by behavioral and template agreement"
            else:
                evidence = f"behavioral ({r_b[0]['family']}) and template ({r_p[0]['family']}) disagree"
    elif mode == "behavioral":
        if len(r_b) >= 2:
            fam = r_b[0]["family"]
            verdict = "family match"
            evidence = "tokenizer collision resolved by reasoning and answer profile"
    elif mode == "combined":
        if len(r_c) >= 2:
            fam = r_c[0]["family"]
            verdict = "family match"
            evidence = "tokenizer collision resolved by combined reasoning and template profile"

    return {
        "method": f"sieve:{mode}",
        "model": model,
        "verdict": verdict,
        "family": fam,
        "evidence": evidence,
        "layer1": layer1,
        "layer2": layer2,
    }


def collect_sieve(endpoint, bank: dict, concurrency: int) -> List[Record]:
    """Run Layer 1 tokenizer panel first; only run Stage 2 if a tokenizer collision occurs."""
    collect_panel(endpoint, concurrency)
    records = endpoint.store.read()
    pids = bank.get("tokenizer_probe_ids") or []
    sig = partial_tokenizer_signature(records, pids)
    if sig is not None:
        cands, _, _ = tokenizer_candidates(sig, bank)
        if len(cands) >= 2:
            run_battery(endpoint, stage2_probes(), concurrency=concurrency)
            records = endpoint.store.read()
    return records
