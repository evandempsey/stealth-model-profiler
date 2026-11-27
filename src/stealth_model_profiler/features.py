"""Extract tokenizer delta features from raw probe records."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .client import normalize_usage
from .store import Record


def _by_tag(records: Iterable[Record], channel: str) -> Dict[str, Record]:
    out: Dict[str, Record] = {}
    for rec in records:
        if rec.channel != channel or not rec.ok:
            continue
        out[rec.tag] = rec
    return out


def tokenizer_features(records: List[Record]) -> Dict[str, Any]:
    tagged = _by_tag(records, "tokenizer")
    base = tagged.get("tok:base")
    if not base:
        return {}
    base_tokens = normalize_usage(base.body).get("prompt")
    if base_tokens is None:
        return {}
    deltas: Dict[str, int] = {}
    for tag, rec in tagged.items():
        if not tag.startswith("tok:") or tag == "tok:base":
            continue
        val = normalize_usage(rec.body).get("prompt")
        if val is None:
            continue
        deltas[tag[4:]] = val - base_tokens
    return {"base_prompt_tokens": base_tokens, "deltas": deltas}
