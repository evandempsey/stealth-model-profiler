"""Passive channel: what the gateway's own catalogue says about a listing.

This costs no model tokens and is sometimes decisive — a declared tokenizer family
or an unusual reasoning-control surface narrows the lineage immediately. It is
also the least trustworthy channel: these fields are published by whoever created
the listing, and a stealth listing has an obvious incentive to be vague. So it is
weighted as a prior, never as proof, and its provenance is always shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"

# OpenRouter's declared tokenizer family -> the lab that vocabulary belongs to.
TOKENIZER_FAMILY_LAB = {
    "qwen3": "alibaba",
    "qwen": "alibaba",
    "mistral": "mistral",
    "deepseek": "deepseek",
    "gpt": "openai",
    "claude": "anthropic",
    "gemini": "google",
    "gemma": "google",
    "llama2": "meta",
    "llama3": "meta",
    "llama4": "meta",
    "grok": "xai",
    "cohere": "cohere",
    "nova": "amazon",
}


@dataclass
class CatalogRecord:
    slug: str
    name: str = ""
    tokenizer_declared: Optional[str] = None
    hugging_face_id: Optional[str] = None
    context_length: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    supported_parameters: List[str] = field(default_factory=list)
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: Dict[str, Any] = field(default_factory=dict)
    knowledge_cutoff: Optional[str] = None
    input_modalities: List[str] = field(default_factory=list)
    output_modalities: List[str] = field(default_factory=list)
    created: Optional[int] = None
    is_moderated: Optional[bool] = None
    per_request_limits: Any = None
    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def declared_lab(self) -> Optional[str]:
        if not self.tokenizer_declared:
            return None
        return TOKENIZER_FAMILY_LAB.get(self.tokenizer_declared.strip().lower())

    def features(self) -> Dict[str, Any]:
        return {
            "tokenizer_declared": self.tokenizer_declared,
            "declared_lab": self.declared_lab,
            "hugging_face_id": self.hugging_face_id,
            "context_length": self.context_length,
            "max_completion_tokens": self.max_completion_tokens,
            "supported_parameters": sorted(self.supported_parameters),
            "default_parameters": self.default_parameters,
            "reasoning": self.reasoning,
            "knowledge_cutoff": self.knowledge_cutoff,
            "input_modalities": sorted(self.input_modalities),
            "output_modalities": sorted(self.output_modalities),
            "is_moderated": self.is_moderated,
            "endpoint_count": len(self.endpoints),
            "endpoint_providers": sorted({e.get("provider_name") for e in self.endpoints
                                          if e.get("provider_name")}),
            "endpoint_quantizations": sorted({e.get("quantization") for e in self.endpoints
                                              if e.get("quantization")}),
        }


def _from_model(model: Dict[str, Any], endpoints: Optional[List[Dict]] = None) -> CatalogRecord:
    arch = model.get("architecture") or {}
    top = model.get("top_provider") or {}
    return CatalogRecord(
        slug=model.get("id", ""),
        name=model.get("name", ""),
        tokenizer_declared=arch.get("tokenizer"),
        hugging_face_id=model.get("hugging_face_id"),
        context_length=model.get("context_length"),
        max_completion_tokens=top.get("max_completion_tokens"),
        supported_parameters=list(model.get("supported_parameters") or []),
        default_parameters=model.get("default_parameters") or {},
        reasoning=model.get("reasoning") or {},
        knowledge_cutoff=model.get("knowledge_cutoff"),
        input_modalities=list(arch.get("input_modalities") or []),
        output_modalities=list(arch.get("output_modalities") or []),
        created=model.get("created"),
        is_moderated=top.get("is_moderated"),
        per_request_limits=model.get("per_request_limits"),
        endpoints=endpoints or [],
        raw=model,
    )


def fetch_catalog(slug: str, timeout: float = 30.0) -> Optional[CatalogRecord]:
    """Fetch one listing plus its per-provider endpoint records.

    Stealth listings are *not* present in the ``/models`` dump — a live check
    showed four stealth models served while the catalogue listed none — so the
    per-model endpoints route is used as a fallback. Without it the tool would
    return nothing for exactly the listings it exists to profile.
    """
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        endpoints: List[Dict] = []
        endpoint_record: Dict = {}
        try:
            ep = client.get(f"{OPENROUTER_MODELS}/{slug}/endpoints")
            if ep.status_code == 200:
                endpoint_record = ep.json().get("data") or {}
                endpoints = endpoint_record.get("endpoints") or []
        except (httpx.HTTPError, ValueError):
            endpoint_record = {}

        try:
            resp = client.get(OPENROUTER_MODELS)
            resp.raise_for_status()
            models = resp.json().get("data") or []
        except (httpx.HTTPError, ValueError):
            models = []

        match = next((m for m in models if m.get("id") == slug), None)
        if match is not None:
            # The catalogue entry is the richer record; use it when present.
            return _from_model(match, endpoints)
        if endpoint_record.get("id") == slug:
            return _from_model(endpoint_record, endpoints)
        return None
