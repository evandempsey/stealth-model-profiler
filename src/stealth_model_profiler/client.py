"""HTTP client with budget, retry, and JSONL recording."""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from .config import OPENAI, Provider
from .store import RecordStore, new_record

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    """Hard stop on request count and (where the gateway reports it) spend.

    Charged from several worker threads during a parallel battery, so the counter
    update is guarded; a race here would let a run overspend its own limit.
    """

    max_calls: int = 4000
    max_spend_usd: Optional[float] = 2.0
    calls: int = 0
    spend_usd: float = 0.0
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def charge(self) -> None:
        """Account for one outbound request attempt."""
        with self._lock:
            if self.max_spend_usd is not None and self.spend_usd >= self.max_spend_usd:
                raise BudgetExceeded(f"reported spend reached ${self.spend_usd:.4f}")
            if self.calls + 1 > self.max_calls:
                raise BudgetExceeded(f"call budget exhausted ({self.max_calls} calls)")
            self.calls += 1

    def add_spend(self, cost: Optional[float]) -> None:
        """Record reported spend.

        Kept separate from :meth:`charge` because a priced response is not a new
        request; folding the two together double-counts every call.
        """
        if not cost or not math.isfinite(cost) or cost < 0:
            return
        with self._lock:
            self.spend_usd += cost
            if self.max_spend_usd is not None and self.spend_usd >= self.max_spend_usd:
                raise BudgetExceeded(
                    f"spend budget exhausted (${self.max_spend_usd:.2f}); "
                    f"spent ${self.spend_usd:.4f}"
                )


@dataclass
class CallOutcome:
    ok: bool
    status: Optional[int]
    body: Optional[Dict[str, Any]]
    text: Optional[str]
    latency_ms: float
    attempts: int = 1
    error: Optional[str] = None
    stream: Optional[List[Dict[str, Any]]] = None

    @property
    def usage(self) -> Dict[str, Any]:
        if not self.body:
            return {}
        return self.body.get("usage") or self.body.get("usageMetadata") or {}


def normalize_usage(body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not body:
        return {}
    raw = body.get("usage") or {}
    if not isinstance(raw, dict):
        return {}
    return {
        "raw": raw,
        "prompt": raw.get("prompt_tokens"),
        "completion": raw.get("completion_tokens"),
        "total": raw.get("total_tokens"),
    }


@dataclass
class Endpoint:
    provider: Provider
    model: str
    store: RecordStore
    budget: Budget = field(default_factory=Budget)
    timeout: float = 120.0
    max_retries: int = 4
    provider_pin: Optional[str] = None
    _client: Optional[httpx.Client] = None

    def __post_init__(self) -> None:
        if self.max_retries < 1:
            raise ValueError("max_retries must allow at least one attempt")
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(self.timeout, connect=15.0))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    # -- request construction -------------------------------------------------

    def _build(self, body: Dict[str, Any], stream: bool) -> tuple[str, Dict[str, str], Dict[str, Any]]:
        provider = self.provider
        if provider.protocol != OPENAI:
            raise ValueError(f"unsupported protocol {provider.protocol!r}")
        key = os.environ.get(provider.key_env, "")
        headers = {"Content-Type": "application/json", **provider.extra_headers}
        url = provider.base_url.rstrip("/") + "/chat/completions"
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": self.model, **body}
        if self.provider_pin and "provider" not in payload:
            payload["provider"] = {"order": [self.provider_pin], "allow_fallbacks": False}
        if stream:
            payload["stream"] = True
            payload.setdefault("stream_options", {"include_usage": True})
        return url, headers, payload

    # -- execution ------------------------------------------------------------

    def call(self, channel: str, tag: str, body: Dict[str, Any], *,
             raw: bool = False, collect_stream: bool = False) -> CallOutcome:
        """Issue one request.

        ``raw=True`` sends ``body`` untouched (minus the model) and is only valid
        for the OpenAI protocol; it exists for malformed-request probes.
        """
        if raw and self.provider.protocol != OPENAI:
            raise ValueError("raw bodies are only supported on the OpenAI protocol")
        url, headers, payload = self._build(body, stream=collect_stream)
        attempts = 0
        last_error: Optional[str] = None
        stream_events: Optional[List[Dict[str, Any]]] = None

        while attempts < self.max_retries:
            attempts += 1
            self.budget.charge()
            started = time.perf_counter()
            try:
                if collect_stream:
                    status, text, parsed, events = self._post_stream(url, headers, payload)
                    stream_events = events
                else:
                    resp = self._client.post(url, headers=headers, json=payload)
                    status = resp.status_code
                    text = resp.text
                    try:
                        parsed = resp.json()
                    except (json.JSONDecodeError, ValueError):
                        parsed = None
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                latency = (time.perf_counter() - started) * 1000
                outcome = CallOutcome(False, None, None, None, latency, attempts, last_error)
                self._record(channel, tag, payload, outcome)
                if attempts >= self.max_retries:
                    return outcome
                time.sleep(min(2 ** attempts * 0.5, 8) * (0.7 + random.random() * 0.6))
                continue

            latency = (time.perf_counter() - started) * 1000
            error = None
            if status >= 400:
                error = (text or "")[:2000]
            outcome = CallOutcome(
                ok=200 <= status < 300 and isinstance(parsed, dict) and bool(parsed) and not parsed.get("error"),
                status=status,
                body=parsed if isinstance(parsed, dict) else None,
                text=text,
                latency_ms=latency,
                attempts=attempts,
                error=error,
                stream=stream_events,
            )
            self._record(channel, tag, payload, outcome)
            usage = (parsed or {}).get("usage") if isinstance(parsed, dict) else None
            if isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float)):
                self.budget.add_spend(usage["cost"])
            if status in RETRY_STATUS and attempts < self.max_retries:
                time.sleep(min(2 ** attempts * 0.5, 8) * (0.7 + random.random() * 0.6))
                continue
            return outcome

        latency = (time.perf_counter() - started) * 1000
        outcome = CallOutcome(False, None, None, None, latency, attempts, last_error or "retries exhausted")
        self._record(channel, tag, payload, outcome)
        return outcome

    def _post_stream(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]):
        events: List[Dict[str, Any]] = []
        collected: List[str] = []
        status = 0
        with self._client.stream("POST", url, headers=headers, json=payload) as resp:
            status = resp.status_code
            for line in resp.iter_lines():
                collected.append(line)
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    events.append(json.loads(data))
                except json.JSONDecodeError:
                    continue
        text = "\n".join(collected)
        parsed = None
        if status < 400:
            for ev in events:
                if isinstance(ev, dict) and ev.get("choices"):
                    parsed = ev
                    break
            usage = next((ev["usage"] for ev in reversed(events)
                          if isinstance(ev, dict) and isinstance(ev.get("usage"), dict)), None)
            if usage is not None:
                parsed = {**(parsed or {}), "usage": usage}
        else:
            try:
                parsed = json.loads(text)
            except ValueError:
                pass
        return status, text, parsed, events

    def _record(self, channel: str, tag: str, payload: Dict[str, Any], outcome: CallOutcome) -> None:
        redacted = dict(payload)
        self.store.append(new_record(
            channel=channel,
            tag=tag,
            endpoint=self.provider.name,
            model=self.model,
            request=redacted,
            status=outcome.status,
            body=outcome.body,
            text=outcome.text,
            latency_ms=outcome.latency_ms,
            error=outcome.error,
            stream=outcome.stream,
        ))
