"""Discover stealth listings and tell new ones from old.

The catalogue is not a usable source: OpenRouter omits stealth listings from
``/api/v1/models`` entirely, so a watcher built on that dump would miss every
launch. The ``/stealth`` page is the authoritative surface, and it embeds a
schema.org ``ItemList`` naming the current slots.

Name patterns are also unreliable — the programme has used ``*-alpha`` and plain
codewords — so slots are keyed by slug and diffed against the previous check.

Liveness is decided by an actual one-token completion, not by presence on the
page: retired slots keep a live documentation page and a 200 on the metadata
route, but a real request returns 404 carrying the reveal notice.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .store import write_json

STEALTH_URL = "https://openrouter.ai/stealth"
FLIGHT_CHUNK = re.compile(r'self\.__next_f\.push\(\[1,(".*?")\]\)', re.S)
SLUG = re.compile(r"stealth/[a-z0-9][a-z0-9._-]*")

# Retired slots announce themselves in one of two shapes: the stealth page says
# "use X at <url>", while a real request returns "this model was X. Use it now:
# <url>". Both are worth reading, and neither is guaranteed to be present.
NEXT_AT = re.compile(r"use\s+(.+?)\s+at\s+(https://openrouter\.ai/\S+?)[.)]", re.I)
NEXT_NOW = re.compile(r"use it now:?\s*(https://openrouter\.ai/\S+?)[.\s]*$", re.I)
# Sentence end, not the first dot: model names contain dots ("GLM-5.3 Flash").
ITEM_LIST = re.compile(r'"@type"\s*:\s*"ItemList"')
REVEALED = re.compile(r"this model was (?:an early version of )?(.+?)\.(?:\s+use it now|\s*$)",
                      re.I | re.S)
NO_PROVIDER = re.compile(r"no provider serves", re.I)


@dataclass
class Slot:
    slug: str
    name: str = ""
    live: Optional[bool] = None
    note: str = ""
    successor: str = ""
    revealed: str = ""

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def _unescape(text: str) -> str:
    return text.replace('\\"', '"').replace("\\\\", "\\")


def parse_slots(html: str) -> List[Slot]:
    """Extract slots from the page's schema.org ItemList, with a slug fallback."""
    slots: Dict[str, Slot] = {}

    chunks = FLIGHT_CHUNK.findall(html)
    payload = ""
    for chunk in chunks:
        try:
            payload += _unescape(json.loads(chunk))
        except (ValueError, TypeError):
            continue
    source = payload or html

    marker = ITEM_LIST.search(source)
    if marker:
        start = source.rfind("{", 0, marker.start())
        try:
            obj, _ = json.JSONDecoder().raw_decode(source, start)
        except (ValueError, TypeError):
            obj = {}
        for item in obj.get("itemListElement") or []:
            match = SLUG.search(item.get("item") or "")
            if match:
                slots[match.group(0)] = Slot(slug=match.group(0),
                                             name=item.get("name") or "")

    if not slots:  # the page shape can change; degrade rather than fail
        for slug in SLUG.findall(source):
            slug = slug.rstrip(".")
            slots[slug] = Slot(slug=slug)

    return sorted(slots.values(), key=lambda s: s.slug)


def interpret_message(message: str) -> Tuple[str, str, str]:
    """Read a retired-slot notice: (revealed name, successor url, remark)."""
    message = message or ""
    revealed = ""
    successor = ""

    match = NEXT_AT.search(message)
    if match:
        revealed = match.group(1).strip()
        successor = match.group(2).strip()
    else:
        match = NEXT_NOW.search(message)
        if match:
            successor = match.group(1).strip()

    named = REVEALED.search(message)
    if named and not revealed:
        revealed = named.group(1).strip()

    if NO_PROVIDER.search(message):
        return "", "", "no provider serves this model anymore"
    return revealed, successor, ""


def check_liveness(slots: List[Slot], store, max_calls: int = 64, max_spend: float = 2.0) -> None:
    """Send one minimal completion to each slot; a 404 means it is retired."""
    from .client import Budget, Endpoint
    from .config import get_provider

    provider = get_provider("openrouter")
    budget = Budget(max_calls=max_calls, max_spend_usd=max_spend)
    for slot in slots:
        endpoint = Endpoint(provider=provider, model=slot.slug, store=store,
                            budget=budget, max_retries=1)
        try:
            outcome = endpoint.call("stealth-watch", f"watch:{slot.slug}",
                                {"messages": [{"role": "user", "content": "hi"}],
                                 "max_tokens": 1, "temperature": 0.0})
        finally:
            endpoint.close()
        slot.live = True if outcome.ok else False if outcome.status in (404, 410) else None
        if outcome.ok:
            slot.note = ""
            continue
        body = outcome.body or {}
        error = body.get("error")
        message = str(error.get("message") or "") if isinstance(error, dict) else ""
        if slot.live is False:
            slot.revealed, slot.successor, slot.note = interpret_message(message)
        else:
            slot.note = f"liveness check failed ({outcome.status or 'transport error'})"


def load_state(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(path: Path, slots: List[Slot]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checked": time.time(), "slots": {s.slug: s.to_json() for s in slots}}
    write_json(path, payload)


def new_slugs(previous: Dict[str, Any], current: List[Slot]) -> List[str]:
    """Slugs absent from the previous snapshot.

    An empty snapshot is a baseline, not a discovery, so it reports nothing; a
    fresh install should not announce four "new" models that are months old.
    """
    old = set((previous.get("slots") or {}).keys())
    if not old:
        return []
    return sorted(s.slug for s in current if s.slug not in old)


def fetch_slots(timeout: float = 30.0) -> List[Slot]:
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(STEALTH_URL)
        resp.raise_for_status()
        return parse_slots(resp.text)
