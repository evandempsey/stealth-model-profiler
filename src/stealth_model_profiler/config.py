"""OpenRouter endpoint configuration and credential loading."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

OPENAI = "openai"


@dataclass(frozen=True)
class Provider:
    name: str
    protocol: str
    base_url: str
    key_env: str
    default_model: str
    lab: str
    extra_headers: Mapping[str, str] = field(default_factory=dict)


OPENROUTER = Provider(
    "openrouter",
    OPENAI,
    "https://openrouter.ai/api/v1",
    "OPENROUTER_API_KEY",
    "stealth/ox-alpha",
    "openrouter",
)

PROVIDERS: Dict[str, Provider] = {"openrouter": OPENROUTER}

LABS: Dict[str, str] = {
    "zai": "Z.ai / Zhipu",
    "minimax": "MiniMax",
    "moonshot": "Moonshot AI",
    "deepseek": "DeepSeek",
    "alibaba": "Alibaba (Qwen)",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "mistral": "Mistral",
    "xai": "xAI",
    "meta": "Meta",
    "xiaomi": "Xiaomi (MiMo)",
    "meituan": "Meituan (LongCat)",
}

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def parse_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if val[:1] in ("'", '"') and val[-1:] == val[:1] and len(val) >= 2:
            val = val[1:-1]
        out[key] = val
    return out


def default_env_candidates() -> Iterable[Path]:
    env_override = os.environ.get("MPROF_ENV_FILE")
    if env_override:
        yield Path(env_override).expanduser()
        return
    yield Path.cwd() / ".env"
    yield Path.home() / ".config" / "model-profiler" / ".env"


def load_credentials(explicit: Optional[str] = None) -> tuple[Dict[str, str], Optional[Path]]:
    candidates = [Path(explicit).expanduser()] if explicit else list(default_env_candidates())
    loaded: Dict[str, str] = {}
    used: Optional[Path] = None
    allowed = {p.key_env for p in PROVIDERS.values()}
    for path in candidates:
        if not path.is_file():
            continue
        used = path
        for key, val in parse_env_file(path).items():
            if key not in allowed or not val:
                continue
            if key not in os.environ:
                loaded[key] = os.environ[key] = val
        break
    return loaded, used


def get_provider(name: str = "openrouter") -> Provider:
    key = name.lower()
    if key not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; known: {', '.join(sorted(PROVIDERS))}")
    return PROVIDERS[key]
