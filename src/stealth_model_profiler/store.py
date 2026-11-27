"""Append-only record store: every probe call is persisted before it is scored.

The store is the unit of reproducibility. A profiling run can be re-scored offline
(``stealth-model-profiler replay``) without spending another token, and any published claim can be
traced back to the exact request and response that produced it.
"""

from __future__ import annotations

import json
import math
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class Record:
    channel: str
    tag: str
    endpoint: str
    model: str
    request: Dict[str, Any]
    status: Optional[int]
    body: Optional[Dict[str, Any]]
    text: Optional[str]
    latency_ms: float
    ts: float
    error: Optional[str] = None
    stream: Optional[List[Dict[str, Any]]] = None

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return (self.status is not None and 200 <= self.status < 300
                and isinstance(self.body, dict) and bool(self.body)
                and not self.body.get("error"))


class RecordStore:
    """Thread-safe JSONL writer/reader."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: Record) -> None:
        line = json.dumps(record.to_json(), ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("ab+") as fh:
                size = fh.tell()
                if size:
                    fh.seek(-1, 2)
                    if fh.read(1) != b"\n":
                        fh.seek(0)
                        tail = fh.read().rsplit(b"\n", 1)[-1]
                        try:
                            Record(**json.loads(tail))
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f"recording ends in an incomplete record: {self.path}") from exc
                        fh.write(b"\n")
                fh.write((line + "\n").encode("utf-8"))

    def read(self) -> List[Record]:
        out: List[Record] = []
        if not self.path.is_file():
            return out
        with self.path.open("rb") as fh:
            for number, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    out.append(Record(**json.loads(line)))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    if not line.endswith(b"\n"):
                        break  # a torn, uncommitted final write
                    raise ValueError(f"invalid JSON at {self.path}:{number}") from exc
                except TypeError as exc:
                    raise ValueError(f"invalid record at {self.path}:{number}") from exc
        return out

    def __iter__(self) -> Iterator[Record]:
        return iter(self.read())


def new_record(channel: str, tag: str, endpoint: str, model: str,
               request: Dict[str, Any], status: Optional[int],
               body: Optional[Dict[str, Any]], text: Optional[str],
               latency_ms: float, error: Optional[str] = None,
               stream: Optional[List[Dict[str, Any]]] = None) -> Record:
    return Record(channel=channel, tag=tag, endpoint=endpoint, model=model,
                  request=request, status=status, body=body, text=text,
                  latency_ms=latency_ms, ts=time.time(), error=error, stream=stream)


def reported_providers(records) -> dict[str, int]:
    names = (r.body.get("provider") for r in records if r.ok and r.endpoint == "openrouter")
    return dict(Counter(name.strip() for name in names if isinstance(name, str) and name.strip()))


def reported_spend(records) -> float:
    usage = (r.body.get("usage") for r in records if isinstance(r.body, dict))
    costs = (item.get("cost") for item in usage if isinstance(item, dict))
    return sum(c for c in costs if isinstance(c, (int, float)) and math.isfinite(c) and c >= 0)


def write_json(path: Path, payload: Any) -> None:
    """Replace one complete JSON document, including on a concurrent read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8",
                                     delete=False) as fh:
        tmp = Path(fh.name)
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
