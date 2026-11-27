"""Concurrent probe battery runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence

from .client import Endpoint
from .probes import Probe
from .store import Record


def run_battery(endpoint: Endpoint, probes: Sequence[Probe], *,
                concurrency: int = 4) -> List[Record]:
    before = len(endpoint.store.read())
    if probes:
        workers = max(1, min(concurrency, len(probes)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(
                lambda probe: endpoint.call(probe.channel, probe.id, probe.body,
                                            raw=probe.raw, collect_stream=probe.stream),
                probes,
            ))
    return endpoint.store.read()[before:]
