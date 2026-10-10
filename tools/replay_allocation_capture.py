"""Adapt actual bounded device and native-heap journals to allocation receipts.

This module does not install an allocator. Native device events come from the
existing ledger; the optional pinned Memray collector owns host interception.
Neither adapter authenticates source/build identity or endpoint completion.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from tools.audit_replay_allocations import InvalidReceipt, _integer, _require

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

HEAP_ALLOCATORS = frozenset(
    (
        "MALLOC",
        "REALLOC",
        "CALLOC",
        "POSIX_MEMALIGN",
        "ALIGNED_ALLOC",
        "MEMALIGN",
        "VALLOC",
        "PVALLOC",
    )
)
EXCLUDED_ALLOCATORS = frozenset(
    (
        "MMAP",
        "MUNMAP",
        "PYMALLOC_FREE",
        "PYMALLOC_MALLOC",
        "PYMALLOC_CALLOC",
        "PYMALLOC_REALLOC",
    )
)


def normalize_heap_records(
    records: Iterable[Any],
    allocator_names: Mapping[int, str],
    *,
    markers: Mapping[str, tuple[int, int]],
    max_events: int = 1 << 20,
) -> dict[str, Any]:
    """Normalize malloc-family requests begun during one continuous tracker.

    Unknown/pre-tracker frees are outside this owner, never invented as tracked
    releases. Mapped memory and Python object-suballocator events are separate
    ownership layers and explicitly excluded. Marker matches must be unique;
    source address reuse must not turn an old allocation into a phase boundary.
    """
    _require(
        type(max_events) is int and 1 <= max_events <= 1 << 20,
        "invalid host event limit",
    )
    events = []
    live: dict[int, tuple[str, int]] = {}
    excluded: dict[str, int] = {}
    positions: dict[str, int] = {}
    raw_count = 0
    for raw_count, record in enumerate(records, 1):
        _require(raw_count <= max_events, "host capture exceeds its finite event limit")
        _require(
            _integer(record.n_allocations, "host request count") == 1,
            "aggregated host records cannot prove event coverage",
        )
        kind = allocator_names.get(record.allocator)
        address = _integer(record.address, "host allocation address")
        size = _integer(record.size, "host requested bytes")
        if kind in EXCLUDED_ALLOCATORS:
            excluded[kind] = excluded.get(kind, 0) + 1
            continue
        if kind == "FREE":
            owned = live.pop(address, None)
            if owned is None:
                excluded["pre-tracker/untracked free"] = (
                    excluded.get("pre-tracker/untracked free", 0) + 1
                )
                continue
            name, size = owned
            operation = "release"
        else:
            _require(kind in HEAP_ALLOCATORS, "unsupported host allocator kind")
            _require(address != 0, "null successful host allocation")
            _require(address not in live, "host address reused without a release event")
            name = f"host-request-{raw_count}"
            live[address] = name, size
            operation = "allocate"
            for marker, identity in markers.items():
                if identity == (address, size):
                    _require(marker not in positions, "ambiguous/reused phase marker")
                    positions[marker] = len(events)
        events.append(
            {
                "sequence": len(events),
                "kind": operation,
                "allocation_id": name,
                "requested_bytes": size,
            }
        )
    _require(set(positions) == set(markers), "unobserved host phase marker")
    return {
        "domain": {
            "owner": "memray-malloc-family",
            "space": "host",
            "counter": "memray-1.20.0-native-heap",
        },
        "initial_live": {},
        "events": events,
        "event_end": len(events),
        "dropped_events": 0,
        "marker_positions": positions,
        "excluded": excluded,
        "raw_count": raw_count,
    }


def read_memray_heap(
    path: Path,
    *,
    markers: Mapping[str, tuple[int, int]],
    max_events: int = 1 << 20,
) -> dict[str, Any]:
    """Require finalized, nonaggregated data from the qualified profiler version."""
    import memray

    _require(memray.__version__ == "1.20.0", "host adapter requires Memray 1.20.0")
    with memray.FileReader(str(path)) as reader:
        metadata = reader.metadata
        _require(
            metadata.file_format == memray.FileFormat.ALL_ALLOCATIONS,
            "aggregated host capture",
        )
        _require(
            not metadata.trace_python_allocators,
            "Python suballocator capture is a different owner",
        )
        _require(
            metadata.end_time >= metadata.start_time, "host capture was not finalized"
        )
        journal = normalize_heap_records(
            reader.get_allocation_records(),
            {int(value): value.name for value in memray.AllocatorType},
            markers=markers,
            max_events=max_events,
        )
        _require(
            journal["raw_count"] == metadata.total_allocations,
            "host event/footer count mismatch",
        )
    return journal


class JournalWindows:
    """Carry live owners and exact event cursors across every declared phase."""

    def __init__(self) -> None:
        self.domain: dict[str, str] | None = None
        self.initial: dict[str, int] | None = None
        self.live: dict[str, int] = {}
        self.cursor = 0
        self.prefix: list[dict[str, Any]] = []

    def take(self, journal: dict[str, Any], end: int | None = None) -> dict[str, Any]:
        """Return one strict observation, refusing dropped or rewritten history.

        The caller independently establishes synchronized endpoint boundaries
        and source-matched capture. These coverage assertions cannot be inferred
        from a counter snapshot or supplied by a synthetic fixture alone.
        """
        _require(journal["dropped_events"] == 0, "journal dropped events")
        _require(
            journal["event_end"] == len(journal["events"]), "journal has lost events"
        )
        if self.domain is None:
            self.domain = dict(journal["domain"])
            self.initial = dict(journal["initial_live"])
            self.live = dict(self.initial)
        _require(
            journal["domain"] == self.domain
            and journal["initial_live"] == self.initial,
            "journal owner/initial snapshot changed",
        )
        events = journal["events"]
        _require(events[: self.cursor] == self.prefix, "journal history changed")
        end = journal["event_end"] if end is None else _integer(end, "window event end")
        _require(
            self.cursor <= end <= journal["event_end"], "invalid window event range"
        )
        initial_live = dict(self.live)
        live = dict(self.live)
        live_bytes = peak = sum(live.values())
        count = requested = 0
        window_events = copy.deepcopy(events[self.cursor : end])
        for sequence, event in enumerate(window_events, self.cursor):
            _require(event["sequence"] == sequence, "journal event reordered")
            name = event["allocation_id"]
            size = _integer(event["requested_bytes"], "event bytes")
            if event["kind"] == "allocate":
                _require(name not in live, "journal owner already live")
                live[name] = size
                count += 1
                requested += size
                live_bytes += size
                peak = max(peak, live_bytes)
            elif event["kind"] == "release":
                _require(
                    name in live and live[name] == size,
                    "journal release ownership mismatch",
                )
                del live[name]
                live_bytes -= size
            else:
                raise InvalidReceipt("unknown journal event kind")
        metrics = {
            "allocation_count": count,
            "requested_bytes": requested,
            "peak_live_bytes": peak,
            "live_bytes": live_bytes,
        }
        for name, value in metrics.items():
            _integer(value, name)
        observation = {
            "domain": dict(self.domain),
            "coverage": {
                "complete": True,
                "started_before_window": True,
                "ended_after_window": True,
                "synchronized": True,
                "dropped_events": 0,
            },
            "initial_live": initial_live,
            "event_begin": self.cursor,
            "event_end": end,
            "events": window_events,
            "metrics": metrics,
        }
        next_prefix = copy.deepcopy(events[:end])
        self.cursor = end
        self.prefix = next_prefix
        self.live = live
        return observation
