"""Independent fixed-width trace decoding. Successful parsing is not capture PASS."""

from __future__ import annotations

import hashlib
import os
import stat
import struct
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

PHASE = 0x434350480001
CC_PLAN = 0x434350410001
T_PLAN = 0x434354410001
CONTRACT = 0x434343500002
FORMATS = {
    0x434354494D453031: (20, 14, 1 << 21),
    0x4343534F55524331: (19, 24, 1 << 18),
    0x4343475241504831: (12, 8, 1 << 18),
    0x43434C4547414331: (14, 8, 1 << 18),
}


class InvalidTrace(ValueError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise InvalidTrace(message)


def regular_stream(path: Path) -> BinaryIO:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        )
    except OSError as error:
        raise InvalidTrace("trace open refused") from error
    try:
        require(
            stat.S_ISREG(os.fstat(descriptor).st_mode),
            "trace is not an opened regular file",
        )
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


@dataclass
class BinaryTrace:
    path: Path
    magic: int
    record_words: int
    metadata: tuple[int, ...]
    count: int
    offset: int
    identity: tuple[int, int, int, int]
    sha256: str

    def records(self) -> Iterator[tuple[int, ...]]:
        codec = struct.Struct("<" + "Q" * self.record_words)
        with regular_stream(self.path) as stream:
            before = os.fstat(stream.fileno())
            require(
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                == self.identity,
                "trace changed before record read",
            )
            stream.seek(self.offset)
            for _ in range(self.count):
                raw = stream.read(codec.size)
                require(len(raw) == codec.size, "short trace record")
                yield codec.unpack(raw)
            require(not stream.read(1), "unexpected trace tail")
            after = os.fstat(stream.fileno())
            require(
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                == self.identity,
                "trace changed during record read",
            )


def open_trace(path: Path) -> BinaryTrace:
    with regular_stream(path) as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(40)
        require(len(raw) == 40, "short binary header")
        magic, version, words, metadata_words, count = struct.unpack("<5Q", raw)
        require(magic in FORMATS and version == 1, "unsupported trace schema")
        expected_words, expected_meta, maximum = FORMATS[magic]
        require(
            words == expected_words
            and metadata_words == expected_meta
            and count <= maximum,
            "trace ABI/size/capacity mismatch",
        )
        raw_metadata = stream.read(metadata_words * 8)
        require(len(raw_metadata) == metadata_words * 8, "short metadata")
        metadata = struct.unpack("<" + "Q" * metadata_words, raw_metadata)
        require(
            count == metadata[0] and count <= metadata[1] <= maximum,
            "count differs from immutable metadata",
        )
        offset = 40 + metadata_words * 8
        require(before.st_size == offset + count * words * 8, "binary size mismatch")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        require(
            identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "trace changed during hash",
        )
    return BinaryTrace(path, magic, words, metadata, count, offset, identity, digest)


def source_inventory(trace: BinaryTrace) -> dict[str, Any]:
    require(trace.magic == 0x4343534F55524331, "not a source journal")
    meta = trace.metadata
    require(
        all(meta[i] == 0 for i in (2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 20, 21)),
        "source loss/error/native failure",
    )
    require(
        meta[10] > 0 and meta[11] == CONTRACT and meta[19] == 1,
        "source lifecycle/contract mismatch",
    )
    require(
        all(meta[i] > 0 for i in (7, 8, 9, 22)) and meta[8] == meta[9],
        "unqualified PID/thread domain",
    )
    stack = []
    phases = {}
    closed = set()
    plans = {}
    begins = []
    last = meta[17]
    for entry in trace.records():
        timestamp, size = entry[:2]
        require(
            meta[17] <= timestamp <= meta[18] and timestamp >= last,
            "source clock/order mismatch",
        )
        last = timestamp
        require(
            size in (10, 12, 14, 17) and all(value == 0 for value in entry[2 + size :]),
            "source record width/padding",
        )
        row = entry[2 : 2 + size]
        if row[0] in (1, 2):
            require((row[0], size) in ((1, 10), (2, 14)), "legacy source shape")
            continue
        if row[0] == PHASE:
            require(
                size == 12 and row[2] > 0 and 1 <= row[4] <= 32 and row[10] == meta[10],
                "phase shape/identity/permit",
            )
            if row[1] == 1:
                require(
                    row[2] not in phases and row[2] not in plans,
                    "duplicate phase identity",
                )
                require(row[3] == (stack[-1] if stack else 0), "phase parent mismatch")
                require(row[7] == 0 or row[7] in plans, "phase references unknown plan")
                require(
                    row[8] == 0 or row[8] in closed,
                    "dependency is not an already closed producer declaration",
                )
                phases[row[2]] = {"words": row, "start": timestamp, "end": None}
                stack.append(row[2])
                begins.append(row[2])
            else:
                require(
                    row[1] == 2 and stack and stack[-1] == row[2],
                    "unmatched/reordered phase end",
                )
                phase = phases[row[2]]
                require(
                    phase["words"][2:11] == row[2:11], "end payload/parent mismatch"
                )
                require(
                    row[11] in (0, 2) and (row[11] != 2 or row[4] == 29),
                    "scientific exception/unknown phase status",
                )
                phase.update(end=timestamp, status=row[11])
                closed.add(row[2])
                stack.pop()
        elif row[0] in (CC_PLAN, T_PLAN):
            require((row[0], size) in ((CC_PLAN, 14), (T_PLAN, 17)), "admission ABI")
            require(
                row[1] == 1
                and row[2] > 0
                and row[2] not in plans
                and row[2] not in phases,
                "admission identity",
            )
            require(
                stack and row[3] == stack[-1], "admission outside actual native parent"
            )
            permit = row[12] if row[0] == CC_PLAN else row[15]
            require(permit == meta[10] and row[-1] == 0, "admission permit/status")
            plans[row[2]] = row
        else:
            raise InvalidTrace("unknown source record layout")
    require(not stack and len(closed) == len(phases), "source open tail")
    roots = [phase for phase in phases.values() if phase["words"][3] == 0]
    require(
        len(roots) == 1 and roots[0]["words"][4] == 1,
        "missing/extra scientific source root",
    )
    return {
        "phases": phases,
        "plans": plans,
        "begin_order": begins,
        "site_counts": Counter(p["words"][4] for p in phases.values()),
        "root": roots[0],
        "pid": meta[7],
        "tid": meta[8],
        "pid_starttime": meta[22],
        "permit": meta[10],
    }


def activity_inventory(trace: BinaryTrace) -> dict[str, Any]:
    require(trace.magic == 0x434354494D453031, "not an activity journal")
    meta = trace.metadata
    require(
        all(meta[i] == 0 for i in (2, 3, 4, 5, 6, 9, 12, 13)),
        "activity loss/error/pending/start failure",
    )
    require(
        meta[7] == 28 and meta[8] == 1 and meta[10] > 0,
        "CUPTI version/lifecycle/kernel coverage",
    )
    counts = Counter()
    correlations = {}
    gpu = []
    api = []
    waits = []
    for row in trace.records():
        kind = row[0]
        require(
            kind in (1, 2, 4, 5, 10, 22, 38, 39, 48),
            "unknown/serialized kernel activity",
        )
        require(row[18:] == (0, 0), "nonzero reserved activity fields")
        counts[kind] += 1
        if kind == 39:
            key = (row[1], row[5])
            require(row[5] > 0 and row[7] > 0, "invalid external correlation")
            if key in correlations:
                require(correlations[key] == row[7], "ambiguous external correlation")
            correlations[key] = row[7]
            continue
        require(
            row[3] > 0 and row[4] >= row[3], "unknown/incomplete activity timestamp"
        )
        if kind in (1, 2, 10, 22):
            gpu.append(row)
        elif kind == 38:
            waits.append(row)
        else:
            api.append(row)
    require(
        counts[10] == meta[10]
        and counts[2] == meta[11]
        and sum(counts.values()) == trace.count,
        "retained family count mismatch",
    )
    return {
        "counts": counts,
        "gpu": gpu,
        "api": api,
        "waits": waits,
        "external": correlations,
        "queued_completed": "UNAVAILABLE where raw timestamp is zero; not a measured zero latency",
    }
