"""Bounded, data-only reuse of deterministic lowering before binary lookup.

Callers own the semantic recipe and its verified dependency closure. Entries
contain UTF-8 sources, never executable Python or serialized IR objects. Binary
consumers must still validate their usual source/header/toolchain identities.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from .provenance import atomic_json, canonical_hash, file_hash
from .source_cache import cache_source

if TYPE_CHECKING:
    from collections.abc import Callable


def cached_sources(
    cache: Path | None,
    recipe: dict[str, Any],
    produce: Callable[[], tuple[str, ...]],
    *,
    max_unit_bytes: int = 64 << 20,
    max_total_bytes: int = 64 << 20,
    expected_units: int | None = None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Reuse checked source bytes, or atomically publish one complete recipe.

    Missing entries regenerate; corrupt/partial entries fail closed. Concurrent
    producers may duplicate lowering, but a losing publisher validates both the
    winner's recipe and exact bytes. ``None`` disables persistent source reuse
    while preserving the same byte budgets and the downstream binary cache.
    Reported host spans are exclusive and never include compiler subprocesses.
    """
    started = perf_counter()
    identity = {"schema": "generativeqc.semantic-sources.v1", "recipe": recipe}
    key = canonical_hash(identity)
    destination = None if cache is None else Path(cache) / "semantic-sources" / key

    def validate(sources: tuple[str, ...]) -> tuple[int, ...]:
        if not 0 < len(sources) <= 4096 or any(
            not isinstance(source, str) or not source for source in sources
        ):
            raise ValueError("semantic source cache requires nonempty source units")
        if expected_units is not None and len(sources) != expected_units:
            raise ValueError("semantic source cache unit inventory mismatch")
        sizes = tuple(len(source.encode("utf-8")) for source in sources)
        if any(size > max_unit_bytes for size in sizes) or sum(sizes) > max_total_bytes:
            raise ValueError("semantic source cache byte budget exceeded")
        return sizes

    def snapshot(path: Path, limit: int) -> bytes:
        if limit < 0:
            raise ValueError("source byte budget exceeded")
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("source byte budget exceeded")
        return data

    def load(folder: Path) -> tuple[str, ...]:
        try:
            manifest = folder / "sources.json"
            metadata = json.loads(snapshot(manifest, 1 << 20).decode("utf-8"))
            if metadata["key"] != key or canonical_hash(metadata["identity"]) != key:
                raise ValueError("recipe mismatch")
            records = metadata["sources"]
            if not isinstance(records, list) or not 0 < len(records) <= 4096:
                raise ValueError("invalid unit inventory")
            if expected_units is not None and len(records) != expected_units:
                raise ValueError("unit inventory mismatch")
            sources, total = [], 0
            for index, record in enumerate(records):
                path = folder / f"{index}.cu"
                data = snapshot(path, min(max_unit_bytes, max_total_bytes - total))
                size = len(data)
                total += size
                if (
                    record["bytes"] != size
                    or record["sha256"] != hashlib.sha256(data).hexdigest()
                ):
                    raise ValueError("source hash mismatch")
                sources.append(data.decode("utf-8"))
            result = tuple(sources)
            validate(result)
            return result
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError("semantic source cache integrity failure") from error

    hit = destination is not None and destination.exists()
    if hit:
        sources = load(destination)
        lookup_seconds = perf_counter() - started
        generation_seconds = publication_seconds = 0.0
    else:
        lookup_seconds = perf_counter() - started
        generation_started = perf_counter()
        sources = tuple(produce())
        sizes = validate(sources)
        generation_seconds = perf_counter() - generation_started
        publication_started = perf_counter()
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".sources-", dir=destination.parent
            ) as temporary:
                folder = Path(temporary)
                records = []
                for index, (source, size) in enumerate(zip(sources, sizes)):
                    path = folder / f"{index}.cu"
                    cache_source(path, source)
                    records.append({"bytes": size, "sha256": file_hash(path)})
                atomic_json(
                    folder / "sources.json",
                    {
                        "identity": identity,
                        "key": key,
                        "sources": records,
                    },
                )
                try:
                    os.rename(folder, destination)
                except OSError:
                    if not destination.is_dir():
                        raise
                    if load(destination) != sources:
                        raise ValueError(
                            "semantic source cache nondeterministic producer"
                        )
        publication_seconds = perf_counter() - publication_started
    return sources, {
        "key": key,
        "hit": hit,
        "enabled": cache is not None,
        "units": len(sources),
        "source_bytes": sum(len(source.encode("utf-8")) for source in sources),
        "lookup_seconds": lookup_seconds,
        "generation_seconds": generation_seconds,
        "publication_seconds": publication_seconds,
    }
