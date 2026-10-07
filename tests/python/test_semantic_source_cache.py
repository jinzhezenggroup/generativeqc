"""Device-free integrity, boundedness and publication races for source reuse."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc_compiler.common.semantic_source_cache import cached_sources

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self


def forbidden() -> tuple[str, ...]:
    raise AssertionError("a semantic source hit must not lower IR")


def test_hit_recipe_invalidation_and_no_cache(tmp_path: Path) -> None:
    sources = ("// first\n", "// second\n")
    first, miss = cached_sources(tmp_path, {"recipe": 1}, lambda: sources)
    replay, hit = cached_sources(tmp_path, {"recipe": 1}, forbidden)
    assert first == replay == sources
    assert not miss["hit"] and hit["hit"]
    assert hit["units"] == 2 and hit["source_bytes"] == sum(map(len, sources))
    assert hit["generation_seconds"] == hit["publication_seconds"] == 0.0
    _, changed = cached_sources(tmp_path, {"recipe": 2}, lambda: sources)
    assert changed["key"] != hit["key"] and not changed["hit"]
    _, disabled = cached_sources(None, {"recipe": 1}, lambda: sources)
    assert not disabled["hit"] and not disabled["enabled"]
    assert disabled["key"] == hit["key"]


@pytest.mark.parametrize(
    "damage", ("source", "hash", "recipe", "partial", "json", "inventory")
)
def test_corrupt_entries_fail_before_generation(tmp_path: Path, damage: str) -> None:
    _, work = cached_sources(tmp_path, {}, lambda: ("// intact\n",))
    folder = tmp_path / "semantic-sources" / work["key"]
    manifest = folder / "sources.json"
    if damage == "source":
        (folder / "0.cu").write_text("// corrupt\n")
    elif damage == "partial":
        manifest.unlink()
    elif damage == "json":
        manifest.write_text("{")
    else:
        metadata = json.loads(manifest.read_text())
        if damage == "hash":
            metadata["sources"][0]["sha256"] = "0" * 64
        elif damage == "recipe":
            metadata["identity"]["recipe"] = {"changed": True}
        else:
            metadata["sources"] = []
        manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="integrity"):
        cached_sources(tmp_path, {}, forbidden)


@pytest.mark.parametrize("sources", ((), ("",), ("a" * 5,), ("aaa", "aaa")))
def test_generation_byte_budgets(tmp_path: Path, sources: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="nonempty|budget"):
        cached_sources(
            tmp_path, {}, lambda: sources, max_unit_bytes=4, max_total_bytes=5
        )
    assert not (tmp_path / "semantic-sources").exists()


def test_hit_rechecks_budget(tmp_path: Path) -> None:
    cached_sources(tmp_path, {}, lambda: ("12345",))
    with pytest.raises(ValueError, match="integrity"):
        cached_sources(tmp_path, {}, forbidden, max_unit_bytes=4)


def test_truncated_inventory_rejects_even_intact_remaining_sources(
    tmp_path: Path,
) -> None:
    _, work = cached_sources(
        tmp_path, {}, lambda: ("first", "second"), expected_units=2
    )
    manifest = tmp_path / "semantic-sources" / work["key"] / "sources.json"
    metadata = json.loads(manifest.read_text())
    metadata["sources"].pop()
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="integrity"):
        cached_sources(tmp_path, {}, forbidden, expected_units=2)


def test_wrong_producer_inventory_is_not_published(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inventory"):
        cached_sources(tmp_path, {}, lambda: ("only one",), expected_units=2)
    assert not (tmp_path / "semantic-sources").exists()


def test_concurrent_producers_publish_only_complete_entries(tmp_path: Path) -> None:
    barrier = Barrier(4)

    def run() -> tuple[tuple[str, ...], dict]:
        def produce() -> tuple[str, ...]:
            barrier.wait(timeout=10)
            return ("// first\n", "// second\n")

        return cached_sources(tmp_path, {"concurrent": True}, produce)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: run(), range(4)))
    assert all(value == results[0][0] for value, _ in results)
    assert all(not work["hit"] for _, work in results)
    replay, work = cached_sources(tmp_path, {"concurrent": True}, forbidden)
    assert replay == results[0][0] and work["hit"]
    assert not list((tmp_path / "semantic-sources").glob(".sources-*"))


def test_concurrent_nondeterministic_producers_fail_closed(tmp_path: Path) -> None:
    barrier = Barrier(2)

    def run(index: int) -> str:
        def produce() -> tuple[str, ...]:
            barrier.wait(timeout=10)
            return (f"// different {index}\n",)

        try:
            cached_sources(tmp_path, {}, produce)
        except ValueError as error:
            return str(error)
        return "published"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(2)))
    assert sorted(results) == [
        "published",
        "semantic source cache nondeterministic producer",
    ]


def test_returned_source_is_the_verified_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacement after hashing cannot substitute unchecked scientific source."""
    from generativeqc_compiler.common import semantic_source_cache as cache_module

    original_source = "const int x=1;"
    replacement = "const int x=2;"
    _, work = cached_sources(tmp_path, {}, lambda: (original_source,))
    unit = tmp_path / "semantic-sources" / work["key"] / "0.cu"
    original_hash = cache_module.hashlib.sha256

    def replace_after_hash(data: bytes = b"", **kwargs: Any) -> Any:
        digest = original_hash(data, **kwargs)
        if data == original_source.encode():
            unit.write_text(replacement)
        return digest

    original_file_hash = cache_module.file_hash

    def replace_after_file_hash(path: Path) -> str:
        digest = original_file_hash(path)
        if path == unit:
            unit.write_text(replacement)
        return digest

    monkeypatch.setattr(cache_module.hashlib, "sha256", replace_after_hash)
    monkeypatch.setattr(cache_module, "file_hash", replace_after_file_hash)
    replay, hit = cached_sources(tmp_path, {}, forbidden)
    assert hit["hit"] and replay == (original_source,)
    assert unit.read_text() == replacement
    with pytest.raises(ValueError, match="integrity"):
        cached_sources(tmp_path, {}, forbidden)


@pytest.mark.parametrize("damaged", ("manifest", "unit", "aggregate"))
def test_load_uses_bounded_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damaged: str
) -> None:
    """No stat/read race can turn admitted metadata into an unbounded read."""
    from pathlib import Path

    sources = ("first", "second")
    _, work = cached_sources(tmp_path, {}, lambda: sources)
    folder = tmp_path / "semantic-sources" / work["key"]
    target = folder / ("sources.json" if damaged == "manifest" else "1.cu")
    limit = (1 << 20) if damaged == "manifest" else (6 if damaged == "unit" else 5)
    target.write_bytes(b"x" * (limit + 10))
    original_open = Path.open
    reads = []

    class BoundedReader:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> bool | None:
            return self.stream.__exit__(*args)

        def read(self, size: int = -1) -> bytes:
            reads.append(size)
            assert size == limit + 1
            return self.stream.read(size)

    def checked_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        stream = original_open(path, *args, **kwargs)
        return BoundedReader(stream) if path == target else stream

    monkeypatch.setattr(Path, "open", checked_open)
    with pytest.raises(ValueError, match="integrity"):
        cached_sources(
            tmp_path,
            {},
            forbidden,
            max_unit_bytes=6,
            max_total_bytes=10 if damaged == "aggregate" else 11,
        )
    assert reads == [limit + 1]
