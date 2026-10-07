"""Device-free integrity, boundedness and publication races for source reuse."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common.semantic_source_cache import cached_sources

if TYPE_CHECKING:
    from pathlib import Path


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
