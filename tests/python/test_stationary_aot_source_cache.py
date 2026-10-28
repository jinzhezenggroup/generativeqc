"""AOT generators reuse checked scientific sources across build directories."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc_compiler.integral import first_derivative_schedule as primitives
from generativeqc_compiler.method import stationary_cuda as lowering

from tools import generate_native_stationary_cuda as native
from tools import generate_stationary_force_aot as legacy

if TYPE_CHECKING:
    from pathlib import Path


def test_composite_source_cache_preserves_external_derivative_selection(
    tmp_path: Path,
) -> None:
    """Cached range/VV10 wrappers must preserve the requested source layout."""
    emit = lowering.emit_stationary_profile_aot_wrapper_cuda
    semilocal = emit("wb97mv_rks", include_external=False, source_cache=tmp_path)
    complete = emit("wb97mv_rks", include_external=True, source_cache=tmp_path)
    assert complete != semilocal
    assert complete == emit("wb97mv_rks", include_external=True)
    assert semilocal == emit("wb97mv_rks", include_external=False)
    assert complete == emit("wb97mv_rks", include_external=True, source_cache=tmp_path)


@pytest.mark.parametrize("kind", ["sp", "spd", "native"])
def test_aot_wrapper_source_hit_preserves_bytes_and_skips_lowering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The output directory is build ownership, never scientific source identity."""
    generator = native if kind == "native" else legacy
    selection = (
        []
        if kind == "native"
        else ["--wrapper-only"]
        if kind == "sp"
        else ["--component-domain", "spd"]
    )
    cache = tmp_path / "shared-sources"

    def generate(output: Path, *, cached: bool = True) -> str:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "generate",
                "--profile",
                "pbe0_rks",
                "--output",
                str(output),
                *selection,
                *(["--source-cache", str(cache)] if cached else []),
            ],
        )
        generator.main()
        return output.read_text()

    uncached = generate(tmp_path / "uncached" / "wrapper.cu", cached=False)
    first = generate(tmp_path / "sm80-build" / "wrapper.cu")
    assert first == uncached

    def forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("AOT source cache hit repeated geometry lowering")

    monkeypatch.setattr(lowering, "emit_geometry_cuda", forbidden)
    second = tmp_path / "sm120-build" / "wrapper.cu"
    assert generate(second) == first
    timestamp = second.stat().st_mtime_ns
    assert generate(second) == first
    assert second.stat().st_mtime_ns == timestamp

    # The CLI must propagate integrity failures, without replacing a valid output.
    entry = next((cache / "semantic-sources").glob("*/0.cu"))
    entry.write_text("// corrupted source\n")
    with pytest.raises(ValueError, match="integrity"):
        generate(second)
    assert second.read_text() == first


def test_all_aot_primitive_shards_replay_as_one_checked_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse the integral-owned ordered inventory without caching mutable IR."""
    sources = tuple(
        ((), f"// primitive shard {index}\n")
        for index in range(legacy.QUALIFIED_SPD_AOT_SHARDS)
    )
    monkeypatch.setattr(primitives, "derivative_cuda_sources", lambda domain: sources)
    cache = tmp_path / "sources"

    def generate(build: str) -> dict[str, str]:
        output = tmp_path / build
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "generate",
                "--component-domain",
                "spd",
                "--all-shards",
                "--source-cache",
                str(cache),
                "--output",
                str(output),
            ],
        )
        legacy.main()
        return {p.name: p.read_text() for p in output.glob("*.cu")}

    first = generate("sm80-build")
    assert len(first) == len(sources)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("AOT source hit repeated primitive lowering")

    monkeypatch.setattr(primitives, "derivative_cuda_sources", forbidden)
    assert generate("sm120-build") == first


def test_shared_automatic_native_package_uses_the_same_source_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optional AOT cache preserves #2835's staged producer/consumer split."""
    cache = tmp_path / "sources"
    profile = lowering.AUTOMATIC_STATIONARY_AOT_PROFILE_NAMES[0]

    def generate(build: str) -> dict[str, str]:
        output = tmp_path / build
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "generate",
                "--all-profiles",
                "--profiles",
                profile,
                "--source-cache",
                str(cache),
                "--output-dir",
                str(output),
            ],
        )
        native.main()
        return {p.name: p.read_text() for p in output.iterdir()}

    first = generate("sm80-build")
    assert {name.rpartition(".")[2] for name in first} == {"cu", "cpp"}

    def forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("shared native hit repeated geometry lowering")

    monkeypatch.setattr(lowering, "emit_geometry_cuda", forbidden)
    assert generate("sm120-build") == first
