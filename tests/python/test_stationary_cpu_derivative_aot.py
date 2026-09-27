"""Packaged CPU stationary derivative inventory and runtime selection."""

from __future__ import annotations

import ctypes as ct
import os
import typing
from pathlib import Path

import pytest
from vibeqc import _stationary_cpu_components as components
from vibeqc_compiler.integral import first_derivative_schedule as schedule


def test_cpu_aot_inventory_is_complete_and_link_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule.derivative_cpu_aot_sources.cache_clear()
    calls = []

    def emit(requests: typing.Any, *, symbol: str) -> str:
        calls.append((tuple(requests), symbol))
        return f"// {symbol} {len(requests)}\n"

    monkeypatch.setattr(schedule, "emit_first_derivative_cpu", emit)
    try:
        sources = schedule.derivative_cpu_aot_sources()
    finally:
        schedule.derivative_cpu_aot_sources.cache_clear()

    assert len(sources) == schedule.CPU_AOT_SHARDS == 46
    assert sum(len(requests) for requests, _ in sources) == 362
    assert [symbol for _, symbol in calls] == [
        schedule.cpu_aot_symbol(shard)
        for shard in range(schedule.CPU_AOT_SHARDS)
    ]
    assert all(
        len(requests) <= schedule.REQUESTS_PER_UNIT for requests, _ in sources
    )


def test_packaged_dispatch_loader_is_all_or_nothing() -> None:
    signature = ct.CFUNCTYPE(
        ct.c_int,
        ct.c_uint,
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.POINTER(ct.c_double),
    )

    @signature
    def dispatch(
        kind: int,
        records: typing.Any,
        count: int,
        output: typing.Any,
    ) -> int:
        return 0

    library = type("Library", (), {})()
    for shard in range(schedule.CPU_AOT_SHARDS):
        setattr(library, schedule.cpu_aot_symbol(shard), dispatch)
    loaded = components._packaged_aot_dispatchers(library)
    assert len(loaded) == schedule.CPU_AOT_SHARDS

    delattr(library, schedule.cpu_aot_symbol(7))
    assert components._packaged_aot_dispatchers(library) == ()


def test_component_executor_uses_packaged_aot_without_runtime_compilation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from vibeqc import _stationary_cpu_components as component_module
    from vibeqc import _stationary_cpu_streaming as streaming_module

    dispatch_signature = ct.CFUNCTYPE(
        ct.c_int,
        ct.c_uint,
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.POINTER(ct.c_double),
    )

    @dispatch_signature
    def dispatch(
        kind: int,
        records: typing.Any,
        count: int,
        output: typing.Any,
    ) -> int:
        return 0

    @ct.CFUNCTYPE(ct.c_int)
    def contract() -> int:
        return 0

    library = type("Library", (), {})()
    library.vibeqc_component_contract_cpu = contract
    for shard in range(schedule.CPU_AOT_SHARDS):
        setattr(library, schedule.cpu_aot_symbol(shard), dispatch)

    labels = schedule.CPU_AOT_COMPONENTS
    aos = []
    primitive_rows = []
    for index, label in enumerate(labels):
        powers = [label.count(axis) for axis in "xyz"]
        row = [0.0] * 16
        row[:4] = [0, index, 1, 1]
        row[4:7] = powers
        row[7] = 1.0
        aos.extend(row)
        primitive_rows.extend((0.5 + 0.01 * index, 1.0))
    basis = SimpleNamespace(
        natom=1,
        nprimitive=len(labels),
        shells=(SimpleNamespace(angular_momentum=2),),
        packed=__import__("numpy").asarray(
            [0.0, 0.0, 0.0, *primitive_rows, *aos], dtype=float
        ),
    )

    monkeypatch.setattr(
        component_module,
        "derivative_sources",
        lambda domain: pytest.fail("runtime derivative source generation"),
    )
    monkeypatch.setattr(
        streaming_module,
        "compile_runtime",
        lambda *args, **kwargs: pytest.fail("runtime component compilation"),
    )

    executor = streaming_module.CompiledComponentExecutor(
        basis,
        tmp_path,
        1,
        None,  # type: ignore[arg-type]
        aot_library=library,
    )
    assert executor.compilation_work["primitive_packaged_aot"] == 1
    assert executor.compilation_work["primitive_runtime_compilations"] == 0
    assert executor.compilation_work["component_contract_runtime_compilations"] == 0
    assert len(executor.dispatchers) == schedule.CPU_AOT_SHARDS


def test_native_library_exports_stationary_cpu_aot() -> None:
    path = os.environ.get("VIBEQC_LIBRARY")
    if not path:
        pytest.skip("native VibeQC library not supplied")
    library = ct.CDLL(str(Path(path)))
    assert library.vibeqc_component_contract_cpu
    for shard in range(schedule.CPU_AOT_SHARDS):
        assert getattr(library, schedule.cpu_aot_symbol(shard))


def test_cmake_registers_the_fixed_cpu_aot_inventory() -> None:
    root = Path(__file__).resolve().parents[2]
    top = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    generated = (root / "cmake/VibeQCGeneratedSources.cmake").read_text(
        encoding="utf-8"
    )
    assert "VIBEQC_ENABLE_STATIONARY_CPU_FORCE_AOT" in top
    assert "foreach(_vibeqc_stationary_cpu_shard RANGE 0 45)" in generated
    assert "generate_stationary_cpu_derivative_aot.py" in generated
    assert "ADD_TO_TARGET" in generated
    assert "-ffp-contract=off" in generated
    identity = (root / "cmake/VibeQCSourceIdentity.json").read_text(
        encoding="utf-8"
    )
    assert "tools/generate_stationary_cpu_derivative_aot.py" in identity
