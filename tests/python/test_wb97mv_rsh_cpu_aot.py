"""WB97M-V CPU range-exchange derivatives load packaged weighted-ERI AOT."""

from __future__ import annotations

import ctypes as ct
import json
import os
import shutil
import typing
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import _stationary_rsh_cpu as runtime
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.integral.derivative_aot_registry import (
    radial_inventory_from_payload,
)
from generativeqc_compiler.integral.ir import four_center_eri_operator
from generativeqc_compiler.integral.range_separation import CoulombKernel
from generativeqc_compiler.integral.rsh_cpu_aot import (
    AOT_ANGULAR_DOMAIN,
    component_groups,
    entry_prefix,
    inventory_size,
    program_source,
)
from generativeqc_compiler.integral.weighted_eri import (
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)
from generativeqc_compiler.integral.weighted_eri_native import emit_weighted_eri_runtime
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.spec import RangeSeparatedExchangePrimitive


def _wb97mv_ranges() -> tuple[RangeSeparatedExchangePrimitive, ...]:
    method = resolve_method("WB97M-V", spin="unpolarized")
    return tuple(
        primitive
        for primitive in method.primitives
        if type(primitive) is RangeSeparatedExchangePrimitive
    )


def _radial(primitive: RangeSeparatedExchangePrimitive) -> CoulombKernel:
    return CoulombKernel(
        {
            "short-range": "short_range",
            "long-range": "long_range",
        }[primitive.operator],
        float(primitive.omega),
    )


def _fake_s_basis() -> SimpleNamespace:
    centers = np.array(
        [
            [0.13, -0.31, 0.24],
            [-0.43, 0.27, 0.51],
            [0.68, -0.14, -0.22],
            [-0.21, 0.48, -0.63],
        ]
    )
    primitives = np.array([[0.57, 0.83], [0.71, -0.19], [0.89, 0.67], [1.13, 0.42]])
    aos = np.zeros((4, 16))
    for index in range(4):
        aos[index, :8] = (index, index, 1, 1, 0, 0, 0, 1)
    return SimpleNamespace(
        natom=4,
        nao=4,
        nprimitive=4,
        shells=tuple(SimpleNamespace(angular_momentum=0) for _ in range(4)),
        packed=np.concatenate((centers.ravel(), primitives.ravel(), aos.ravel())),
    )


def test_wb97mv_rsh_aot_inventory_is_bounded_and_complete() -> None:
    assert inventory_size() == 34
    assert len(_wb97mv_ranges()) == 2
    count = 0
    prefixes = set()
    for primitive in _wb97mv_ranges():
        radial = _radial(primitive)
        for angular_value in product(AOT_ANGULAR_DOMAIN, repeat=4):
            angular = tuple(angular_value)
            groups = component_groups(angular)
            assert len(groups) == (2 if angular == (1, 1, 1, 1) else 1)
            assert all(1 <= len(group) <= 64 for group in groups)
            for group_index, group in enumerate(groups):
                source, selected, prefix = program_source(radial, angular, group_index)
                assert selected == group
                assert f"{prefix}_identity_v2" in source
                assert f"{prefix}_create_v2" in source
                assert "GENERATIVEQC_API" in source
                assert prefix not in prefixes
                prefixes.add(prefix)
                count += 1
    assert count == 34


def test_weighted_runtime_default_symbols_remain_compatible() -> None:
    radial = CoulombKernel("short_range", 0.3)
    integral = build_weighted_eri_ir(
        (0, 0, 0, 0), operator=four_center_eri_operator(radial)
    )
    kernel = build_weighted_eri_kernel(integral, (0,))
    source = emit_weighted_eri_runtime(kernel, backend="cpu")
    assert "generativeqc_weighted_identity_v2" in source
    assert "generativeqc_weighted_create_v2" in source
    assert "GENERATIVEQC_API" not in source


def test_range_exchange_prefers_packaged_cpu_aot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compiler_path = shutil.which("c++")
    if compiler_path is None:
        pytest.skip("native C++ compiler unavailable")
    compiler = CppCompilerAdapter(Path(compiler_path))
    primitive = next(
        item for item in _wb97mv_ranges() if item.operator == "short-range"
    )
    radial = _radial(primitive)
    prefix = entry_prefix(radial, (0, 0, 0, 0), 0)
    library = type("Library", (), {})()
    setattr(library, f"{prefix}_identity_v2", object())

    packaged_calls = []

    def packaged(
        integral: typing.Any,
        candidate_library: typing.Any,
        *,
        component_indices: typing.Any,
        entry_prefix: str,
    ) -> typing.Any:
        packaged_calls.append((tuple(component_indices), entry_prefix))
        return SimpleNamespace(backend="cpu")

    def forbidden_compile(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("packaged WB97M-V RSH path reached runtime compilation")

    class FakePlan:
        def __init__(self, artifact: typing.Any, **kwargs: typing.Any) -> None:
            self.artifact = artifact

        def raw(
            self,
            primitives: typing.Any,
            centers: typing.Any,
            component_indices: typing.Any,
        ) -> typing.Any:
            return SimpleNamespace(
                values=np.zeros((1, 13)),
                diagnostics={"records": 1},
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(runtime, "packaged_weighted_eri", packaged)
    monkeypatch.setattr(runtime, "compile_weighted_eri", forbidden_compile)
    monkeypatch.setattr(runtime, "PreparedWeightedEri", FakePlan)

    executor = runtime.RangeExchangeExecutor(
        _fake_s_basis(),
        tmp_path,
        8,
        compiler,
        aot_library=library,
    )
    try:
        owners, values = executor.integral(primitive, (0, 1, 2, 3), 1.0)
    finally:
        executor.close()
    assert owners == [0, 1, 2, 3]
    np.testing.assert_array_equal(values, 0)
    assert packaged_calls == [((0,), prefix)]
    assert executor.packaged_aot_plans == 1
    assert executor.runtime_compilations == 0


def test_grouped_aot_reuses_one_prepared_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compiler_path = shutil.which("c++")
    if compiler_path is None:
        pytest.skip("native C++ compiler unavailable")
    compiler = CppCompilerAdapter(Path(compiler_path))
    primitive = next(
        item for item in _wb97mv_ranges() if item.operator == "short-range"
    )
    radial = _radial(primitive)
    angular = (1, 1, 1, 1)
    prefix = entry_prefix(radial, angular, 0)
    library = type("Library", (), {})()
    setattr(library, f"{prefix}_identity_v2", object())

    centers = np.asarray(
        [
            [0.13, -0.31, 0.24],
            [-0.43, 0.27, 0.51],
            [0.68, -0.14, -0.22],
            [-0.21, 0.48, -0.63],
        ]
    )
    primitives = []
    aos = []
    for atom in range(4):
        for axis in range(2):
            primitive_index = len(primitives)
            primitives.append((0.57 + 0.07 * primitive_index, 1.0))
            row = [0.0] * 16
            row[:4] = [atom, primitive_index, 1, 1]
            row[4 + axis] = 1
            row[7] = 1.0
            aos.append(row)
    basis = SimpleNamespace(
        natom=4,
        nao=len(aos),
        nprimitive=len(primitives),
        shells=tuple(SimpleNamespace(angular_momentum=1) for _ in range(4)),
        packed=np.concatenate(
            (
                centers.ravel(),
                np.asarray(primitives).ravel(),
                np.asarray(aos).ravel(),
            )
        ),
    )

    packaged_calls = []
    plans = []

    def packaged(
        integral: typing.Any,
        candidate_library: typing.Any,
        *,
        component_indices: typing.Any,
        entry_prefix: str,
    ) -> typing.Any:
        packaged_calls.append((tuple(component_indices), entry_prefix))
        return SimpleNamespace(backend="cpu")

    def forbidden_compile(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("grouped AOT reuse reached runtime compilation")

    class FakePlan:
        def __init__(self, artifact: typing.Any, **kwargs: typing.Any) -> None:
            plans.append(self)

        def raw(
            self,
            primitives: typing.Any,
            centers: typing.Any,
            component_indices: typing.Any,
        ) -> typing.Any:
            return SimpleNamespace(
                values=np.zeros((1, 13)),
                diagnostics={"records": 1},
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(runtime, "packaged_weighted_eri", packaged)
    monkeypatch.setattr(runtime, "compile_weighted_eri", forbidden_compile)
    monkeypatch.setattr(runtime, "PreparedWeightedEri", FakePlan)

    executor = runtime.RangeExchangeExecutor(
        basis,
        tmp_path,
        8,
        compiler,
        aot_library=library,
    )
    try:
        executor.integral(primitive, (0, 2, 4, 6), 1.0)
        executor.integral(primitive, (1, 3, 5, 7), 1.0)
    finally:
        executor.close()
    assert len(plans) == 1
    assert len(packaged_calls) == 1
    assert executor.packaged_aot_plans == 1
    assert executor.runtime_compilations == 0


def test_native_library_exports_wb97mv_rsh_aot() -> None:
    path = os.environ.get("GENERATIVEQC_LIBRARY")
    if not path:
        pytest.skip("native GenerativeQC library not supplied")
    library = ct.CDLL(str(Path(path)))
    count = 0
    for primitive in _wb97mv_ranges():
        radial = _radial(primitive)
        for angular_value in product(AOT_ANGULAR_DOMAIN, repeat=4):
            angular = tuple(angular_value)
            for group_index in range(len(component_groups(angular))):
                prefix = entry_prefix(radial, angular, group_index)
                assert getattr(library, f"{prefix}_identity_v2")
                assert getattr(library, f"{prefix}_create_v2")
                assert getattr(library, f"{prefix}_run_v2")
                count += 1
    assert count == 34


def test_range_aot_manifest_covers_current_wb97mv_operator_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (root / "manifests/derivative_aot_radials.json").read_text(encoding="utf-8")
    )
    packaged = radial_inventory_from_payload(payload, backend="cpu")
    expected = tuple(_radial(primitive) for primitive in _wb97mv_ranges())
    assert packaged == expected


def test_cmake_packages_method_neutral_range_derivative_inventory() -> None:
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "cmake/GenerativeQCGeneratedSources.cmake").read_text(
        encoding="utf-8"
    )
    generator = (root / "tools/generate_derivative_range_aot.py").read_text(
        encoding="utf-8"
    )
    assert "GENERATIVEQC_RANGE_DERIVATIVE_CPU_AOT_SOURCES" in cmake
    assert "generate_derivative_range_aot.py" in cmake
    assert "manifests/derivative_aot_radials.json" in cmake
    assert "generativeqc_derivative_range_" in cmake
    assert "WB97M-V" not in generator
    assert "resolve_method" not in generator
    assert 'if(_generativeqc_range_shell STREQUAL "1111")' in cmake
