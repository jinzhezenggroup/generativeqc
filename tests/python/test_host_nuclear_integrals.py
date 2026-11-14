"""Independent actual S/H0/J/K nuclear directions without derivative tensors."""

from __future__ import annotations

import ctypes as ct
import itertools
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.factorized_matrix import emit_factorized_matrix_cpu
from generativeqc_compiler.integral.factorized_second import emit_factorized_second
from generativeqc_compiler.integral.one_electron_cpu import emit_one_electron_cpu
from generativeqc_compiler.integral.one_electron_mixed import (
    emit_one_electron_mixed_cpu,
)
from generativeqc_compiler.method.host_hessian_aot import (
    emit_registry,
    parse_inventory,
    programs,
)
from generativeqc_compiler.method.spec import resolve_method

from tools.generate_factorized_second import (
    emit_first_cpu_source,
    emit_value_cpu_source,
)
from tools.generate_host_nuclear_integrals import generate

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
SIZE_POINTER = ct.POINTER(ct.c_size_t)
CENTERS = np.array([[0.13, -0.21, 0.34], [1.7, 0.28, -0.42]])
MOTION = np.array([[0.16, -0.09, 0.13], [-0.12, 0.07, -0.11]])


def _pointer(values: np.ndarray) -> Any:
    return values.ctypes.data_as(POINTER)


def _method(spins: int) -> bytes:
    return resolve_method(
        "B3LYP", spin="unpolarized" if spins == 1 else "polarized"
    ).identity.encode()


def test_map_generation_requires_the_declared_projection_and_no_runtime(
    tmp_path: Path,
) -> None:
    """An exact linked closure is required without importing runtime or oracles."""
    payload = json.loads((ROOT / "manifests/host_hessian_aot_b3lyp.json").read_text())
    payload["arrays"] = ["xc", "coefficient_response"]
    with pytest.raises(ValueError, match="explicit integral_direction leaf"):
        generate(payload)
    inventory = ROOT / "manifests/host_hessian_aot_b3lyp.json"
    output = tmp_path / "maps.hpp"
    script = (
        "import importlib.abc, runpy, sys\n"
        "class Guard(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname.split('.')[0] in {'generativeqc', 'pyscf', 'cupy', 'torch'}:\n"
        "   raise RuntimeError('forbidden build-time dependency: ' + fullname)\n"
        "sys.meta_path.insert(0, Guard())\n"
        f"sys.argv = ['generator', '--inventory', {str(inventory)!r}, '--output', {str(output)!r}]\n"
        f"runpy.run_path({str(ROOT / 'tools/generate_host_nuclear_integrals.py')!r}, run_name='__main__')\n"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert output.read_text() == generate(json.loads(inventory.read_text()))


def test_factorized_source_generation_has_no_runtime_or_oracle_dependency(
    tmp_path: Path,
) -> None:
    """Actual build-only generation works in an isolated import-guarded process."""
    output = tmp_path / "factorized.hpp"
    first_output = tmp_path / "factorized_first.cpp"
    value_output = tmp_path / "factorized_value.cpp"
    matrix_output = tmp_path / "factorized_matrix.cpp"
    script = (
        "import importlib.abc, runpy, sys\n"
        "class Guard(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname.split('.')[0] in {'generativeqc', 'pyscf', 'cupy', 'torch'}:\n"
        "   raise RuntimeError('forbidden build-time dependency: ' + fullname)\n"
        "sys.meta_path.insert(0, Guard())\n"
        f"sys.argv = ['generator', '--output', {str(output)!r}, '--first-output', {str(first_output)!r}, '--value-output', {str(value_output)!r}, '--matrix-output', {str(matrix_output)!r}]\n"
        f"runpy.run_path({str(ROOT / 'tools/generate_factorized_second.py')!r}, run_name='__main__')\n"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert output.read_text() == emit_factorized_second()
    assert matrix_output.read_text() == emit_factorized_matrix_cpu()


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any) -> Any:
    """Compile current consumer, shared S/T/V and selected exact Graph leaves."""
    directory = tmp_path_factory.mktemp("host-nuclear-integrals")
    payload = json.loads((ROOT / "manifests/host_hessian_aot_b3lyp.json").read_text())
    payload["arrays"] = ["xc", "coefficient_response", "integral_direction"]
    inventory = parse_inventory(payload)
    sources = []
    for entry in programs(inventory):
        source = directory / (entry.name + ".cpp")
        source.write_text(entry.emit())
        sources.append(source)
    registry = directory / "registry.cpp"
    registry.write_text(emit_registry(inventory))
    (directory / "host_nuclear_integral_maps.hpp").write_text(generate(payload))
    (directory / "generated_one_electron_cpu.hpp").write_text(emit_one_electron_cpu())
    (directory / "host_factorized_second.hpp").write_text(emit_factorized_second())
    first_source = directory / "host_factorized_first.cpp"
    first_source.write_text(emit_first_cpu_source())
    value_source = directory / "host_factorized_value.cpp"
    value_source.write_text(emit_value_cpu_source())
    matrix_source = directory / "host_factorized_matrix.cpp"
    matrix_source.write_text(emit_factorized_matrix_cpu())
    mixed_source = directory / "one_electron_mixed.cpp"
    mixed_source.write_text(emit_one_electron_mixed_cpu())
    core = Path(policy_library._name).resolve()
    library = required_native_cxx.build_shared(
        [
            first_source,
            value_source,
            matrix_source,
            mixed_source,
            *sources,
            registry,
            ROOT / "src/dft/host_nuclear_integrals.cpp",
            ROOT / "tests/native/host_nuclear_integrals_probe.cpp",
        ],
        directory / "integrals.so",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-isystem",
            str(directory),
        ],
        link_args=[str(core), "-Wl,-rpath," + str(core.parent)],
        compile_timeout=600,
    )
    loaded = ct.CDLL(str(library))
    loaded.host_integral_error.restype = ct.c_char_p
    loaded.host_integral_create.argtypes = [
        POINTER,
        ct.c_uint,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_size_t,
        POINTER,
        ct.c_char_p,
        ct.c_size_t,
    ]
    loaded.host_integral_create.restype = ct.c_void_p
    loaded.host_integral_create_sites.argtypes = [
        POINTER,
        ct.c_size_t,
        ct.c_size_t,
        POINTER,
        ct.c_char_p,
        ct.c_size_t,
    ]
    loaded.host_integral_create_sites.restype = ct.c_void_p
    loaded.host_integral_destroy.argtypes = [ct.c_void_p]
    loaded.host_integral_coordinate_mode.argtypes = [
        ct.c_void_p,
        ct.c_uint,
        ct.c_char_p,
        SIZE_POINTER,
    ]
    loaded.host_integral_prepare_coordinates.argtypes = [ct.c_void_p]
    loaded.host_integral_resident_work.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_integral_density_response_terms.argtypes = [ct.c_void_p]
    loaded.host_integral_density_response_terms.restype = ct.c_size_t
    loaded.host_integral_density_response.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        ct.c_size_t,
    ]
    loaded.host_integral_bind.argtypes = [
        ct.c_void_p,
        ct.c_char_p,
        SIZE_POINTER,
        ct.c_size_t,
    ]
    loaded.host_integral_sizes.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_integral_schedule.argtypes = [
        ct.c_void_p,
        ct.c_uint,
        ct.c_char_p,
        ct.c_size_t,
    ]
    loaded.host_integral_failure.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t]
    loaded.host_integral_model.argtypes = [ct.c_void_p, ct.c_uint]
    loaded.host_integral_packed.argtypes = [ct.c_void_p]
    loaded.host_integral_packed.restype = POINTER
    loaded.host_integral_charge.argtypes = [ct.c_void_p, ct.c_int]
    loaded.host_integral_plan_count.argtypes = [ct.c_void_p, ct.c_size_t]
    loaded.host_integral_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
    ]
    loaded.host_integral_work.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_curvature_bind.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        ct.c_char_p,
        SIZE_POINTER,
        ct.c_size_t,
    ]
    loaded.host_curvature_resources.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_curvature_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
    ]
    loaded.host_curvature_work.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_curvature_eri_orders.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_curvature_matrix_fields.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_column_bind.argtypes = loaded.host_curvature_bind.argtypes
    loaded.host_matrix_bind.argtypes = loaded.host_curvature_bind.argtypes
    loaded.host_projected_matrix_bind.argtypes = loaded.host_curvature_bind.argtypes
    loaded.host_projected_matrix_supported.argtypes = [ct.c_void_p]
    loaded.host_integral_create_mixed_sites.argtypes = [
        POINTER,
        ct.c_int,
        ct.c_size_t,
        POINTER,
        ct.c_char_p,
    ]
    loaded.host_integral_create_mixed_sites.restype = ct.c_void_p
    loaded.host_curvature_density_projection_terms.argtypes = [ct.c_void_p]
    loaded.host_curvature_density_projection_terms.restype = ct.c_size_t
    loaded.host_matrix_apply.argtypes = [ct.c_void_p, POINTER, ct.c_size_t]
    loaded.host_column_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
    ]
    loaded.host_column_response.argtypes = [ct.c_void_p] + [POINTER, ct.c_size_t] * 4
    loaded.host_curvature_plan_schedule.argtypes = [ct.c_void_p, ct.c_uint]
    return loaded


def test_generated_integral_registries_coexist_without_symbol_interposition(
    tmp_path: Path, required_native_cxx: Any
) -> None:
    """Distinct inventories in loaded DSOs must not share ELF GNU-unique tables."""
    payload = json.loads((ROOT / "manifests/host_hessian_aot_b3lyp.json").read_text())
    payload["arrays"] = ["xc", "coefficient_response", "integral_direction"]
    loaded = []
    for label, spins in (
        ("both", ["unpolarized", "polarized"]),
        ("rks", ["unpolarized"]),
    ):
        directory = tmp_path / label
        directory.mkdir()
        inventory = {**payload, "spins": spins}
        (directory / "maps.hpp").write_text(generate(inventory))
        probe = directory / "probe.cpp"
        probe.write_text(
            '#include "maps.hpp"\n#include <string_view>\n'
            'extern "C" int registry_outputs(const char* name) {\n'
            " for (const auto& entry : generativeqc_host_integral_maps::programs())\n"
            "  if (entry.name == name) return static_cast<int>(entry.outputs.size());\n"
            " return -1; }\n"
        )
        library = required_native_cxx.build_shared(
            [probe],
            directory / "probe.so",
            compile_args=["-std=c++20", "-O2"],
            compile_timeout=180,
        )
        module = ct.CDLL(str(library))
        module.registry_outputs.argtypes = [ct.c_char_p]
        loaded.append(module)
    for module in loaded:
        assert module.registry_outputs(b"quartet_response_weight_unpolarized") == 2
        assert module.registry_outputs(b"density_response_unpolarized") == 1
    assert loaded[0].registry_outputs(b"quartet_response_weight_polarized") == 2
    assert loaded[1].registry_outputs(b"quartet_response_weight_polarized") == -1


@contextmanager
def _owner(
    native: Any,
    angular: int = 1,
    spins: int = 1,
    spherical: bool = True,
    contracted: bool = True,
    number: int = 1,
    core: int = 0,
    density: np.ndarray | None = None,
    cadence: int = 64,
    centers: np.ndarray = CENTERS,
    eri_schedule: int = 1,
) -> Any:
    basis = (2 * angular + 1 if spherical else (angular + 1) * (angular + 2) // 2) + 1
    if density is None:
        rng = np.random.default_rng(882)
        density = rng.normal(size=(spins, basis, basis)) * 0.14
        density += density.transpose(0, 2, 1).copy()
    density = np.ascontiguousarray(density, dtype=np.float64)
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    handle = native.host_integral_create(
        _pointer(centers),
        angular,
        spherical,
        contracted,
        number,
        core,
        spins,
        _pointer(density),
        _method(spins),
        cadence,
    )
    if not handle:
        raise RuntimeError(native.host_integral_error().decode())
    try:
        assert (
            native.host_integral_schedule(handle, eri_schedule, _method(spins), cadence)
            == 0
        ), native.host_integral_error().decode()
        sizes = (ct.c_size_t * 7)()
        native.host_integral_sizes(handle, sizes)
        yield handle, density, tuple(sizes)
    finally:
        native.host_integral_destroy(handle)


def _apply(
    native: Any, handle: int, density: np.ndarray, direction: np.ndarray = MOTION
) -> tuple:
    overlap, fock = np.full(density.shape[1:], -781.25), np.full_like(density, -781.25)
    status = native.host_integral_apply(
        handle,
        _pointer(direction),
        direction.size,
        _pointer(overlap),
        overlap.size,
        _pointer(fock),
        fock.size,
    )
    if status:
        raise RuntimeError(native.host_integral_error().decode())
    return overlap, fock


RIGHT = np.array([[-0.08, 0.12, -0.06], [0.11, -0.05, 0.09]])


def _bind_column(
    native: Any, handle: Any, weighted: np.ndarray, cadence: int = 64
) -> tuple[int, ...]:
    """Select the bounded whole-column schedule on the same immutable owner."""
    assert (
        native.host_column_bind(
            handle,
            _pointer(weighted),
            weighted.size,
            _method(len(weighted)),
            None,
            cadence,
        )
        == 0
    ), native.host_integral_error().decode()
    resources = (ct.c_size_t * 4)()
    native.host_curvature_resources(handle, resources)
    return tuple(resources)


def _column(native: Any, handle: Any, right: np.ndarray = RIGHT) -> np.ndarray:
    """Read a whole physical gradient and Hessian action in one source sweep."""
    output = np.full(2 + 2 * right.size, -781.25)
    assert (
        native.host_column_apply(
            handle, _pointer(right), right.size, _pointer(output), output.size
        )
        == 0
    ), native.host_integral_error().decode()
    return output


def _response_column(
    native: Any,
    handle: Any,
    right: np.ndarray,
    density_right: np.ndarray,
    weighted_right: np.ndarray,
    output: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    """Supply explicit right D/W motion without claiming it solves CPKS."""
    if output is None:
        output = np.full(2 + 2 * right.size, -781.25)
    status = native.host_column_response(
        handle,
        _pointer(right),
        right.size,
        _pointer(density_right),
        density_right.size,
        _pointer(weighted_right),
        weighted_right.size,
        _pointer(output),
        output.size,
    )
    return status, output


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("spherical", [False, True])
def test_supplied_dw_column_against_independent_primal(
    native: Any, spins: int, spherical: bool, tmp_path: Path
) -> None:
    """All left components differentiate only geometry, with D/W on the right."""
    with _curvature_owner(
        native, angular=3, spherical=spherical, spins=spins, contracted=False
    ) as (handle, density, weighted, _, _):
        resources = _bind_column(native, handle, weighted)
        random = np.random.default_rng(578)
        moving = random.normal(scale=0.02, size=density.shape)
        density_right = np.ascontiguousarray(moving + moving.transpose(0, 2, 1))
        moving = random.normal(scale=0.01, size=weighted.shape)
        weighted_right = np.ascontiguousarray(moving + moving.transpose(0, 2, 1))
        status, actual = _response_column(
            native, handle, RIGHT, density_right, weighted_right
        )
        assert status == 0, native.host_integral_error().decode()
        work = (ct.c_size_t * 8)()
        native.host_curvature_work(handle, work)
        assert tuple(work)[4:6] == resources[1:3]
        assert work[7] == resources[3]
        steps = np.array([3e-3, 1e-3, 3e-4])
        legs = np.eye(CENTERS.size).reshape(-1, *CENTERS.shape)
        corners, left_corners, right_corners = [], [], []
        for step in steps:
            rows, left_rows = [], []
            for leg in legs:
                rows.append(
                    [
                        _oracle(
                            CENTERS + step * (first * leg + second * RIGHT),
                            density + second * step * density_right,
                            3,
                            spherical,
                            False,
                            1,
                            0,
                            weighted + second * step * weighted_right,
                        )
                        for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                    ]
                )
                left_rows.append(
                    [
                        _oracle(
                            CENTERS + sign * step * leg,
                            density,
                            3,
                            spherical,
                            False,
                            1,
                            0,
                            weighted,
                        )
                        for sign in (1, -1)
                    ]
                )
            corners.append(rows)
            left_corners.append(left_rows)
            right_corners.append(
                [
                    _oracle(
                        CENTERS + sign * step * RIGHT,
                        density + sign * step * density_right,
                        3,
                        spherical,
                        False,
                        1,
                        0,
                        weighted + sign * step * weighted_right,
                    )
                    for sign in (1, -1)
                ]
            )
        corners = np.asarray(corners, dtype=np.float64)
        left_corners = np.asarray(left_corners, dtype=np.float64)
        right_corners = np.asarray(right_corners, dtype=np.float64)
        gradient = (left_corners[:, :, 0] - left_corners[:, :, 1]) / (
            2 * steps[:, None]
        )
        mixed = (
            corners[:, :, 0] - corners[:, :, 1] - corners[:, :, 2] + corners[:, :, 3]
        ) / (4 * steps[:, None] ** 2)
        right_control = (right_corners[:, 0] - right_corners[:, 1]) / (2 * steps)
        expected = np.column_stack((right_control, gradient, mixed))
        errors = np.abs(expected - actual[None, 1:])
        np.savez_compressed(
            tmp_path / "supplied-dw-integral-column.npz",
            centers=CENTERS,
            density=density,
            weighted_density=weighted,
            density_right=density_right,
            weighted_right=weighted_right,
            right=RIGHT,
            native=actual,
            steps=steps,
            primal_corners=corners,
            left_corners=left_corners,
            right_corners=right_corners,
            errors=errors,
            spherical=spherical,
            resources=resources,
            work=list(work),
        )
        np.testing.assert_allclose(
            actual[0],
            _oracle(CENTERS, density, 3, spherical, False, 1, 0, weighted),
            atol=3e-11,
            rtol=3e-10,
        )
        np.testing.assert_allclose(actual[1:], expected[-1], atol=3e-8, rtol=2e-6)
        assert errors[-1].max() < max(errors[0].max() / 8, 5e-9), errors
        frozen = _column(native, handle)
        status, response = _response_column(
            native, handle, np.zeros_like(RIGHT), density_right, weighted_right
        )
        assert status == 0
        np.testing.assert_array_equal(actual[2:8], frozen[2:8])
        np.testing.assert_allclose(
            actual[1], frozen[1] + response[1], atol=3e-12, rtol=3e-11
        )
        np.testing.assert_allclose(
            actual[8:], frozen[8:] + response[8:], atol=3e-12, rtol=3e-11
        )
        split = []
        for first, second in (
            (density_right, np.empty(0)),
            (np.empty(0), weighted_right),
        ):
            status, part = _response_column(
                native, handle, np.zeros_like(RIGHT), first, second
            )
            assert status == 0
            split.append(part)
        np.testing.assert_allclose(
            response[1], split[0][1] + split[1][1], atol=3e-12, rtol=3e-11
        )
        np.testing.assert_allclose(
            response[8:], split[0][8:] + split[1][8:], atol=3e-12, rtol=3e-11
        )


@pytest.mark.parametrize("spins", [1, 2])
def test_supplied_dw_column_refusal_alias_mutation_and_recovery(
    native: Any, spins: int
) -> None:
    """Invalid and changing right inputs cannot publish partially accumulated fields."""
    with _curvature_owner(native, angular=1, spins=spins) as (
        handle,
        density,
        weighted,
        sizes,
        _,
    ):
        _bind_column(native, handle, weighted)
        density_right, weighted_right = 0.07 * density, -0.04 * weighted
        status, actual = _response_column(
            native, handle, RIGHT, density_right, weighted_right
        )
        assert status == 0, native.host_integral_error().decode()
        for index in range(2):
            malformed = [density_right.copy(), weighted_right.copy()]
            for invalid in (
                np.full(density.shape, np.nan),
                np.full(density.shape, np.inf),
                np.arange(density.size, dtype=float).reshape(density.shape),
                density_right.ravel()[:-1],
            ):
                malformed[index] = invalid
                status, output = _response_column(native, handle, RIGHT, *malformed)
                assert status != 0
                np.testing.assert_array_equal(output, -781.25)
            counts = [density.size, weighted.size]
            pointers = [_pointer(density_right), _pointer(weighted_right)]
            pointers[index] = POINTER()
            output = np.full_like(actual, -781.25)
            assert (
                native.host_column_response(
                    handle,
                    _pointer(RIGHT),
                    RIGHT.size,
                    pointers[0],
                    counts[0],
                    pointers[1],
                    counts[1],
                    _pointer(output),
                    output.size,
                )
                != 0
            )
            np.testing.assert_array_equal(output, -781.25)
        for index in range(3):
            shared = np.full(max(actual.size, density.size, RIGHT.size), -781.25)
            inputs = [RIGHT.copy(), density_right.copy(), weighted_right.copy()]
            shared[: inputs[index].size] = inputs[index].ravel()
            inputs[index] = shared[: inputs[index].size].reshape(inputs[index].shape)
            status, output = _response_column(
                native, handle, *inputs, output=shared[: actual.size]
            )
            assert status == 0, native.host_integral_error().decode()
            np.testing.assert_allclose(output, actual, atol=3e-12, rtol=3e-11)
        for mode in (8, 9, 10):
            _bind_column(native, handle, weighted)
            inputs = [RIGHT.copy(), density_right.copy(), weighted_right.copy()]
            native.host_integral_failure(handle, mode, sizes[0] ** 4)
            status, output = _response_column(native, handle, *inputs)
            assert status != 0
            np.testing.assert_array_equal(output, -781.25)
            status, recovered = _response_column(
                native, handle, RIGHT, density_right, weighted_right
            )
            assert status == 0, native.host_integral_error().decode()
            np.testing.assert_allclose(recovered, actual, atol=3e-12, rtol=3e-11)


@pytest.mark.parametrize(
    ("angular", "spherical", "spins", "number", "core"),
    [
        (angular, spherical, spins, 1, 0)
        for angular in range(4)
        for spherical in (False, True)
        for spins in (1, 2)
    ]
    + [(1, True, 2, 28, 10), (1, True, 2, 53, 28), (1, True, 2, 55, 46)],
)
def test_frozen_column_against_independent_primal_energy(
    native: Any,
    angular: int,
    spherical: bool,
    spins: int,
    number: int,
    core: int,
    tmp_path: Path,
) -> None:
    """All physical coordinates match independent libcint frozen D/W energies."""
    contracted = angular <= 1
    with _curvature_owner(
        native,
        angular=angular,
        spherical=spherical,
        spins=spins,
        contracted=contracted,
        number=number,
        core=core,
    ) as (handle, density, weighted, sizes, scalar_resources):
        scalar = _mixed(native, handle)
        resources = _bind_column(native, handle, weighted)
        actual = _column(native, handle)
        np.testing.assert_allclose(
            [
                actual[0],
                actual[2:8] @ MOTION.ravel(),
                actual[1],
                actual[8:] @ MOTION.ravel(),
            ],
            scalar,
            atol=3e-12,
            rtol=3e-11,
        )
        primal = _oracle(
            CENTERS, density, angular, spherical, contracted, number, core, weighted
        )
        np.testing.assert_allclose(actual[0], primal, atol=3e-11, rtol=3e-10)
        steps = np.array([3e-3, 1e-3, 3e-4])
        legs = np.eye(CENTERS.size).reshape(-1, *CENTERS.shape)
        corners = np.array(
            [
                [
                    [
                        _oracle(
                            CENTERS + step * (first * leg + second * RIGHT),
                            density,
                            angular,
                            spherical,
                            contracted,
                            number,
                            core,
                            weighted,
                        )
                        for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                    ]
                    for leg in legs
                ]
                for step in steps
            ]
        )
        gradients = (
            corners[:, :, 0] + corners[:, :, 1] - corners[:, :, 2] - corners[:, :, 3]
        ) / (4 * steps[:, None])
        mixed = (
            corners[:, :, 0] - corners[:, :, 1] - corners[:, :, 2] + corners[:, :, 3]
        ) / (4 * steps[:, None] ** 2)
        expected = np.concatenate((gradients, mixed), axis=1)
        errors = np.abs(expected - actual[None, 2:])
        np.testing.assert_allclose(actual[2:], expected[-1], atol=3e-8, rtol=2e-6)
        assert errors[-1].max() < max(errors[0].max() / 8, 5e-9), errors
        np.testing.assert_allclose(
            actual[2:8] @ RIGHT.ravel(), actual[1], atol=3e-12, rtol=3e-11
        )
        work = (ct.c_size_t * 8)()
        native.host_curvature_work(handle, work)
        assert work[0] == work[1] == 1
        assert tuple(work)[2:6] == (
            sizes[0] ** 2,
            sizes[0] ** 4,
            resources[1],
            resources[2],
        )
        assert work[7] == resources[3] >= scalar_resources[3]
        assert resources[2] == scalar_resources[2]
        np.savez_compressed(
            tmp_path / "frozen-integral-column.npz",
            native=actual,
            density=density,
            weighted_density=weighted,
            primal_baseline=primal,
            centers=CENTERS,
            right=RIGHT,
            steps=steps,
            primal_corners=corners,
            errors=errors,
            resources=resources,
            work=tuple(work),
            angular=angular,
            spherical=spherical,
            spins=spins,
            number=number,
            core=core,
            method_identity=_method(spins).decode(),
            basis_sizes=sizes,
            ao_basis_packed=np.ctypeslib.as_array(
                native.host_integral_packed(handle),
                shape=(3 * sizes[1] + 2 * sizes[6] + 16 * sizes[0],),
            ).copy(),
        )


@pytest.mark.parametrize("cadence", [1, 4096])
def test_frozen_column_transaction_caps_alias_owner_and_recovery(
    native: Any, cadence: int
) -> None:
    """Whole columns retain all-or-nothing publication and authenticated owners."""
    with _curvature_owner(native, spins=2) as (handle, density, weighted, sizes, _):
        for schedule in (0, 1, 2, 3):
            assert native.host_curvature_plan_schedule(handle, schedule) == 0
        for schedule in (4, 4294967295):
            assert native.host_curvature_plan_schedule(handle, schedule) != 0
        refused = np.full(14, -781.25)
        assert (
            native.host_column_apply(handle, _pointer(RIGHT), 6, _pointer(refused), 14)
            != 0
        )
        np.testing.assert_array_equal(refused, -781.25)
        resources = _bind_column(native, handle, weighted, cadence)
        actual = _column(native, handle)
        for index in range(4):
            limits = list(resources)
            limits[index] -= 1
            assert (
                native.host_column_bind(
                    handle,
                    _pointer(weighted),
                    weighted.size,
                    _method(2),
                    (ct.c_size_t * 4)(*limits),
                    cadence,
                )
                != 0
            )
        for method, invalid_cadence in (
            (b"forged", cadence),
            (_method(1), cadence),
            (_method(2), 0),
            (_method(2), 4097),
        ):
            assert (
                native.host_column_bind(
                    handle,
                    _pointer(weighted),
                    weighted.size,
                    method,
                    None,
                    invalid_cadence,
                )
                != 0
            )
        for mode in range(1, 8):
            native.host_integral_model(handle, mode)
            assert (
                native.host_column_bind(
                    handle, _pointer(weighted), weighted.size, _method(2), None, cadence
                )
                != 0
            )
            native.host_integral_model(handle, 0)
        for mode, at in (
            (1, 1),
            (1, 4),
            (2, 3),
            (3, 5),
            (4, 3),
            (5, 3),
            (6, 3),
            (7, 0),
        ):
            output = np.full_like(actual, -781.25)
            before, after = (ct.c_size_t * 8)(), (ct.c_size_t * 8)()
            native.host_curvature_work(handle, before)
            native.host_integral_failure(handle, mode, at)
            assert (
                native.host_column_apply(
                    handle, _pointer(RIGHT), 6, _pointer(output), output.size
                )
                != 0
            )
            np.testing.assert_array_equal(output, -781.25)
            native.host_curvature_work(handle, after)
            assert after[1] == before[1]
            if mode == 7:
                assert after[3] - before[3] == sizes[0] ** 4
            native.host_integral_failure(handle, 0, 0)
            np.testing.assert_array_equal(_column(native, handle), actual)
        for values in (density, weighted):
            saved = values.copy()
            values[0, 0, 0] += 0.125
            output = np.full_like(actual, -781.25)
            assert (
                native.host_column_apply(
                    handle, _pointer(RIGHT), 6, _pointer(output), output.size
                )
                != 0
            )
            np.testing.assert_array_equal(output, -781.25)
            values[:] = saved
        for pointer in (
            _pointer(density),
            _pointer(weighted),
            native.host_integral_packed(handle),
        ):
            assert (
                native.host_column_apply(
                    handle, _pointer(RIGHT), 6, pointer, actual.size
                )
                != 0
            )
        for right_count, output_count in (
            (0, actual.size),
            (5, actual.size),
            (6, actual.size - 1),
        ):
            output = np.full_like(actual, -781.25)
            assert (
                native.host_column_apply(
                    handle, _pointer(RIGHT), right_count, _pointer(output), output_count
                )
                != 0
            )
            np.testing.assert_array_equal(output, -781.25)
        for right_pointer, output_pointer in (
            (None, _pointer(actual)),
            (_pointer(RIGHT), None),
        ):
            assert (
                native.host_column_apply(
                    handle, right_pointer, 6, output_pointer, actual.size
                )
                != 0
            )
        malformed = RIGHT.copy()
        malformed[0, 0] = np.nan
        output = np.full_like(actual, -781.25)
        assert (
            native.host_column_apply(
                handle, _pointer(malformed), 6, _pointer(output), output.size
            )
            != 0
        )
        np.testing.assert_array_equal(output, -781.25)
        assert (
            native.host_curvature_apply(
                handle, _pointer(MOTION), 6, _pointer(RIGHT), 6, _pointer(output), 4
            )
            != 0
        )
        shared = np.full_like(actual, -781.25)
        shared[:6] = RIGHT.ravel()
        assert (
            native.host_column_apply(
                handle, _pointer(shared), 6, _pointer(shared), shared.size
            )
            == 0
        )
        np.testing.assert_array_equal(shared, actual)
        rigid = _column(native, handle, np.tile([0.17, -0.08, 0.11], (2, 1)))
        np.testing.assert_array_equal(rigid[8:], 0)
        assert rigid[1] == 0
        np.testing.assert_allclose(rigid[:1], actual[:1], atol=0, rtol=0)
        np.testing.assert_allclose(rigid[2:8], actual[2:8], atol=0, rtol=0)
        first, second = _column(native, handle, MOTION), _column(native, handle, RIGHT)
        combined = _column(native, handle, np.ascontiguousarray(MOTION + 2 * RIGHT))
        np.testing.assert_allclose(
            combined[8:], first[8:] + 2 * second[8:], atol=3e-12, rtol=3e-11
        )


@pytest.mark.parametrize("spins", [1, 2])
def test_frozen_column_empty_density_retains_only_constraint(
    native: Any, spins: int
) -> None:
    """Vacuum D does not remove finite W; completely empty D/W is exactly zero."""
    density = np.zeros((spins, 2, 2))
    with _curvature_owner(native, angular=0, spins=spins, density=density) as (
        handle,
        _,
        weighted,
        _,
        _,
    ):
        scalar = _mixed(native, handle)
        _bind_column(native, handle, weighted)
        actual = _column(native, handle)
        np.testing.assert_allclose(
            actual[8:] @ MOTION.ravel(), scalar[3], atol=3e-12, rtol=3e-11
        )
        assert np.linalg.norm(actual) > 0
        weighted.fill(0)
        _bind_column(native, handle, weighted)
        np.testing.assert_array_equal(_column(native, handle), 0)


def test_frozen_column_one_empty_uks_spin(native: Any) -> None:
    """The nonvacuum UKS empty-spin limit follows the same canonical weights."""
    density = np.array([[[0.3, -0.1], [-0.1, 0.2]], [[0.0, 0.0], [0.0, 0.0]]])
    with _curvature_owner(native, angular=0, spins=2, density=density) as (
        handle,
        _,
        weighted,
        _,
        _,
    ):
        scalar = _mixed(native, handle)
        _bind_column(native, handle, weighted)
        actual = _column(native, handle)
        np.testing.assert_allclose(
            [
                actual[0],
                actual[2:8] @ MOTION.ravel(),
                actual[1],
                actual[8:] @ MOTION.ravel(),
            ],
            scalar,
            atol=3e-12,
            rtol=3e-11,
        )


@pytest.mark.parametrize("angular", range(4))
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("spins", [1, 2])
def test_eri_symmetry_schedules_preserve_all_physical_sources(
    native: Any, angular: int, spherical: bool, spins: int, tmp_path: Path
) -> None:
    """Distinct ordered incidences survive canonical integral reuse, including D/W motion."""
    results = []
    plans = []
    work_counts = []
    for schedule in (0, 1):
        with _curvature_owner(
            native,
            angular=angular,
            spherical=spherical,
            spins=spins,
            contracted=angular <= 1,
            eri_schedule=schedule,
        ) as (handle, density, weighted, _, plan):
            overlap, fock = _apply(native, handle, density)
            mixed = _mixed(native, handle)
            column_plan = _bind_column(native, handle, weighted)
            frozen = _column(native, handle)
            rng = np.random.default_rng(875)
            moving = rng.normal(scale=0.02, size=density.shape)
            density_right = np.ascontiguousarray(moving + moving.transpose(0, 2, 1))
            moving = rng.normal(scale=0.01, size=weighted.shape)
            weighted_right = np.ascontiguousarray(moving + moving.transpose(0, 2, 1))
            status, moving_column = _response_column(
                native, handle, RIGHT, density_right, weighted_right
            )
            assert status == 0, native.host_integral_error().decode()
            translation = np.tile([0.125, -0.25, 0.5], (2, 1))
            status, translated_density = _response_column(
                native, handle, translation, density_right, weighted_right
            )
            assert status == 0, native.host_integral_error().decode()
            work = (ct.c_size_t * 8)()
            native.host_curvature_work(handle, work)
            results.append(
                np.concatenate(
                    [
                        overlap.ravel(),
                        fock.ravel(),
                        mixed,
                        frozen,
                        moving_column,
                        translated_density,
                    ]
                )
            )
            plans.append((plan, column_plan))
            work_counts.append(tuple(work))
    np.testing.assert_allclose(results[1], results[0], atol=3e-12, rtol=3e-11)
    assert plans[1][0][2] < plans[0][0][2]
    assert plans[1][1][2] < plans[0][1][2]
    assert plans[1][0][0:2] == plans[0][0][0:2]
    assert plans[1][1][0:2] == plans[0][1][0:2]
    assert work_counts[1][3] == work_counts[0][3]
    assert work_counts[1][7] == work_counts[0][7]
    np.savez_compressed(
        tmp_path / "eri-symmetry-schedules.npz",
        ordered=results[0],
        symmetry_unique=results[1],
        plans=plans,
        work=work_counts,
    )


def test_eri_schedule_selection_failure_preserves_bound_sources(native: Any) -> None:
    """The test selector must not leave limits or owners on different schedules."""
    with _owner(native) as (handle, density, sizes):
        expected = _apply(native, handle, density)
        for schedule, method in ((2, _method(1)), (0, b"invalid-method")):
            assert native.host_integral_schedule(handle, schedule, method, 64) != 0
            for actual, baseline in zip(
                _apply(native, handle, density), expected, strict=True
            ):
                np.testing.assert_array_equal(actual, baseline)
            after = (ct.c_size_t * 7)()
            native.host_integral_sizes(handle, after)
            assert tuple(after) == sizes
        weighted = np.zeros_like(density)
        _bind_column(native, handle, weighted)
        baseline = _column(native, handle)
        assert native.host_integral_schedule(handle, 0, _method(1), 64) != 0
        np.testing.assert_array_equal(_column(native, handle), baseline)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("atoms", [8, 32])
def test_sparse_first_direction_avoids_common_translation_eri_work(
    native: Any, atoms: int, spins: int, tmp_path: Path
) -> None:
    """Sparse physical motion reduces derivative work without screening densities."""
    centers = np.ascontiguousarray(
        np.column_stack(
            (
                np.arange(atoms) * 1.5,
                0.19 * np.sin(np.arange(atoms)),
                0.13 * np.cos(np.arange(atoms)),
            )
        )
    )
    rng = np.random.default_rng(390)
    density = rng.normal(scale=0.02, size=(spins, atoms, atoms))
    density = np.ascontiguousarray(density + density.transpose(0, 2, 1))
    direction = np.zeros_like(centers)
    direction[atoms // 2] = [0.13, -0.07, 0.11]
    outputs, records = [], []
    for schedule in (0, 1):
        handle = native.host_integral_create_sites(
            _pointer(centers), atoms, spins, _pointer(density), _method(spins), 64
        )
        assert handle, native.host_integral_error().decode()
        try:
            assert (
                native.host_integral_schedule(handle, schedule, _method(spins), 64) == 0
            ), native.host_integral_error().decode()
            sizes = (ct.c_size_t * 7)()
            native.host_integral_sizes(handle, sizes)
            started = time.perf_counter()
            outputs.append(_apply(native, handle, density, direction))
            seconds = time.perf_counter() - started
            work = (ct.c_size_t * 8)()
            native.host_integral_work(handle, work)
            records.append(
                {"seconds": seconds, "work": tuple(work), "plan": tuple(sizes)}
            )
            if schedule:
                assert 0 < work[5] <= 4 / atoms * sizes[4]
                shifted = _apply(
                    native, handle, density, direction + [0.25, -0.5, 0.125]
                )
                for actual, expected in zip(shifted, outputs[-1], strict=True):
                    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)
                before_translation = (ct.c_size_t * 8)()
                native.host_integral_work(handle, before_translation)
                translated = _apply(
                    native, handle, density, np.tile([0.25, -0.5, 0.125], (atoms, 1))
                )
                for actual in translated:
                    np.testing.assert_allclose(actual, 0, atol=2e-12)
                native.host_integral_work(handle, work)
                assert work[5] == before_translation[5]
        finally:
            native.host_integral_destroy(handle)
    for actual, expected in zip(outputs[1], outputs[0], strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)
    (tmp_path / "sparse-direction-work.json").write_text(
        json.dumps(
            {
                "scope": "frozen-density source only, not a complete electronic endpoint",
                "atoms": atoms,
                "spins": spins,
                "schedules": records,
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        tmp_path / "sparse-direction-results.npz",
        centers=centers,
        density=density,
        direction=direction,
        ordered_overlap=outputs[0][0],
        ordered_fock=outputs[0][1],
        optimized_overlap=outputs[1][0],
        optimized_fock=outputs[1][1],
    )


def test_first_direction_has_no_small_displacement_screen(native: Any) -> None:
    """Distinct, tiny motions retain the exact linear operator in both spins."""
    direction = np.zeros_like(CENTERS)
    direction[1] = [0.13, -0.07, 0.11]
    for spins in (1, 2):
        results = []
        for schedule in (0, 1):
            with _owner(native, spins=spins, eri_schedule=schedule) as (
                handle,
                density,
                _,
            ):
                actual = _apply(native, handle, density, direction)
                tiny = _apply(native, handle, density, direction * 1e-12)
                for small, expected in zip(tiny, actual, strict=True):
                    np.testing.assert_allclose(
                        small, expected * 1e-12, atol=3e-26, rtol=2e-12
                    )
                results.append(tiny)
        for actual, expected in zip(results[1], results[0], strict=True):
            np.testing.assert_allclose(actual, expected, atol=3e-26, rtol=2e-12)


@pytest.mark.parametrize("spins", [1, 2])
def test_common_translation_keeps_explicit_density_curvature(
    native: Any, spins: int, tmp_path: Path
) -> None:
    """Zero integral direction does not remove gradient times moving-D/W terms."""
    translation = np.tile([0.125, -0.25, 0.5], (2, 1))
    with _curvature_owner(native, spins=spins) as (handle, density, weighted, _, _):
        _bind_column(native, handle, weighted)
        rng = np.random.default_rng(487)
        moving = rng.normal(scale=0.02, size=density.shape)
        density_right = np.ascontiguousarray(moving + moving.transpose(0, 2, 1))
        moving = rng.normal(scale=0.01, size=weighted.shape)
        weighted_right = np.ascontiguousarray(moving + moving.transpose(0, 2, 1))
        frozen = _column(native, handle, translation)
        np.testing.assert_allclose(frozen[8:], 0, atol=2e-13)
        status, actual = _response_column(
            native, handle, translation, density_right, weighted_right
        )
        assert status == 0, native.host_integral_error().decode()
        assert np.max(np.abs(actual[8:])) > 1e-5
        orders, work = (ct.c_size_t * 3)(), (ct.c_size_t * 8)()
        native.host_curvature_eri_orders(handle, orders)
        native.host_curvature_work(handle, work)
        assert sum(orders) == work[5]
        assert orders[0] > 0 and orders[1] > 0 and orders[2] == 0
        steps = np.array([1e-3, 3e-4])
        references = []
        for step in steps:
            column = []
            for leg in np.eye(6).reshape(-1, 2, 3):
                corners = [
                    _oracle(
                        CENTERS + step * (left_sign * leg + right_sign * translation),
                        density + right_sign * step * density_right,
                        1,
                        True,
                        True,
                        1,
                        0,
                        weighted + right_sign * step * weighted_right,
                    )
                    for left_sign, right_sign in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                ]
                column.append(
                    (corners[0] - corners[1] - corners[2] + corners[3]) / (4 * step**2)
                )
            references.append(column)
        np.testing.assert_allclose(actual[8:], references[-1], atol=2e-8, rtol=3e-6)
        np.savez_compressed(
            tmp_path / "common-translation-density-curvature.npz",
            density=density,
            weighted=weighted,
            density_right=density_right,
            weighted_right=weighted_right,
            translation=translation,
            native=actual,
            steps=steps,
            independent_mixed=references,
            eri_derivative_orders=tuple(orders),
        )


@pytest.mark.parametrize("spins", [1, 2])
def test_curvature_keeps_arbitrarily_small_distinct_right_motion(
    native: Any,
    spins: int,
) -> None:
    """Only exact relative zero may lower the requested derivative order."""
    with _curvature_owner(native, spins=spins) as (handle, _, weighted, _, _):
        _bind_column(native, handle, weighted)
        baseline = _column(native, handle, RIGHT)
        scale = 1e-20
        actual = _column(native, handle, scale * RIGHT)
        np.testing.assert_allclose(
            actual[8:], scale * baseline[8:], atol=3e-32, rtol=3e-11
        )
        assert np.max(abs(actual[8:])) > 1e-25


def test_coincident_distinct_atom_owners_keep_physical_curvature(native: Any) -> None:
    """Coincident coordinates do not identify independently moving nuclei."""
    centers = np.zeros((2, 3))
    density = np.array([[[0.2, 0.03], [0.03, 0.16]], [[0.1, 0.02], [0.02, 0.08]]])
    weighted = np.stack([np.eye(2) * 0.03, np.eye(2) * 0.02])
    right = np.array([[0.1, 0.05, -0.03], [-0.04, 0.07, 0.09]])
    columns = []
    for schedule in (0, 1):
        handle = native.host_integral_create_sites(
            _pointer(centers), 2, 2, _pointer(density), _method(2), 64
        )
        assert handle, native.host_integral_error().decode()
        try:
            assert native.host_integral_schedule(handle, schedule, _method(2), 64) == 0
            _bind_column(native, handle, weighted)
            actual = _column(native, handle, right)
            assert np.max(np.abs(actual[8:])) > 1e-6
            columns.append(actual)
            orders = (ct.c_size_t * 3)()
            native.host_curvature_eri_orders(handle, orders)
            assert orders[2] > 0
            if schedule:
                assert orders[0] > 0
            else:
                assert orders[0] == orders[1] == 0
        finally:
            native.host_integral_destroy(handle)
    np.testing.assert_allclose(columns[1], columns[0], atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("motion", ["dense", "sparse"])
@pytest.mark.parametrize("eri_schedule", [0, 1])
@pytest.mark.parametrize("atoms", [3, 8, 32])
def test_frozen_column_complete_source_reuse_at_larger_coordinate_counts(
    native: Any, atoms: int, eri_schedule: int, motion: str, tmp_path: Path
) -> None:
    """One ERI sweep per right leg, not per physical left coordinate at 32 sites."""
    centers = np.column_stack(
        (
            np.arange(atoms) * 1.5,
            0.19 * np.sin(np.arange(atoms)),
            0.13 * np.cos(np.arange(atoms)),
        )
    )
    rng = np.random.default_rng(190)
    density = rng.normal(size=(2, atoms, atoms)) * 0.02
    density += density.transpose(0, 2, 1).copy()
    weighted = rng.normal(size=density.shape) * 0.01
    weighted += weighted.transpose(0, 2, 1).copy()
    right = rng.normal(size=centers.shape) * 0.1
    if motion == "sparse":
        right[1:] = 0
    handle = native.host_integral_create_sites(
        _pointer(centers), atoms, 2, _pointer(density), _method(2), 64
    )
    assert handle, native.host_integral_error().decode()
    try:
        assert (
            native.host_integral_schedule(handle, eri_schedule, _method(2), 64) == 0
        ), native.host_integral_error().decode()
        resources = _bind_column(native, handle, weighted)
        start = time.perf_counter()
        actual = _column(native, handle, right)
        seconds = time.perf_counter() - start
        work = (ct.c_size_t * 8)()
        native.host_curvature_work(handle, work)
        orders = (ct.c_size_t * 3)()
        native.host_curvature_eri_orders(handle, orders)
        assert sum(orders) == work[5]
        assert orders[2] > 0
        if eri_schedule:
            assert orders[0] > 0
            if motion == "sparse":
                assert orders[1] > 0
        else:
            assert orders[0] == orders[1] == 0
        assert work[0] == work[1] == 1
        assert work[2] == atoms**2
        pairs = atoms * (atoms + 1) // 2
        primitive_work = pairs * (pairs + 1) // 2 if eri_schedule else atoms**4
        assert work[3] == atoms**4
        assert work[5] == resources[2] == primitive_work
        assert work[4] == resources[1] == atoms**2 * (3 + 6 * atoms)
        assert work[7] == resources[3]
        assert resources[0] < 4096 + 160 * atoms + 32 * atoms**2
        sizes = (ct.c_size_t * 7)()
        native.host_integral_sizes(handle, sizes)
        np.testing.assert_allclose(
            actual[2 : 2 + 3 * atoms].reshape(atoms, 3).sum(axis=0), 0, atol=2e-12
        )
        np.testing.assert_allclose(
            actual[2 + 3 * atoms :].reshape(atoms, 3).sum(axis=0), 0, atol=2e-12
        )
        np.savez_compressed(
            tmp_path / "frozen-column-source-reuse.npz",
            native=actual,
            density=density,
            weighted_density=weighted,
            centers=centers,
            right=right,
            resources=resources,
            work=tuple(work),
            eri_derivative_orders=tuple(orders),
            motion=motion,
            seconds=seconds,
            method_identity=_method(2).decode(),
            eri_schedule=eri_schedule,
            basis_sizes=tuple(sizes),
            ao_basis_packed=np.ctypeslib.as_array(
                native.host_integral_packed(handle),
                shape=(3 * sizes[1] + 2 * sizes[6] + 16 * sizes[0],),
            ).copy(),
        )
        if atoms == 3:
            _qualify_three_slot_column(
                native, handle, centers, density, weighted, right, actual, tmp_path
            )
    finally:
        native.host_integral_destroy(handle)


def _qualify_three_slot_column(
    native: Any,
    handle: Any,
    centers: np.ndarray,
    density: np.ndarray,
    weighted: np.ndarray,
    right: np.ndarray,
    actual: np.ndarray,
    tmp_path: Path,
) -> None:
    """Check an independent ionic slot and complete frozen matrix, without SCF."""
    gto = pytest.importorskip("pyscf.gto")

    def energy(geometry: np.ndarray) -> float:
        molecule = gto.M(
            atom=[("H", position) for position in geometry],
            basis={"H": [[0, [0.6, 1]]]},
            spin=1,
            unit="Bohr",
            verbose=0,
        )
        overlap = molecule.intor("int1e_ovlp")
        normalization = 1 / np.sqrt(np.diag(overlap))
        scale = normalization[:, None] * normalization[None, :]
        hcore = molecule.intor("int1e_kin")
        for position in geometry:
            with molecule.with_rinv_origin(position):
                hcore -= molecule.intor("int1e_rinv")
        hcore *= scale
        overlap *= scale
        eri = molecule.intor("int2e", aosym="s1") * (
            normalization[:, None, None, None]
            * normalization[None, :, None, None]
            * normalization[None, None, :, None]
            * normalization[None, None, None, :]
        )
        total = density.sum(axis=0)
        return float(
            np.einsum("sij,ij->", density, hcore)
            + 0.5 * np.einsum("ij,kl,ijkl->", total, total, eri)
            - 0.1 * np.einsum("sij,skl,ikjl->", density, density, eri)
            - np.einsum("sij,ij->", weighted, overlap)
        )

    primal = energy(centers)
    np.testing.assert_allclose(actual[0], primal, atol=3e-11, rtol=3e-10)
    steps = np.array([3e-3, 1e-3, 3e-4])
    legs = np.eye(centers.size).reshape(-1, *centers.shape)
    corners = np.array(
        [
            [
                [
                    energy(centers + step * (first * leg + second * right))
                    for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                ]
                for leg in legs
            ]
            for step in steps
        ]
    )
    gradient = (
        corners[:, :, 0] + corners[:, :, 1] - corners[:, :, 2] - corners[:, :, 3]
    ) / (4 * steps[:, None])
    mixed = (
        corners[:, :, 0] - corners[:, :, 1] - corners[:, :, 2] + corners[:, :, 3]
    ) / (4 * steps[:, None] ** 2)
    expected = np.concatenate((gradient, mixed), axis=1)
    errors = np.abs(expected - actual[None, 2:])
    np.testing.assert_allclose(actual[2:], expected[-1], atol=3e-8, rtol=2e-6)
    assert errors[-1].max() < max(errors[0].max() / 8, 5e-9), errors
    matrix = np.column_stack(
        [_column(native, handle, leg)[2 + centers.size :] for leg in legs]
    )
    np.testing.assert_allclose(matrix, matrix.T, atol=3e-12, rtol=3e-11)
    np.testing.assert_allclose(
        matrix @ right.ravel(), actual[2 + centers.size :], atol=3e-12, rtol=3e-11
    )
    np.testing.assert_allclose(matrix.reshape(3, 3, 3, 3).sum(axis=0), 0, atol=3e-12)
    np.savez_compressed(
        tmp_path / "frozen-column-three-slot.npz",
        native=actual,
        density=density,
        weighted_density=weighted,
        centers=centers,
        right=right,
        steps=steps,
        primal_corners=corners,
        primal_baseline=primal,
        errors=errors,
        matrix=matrix,
    )


@pytest.mark.parametrize("angular", range(4))
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("spins", [1, 2])
def test_frozen_curvature_against_independent_primal_energy(
    native: Any,
    angular: int,
    spherical: bool,
    spins: int,
    tmp_path: Path,
) -> None:
    """Complete ordered contracted/spherical frozen sources match libcint energy."""
    contracted = angular <= 1
    with _curvature_owner(
        native, angular=angular, spherical=spherical, spins=spins, contracted=contracted
    ) as (handle, density, weighted, sizes, resources):
        actual = _mixed(native, handle)
        primal = _oracle(
            CENTERS, density, angular, spherical, contracted, 1, 0, weighted
        )
        np.testing.assert_allclose(actual[0], primal, atol=3e-11, rtol=3e-10)
        steps = np.array([1e-2, 3e-3, 1e-3])
        corners = np.array(
            [
                [
                    _oracle(
                        CENTERS + step * (first * MOTION + second * RIGHT),
                        density,
                        angular,
                        spherical,
                        contracted,
                        1,
                        0,
                        weighted,
                    )
                    for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                ]
                for step in steps
            ]
        )
        expected = np.column_stack(
            [
                (corners[:, 0] + corners[:, 1] - corners[:, 2] - corners[:, 3])
                / (4 * steps),
                (corners[:, 0] - corners[:, 1] + corners[:, 2] - corners[:, 3])
                / (4 * steps),
                (corners[:, 0] - corners[:, 1] - corners[:, 2] + corners[:, 3])
                / (4 * steps**2),
            ]
        )
        errors = np.abs(expected - actual[None, 1:])
        np.testing.assert_allclose(actual[1:], expected[-1], atol=3e-8, rtol=2e-6)
        assert errors[-1].max() < max(errors[0].max() / 8, 5e-9), errors
        work = (ct.c_size_t * 8)()
        native.host_curvature_work(handle, work)
        assert work[0] == work[1] == 1
        assert tuple(work)[2:6] == (
            sizes[0] ** 2,
            sizes[0] ** 4,
            resources[1],
            resources[2],
        )
        assert work[7] == resources[3]
        np.savez_compressed(
            tmp_path / "frozen-integral-curvature.npz",
            native=actual,
            density=density,
            weighted_density=weighted,
            primal_baseline=primal,
            centers=CENTERS,
            left=MOTION,
            right=RIGHT,
            steps=steps,
            primal_corners=corners,
            errors=errors,
            resources=resources,
            work=tuple(work),
        )


@pytest.mark.parametrize("spins", [1, 2])
def test_frozen_curvature_symmetry_translation_and_bilinearity(
    native: Any, spins: int
) -> None:
    """D/W stay fixed while both nuclear legs obey the mixed-source algebra."""
    with _curvature_owner(native, spins=spins) as (handle, _, _, _, _):
        actual = _mixed(native, handle)
        reverse = _mixed(native, handle, RIGHT, MOTION)
        np.testing.assert_allclose(
            actual, reverse[[0, 2, 1, 3]], atol=8e-13, rtol=3e-11
        )
        other = _mixed(native, handle, RIGHT, RIGHT)
        combined = _mixed(native, handle, 2 * MOTION - 0.3 * RIGHT, RIGHT)
        np.testing.assert_allclose(
            combined[[1, 3]],
            2 * actual[[1, 3]] - 0.3 * other[[1, 3]],
            atol=8e-13,
            rtol=3e-11,
        )
        translation = np.tile([0.125, -0.25, 0.375], (2, 1))
        np.testing.assert_array_equal(_mixed(native, handle, translation)[[1, 3]], 0)
        np.testing.assert_array_equal(
            _mixed(native, handle, right=translation)[[2, 3]], 0
        )
        for alias_left in (True, False):
            buffer = (MOTION if alias_left else RIGHT).copy()
            left, right = (buffer, RIGHT) if alias_left else (MOTION, buffer)
            assert (
                native.host_curvature_apply(
                    handle, _pointer(left), 6, _pointer(right), 6, _pointer(buffer), 4
                )
                == 0
            )
            np.testing.assert_array_equal(buffer.ravel()[:4], actual)


@pytest.mark.parametrize("cadence", [17, 64])
def test_frozen_curvature_caps_owner_model_and_recovery(
    native: Any, cadence: int
) -> None:
    """Late/final failure never publishes and malformed bindings recover cleanly."""
    with _curvature_owner(native, spins=2, cadence=cadence) as (
        handle,
        density,
        weighted,
        sizes,
        resources,
    ):
        actual = _mixed(native, handle)
        for index in range(4):
            limits = list(resources)
            limits[index] -= 1
            assert (
                native.host_curvature_bind(
                    handle,
                    _pointer(weighted),
                    weighted.size,
                    _method(2),
                    (ct.c_size_t * 4)(*limits),
                    cadence,
                )
                != 0
            )
        for method, invalid_cadence in (
            (b"forged", cadence),
            (_method(1), cadence),
            (_method(2), 0),
            (_method(2), 4097),
        ):
            assert (
                native.host_curvature_bind(
                    handle,
                    _pointer(weighted),
                    weighted.size,
                    method,
                    None,
                    invalid_cadence,
                )
                != 0
            )
        for mode in range(1, 8):
            native.host_integral_model(handle, mode)
            assert (
                native.host_curvature_bind(
                    handle, _pointer(weighted), weighted.size, _method(2), None, cadence
                )
                != 0
            )
            native.host_integral_model(handle, 0)
        for mode, at in (
            (1, 1),
            (1, 4),
            (2, 3),
            (3, 5),
            (4, 3),
            (5, 3),
            (6, 3),
            (7, 0),
        ):
            output = np.full(4, -781.25)
            before = (ct.c_size_t * 8)()
            native.host_curvature_work(handle, before)
            native.host_integral_failure(handle, mode, at)
            assert (
                native.host_curvature_apply(
                    handle, _pointer(MOTION), 6, _pointer(RIGHT), 6, _pointer(output), 4
                )
                != 0
            )
            np.testing.assert_array_equal(output, -781.25)
            after = (ct.c_size_t * 8)()
            native.host_curvature_work(handle, after)
            assert after[1] == before[1]
            if mode == 7:
                assert after[3] - before[3] == sizes[0] ** 4
            native.host_integral_failure(handle, 0, 0)
            np.testing.assert_array_equal(_mixed(native, handle), actual)
        for values in (density, weighted):
            saved = values.copy()
            values[0, 0, 0] += 0.125
            output = np.full(4, -781.25)
            assert (
                native.host_curvature_apply(
                    handle, _pointer(MOTION), 6, _pointer(RIGHT), 6, _pointer(output), 4
                )
                != 0
            )
            np.testing.assert_array_equal(output, -781.25)
            values[:] = saved
            np.testing.assert_array_equal(_mixed(native, handle), actual)
        for count in (0, weighted.size - 1):
            assert (
                native.host_curvature_bind(
                    handle, _pointer(weighted), count, _method(2), None, cadence
                )
                != 0
            )
        for output in (
            _pointer(density),
            _pointer(weighted),
            native.host_integral_packed(handle),
        ):
            assert (
                native.host_curvature_apply(
                    handle, _pointer(MOTION), 6, _pointer(RIGHT), 6, output, 4
                )
                != 0
            )
        np.testing.assert_array_equal(_mixed(native, handle), actual)


@pytest.mark.parametrize("spins", [1, 2])
def test_frozen_curvature_empty_density_channels(native: Any, spins: int) -> None:
    """Zero D still retains W constraint curvature; zero D/W gives exact zeros."""
    density = np.zeros((spins, 4, 4))
    with _curvature_owner(native, spins=spins, density=density) as (
        handle,
        density,
        weighted,
        _,
        _,
    ):
        actual = _mixed(native, handle)
        primal = _oracle(CENTERS, density, 1, True, True, 1, 0, weighted)
        np.testing.assert_allclose(actual[0], primal, atol=3e-12, rtol=3e-11)
        weighted[:] = 0
        assert (
            native.host_curvature_bind(
                handle, _pointer(weighted), weighted.size, _method(spins), None, 64
            )
            == 0
        )
        np.testing.assert_array_equal(_mixed(native, handle), 0)


def test_frozen_curvature_invalid_directions_and_weighted_density(native: Any) -> None:
    """Own entry bindings refuse malformed data transactionally, then recover."""
    with _curvature_owner(native, spins=2) as (handle, _, weighted, _, _):
        actual = _mixed(native, handle)
        for argument, value in ((0, np.nan), (0, np.inf), (1, np.nan), (1, np.inf)):
            legs = [MOTION.copy(), RIGHT.copy()]
            legs[argument].flat[0] = value
            output = np.full(4, -781.25)
            assert (
                native.host_curvature_apply(
                    handle,
                    _pointer(legs[0]),
                    6,
                    _pointer(legs[1]),
                    6,
                    _pointer(output),
                    4,
                )
                != 0
            )
            np.testing.assert_array_equal(output, -781.25)
        for argument in (1, 3, 5):
            output = np.full(4, -781.25)
            pointers = [
                handle,
                _pointer(MOTION),
                6,
                _pointer(RIGHT),
                6,
                _pointer(output),
                4,
            ]
            pointers[argument] = None
            assert native.host_curvature_apply(*pointers) != 0
            np.testing.assert_array_equal(output, -781.25)
        for argument in (2, 4, 6):
            output = np.full(4, -781.25)
            pointers = [
                handle,
                _pointer(MOTION),
                6,
                _pointer(RIGHT),
                6,
                _pointer(output),
                4,
            ]
            pointers[argument] -= 1
            assert native.host_curvature_apply(*pointers) != 0
            np.testing.assert_array_equal(output, -781.25)
        saved = weighted.copy()
        for value in (np.nan, np.inf, saved[0, 0, 1] + 0.25):
            weighted[0, 0, 1] = value
            assert (
                native.host_curvature_bind(
                    handle, _pointer(weighted), weighted.size, _method(2), None, 64
                )
                != 0
            )
            weighted[:] = saved
            np.testing.assert_array_equal(_mixed(native, handle), actual)


def test_frozen_curvature_one_empty_uks_spin(native: Any) -> None:
    """A nonvacuum occupation-one UKS channel has no opposite-spin exchange."""
    density = np.zeros((2, 4, 4))
    density[0] = np.diag([0.3, 0.2, 0.1, 0.05])
    with _curvature_owner(native, spins=2, density=density) as (
        handle,
        density,
        weighted,
        _,
        _,
    ):
        actual = _mixed(native, handle)
        energy = _oracle(CENTERS, density, 1, True, True, 1, 0, weighted)
        np.testing.assert_allclose(actual[0], energy, atol=3e-11, rtol=3e-10)
        step = 1e-3
        corners = [
            _oracle(
                CENTERS + step * (first * MOTION + second * RIGHT),
                density,
                1,
                True,
                True,
                1,
                0,
                weighted,
            )
            for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
        ]
        mixed = (corners[0] - corners[1] - corners[2] + corners[3]) / (4 * step**2)
    np.testing.assert_allclose(actual[3], mixed, atol=3e-8, rtol=2e-6)


@pytest.mark.parametrize("number,core,spins", [(28, 10, 1), (53, 28, 2), (55, 46, 2)])
def test_frozen_curvature_ionic_charge_not_residual_ecp(
    native: Any,
    number: int,
    core: int,
    spins: int,
    tmp_path: Path,
) -> None:
    """Ionic-tail controls are not residual-ECP or complete heavy-atom endpoints."""
    with _curvature_owner(native, spins=spins, number=number, core=core) as (
        handle,
        density,
        weighted,
        _,
        _,
    ):
        actual = _mixed(native, handle)
        primal = _oracle(CENTERS, density, 1, True, True, number, core, weighted)
        np.testing.assert_allclose(actual[0], primal, atol=3e-11, rtol=3e-10)
        steps = np.array([1e-2, 3e-3, 1e-3])
        corners = np.array(
            [
                [
                    _oracle(
                        CENTERS + step * (first * MOTION + second * RIGHT),
                        density,
                        1,
                        True,
                        True,
                        number,
                        core,
                        weighted,
                    )
                    for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                ]
                for step in steps
            ]
        )
        expected = np.column_stack(
            [
                (corners[:, 0] + corners[:, 1] - corners[:, 2] - corners[:, 3])
                / (4 * steps),
                (corners[:, 0] - corners[:, 1] + corners[:, 2] - corners[:, 3])
                / (4 * steps),
                (corners[:, 0] - corners[:, 1] - corners[:, 2] + corners[:, 3])
                / (4 * steps**2),
            ]
        )
        errors = np.abs(expected - actual[None, 1:])
        np.testing.assert_allclose(actual[1:], expected[-1], atol=3e-8, rtol=2e-6)
        assert errors[-1].max() < max(errors[0].max() / 8, 5e-9), errors
        np.savez_compressed(
            tmp_path / "ionic-tail-curvature.npz",
            native=actual,
            density=density,
            weighted_density=weighted,
            centers=CENTERS,
            left=MOTION,
            right=RIGHT,
            steps=steps,
            primal_baseline=primal,
            primal_corners=corners,
            errors=errors,
            number=number,
            core=core,
        )


@contextmanager
def _curvature_owner(native: Any, **options: Any) -> Any:
    """Bind actual normalized AO owners and authenticated immutable D/W panels."""
    with _owner(native, **options) as (handle, density, sizes):
        weighted_density = np.random.default_rng(438).normal(size=density.shape) * 0.07
        weighted_density += weighted_density.transpose(0, 2, 1).copy()
        weighted_density = np.ascontiguousarray(weighted_density)
        assert (
            native.host_curvature_bind(
                handle,
                _pointer(weighted_density),
                weighted_density.size,
                _method(len(density)),
                None,
                options.get("cadence", 64),
            )
            == 0
        ), native.host_integral_error().decode()
        resources = (ct.c_size_t * 4)()
        native.host_curvature_resources(handle, resources)
        yield handle, density, weighted_density, sizes, tuple(resources)


def _mixed(
    native: Any, handle: Any, left: np.ndarray = MOTION, right: np.ndarray = RIGHT
) -> np.ndarray:
    """Obtain one frozen source transaction without invoking SCF or response."""
    output = np.full(4, -781.25)
    assert (
        native.host_curvature_apply(
            handle,
            _pointer(left),
            left.size,
            _pointer(right),
            right.size,
            _pointer(output),
            output.size,
        )
        == 0
    ), native.host_integral_error().decode()
    return output


def _oracle(
    centers: np.ndarray,
    density: np.ndarray,
    angular: int,
    spherical: bool,
    contracted: bool,
    number: int,
    core: int,
    weighted_density: np.ndarray | None = None,
) -> Any:
    """Independent normalized libcint S/T/V_eff/ERI values, no native AD."""
    gto = pytest.importorskip("pyscf.gto")
    from pyscf.data.elements import ELEMENTS

    first, second = ELEMENTS[number] + "0", "H1"
    first_primitives = [[0.8 + 0.1 * angular, 1]]
    second_primitives = [[0.6, 1]]
    if contracted:
        first_primitives = [[0.8 + 0.1 * angular, 0.7], [0.23 + 0.02 * angular, 0.3]]
        second_primitives = [[0.6, 0.7], [0.18, 0.3]]
    molecule = gto.M(
        atom=[(first, centers[0]), (second, centers[1])],
        basis={
            first: [[angular, *first_primitives]],
            second: [[0, *second_primitives]],
        },
        cart=not spherical,
        unit="Bohr",
        spin=(number + 1) % 2,
        verbose=0,
    )
    overlap = molecule.intor("int1e_ovlp")
    normalization = 1 / np.sqrt(np.diag(overlap))
    scale = normalization[:, None] * normalization[None, :]
    hcore = molecule.intor("int1e_kin")
    for atom, charge in enumerate((number - core, 1)):
        with molecule.with_rinv_origin(centers[atom]):
            hcore -= charge * molecule.intor("int1e_rinv")
    overlap = overlap * scale
    hcore *= scale
    eri = molecule.intor("int2e", aosym="s1")
    eri *= (
        normalization[:, None, None, None]
        * normalization[None, :, None, None]
        * normalization[None, None, :, None]
        * normalization[None, None, None, :]
    )
    coulomb = np.einsum("ijkl,kl->ij", eri, density.sum(axis=0))
    exchange = np.einsum("ikjl,skl->sij", eri, density)
    fock = hcore[None] + coulomb[None] - (0.1 if len(density) == 1 else 0.2) * exchange
    if weighted_density is None:
        return overlap, fock
    return float(
        np.einsum("sij,ij->", density, hcore)
        + 0.5 * np.einsum("sij,sij->", density, fock - hcore[None])
        - np.einsum("sij,ij->", weighted_density, overlap)
    )


def _matrix_bind(native: Any, handle: Any, weighted: np.ndarray) -> tuple[int, ...]:
    """Reserve the frozen nuclear matrix, not the density-response contribution."""
    assert (
        native.host_matrix_bind(
            handle, _pointer(weighted), weighted.size, _method(len(weighted)), None, 64
        )
        == 0
    ), native.host_integral_error().decode()
    resources = (ct.c_size_t * 4)()
    native.host_curvature_resources(handle, resources)
    return tuple(resources)


def _matrix(native: Any, handle: Any, dimension: int) -> np.ndarray:
    output = np.full(1 + dimension + dimension**2, -781.25)
    assert native.host_matrix_apply(handle, _pointer(output), output.size) == 0, (
        native.host_integral_error().decode()
    )
    return output


def _independent_matrix_primitive_budget(native: Any, handle: Any) -> int:
    """Enumerate a small public AO recipe, independently of the grouped planner."""
    sizes = (ct.c_size_t * 7)()
    native.host_integral_sizes(handle, sizes)
    basis, atoms, primitives = sizes[0], sizes[1], sizes[6]
    packed = np.ctypeslib.as_array(
        native.host_integral_packed(handle),
        shape=(3 * atoms + 2 * primitives + 16 * basis,),
    )
    records = packed[3 * atoms + 2 * primitives :].reshape(basis, 16)
    pairs = list(itertools.combinations_with_replacement(range(basis), 2))
    work = 0
    for first_index, first in enumerate(pairs):
        for second in pairs[: first_index + 1]:
            quartet = records[list(first + second)]
            owners = len(set(quartet[:, 0].astype(int)))
            radial_cartesian = int(np.prod(quartet[:, 2] * quartet[:, 3]))
            work += radial_cartesian * (1 if owners == 1 else 3 * (owners - 1))
    return work


def _projected_matrix_bind(
    native: Any, handle: Any, weighted: np.ndarray
) -> tuple[int, ...]:
    """Select the explicitly budgeted polynomial-density representation."""
    assert (
        native.host_projected_matrix_bind(
            handle,
            _pointer(weighted),
            weighted.size,
            _method(weighted.shape[0]),
            None,
            64,
        )
        == 0
    ), native.host_integral_error().decode()
    sizes = (ct.c_size_t * 4)()
    native.host_curvature_resources(handle, sizes)
    return tuple(sizes)


@pytest.mark.parametrize("angular", [0, 1, 2, 3])
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("spins", [1, 2])
def test_projected_matrix_matches_public_and_independent_primal(
    native: Any, angular: int, spherical: bool, spins: int, tmp_path: Path
) -> None:
    """A density congruence must preserve the full frozen scalar/gradient/raw H."""
    contracted = angular <= 1
    with _curvature_owner(
        native, angular=angular, spherical=spherical, spins=spins, contracted=contracted
    ) as (handle, density, weighted, _, _):
        _matrix_bind(native, handle, weighted)
        expected = _matrix(native, handle, 6)
        baseline_work = (ct.c_size_t * 8)()
        native.host_curvature_work(handle, baseline_work)
        assert native.host_projected_matrix_supported(handle) == 1
        resources = _projected_matrix_bind(native, handle, weighted)
        actual = _matrix(native, handle, 6)
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
        hessian = actual[7:].reshape(6, 6)
        np.testing.assert_allclose(hessian, hessian.T, atol=3e-12, rtol=3e-11)
        np.testing.assert_allclose(
            hessian.reshape(2, 3, 2, 3).sum(axis=2), 0, atol=3e-12
        )
        steps = np.array([3e-3, 1e-3, 3e-4])
        expected_actions = []
        for step in steps:
            values = np.array(
                [
                    [
                        _oracle(
                            CENTERS + step * (first * left + second * RIGHT),
                            density,
                            angular,
                            spherical,
                            contracted,
                            1,
                            0,
                            weighted,
                        )
                        for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                    ]
                    for left in np.eye(6).reshape(6, 2, 3)
                ]
            )
            expected_actions.append(
                (values[:, 0] - values[:, 1] - values[:, 2] + values[:, 3])
                / (4 * step**2)
            )
        errors = np.abs(np.asarray(expected_actions) - hessian @ RIGHT.ravel())
        np.testing.assert_allclose(
            expected_actions[-1], hessian @ RIGHT.ravel(), atol=3e-8, rtol=2e-6
        )
        assert errors[-1].max() < max(errors[0].max() / 8, 5e-9)
        work = (ct.c_size_t * 8)()
        native.host_curvature_work(handle, work)
        assert 0 < work[5] <= resources[2] and work[7] <= resources[3]
        if spherical and angular >= 2:
            assert work[5] < baseline_work[5]
        projection_terms = native.host_curvature_density_projection_terms(handle)
        assert projection_terms > 0
        np.testing.assert_array_equal(_matrix(native, handle, 6), actual)
        assert (
            native.host_curvature_density_projection_terms(handle) == projection_terms
        )
        np.savez_compressed(
            tmp_path / "projected-frozen-matrix.npz",
            matrix=actual,
            public_reference=expected,
            independent_actions=np.asarray(expected_actions),
            errors=errors,
            resources=np.array(resources),
            work=np.array(work),
            public_work=np.array(baseline_work),
        )


@pytest.mark.parametrize("spins", [1, 2])
def test_projected_matrix_caps_and_unsafe_scaling_keep_public_source(
    native: Any, spins: int
) -> None:
    """Extra owned snapshots have real caps; finite unsafe D still has a fallback."""
    with _curvature_owner(native, angular=2, spins=spins, contracted=False) as (
        handle,
        density,
        weighted,
        _,
        _,
    ):
        resources = _projected_matrix_bind(native, handle, weighted)
        expected = _matrix(native, handle, 6)
        for field in range(4):
            limits = (ct.c_size_t * 4)(*resources)
            limits[field] -= 1
            assert (
                native.host_projected_matrix_bind(
                    handle,
                    _pointer(weighted),
                    weighted.size,
                    _method(spins),
                    limits,
                    64,
                )
                != 0
            )
            np.testing.assert_array_equal(_matrix(native, handle, 6), expected)
        output = np.full_like(expected, -781.25)
        original_density = density.copy()
        density[0, 0, 0] += 0.125
        assert native.host_projected_matrix_supported(handle) == 1
        assert native.host_matrix_apply(handle, _pointer(output), output.size) != 0
        np.testing.assert_array_equal(output, -781.25)
        density[:] = original_density
        np.testing.assert_array_equal(_matrix(native, handle, 6), expected)
        sizes = (ct.c_size_t * 7)()
        native.host_integral_sizes(handle, sizes)
        packed = np.ctypeslib.as_array(
            native.host_integral_packed(handle),
            shape=(3 * sizes[1] + 2 * sizes[6] + 16 * sizes[0],),
        )
        coefficient = 3 * sizes[1] + 2 * sizes[6] + 7
        original_coefficient = packed[coefficient]
        packed[coefficient] *= 1.125
        assert native.host_projected_matrix_supported(handle) == 1
        assert native.host_matrix_apply(handle, _pointer(output), output.size) != 0
        np.testing.assert_array_equal(output, -781.25)
        packed[coefficient] = original_coefficient
        np.testing.assert_array_equal(_matrix(native, handle, 6), expected)
        density[...] *= 1e60
        assert native.host_projected_matrix_supported(handle) == 0
        assert (
            native.host_projected_matrix_bind(
                handle, _pointer(weighted), weighted.size, _method(spins), None, 64
            )
            != 0
        )
        _matrix_bind(native, handle, weighted)


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("spins", [1, 2])
def test_projected_matrix_mixed_shells_and_physical_owners(
    native: Any, spherical: bool, spins: int, tmp_path: Path
) -> None:
    """Larger mixed contractions include distinct shells sharing one center."""
    gto = pytest.importorskip("pyscf.gto")
    centers = np.array([[0.13, -0.21, 0.34], [1.7, 0.28, -0.42], [-0.8, 1.5, 0.19]])
    dimension = centers.size
    count = 14 if spherical else 16
    rng = np.random.default_rng(24184)
    density = rng.normal(size=(spins, count, count)) * 0.025
    density += density.transpose(0, 2, 1).copy()
    weighted = rng.normal(size=density.shape) * 0.01
    weighted += weighted.transpose(0, 2, 1).copy()
    direction = rng.normal(size=centers.shape) * 0.1
    handle = native.host_integral_create_mixed_sites(
        _pointer(centers), spherical, spins, _pointer(density), _method(spins)
    )
    assert handle, native.host_integral_error().decode()
    try:
        _matrix_bind(native, handle, weighted)
        expected = _matrix(native, handle, dimension)
        before = (ct.c_size_t * 8)()
        native.host_curvature_work(handle, before)
        resources = _projected_matrix_bind(native, handle, weighted)
        actual = _matrix(native, handle, dimension)
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
        matrix = actual[1 + dimension :].reshape(dimension, dimension)
        np.testing.assert_allclose(matrix, matrix.T, atol=3e-12, rtol=3e-11)
        np.testing.assert_allclose(matrix @ np.tile(np.eye(3), (3, 1)), 0, atol=3e-12)

        def primal(positions: np.ndarray) -> float:
            molecule = gto.M(
                atom=list(zip(("H0", "H1", "H2"), positions, strict=True)),
                basis={
                    "H0": [[0, [0.5, 1]], [2, [0.8, 0.7], [0.23, 0.3]]],
                    "H1": [[1, [0.7, 1]]],
                    "H2": [[2, [0.6, 1]]],
                },
                cart=not spherical,
                unit="Bohr",
                spin=1,
                verbose=0,
            )
            overlap = molecule.intor("int1e_ovlp")
            normalization = 1 / np.sqrt(np.diag(overlap))
            scale = normalization[:, None] * normalization[None, :]
            hcore = (molecule.intor("int1e_kin") + molecule.intor("int1e_nuc")) * scale
            eri = molecule.intor("int2e", aosym="s1")
            eri *= (
                normalization[:, None, None, None]
                * normalization[None, :, None, None]
                * normalization[None, None, :, None]
                * normalization[None, None, None, :]
            )
            total = density.sum(axis=0)
            return float(
                np.einsum("ij,ij->", total, hcore)
                - np.einsum("sij,ij->", weighted, overlap * scale)
                + 0.5 * np.einsum("ij,kl,ijkl->", total, total, eri)
                - (0.05 if spins == 1 else 0.1)
                * np.einsum("sik,sjl,ijkl->", density, density, eri)
            )

        actions = []
        for step in (3e-3, 1e-3, 3e-4):
            corners = np.array(
                [
                    [
                        primal(centers + step * (first * left + second * direction))
                        for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                    ]
                    for left in np.eye(dimension).reshape(dimension, 3, 3)
                ]
            )
            actions.append(
                (corners[:, 0] - corners[:, 1] - corners[:, 2] + corners[:, 3])
                / (4 * step**2)
            )
        np.testing.assert_allclose(
            actions[-1], matrix @ direction.ravel(), atol=3e-8, rtol=2e-6
        )
        after = (ct.c_size_t * 8)()
        native.host_curvature_work(handle, after)
        assert 0 < after[5] <= resources[2] and after[7] <= resources[3]
        if spherical:
            assert after[5] < before[5]
        np.savez_compressed(
            tmp_path / "mixed-projected-matrix.npz",
            matrix=actual,
            public_reference=expected,
            independent_actions=np.asarray(actions),
            work=np.array(after),
            public_work=np.array(before),
            resources=np.array(resources),
        )
    finally:
        native.host_integral_destroy(handle)


@pytest.mark.parametrize("angular", [0, 1, 2, 3])
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("spins", [1, 2])
def test_matrix_primitive_budget_admits_physical_owner_envelope(
    native: Any, angular: int, spherical: bool, spins: int
) -> None:
    """A finite cap admits real work, not a fictitious four-owner worst case."""
    with _curvature_owner(
        native, angular=angular, spherical=spherical, spins=spins, contracted=True
    ) as (handle, _, weighted, _, _):
        resources = _matrix_bind(native, handle, weighted)
        budget = _independent_matrix_primitive_budget(native, handle)
        limits = (ct.c_size_t * 4)(resources[0], resources[1], budget, resources[3])
        assert (
            native.host_matrix_bind(
                handle, _pointer(weighted), weighted.size, _method(spins), limits, 64
            )
            == 0
        ), native.host_integral_error().decode()
        limits[2] = resources[2] - 1
        assert (
            native.host_matrix_bind(
                handle, _pointer(weighted), weighted.size, _method(spins), limits, 64
            )
            != 0
        )


@pytest.mark.parametrize("atoms", [1, 3, 5, 16, 32])
@pytest.mark.parametrize("coincident", [False, True])
def test_matrix_primitive_budget_keeps_distinct_site_owners(
    native: Any, atoms: int, coincident: bool
) -> None:
    """Physical ownership, not numeric separation or a 3N bound, prices the cap."""
    centers = np.zeros((atoms, 3))
    if not coincident:
        centers[:, 0] = np.arange(atoms) * 1.4
    density = np.eye(atoms)[None] * 0.03
    weighted = density * 0.3
    handle = native.host_integral_create_sites(
        _pointer(centers), atoms, 1, _pointer(density), _method(1), 64
    )
    assert handle, native.host_integral_error().decode()
    try:
        resources = _matrix_bind(native, handle, weighted)
        budget = _independent_matrix_primitive_budget(native, handle)
        limits = (ct.c_size_t * 4)(resources[0], resources[1], budget, resources[3])
        assert (
            native.host_matrix_bind(
                handle, _pointer(weighted), weighted.size, _method(1), limits, 64
            )
            == 0
        ), native.host_integral_error().decode()
        limits[2] = resources[2] - 1
        assert (
            native.host_matrix_bind(
                handle, _pointer(weighted), weighted.size, _method(1), limits, 64
            )
            != 0
        )
    finally:
        native.host_integral_destroy(handle)


@pytest.mark.parametrize(
    ("angular", "spherical", "spins", "number", "core"),
    [
        (angular, spherical, spins, 1, 0)
        for angular in range(4)
        for spherical in (False, True)
        for spins in (1, 2)
    ]
    + [(1, True, 2, 28, 10), (1, True, 2, 53, 28), (1, True, 2, 55, 46)],
)
def test_local_curvature_matrix_against_stream_and_independent_primal(
    native: Any,
    angular: int,
    spherical: bool,
    spins: int,
    number: int,
    core: int,
    tmp_path: Path,
) -> None:
    """Raw frozen H from one weight sweep, with independent physical-center gates."""
    contracted = angular <= 1
    with _curvature_owner(
        native,
        angular=angular,
        spherical=spherical,
        spins=spins,
        contracted=contracted,
        number=number,
        core=core,
    ) as (handle, density, weighted, sizes, _):
        _bind_column(native, handle, weighted)
        columns = [
            _column(native, handle, direction)
            for direction in np.eye(6).reshape(6, 2, 3)
        ]
        resources = _matrix_bind(native, handle, weighted)
        actual = _matrix(native, handle, 6)
        hessian = actual[7:].reshape(6, 6)
        np.testing.assert_allclose(actual[0], columns[0][0], atol=3e-12, rtol=3e-11)
        np.testing.assert_allclose(actual[1:7], columns[0][2:8], atol=3e-12, rtol=3e-11)
        np.testing.assert_allclose(
            hessian,
            np.column_stack([item[8:] for item in columns]),
            atol=3e-12,
            rtol=3e-11,
        )
        np.testing.assert_allclose(hessian, hessian.T, atol=3e-12, rtol=3e-11)
        np.testing.assert_allclose(
            hessian.reshape(2, 3, 2, 3).sum(axis=2), 0, atol=3e-12
        )
        steps = np.array([3e-3, 1e-3, 3e-4])
        corners = np.array(
            [
                [
                    [
                        _oracle(
                            CENTERS + step * (first * leg + second * RIGHT),
                            density,
                            angular,
                            spherical,
                            contracted,
                            number,
                            core,
                            weighted,
                        )
                        for first, second in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                    ]
                    for leg in np.eye(6).reshape(6, 2, 3)
                ]
                for step in steps
            ]
        )
        independent_gradient = (
            corners[:, :, 0] + corners[:, :, 1] - corners[:, :, 2] - corners[:, :, 3]
        ) / (4 * steps[:, None])
        independent_action = (
            corners[:, :, 0] - corners[:, :, 1] - corners[:, :, 2] + corners[:, :, 3]
        ) / (4 * steps[:, None] ** 2)
        expected = np.column_stack((actual[1:7], hessian @ RIGHT.ravel())).T
        references = np.stack((independent_gradient, independent_action), axis=1)
        errors = np.abs(references - expected)
        np.testing.assert_allclose(references[-1], expected, atol=3e-8, rtol=2e-6)
        assert errors[-1].max() < max(errors[0].max() / 8, 5e-9)
        work, orders = (ct.c_size_t * 8)(), (ct.c_size_t * 3)()
        native.host_curvature_work(handle, work)
        native.host_curvature_eri_orders(handle, orders)
        assert work[0] == work[1] == 1
        assert work[3] == sizes[0] ** 4
        assert work[4] == resources[1] and work[5] <= resources[2]
        assert work[7] == resources[3] and sum(orders) == work[5]
        assert orders[0] > 0 and orders[1] == 0 and orders[2] > 0
        matrix_fields = (ct.c_size_t * 3)()
        native.host_curvature_matrix_fields(handle, matrix_fields)
        calls, coordinates, partials = matrix_fields
        assert 0 < calls <= orders[2]
        assert 3 * calls <= coordinates <= 9 * calls
        assert 3 * coordinates <= partials <= 81 * calls
        np.savez_compressed(
            tmp_path / "local-frozen-matrix.npz",
            matrix=actual,
            density=density,
            weighted_density=weighted,
            steps=steps,
            corners=corners,
            errors=errors,
            work=np.array(work),
            resources=np.array(resources),
        )


def test_local_curvature_matrix_caps_alias_owner_and_recovery(native: Any) -> None:
    """Admission and failed matrix sweeps preserve output and remain recoverable."""
    with _curvature_owner(native, spins=2) as (handle, density, weighted, _, _):
        resources = _matrix_bind(native, handle, weighted)
        expected = _matrix(native, handle, 6)
        for index in range(4):
            limits = list(resources)
            limits[index] -= 1
            assert (
                native.host_matrix_bind(
                    handle,
                    _pointer(weighted),
                    weighted.size,
                    _method(2),
                    (ct.c_size_t * 4)(*limits),
                    64,
                )
                != 0
            )
        for mode, at in (
            (1, 1),
            (1, 4),
            (2, 3),
            (3, 5),
            (4, 3),
            (5, 3),
            (6, 3),
            (7, 0),
        ):
            output = np.full_like(expected, -781.25)
            native.host_integral_failure(handle, mode, at)
            assert native.host_matrix_apply(handle, _pointer(output), output.size) != 0
            np.testing.assert_array_equal(output, -781.25)
            native.host_integral_failure(handle, 0, 0)
            np.testing.assert_array_equal(_matrix(native, handle, 6), expected)
        output = np.full_like(expected, -781.25)
        assert native.host_matrix_apply(handle, _pointer(output), output.size - 1) != 0
        np.testing.assert_array_equal(output, -781.25)
        original = weighted.copy()
        weighted[0, 0, 0] += 0.125
        assert native.host_matrix_apply(handle, _pointer(output), output.size) != 0
        np.testing.assert_array_equal(output, -781.25)
        weighted[:] = original
        np.testing.assert_array_equal(_matrix(native, handle, 6), expected)
        assert native.host_matrix_apply(handle, _pointer(density), output.size) != 0
        assert native.host_matrix_apply(handle, _pointer(weighted), output.size) != 0


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("angular", [0, 1, 2, 3])
@pytest.mark.parametrize("spherical", [False, True])
def test_local_matrix_plus_coordinate_density_response_matches_full_column(
    native: Any, spins: int, angular: int, spherical: bool
) -> None:
    """The full D'/W' term is not discarded when freezing the primitive sweep."""
    with _curvature_owner(
        native,
        spins=spins,
        angular=angular,
        spherical=spherical,
        contracted=angular <= 1,
    ) as (handle, density, weighted, _, _):
        rng = np.random.default_rng(1314)
        moving = rng.normal(size=density.shape) * 0.02
        moving += moving.transpose(0, 2, 1).copy()
        weighted_moving = rng.normal(size=density.shape) * 0.01
        weighted_moving += weighted_moving.transpose(0, 2, 1).copy()
        _bind_column(native, handle, weighted)
        status, expected = _response_column(
            native, handle, RIGHT, moving, weighted_moving
        )
        assert status == 0, native.host_integral_error().decode()
        _matrix_bind(native, handle, weighted)
        frozen = _matrix(native, handle, 6)
        _coordinate_sources(native, handle, spins)
        terms = native.host_integral_density_response_terms(handle)
        output = np.full(6, -781.25)
        assert (
            native.host_integral_density_response(
                handle,
                _pointer(moving),
                moving.size,
                _pointer(weighted_moving),
                weighted_moving.size,
                _pointer(output),
                output.size,
                terms - 1,
            )
            != 0
        )
        np.testing.assert_array_equal(output, -781.25)
        before, after = (ct.c_size_t * 8)(), (ct.c_size_t * 8)()
        native.host_integral_work(handle, before)
        assert (
            native.host_integral_density_response(
                handle,
                _pointer(moving),
                moving.size,
                _pointer(weighted_moving),
                weighted_moving.size,
                _pointer(output),
                output.size,
                terms,
            )
            == 0
        ), native.host_integral_error().decode()
        native.host_integral_work(handle, after)
        assert tuple(before) == tuple(after)
        np.testing.assert_allclose(
            frozen[7:].reshape(6, 6) @ RIGHT.ravel() + output,
            expected[8:],
            atol=3e-12,
            rtol=3e-11,
        )
        actual_action = frozen[7:].reshape(6, 6) @ RIGHT.ravel() + output
        errors = []
        for step in (3e-3, 1e-3, 3e-4):
            corners = np.array(
                [
                    [
                        _oracle(
                            CENTERS + step * (left_sign * leg + right_sign * RIGHT),
                            density + right_sign * step * moving,
                            angular,
                            spherical,
                            angular <= 1,
                            1,
                            0,
                            weighted + right_sign * step * weighted_moving,
                        )
                        for left_sign, right_sign in (
                            (1, 1),
                            (1, -1),
                            (-1, 1),
                            (-1, -1),
                        )
                    ]
                    for leg in np.eye(6).reshape(6, 2, 3)
                ]
            )
            reference = (
                corners[:, 0] - corners[:, 1] - corners[:, 2] + corners[:, 3]
            ) / (4 * step**2)
            errors.append(float(np.max(np.abs(reference - actual_action))))
        np.testing.assert_allclose(reference, actual_action, atol=3e-8, rtol=2e-6)
        assert errors[-1] < max(errors[0] / 8, 5e-9)
        alias = np.empty(max(moving.size, 6))
        alias[: moving.size] = moving.ravel()
        assert (
            native.host_integral_density_response(
                handle,
                _pointer(alias),
                moving.size,
                _pointer(weighted_moving),
                weighted_moving.size,
                _pointer(alias),
                6,
                terms,
            )
            == 0
        )
        np.testing.assert_array_equal(alias.ravel()[:6], output)
        moving[0, 0, 0] = np.nan
        output.fill(-781.25)
        assert (
            native.host_integral_density_response(
                handle,
                _pointer(moving),
                moving.size,
                _pointer(weighted_moving),
                weighted_moving.size,
                _pointer(output),
                output.size,
                terms,
            )
            != 0
        )
        np.testing.assert_array_equal(output, -781.25)


@pytest.mark.parametrize("atoms", [3, 8, 32])
def test_local_curvature_matrix_many_owners_and_bounded_work(
    native: Any, atoms: int, tmp_path: Path
) -> None:
    """Local right work is bounded by nine, not 3N; this is source-only evidence."""
    centers = np.column_stack(
        (
            np.arange(atoms) * 1.5,
            0.19 * np.sin(np.arange(atoms)),
            0.13 * np.cos(np.arange(atoms)),
        )
    )
    rng = np.random.default_rng(1955)
    density = rng.normal(size=(2, atoms, atoms)) * 0.02
    density += density.transpose(0, 2, 1).copy()
    weighted = density * 0.3
    dense = rng.normal(size=centers.shape) * 0.1
    sparse = dense.copy()
    sparse[1:] = 0
    handle = native.host_integral_create_sites(
        _pointer(centers), atoms, 2, _pointer(density), _method(2), 64
    )
    assert handle, native.host_integral_error().decode()
    try:
        _bind_column(native, handle, weighted)
        expected = [_column(native, handle, right) for right in (dense, sparse)]
        resources = _matrix_bind(native, handle, weighted)
        started = time.perf_counter()
        actual = _matrix(native, handle, centers.size)
        seconds = time.perf_counter() - started
        for right, reference in zip((dense, sparse), expected, strict=True):
            np.testing.assert_allclose(
                actual[1 : 1 + centers.size],
                reference[2 : 2 + centers.size],
                atol=3e-12,
                rtol=3e-11,
            )
            np.testing.assert_allclose(
                actual[1 + centers.size :].reshape(centers.size, -1) @ right.ravel(),
                reference[2 + centers.size :],
                atol=3e-12,
                rtol=3e-11,
            )
        work = (ct.c_size_t * 8)()
        native.host_curvature_work(handle, work)
        assert work[0] == work[1] == 1
        assert (
            work[3] == atoms**4 and work[5] <= resources[2] and work[7] == resources[3]
        )
        np.savez_compressed(
            tmp_path / "local-many-owner-matrix.npz",
            matrix=actual,
            resources=np.array(resources),
            work=np.array(work),
            seconds=seconds,
        )
    finally:
        native.host_integral_destroy(handle)


@pytest.mark.parametrize("coincident", [False, True])
def test_local_curvature_matrix_ordered_comparison_keeps_physical_owners(
    native: Any, coincident: bool
) -> None:
    centers = CENTERS.copy()
    if coincident:
        centers[1] = centers[0]
    results = []
    for schedule in (0, 1):
        with _curvature_owner(
            native, angular=1, spins=2, centers=centers, eri_schedule=schedule
        ) as (handle, _, weighted, _, _):
            _matrix_bind(native, handle, weighted)
            results.append(_matrix(native, handle, 6))
    np.testing.assert_allclose(results[0], results[1], atol=3e-12, rtol=3e-11)
    if coincident:
        assert np.max(np.abs(results[1][7:])) > 1e-5


def _coordinate_sources(native: Any, handle: Any, spins: int) -> tuple[int, ...]:
    """Explicitly reserve, then prepare the optional resident first-source panel."""
    assert native.host_integral_coordinate_mode(handle, 1, _method(spins), None) == 0, (
        native.host_integral_error().decode()
    )
    sizes = (ct.c_size_t * 7)()
    native.host_integral_sizes(handle, sizes)
    assert native.host_integral_prepare_coordinates(handle) == 0, (
        native.host_integral_error().decode()
    )
    return tuple(sizes)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("spherical", [True, False])
@pytest.mark.parametrize("angular", [0, 1, 2, 3])
def test_coordinate_first_sources_against_stream_and_independent_values(
    native: Any, angular: int, spherical: bool, spins: int
) -> None:
    """All Cartesian sources and arbitrary replays retain independent primal gates."""
    contracted = angular <= 1
    directions = list(np.eye(6).reshape(6, 2, 3)) + [
        MOTION,
        np.tile([0.2, -0.3, 0.1], (2, 1)),
        MOTION * 1e-14,
    ]
    with _owner(native, angular, spins, spherical, contracted) as (
        handle,
        density,
        sizes,
    ):
        streamed = [_apply(native, handle, density, right) for right in directions]
        panel = _coordinate_sources(native, handle, spins)
        assert panel[2] > sizes[2] + 6 * (spins + 1) * sizes[0] ** 2 * 8
        before = (ct.c_size_t * 8)()
        native.host_integral_work(handle, before)
        assert before[0] == before[1] == 1
        assert before[7] == panel[5]
        for right, expected in zip(directions, streamed, strict=True):
            actual = _apply(native, handle, density, right)
            for result, reference in zip(actual, expected, strict=True):
                np.testing.assert_allclose(result, reference, atol=3e-13, rtol=2e-11)
        after, resident = (ct.c_size_t * 8)(), (ct.c_size_t * 2)()
        native.host_integral_work(handle, after)
        native.host_integral_resident_work(handle, resident)
        assert tuple(after) == tuple(before)
        assert tuple(resident) == (1, len(directions))
        actual = _apply(native, handle, density)
        errors = []
        for step in (1e-3, 2e-4, 4e-5):
            positive = _oracle(
                CENTERS + step * MOTION, density, angular, spherical, contracted, 1, 0
            )
            negative = _oracle(
                CENTERS - step * MOTION, density, angular, spherical, contracted, 1, 0
            )
            numerical = [
                (first - second) / (2 * step)
                for first, second in zip(positive, negative, strict=True)
            ]
            errors.append(
                max(
                    float(np.max(np.abs(first - second)))
                    for first, second in zip(actual, numerical, strict=True)
                )
            )
        for result, reference in zip(actual, numerical, strict=True):
            np.testing.assert_allclose(result, reference, atol=2e-8, rtol=3e-7)
        assert errors[-1] < errors[0] / 30
        for result, reference in zip(
            _apply(native, handle, density, MOTION * 1e-14), actual, strict=True
        ):
            np.testing.assert_allclose(
                result / 1e-14, reference, atol=3e-13, rtol=2e-11
            )


def test_coordinate_first_sources_caps_staleness_alias_and_recovery(
    native: Any,
) -> None:
    """A failed/stale bank never publishes, and original source remains recoverable."""
    with _owner(native, spins=2) as (handle, density, sizes):
        expected = _apply(native, handle, density)
        panel = _coordinate_sources(native, handle, 2)
        for index in range(4):
            limits = list(panel[2:6])
            limits[index] -= 1
            assert (
                native.host_integral_coordinate_mode(
                    handle, 1, _method(2), (ct.c_size_t * 4)(*limits)
                )
                != 0
            )
            for result, reference in zip(
                _apply(native, handle, density), expected, strict=True
            ):
                np.testing.assert_allclose(result, reference, atol=2e-13)
        assert native.host_integral_coordinate_mode(handle, 1, _method(2), None) == 0
        with pytest.raises(RuntimeError, match="unprepared"):
            _apply(native, handle, density)
        assert native.host_integral_prepare_coordinates(handle) == 0
        original = density.copy()
        density[0, 0, 0] += 0.125
        with pytest.raises(RuntimeError, match="changed"):
            _apply(native, handle, density)
        density[:] = original
        packed = native.host_integral_packed(handle)
        changed_index = 3 * sizes[1] + 2 * sizes[6] + 7
        value = packed[changed_index]
        packed[changed_index] += 0.125
        with pytest.raises(RuntimeError, match="changed"):
            _apply(native, handle, density)
        packed[changed_index] = value
        native.host_integral_charge(handle, 2)
        with pytest.raises(RuntimeError, match="charges changed"):
            _apply(native, handle, density)
        native.host_integral_charge(handle, 1)
        overlap, fock = (
            np.full(density.shape[1:], -781.25),
            np.full_like(density, -781.25),
        )
        overlap.ravel()[:6] = MOTION.ravel()
        assert (
            native.host_integral_apply(
                handle,
                _pointer(overlap),
                6,
                _pointer(overlap),
                overlap.size,
                _pointer(fock),
                fock.size,
            )
            == 0
        )
        for result, reference in zip((overlap, fock), expected, strict=True):
            np.testing.assert_allclose(result, reference, atol=2e-13)
        for mode, at in ((1, 4), (2, 3), (3, 5), (4, 3), (5, 3), (6, 3), (7, 0)):
            native.host_integral_failure(handle, mode, at)
            assert native.host_integral_prepare_coordinates(handle) != 0
            native.host_integral_failure(handle, 0, 0)
            with pytest.raises(RuntimeError, match="unprepared"):
                _apply(native, handle, density)
            assert native.host_integral_prepare_coordinates(handle) == 0
        work = (ct.c_size_t * 8)()
        native.host_integral_work(handle, work)
        assert work[0] > work[1] > 1
        assert native.host_integral_coordinate_mode(handle, 0, _method(2), None) == 0
        for result, reference in zip(
            _apply(native, handle, density), expected, strict=True
        ):
            np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("atoms", [3, 8, 32])
def test_coordinate_first_sources_dense_sparse_work(native: Any, atoms: int) -> None:
    """Resident replay at 32 sites is not a complete-electronic scale qualification."""
    centers = np.column_stack(
        (
            np.arange(atoms) * 1.5,
            0.19 * np.sin(np.arange(atoms)),
            0.13 * np.cos(np.arange(atoms)),
        )
    )
    rng = np.random.default_rng(1711)
    density = rng.normal(size=(2, atoms, atoms)) * 0.02
    density += density.transpose(0, 2, 1).copy()
    dense = rng.normal(size=centers.shape) * 0.1
    sparse = dense.copy()
    sparse[1:] = 0
    handle = native.host_integral_create_sites(
        _pointer(centers), atoms, 2, _pointer(density), _method(2), 64
    )
    assert handle, native.host_integral_error().decode()
    try:
        expected = [_apply(native, handle, density, right) for right in (dense, sparse)]
        _coordinate_sources(native, handle, 2)
        before, after = (ct.c_size_t * 8)(), (ct.c_size_t * 8)()
        native.host_integral_work(handle, before)
        for right, reference in zip((dense, sparse), expected, strict=True):
            for result, baseline in zip(
                _apply(native, handle, density, right), reference, strict=True
            ):
                np.testing.assert_allclose(result, baseline, atol=3e-13, rtol=2e-11)
        native.host_integral_work(handle, after)
        assert tuple(after) == tuple(before)
    finally:
        native.host_integral_destroy(handle)


def test_coordinate_first_sources_transaction_and_extreme_density(native: Any) -> None:
    """Direction aliases, late mutation and nonfinite data cannot publish a bank replay."""
    with _owner(native, spins=2) as (handle, density, _):
        expected = _apply(native, handle, density)
        _coordinate_sources(native, handle, 2)
        motion = MOTION.copy()
        overlap, fock = (
            np.full(density.shape[1:], -781.25),
            np.full_like(density, -781.25),
        )

        def refused() -> None:
            assert (
                native.host_integral_apply(
                    handle,
                    _pointer(motion),
                    motion.size,
                    _pointer(overlap),
                    overlap.size,
                    _pointer(fock),
                    fock.size,
                )
                != 0
            )
            np.testing.assert_array_equal(overlap, -781.25)
            np.testing.assert_array_equal(fock, -781.25)

        for value in (np.nan, np.inf):
            motion[0, 0] = value
            refused()
        motion[:] = MOTION
        native.host_integral_failure(handle, 11, 2)
        refused()
        assert motion[0, 0] != MOTION[0, 0]
        motion[:] = MOTION
        native.host_integral_failure(handle, 0, 0)
        original_density = density.copy()
        density[0, 0, 0] += 0.125
        refused()
        density[:] = original_density
        for result, reference in zip(
            _apply(native, handle, density), expected, strict=True
        ):
            np.testing.assert_allclose(result, reference, atol=3e-13)
    density = np.full((2, 2, 2), 1e51)
    with _owner(native, angular=0, spins=2, density=density) as (handle, density, _):
        expected = _apply(native, handle, density)
        assert native.host_integral_coordinate_mode(handle, 1, _method(2), None) != 0
        assert "density range" in native.host_integral_error().decode()
        for result, reference in zip(
            _apply(native, handle, density), expected, strict=True
        ):
            np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("spherical", [True, False])
@pytest.mark.parametrize("angular", [0, 1, 2, 3])
def test_real_frozen_integral_direction_against_independent_values(
    native: Any,
    angular: int,
    spherical: bool,
    spins: int,
    tmp_path: Path,
    record_testsuite_property: Any,
) -> None:
    contracted = angular <= 1
    with _owner(
        native, angular=angular, spherical=spherical, spins=spins, contracted=contracted
    ) as (handle, density, sizes):
        actual = _apply(native, handle, density)
        errors = []
        primal_overlaps, primal_focks = [], []
        for step in (1e-3, 2e-4, 4e-5):
            positive = _oracle(
                CENTERS + step * MOTION, density, angular, spherical, contracted, 1, 0
            )
            negative = _oracle(
                CENTERS - step * MOTION, density, angular, spherical, contracted, 1, 0
            )
            primal_overlaps.append([positive[0], negative[0]])
            primal_focks.append([positive[1], negative[1]])
            numerical = [
                (first - second) / (2 * step)
                for first, second in zip(positive, negative, strict=True)
            ]
            errors.append(
                [
                    float(np.max(np.abs(first - second)))
                    for first, second in zip(actual, numerical, strict=True)
                ]
            )
        for first, second in zip(actual, numerical, strict=True):
            np.testing.assert_allclose(first, second, atol=2e-8, rtol=3e-7)
        assert max(errors[-1]) < max(errors[0]) / 30
        work = (ct.c_size_t * 8)()
        native.host_integral_work(handle, work)
        assert work[0] == work[1] == 1
        assert tuple(work)[2:5] == (sizes[0] ** 2, sizes[0] ** 4, sizes[3])
        assert 0 < work[5] < sizes[4]
        assert work[7] == sizes[5]
        np.savez_compressed(
            tmp_path / "independent-primal-control.npz",
            centers=CENTERS,
            motion=MOTION,
            density=density,
            native_overlap=actual[0],
            native_fock=actual[1],
            steps=np.array([1e-3, 2e-4, 4e-5]),
            primal_overlaps=np.array(primal_overlaps),
            primal_focks=np.array(primal_focks),
            errors=np.array(errors),
        )
        record_testsuite_property(
            f"integral_l{angular}_{'sph' if spherical else 'cart'}_s{spins}",
            json.dumps({"sizes": sizes, "errors": errors, "work": list(work)}),
        )
        translated = _apply(native, handle, density, np.tile([0.2, -0.3, 0.1], (2, 1)))
        for output in translated:
            np.testing.assert_allclose(output, 0, atol=5e-13)


@pytest.mark.parametrize("spins", [1, 2])
def test_effective_point_core_is_not_an_ecp_residual(native: Any, spins: int) -> None:
    with _owner(native, angular=1, spins=spins, number=28, core=18) as (
        handle,
        density,
        _,
    ):
        actual = _apply(native, handle, density)
        for step in (2e-4, 4e-5):
            positive = _oracle(CENTERS + step * MOTION, density, 1, True, True, 28, 18)
            negative = _oracle(CENTERS - step * MOTION, density, 1, True, True, 28, 18)
            for first, second, expected in zip(positive, negative, actual, strict=True):
                np.testing.assert_allclose(
                    expected, (first - second) / (2 * step), atol=3e-8, rtol=5e-7
                )


def test_zero_density_core_and_direction_linearity(native: Any) -> None:
    density = np.zeros((2, 4, 4))
    with _owner(native, spins=2, density=density) as (handle, density, _):
        overlap, fock = _apply(native, handle, density)
        np.testing.assert_array_equal(fock[0], fock[1])
        positive = _oracle(CENTERS + 4e-5 * MOTION, density, 1, True, True, 1, 0)
        negative = _oracle(CENTERS - 4e-5 * MOTION, density, 1, True, True, 1, 0)
        for expected, first, second in zip(
            (overlap, fock), positive, negative, strict=True
        ):
            np.testing.assert_allclose(
                expected, (first - second) / 8e-5, atol=2e-9, rtol=2e-7
            )
        other = np.array([[-0.07, 0.03, 0.02], [0.06, -0.13, 0.09]])
        second = _apply(native, handle, density, other)
        combined = _apply(native, handle, density, 2 * MOTION - 0.3 * other)
        for first, other_result, actual in zip(
            (overlap, fock), second, combined, strict=True
        ):
            np.testing.assert_allclose(
                actual, 2 * first - 0.3 * other_result, atol=4e-14, rtol=1e-12
            )


@pytest.mark.parametrize("cadence", [17, 64])
def test_resource_model_alias_epoch_failure_and_recovery(
    native: Any, cadence: int
) -> None:
    with _owner(native, spins=2, cadence=cadence) as (handle, density, sizes):
        expected = _apply(native, handle, density)
        for index in range(4):
            limits = list(sizes[2:6])
            limits[index] -= 1
            assert (
                native.host_integral_bind(
                    handle, _method(2), (ct.c_size_t * 4)(*limits), 64
                )
                != 0
            )
        for method, invalid_cadence in (
            (b"forged", 64),
            (_method(1), 64),
            (_method(2), 0),
            (_method(2), 4097),
        ):
            assert native.host_integral_bind(handle, method, None, invalid_cadence) != 0
        for count in (0, ct.c_size_t(-1).value):
            assert native.host_integral_plan_count(handle, count) != 0
        for model in range(1, 8):
            native.host_integral_model(handle, model)
            assert native.host_integral_bind(handle, _method(2), None, 64) != 0
            native.host_integral_model(handle, 0)
        for mode, at in (
            (1, 1),
            (1, 4),
            (2, 3),
            (3, 5),
            (4, 3),
            (5, 3),
            (6, 3),
            (7, 0),
        ):
            outputs = [np.full_like(output, -781.25) for output in expected]
            before = (ct.c_size_t * 8)()
            native.host_integral_work(handle, before)
            native.host_integral_failure(handle, mode, at)
            assert (
                native.host_integral_apply(
                    handle,
                    _pointer(MOTION),
                    6,
                    _pointer(outputs[0]),
                    outputs[0].size,
                    _pointer(outputs[1]),
                    outputs[1].size,
                )
                != 0
            )
            for output in outputs:
                np.testing.assert_array_equal(output, -781.25)
            after = (ct.c_size_t * 8)()
            native.host_integral_work(handle, after)
            assert after[1] == before[1]
            if mode == 7:
                assert after[3] - before[3] == sizes[0] ** 4
            native.host_integral_failure(handle, 0, 0)
            for output, baseline in zip(
                _apply(native, handle, density), expected, strict=True
            ):
                np.testing.assert_array_equal(output, baseline)
        overlap = np.full_like(expected[0], -781.25)
        assert (
            native.host_integral_apply(
                handle,
                _pointer(MOTION),
                6,
                _pointer(overlap),
                overlap.size,
                _pointer(density),
                density.size,
            )
            != 0
        )
        assert (
            native.host_integral_apply(
                handle,
                _pointer(MOTION),
                6,
                _pointer(overlap),
                overlap.size,
                _pointer(overlap),
                density.size,
            )
            != 0
        )
        assert (
            native.host_integral_apply(
                handle,
                _pointer(MOTION),
                6,
                native.host_integral_packed(handle),
                overlap.size,
                _pointer(expected[1]),
                density.size,
            )
            != 0
        )
        output = np.full_like(density, -781.25)
        output.ravel()[:6] = MOTION.ravel()
        assert (
            native.host_integral_apply(
                handle,
                _pointer(output),
                6,
                _pointer(overlap),
                overlap.size,
                _pointer(output),
                output.size,
            )
            == 0
        )
        np.testing.assert_array_equal(output, expected[1])
        np.testing.assert_array_equal(overlap, expected[0])
        with _owner(native, spins=2, density=density, cadence=17) as (other, _, _):
            for actual, baseline in zip(
                _apply(native, other, density), expected, strict=True
            ):
                np.testing.assert_array_equal(actual, baseline)


def test_malformed_inputs_and_density_fail_without_publication(native: Any) -> None:
    """Public shape/domain checks precede publication and remain recoverable."""
    with _owner(native, spins=2) as (handle, density, sizes):
        expected = _apply(native, handle, density)
        outputs = [np.full_like(value, -781.25) for value in expected]
        direction = MOTION.copy()

        def rejected(
            direction_count: int = 6,
            overlap_count: int = outputs[0].size,
            fock_count: int = outputs[1].size,
            null_input: bool = False,
        ) -> None:
            assert (
                native.host_integral_apply(
                    handle,
                    None if null_input else _pointer(direction),
                    direction_count,
                    _pointer(outputs[0]),
                    overlap_count,
                    _pointer(outputs[1]),
                    fock_count,
                )
                != 0
            )
            for output in outputs:
                np.testing.assert_array_equal(output, -781.25)

        rejected(direction_count=5)
        rejected(overlap_count=outputs[0].size - 1)
        rejected(fock_count=outputs[1].size - 1)
        rejected(null_input=True)
        direction[0, 0] = np.nan
        rejected()
        direction[:] = MOTION
        original = density.copy()
        for invalid in (np.nan, np.inf, original[0, 0, 1] + 0.2):
            density[0, 0, 1] = invalid
            rejected()
            density[:] = original
        packed = native.host_integral_packed(handle)
        metadata = 3 * sizes[1] + 2 * sizes[6]
        for index, invalid in (
            (3 * sizes[1], 0),
            (metadata, sizes[1]),
            (metadata + 1, sizes[6]),
            (metadata + 2, 0),
            (metadata + 3, 4),
            (metadata + 4, 1.5),
            (metadata + 7, np.nan),
        ):
            original_value = packed[index]
            packed[index] = invalid
            rejected()
            packed[index] = original_value
        for actual, baseline in zip(
            _apply(native, handle, density), expected, strict=True
        ):
            np.testing.assert_array_equal(actual, baseline)
