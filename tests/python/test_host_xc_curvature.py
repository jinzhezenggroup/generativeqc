"""Independent frozen-D semilocal mixed-energy geometry, never an SCF Hessian."""

from __future__ import annotations

import ctypes as ct
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.method.host_hessian_aot import (
    emit_registry,
    parse_inventory,
    programs,
)
from generativeqc_compiler.xc.atomic_grid_native import emit_atomic_grid_header
from test_host_grid_mixed_geometry import (
    CENTERS,
    LEFT,
    RIGHT,
    _ao,
    _grid,
    _owner,
    _pointer,
)

from tools.generate_grid_geometry_response import generate
from tools.generate_host_grid_maps import generate_curvature

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
SIZE_POINTER = ct.POINTER(ct.c_size_t)
SEMILOCAL = ".08*LDA_X+.72*GGA_X_B88+.19*LDA_C_VWN_RPA+.81*GGA_C_LYP"


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any, policy_library: Any) -> Any:
    directory = tmp_path_factory.mktemp("host-xc-curvature")
    payload = json.loads((ROOT / "manifests/host_hessian_aot_b3lyp.json").read_text())
    payload["arrays"] = [
        "ao_first_motion",
        "xc",
        "coefficient_response",
        "gga_energy_mixed",
    ]
    inventory = parse_inventory(payload)
    sources = []
    for entry in programs(inventory):
        source = directory / (entry.name + ".cpp")
        source.write_text(entry.emit())
        sources.append(source)
    registry = directory / "registry.cpp"
    registry.write_text(emit_registry(inventory))
    (directory / "host_grid_geometry_ad.hpp").write_text(generate("cpu"))
    (directory / "host_xc_curvature_maps.hpp").write_text(generate_curvature())
    (directory / "generated_atomic_grid.hpp").write_text(emit_atomic_grid_header())
    sources.extend(
        [
            registry,
            ROOT / "src/dft/grid.cpp",
            ROOT / "src/dft/host_grid_geometry.cpp",
            ROOT / "src/dft/host_grid_mixed_geometry.cpp",
            ROOT / "src/dft/host_grid_column_geometry.cpp",
            ROOT / "src/dft/host_xc_curvature.cpp",
            ROOT / "src/dft/host_xc_column.cpp",
            ROOT / "tests/native/host_grid_mixed_geometry_probe.cpp",
            ROOT / "tests/native/host_xc_curvature_probe.cpp",
            ROOT / "tests/native/host_xc_column_probe.cpp",
        ]
    )
    core = Path(policy_library._name).resolve()
    flags = [
        "-std=c++20",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "include"),
        "-I" + str(directory),
    ]
    link = [str(core), "-Wl,-rpath," + str(core.parent)]
    library = required_native_cxx.build_shared(
        sources, directory / "curvature.so", compile_args=flags, link_args=link
    )
    standalone = required_native_cxx.build_executable(
        sources,
        directory / "standalone",
        compile_args=[*flags, "-DGENERATIVEQC_CURVATURE_STANDALONE=1"],
        link_args=link,
    )
    loaded = ct.CDLL(str(library))
    loaded.standalone = standalone
    loaded.host_mixed_error.restype = ct.c_char_p
    loaded.host_mixed_create.argtypes = [
        POINTER,
        ct.POINTER(ct.c_int),
        ct.c_size_t,
        ct.c_uint,
        ct.c_uint,
        ct.c_uint,
        ct.c_size_t,
        ct.c_bool,
    ]
    loaded.host_mixed_create.restype = ct.c_void_p
    loaded.host_mixed_destroy.argtypes = [ct.c_void_p]
    loaded.host_mixed_shape.argtypes = [ct.c_void_p, SIZE_POINTER]
    loaded.host_mixed_reader.argtypes = [ct.c_void_p]
    loaded.host_mixed_reader.restype = ct.c_void_p
    loaded.host_mixed_export.argtypes = [
        ct.c_void_p,
        POINTER,
        POINTER,
        POINTER,
        ct.POINTER(ct.c_uint32),
        POINTER,
    ]
    loaded.host_curvature_error.restype = ct.c_char_p
    loaded.host_curvature_create.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        ct.c_char_p,
        SIZE_POINTER,
    ]
    loaded.host_curvature_create.restype = ct.c_void_p
    loaded.host_curvature_destroy.argtypes = [ct.c_void_p]
    loaded.host_curvature_plan.argtypes = [ct.c_void_p, ct.c_size_t, SIZE_POINTER]
    loaded.host_curvature_apply.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
    ]
    loaded.host_curvature_fault.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t]
    loaded.host_curvature_work.argtypes = [ct.c_void_p, SIZE_POINTER]
    return loaded


def _density(spins: int, basis: int) -> np.ndarray:
    random = np.random.default_rng(921 + spins + basis)
    factors = random.normal(size=(spins, basis, basis)) * 0.12
    return np.ascontiguousarray(
        factors @ factors.transpose(0, 2, 1) + np.eye(basis) * 0.08
    )


@contextmanager
def _source(native: Any, reader: int, spins: int, density: np.ndarray) -> Any:
    handle = native.host_curvature_create(
        reader, spins, _pointer(density), density.size, None, None
    )
    if not handle:
        raise RuntimeError(native.host_curvature_error().decode())
    try:
        yield handle
    finally:
        native.host_curvature_destroy(handle)


def _apply(
    native: Any,
    handle: int,
    left: np.ndarray = LEFT,
    right: np.ndarray = RIGHT,
    output: np.ndarray | None = None,
) -> tuple:
    left, right = (
        np.ascontiguousarray(left, dtype=float),
        np.ascontiguousarray(right, dtype=float),
    )
    if output is None:
        output = np.full(4, -9841.5)
    status = native.host_curvature_apply(
        handle,
        _pointer(left),
        left.size,
        _pointer(right),
        right.size,
        _pointer(output),
        output.size,
    )
    return status, output


def _primal(
    density: np.ndarray,
    centers: np.ndarray,
    points: np.ndarray,
    weight: np.ndarray,
    maximum: int,
    cartesian: bool,
) -> float:
    """Independent libcint/Libxc primal energy on the actual displaced grid rule."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    ao = _ao(centers, points, maximum, cartesian)
    rho = np.einsum("pi,sij,pj->sp", ao[0], density, ao[0])
    gradients = np.stack(
        [
            np.einsum("pi,sij,pj->sp", component, density, ao[0])
            + np.einsum("pi,sij,pj->sp", ao[0], density, component)
            for component in ao[1:]
        ],
        axis=1,
    )
    fields = np.concatenate((rho[:, None], gradients), axis=1)
    spins = density.shape[0]
    energy = libxc.eval_xc(
        SEMILOCAL, fields if spins == 2 else fields[0], spin=spins - 1, deriv=0
    )[0]
    return float(np.dot(weight, rho.sum(axis=0) * energy))


CASES = [(spins, version, False) for spins in (1, 2) for version in (1, 2, 3, 4)] + [
    (1, 3, True),
    (2, 3, True),
]


@pytest.mark.parametrize(("spins", "version", "cartesian"), CASES)
def test_frozen_xc_energy_mixed_against_independent_primal(
    native: Any,
    spins: int,
    version: int,
    cartesian: bool,
    tmp_path: Path,
    record_testsuite_property: Any,
) -> None:
    maximum = 3 if version == 3 else 1
    with _owner(native, version, maximum=maximum, cartesian=cartesian, tile=257) as (
        handle,
        shape,
    ):
        density = _density(spins, shape[1])
        reader = native.host_mixed_reader(handle)
        points, weight, _, _, _ = _grid(native, handle, shape)
        baseline = _primal(density, CENTERS, points, weight, maximum, cartesian)
        with _source(native, reader, spins, density) as source:
            status, actual = _apply(native, source)
            assert status == 0, native.host_curvature_error().decode()
            np.testing.assert_allclose(actual[0], baseline, atol=3e-11, rtol=2e-12)
            primals, errors = [], []
            steps = [1e-2, 3e-3, 1e-3]
            for step in steps:
                corners = []
                for sign_left, sign_right in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    displaced = CENTERS + step * (sign_left * LEFT + sign_right * RIGHT)
                    with _owner(
                        native,
                        version,
                        maximum=maximum,
                        cartesian=cartesian,
                        centers=displaced,
                        tile=257,
                    ) as (shifted, shifted_shape):
                        shifted_points, shifted_weights, _, _, _ = _grid(
                            native, shifted, shifted_shape
                        )
                    corners.append(
                        _primal(
                            density,
                            displaced,
                            shifted_points,
                            shifted_weights,
                            maximum,
                            cartesian,
                        )
                    )
                numerical = np.array(
                    [
                        (corners[0] + corners[1] - corners[2] - corners[3])
                        / (4 * step),
                        (corners[0] - corners[1] + corners[2] - corners[3])
                        / (4 * step),
                        (corners[0] - corners[1] - corners[2] + corners[3])
                        / (4 * step * step),
                    ]
                )
                primals.append(corners)
                errors.append(np.abs(actual[1:] - numerical))
            np.savez_compressed(
                tmp_path / "frozen-xc-curvature.npz",
                centers=CENTERS,
                density=density,
                left=LEFT,
                right=RIGHT,
                native=actual,
                primal_baseline=baseline,
                primal_corners=primals,
                steps=steps,
                errors=errors,
                grid_version=version,
                cartesian=cartesian,
                maximum=maximum,
            )
            np.testing.assert_allclose(actual[1:], numerical, atol=3e-7, rtol=2e-6)
            assert errors[-1].max() < max(errors[0].max() / 12, 3e-8)
            work = (ct.c_size_t * 5)()
            resources = (ct.c_size_t * 4)()
            native.host_curvature_work(source, work)
            assert native.host_curvature_plan(reader, spins, resources) == 0
            assert work[1] == 1 and work[3] == shape[2] and work[4] == resources[1]
            record_testsuite_property(
                f"frozen_xc_s{spins}_v{version}_cart{cartesian}",
                json.dumps(
                    {
                        "errors": np.array(errors).tolist(),
                        "work": list(work),
                        "resources": list(resources),
                    }
                ),
            )


def test_bilinear_symmetry_scaling_translation_and_tile_reuse(
    native: Any,
) -> None:
    with _owner(native, maximum=1, tile=53) as (owner, shape):
        density = _density(2, shape[1])
        reader = native.host_mixed_reader(owner)
        with _source(native, reader, 2, density) as source:
            status, actual = _apply(native, source)
            assert status == 0, native.host_curvature_error().decode()
            status, swapped = _apply(native, source, RIGHT, LEFT)
            assert status == 0
            np.testing.assert_allclose(
                actual[[0, 1, 2, 3]], swapped[[0, 2, 1, 3]], atol=2e-12, rtol=2e-12
            )
            status, scaled = _apply(native, source, 2 * LEFT, -3 * RIGHT)
            assert status == 0
            np.testing.assert_allclose(
                scaled, actual * [1, 2, -3, -6], atol=2e-12, rtol=2e-12
            )
            translation = np.tile([0.2, -0.1, 0.03], (3, 1))
            status, translated = _apply(native, source, translation, RIGHT)
            assert status == 0
            np.testing.assert_allclose(translated[[1, 3]], 0, atol=2e-13)
            alias = LEFT.copy()
            status, aliased = _apply(native, source, alias, RIGHT, alias.ravel()[:4])
            assert status == 0
            np.testing.assert_array_equal(aliased, actual)
        with (
            _owner(native, maximum=1, tile=37) as (other, _),
            _source(native, native.host_mixed_reader(other), 2, density) as source,
        ):
            status, retiled = _apply(native, source)
            assert status == 0
            np.testing.assert_allclose(retiled, actual, atol=2e-12, rtol=2e-12)


def test_nonvacuum_empty_spin_retains_shared_response_domain_refusal(
    native: Any,
) -> None:
    """A finite restricted path is not a newly qualified full ingredient Hessian."""
    with _owner(native, maximum=1, tile=53) as (owner, shape):
        density = _density(2, shape[1])
        density[1] = 0
        reader = native.host_mixed_reader(owner)
        with _source(native, reader, 2, density) as source:
            status, output = _apply(native, source)
            assert status != 0
            assert b"mixed action failed" in native.host_curvature_error()
            np.testing.assert_array_equal(output, -9841.5)
        with _source(native, reader, 2, _density(2, shape[1])) as source:
            assert _apply(native, source)[0] == 0


@pytest.mark.parametrize("spins", [1, 2])
def test_exact_linked_vacuum_is_lazy_without_density_or_tangent_floors(
    native: Any, spins: int
) -> None:
    with _owner(native, maximum=1, tile=53) as (owner, shape):
        reader = native.host_mixed_reader(owner)
        for scale in (0, 1e-22):
            density = scale * _density(spins, shape[1])
            with _source(native, reader, spins, density) as source:
                status, output = _apply(native, source)
                assert status == 0, native.host_curvature_error().decode()
                np.testing.assert_array_equal(output, 0)


def test_caps_identity_shapes_density_owner_failure_alias_and_recovery(
    native: Any,
) -> None:
    with _owner(native, maximum=0, tile=47) as (owner, shape):
        reader = native.host_mixed_reader(owner)
        density = _density(1, shape[1])
        resources = (ct.c_size_t * 4)()
        assert native.host_curvature_plan(reader, 1, resources) == 0
        for index in range(4):
            caps = (ct.c_size_t * 4)(*resources)
            caps[index] -= 1
            assert not native.host_curvature_create(
                reader, 1, _pointer(density), density.size, None, caps
            )
        assert not native.host_curvature_create(
            reader, 1, _pointer(density), density.size, b"wrong", None
        )
        assert not native.host_curvature_create(
            reader, 3, _pointer(density), density.size, None, None
        )
        assert not native.host_curvature_create(
            reader, 1, _pointer(density), density.size - 1, None, None
        )
        assert not native.host_curvature_create(
            reader, 1, POINTER(), density.size, None, None
        )
        for mode in (1, 2, 3):
            with _source(native, reader, 1, density) as source:
                native.host_curvature_fault(source, mode, shape[2])
                status, output = _apply(native, source)
                assert status != 0
                np.testing.assert_array_equal(output, -9841.5)
                native.host_curvature_fault(source, 0, 0)
                assert _apply(native, source)[0] == 0
        with _source(native, reader, 1, density) as source:
            before = density.copy()
            assert _apply(native, source, output=density.ravel()[:4])[0] != 0
            np.testing.assert_array_equal(density, before)
            status, output = _apply(native, source, LEFT.ravel()[:-1])
            assert status != 0
            np.testing.assert_array_equal(output, -9841.5)
            saved = density[0, 0, 0]
            for invalid in (np.nan, np.inf):
                density[0, 0, 0] = invalid
                status, output = _apply(native, source)
                assert status != 0
                np.testing.assert_array_equal(output, -9841.5)
            density[0, 0, 0] = saved
            assert _apply(native, source)[0] == 0
        negative = -np.eye(shape[1])[None]
        with _source(native, reader, 1, negative) as source:
            status, output = _apply(native, source)
            assert status != 0
            np.testing.assert_array_equal(output, -9841.5)


def test_native_frozen_xc_curvature_runs_without_python_or_compiler(
    native: Any, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    subprocess.run(
        [str(native.standalone)],
        env={**os.environ, "PATH": str(empty), "PYTHONPATH": ""},
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
    )
