"""Qualify bounded native AO XC response without a molecular/Python runtime."""

from __future__ import annotations

import ctypes as ct
import json
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.method.host_hessian_aot import (
    emit_registry,
    parse_inventory,
    programs,
)

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
METHOD = ".08*LDA_X+.72*GGA_X_B88+.19*LDA_C_VWN_RPA+.81*GGA_C_LYP"


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any) -> tuple:
    """Link the real scalar AD and native tile consumer; no public runtime."""
    directory = tmp_path_factory.mktemp("host-xc-response")
    payload = json.loads((ROOT / "manifests/host_hessian_aot_b3lyp.json").read_text())
    payload["arrays"] = ["xc", "coefficient_response"]
    inventory = parse_inventory(payload)
    selected = programs(inventory)
    sources = []
    for entry in selected:
        source = directory / (entry.name + ".cpp")
        source.write_text(entry.emit())
        sources.append(source)
    registry = directory / "registry.cpp"
    registry.write_text(emit_registry(inventory))
    probe = directory / "probe.cpp"
    probe.write_text(
        r"""
#include "dft/host_xc_response.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using namespace generativeqc::dft;
struct Handle {
  bool alive = true;
  int failure = 0;
  std::unique_ptr<HostGgaXcResponse> response;
};
thread_local std::string error;
extern "C" const char* last_error() { return error.c_str(); }
extern "C" std::size_t query(std::size_t basis, std::size_t points,
                             std::size_t spins, std::size_t tile) {
  try { return HostGgaXcResponse::workspace_bytes({basis, points, spins, tile}); }
  catch (const std::exception& failure) { error = failure.what(); return SIZE_MAX; }
}
extern "C" void* create(std::size_t basis, std::size_t points, std::size_t spins,
                        std::size_t tile, std::size_t budget, const double* density,
                        const double* ao, const double* weights,
                        const char* xc, const char* coefficients) {
  try {
    const HostGgaResponseShape shape{basis, points, spins, tile};
    HostGgaXcResponse::workspace_bytes(shape);
    auto handle = std::make_unique<Handle>();
    auto* owner = handle.get();
    handle->response = std::make_unique<HostGgaXcResponse>(
      shape, budget, std::span(density, spins * basis * basis), xc, coefficients,
      [=](std::size_t begin, std::size_t count, std::span<double> values,
           std::span<double> quadrature) {
        if (owner->failure == 1) throw std::runtime_error("injected tile failure");
        if (owner->failure == 4) {
          std::vector<double> direction(spins * basis * basis), result(direction.size());
          owner->response->apply(direction, result);
        }
        for (std::size_t component = 0; component < 4; ++component)
          std::copy_n(ao + (component * points + begin) * basis, count * basis,
                      values.begin() + component * count * basis);
        std::copy_n(weights + begin, count, quadrature.begin());
        if (owner->failure == 2) owner->alive = false;
        if (owner->failure == 3) values.back() = std::numeric_limits<double>::quiet_NaN();
      },
      [owner] { if (!owner->alive) throw std::runtime_error("stationary owner expired"); });
    return handle.release();
  } catch (const std::exception& failure) { error = failure.what(); return nullptr; }
}
extern "C" void destroy(Handle* handle) { delete handle; }
extern "C" void set_failure(Handle* handle, int failure) {
  handle->failure = failure;
  handle->alive = failure != 5;
}
extern "C" int apply(Handle* handle, const double* direction, std::size_t input_count,
                      double* output, std::size_t output_count) {
  try { handle->response->apply({direction, input_count}, {output, output_count}); return 0; }
  catch (const std::exception& failure) { error = failure.what(); return -1; }
}
extern "C" std::size_t batch_query(std::size_t basis, std::size_t points,
    std::size_t spins, std::size_t tile, std::size_t directions) {
  try { return HostGgaXcResponse::batch_workspace_bytes({basis, points, spins, tile}, directions); }
  catch (const std::exception& failure) { error = failure.what(); return SIZE_MAX; }
}
extern "C" int apply_batch(Handle* handle, const double* directions, std::size_t input_count,
    double* output, std::size_t output_count, double* workspace, std::size_t workspace_count) {
  try {
    handle->response->apply_batch({directions, input_count}, {output, output_count}, {workspace, workspace_count});
    return 0;
  } catch (const std::exception& failure) { error = failure.what(); return -1; }
}
extern "C" std::size_t scratch(Handle* handle) { return handle->response->scratch_bytes(); }
extern "C" std::size_t cache_query(std::size_t basis, std::size_t points,
                                   std::size_t spins, std::size_t tile,
                                   std::size_t maximum) {
  try { return HostGgaXcResponse::primal_cache_bytes({basis, points, spins, tile}, maximum); }
  catch (const std::exception& failure) { error = failure.what(); return SIZE_MAX; }
}
extern "C" int admit_cache(Handle* handle, std::size_t maximum) {
  try { return handle->response->admit_primal_cache(maximum) ? 1 : 0; }
  catch (const std::exception& failure) { error = failure.what(); return -1; }
}
extern "C" void cache_work(Handle* handle, std::size_t* output) {
  const auto work = handle->response->work();
  output[0] = work.xc_evaluated_points;
  output[1] = work.cached_xc_points;
}
extern "C" void work(Handle* handle, std::size_t* output) {
  const auto work = handle->response->work();
  output[0] = work.actions_started;
  output[1] = work.actions_completed;
  output[2] = work.point_visits;
  output[3] = work.tile_reads;
}
"""
    )
    compiler = replace(
        required_native_cxx, cache=shutil.which("sccache") or required_native_cxx.cache
    )
    subprocess.run([compiler.cache, "--version"], check=True, capture_output=True)
    library = compiler.build_shared(
        [*sources, registry, ROOT / "src/dft/host_xc_response.cpp", probe],
        directory / "response.so",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I" + str(ROOT / "src"),
        ],
    )
    loaded = ct.CDLL(str(library))
    loaded.last_error.restype = ct.c_char_p
    loaded.query.argtypes = [ct.c_size_t] * 4
    loaded.query.restype = ct.c_size_t
    loaded.create.argtypes = [ct.c_size_t] * 5 + [POINTER] * 3 + [ct.c_char_p] * 2
    loaded.create.restype = ct.c_void_p
    loaded.destroy.argtypes = [ct.c_void_p]
    loaded.set_failure.argtypes = [ct.c_void_p, ct.c_int]
    loaded.apply.argtypes = [ct.c_void_p, POINTER, ct.c_size_t, POINTER, ct.c_size_t]
    loaded.apply.restype = ct.c_int
    loaded.batch_query.argtypes = [ct.c_size_t] * 5
    loaded.batch_query.restype = ct.c_size_t
    loaded.apply_batch.argtypes = [
        ct.c_void_p,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
        POINTER,
        ct.c_size_t,
    ]
    loaded.apply_batch.restype = ct.c_int
    loaded.scratch.argtypes = [ct.c_void_p]
    loaded.scratch.restype = ct.c_size_t
    loaded.cache_query.argtypes = [ct.c_size_t] * 5
    loaded.cache_query.restype = ct.c_size_t
    loaded.admit_cache.argtypes = [ct.c_void_p, ct.c_size_t]
    loaded.admit_cache.restype = ct.c_int
    loaded.cache_work.argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
    loaded.work.argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t)]
    return loaded, {entry.name: entry.identity.encode() for entry in selected}


def _pointer(values: np.ndarray) -> POINTER:
    return values.ctypes.data_as(POINTER)


@dataclass
class Response:
    """Keep borrowed buffers alive for every native action."""

    library: Any
    handle: int
    density: np.ndarray
    ao: np.ndarray
    weights: np.ndarray
    tile: int

    def apply(
        self, direction: np.ndarray, output: np.ndarray | None = None
    ) -> np.ndarray:
        direction = np.ascontiguousarray(direction, dtype=np.float64)
        if output is None:
            output = np.empty_like(direction)
        result = self.library.apply(
            self.handle,
            _pointer(direction),
            direction.size,
            _pointer(output),
            output.size,
        )
        if result:
            raise RuntimeError(self.library.last_error().decode())
        return output

    def batch_bytes(self, count: int) -> int:
        """Query only additional caller-owned staging, never allocate a full grid."""
        spins, basis, _ = self.density.shape
        needed = self.library.batch_query(
            basis, self.weights.size, spins, self.tile, count
        )
        if needed == ct.c_size_t(-1).value:
            raise RuntimeError(self.library.last_error().decode())
        return needed

    def apply_batch(
        self,
        directions: np.ndarray,
        output: np.ndarray | None = None,
        workspace: np.ndarray | None = None,
    ) -> np.ndarray:
        """Keep the explicit workspace and all borrowed inputs alive synchronously."""
        directions = np.ascontiguousarray(directions, dtype=np.float64)
        if output is None:
            output = np.empty_like(directions)
        if workspace is None:
            workspace = np.empty(
                self.batch_bytes(directions.size // self.density.size) // 8
            )
        result = self.library.apply_batch(
            self.handle,
            _pointer(directions),
            directions.size,
            _pointer(output),
            output.size,
            _pointer(workspace),
            workspace.size,
        )
        if result:
            raise RuntimeError(self.library.last_error().decode())
        return output


@contextmanager
def _response(
    native: tuple,
    density: np.ndarray,
    ao: np.ndarray,
    weights: np.ndarray,
    tile: int,
    *,
    budget_delta: int = 0,
    bad_identity: bool = False,
) -> Any:
    library, identities = native
    density, ao, weights = [
        np.ascontiguousarray(value, dtype=np.float64)
        for value in (density, ao, weights)
    ]
    spins, basis, _ = density.shape
    points = weights.size
    spin = "polarized" if spins == 2 else "unpolarized"
    budget = library.query(basis, points, spins, tile)
    handle = library.create(
        basis,
        points,
        spins,
        tile,
        budget + budget_delta,
        _pointer(density),
        _pointer(ao),
        _pointer(weights),
        b"wrong-identity" if bad_identity else identities["xc_" + spin],
        identities["coefficient_response_" + spin],
    )
    if not handle:
        raise RuntimeError(library.last_error().decode())
    try:
        yield Response(library, handle, density, ao, weights, tile)
    finally:
        library.destroy(handle)


def _inputs(spins: int) -> tuple:
    generator = np.random.default_rng(2844)
    basis, points = 4, 19
    ao = generator.normal(size=(4, points, basis)) * 0.2
    ao[0] += 0.7
    density = []
    for _ in range(spins):
        factor = generator.normal(size=(basis, basis)) * 0.1
        density.append(factor @ factor.T + np.eye(basis) * 0.3)
    direction = generator.normal(size=(spins, basis, basis)) * 0.04
    direction = direction + direction.transpose(0, 2, 1)
    weights = generator.uniform(0.01, 0.4, size=points)
    weights[3] *= -1
    weights[7] = 0
    return np.array(density), ao, weights, direction


def _independent_fock(
    density: np.ndarray, ao: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Independent Libxc first potentials, not production response algebra."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    spins = len(density)
    value, gradient = ao[0], ao[1:]
    rho = np.einsum("pi,sij,pj->sp", value, density, value)
    grad_rho = np.einsum("kpi,sij,pj->skp", gradient, density, value) + np.einsum(
        "pi,sij,kpj->skp", value, density, gradient
    )
    ingredients = np.concatenate((rho[:, None, :], grad_rho), axis=1)
    _, potentials, _, _ = libxc.eval_xc(
        METHOD, ingredients if spins == 2 else ingredients[0], spin=spins - 1, deriv=1
    )
    result = np.zeros_like(density)
    for spin in range(spins):
        if spins == 1:
            scalar = potentials[0]
            vector = 2 * potentials[1] * grad_rho[0]
        else:
            scalar = potentials[0][:, spin]
            vector = (
                2 * potentials[1][:, 2 * spin] * grad_rho[spin]
                + potentials[1][:, 1] * grad_rho[1 - spin]
            )
        result[spin] = np.einsum("p,pi,pj->ij", weights * scalar, value, value)
        contribution = np.einsum("p,kp,kpi,pj->ij", weights, vector, gradient, value)
        result[spin] += contribution + contribution.T
    return result


@pytest.mark.parametrize("spins", [1, 2])
def test_full_ao_response_against_three_independent_potential_steps(
    native: tuple, spins: int
) -> None:
    density, ao, weights, direction = _inputs(spins)
    with _response(native, density, ao, weights, 4) as response:
        actual = response.apply(direction)
        for step in (1e-3, 3e-4, 1e-4):
            reference = sum(
                coefficient
                * _independent_fock(density + offset * step * direction, ao, weights)
                for offset, coefficient in ((-2, 1), (-1, -8), (1, 8), (2, -1))
            ) / (12 * step)
            np.testing.assert_allclose(actual, reference, atol=3e-9, rtol=3e-8)
        np.testing.assert_allclose(actual, actual.transpose(0, 2, 1), atol=2e-14)


@pytest.mark.parametrize("spins", [1, 2])
def test_tiling_bilinearity_aliasing_and_numeric_capacity(
    native: tuple, spins: int
) -> None:
    density, ao, weights, direction = _inputs(spins)
    library, _ = native
    other = np.flip(direction, axis=1).copy()
    other = other + other.transpose(0, 2, 1)
    results = []
    for tile in (1, 4, weights.size, weights.size + 10):
        with _response(native, density, ao, weights, tile) as response:
            actual = response.apply(direction)
            results.append(actual)
            moved = response.apply(other)
            assert np.sum(direction * moved) == pytest.approx(
                np.sum(other * actual), abs=2e-12
            )
            aliased = direction.copy()
            np.testing.assert_allclose(
                response.apply(aliased, aliased), actual, atol=2e-14
            )
            np.testing.assert_array_equal(response.apply(np.zeros_like(direction)), 0)
            assert library.scratch(response.handle) <= library.query(
                density.shape[-1], weights.size, spins, tile
            )
            work = (ct.c_size_t * 4)()
            library.work(response.handle, work)
            assert work[0] >= work[1] > 0
            assert work[2] >= weights.size and work[3] > 0
    for actual in results[1:]:
        np.testing.assert_allclose(actual, results[0], atol=2e-14)
    assert library.query(4, 1000, spins, 4) == library.query(4, 1_000_000, spins, 4)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("cached", [False, True])
def test_nonsymmetric_density_directions_keep_full_ao_semantics(
    native: tuple, spins: int, cached: bool
) -> None:
    """Neither input is implicitly replaced by one of its matrix triangles."""
    density, ao, weights, direction = _inputs(spins)
    generator = np.random.default_rng(2848)
    other = generator.normal(size=density.shape) * 0.03
    density += other - other.transpose(0, 2, 1)
    direction += generator.normal(size=direction.shape) * 0.02
    with _response(native, density, ao, weights, 4) as response:
        if cached:
            assert (
                response.library.admit_cache(response.handle, _cache_bytes(response))
                == 1
            )
        actual = response.apply(direction)
        for step in (1e-3, 3e-4, 1e-4):
            reference = sum(
                coefficient
                * _independent_fock(density + offset * step * direction, ao, weights)
                for offset, coefficient in ((-2, 1), (-1, -8), (1, 8), (2, -1))
            ) / (12 * step)
            np.testing.assert_allclose(actual, reference, atol=3e-9, rtol=3e-8)
        np.testing.assert_array_equal(response.apply(direction), actual)


@pytest.mark.parametrize("basis", (4, 17, 81))
def test_ordered_packing_preserves_vacuum_boundary_classification(
    native: tuple, basis: int
) -> None:
    """Enough repeated points activate packing; a one-point tile is the control.

    BLAS reassociation can move a near-boundary density across the linked lazy
    vacuum branch, even when ordinary finite-density error appears negligible.
    Exercise nearby representable density scales without changing that model.
    """
    points = (4 * basis + 8) // 9
    for seed in range(8):
        generator = np.random.default_rng(seed + 2848)
        factor = generator.normal(size=(basis, basis))
        base_density = factor @ factor.T / basis + np.eye(basis) * 0.4
        values = generator.normal(size=basis)
        ao = np.zeros((4, points, basis))
        ao[0] = values
        rho = 0.0
        for row in range(basis):
            for column in range(basis):
                rho += base_density[row, column] * (values[row] * values[column])
        central = 1e-18 / rho
        scales = [central]
        left = right = central
        for _ in range(16):
            left = np.nextafter(left, 0)
            right = np.nextafter(right, np.inf)
            scales.extend((left, right))
        weights = np.ones(points) / points
        direction = (np.eye(basis) * 0.1)[None]
        for scale in scales:
            density = (scale * base_density)[None]
            with _response(native, density, ao, weights, 1) as scalar:
                control = scalar.apply(direction)
            with _response(native, density, ao, weights, points) as packed:
                actual = packed.apply(direction)
            np.testing.assert_array_equal(
                actual.view(np.uint64), control.view(np.uint64)
            )


def test_cross_spin_coupling_and_restricted_equivalence(native: tuple) -> None:
    density, ao, weights, direction = _inputs(2)
    direction[0] = 0
    with _response(native, density, ao, weights, 4) as response:
        actual = response.apply(direction)
        assert np.max(np.abs(actual[0])) > 1e-6
    with _response(native, density[::-1], ao, weights, 4) as response:
        swapped = response.apply(direction[::-1])
        np.testing.assert_allclose(swapped[::-1], actual, atol=2e-13)
    restricted_density = density[:1] + density[1:]
    restricted_direction = direction[:1] + direction[1:]
    with _response(native, restricted_density, ao, weights, 4) as response:
        restricted = response.apply(restricted_direction)
    with _response(
        native, np.repeat(restricted_density / 2, 2, axis=0), ao, weights, 4
    ) as response:
        unrestricted = response.apply(np.repeat(restricted_direction / 2, 2, axis=0))
    np.testing.assert_allclose(
        unrestricted, np.repeat(restricted, 2, axis=0), atol=3e-13
    )


@pytest.mark.parametrize("failure", [1, 2, 3, 4, 5])
def test_failed_provider_lifetime_reentry_and_recovery(
    native: tuple, failure: int
) -> None:
    density, ao, weights, direction = _inputs(2)
    library, _ = native
    with _response(native, density, ao, weights, 4) as response:
        expected = response.apply(direction)
        library.set_failure(response.handle, failure)
        output = np.full_like(direction, -9182.5)
        with pytest.raises(RuntimeError):
            response.apply(direction, output)
        np.testing.assert_array_equal(output, -9182.5)
        library.set_failure(response.handle, 0)
        np.testing.assert_allclose(response.apply(direction), expected, atol=2e-14)


def test_late_nonfinite_grid_failure_is_transactional(native: tuple) -> None:
    density, ao, weights, direction = _inputs(2)
    with _response(native, density, ao, weights, 4) as response:
        expected = response.apply(direction)
        saved = response.ao[0, -1, -1]
        response.ao[0, -1, -1] = np.nan
        output = np.full_like(direction, -9182.5)
        with pytest.raises(RuntimeError, match="nonfinite"):
            response.apply(direction, output)
        np.testing.assert_array_equal(output, -9182.5)
        response.ao[0, -1, -1] = saved
        np.testing.assert_allclose(response.apply(direction), expected, atol=2e-14)


def test_budget_identity_shape_and_undefined_point_rejections(native: tuple) -> None:
    density, ao, weights, direction = _inputs(2)
    library, _ = native
    for shape in ((0, 1, 2, 1), (4, 0, 2, 1), (4, 1, 3, 1), (4, 1, 2, 0)):
        assert library.query(*shape) == ct.c_size_t(-1).value
    assert library.query(ct.c_size_t(-1), 10, 2, 1) == ct.c_size_t(-1).value
    with (
        pytest.raises(RuntimeError, match="budget"),
        _response(native, density, ao, weights, 4, budget_delta=-1),
    ):
        pytest.fail("under-budget owner must not be constructed")
    with (
        pytest.raises(RuntimeError, match="identity"),
        _response(native, density, ao, weights, 4, bad_identity=True),
    ):
        pytest.fail("unknown identity must not trigger compilation or fallback")
    with _response(native, density, ao, weights, 4) as response:
        for invalid in (-density,):
            response.density[:] = invalid
            with pytest.raises(RuntimeError, match="nonnegative"):
                response.apply(direction)
        response.density[:] = np.nan
        with pytest.raises(RuntimeError, match="nonfinite"):
            response.apply(direction)
        response.density[:] = _inputs(2)[0]
        with pytest.raises(RuntimeError, match="alias"):
            response.apply(direction, response.density)
        with pytest.raises(RuntimeError, match="shape"):
            response.apply(direction.ravel()[:-1])
        direction[0, 0, 0] = np.inf
        with pytest.raises(RuntimeError, match="nonfinite"):
            response.apply(direction)


@pytest.mark.parametrize("spins", [1, 2])
def test_production_vacuum_is_lazy_without_density_or_tangent_floors(
    native: tuple, spins: int
) -> None:
    density, ao, weights, direction = _inputs(spins)
    for scale in (0.0, 1e-22):
        with _response(native, scale * density, ao, weights, 4) as response:
            np.testing.assert_array_equal(response.apply(direction), 0)
            np.testing.assert_array_equal(response.apply(-direction), 0)
    selected = np.arange(weights.size) % 3 != 0
    mixed = ao.copy()
    mixed[:, ~selected] = 0
    with (
        _response(native, density, mixed, weights, 4) as response,
        _response(native, density, ao[:, selected], weights[selected], 4) as active,
    ):
        np.testing.assert_allclose(
            response.apply(direction), active.apply(direction), atol=2e-14
        )


def test_nonvacuum_spin_endpoint_and_incompatible_zero_gradient_fail_closed(
    native: tuple,
) -> None:
    density, ao, weights, direction = _inputs(2)
    density[0] = 0
    with _response(native, density, ao, weights, 4) as response:
        output = np.full_like(direction, -9182.5)
        with pytest.raises(RuntimeError, match="nonfinite"):
            response.apply(direction, output)
        np.testing.assert_array_equal(output, -9182.5)

    density = np.array([[[1.0, 0.0], [0.0, -1.0]]])
    ao = np.zeros((4, 1, 2))
    ao[0, 0] = [1, 1]
    ao[1, 0] = [1, 0]
    with _response(native, density, ao, np.ones(1), 1) as response:
        output = np.full_like(density, -9182.5)
        with pytest.raises(RuntimeError, match="zero gradient"):
            response.apply(np.ones_like(density), output)
        np.testing.assert_array_equal(output, -9182.5)


def _cache_work(response: Response) -> tuple[int, int]:
    """Observe semantic scalar-evaluation/replay work, not internal call counts."""
    work = (ct.c_size_t * 2)()
    response.library.cache_work(response.handle, work)
    return tuple(work)


def _cache_bytes(response: Response, maximum: int = ct.c_size_t(-1).value) -> int:
    return response.library.cache_query(
        response.density.shape[-1],
        response.weights.size,
        len(response.density),
        4,
        maximum,
    )


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("tile", [1, 4, 64])
@pytest.mark.parametrize("capacity", ["none", "one_tile", "full"])
def test_exact_primal_bank_prefix_budget_and_independent_response(
    native: tuple, spins: int, tile: int, capacity: str
) -> None:
    """Bounded prefixes replay exact AD partials; every uncached point streams."""
    density, ao, weights, direction = _inputs(spins)
    with (
        _response(native, density, ao, weights, tile) as response,
        _response(native, density, ao, weights, tile) as streamed,
    ):
        library = response.library
        full = library.cache_query(4, weights.size, spins, tile, ct.c_size_t(-1).value)
        per_point = full // weights.size
        points = {"none": 0, "one_tile": min(tile, weights.size), "full": weights.size}[
            capacity
        ]
        allowance = points * per_point
        planned = library.cache_query(4, weights.size, spins, tile, allowance)
        base = library.scratch(response.handle)
        assert library.admit_cache(response.handle, allowance) == int(planned > 0)
        assert library.scratch(response.handle) == base + planned <= base + allowance
        assert _cache_work(response) == (0, 0)
        actual = response.apply(direction)
        np.testing.assert_array_equal(actual, streamed.apply(direction))
        np.testing.assert_array_equal(response.apply(-direction), -actual)
        aliased = direction.copy()
        np.testing.assert_array_equal(response.apply(aliased, aliased), actual)
        prefix = planned // per_point
        assert _cache_work(response) == (
            weights.size + 2 * (weights.size - prefix),
            2 * prefix,
        )
        for step in (1e-3, 3e-4, 1e-4):
            reference = sum(
                coefficient
                * _independent_fock(density + offset * step * direction, ao, weights)
                for offset, coefficient in ((-2, 1), (-1, -8), (1, 8), (2, -1))
            ) / (12 * step)
            np.testing.assert_allclose(actual, reference, atol=3e-9, rtol=3e-8)
        assert library.admit_cache(response.handle, allowance) == -1


@pytest.mark.parametrize("spins", [1, 2])
def test_primal_bank_authenticates_exact_fields_and_recovers(
    native: tuple, spins: int
) -> None:
    """Finite changed primal inputs refuse instead of silently refreshing a bank."""
    density, ao, weights, direction = _inputs(spins)
    with _response(native, density, ao, weights, 4) as response:
        assert (
            response.library.admit_cache(response.handle, _cache_bytes(response)) == 1
        )
        expected = response.apply(direction)
        for values, index in ((response.density, (0, 0, 0)), (response.ao, (0, 0, 0))):
            saved = values[index]
            values[index] += 0.01
            output = np.full_like(direction, -9182.5)
            with pytest.raises(RuntimeError, match="cached point inputs changed"):
                response.apply(direction, output)
            np.testing.assert_array_equal(output, -9182.5)
            values[index] = saved
            np.testing.assert_array_equal(response.apply(direction), expected)
        assert _cache_work(response)[0] == weights.size


@pytest.mark.parametrize("failure", [1, 2, 3, 4, 5])
def test_primal_bank_does_not_bypass_live_provider_checks(
    native: tuple, failure: int
) -> None:
    """Even fully retained partials still consume and authenticate each AO tile."""
    density, ao, weights, direction = _inputs(2)
    with _response(native, density, ao, weights, 4) as response:
        library = response.library
        assert library.admit_cache(response.handle, _cache_bytes(response)) == 1
        expected = response.apply(direction)
        library.set_failure(response.handle, failure)
        output = np.full_like(direction, -9182.5)
        with pytest.raises(RuntimeError):
            response.apply(direction, output)
        np.testing.assert_array_equal(output, -9182.5)
        library.set_failure(response.handle, 0)
        np.testing.assert_array_equal(response.apply(direction), expected)


def test_failed_first_primal_preparation_cannot_publish_a_partial_bank(
    native: tuple,
) -> None:
    """A late failure must recompute the prefix before it can ever be reused."""
    density, ao, weights, direction = _inputs(2)
    with _response(native, density, ao, weights, 4) as response:
        per_point = _cache_bytes(response) // weights.size
        assert response.library.admit_cache(response.handle, 8 * per_point) == 1
        saved = response.ao[0, -1, -1]
        response.ao[0, -1, -1] = np.nan
        output = np.full_like(direction, -9182.5)
        with pytest.raises(RuntimeError, match="nonfinite"):
            response.apply(direction, output)
        np.testing.assert_array_equal(output, -9182.5)
        before = _cache_work(response)
        assert before[1] == 0
        response.ao[0, -1, -1] = saved
        expected = response.apply(direction)
        assert _cache_work(response) == (before[0] + weights.size, 0)
        np.testing.assert_array_equal(response.apply(direction), expected)
        assert _cache_work(response) == (before[0] + 2 * weights.size - 8, 8)


@pytest.mark.parametrize("spins", [1, 2])
def test_cached_primal_vacuum_remains_lazy(native: tuple, spins: int) -> None:
    """Caching never adds a floor or evaluates an inactive singular XC branch."""
    density, ao, weights, direction = _inputs(spins)
    for scale in (0.0, 1e-22):
        with _response(native, scale * density, ao, weights, 4) as response:
            assert (
                response.library.admit_cache(response.handle, _cache_bytes(response))
                == 1
            )
            np.testing.assert_array_equal(response.apply(direction), 0)
            np.testing.assert_array_equal(response.apply(-direction), 0)
            assert _cache_work(response) == (weights.size, weights.size)


def test_primal_cache_planning_is_bounded_without_full_grid_allocation(
    native: tuple,
) -> None:
    """Huge grids and byte allowances retain checked signed-domain capacities."""
    library, _ = native
    maximum = ct.c_size_t(-1).value
    for spins in (1, 2):
        for allowance in (0, 1, 4096, 1 << 20):
            planned = library.cache_query(4, maximum, spins, 4, allowance)
            assert planned <= allowance
        assert library.cache_query(0, 19, spins, 4, 4096) == maximum
        density, ao, weights, _ = _inputs(spins)
        with _response(native, density, ao, weights, 4) as response:
            assert library.admit_cache(response.handle, 1) == 0
            full = _cache_bytes(response)
            assert library.admit_cache(response.handle, full) == 1
            assert library.admit_cache(response.handle, full) == 1


def _batch_directions(direction: np.ndarray, count: int) -> np.ndarray:
    """Include a nonsymmetric ordered direction, an exact zero and a signed mix."""
    return np.array(
        (
            direction,
            np.flip(direction, axis=1),
            np.zeros_like(direction),
            -0.5 * direction,
        )
    )[:count].copy()


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("count", (1, 2, 3, 4))
@pytest.mark.parametrize("tile", (1, 4, 19, 29))
@pytest.mark.parametrize("cached", (False, True))
def test_batch_matches_complete_native_directions_and_exact_bounds(
    native: tuple, spins: int, count: int, tile: int, cached: bool
) -> None:
    """Same-frame sharing preserves every independent output bit and canary."""
    density, ao, weights, direction = _inputs(spins)
    directions = _batch_directions(direction, count)
    with _response(native, density, ao, weights, tile) as response:
        if cached:
            assert (
                response.library.admit_cache(response.handle, _cache_bytes(response))
                == 1
            )
        expected = np.array([response.apply(values) for values in directions])
        before = directions.copy()
        owned = response.library.scratch(response.handle)
        scratch = np.full(response.batch_bytes(count) // 8 + 2, -713.25)
        output = np.full(directions.size + 2, -781.25)
        actual = output[1:-1].reshape(directions.shape)
        response.apply_batch(directions, actual, scratch[1:-1])
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        np.testing.assert_array_equal(output[[0, -1]], -781.25)
        np.testing.assert_array_equal(scratch[[0, -1]], -713.25)
        np.testing.assert_array_equal(directions, before)
        assert response.library.scratch(response.handle) == owned
        inplace = directions.copy()
        response.apply_batch(inplace, inplace)
        np.testing.assert_array_equal(inplace.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("count", (2, 3, 4))
def test_batch_against_independent_multistep_potential_response(
    native: tuple, spins: int, count: int
) -> None:
    """Independent Libxc gradients cover both spins and nonsymmetric tangents."""
    density, ao, weights, direction = _inputs(spins)
    directions = _batch_directions(direction, count)
    if spins == 2:
        directions[0, 1] = 0
        directions[1, 0] = 0
    with _response(native, density, ao, weights, 4) as response:
        actual = response.apply_batch(directions)
    for values, result in zip(directions, actual, strict=True):
        for step in (1e-3, 3e-4, 1e-4):
            reference = sum(
                coefficient
                * _independent_fock(density + offset * step * values, ao, weights)
                for offset, coefficient in ((-2, 1), (-1, -8), (1, 8), (2, -1))
            ) / (12 * step)
            np.testing.assert_allclose(result, reference, atol=3e-9, rtol=3e-8)
    if spins == 2:
        assert np.max(np.abs(actual[0, 1])) > 1e-8
        assert np.max(np.abs(actual[1, 0])) > 1e-8


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("failure", (1, 2, 3, 4, 5))
@pytest.mark.parametrize("cached", (False, True))
def test_batch_failure_is_transactional_and_recovers(
    native: tuple, spins: int, failure: int, cached: bool
) -> None:
    """Expired/reentrant/nonfinite readers cannot publish any successful lane."""
    density, ao, weights, direction = _inputs(spins)
    directions = _batch_directions(direction, 4)
    with _response(native, density, ao, weights, 4) as response:
        if cached:
            assert (
                response.library.admit_cache(response.handle, _cache_bytes(response))
                == 1
            )
        expected = response.apply_batch(directions)
        output = np.full_like(directions, -9182.5)
        response.library.set_failure(response.handle, failure)
        with pytest.raises(RuntimeError):
            response.apply_batch(directions, output)
        np.testing.assert_array_equal(output, -9182.5)
        response.library.set_failure(response.handle, 0)
        np.testing.assert_array_equal(response.apply_batch(directions), expected)


@pytest.mark.parametrize("spins", (1, 2))
def test_batch_capacity_aliases_nonfinite_and_shape_admission(
    native: tuple, spins: int
) -> None:
    """Caller workspace is priced, disjoint and checked before publication."""
    density, ao, weights, direction = _inputs(spins)
    directions = _batch_directions(direction, 4)
    with _response(native, density, ao, weights, 4) as response:
        output = np.full_like(directions, -9182.5)
        needed = response.batch_bytes(4) // 8
        with pytest.raises(RuntimeError, match="workspace capacity"):
            response.apply_batch(directions, output, np.empty(needed - 1))
        storage = np.full(needed + directions.size, -713.25)
        storage[: directions.size] = directions.ravel()
        aliased = storage[: directions.size].reshape(directions.shape)
        with pytest.raises(RuntimeError, match="workspace aliases"):
            response.apply_batch(aliased, output, storage[:needed])
        with pytest.raises(RuntimeError, match="workspace aliases"):
            response.apply_batch(directions, aliased, storage[:needed])
        invalid = directions.copy()
        invalid[-1, -1, -1, -1] = np.nan
        with pytest.raises(RuntimeError, match="nonfinite"):
            response.apply_batch(invalid, output)
        with pytest.raises(RuntimeError, match="matrix shape"):
            response.apply_batch(directions.ravel()[:-1], output, np.empty(needed))
        np.testing.assert_array_equal(output, -9182.5)
        expected = np.array([response.apply(values) for values in directions])
        np.testing.assert_array_equal(response.apply_batch(directions), expected)
    arena = np.full(needed + density.size, -713.25)
    arena[: density.size] = density.ravel()
    borrowed = arena[: density.size].reshape(density.shape)
    with _response(native, borrowed, ao, weights, 4) as response:
        with pytest.raises(RuntimeError, match="workspace aliases"):
            response.apply_batch(directions, workspace=arena[:needed])
        with pytest.raises(RuntimeError, match="reference alias"):
            response.apply_batch(
                directions, arena[: directions.size].reshape(directions.shape)
            )
        np.testing.assert_array_equal(borrowed, density)


@pytest.mark.parametrize("spins", (1, 2))
def test_batch_partial_cache_preparation_and_mutation_recovery(
    native: tuple, spins: int
) -> None:
    """A failed first batch cannot reuse a partial bank; later hits stay live."""
    density, ao, weights, direction = _inputs(spins)
    directions = _batch_directions(direction, 3)
    with _response(native, density, ao, weights, 4) as response:
        assert (
            response.library.admit_cache(response.handle, _cache_bytes(response)) == 1
        )
        saved = response.ao[0, -1, -1]
        response.ao[0, -1, -1] = np.nan
        output = np.full_like(directions, -9182.5)
        with pytest.raises(RuntimeError, match="nonfinite"):
            response.apply_batch(directions, output)
        np.testing.assert_array_equal(output, -9182.5)
        assert _cache_work(response)[1] == 0
        response.ao[0, -1, -1] = saved
        expected = response.apply_batch(directions)
        assert _cache_work(response)[1] == 0
        for values, index in ((response.density, (0, 0, 0)), (response.ao, (0, 0, 0))):
            previous = values[index]
            values[index] += 0.01
            with pytest.raises(RuntimeError, match="cached point inputs changed"):
                response.apply_batch(directions, output)
            np.testing.assert_array_equal(output, -9182.5)
            values[index] = previous
            np.testing.assert_array_equal(response.apply_batch(directions), expected)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("scale", (0.0, 1e-22, 1e-12, 1e-8))
def test_batch_retains_lazy_vacuum_branch(
    native: tuple, spins: int, scale: float
) -> None:
    """No shared-field floor or inverse-power evaluation may change admission."""
    density, ao, weights, direction = _inputs(spins)
    directions = _batch_directions(direction, 4)
    with _response(native, scale * density, ao, weights, 4) as response:
        expected = np.array([response.apply(values) for values in directions])
        assert (
            response.library.admit_cache(response.handle, _cache_bytes(response)) == -1
        )
        actual = response.apply_batch(directions)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_batch_planning_bounds_and_repeated_provider_work(native: tuple) -> None:
    """Grid size is streamed, direction work stays explicit, overflows fail closed."""
    library, _ = native
    maximum = ct.c_size_t(-1).value
    for spins in (1, 2):
        for count in (1, 2, 3, 4):
            planned = library.batch_query(4, 1000, spins, 4, count)
            assert planned == library.batch_query(4, maximum, spins, 4, count)
            assert (planned == 0) == (count == 1)
        for count in (0, 5, maximum):
            assert library.batch_query(4, 19, spins, 4, count) == maximum
        assert library.batch_query(maximum, 19, spins, 4, 4) == maximum
        density, ao, weights, direction = _inputs(spins)
        with _response(native, density, ao, weights, 4) as response:
            response.apply_batch(_batch_directions(direction, 4))
            work = (ct.c_size_t * 4)()
            library.work(response.handle, work)
            assert work[0] == work[1] == 4
            assert work[2] == 4 * weights.size
            assert work[3] == (weights.size + 3) // 4
