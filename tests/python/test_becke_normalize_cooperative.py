"""Serial AD, independent NumPy and cooperative CUDA normalization gates."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.method.stationary_becke_phased import (
    emit_stationary_phased_becke_cuda,
)
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials

HARNESS = r"""
extern "C" __global__ void normalize_test(size_t atoms, size_t points,
    double* fields, size_t* zeros, double* maximum, const int64_t* owners,
    double* seeds, int* error, int cooperative) {
  using namespace generativeqc_stationary_cuda;
  PhasedBeckeInput input{};
  input.work = {atoms, points, nullptr, fields, zeros, maximum};
  input.owners = owners;
  input.seeds = seeds;
  input.error = error;
  if (cooperative) normalize_cooperative(input);
  else {
    const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point < points && !generativeqc_grid_phased::point_normalize_phase(
            input.work, point, input.owner(point), seeds[point], local_ratio))
      atomicExch(error, 1);
  }
}
"""


def test_normalize_emission_bounds_and_serial_control() -> None:
    """Authenticated domains retain the constant shared-memory bound."""
    source = emit_stationary_phased_becke_cuda(128)
    assert "stationary_becke_normalize_max_atoms = 128;" in source
    assert "stationary_becke_cooperative_normalize = true;" in source
    control = emit_stationary_phased_becke_cuda(cooperative_normalize=False)
    assert "stationary_becke_cooperative_normalize = false;" in control
    with pytest.raises(ValueError, match="selection must be boolean"):
        emit_stationary_phased_becke_cuda(cooperative_normalize=1)
    with pytest.raises(ValueError, match="atom limit"):
        emit_stationary_phased_becke_cuda(129)


@pytest.mark.parametrize("selection", [0, 1, "on"])
def test_normalize_control_rejects_non_boolean_before_allocation(
    selection: Any,
) -> None:
    """A malformed qualification control cannot create a native owner."""
    from generativeqc._stationary_cuda import _CudaSources

    with pytest.raises(TypeError, match="normalization selection"):
        _CudaSources(
            None, None, None, None, None, None, None, becke_normalize=selection
        )


def test_normalize_explicit_control_requires_versioned_abi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Legacy artifacts fail explicit requests before their allocation ABI."""
    from generativeqc import _stationary_cuda as runtime

    artifact = SimpleNamespace(
        library=tmp_path / "legacy.so", metadata={"binary_sha256": "legacy"}
    )
    monkeypatch.setattr(runtime, "file_hash", lambda _path: "legacy")
    monkeypatch.setattr(runtime.ct, "CDLL", lambda _path: SimpleNamespace())
    with pytest.raises(ValueError, match="lacks normalization schedule control"):
        runtime._CudaSources(
            SimpleNamespace(natom=96),
            artifact,
            None,
            None,
            None,
            None,
            None,
            becke_normalize=True,
        )


@pytest.fixture
def physical_normalizer(monkeypatch: pytest.MonkeyPatch, request: Any) -> Any:
    """Check both requests against the unchanged small-domain serial fallback.

    These physical oracles have three atoms. The native phased reservation is
    intentionally restricted to domains above 32 atoms, so this must not be
    labeled physical qualification of cooperative normalization. The admitted
    schedule is independently covered by CUDA and complete 48/96-atom gates.
    """
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("real-GPU gates require a finite Slurm allocation")
    import ctypes as ct

    from generativeqc import _stationary_cuda as runtime
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    original = runtime._CudaSources.__init__
    selected = request.param
    observed = []
    if directory := os.environ.get("GENERATIVEQC_STATIONARY_EVIDENCE"):
        monkeypatch.setenv(
            "GENERATIVEQC_STATIONARY_EVIDENCE",
            str(Path(directory) / ("candidate" if selected else "baseline")),
        )

    def initialize(source: Any, *values: Any, **options: Any) -> None:
        options["phased_becke"] = True
        options["becke_normalize"] = selected
        original(source, *values, **options)
        function = source.library.stationary_becke_normalize_metrics_v1
        function.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64), ct.c_size_t]
        output = (ct.c_uint64 * 4)()
        assert function(source.handle, output, 4) == 0
        assert source.natom == 3 and source.resources.phased_becke_bytes == 0
        assert tuple(output[:2]) == (0, 0)
        observed.append(selected)

    monkeypatch.setattr(runtime._CudaSources, "__init__", initialize)
    yield CudaCompilerAdapter(
        Path(os.environ["CUDACXX"]), cuda_target_info("sm_120"), compile_timeout=600
    )
    assert observed, "the physical gate must exercise the controlled native owner"


@pytest.mark.parametrize("physical_normalizer", [False, True], indirect=True)
@pytest.mark.parametrize("spin", ["rks", "uks"])
def test_physical_normalize_fallback_independent_gradient(
    physical_normalizer: Any, spin: str
) -> None:
    """Reuse analytic oracles, stale/replayed geometry and CPU-fallback guards."""
    import test_dft_complete_cuda as complete

    if spin == "rks":
        complete.test_complete_cuda_independent_analytic(
            "pbe-rks", "water", physical_normalizer
        )
    else:
        complete.test_complete_cuda_open_shell_uks_independent_analytic(
            "pbe-uks", physical_normalizer
        )


@pytest.mark.parametrize("physical_normalizer", [True], indirect=True)
def test_physical_normalize_fallback_reconverged_finite_difference(
    physical_normalizer: Any,
) -> None:
    """Check the three-step directional curve and reconverged replay contract."""
    import test_dft_complete_cuda as complete

    complete.test_cuda_reconverged_finite_differences_and_replay(
        "pbe-rks", physical_normalizer
    )


@pytest.fixture(scope="module")
def cuda_normalizer() -> tuple[Any, Any]:
    """Compilation/execution requires scheduler-assigned device visibility."""
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("real-GPU gates require a finite Slurm allocation")
    cupy = pytest.importorskip("cupy")
    source = (
        emit_grid_adjoint()
        + emit_grid_partials(3, device=True)
        + emit_stationary_phased_becke_cuda()
        + HARNESS
    )
    module = cupy.RawModule(
        code=source,
        backend="nvcc",
        options=("-std=c++17", "--expt-relaxed-constexpr", "--fmad=false"),
    )
    return cupy, module.get_function("normalize_test")


@pytest.mark.parametrize("atoms", [1, 3, 33, 48, 96, 128, 129])
@pytest.mark.parametrize("points", [1, 17, 32, 33, 257])
def test_cuda_normalize_matches_serial_and_independent_formula(
    cuda_normalizer: tuple[Any, Any], atoms: int, points: int
) -> None:
    """Cover partial blocks, exact-zero products/owners and dense fallback."""
    cupy, kernel = cuda_normalizer
    random = np.random.default_rng(1894 + atoms + points)
    logs = random.uniform(-750, 20, (atoms, points))
    zeros = random.integers(0, 3, (atoms, points), dtype=np.uint64)
    zeros[0] = 0
    owners = random.integers(0, atoms, points, dtype=np.int64)
    seeds = random.normal(size=points)
    maximum = np.max(np.where(zeros == 0, logs, -np.inf), axis=0)
    products = np.exp(np.where(zeros == 0, logs - maximum, -np.inf))
    total = np.zeros(points)
    for row in products:
        total += row
    objective_owner = products[owners, np.arange(points)]
    bars = np.broadcast_to(-objective_owner / (total * total), products.shape).copy()
    bars[owners, np.arange(points)] += 1 / total
    bars *= seeds
    results = []
    for cooperative in (0, 1):
        fields = np.full((12, atoms, points), np.nan)
        fields[4] = logs
        device_fields = cupy.asarray(fields)
        device_zeros = cupy.asarray(zeros)
        device_maximum = cupy.full(points, cupy.nan)
        device_owners = cupy.asarray(owners)
        device_seeds = cupy.asarray(seeds)
        error = cupy.zeros(1, dtype=cupy.int32)
        width = 8 if cooperative else 128
        kernel(
            ((points + width - 1) // width,),
            (8, 16) if cooperative else (128,),
            (
                np.uint64(atoms),
                np.uint64(points),
                device_fields,
                device_zeros,
                device_maximum,
                device_owners,
                device_seeds,
                error,
                np.int32(cooperative),
            ),
        )
        assert int(error.get()[0]) == 0
        result = device_fields.get()[5:7]
        np.testing.assert_array_equal(device_maximum.get(), maximum)
        np.testing.assert_allclose(result[0], products, rtol=3e-14, atol=1e-300)
        np.testing.assert_allclose(result[1], bars, rtol=3e-14, atol=3e-14)
        results.append(result)
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("invalid", ["all-zero", "owner", "seed", "sticky"])
def test_cuda_normalize_failure_is_sticky(
    cuda_normalizer: tuple[Any, Any], invalid: str
) -> None:
    """No inactive point skips a block barrier or clears an existing error."""
    cupy, kernel = cuda_normalizer
    atoms, points = 96, 33
    fields = cupy.zeros((12, atoms, points))
    zeros = cupy.zeros((atoms, points), dtype=cupy.uint64)
    owners = cupy.zeros(points, dtype=cupy.int64)
    seeds = cupy.ones(points)
    error = cupy.array([int(invalid == "sticky")], dtype=cupy.int32)
    if invalid == "all-zero":
        zeros[:, 0] = 1
    elif invalid == "owner":
        owners[0] = atoms
    elif invalid == "seed":
        seeds[0] = cupy.nan
    kernel(
        (5,),
        (8, 16),
        (
            np.uint64(atoms),
            np.uint64(points),
            fields,
            zeros,
            cupy.zeros(points),
            owners,
            seeds,
            error,
            np.int32(1),
        ),
    )
    assert int(error.get()[0]) == 1
