"""Explicit U1 science and actual CUDA launchers, not endpoint admission."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.common.compiler_cache import (
    resolve_compiler_cache,
    run_cached_compiler,
)
from generativeqc_compiler.common.compiler_process import run_compiler
from generativeqc_compiler.common.cuda_resources import parse_resources
from generativeqc_compiler.common.provenance import file_hash
from generativeqc_compiler.dft.xc_point_cuda import emit_device_point_value
from generativeqc_compiler.xc.automatic_cuda import (
    emit_automatic_cuda_descriptor,
    emit_automatic_cuda_provider,
)
from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.bulk_runtime import build_bulk_runtime_program
from generativeqc_compiler.xc.density_work import (
    DENSITY_WORK_DOMAIN,
    DENSITY_WORK_DOMAIN_VERSION,
)
from generativeqc_compiler.xc.libxc_work import automatic_work_policy
from test_libxc_automatic_point_native import NAMES as ALL_NAMES
from test_libxc_automatic_point_native import _inputs, _oracle

if TYPE_CHECKING:
    from generativeqc_compiler.xc.automatic_cuda import AutomaticCudaPointProvider

NAMES = ("LDA_C_VWN_4", "LDA_C_CHACHIYO", "LDA_C_PZ_MOD")


def density_inputs(name: str) -> np.ndarray:
    """Probe both sides of screening and the distinct U1/P2 floor interval."""
    threshold = automatic_work_policy(name).density_threshold
    return np.array(
        [
            0.0,
            np.nextafter(0.0, 1.0),
            0.25 * threshold,
            np.nextafter(threshold, 0.0),
            threshold,
            np.nextafter(threshold, np.inf),
            1.1 * threshold,
            np.nextafter(2.0 * threshold, 0.0),
            2.0 * threshold,
            np.nextafter(2.0 * threshold, np.inf),
            0.62,
            12.0,
        ]
    )


@pytest.mark.parametrize("name", NAMES)
def test_restricted_graph_is_explicit_and_backend_neutral(name: str) -> None:
    program = build_bulk_runtime_program(
        name, spin="unpolarized", domain=DENSITY_WORK_DOMAIN
    )
    cpu = emit_automatic_point(name, spin="unpolarized")
    cuda = emit_automatic_point(name, backend="cuda", spin="unpolarized")
    provider = emit_automatic_cuda_provider(name)
    assert (
        cpu.expression_identity == cuda.expression_identity == program.expression_hash
    )
    assert cpu.mathematical_contract == cuda.mathematical_contract
    assert cpu.source_identity != cuda.source_identity
    assert provider.restricted == cuda
    assert provider.point.spin == "polarized"
    assert provider.point.domain_version == DENSITY_WORK_DOMAIN_VERSION
    assert provider.point.mathematical_contract != cuda.mathematical_contract
    assert provider.expression_identity not in (
        provider.point.expression_identity,
        cuda.expression_identity,
    )
    assert provider == emit_automatic_cuda_provider(name)


@pytest.mark.parametrize(
    "changes",
    [
        {"spin": "polarized"},
        {"backend": "cpu"},
        {"features": ("rho_a", "rho_b")},
        {"outputs": ((), (0,), (1,))},
        {"domain_version": 4},
        {"domain": "libxc-7.0/work-semilocal-v1"},
        {"ingredient_mask": 7},
        {"name": "LDA_C_PZ_MOD"},
    ],
)
def test_provider_rejects_stale_restricted_contract(changes: dict) -> None:
    provider = emit_automatic_cuda_provider(NAMES[0])
    with pytest.raises(ValueError, match="restricted layout"):
        replace(provider, restricted=replace(provider.restricted, **changes))


def test_provider_identity_binds_both_layouts_and_exact_source() -> None:
    provider = emit_automatic_cuda_provider(NAMES[0])
    for field in ("point", "restricted"):
        point = getattr(provider, field)
        changed = replace(
            provider, **{field: replace(point, expression_identity="changed-import")}
        )
        assert changed.expression_identity != provider.expression_identity
        assert changed.source_identity != provider.source_identity
        changed = replace(
            provider, **{field: replace(point, source=point.source + "\n")}
        )
        assert changed.expression_identity == provider.expression_identity
        assert changed.source_identity != provider.source_identity
    changed = replace(provider, source=provider.source + "\n")
    assert changed.expression_identity == provider.expression_identity
    assert changed.source_identity != provider.source_identity


def test_provider_rejects_joint_stale_domain_and_invalid_tile_contract() -> None:
    provider = emit_automatic_cuda_provider(NAMES[0])
    for changes in ({"domain_version": 4}, {"domain": "stale-worker"}):
        with pytest.raises(ValueError, match="restricted layout"):
            replace(
                provider,
                point=replace(provider.point, **changes),
                restricted=replace(provider.restricted, **changes),
            )
    for changes in (
        {"feature_terms": 4},
        {"point": replace(provider.point, outputs=((),))},
    ):
        with pytest.raises(ValueError, match="restricted layout"):
            replace(provider, **changes)


def test_unpolarized_lowering_does_not_claim_other_ingredients() -> None:
    with pytest.raises(ValueError, match="density-only"):
        emit_automatic_point("GGA_X_APBE", spin="unpolarized")
    with pytest.raises(ValueError, match="polarized derivative"):
        emit_automatic_point(
            NAMES[0], backend="cuda", spin="unpolarized", equal_spin=True
        )


@pytest.mark.parametrize("name", NAMES)
def test_actual_restricted_scalar_matches_independent_u1(
    name: str, tmp_path: Path, native_cxx: object
) -> None:
    """Execute CUDA scalar bytes on the host; GPU traversal has a separate gate."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    assert libxc.__version__ == "7.0.0"
    point = emit_automatic_point(name, backend="cuda", spin="unpolarized")
    source = tmp_path / "restricted.cpp"
    source.write_text(
        "#include <cmath>\n#include <cstddef>\nusing std::isfinite;\n"
        + emit_device_point_value()
        + point.source.replace("__device__ ", "")
        + f"""
extern "C" void evaluate(const double* density, double* output, std::size_t count) {{
  for (std::size_t index = 0; index < count; ++index) {{
    const double rho[2]{{0.5 * density[index], 0.5 * density[index]}};
    const double gradient[2][3]{{}}, tau[2]{{}};
    const auto value = {point.function_name}(rho, gradient, tau);
    output[3 * index] = value.energy;
    output[3 * index + 1] = 0.5 * (value.rho[0] + value.rho[1]);
    output[3 * index + 2] = value.valid ? 1.0 : 0.0;
  }}
}}
extern "C" bool accepts_pair(double first, double second) {{
  const double rho[2]{{first, second}}, gradient[2][3]{{}}, tau[2]{{}};
  return {point.function_name}(rho, gradient, tau).valid;
}}
"""
    )
    binary = native_cxx.build_shared([source], tmp_path / "restricted.so")
    library = ctypes.CDLL(str(binary))
    pointer = ctypes.POINTER(ctypes.c_double)
    library.evaluate.argtypes = (pointer, pointer, ctypes.c_size_t)
    library.evaluate.restype = None
    rho = density_inputs(name)
    observed = np.empty((len(rho), 3))
    library.evaluate(
        rho.ctypes.data_as(pointer), observed.ctypes.data_as(pointer), len(rho)
    )
    energy, vxc, _, _ = libxc.eval_xc(name, rho, spin=0, deriv=1)
    np.testing.assert_array_equal(observed[:, 2], 1.0)
    np.testing.assert_allclose(observed[:, 0], rho * energy, rtol=2e-10, atol=1e-30)
    np.testing.assert_allclose(observed[:, 1], vxc[0], rtol=2e-10, atol=2e-10)
    library.accepts_pair.argtypes = (ctypes.c_double, ctypes.c_double)
    library.accepts_pair.restype = ctypes.c_bool
    for first, second in ((0.3, 0.2), (-1.0, -1.0), (np.nan, np.nan), (np.inf, np.inf)):
        assert not library.accepts_pair(first, second)


def _provider_probe(provider: AutomaticCudaPointProvider) -> str:
    """Wrap complete emitted launchers without substituting the point or traversal."""
    namespace = provider.namespace
    return (
        "#include <vector>\n"
        + provider.source
        + emit_automatic_cuda_descriptor(provider)
        + f"""
extern "C" unsigned domain_version() {{ return {namespace}::program.domain_version; }}
extern "C" const char* expression_identity() {{ return {namespace}::program.expression_identity; }}
extern "C" const char* source_identity() {{ return {namespace}::program.source_identity; }}
extern "C" int probe(const double* features, const double* weights,
                     double* coefficients, double* totals, std::size_t count,
                     std::size_t spins, std::size_t tile_points) {{
  const std::size_t channels = spins * {provider.feature_terms};
  double *device_features{{}}, *device_weights{{}}, *device_coefficients{{}}, *device_totals{{}};
  int *device_error{{}};
  cudaError_t status = cudaMalloc(&device_features, count * channels * sizeof(double));
  if (status == cudaSuccess) status = cudaMalloc(&device_weights, count * sizeof(double));
  if (status == cudaSuccess) status = cudaMalloc(&device_coefficients, count * channels * sizeof(double));
  if (status == cudaSuccess) status = cudaMalloc(&device_totals, 3 * count * sizeof(double));
  if (status == cudaSuccess) status = cudaMalloc(&device_error, sizeof(int));
  if (status == cudaSuccess) status = cudaMemset(device_error, 0, sizeof(int));
  if (status == cudaSuccess) status = cudaMemcpy(device_features, features, count * channels * sizeof(double), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) status = cudaMemcpy(device_weights, weights, count * sizeof(double), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {{
    const auto& program = {namespace}::program;
    if (tile_points)
      program.launch_batch(nullptr, device_features, device_weights, count, spins,
          device_coefficients, device_totals, device_error, program.functional_code, 1, 1, tile_points);
    else
      program.launch(nullptr, device_features, device_weights, count, spins,
          device_coefficients, device_totals, device_error, program.functional_code, 1, 1, nullptr);
    status = cudaGetLastError();
  }}
  if (status == cudaSuccess) status = cudaMemcpy(coefficients, device_coefficients, count * channels * sizeof(double), cudaMemcpyDeviceToHost);
  if (status == cudaSuccess) status = cudaMemcpy(totals, device_totals, 3 * count * sizeof(double), cudaMemcpyDeviceToHost);
  int error = 0;
  if (status == cudaSuccess) status = cudaMemcpy(&error, device_error, sizeof(int), cudaMemcpyDeviceToHost);
  if (device_error) cudaFree(device_error);
  if (device_totals) cudaFree(device_totals);
  if (device_coefficients) cudaFree(device_coefficients);
  if (device_weights) cudaFree(device_weights);
  if (device_features) cudaFree(device_features);
  return status == cudaSuccess ? -error : static_cast<int>(status);
}}
extern "C" bool rejects_binding(unsigned spins, unsigned code, double exchange,
                                 double correlation, unsigned tile, bool response) {{
  const auto& program = {namespace}::program;
  const double delta = 0;
  try {{
    if (tile == 0)
      program.launch(nullptr, nullptr, nullptr, 0, spins, nullptr, nullptr,
          nullptr, code, exchange, correlation, response ? &delta : nullptr);
    else
      program.launch_batch(nullptr, nullptr, nullptr, 0, spins, nullptr, nullptr,
          nullptr, code, exchange, correlation, tile == 1 ? 0 : tile);
  }} catch (const std::invalid_argument&) {{ return true; }}
  return false;
}}
"""
    )


@pytest.fixture(scope="module", params=ALL_NAMES)
def actual_cuda_provider(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[AutomaticCudaPointProvider, ctypes.CDLL, Path]:
    if os.environ.get("GENERATIVEQC_XC_CUDA_TEST") != "1":
        pytest.skip("opt-in actual-provider Slurm CUDA gate")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU gate requires srun"
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "preserve Slurm GPU visibility"
    provider = emit_automatic_cuda_provider(str(request.param))
    directory = tmp_path_factory.mktemp(f"provider-{provider.point.name}")
    source = directory / "provider.cu"
    source.write_text(_provider_probe(provider))
    compiler = shutil.which(os.environ.get("GENERATIVEQC_NVCC", "nvcc"))
    assert compiler is not None, "opt-in CUDA gate requires NVCC"
    version = run_compiler(
        [compiler, "--version"], 30, label="provider compiler version"
    )
    assert version.returncode == 0 and not version.timed_out
    launcher = resolve_compiler_cache()
    root = Path(__file__).resolve().parents[2]
    compiled = run_cached_compiler(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-arch=sm_120",
            "--fmad=false",
            "-Xcompiler=-fPIC",
            "-Xptxas=-v",
            "-I",
            str(root / "src"),
            "-I",
            str(root / "include"),
            "-c",
            str(source),
            "-o",
            str(directory / "provider.o"),
        ],
        180,
        label="actual automatic CUDA provider",
    )
    (directory / "compile.log").write_text(compiled.stdout + compiled.stderr)
    assert compiled.returncode == 0 and not compiled.timed_out, compiled.stderr
    binary = directory / "provider.so"
    linked = run_compiler(
        [compiler, "-shared", str(directory / "provider.o"), "-o", str(binary)],
        60,
        label="actual automatic CUDA provider link",
    )
    (directory / "link.log").write_text(linked.stdout + linked.stderr)
    assert linked.returncode == 0 and not linked.timed_out, linked.stderr
    resources = parse_resources(compiled.stderr)
    point_resources = tuple(
        resource
        for resource in resources
        if provider.provider_name in resource.function
    )
    assert len(point_resources) == 4
    assert all(0 < resource.registers <= 255 for resource in resources)
    assert all(resource.stack_bytes <= 4096 for resource in resources)
    assert all(
        resource.spill_load_bytes + resource.spill_store_bytes <= 4096
        for resource in resources
    )
    library = ctypes.CDLL(str(binary))
    pointer = ctypes.POINTER(ctypes.c_double)
    library.probe.argtypes = (
        pointer,
        pointer,
        pointer,
        pointer,
        *([ctypes.c_size_t] * 3),
    )
    library.probe.restype = ctypes.c_int
    library.domain_version.restype = ctypes.c_uint
    assert library.domain_version() == provider.point.domain_version
    for field in ("expression_identity", "source_identity"):
        getattr(library, field).restype = ctypes.c_char_p
        assert getattr(library, field)().decode() == getattr(provider, field)
    (directory / "artifact.json").write_text(
        json.dumps(
            {
                "claim": "actual-provider-point-gate-not-KS-or-force-admission",
                "polarized": provider.point.mathematical_contract,
                "restricted": provider.restricted.mathematical_contract,
                "expression_identity": provider.expression_identity,
                "source_identity": provider.source_identity,
                "source_sha256": file_hash(source),
                "object_sha256": file_hash(directory / "provider.o"),
                "binary_sha256": file_hash(binary),
                "compiler_version": version.stdout,
                "cache": {"path": str(launcher.path), "version": launcher.version},
                "resources": [asdict(resource) for resource in resources],
                "point_resources": [asdict(resource) for resource in point_resources],
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            },
            indent=2,
        )
    )
    return provider, library, directory


def _pack_tiles(rows: np.ndarray, tile: int) -> np.ndarray:
    """Match the compact final partial tile, not padded channel strides."""
    tile = tile or len(rows)
    return np.concatenate(
        [rows[begin : begin + tile].T.ravel() for begin in range(0, len(rows), tile)]
    )


def _unpack_totals(packed: np.ndarray, count: int, tile: int) -> np.ndarray:
    """Separate energy and physical-spin counts without assuming full tiles."""
    tile = tile or count
    return np.concatenate(
        [
            packed[3 * begin : 3 * min(begin + tile, count)].reshape(3, -1).T
            for begin in range(0, count, tile)
        ]
    )


@pytest.mark.parametrize("spin", (0, 1))
@pytest.mark.parametrize("knob", ("0", "1"))
@pytest.mark.parametrize("tile", (0, 1, 7, 16, 64))
def test_actual_cuda_provider_matches_independent_spin_contract(
    actual_cuda_provider: tuple[AutomaticCudaPointProvider, ctypes.CDLL, Path],
    spin: int,
    knob: str,
    tile: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, library, directory = actual_cuda_provider
    name, width = provider.point.name, provider.feature_terms
    inputs = _inputs(name, spin)
    if width == 1:
        boundary = np.zeros((len(density_inputs(name)), 10))
        boundary[:, 0] = boundary[:, 1] = 0.5 * density_inputs(name)
        inputs = np.concatenate((inputs, boundary))
    oracle = _oracle(name, inputs, spin)
    spins = spin + 1
    features = np.empty((len(inputs), spins * width))
    expected_coefficients = np.empty_like(features)
    for channel in range(spins):
        features[:, channel * width] = (
            inputs[:, channel] if spin else inputs[:, :2].sum(axis=1)
        )
        expected_coefficients[:, channel * width] = oracle[:, 1 + channel]
        if width >= 4:
            gradient = inputs[:, 2:8].reshape(-1, 2, 3)
            features[:, channel * width + 1 : channel * width + 4] = (
                gradient[:, channel] if spin else gradient.sum(axis=1)
            )
            expected_coefficients[:, channel * width + 1 : channel * width + 4] = (
                oracle[:, 3 + channel * 3 : 6 + channel * 3] if spin else oracle[:, 2:5]
            )
        if width == 5:
            features[:, channel * width + 4] = (
                inputs[:, 8 + channel] if spin else inputs[:, 8:].sum(axis=1)
            )
            expected_coefficients[:, channel * width + 4] = (
                oracle[:, 9 + channel] if spin else oracle[:, 5]
            )
    packed = np.ascontiguousarray(_pack_tiles(features, tile))
    weights = np.linspace(0.125, 1.25, len(inputs))
    coefficients = np.empty_like(packed)
    totals = np.empty(3 * len(inputs))
    pointer = ctypes.POINTER(ctypes.c_double)
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", knob)
    status = library.probe(
        packed.ctypes.data_as(pointer),
        weights.ctypes.data_as(pointer),
        coefficients.ctypes.data_as(pointer),
        totals.ctypes.data_as(pointer),
        len(inputs),
        spins,
        tile,
    )
    assert status == 0, (name, spin, knob, tile, status)
    expected_totals = (
        np.column_stack((oracle[:, 0], inputs[:, 0], inputs[:, 1])) * weights[:, None]
    )
    expected_totals = _pack_tiles(expected_totals, tile)
    expected_coefficients = _pack_tiles(expected_coefficients, tile)
    (directory / f"spin-{spin}-knob-{knob}-tile-{tile}.json").write_text(
        json.dumps(
            {
                "name": name,
                "spin": spin,
                "knob": knob,
                "tile": tile,
                "features": packed.tolist(),
                "weights": weights.tolist(),
                "observed_coefficients": coefficients.tolist(),
                "reference_coefficients": expected_coefficients.tolist(),
                "observed_totals": totals.tolist(),
                "reference_totals": expected_totals.tolist(),
                "coefficient_error": float(
                    np.max(np.abs(coefficients - expected_coefficients))
                ),
                "totals_error": float(np.max(np.abs(totals - expected_totals))),
            },
            indent=2,
        )
    )
    np.testing.assert_allclose(
        coefficients, expected_coefficients, rtol=2e-10, atol=2e-10
    )
    observed_rows = _unpack_totals(totals, len(inputs), tile)
    expected_rows = _unpack_totals(expected_totals, len(inputs), tile)
    np.testing.assert_allclose(
        observed_rows[:, 1:], expected_rows[:, 1:], rtol=2e-10, atol=1e-30
    )
    # Preserve the original GGA/MGGA point gate; cancellation can make energy
    # essentially zero while its separate terms remain appreciable. Density
    # work additionally needs a strict absolute gate on its active tail energy.
    np.testing.assert_allclose(
        observed_rows[:, 0],
        expected_rows[:, 0],
        rtol=2e-10,
        atol=1e-30 if width == 1 else 2e-10,
    )


def test_actual_cuda_provider_rejects_invalid_bindings_and_inputs(
    actual_cuda_provider: tuple[AutomaticCudaPointProvider, ctypes.CDLL, Path],
) -> None:
    provider, library, _ = actual_cuda_provider
    from generativeqc_compiler.xc.automatic_semilocal import automatic_functional_code

    code = automatic_functional_code(provider.point.name)
    library.rejects_binding.argtypes = (
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint,
        ctypes.c_bool,
    )
    library.rejects_binding.restype = ctypes.c_bool
    for tile in (0, 2):
        assert not library.rejects_binding(1, code, 1, 1, tile, False)
        for spins, selector, exchange, correlation in (
            (0, code, 1, 1),
            (3, code, 1, 1),
            (1, code + 1, 1, 1),
            (1, code, 0.5, 1),
            (1, code, 1, 0.5),
        ):
            assert library.rejects_binding(
                spins, selector, exchange, correlation, tile, False
            )
    assert library.rejects_binding(1, code, 1, 1, 1, False)
    assert library.rejects_binding(1, code, 1, 1, 0, True)
    pointer = ctypes.POINTER(ctypes.c_double)
    for density in (-1.0, np.nan, np.inf):
        features = np.zeros(provider.feature_terms)
        features[0] = density
        weights, coefficients, totals = np.ones(1), np.empty_like(features), np.empty(3)
        for tile in (0, 1):
            assert (
                library.probe(
                    features.ctypes.data_as(pointer),
                    weights.ctypes.data_as(pointer),
                    coefficients.ctypes.data_as(pointer),
                    totals.ctypes.data_as(pointer),
                    1,
                    1,
                    tile,
                )
                == -3
            )
