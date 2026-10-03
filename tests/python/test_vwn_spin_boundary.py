"""Independent VWN closed-form checks at empty and nearly empty spin channels.

The oracle evaluates the published rs/spin interpolation at high precision;
it does not call the Graph, Maple importer or native implementation. Only
energy and first derivatives are tested at zero minority density, where a
second density derivative need not be finite.
"""

import ctypes
import os
import shutil
import subprocess
import typing
from collections.abc import Callable
from fractions import Fraction
from functools import cache

import numpy as np
import pytest
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.integral.cuda import CudaEmitter
from generativeqc_compiler.integral.expr import Expr, Graph
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter
from generativeqc_compiler.xc.expression_dispatch import build_energy_expression
from generativeqc_compiler.xc.spec import FunctionalSpec

mp = pytest.importorskip("mpmath", reason="high-precision independent VWN oracle")
Evaluator = Callable[[str, float, float], np.ndarray]


def _closed_form(name: str, alpha: typing.Any, beta: typing.Any) -> typing.Any:
    """Original VWN Hartree parameters and interpolation, evaluated with mpmath."""
    density = alpha + beta
    rs = (3 / (4 * mp.pi * density)) ** (mp.mpf(1) / 3)
    x = mp.sqrt(rs)
    if name == "LDA_C_VWN_RPA":
        parameters = (
            ("13.0720", "42.7198", "-0.409286"),
            ("20.1231", "101.578", "-0.743294"),
        )
    else:
        parameters = (
            ("3.72744", "12.9352", "-0.10498"),
            ("7.06042", "18.0578", "-0.32500"),
            ("1.13107", "13.0045", "-0.0047584"),
        )
    amplitudes = (mp.mpf("0.0310907"), mp.mpf("0.01554535"), -1 / (6 * mp.pi**2))
    values = []
    for amplitude, raw in zip(amplitudes[: len(parameters)], parameters, strict=True):
        b, c, x0 = map(mp.mpf, raw)
        q = mp.sqrt(4 * c - b * b)
        polynomial = rs + b * x + c
        offset = b * x0 / (x0 * x0 + b * x0 + c)
        values.append(
            amplitude
            * (
                mp.log(rs / polynomial)
                + (2 * b / q - offset * 2 * (2 * x0 + b) / q) * mp.atan(q / (2 * x + b))
                - offset * mp.log((x - x0) ** 2 / polynomial)
            )
        )
    exponent = mp.mpf(4) / 3
    spin = (
        (2 * alpha / density) ** exponent + (2 * beta / density) ** exponent - 2
    ) / (2**exponent - 2)
    if name == "LDA_C_VWN_RPA":
        return density * (values[0] * (1 - spin) + values[1] * spin)
    zeta = (alpha - beta) / density
    curvature = 4 / (9 * (2 ** (mp.mpf(1) / 3) - 1))
    return density * (
        values[0]
        + values[2] * spin * (1 - zeta**4) / curvature
        + (values[1] - values[0]) * spin * zeta**4
    )


@cache
def _graph(name: str) -> tuple[Graph, tuple[Expr, ...]]:
    """Build the production adapter, including its derivatives, once per family."""
    spec = FunctionalSpec(name, ((name, Fraction(1)),), spin="polarized")
    graph, energy, variables = build_energy_expression(spec)
    roots = (energy, *(graph.differentiate(energy, x) for x in variables[:2]))
    return graph, roots


def _actual(name: str, alpha: float, beta: float) -> np.ndarray:
    graph, roots = _graph(name)
    inputs = {
        "rho_a": alpha,
        "rho_b": beta,
        "sigma_aa": 0.0,
        "sigma_ab": 0.0,
        "sigma_bb": 0.0,
        "tau_a": 0.0,
        "tau_b": 0.0,
    }
    return np.asarray(evaluate_array_graph(graph, roots, inputs), dtype=float)


@pytest.fixture(scope="module", params=("graph", "scalar_c", "cuda"))
def evaluate(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Evaluator:
    """Check DAG, optimized host arithmetic and opt-in Slurm CUDA arithmetic."""
    if request.param == "graph":
        return _actual
    cuda = request.param == "cuda"
    if cuda and os.environ.get("GENERATIVEQC_XC_CUDA_TEST") != "1":
        pytest.skip("opt-in Slurm CUDA gate")
    if cuda and not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("CUDA qualification requires a finite Slurm GPU allocation")
    compiler, cache = shutil.which("nvcc" if cuda else "c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("emitted arithmetic requires ccache and a C++ compiler")
    subprocess.run([cache, "--version"], check=True, capture_output=True, timeout=10)
    folder = tmp_path_factory.mktemp("vwn_spin_boundary")
    source = ["#include <cmath>\n", "#include <cuda_runtime.h>\n" if cuda else ""]
    for name in ("LDA_C_VWN", "LDA_C_VWN_RPA"):
        graph, roots = _graph(name)
        emitter = (CudaEmitter if cuda else ScalarCEmitter)(
            graph, {"rho_a": "a", "rho_b": "b"}
        )
        emitter.emit(roots)
        declaration = (
            f"__global__ void {name}_kernel" if cuda else f'extern "C" int {name}'
        )
        source.append(f"{declaration}(double a, double b, double* out) {{")
        source.extend(emitter.lines)
        source.extend(
            f"out[{i}] = {emitter.names[root.identifier]};"
            for i, root in enumerate(roots)
        )
        source.append("}" if cuda else "return 0; }")
        if cuda:
            # Synchronous copy observes kernel failures; free even on failure.
            source.append(f"""
extern "C" int {name}(double a, double b, double* out) {{
  double* device = nullptr;
  cudaError_t status = cudaMalloc(&device, 3 * sizeof(double));
  if (status != cudaSuccess) return static_cast<int>(status);
  {name}_kernel<<<1, 1>>>(a, b, device);
  status = cudaGetLastError();
  if (status == cudaSuccess)
    status = cudaMemcpy(out, device, 3 * sizeof(double), cudaMemcpyDeviceToHost);
  const auto released = cudaFree(device);
  return static_cast<int>(status == cudaSuccess ? released : status);
}}
""")
    path = folder / ("vwn.cu" if cuda else "vwn.cpp")
    path.write_text("\n".join(source))
    obj, library_path = folder / "vwn.o", folder / "vwn.so"
    # Separate compilation from linking so the shared compiler cache is used.
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O3",
            *(["--gpu-architecture=sm_120", "-Xcompiler=-fPIC"] if cuda else ["-fPIC"]),
            "-c",
            str(path),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run(
        [compiler, "-shared", str(obj), "-o", str(library_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    library = ctypes.CDLL(str(library_path))

    def compiled(name: str, alpha: float, beta: float) -> np.ndarray:
        function = getattr(library, name)
        function.argtypes = (
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        )
        function.restype = ctypes.c_int
        result = np.empty(3, dtype=np.float64)
        assert (
            function(
                alpha, beta, result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )
            == 0
        )
        return result

    return compiled


@pytest.mark.parametrize("name", ("LDA_C_VWN", "LDA_C_VWN_RPA"))
@pytest.mark.parametrize("majority", (1e-12, 0.010594746189871745, 1e4))
@pytest.mark.parametrize("ratio", (0.0, 1e-30, 1e-18, 1e-12))
@pytest.mark.parametrize("flip", (False, True))
def test_vwn_first_derivatives_match_high_precision_spin_limit(
    evaluate: Evaluator, name: str, majority: float, ratio: float, flip: bool
) -> None:
    """Protect both spin endpoints and minority fractions lost by 1 +/- zeta."""
    alpha, beta = (majority * ratio, majority) if flip else (majority, majority * ratio)
    with mp.workdps(90):
        a, b = mp.mpf(alpha), mp.mpf(beta)
        # One-sided differentiation remains in the physical domain at zero.
        expected = np.asarray(
            [
                _closed_form(name, a, b),
                mp.diff(lambda x: _closed_form(name, x, b), a, direction=1),
                mp.diff(lambda x: _closed_form(name, a, x), b, direction=1),
            ],
            dtype=float,
        )
    np.testing.assert_allclose(
        evaluate(name, alpha, beta), expected, rtol=5e-12, atol=5e-14
    )


@pytest.mark.parametrize("name", ("LDA_C_VWN", "LDA_C_VWN_RPA"))
@pytest.mark.parametrize("flip", (False, True))
def test_one_ulp_majority_perturbation_does_not_create_minority_density(
    evaluate: Evaluator, name: str, flip: bool
) -> None:
    """Retain the density that exposed the full-AO CPU/CUDA regression."""
    density = 0.010594746189871745
    neighbors = (np.nextafter(density, 0.0), density, np.nextafter(density, np.inf))
    potentials = [
        evaluate(name, 0.0, n)[1] if flip else evaluate(name, n, 0.0)[2]
        for n in neighbors
    ]
    assert np.ptp(potentials) < 5e-13
