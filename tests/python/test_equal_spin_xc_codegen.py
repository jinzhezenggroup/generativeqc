"""Equal-spin CSE preserves polarized partials and Libxc work-input policies."""

import ctypes
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.dft._generated_native_semilocal import SEMILOCAL_FAMILIES
from generativeqc_compiler.dft._generated_split_hybrid_consumers import (
    SPLIT_HYBRID_POINT_CONSUMERS,
)
from generativeqc_compiler.dft.ao_cuda import (
    emit_native_equal_spin_point_dispatch,
    emit_native_xc_contraction_kernels,
    emit_native_xc_point_feature_terms,
)
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.xc._generated_split_hybrids import SPLIT_HYBRIDS
from generativeqc_compiler.xc.equal_spin import equal_spin_roots
from generativeqc_compiler.xc.geometry_cuda import _emit_composed_point
from generativeqc_compiler.xc.semilocal_codegen import build_roots

from tools.generate_xc_b3lyp_cuda import emit_b3lyp_device
from tools.generate_xc_r2scan_cuda import emit_r2scan_device
from tools.generate_xc_split_hybrid_cuda import (
    emit_split_hybrid_device,
    emit_split_hybrid_device_body,
)
from tools.generate_xc_wb97mv_cuda import emit_wb97mv_device

ROOT = Path(__file__).resolve().parents[2]
POINT_CONSUMERS = (
    tuple(
        (
            record["code"],
            5 if record["requires_tau"] else 4 if record["requires_gradient"] else 1,
        )
        for record in SEMILOCAL_FAMILIES
        if record["cuda_ks"]
    )
    + SPLIT_HYBRID_POINT_CONSUMERS
)


def test_equal_spin_binds_after_differentiation_and_keeps_cross_sigma() -> None:
    graph = Graph()
    alpha = graph.variable("rho_a")
    beta = graph.variable("rho_b")
    cross = graph.variable("sigma_ab")
    energy = alpha.pow(2) + 3 * beta.pow(2) + cross
    roots = (
        energy,
        graph.differentiate(energy, alpha),
        graph.differentiate(energy, beta),
    )
    restricted = equal_spin_roots(graph, roots)
    values = {"rho_a": 2.0, "sigma_ab": 5.0}
    assert tuple(graph.evaluate(root, values) for root in restricted) == (
        21.0,
        4.0,
        12.0,
    )
    variables = {
        graph.nodes[index].payload
        for index in graph.topological_order(restricted)
        if graph.nodes[index].operation == "variable"
    }
    assert variables == {"rho_a", "sigma_ab"}


@pytest.mark.parametrize("method", ("B3LYP", "WB97M-V"))
def test_equal_spin_removes_repeated_scalar_work(method: str) -> None:
    spec = resolve_method(method, spin="polarized").primitives[0].functional
    outputs = ((), *((index,) for index in range(len(spec.features))))
    graph, roots, _ = build_roots(spec, outputs, production=True)
    bound = equal_spin_roots(graph, roots)
    assert len(graph.topological_order(bound)) < len(graph.topological_order(roots))


def test_native_equal_spin_selector_defaults_on_for_rks_and_preserves_uks_opt_out(
    tmp_path: Path,
) -> None:
    """Execute the production host selector, not a Python policy imitation."""
    compiler = shutil.which("c++")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if compiler is None or launcher is None:
        pytest.skip("selector qualification requires C++ and a compiler cache")
    subprocess.run([launcher, "--version"], check=True, capture_output=True, timeout=30)
    generated = emit_native_xc_contraction_kernels()
    selector = generated.split("bool use_equal_spin_points(", 1)[1].split("\n}", 1)[0]
    source = tmp_path / "selector.cpp"
    source.write_text(
        "#include <cstdlib>\n#include <cstring>\n#include <cstdint>\n"
        "#include <stdexcept>\n#include <cassert>\n#include <initializer_list>\n"
        + "bool use_equal_spin_points("
        + selector
        + "\n}\n"
        + r"""
int main() {
  unsetenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN");
  assert(use_equal_spin_points(1));
  assert(!use_equal_spin_points(2));
  setenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", "1", 1);
  assert(use_equal_spin_points(1));
  assert(!use_equal_spin_points(2));
  setenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", "0", 1);
  assert(!use_equal_spin_points(1));
  setenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", "yes", 1);
  try { use_equal_spin_points(1); } catch (const std::invalid_argument&) { return 0; }
  return 1;
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "selector"
    subprocess.run(
        [launcher, compiler, "-std=c++17", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=30)
    assert (
        "evaluate_point_tiles<feature_terms, false, batched, EqualSpinPointEvaluator<bound_functional>>"
        in generated
    )
    assert "if constexpr (!response)" in generated
    assert "functional > static_cast<I>(SemilocalFamily::Pbe)" not in generated
    assert "spins == 1 ? 0.5 : 1.0" in generated


def test_method_bound_dispatch_uses_exact_registered_code_and_width(
    tmp_path: Path,
) -> None:
    """Run the emitted dispatch with a host stub of the CUDA launch primitive."""
    compiler = shutil.which("c++")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if compiler is None or launcher is None:
        pytest.skip("dispatch qualification requires C++ and a compiler cache")
    subprocess.run([launcher, "--version"], check=True, capture_output=True, timeout=30)
    dispatch = re.sub(r"<<<[^>]*>>>", "", emit_native_equal_spin_point_dispatch())
    checks = []
    for code, width in POINT_CONSUMERS:
        for batched in ("false", "true"):
            checks.extend(
                (
                    f"  invoke<{width},{batched}>({code});",
                    f"  assert(observed_functional == {code} && observed_width == {width} && observed_batch == {batched});",
                )
            )
        wrong_width = 4 if width != 4 else 5
        checks.extend(
            (
                f"  try {{ invoke<{wrong_width},false>({code}); assert(false); }} catch (const std::invalid_argument&) {{}}",
                f"  try {{ invoke<{width},false>({code},2); assert(false); }} catch (const std::invalid_argument&) {{}}",
                f"  assert(point_feature_terms({code}) == {width});",
            )
        )
    source = tmp_path / "dispatch.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include "dft/semilocal_family.hpp"
using generativeqc::dft::SemilocalFamily;
using generativeqc::dft::semilocal_family_code;
using I = long;
using cudaStream_t = void*;
I observed_functional = -1, observed_width = -1;
bool observed_batch = false;
template <I width, bool batched, I functional, class... Args>
void evaluate_equal_spin_points(Args...) {
  static_assert(functional >= 0);
  observed_functional = functional;
  observed_width = width;
  observed_batch = batched;
}
"""
        + dispatch
        + emit_native_xc_point_feature_terms()
        + r"""
template <I width, bool batched>
void invoke(unsigned functional, unsigned spins = 1) {
  launch_equal_spin_points<width,128,batched>(nullptr,nullptr,nullptr,3,spins,nullptr,
                                             nullptr,nullptr,functional,1,1,2);
}
int main() {
"""
        + "\n".join(checks)
        + r"""
  try { invoke<4,false>(99999); assert(false); } catch (const std::invalid_argument&) {}
  try { point_feature_terms(99999); assert(false); } catch (const std::invalid_argument&) {}
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "dispatch"
    subprocess.run(
        [
            launcher,
            compiler,
            "-std=c++17",
            f"-I{ROOT / 'src'}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=30)


def _m06_integer_power_point_header() -> str:
    """Exercise the explicit experiment without replacing default qualification."""
    body, _, _ = emit_split_hybrid_device_body("M06-2X", lower_integer_powers=True)
    return (
        "namespace generativeqc::dft::generated::integer_power_point {\n" + body + "}\n"
    )


@pytest.fixture(scope="module")
def compiled_points(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    """Compile the actual CUDA scalar bodies as strict host C++ for parity."""
    compiler = shutil.which("c++")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if compiler is None or launcher is None:
        pytest.skip("host point qualification requires C++ and a compiler cache")
    subprocess.run([launcher, "--version"], check=True, capture_output=True, timeout=30)
    directory = tmp_path_factory.mktemp("equal-spin-xc")
    generators = (
        emit_b3lyp_device,
        emit_r2scan_device,
        emit_wb97mv_device,
        lambda: emit_split_hybrid_device("M06-2X"),
        _m06_integer_power_point_header,
    )
    symbols = (
        "b3lyp_device",
        "r2scan_device",
        "wb97mv_device",
        "m06_2x_device",
        "integer_power_point::m06_2x_device",
    )
    lines = ["#define __device__", "#include <cmath>"]
    for index, generator in enumerate(generators):
        header = directory / f"point_{index}.hpp"
        header.write_text(generator(), encoding="utf-8")
        lines.append(f'#include "{header.name}"')
    lines.extend(
        (
            "using namespace generativeqc::dft::generated;",
            "template <class Value> void store(Value value, double* output) {",
            "  output[0] = value.energy_density;",
            "  for (unsigned index = 0; index < 7; ++index) output[index + 1] = 0;",
            "  for (unsigned index = 0; index < sizeof(value.feature_derivative) / sizeof(double); ++index)",
            "    output[index + 1] = value.feature_derivative[index];",
            "}",
            'extern "C" void evaluate(int method, bool restricted, const double* input, double* output) {',
            "  switch (method) {",
        )
    )
    for index, symbol in enumerate(symbols):
        arguments = ", ".join(
            f"input[{feature}]" for feature in range(5 if index == 0 else 7)
        )
        lines.extend(
            (
                f"    case {index}: store(restricted ? {symbol}_equal_spin({arguments})",
                f"                                    : {symbol}({arguments}), output); break;",
            )
        )
    lines.extend(("  }", "}"))
    for method_index, identifier in enumerate(("M06-2X", "MN15")):
        for spin in ("unpolarized", "polarized"):
            semilocal = resolve_method(identifier, spin=spin).primitives[0].functional
            lines.extend(
                (
                    f"namespace force_point_{method_index}_{spin} {{",
                    "using std::isfinite;",
                    _emit_composed_point(
                        SPLIT_HYBRIDS[identifier]["functional_code"], semilocal
                    ),
                    "}",
                )
            )
    lines.extend(
        (
            'extern "C" void evaluate_force(int method, bool restricted, const double* input, double* output) {',
            "  const double gradient[2][3] = {{input[2],input[3],input[4]}, {input[5],input[6],input[7]}};",
            "  switch (method) {",
        )
    )
    for method_index in range(2):
        lines.extend(
            (
                f"    case {method_index}: {{",
                "  const auto store_force = [output](const auto& value) {",
                "  output[0] = value.energy;",
                "  for (unsigned spin=0; spin<2; ++spin) {",
                "    output[spin+1] = value.rho[spin];",
                "    for (unsigned coordinate=0; coordinate<3; ++coordinate)",
                "      output[3+3*spin+coordinate] = value.gradient[spin][coordinate];",
                "    output[9+spin] = value.kinetic[spin];",
                "  }",
                "  };",
                f"  if (restricted) store_force(force_point_{method_index}_unpolarized::stationary_evaluate_point<true>(input,gradient,input+8));",
                f"  else store_force(force_point_{method_index}_polarized::stationary_evaluate_point<false>(input,gradient,input+8));",
                "  break;",
                "}",
            )
        )
    lines.extend(("  }", "}"))
    source = directory / "probe.cpp"
    source.write_text("\n".join(lines), encoding="utf-8")
    library = directory / "probe.so"
    subprocess.run(
        [
            launcher,
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    loaded = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_double)
    loaded.evaluate.argtypes = (ctypes.c_int, ctypes.c_bool, pointer, pointer)
    loaded.evaluate.restype = None
    loaded.evaluate_force.argtypes = (ctypes.c_int, ctypes.c_bool, pointer, pointer)
    loaded.evaluate_force.restype = None
    return loaded


def point_values(
    library: ctypes.CDLL, method: int, restricted: bool, inputs: np.ndarray
) -> np.ndarray:
    """Evaluate both scalar implementations in one binary without GPU access."""
    output = np.empty((len(inputs), 8))
    pointer = ctypes.POINTER(ctypes.c_double)
    for row, result in zip(inputs, output, strict=True):
        library.evaluate(
            method,
            restricted,
            row.ctypes.data_as(pointer),
            result.ctypes.data_as(pointer),
        )
    return output


def physical_equal_spin_points() -> np.ndarray:
    """Include both ordinary molecular inputs and independent work cross-sigma."""
    rng = np.random.default_rng(20261011)
    density = 10.0 ** rng.uniform(-2, 0, 32)
    sigma = density ** (8 / 3) * rng.uniform(0.01, 0.3, len(density))
    kinetic = density ** (5 / 3) * rng.uniform(0.5, 2, len(density))
    return np.column_stack((density, density, sigma, sigma, sigma, kinetic, kinetic))


@pytest.mark.parametrize("method", range(5))
def test_emitted_equal_spin_points_preserve_every_polarized_partial(
    compiled_points: ctypes.CDLL, method: int
) -> None:
    inputs = physical_equal_spin_points()
    cross_inputs = inputs.copy()
    cross_inputs[:, 3] = 0
    inputs = np.concatenate((inputs, cross_inputs))
    if method in (1, 3, 4):
        boundaries = np.asarray(
            [
                (density, density, sigma, 0, sigma, kinetic, kinetic)
                for density in (0, 1e-30, 4e-16, 1e-15, 1e-14, 1e-10, 0.2)
                for sigma, kinetic in ((0, 0), (1e-40, 1e-20), (0.01, 0.001))
            ],
            dtype=np.float64,
        )
        inputs = np.concatenate((inputs, boundaries))
    general = point_values(compiled_points, method, False, inputs)
    restricted = point_values(compiled_points, method, True, inputs)
    assert np.all(np.isfinite(general)) and np.all(np.isfinite(restricted))
    np.testing.assert_array_equal(restricted, general)


@pytest.mark.parametrize(
    ("method", "libxc_code"),
    ((0, "B3LYP"), (1, "R2SCAN"), (2, "WB97M_V"), (3, "450,236"), (4, "450,236")),
)
def test_equal_spin_points_match_independent_libxc(
    compiled_points: ctypes.CDLL, method: int, libxc_code: str
) -> None:
    pytest.importorskip("pyscf")
    from pyscf.dft import libxc

    from tools.validate_split_hybrid_cuda import libxc_density

    inputs = physical_equal_spin_points()
    density = libxc_density(inputs)
    exc, derivatives, _, _ = libxc.eval_xc(
        libxc_code, density[:, :4] if method == 0 else density, spin=1, deriv=1
    )
    kinetic = np.zeros((len(inputs), 2)) if method == 0 else derivatives[3]
    reference = np.column_stack(
        (exc * inputs[:, :2].sum(axis=1), derivatives[0], derivatives[1], kinetic)
    )
    values = point_values(compiled_points, method, True, inputs)
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(reference))
    np.testing.assert_allclose(values[:, 0], reference[:, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(values[:, 1:], reference[:, 1:], rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("restricted", (False, True))
@pytest.mark.parametrize(("method", "libxc_code"), ((0, "450,236"), (1, "268,269")))
def test_integer_power_force_point_matches_independent_libxc_and_finite_difference(
    compiled_points: ctypes.CDLL, restricted: bool, method: int, libxc_code: str
) -> None:
    """Exercise the production force emitter, including Cartesian and tau ABIs."""
    pytest.importorskip("pyscf")
    from pyscf.dft import libxc

    from tools.validate_split_hybrid_cuda import libxc_density

    rng = np.random.default_rng(20261014)
    density = 10.0 ** rng.uniform(-2, 0, (32, 2))
    gradient = rng.normal(size=(32, 2, 3)) * density[..., None] ** (4 / 3) * 0.1
    kinetic = density ** (5 / 3) * rng.uniform(0.5, 2, (32, 2))
    if restricted:
        density[:, 1] = density[:, 0]
        gradient[:, 1] = gradient[:, 0]
        kinetic[:, 1] = kinetic[:, 0]
    inputs = np.column_stack((density, gradient.reshape(32, 6), kinetic))
    sigma = np.column_stack(
        (
            np.sum(gradient[:, 0] ** 2, axis=1),
            np.sum(gradient[:, 0] * gradient[:, 1], axis=1),
            np.sum(gradient[:, 1] ** 2, axis=1),
        )
    )
    features = np.column_stack((density, sigma, kinetic))
    exc, derivatives, _, _ = libxc.eval_xc(
        libxc_code, libxc_density(features), spin=1, deriv=1
    )
    reference_gradient = np.empty_like(gradient)
    reference_gradient[:, 0] = (
        2 * derivatives[1][:, 0, None] * gradient[:, 0]
        + derivatives[1][:, 1, None] * gradient[:, 1]
    )
    reference_gradient[:, 1] = (
        2 * derivatives[1][:, 2, None] * gradient[:, 1]
        + derivatives[1][:, 1, None] * gradient[:, 0]
    )
    reference = np.column_stack(
        (
            exc * density.sum(axis=1),
            derivatives[0],
            reference_gradient.reshape(32, 6),
            0.5 * derivatives[3],
        )
    )
    pointer = ctypes.POINTER(ctypes.c_double)

    def evaluate(values: np.ndarray) -> np.ndarray:
        output = np.empty((len(values), 11))
        for point, result in zip(values, output, strict=True):
            compiled_points.evaluate_force(
                method,
                restricted,
                point.ctypes.data_as(pointer),
                result.ctypes.data_as(pointer),
            )
        return output

    actual = evaluate(inputs)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual[:, 0], reference[:, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual[:, 1:], reference[:, 1:], rtol=2e-10, atol=2e-10)
    displacement = inputs * rng.uniform(-0.1, 0.1, inputs.shape)
    if restricted:
        displacement[:, 1] = displacement[:, 0]
        displacement[:, 5:8] = displacement[:, 2:5]
        displacement[:, 9] = displacement[:, 8]
    step = 1e-5
    finite_difference = (
        evaluate(inputs + step * displacement)[:, 0]
        - evaluate(inputs - step * displacement)[:, 0]
    ) / (2 * step)
    partials = actual[:, 1:].copy()
    partials[:, -2:] *= 2
    predicted = np.sum(partials * displacement, axis=1)
    np.testing.assert_allclose(predicted, finite_difference, rtol=2e-5, atol=2e-8)
