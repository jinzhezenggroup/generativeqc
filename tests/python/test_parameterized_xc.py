"""Independent numerical gates for mathematical-parameter XC lowering."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.bulk_runtime import build_bulk_runtime_program
from generativeqc_compiler.xc.libxc_bulk import read_catalog
from generativeqc_compiler.xc.libxc_maple import MapleImportError
from test_libxc_automatic_point_native import _inputs, _oracle

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
RECORDS = {
    r["name"]: r
    for r in read_catalog()["registrations"]
    if r["entry"] == "maple/mgga_exc/mgga_x_ktbm.mpl"
}
PARAMETERS = tuple(
    sorted(
        name
        for name in RECORDS["MGGA_X_KTBM_0"]["bindings"]
        if name.startswith("params_a_")
    )
)


@pytest.fixture(scope="module", params=(False, True))
def parameter_point(
    tmp_path_factory: pytest.TempPathFactory,
    native_cxx: NativeCxx,
    request: pytest.FixtureRequest,
) -> tuple[ctypes.CDLL, bool]:
    """Compile one mathematical body and reuse it with every registered row."""
    directory = tmp_path_factory.mktemp("parameterized-xc")
    prepared = bool(request.param)
    point = emit_automatic_point(
        "MGGA_X_KTBM_0", parameter_names=PARAMETERS, precompute_parameters=prepared
    )
    source = directory / "point.cpp"
    source.write_text(
        '#include <cmath>\n#include <stdexcept>\n#include "dft/xc.hpp"\n'
        "using generativeqc::dft::SemilocalPointValue;\n"
        + point.source
        + f"""
extern "C" int evaluate(const double* input, const double* parameters,
                        double* output, std::size_t count) {{
  for (std::size_t i = 0; i < count; ++i) {{
    const double* x = input + 10*i;
    const double gradient[2][3]{{{{x[2],x[3],x[4]}},{{x[5],x[6],x[7]}}}};
    try {{
      auto value = {point.function_name}(x, gradient, x+8, parameters);
      double* out = output + 11*i;
      out[0] = value.energy;
      for (unsigned spin = 0; spin < 2; ++spin) {{
        out[1+spin] = value.rho[spin];
        out[9+spin] = value.kinetic[spin];
        for (unsigned axis = 0; axis < 3; ++axis)
          out[3+3*spin+axis] = value.gradient[spin][axis];
      }}
    }} catch (const std::domain_error&) {{ return 1; }}
  }}
  return 0;
}}
"""
    )
    library = ctypes.CDLL(
        str(
            native_cxx.build_shared(
                [source],
                directory / "point.so",
                compile_args=(
                    "-std=c++20",
                    "-O3",
                    "-ffp-contract=off",
                    "-I",
                    str(ROOT / "src"),
                    "-I",
                    str(ROOT / "include"),
                ),
            )
        )
    )
    vector = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    library.evaluate.argtypes = (vector, vector, vector, ctypes.c_size_t)
    library.evaluate.restype = ctypes.c_int
    return library, prepared


@pytest.mark.parametrize("name", tuple(RECORDS))
@pytest.mark.parametrize("spin", (0, 1))
def test_parameter_rows_match_independent_libxc(
    parameter_point: tuple[ctypes.CDLL, bool], name: str, spin: int
) -> None:
    library, prepared = parameter_point
    inputs = np.ascontiguousarray(_inputs(name, spin))
    point = emit_automatic_point(
        name, parameter_names=PARAMETERS, precompute_parameters=prepared
    )
    parameters = np.array([value for _, value in point.parameters])
    actual = np.empty((len(inputs), 11))
    assert library.evaluate(inputs, parameters, actual, len(inputs)) == 0
    assert np.all(actual[[0, 4]] == 0)
    observed = (
        actual
        if spin
        else np.column_stack(
            (
                actual[:, 0],
                actual[:, 1:3].mean(axis=1),
                0.5 * (actual[:, 3:6] + actual[:, 6:9]),
                actual[:, 9:11].mean(axis=1),
            )
        )
    )
    np.testing.assert_allclose(
        observed, _oracle(name, inputs, spin), atol=2e-11, rtol=2e-9
    )
    interior = inputs[10:]
    direction = (
        np.random.default_rng(11212026).normal(size=interior.shape)
        * np.abs(interior)
        * 0.1
    )
    if not spin:
        direction[:, 1] = direction[:, 0]
        direction[:, 5:8] = direction[:, 2:5]
        direction[:, 9] = direction[:, 8]
    step = 1e-5
    expected = (
        _oracle(name, interior + step * direction, spin)[:, 0]
        - _oracle(name, interior - step * direction, spin)[:, 0]
    ) / (2 * step)
    derivative = np.sum(
        np.column_stack((actual[10:, 1:9], 2 * actual[10:, 9:11])) * direction, axis=1
    )
    np.testing.assert_allclose(derivative, expected, atol=2e-10, rtol=2e-7)


def test_symbolic_program_retains_registered_values() -> None:
    name = "MGGA_X_KTBM_1"
    symbolic = build_bulk_runtime_program(name, parameter_names=PARAMETERS)
    literal = build_bulk_runtime_program(name)
    x = np.array([[0.2], [0.1], [0.01], [0.002], [0.005], [0.2], [0.1]])
    np.testing.assert_allclose(symbolic.evaluate(x), literal.evaluate(x), atol=2e-14)
    other = build_bulk_runtime_program("MGGA_X_KTBM_0", parameter_names=PARAMETERS)
    assert symbolic.expression_hash != other.expression_hash
    assert not np.allclose(symbolic.evaluate(x), other.evaluate(x))


def test_prepared_products_preserve_lazy_input_arithmetic() -> None:
    """Preparation cannot speculate a singular inactive arm onto the host."""
    from generativeqc_compiler.integral.expr import Graph
    from generativeqc_compiler.xc.parameter_products import prepare_parameter_products

    graph = Graph()
    x, coefficient = graph.variable("x"), graph.variable("coefficient")
    product = graph.multiply_many((coefficient, graph.constant(0.3), x))
    energy = graph.select_le(x, graph.constant(0), graph.constant(0), product / x)
    derivative = graph.differentiate(energy, x)
    parameters = (("coefficient", 1.7),)
    roots, prepared = prepare_parameter_products(
        graph, (energy, derivative), parameters
    )
    for value in (-1.0, 0.0, 0.1, 2.0):
        expected = [
            graph.evaluate(root, {"x": value, **dict(parameters)})
            for root in (energy, derivative)
        ]
        actual = [
            graph.evaluate(root, {"x": value, **dict(prepared)}) for root in roots
        ]
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    # Paired and general workers must agree on the entire launch-data layout,
    # including products that occur only in independent polarized derivatives.
    general = emit_automatic_point(
        "MGGA_X_KTBM_0",
        backend="cuda",
        parameter_names=PARAMETERS,
        precompute_parameters=True,
    )
    paired = emit_automatic_point(
        "MGGA_X_KTBM_0",
        backend="cuda",
        equal_spin=True,
        parameter_names=PARAMETERS,
        precompute_parameters=True,
    )
    assert general.parameters == paired.parameters


@pytest.mark.parametrize(
    "parameters",
    [
        ("p_a_dens_threshold",),
        ("params_a_absent",),
        (PARAMETERS[0], PARAMETERS[0]),
    ],
)
def test_symbolic_parameters_reject_ambiguous_or_worker_bindings(
    parameters: tuple[str, ...],
) -> None:
    with pytest.raises(MapleImportError):
        emit_automatic_point("MGGA_X_KTBM_0", parameter_names=parameters)


def test_package_preserves_registered_science_and_selection() -> None:
    """Request ordering cannot alter workers, rows, or CPU/CUDA admission."""
    from generativeqc_compiler.xc.automatic_cuda import emit_automatic_cuda_provider
    from generativeqc_compiler.xc.parameter_cuda_package import (
        parameterized_cuda_package,
    )

    names = tuple(RECORDS)
    workers, providers = parameterized_cuda_package(names)
    assert workers
    assert (workers, providers) == parameterized_cuda_package(
        (*reversed(names), names[0])
    )
    for name, provider in providers.items():
        literal = emit_automatic_cuda_provider(name)
        assert provider.point == literal.point
        assert provider.restricted == literal.restricted
        assert provider.expression_identity == literal.expression_identity
        assert provider.source_identity != literal.source_identity
    # A singleton retains literal constant folding; unrelated registrations do
    # not enter a family merely because their physical point layouts match.
    for name in (names[0], "GGA_X_APBE"):
        assert parameterized_cuda_package((name,)) == (
            {},
            {name: emit_automatic_cuda_provider(name)},
        )
    subset_workers, subset = parameterized_cuda_package(names[:2])
    assert set(subset) == set(names[:2])
    assert set(subset_workers).isdisjoint(workers)


def test_parameterized_descriptors_admit_the_canonical_cpu_science(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Compile native descriptors for all rows, including rejection of stale math."""
    import subprocess

    from generativeqc_compiler.xc.automatic_cuda import emit_automatic_cuda_descriptor
    from generativeqc_compiler.xc.parameter_cuda_package import (
        parameterized_cuda_package,
    )

    _, providers = parameterized_cuda_package(tuple(RECORDS))
    (tmp_path / "cuda_runtime_api.h").write_text("using cudaStream_t = void*;\n")
    source = '#include <cassert>\n#include "dft/cuda_xc_point_program.hpp"\n'
    checks = []
    for name, provider in providers.items():
        source += (
            f"namespace {provider.namespace} {{\n"
            + """
void launch(cudaStream_t, const double*, const double*, std::size_t, std::size_t,
            double*, double*, int*, std::uint32_t, double, double, const double*) {}
void launch_batch(cudaStream_t, const double*, const double*, std::size_t,
                  std::size_t, double*, double*, int*, std::uint32_t, double,
                  double, std::size_t) {}
}
"""
            + emit_automatic_cuda_descriptor(provider)
        )
        cpu = emit_automatic_point(name)
        checks.append(f"""
  {{
    const auto& program = {provider.namespace}::program;
    assert(program.matches_cpu_science("{cpu.expression_identity}", "",
                                      {cpu.ingredient_mask}, {cpu.domain_version}));
    assert(!program.matches_cpu_science("stale", "",
                                       {cpu.ingredient_mask}, {cpu.domain_version}));
  }}
""")
    path = tmp_path / "descriptors.cpp"
    path.write_text(source + "int main() {\n" + "".join(checks) + "}\n")
    executable = native_cxx.build_executable(
        [path],
        tmp_path / "descriptors",
        compile_args=("-std=c++20", "-I", str(tmp_path), "-I", str(ROOT / "src")),
    )
    subprocess.run([executable], capture_output=True, check=True, timeout=10)


@pytest.mark.parametrize("share_parameters", (False, True))
def test_package_descriptor_selects_explicit_execution_identity(
    share_parameters: bool,
) -> None:
    """Explicit rollback retains specialization; sharing binds its worker identity."""
    from generativeqc_compiler.xc.automatic_cuda import emit_automatic_cuda_provider
    from generativeqc_compiler.xc.parameter_cuda_package import (
        parameterized_cuda_package,
    )

    from tools.generate_libxc_semilocal_cuda_registry import emit_shard, select_entries

    names = ("MGGA_X_KTBM_0", "MGGA_X_KTBM_1")
    entries = select_entries(list(names))
    _, shared = parameterized_cuda_package(names)
    for index, name in enumerate(names):
        literal = emit_automatic_cuda_provider(name)
        provider = shared[name] if share_parameters else literal
        source = emit_shard(index, entries, share_parameters=share_parameters)
        assert provider.source_identity in source
        assert literal.expression_identity in source
