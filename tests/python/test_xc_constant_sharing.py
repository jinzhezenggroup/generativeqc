"""Executable discovery/ABI and independent Libxc gates for specialized sharing."""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.dft.xc_point_cuda import emit_device_point_value
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.xc.automatic_cuda import emit_automatic_cuda_provider
from generativeqc_compiler.xc.constant_parameters import lift_point_constants
from generativeqc_compiler.xc.parameter_cuda_package import (
    parameter_worker_groups,
    parameterized_cuda_package,
)
from test_libxc_automatic_point_native import _inputs, _oracle

if TYPE_CHECKING:
    from conftest import NativeCxx
    from generativeqc_compiler.xc.parameter_cuda_package import ParameterWorkerGroup

# Representatives include scalar families, different source kinds, fixed powers,
# and distinct U1/P2 density ABIs. Discovery itself has no such positive list.
FAMILIES = (
    ("GGA_X_APBE", "GGA_X_PBE_SOL", "GGA_X_XPBE"),
    ("GGA_C_APBE", "GGA_C_PBE_SOL", "GGA_C_XPBE"),
    ("GGA_C_P86VWN", "GGA_C_P86VWN_FT"),
    ("GGA_XC_TH_FCFO", "GGA_XC_TH_FCO"),
    ("GGA_X_B86_MGC", "GGA_X_B86_R"),
    ("GGA_X_B88M", "GGA_X_EB88"),
    ("GGA_X_BKL1", "GGA_X_BKL2"),
    ("LDA_C_BR78", "LDA_C_WIGNER", "LDA_XC_LP_B"),
    ("MGGA_X_MS0", "MGGA_X_MS1"),
    ("MGGA_X_MS2", "MGGA_X_MS2_REV"),
    ("MGGA_X_MVSB", "MGGA_X_MVSBS"),
)


def test_lifting_retains_predicates_and_lazy_singular_arms() -> None:
    graph = Graph()
    x = graph.variable("x")
    threshold = graph.constant(0.3)
    energy = graph.select_le(x, threshold, 0, 1.7 * x.pow(-1.5))
    roots = (energy, graph.differentiate(energy, x))
    lifted, row = lift_point_constants(graph, roots)
    assert row
    for value in (0.0, 0.3, 0.31, 2.0):
        expected = [graph.evaluate(root, {"x": value}) for root in roots]
        actual = [graph.evaluate(root, {"x": value, **dict(row)}) for root in lifted]
        np.testing.assert_array_equal(actual, expected)
    # Altering launch data cannot change the fixed predicate or speculate 0**-1.5.
    assert graph.evaluate(lifted[0], {"x": 0.3, **dict.fromkeys(dict(row), 100)}) == 0


def test_discovery_preserves_specialization_and_scientific_identity() -> None:
    names = (*FAMILIES[0], "GGA_X_PBEK1_VDW", "GGA_X_RPBE")
    groups = parameter_worker_groups(names)
    assert any(set(group.names) == set(FAMILIES[0]) for group in groups)
    shared = {name for group in groups for name in group.names}
    # Different work thresholds and a different formula remain independent.
    assert "GGA_X_PBEK1_VDW" not in shared
    assert "GGA_X_RPBE" not in shared
    assert parameterized_cuda_package(
        names, policy="discover"
    ) == parameterized_cuda_package((*reversed(names), names[0]), policy="discover")
    _, providers = parameterized_cuda_package(names, policy="discover")
    for name, provider in providers.items():
        original = emit_automatic_cuda_provider(name)
        assert provider.point == original.point
        assert provider.restricted == original.restricted
        assert provider.expression_identity == original.expression_identity
        assert (provider.source_identity != original.source_identity) == (
            name in shared
        )
    assert not parameter_worker_groups((names[0],))
    assert not parameterized_cuda_package(names)[0]
    with pytest.raises(ValueError, match="sharing policy"):
        parameterized_cuda_package(names, policy="unknown")


@pytest.mark.parametrize("policy", ("qualified", "discover"))
def test_generator_reports_candidate_and_selected_workers(
    tmp_path: Path, policy: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    report = tmp_path / "sharing.json"
    command = [
        sys.executable,
        str(root / "tools/generate_libxc_semilocal_cuda_registry.py"),
        "--output-directory",
        str(tmp_path / "generated"),
        "--sharing-report",
        str(report),
        "--parameter-policy",
        policy,
    ]
    for name in FAMILIES[0]:
        command.extend(("--functional", name))
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    data = json.loads(report.read_text())
    assert data["registrations"] == len(FAMILIES[0])
    assert data["shared_registrations"] == (
        len(FAMILIES[0]) if policy == "discover" else 0
    )
    (candidate,) = data["candidate_groups"]
    assert set(candidate["names"]) == set(FAMILIES[0])
    assert candidate["selected"] is (policy == "discover")
    assert candidate["default_qualified"] is False


@pytest.fixture(scope="module", params=FAMILIES, ids=lambda names: names[0])
def shared_point(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    native_cxx: NativeCxx,
) -> tuple[ParameterWorkerGroup, ctypes.CDLL]:
    """Compile a single pair of bodies and execute every discovered row through it."""
    (group,) = parameter_worker_groups(request.param)
    point, restricted = group.providers[0].point, group.providers[0].restricted
    directory = tmp_path_factory.mktemp("xc-constant-sharing")
    source = directory / "points.cpp"
    source.write_text(
        "#include <cmath>\n#include <cstddef>\n#define __device__\n"
        "using std::isfinite;\n"
        + emit_device_point_value()
        + point.source
        + restricted.source
        + f"""
extern "C" void evaluate(const double* input, const double* parameters,
                         double* output, std::size_t count, bool paired) {{
  for (std::size_t i = 0; i < count; ++i) {{
    const double* x = input + 10*i;
    const double gradient[2][3]{{{{x[2],x[3],x[4]}},{{x[5],x[6],x[7]}}}};
    const auto value = paired
      ? {restricted.function_name}(x, gradient, x+8, parameters)
      : {point.function_name}(x, gradient, x+8, parameters);
    double* out = output + 12*i;
    out[0] = value.energy;
    for (unsigned spin = 0; spin < 2; ++spin) {{
      out[1+spin] = value.rho[spin];
      out[9+spin] = value.kinetic[spin];
      for (unsigned axis = 0; axis < 3; ++axis)
        out[3+3*spin+axis] = value.gradient[spin][axis];
    }}
    out[11] = value.valid;
  }}
}}
"""
    )
    library = ctypes.CDLL(
        str(
            native_cxx.build_shared(
                [source],
                directory / "points.so",
                compile_args=("-std=c++20", "-O3", "-ffp-contract=off"),
            )
        )
    )
    vector = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    library.evaluate.argtypes = (vector, vector, vector, ctypes.c_size_t, ctypes.c_bool)
    library.evaluate.restype = None
    return group, library


@pytest.mark.parametrize("spin", (0, 1))
def test_discovered_rows_match_independent_libxc(
    shared_point: tuple[ParameterWorkerGroup, ctypes.CDLL], spin: int
) -> None:
    group, library = shared_point
    for provider in group.providers:
        inputs = np.ascontiguousarray(_inputs(provider.point.name, spin))
        row = np.array([value for _, value in provider.point.parameters])
        actual = np.empty((len(inputs), 12))
        library.evaluate(inputs, row, actual, len(inputs), spin == 0)
        assert np.all(actual[:, 11] == 1)
        observed = (
            actual[:, :11]
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
            observed, _oracle(provider.point.name, inputs, spin), atol=2e-11, rtol=2e-9
        )
        # Invalid inputs must keep their device-visible failure contract.
        inputs[0, 0] = -1
        library.evaluate(inputs, row, actual, len(inputs), spin == 0)
        assert actual[0, 11] == 0
