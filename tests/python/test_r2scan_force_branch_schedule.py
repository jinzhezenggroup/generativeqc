"""Executable contracts for consumer-local r2SCAN force branch sharing."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.xc import geometry_cuda
from generativeqc_compiler.xc.semilocal_codegen import emit_r2scan_program

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def force_points(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> ctypes.CDLL:
    """Compile both schedules and the actual stationary Cartesian consumer."""
    directory = tmp_path_factory.mktemp("r2scan-force-branches")
    lines = ["#include <cmath>", "#define __device__", "using std::isfinite;"]
    for coalesced in (False, True):
        lines.extend(
            (
                f"namespace schedule_{int(coalesced)} {{",
                emit_r2scan_program(
                    value_type="Point",
                    function_name="evaluate",
                    identity_constant="kIdentity",
                    qualifier="inline",
                    include_equal_spin=True,
                    coalesce_branches=coalesced,
                ),
                "}",
            )
        )
    lines.extend(
        (
            "namespace stationary {",
            geometry_cuda._emit_stationary_point(2),
            "}",
            'extern "C" void raw(bool coalesced, bool restricted, const double* input, double* output) {',
            "  auto store = [output](const auto& value) {",
            "    output[0] = value.energy_density;",
            "    for (unsigned index = 0; index < 7; ++index)",
            "      output[index + 1] = value.feature_derivative[index];",
            "  };",
        )
    )
    arguments = ", ".join(f"input[{index}]" for index in range(7))
    for coalesced in (False, True):
        lines.extend(
            (
                f"  if (coalesced == {str(coalesced).lower()}) {{",
                f"    if (restricted) store(schedule_{int(coalesced)}::evaluate_equal_spin({arguments}));",
                f"    else store(schedule_{int(coalesced)}::evaluate({arguments}));",
                "  }",
            )
        )
    lines.extend(
        (
            "}",
            'extern "C" bool cartesian(bool restricted, const double* input, double* output) {',
            "  double gradient[2][3];",
            "  for (unsigned spin = 0; spin < 2; ++spin)",
            "    for (unsigned axis = 0; axis < 3; ++axis)",
            "      gradient[spin][axis] = input[2 + 3 * spin + axis];",
            "  const auto value = restricted",
            "      ? stationary::stationary_evaluate_point<true>(input, gradient, input + 8)",
            "      : stationary::stationary_evaluate_point<false>(input, gradient, input + 8);",
            "  output[0] = value.energy;",
            "  for (unsigned spin = 0; spin < 2; ++spin) {",
            "    output[1 + spin] = value.rho[spin];",
            "    for (unsigned axis = 0; axis < 3; ++axis)",
            "      output[3 + 3 * spin + axis] = value.gradient[spin][axis];",
            "    output[9 + spin] = value.kinetic[spin];",
            "  }",
            "  return value.valid;",
            "}",
        )
    )
    source = directory / "points.cpp"
    source.write_text("\n".join(lines), encoding="utf-8")
    library = ctypes.CDLL(
        str(
            native_cxx.build_shared(
                [source],
                directory / "points.so",
                compile_args=("-std=c++17", "-O2", "-ffp-contract=off"),
            )
        )
    )
    pointer = ctypes.POINTER(ctypes.c_double)
    library.raw.argtypes = (ctypes.c_bool, ctypes.c_bool, pointer, pointer)
    library.cartesian.argtypes = (ctypes.c_bool, pointer, pointer)
    library.cartesian.restype = ctypes.c_bool
    return library


def _raw(
    library: ctypes.CDLL, inputs: np.ndarray, *, coalesced: bool, restricted: bool
) -> np.ndarray:
    """Evaluate every feature partial without changing either spin ABI."""
    pointer = ctypes.POINTER(ctypes.c_double)
    inputs = np.ascontiguousarray(inputs, dtype=np.float64)
    outputs = np.empty((len(inputs), 8))
    for row, output in zip(inputs, outputs, strict=True):
        library.raw(
            coalesced,
            restricted,
            row.ctypes.data_as(pointer),
            output.ctypes.data_as(pointer),
        )
    return outputs


@pytest.mark.parametrize("restricted", (False, True))
def test_force_schedule_preserves_all_partial_bits(
    force_points: ctypes.CDLL, restricted: bool
) -> None:
    """Sharing branches must preserve values, not just the summed energy."""
    random = np.random.default_rng(20261015)
    density = 10 ** random.uniform(-24, 1, (512, 2))
    gradient = random.normal(size=(512, 2, 3)) * density[:, :, None]
    sigma = np.column_stack(
        (
            np.sum(gradient[:, 0] ** 2, axis=1),
            np.sum(gradient[:, 0] * gradient[:, 1], axis=1),
            np.sum(gradient[:, 1] ** 2, axis=1),
        )
    )
    tau = density * 10 ** random.uniform(-4, 2, (512, 2))
    inputs = np.column_stack((density, sigma, tau))
    if restricted:
        inputs[:, 1] = inputs[:, 0]
        inputs[:, 3:5] = inputs[:, 2, None]
        inputs[:, 6] = inputs[:, 5]
    control = _raw(force_points, inputs, coalesced=False, restricted=restricted)
    candidate = _raw(force_points, inputs, coalesced=True, restricted=restricted)
    assert np.isfinite(candidate).all()
    np.testing.assert_array_equal(candidate.view(np.uint64), control.view(np.uint64))


def test_force_schedule_matches_independent_high_precision_tails(
    force_points: ctypes.CDLL,
) -> None:
    """Retain the 113-bit Libxc oracle, including exactly empty minority spins."""
    fixture = json.loads(
        (ROOT / "tests/data/xc/r2scan-tail-reference.json").read_text()
    )
    inputs = np.array([point["inputs"] for point in fixture["points"]])
    reference = np.array([point["reference"] for point in fixture["points"]])
    for exchanged in (False, True):
        if exchanged:
            inputs = inputs[:, (1, 0, 4, 3, 2, 6, 5)]
            reference = reference[:, (0, 2, 1, 5, 4, 3, 7, 6)]
        candidate = _raw(force_points, inputs, coalesced=True, restricted=False)
        np.testing.assert_allclose(candidate[:, 0], reference[:, 0], rtol=0, atol=2e-13)
        np.testing.assert_allclose(
            candidate[:, 1:], reference[:, 1:], rtol=2e-10, atol=2e-10
        )


@pytest.mark.parametrize("restricted", (False, True))
def test_stationary_cartesian_partials_match_displaced_energy(
    force_points: ctypes.CDLL, restricted: bool
) -> None:
    """The production consumer keeps independent spin gradients and tau/2."""
    pointer = ctypes.POINTER(ctypes.c_double)
    point = np.array([0.31, 0.22, 0.03, -0.02, 0.01, -0.04, 0.01, 0.02, 0.2, 0.15])
    if restricted:
        point[1] = point[0]
        point[5:8] = point[2:5]
        point[9] = point[8]

    def evaluate(inputs: np.ndarray, equal_spin: bool = False) -> np.ndarray:
        output = np.empty(11)
        assert force_points.cartesian(
            equal_spin, inputs.ctypes.data_as(pointer), output.ctypes.data_as(pointer)
        )
        return output

    candidate = evaluate(point, restricted)
    for feature in range(10):
        step = 1e-5
        plus, minus = point.copy(), point.copy()
        plus[feature] += step
        minus[feature] -= step
        derivative = (evaluate(plus)[0] - evaluate(minus)[0]) / (2 * step)
        scale = 2 if feature >= 8 else 1
        np.testing.assert_allclose(
            scale * candidate[feature + 1], derivative, rtol=2e-7, atol=2e-9
        )
