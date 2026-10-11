"""Physical kinetic predicates preserve full Fock and raw work-domain coefficients."""

from __future__ import annotations

import ctypes
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.libxc_bulk import canonicalize_spin_density_coordinates
from test_libxc_automatic_point_native import _evaluate, _native_source

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads(
    (ROOT / "tests/data/xc/physical-kinetic-predicate/tpss.json").read_text()
)


@pytest.mark.parametrize("restricted", (False, True))
@pytest.mark.parametrize("threshold", (1, 2))
def test_only_exact_kinetic_comparison_changes(
    restricted: bool, threshold: int
) -> None:
    """The algebraic rewrite preserves branch payloads and is idempotent."""
    graph = Graph()
    rho_a, rho_b = (
        (graph.variable("rho") / 2,) * 2
        if restricted
        else (graph.variable("rho_a"), graph.variable("rho_b"))
    )
    if restricted:
        same_spin_sigma = graph.variable("sigma") / 4
        sigma = same_spin_sigma + 2 * same_spin_sigma + same_spin_sigma
        tau_a = tau_b = graph.variable("tau") / 2
    else:
        sigma = (
            graph.variable("sigma_aa")
            + 2 * graph.variable("sigma_ab")
            + graph.variable("sigma_bb")
        )
        tau_a, tau_b = graph.variable("tau_a"), graph.variable("tau_b")
    density = rho_a + rho_b
    zeta = graph.constant(0) if restricted else (rho_a - rho_b) / density
    total_t = tau_a * rho_a.pow(-5 / 3) * ((1 + zeta) / 2).pow(5 / 3)
    total_t += tau_b * rho_b.pow(-5 / 3) * ((1 - zeta) / 2).pow(5 / 3)
    ratio = graph.multiply_many((sigma, density.pow(-4 / 3).pow(2))) / (8 * total_t)
    original = graph.select_le(ratio, threshold, ratio, 1)
    (changed,) = canonicalize_spin_density_coordinates(graph, (original,), rho_a, rho_b)
    if threshold == 1:
        product = density * (tau_a + tau_b)
        physical = graph.select_le(sigma, 8 * product, ratio, 1)
        expected = graph.select_le(
            product, float.fromhex("0x1.0p-1022"), original, physical
        )
        assert changed.identifier == expected.identifier
        assert graph.node(physical).arguments[2:] == graph.node(original).arguments[2:]
    else:
        assert changed.identifier == original.identifier
    (repeated,) = canonicalize_spin_density_coordinates(graph, (changed,), rho_a, rho_b)
    assert repeated.identifier == changed.identifier


@pytest.mark.parametrize(
    ("density", "tau", "sigma"),
    (
        (1e308, 2e-20, 2e289),
        (2e-15, 1e307, 2e293),
        (2e-15, 2e-20, 1e-35),
        (1e200, 1e200, 1e300),
        (1e-100, 1e-224, math.ulp(0.0)),
        (1e-100, 1e-223, math.ulp(0.0)),
        (1e-100, 1e-222, math.ulp(0.0)),
        (1.0, 0.125, math.nextafter(1.0, 0.0)),
        (1.0, 0.125, 1.0),
        (1.0, 0.125, math.nextafter(1.0, math.inf)),
    ),
)
def test_physical_predicate_preserves_positive_work_range(
    density: float, tau: float, sigma: float
) -> None:
    """Exact-rational branch oracles cover overflow ordering and adjacent sides."""
    graph = Graph()
    rho_a, rho_b = graph.variable("rho_a"), graph.variable("rho_b")
    sigma_aa, sigma_ab, sigma_bb = (
        graph.variable(name) for name in ("sigma_aa", "sigma_ab", "sigma_bb")
    )
    tau_a, tau_b = graph.variable("tau_a"), graph.variable("tau_b")
    total_density = rho_a + rho_b
    total_sigma = sigma_aa + 2 * sigma_ab + sigma_bb
    zeta = (rho_a - rho_b) / total_density
    total_t = tau_a * rho_a.pow(-5 / 3) * ((1 + zeta) / 2).pow(5 / 3)
    total_t += tau_b * rho_b.pow(-5 / 3) * ((1 - zeta) / 2).pow(5 / 3)
    ratio = graph.multiply_many((total_sigma, total_density.pow(-4 / 3).pow(2))) / (
        8 * total_t
    )
    original = graph.select_le(ratio, 1, 7, 3)
    (changed,) = canonicalize_spin_density_coordinates(graph, (original,), rho_a, rho_b)
    features = {
        "rho_a": density / 2,
        "rho_b": density / 2,
        "sigma_aa": sigma,
        "sigma_ab": 0.0,
        "sigma_bb": 0.0,
        "tau_a": tau / 2,
        "tau_b": tau / 2,
    }
    expected = Fraction(sigma) <= 8 * Fraction(density) * Fraction(tau)
    assert graph.evaluate(changed, features) == (7 if expected else 3)
    if density * tau <= float.fromhex("0x1.0p-1022"):
        # Preserve the original finite ratio when the physical product rounds
        # to a subnormal or zero; the first tiny row catches removing the guard.
        assert graph.evaluate(changed, features) == graph.evaluate(original, features)


@pytest.fixture(scope="module")
def native_point(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Any:
    """Compile current production point emission through the verified cache."""
    cpu = emit_automatic_point("MGGA_C_TPSS", backend="cpu")
    cuda = emit_automatic_point("MGGA_C_TPSS", backend="cuda")
    assert cpu.expression_identity == cuda.expression_identity
    folder = tmp_path_factory.mktemp("physical-kinetic-predicate")
    source = folder / "point.cpp"
    source.write_text(_native_source(cpu))
    binary = native_cxx.build_shared(
        [source],
        folder / "point.so",
        compile_args=(
            "-std=c++20",
            "-O2",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
    )
    library = ctypes.CDLL(str(binary))
    pointer = ctypes.POINTER(ctypes.c_double)
    library.evaluate.argtypes = (pointer, pointer, ctypes.c_size_t)
    library.evaluate.restype = ctypes.c_int
    return library


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_exact_input_and_empty_spin_coefficients_match_pinned_libxc(
    native_point: Any, case: dict[str, Any]
) -> None:
    """An independent pinned fixture catches both the original and rejected fixes."""
    actual = _evaluate(native_point, np.array([case["inputs"]]))[0]
    assert actual[11] == 1
    np.testing.assert_allclose(actual[:11], case["expected"], atol=2e-11, rtol=2e-9)


@pytest.mark.parametrize("fixed_electrons", (False, True))
def test_rank_one_fock_is_the_inward_psd_energy_derivative(
    native_point: Any, fixed_electrons: bool
) -> None:
    """Adding an unoccupied PSD component tests more than occupied-space tangents."""
    inputs = np.array([FIXTURE["cases"][0]["inputs"]])
    actual = _evaluate(native_point, inputs)[0]
    # Local AO values (1, 0), x-derivatives (1/8, 1), per-spin D=diag(1,0).
    # F_11 is precisely half-vtau. Occupied/tangent entries alone miss this bug.
    occupied = actual[1] + actual[3] / 4 + actual[9] / 64
    expected = actual[9] - (occupied if fixed_electrons else 0)
    assert expected < -3
    errors = []
    for step in (1e-5, 1e-6, 1e-7):
        displaced = inputs.copy()
        if fixed_electrons:
            # D_alpha(h)=diag(1-h,h) keeps trace and PSD for 0<=h<=1.
            displaced[0, 0] *= 1 - step
            displaced[0, 2] *= 1 - step
            displaced[0, 8] *= 1 - step
        displaced[0, 8] += step / 2
        slope = (_evaluate(native_point, displaced)[0, 0] - actual[0]) / step
        errors.append(abs(slope - expected))
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 2e-5
