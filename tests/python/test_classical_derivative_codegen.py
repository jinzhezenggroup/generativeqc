"""Independent energy and compiled-host gates for generated classical responses."""

import ctypes
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.geometry.dispersion_scalar import (
    SCALAR_PROGRAMS,
    dispersion_scalar_program,
    emit_dispersion_scalar_header,
)
from generativeqc_compiler.geometry.gcp_scalar import emit_gcp_scalar_header
from generativeqc_compiler.method.nuclear_repulsion import (
    emit_nuclear_repulsion_header,
    nuclear_repulsion_hvp,
)

from tools.generate_gfn2_pair_native import native_header as emit_gfn2_pair_header
from tools.generativeqc_d3.generate_native_data import render as render_d3_tables

ROOT = Path(__file__).resolve().parents[2]
POINTER = ctypes.POINTER(ctypes.c_double)
D3_CASES = json.loads((ROOT / "tests/data/d3_bj_reference.json").read_text())[
    "fixtures"
]


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    """Compile the actual host adapters through the verified compiler cache."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    directory = tmp_path_factory.mktemp("classical-derivatives")
    (directory / "generated_nuclear_repulsion.hpp").write_text(
        emit_nuclear_repulsion_header()
    )
    (directory / "generated_dispersion_scalar.hpp").write_text(
        emit_dispersion_scalar_header()
    )
    (directory / "d3_data.hpp").write_text(render_d3_tables())
    (directory / "generated_gcp_scalar.hpp").write_text(emit_gcp_scalar_header())
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_method_parameters.py"),
            "--cpp-output",
            str(directory / "generated_method_parameters.hpp"),
        ],
        check=True,
        capture_output=True,
    )
    cases = []
    for index, name in enumerate(SCALAR_PROGRAMS):
        _, names, _ = dispersion_scalar_program(name)
        arguments = ", ".join(f"input[{field}]" for field in range(len(names)))
        cases.append(f"case {index}: {name}({arguments}, output); break;")
    source = directory / "probe.cpp"
    source.write_text(
        "\n".join(
            [
                '#include "molecule/nuclear_gradient.hpp"',
                '#include "dft/dispersion/d4_math.hpp"',
                '#include "dft/dispersion/d3_bj.hpp"',
                '#include "dft/dispersion/gcp_r2scan3c.hpp"',
                'extern "C" void scalar(int kind, const double* input, double* output) {',
                "using namespace generativeqc::generated::dispersion;",
                "switch (kind) {",
                *cases,
                "}",
                "}",
                'extern "C" int nuclear(const double* positions, const double* charges,',
                "    int atoms, double* gradient, int count) {",
                "generativeqc::core::System system;",
                "for (int atom = 0; atom < atoms; ++atom)",
                "  system.atoms.push_back({static_cast<int>(charges[atom]) + 2,",
                "    {positions[3*atom], positions[3*atom+1], positions[3*atom+2]}, 2});",
                "try { generativeqc::molecule::add_nuclear_repulsion_gradient(",
                "  system, std::span<double>(gradient, count)); return 0;",
                "} catch (const std::invalid_argument&) { return 1; }",
                "}",
                'extern "C" void d4_charge(double qmod, double* output) {',
                "auto result = generativeqc::dft::dispersion::math::charge_scale(3.0, 2.0, 4.0, qmod);",
                "output[0] = result.value; output[1] = result.derivative;",
                "}",
                'extern "C" int d3(int atoms, const int* numbers, const double* centers,',
                "    const double* parameters, double* output) {",
                "using namespace generativeqc::dft::dispersion;",
                "D3Parameters model{parameters[0], parameters[1], parameters[2], parameters[3], 0.0};",
                "std::vector<double> workspace(d3_workspace_elements(atoms));",
                "return static_cast<int>(evaluate_d3_bj(atoms, numbers, centers, model,",
                "  d3_host_tables(), workspace.data(), workspace.size(), output, output + 1));",
                "}",
                'extern "C" int gcp_overlap(int first_shell, int second_shell,',
                "    double first_exponent, double second_exponent, double distance, double* output) {",
                "return generativeqc::dft::dispersion::gcp_detail::overlap(distance, first_shell,",
                "  second_shell, first_exponent, second_exponent, output, output + 1);",
                "}",
                'extern "C" int gcp(int atoms, const int* numbers, const double* centers, double* output) {',
                "using namespace generativeqc::dft::dispersion;",
                "return static_cast<int>(evaluate_r2scan3c_gcp(atoms, numbers, centers,",
                "  r2scan3c_gcp_parameters(), output, output + 1));",
                "}",
            ]
        )
    )
    library = directory / "probe.so"
    CppCompilerAdapter(Path(compiler), compile_timeout=120).compile_shared(
        source,
        library,
        includes=(directory, ROOT / "src"),
        options=("-std=c++20", "-ffp-contract=off"),
    )
    result = ctypes.CDLL(str(library))
    result.scalar.argtypes = (ctypes.c_int, POINTER, POINTER)
    result.scalar.restype = None
    result.nuclear.argtypes = (POINTER, POINTER, ctypes.c_int, POINTER, ctypes.c_int)
    result.nuclear.restype = ctypes.c_int
    result.d4_charge.argtypes = (ctypes.c_double, POINTER)
    result.d4_charge.restype = None
    result.d3.argtypes = (
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        POINTER,
        POINTER,
        POINTER,
    )
    result.d3.restype = ctypes.c_int
    result.gcp_overlap.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        POINTER,
    )
    result.gcp_overlap.restype = ctypes.c_int
    result.gcp.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int), POINTER, POINTER)
    result.gcp.restype = ctypes.c_int
    return result


def _pointer(array: np.ndarray) -> POINTER:
    return array.ctypes.data_as(POINTER)


def _evaluate(native: ctypes.CDLL, name: str, values: tuple) -> np.ndarray:
    inputs = np.array(values, dtype=np.float64)
    result = np.zeros(len(dispersion_scalar_program(name)[2]))
    native.scalar(SCALAR_PROGRAMS.index(name), _pointer(inputs), _pointer(result))
    return result


@pytest.mark.parametrize("distance", (0.1, 1.7, 100.0))
@pytest.mark.parametrize("fraction", (-0.8, 0.0, 0.3))
def test_cached_cartesian_norm_pullback_against_primal_only_oracle(
    native: ctypes.CDLL, distance: float, fraction: float
) -> None:
    """Qualify symbolic radial cancellation at zero Cartesian displacement too."""
    mp = pytest.importorskip("mpmath").mp.clone()
    mp.dps = 80
    radius = mp.mpf(distance)
    component = radius * mp.mpf(fraction)
    other_squared = radius * radius - component * component
    seed = mp.mpf("-1.37")
    primal = lambda value: seed * mp.sqrt(value * value + other_squared)
    expected = float(mp.diff(primal, component))
    actual = _evaluate(
        native, "radial_projection", (distance, float(seed), float(component))
    )[0]
    assert actual == pytest.approx(expected, rel=4e-15, abs=1e-16)
    coefficient = _evaluate(native, "radial_scatter", (distance, float(seed)))[0]
    assert coefficient * float(component) == pytest.approx(
        expected, rel=4e-15, abs=1e-16
    )


@pytest.mark.parametrize(
    "script,emit",
    [
        ("generate_nuclear_repulsion.py", emit_nuclear_repulsion_header),
        ("generate_dispersion_scalar.py", emit_dispersion_scalar_header),
        ("generate_gcp_scalar.py", emit_gcp_scalar_header),
        ("generate_gfn2_pair_native.py", emit_gfn2_pair_header),
    ],
)
@pytest.mark.parametrize("python_flags", [(), ("-I", "-S")])
def test_uninstalled_generator_is_deterministic_from_another_directory(
    tmp_path: Path,
    script: str,
    emit: Callable[[], str],
    python_flags: tuple[str, ...],
) -> None:
    """Build-time generators need neither site packages nor inherited paths."""
    output = tmp_path / "generated.hpp"
    subprocess.run(
        [
            sys.executable,
            *python_flags,
            str(ROOT / "tools" / script),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert output.read_text() == emit()


@pytest.mark.parametrize(
    "name,values,energy,scale",
    [
        (
            "gaussian",
            (1.7, 2.3, 4.0),
            lambda value: math.exp(-4 * (value - 2.3) ** 2),
            1.0,
        ),
        (
            "smooth_switch",
            (8.5, 10.0, 2.0),
            lambda value: (
                ((10 - value) / 2) ** 3
                * (10 + ((10 - value) / 2) * (-15 + 6 * ((10 - value) / 2)))
            ),
            1.0,
        ),
        (
            "d3_coordination",
            (2.4, 2.1),
            lambda value: 1 / (1 + math.exp(-16 * (2.1 / value - 1))),
            1 / 2.4,
        ),
        (
            "bj_damping",
            (7.0, 2.0, 3.5, 1.0, 0.7),
            lambda value: 1 / (value**3 + 3.5**6) + 1.4 / (value**4 + 3.5**8),
            2.0,
        ),
        (
            "d4_coordination",
            (2.4, 2.1, 0.9),
            lambda value: 0.45 * (1 + math.erf(7.5 * (2.1 - value) / 2.1)),
            1.0,
        ),
        (
            "eeq_coulomb",
            (2.4, 0.7),
            lambda value: math.erf(0.7 * value) / value,
            1 / 2.4,
        ),
        (
            "eeq_rhs",
            (1.7, 1.4, 0.6),
            lambda value: (
                -1.4
                + 0.6
                * (math.log1p(math.exp(8)) - math.log1p(math.exp(8 - value)))
                / math.sqrt(
                    math.log1p(math.exp(8)) - math.log1p(math.exp(8 - value)) + 1e-14
                )
            ),
            1.0,
        ),
        (
            "zero_regular",
            (2.4, 3.1, 14.0, 6.0, (3.1 / 2.4) ** 14),
            lambda value: value**-6 / (1 + 6 * (3.1 / value) ** 14),
            1 / 2.4,
        ),
    ],
)
def test_scalar_derivatives_against_independent_energies(
    native: ctypes.CDLL,
    name: str,
    values: tuple[float, ...],
    energy: Callable[[float], float],
    scale: float,
) -> None:
    actual = _evaluate(native, name, values)
    assert actual[0] == pytest.approx(energy(values[0]), rel=3e-14, abs=1e-15)
    for step in (1e-4, 3e-5):
        point = values[0]
        derivative = (
            energy(point - 2 * step)
            - 8 * energy(point - step)
            + 8 * energy(point + step)
            - energy(point + 2 * step)
        ) / (12 * step)
        assert actual[1] == pytest.approx(scale * derivative, rel=3e-8, abs=1e-10)


def test_normalization_and_product_directional_responses(native: ctypes.CDLL) -> None:
    values = (0.3, 0.7, 0.4, -0.2)
    for name, energy in (
        (
            "normalize",
            lambda parameter: (0.3 + 0.4 * parameter) / (0.7 - 0.2 * parameter),
        ),
        (
            "product_response",
            lambda parameter: (0.3 + 0.4 * parameter) * (0.7 - 0.2 * parameter),
        ),
    ):
        actual = _evaluate(native, name, values)
        assert actual[0] == pytest.approx(energy(0), abs=1e-15)
        assert actual[1] == pytest.approx(
            (energy(1e-5) - energy(-1e-5)) / 2e-5, abs=1e-10
        )


@pytest.mark.parametrize("exponent", [12.0, 16.0, 19.5])
@pytest.mark.parametrize("absolute", [False, True])
def test_atm_responses_from_independent_cosine_energy(
    native: ctypes.CDLL, exponent: float, absolute: bool
) -> None:
    values = np.array(
        [9.0, 13.0, 10.0, 31.0, -23.0 if absolute else 23.0, 18.0, 12.0, 1.2, exponent]
    )

    def energy(inputs: np.ndarray) -> float:
        lengths = np.sqrt(inputs[:3])
        cosines = [
            (inputs[first] + inputs[second] - inputs[opposite])
            / (2 * lengths[first] * lengths[second])
            for first, second, opposite in ((0, 1, 2), (0, 2, 1), (1, 2, 0))
        ]
        length_product = math.prod(lengths)
        damping = 1 / (1 + 6 * (inputs[3] / length_product) ** (inputs[8] / 3))
        return (
            inputs[7]
            * math.sqrt(abs(math.prod(inputs[4:7])))
            * (1 + 3 * math.prod(cosines))
            / length_product**3
            * damping
        )

    actual = _evaluate(native, "atm_absolute" if absolute else "atm", values)
    assert actual[0] == pytest.approx(energy(values), abs=2e-18, rel=2e-14)
    for output, argument in enumerate((0, 1, 2, 4, 5, 6), start=1):
        step = np.zeros_like(values)
        step[argument] = 1e-4
        expected = (
            energy(values - 2 * step)
            - 8 * energy(values - step)
            + 8 * energy(values + step)
            - energy(values + 2 * step)
        ) / (12e-4)
        if argument < 3:
            expected *= 2
        assert actual[output] == pytest.approx(expected, rel=2e-8, abs=2e-13)


def test_atm_switch_preserves_each_edge_direction(native: ctypes.CDLL) -> None:
    radial = np.array([0.3, -0.4, 0.2])
    weights = np.array([0.4, 0.7, 0.6])
    slopes = np.array([-0.1, -0.2, -0.3])
    distances = np.array([1.5, 2.5, 3.5])
    actual = _evaluate(
        native, "atm_switch", (0.5, *radial, *weights, *slopes, *distances)
    )
    assert actual[0] == pytest.approx(0.5 * math.prod(weights), abs=1e-16)
    for edge in range(3):

        def energy(direction: float, edge_index: int = edge) -> float:
            changed = weights.copy()
            changed[edge_index] += (
                direction * slopes[edge_index] / distances[edge_index]
            )
            return (0.5 + direction * radial[edge_index]) * math.prod(changed)

        assert actual[edge + 1] == pytest.approx(
            (energy(1e-5) - energy(-1e-5)) / 2e-5, abs=2e-12
        )


@pytest.mark.parametrize(
    "first_shell,second_shell",
    [(first, second) for first in (1, 2, 3) for second in (1, 2, 3)],
)
@pytest.mark.parametrize(
    "first_exponent,second_exponent", [(1.3, 1.3), (1.3, 1.34), (1.1, 2.7)]
)
def test_gcp_overlap_shells_against_pinned_reference_and_energy_differences(
    native: ctypes.CDLL,
    first_shell: int,
    second_shell: int,
    first_exponent: float,
    second_exponent: float,
) -> None:
    from tools.generativeqc_gcp.reference import _overlap

    for distance in (2.5, 4.0, 7.0):
        actual = np.zeros(2)
        assert (
            native.gcp_overlap(
                first_shell,
                second_shell,
                first_exponent,
                second_exponent,
                distance,
                _pointer(actual),
            )
            == 1
        )
        expected = _overlap(
            distance, first_shell, second_shell, first_exponent, second_exponent
        )
        np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=1e-9)
        step = 3e-4
        value = lambda shift, center=distance: _overlap(
            center + shift, first_shell, second_shell, first_exponent, second_exponent
        )[0]
        finite_difference = (
            value(-2 * step) - 8 * value(-step) + 8 * value(step) - value(2 * step)
        ) / (12 * step)
        assert actual[1] == pytest.approx(finite_difference, abs=2e-9, rel=2e-8)


def test_complete_gcp_against_independent_upstream_reference(
    native: ctypes.CDLL,
) -> None:
    from tools.generativeqc_gcp.reference import evaluate_r2scan3c_gcp

    numbers = np.array([6, 8, 1, 13], dtype=np.int32)
    positions = np.array(
        [[0.0, 0.0, 0.0], [2.5, 0.1, 0.0], [0.6, 2.7, 0.2], [-1.2, 0.8, 2.4]]
    )
    result = np.zeros(1 + positions.size)
    assert (
        native.gcp(
            len(numbers),
            numbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            _pointer(positions),
            _pointer(result),
        )
        == 0
    )
    expected = evaluate_r2scan3c_gcp(numbers, positions)
    assert result[0] == pytest.approx(expected.energy, abs=2e-13, rel=2e-11)
    np.testing.assert_allclose(
        result[1:].reshape(positions.shape), expected.gradient, atol=2e-11, rtol=2e-9
    )


@pytest.mark.parametrize("charge", [-1.0, 0.0, 1e-300, 0.5, 2.5, 4.5, 100.0])
def test_charge_scaling_saturation_and_response(
    native: ctypes.CDLL, charge: float
) -> None:
    actual = np.zeros(2)
    native.d4_charge(charge, _pointer(actual))
    assert np.all(np.isfinite(actual))
    if charge <= 0:
        np.testing.assert_allclose(actual, (math.exp(3), 0), atol=0, rtol=2e-15)
    elif charge > 1e-200:
        energy = lambda value: math.exp(3 * (1 - math.exp(2 * (1 - 4 / value))))
        assert actual[0] == pytest.approx(energy(charge), abs=2e-14)
        step = 1e-5 * charge
        assert actual[1] == pytest.approx(
            (energy(charge + step) - energy(charge - step)) / (2 * step), abs=2e-8
        )


def test_nuclear_gradient_and_hvp_from_independent_pair_energy(
    native: ctypes.CDLL,
) -> None:
    random = np.random.default_rng(7713)
    centers = random.normal(size=(5, 3)) * 2
    charges = np.array([1, 6, 2, 8, 3], dtype=np.float64)
    direction = random.normal(size=centers.shape)

    def energy(positions: np.ndarray) -> float:
        return sum(
            charges[first]
            * charges[second]
            / np.linalg.norm(positions[first] - positions[second])
            for first in range(len(charges))
            for second in range(first)
        )

    def gradient(positions: np.ndarray) -> np.ndarray:
        result = np.zeros_like(positions)
        for index in np.ndindex(positions.shape):
            shift = np.zeros_like(positions)
            shift[index] = 1e-4
            result[index] = (
                energy(positions - 2 * shift)
                - 8 * energy(positions - shift)
                + 8 * energy(positions + shift)
                - energy(positions + 2 * shift)
            ) / (12e-4)
        return result

    accumulated = np.full_like(centers, 0.25)
    assert (
        native.nuclear(
            _pointer(centers),
            _pointer(charges),
            len(charges),
            _pointer(accumulated),
            centers.size,
        )
        == 0
    )
    np.testing.assert_allclose(
        accumulated - 0.25, gradient(centers), atol=3e-10, rtol=1e-9
    )
    step = 1e-3
    expected = (
        gradient(centers + step * direction) - gradient(centers - step * direction)
    ) / (2 * step)
    actual = nuclear_repulsion_hvp(centers, charges, direction)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=3e-5)
    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-14)
    np.testing.assert_allclose(
        nuclear_repulsion_hvp(centers, charges, np.ones_like(centers)), 0, atol=0
    )


def test_nuclear_invalid_geometry_and_shape(native: ctypes.CDLL) -> None:
    centers = np.zeros((2, 3))
    charges = np.ones(2)
    result = np.zeros_like(centers)
    assert (
        native.nuclear(_pointer(centers), _pointer(charges), 2, _pointer(result), 5)
        == 1
    )
    assert (
        native.nuclear(_pointer(centers), _pointer(charges), 2, _pointer(result), 6)
        == 1
    )
    for invalid in (0.0, math.inf, math.nan):
        centers[1, 0] = invalid
        with pytest.raises(ValueError, match="distinct finite"):
            nuclear_repulsion_hvp(centers, charges, np.zeros_like(centers))


@pytest.mark.parametrize("case", D3_CASES, ids=lambda case: case["name"])
def test_complete_native_d3_against_pinned_independent_goldens(
    native: ctypes.CDLL, case: dict
) -> None:
    numbers = np.array(case["numbers"], dtype=np.int32)
    centers = np.array(case["positions"], dtype=np.float64)
    parameters = np.array(
        [case["parameters"][name] for name in ("s6", "s8", "a1", "a2")]
    )
    output = np.zeros(1 + centers.size)
    status = native.d3(
        len(numbers),
        numbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _pointer(centers),
        _pointer(parameters),
        _pointer(output),
    )
    assert status == 0
    assert output[0] == pytest.approx(case["energy"], abs=2e-15, rel=0)
    np.testing.assert_allclose(
        output[1:].reshape(centers.shape), case["gradient"], atol=2e-14, rtol=0
    )
