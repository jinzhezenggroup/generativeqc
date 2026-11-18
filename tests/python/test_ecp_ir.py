"""ECP scientific identities, derivative rules, and strict lowering boundaries."""

import math
import subprocess
import sys
import typing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral.ecp import build_ecp_ir, gaussian_roots
from generativeqc_compiler.integral.ecp_grid import (
    harmonic_roots,
    radial_map_roots,
)
from generativeqc_compiler.integral.ecp_projector import (
    column_roots,
    direction_roots,
    emit_ecp_quadrature_cpp,
    pair_roots,
    physical_second_roots,
    radial_roots,
    second_pair_roots,
)
from generativeqc_compiler.integral.ir import EcpRadialTerm, OperatorSpec
from generativeqc_compiler.integral.ir_serialization import (
    integral_from_payload,
    integral_to_payload,
)
from generativeqc_compiler.integral.quadrature import legendre_step_roots
from generativeqc_compiler.integral.shell_spec import cartesian_components


def test_frozen_ecp_direction_keeps_independent_relative_slots() -> None:
    """Project the canonical center gradients only after independent-slot AD."""
    graph, (root,) = direction_roots()
    for seed in range(7):
        gradient = [math.sin(0.3 + seed + index) for index in range(6)]
        first = [math.cos(0.2 + seed + axis) for axis in range(3)]
        second = [math.sin(0.4 - seed + axis) for axis in range(3)]
        center = [math.cos(0.7 - seed + axis) for axis in range(3)]
        values = {f"gradient{index}": value for index, value in enumerate(gradient)}
        values.update({f"u0_{axis}": value for axis, value in enumerate(first)})
        values.update({f"u1_{axis}": value for axis, value in enumerate(second)})
        values.update({f"center{axis}": value for axis, value in enumerate(center)})
        expected = sum(
            gradient[axis] * first[axis]
            + gradient[3 + axis] * second[axis]
            - (gradient[axis] + gradient[3 + axis]) * center[axis]
            for axis in range(3)
        )
        assert graph.evaluate(root, values) == pytest.approx(
            expected, abs=3e-15, rel=3e-15
        )
        for axis in range(3):
            values[f"u0_{axis}"] = values[f"u1_{axis}"] = center[axis]
        assert graph.evaluate(root, values) == 0


def test_ecp_codegen_without_site_packages(tmp_path: typing.Any) -> None:
    """CPU builds generate this header before installing Python dependencies."""
    generator = Path(__file__).resolve().parents[2] / "tools/generate_ecp_kernels.py"
    output = tmp_path / "generated" / "generated_ecp_ao.cuh"
    subprocess.run(
        [sys.executable, "-I", "-S", str(generator), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert output.read_text() == emit_ecp_quadrature_cpp()


def test_generated_grid_radial_map_and_jacobian() -> None:
    graph, (radius, weight) = radial_map_roots()
    for z in (-0.999, -0.7, 0.0, 0.4, 0.999):
        values = {"z": z, "weight": 0.37}
        # Independent rational form and its analytic Jacobian.
        assert graph.evaluate(radius, values) == pytest.approx((1 + z) / (1 - z))
        assert graph.evaluate(weight, values) == pytest.approx(0.74 / (1 - z) ** 2)
        derivative = graph.differentiate(radius, graph.variable("z"))
        assert graph.evaluate(weight, values) == pytest.approx(
            0.37 * graph.evaluate(derivative, values)
        )


@pytest.mark.parametrize("order", (8, 16, 32, 96, 256, 512))
def test_legendre_recurrence_joint_ad(order: int) -> None:
    """Compare all loop-carried AD channels to an independent Legendre basis."""
    from numpy.polynomial.legendre import legder, legval

    graph, (following, tangent) = legendre_step_roots()
    coefficients = np.zeros(order + 1)
    coefficients[-1] = 1
    for coordinate in (-1.0, -0.999, -0.43, 0.0, 0.27, 0.999, 1.0):
        value, previous, derivative, previous_derivative = 1.0, 0.0, 0.0, 0.0
        for degree in range(1, order + 1):
            fields = {
                "k": degree,
                "z": coordinate,
                "p": value,
                "previous": previous,
                "dp": derivative,
                "dprevious": previous_derivative,
            }
            next_value = graph.evaluate(following, fields)
            next_derivative = graph.evaluate(tangent, fields)
            previous, value = value, next_value
            previous_derivative, derivative = derivative, next_derivative
        assert value == pytest.approx(
            legval(coordinate, coefficients), abs=3e-12, rel=3e-12
        )
        assert derivative == pytest.approx(
            legval(coordinate, legder(coefficients)), abs=2e-8, rel=2e-11
        )


def test_generated_harmonic_channel_addition_theorem() -> None:
    graph, roots = harmonic_roots()
    rng = np.random.default_rng(17103)
    for _ in range(30):
        a, b = rng.normal(size=(2, 3))
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        ya, yb = (
            np.array(
                [graph.evaluate(r, dict(zip("xyz", v, strict=True))) for r in roots]
            )
            for v in (a, b)
        )
        dot = a @ b
        for angular_momentum, polynomial in enumerate(
            (1.0, dot, (3 * dot * dot - 1) / 2, (5 * dot**3 - 3 * dot) / 2)
        ):
            selected = slice(
                angular_momentum * angular_momentum, (angular_momentum + 1) ** 2
            )
            assert ya[selected] @ yb[selected] == pytest.approx(
                (2 * angular_momentum + 1) / (4 * math.pi) * polynomial, abs=3e-15
            )


@pytest.mark.parametrize("weighted", [False, True])
def test_ecp_center_and_versioned_roundtrip(weighted: typing.Any) -> None:
    ir = build_ecp_ir(
        (1, 2),
        (EcpRadialTerm(-1, 2, 0.7, -3), EcpRadialTerm(3, 0, 1.5, 2)),
        derivatives=True,
        weighted=weighted,
    )
    assert len(ir.signature.shells) == 2
    assert ir.operator.centers == (0, 1, 2)
    assert ir.derivative.recovered_centers(ir.operator) == (2,)
    payload = integral_to_payload(ir)
    assert payload["schema_version"] == 4
    assert integral_from_payload(payload) == ir
    old = {**payload, "schema_version": 1}
    with pytest.raises(ValueError, match="schema version 4"):
        integral_from_payload(old)
    changed = replace(
        ir.operator.external_centers[0], terms=(EcpRadialTerm(-1, 2, 0.8, -3),)
    )
    assert (
        integral_to_payload(
            replace(ir, operator=replace(ir.operator, external_centers=(changed,)))
        )
        != payload
    )
    with pytest.raises(ValueError):
        OperatorSpec("nuclear_attraction", (0, 1, 2), external_centers=(changed,))


def test_generated_center_derivatives_independent_finite_difference() -> None:
    rng = np.random.default_rng(171)
    for l in range(4):
        for component in cartesian_components(l):
            graph, roots = gaussian_roots(component)
            for xyz in (np.zeros(3), rng.normal(size=3)):
                alpha = 0.67
                values = dict(zip("xyz", xyz, strict=True), alpha=alpha)
                actual = np.array([graph.evaluate(root, values) for root in roots])

                def reference(
                    x: typing.Any,
                    alpha: typing.Any = alpha,
                    component: typing.Any = component,
                ) -> typing.Any:
                    return math.exp(-alpha * (x @ x)) * math.prod(
                        x[i] ** component.count(a) for i, a in enumerate("xyz")
                    )

                assert actual[0] == pytest.approx(reference(xyz), abs=1e-14)
                for axis in range(3):
                    delta = np.eye(3)[axis] * 1e-5
                    # A-center derivative is minus the electronic-coordinate derivative.
                    expected = (reference(xyz - delta) - reference(xyz + delta)) / 2e-5
                    assert actual[axis + 1] == pytest.approx(expected, abs=2e-9)


def test_second_ao_roots_against_independent_polynomial_products() -> None:
    """All twenty Cartesian components retain exact center Hessians at nodes."""
    from numpy.polynomial import Polynomial

    rng = np.random.default_rng(2671)
    alpha = 0.67
    for angular in range(4):
        for component in cartesian_components(angular):
            graph, roots = gaussian_roots(component, order=2)
            for coordinates in (np.zeros(3), rng.normal(size=3)):
                actual = np.array(
                    [
                        graph.evaluate(
                            root,
                            dict(zip("xyz", coordinates, strict=True), alpha=alpha),
                        )
                        for root in roots
                    ]
                )
                axis_values = []
                for label, coordinate in zip("xyz", coordinates, strict=True):
                    polynomial = Polynomial([0] * component.count(label) + [1])
                    primal = polynomial(coordinate)
                    first = (
                        polynomial.deriv()(coordinate) - 2 * alpha * coordinate * primal
                    )
                    second = (
                        polynomial.deriv(2)(coordinate)
                        - 4 * alpha * coordinate * polynomial.deriv()(coordinate)
                        + (4 * alpha**2 * coordinate**2 - 2 * alpha) * primal
                    )
                    axis_values.append((primal, -first, second))
                exponential = np.exp(-alpha * (coordinates @ coordinates))
                expected = [
                    exponential * np.prod([values[0] for values in axis_values])
                ]
                for axis in range(3):
                    expected.append(
                        exponential
                        * np.prod(
                            [
                                values[int(index == axis)]
                                for index, values in enumerate(axis_values)
                            ]
                        )
                    )
                for row in range(3):
                    for column in range(row, 3):
                        expected.append(
                            exponential
                            * np.prod(
                                [
                                    values[int(index == row) + int(index == column)]
                                    for index, values in enumerate(axis_values)
                                ]
                            )
                        )
                np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-13)
    for order in (0, 3):
        with pytest.raises(ValueError):
            gaussian_roots("", order=order)


def test_second_pair_roots_and_column_projection_against_independent_blocks() -> None:
    """Canonical bilinear products and right actions have no occupancy factor."""
    rng = np.random.default_rng(2627)
    pair_graph, pair = second_pair_roots()
    column_graph, column = column_roots()
    physical_graph, physical = physical_second_roots()
    for weight in (1.0, -0.31, 0.0):
        first, second = rng.normal(size=(2, 10))
        values = {
            f"{slot}{index}": value
            for slot, array in (("a", first), ("b", second))
            for index, value in enumerate(array)
        }
        values["weight"] = weight
        actual = np.array([pair_graph.evaluate(root, values) for root in pair])
        gradient = (
            np.concatenate((first[1:4] * second[0], second[1:4] * first[0])) * weight
        )
        hessian = np.empty((6, 6))
        for row in range(3):
            for other in range(3):
                field = 4 + next(
                    index
                    for index, pair in enumerate(
                        (row, column) for row in range(3) for column in range(row, 3)
                    )
                    if pair == (min(row, other), max(row, other))
                )
                hessian[row, other] = weight * first[field] * second[0]
                hessian[3 + row, 3 + other] = weight * second[field] * first[0]
                hessian[row, 3 + other] = hessian[3 + other, row] = (
                    weight * first[1 + row] * second[1 + other]
                )
        np.testing.assert_allclose(
            actual,
            np.concatenate(
                ([weight * first[0] * second[0]], gradient, hessian.ravel())
            ),
            atol=2e-15,
        )
        right = rng.normal(size=(3, 3))
        relative = (right[:2] - right[2]).ravel()
        fields = {f"jet{index}": value for index, value in enumerate(actual)}
        fields.update(
            weight=1.0,
            weight_right=0.0,
            **{
                f"right{slot}_{axis}": right[slot, axis]
                for slot in range(2)
                for axis in range(3)
            },
            **{f"center{axis}": right[2, axis] for axis in range(3)},
        )
        projected = [column_graph.evaluate(root, fields) for root in column]
        np.testing.assert_allclose(
            projected,
            np.concatenate(
                (actual[:1], [gradient @ relative], gradient, hessian @ relative)
            ),
            atol=3e-15,
        )
        for weight_right in (0.23, -0.11):
            fields["weight_right"] = weight_right
            expected = np.asarray(projected).copy()
            expected[1] += weight_right * actual[0]
            expected[8:] += weight_right * gradient
            np.testing.assert_allclose(
                [column_graph.evaluate(root, fields) for root in column],
                expected,
                atol=3e-15,
            )
        fields["weight_right"] = 0.0
        incidence = rng.integers(-1, 2, size=(2, 6)).astype(float)
        fields.update(
            {
                f"{slot}{axis}": incidence[index, axis]
                for index, slot in enumerate(("first", "second"))
                for axis in range(6)
            }
        )
        np.testing.assert_allclose(
            [physical_graph.evaluate(root, fields) for root in physical],
            [gradient @ incidence[0], incidence[0] @ hessian @ incidence[1]],
            atol=5e-15,
        )


def test_invalid_ecp_lowerings_fail_closed() -> None:
    term = EcpRadialTerm(-1, 2, 1.0, 1.0)
    for angular in ((0,), (0, 4)):
        with pytest.raises(ValueError):
            build_ecp_ir(angular, (term,))
    with pytest.raises(ValueError):
        build_ecp_ir((0, 0), (term,), weighted=True)
    for kwargs in (
        {"power": -1},
        {"channel": 4},
        {"exponent": 0},
        {"coefficient": float("nan")},
    ):
        with pytest.raises(ValueError):
            replace(term, **kwargs)


@pytest.mark.parametrize("angular", [(0, 3), (3, 2), (3, 3)])
@pytest.mark.parametrize("weighted", [False, True])
def test_f_orbital_ir_derivative_contract(
    angular: typing.Any, weighted: typing.Any
) -> None:
    ir = build_ecp_ir(
        angular,
        (EcpRadialTerm(2, 4, 0.8, -1.2),),
        derivatives=True,
        weighted=weighted,
    )
    assert ir.derivative.recovered_centers(ir.operator) == (2,)
    assert integral_from_payload(integral_to_payload(ir)) == ir


@pytest.mark.parametrize("power", range(5))
def test_radial_measure_and_operator_contract(power: typing.Any) -> None:
    graph, (root,) = radial_roots(power)
    for r in (0.0, 1e-8, 0.37, 1.8, 17.0):
        values = {"r": r, "alpha": 0.63, "coefficient": -1.23, "weight": 0.41}
        expected = -1.23 * 0.41 * r**power * math.exp(-0.63 * r**2)
        assert graph.evaluate(root, values) == pytest.approx(expected, abs=1e-15)
    for bad in (-1, 5, True, 1.5):
        with pytest.raises(ValueError):
            radial_roots(bad)


def test_projector_pair_jets_against_displaced_bilinear() -> None:
    graph, roots = pair_roots()
    rng = np.random.default_rng(17102)
    for _ in range(20):
        a, b = rng.normal(size=(2, 4))
        weight = rng.normal()
        values = {
            f"{ab}{d}": jet[d] for ab, jet in (("a", a), ("b", b)) for d in range(4)
        }
        actual = np.array(
            [graph.evaluate(root, dict(values, weight=weight)) for root in roots]
        )
        assert actual[0] == pytest.approx(weight * a[0] * b[0])
        for d in range(1, 4):
            step = 1e-5
            da = (
                weight * (a[0] + step * a[d]) * b[0]
                - weight * (a[0] - step * a[d]) * b[0]
            ) / (2 * step)
            db = (
                weight * a[0] * (b[0] + step * b[d])
                - weight * a[0] * (b[0] - step * b[d])
            ) / (2 * step)
            np.testing.assert_allclose(
                actual[[d, d + 3]], [da, db], atol=2e-10, rtol=1e-9
            )
