"""Molecular source-cut contracts and complete native RHF nuclear derivatives."""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp

if TYPE_CHECKING:
    from pathlib import Path

COORDINATES = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])


@pytest.fixture
def polynomial_source(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Test orchestration with an explicit fake, not a QC numerical oracle."""
    owners = []

    class PolynomialEndpoint:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = False
            self.properties: list[Any] = []

        def singlepoint(self, atoms: Any, **options: Any) -> Any:
            self.calls += 1
            self.properties.append(options["properties"])
            coordinates = np.asarray([position for number, position in atoms])
            return SimpleNamespace(
                energy=float(np.sum(coordinates * coordinates)),
                forces=-2 * coordinates,
                converged=True,
                executed_backend="cpu_reference",
                iterations=3,
                energy_change=0.0,
                density_rms=0.0,
            )

        def clear_cache(self) -> None:
            self.closed = True

    def factory(self: Any) -> Any:
        owner = PolynomialEndpoint()
        owners.append(owner)
        return owner

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("native molecular frontend reached reference interpreter")

    from generativeqc_compiler.tensor import interpreter

    monkeypatch.setattr(xp.RHF, "_calculator", factory)
    monkeypatch.setattr(interpreter, "execute", forbidden)
    return owners


def test_lowering_is_native_free_and_configuration_is_immutable(
    tmp_path: Path, polynomial_source: list[Any]
) -> None:
    rhf = xp.RHF(("H", "H"))
    compiled = xp.value_and_grad(lambda coordinates: rhf(coordinates), cache=tmp_path)
    lowered = compiled.lower(COORDINATES)
    assert lowered.provenance["molecular_ad"] == "native-analytic-first-order-jet"
    assert len(lowered.provenance["molecular_sources"]) == 1
    assert not polynomial_source
    with pytest.raises(FrozenInstanceError):
        rhf._charge = 2
    value, gradient = compiled(COORDINATES)
    np.testing.assert_allclose(value, np.sum(COORDINATES * COORDINATES))
    np.testing.assert_allclose(gradient, 2 * COORDINATES)
    assert polynomial_source[0].calls == 1
    compiled.close()
    assert polynomial_source[0].closed
    with pytest.raises(RuntimeError, match="closed"):
        compiled(COORDINATES)


def test_compiled_geometry_and_nonlinear_source_chain_rule(
    tmp_path: Path, polynomial_source: list[Any]
) -> None:
    rhf = xp.RHF(("H", "H"))

    def expression(coordinates: Any, scale: Any) -> Any:
        energy = rhf(coordinates * scale)
        return energy * energy + xp.sum(coordinates * coordinates) / 2

    scale = np.asarray(1.2)
    compiled = xp.value_and_grad(expression, argnums=(0, 1), cache=tmp_path)
    value, (coordinate_gradient, scale_gradient) = compiled(COORDINATES, scale)
    norm = np.sum(COORDINATES * COORDINATES)
    energy = norm * scale * scale
    np.testing.assert_allclose(value, energy * energy + norm / 2, atol=1e-14)
    np.testing.assert_allclose(
        coordinate_gradient, 4 * energy * scale * scale * COORDINATES + COORDINATES
    )
    np.testing.assert_allclose(scale_gradient, 4 * energy * norm * scale)
    assert polynomial_source[0].calls == 1
    compiled.close()


def test_jvp_vjp_snapshot_and_repeated_source_deduplication(
    tmp_path: Path, polynomial_source: list[Any]
) -> None:
    rhf = xp.RHF(("H", "H"))

    def expression(coordinates: Any) -> Any:
        return {
            "energy": rhf(coordinates) + rhf(coordinates),
            "norm": xp.sum(coordinates),
        }

    coordinates = COORDINATES.copy()
    direction = np.arange(6, dtype=np.float64).reshape(2, 3) / 10
    value, response = xp.jvp(expression, (coordinates,), (direction,), cache=tmp_path)
    np.testing.assert_allclose(value["energy"], 2 * np.sum(coordinates * coordinates))
    np.testing.assert_allclose(response["energy"], np.sum(4 * coordinates * direction))
    np.testing.assert_allclose(response["norm"], np.sum(direction))
    assert polynomial_source[0].calls == 1
    assert polynomial_source[0].closed
    _value, pullback = xp.vjp(expression, coordinates, cache=tmp_path)
    coordinates[:] = 100
    for weight in (0.5, 2.0):
        (gradient,) = pullback({"energy": xp.asarray(weight), "norm": xp.asarray(0.3)})
        np.testing.assert_allclose(gradient, weight * 4 * COORDINATES + 0.3)
    assert polynomial_source[1].calls == 1
    pullback.close()
    assert polynomial_source[1].closed


def test_energy_only_recomputes_changed_geometry_and_prunes_unused_sources(
    tmp_path: Path, polynomial_source: list[Any]
) -> None:
    rhf = xp.RHF(("H", "H"))
    compiled = xp.compile(lambda coordinates: rhf(coordinates), cache=tmp_path)
    compiled(COORDINATES)
    moved = COORDINATES * 1.1
    np.testing.assert_allclose(compiled(moved), np.sum(moved * moved))
    assert polynomial_source[0].calls == 2
    assert polynomial_source[0].properties == [("energy",), ("energy",)]

    def inactive(coordinates: Any) -> Any:
        rhf(coordinates)
        return xp.sum(coordinates)

    unused = xp.compile(inactive, cache=tmp_path)
    np.testing.assert_allclose(unused(COORDINATES), 0.0)
    assert len(polynomial_source) == 1
    diagnostics = compiled.compilation_info()["molecular_calls"]
    assert diagnostics[0]["calls"] == 2
    assert diagnostics[0]["force_calls"] == 0
    assert diagnostics[0]["total_iterations"] == 6
    unused.close()
    compiled.close()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"elements": ("H",)}, "even electron"),
        ({"elements": ()}, "tuple"),
        ({"elements": ("H", "H"), "backend": "reference"}, "backend"),
        ({"elements": ("H", "H"), "max_iterations": 0}, "iteration"),
        ({"elements": ("H", "H"), "screening_tolerance": 0.0}, "screening"),
        ({"elements": ("H", "H"), "energy_tolerance": float("nan")}, "tolerance"),
    ],
)
def test_molecular_policy_fails_before_native_load(options: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        xp.RHF(**options)


def test_molecular_capture_gates_before_execution(
    tmp_path: Path, polynomial_source: list[Any]
) -> None:
    rhf = xp.RHF(("H", "H"))
    for backend in ("reference", "cuda"):
        compiled = xp.compile(lambda coordinates: rhf(coordinates), backend=backend)
        with pytest.raises(ValueError, match="same backend"):
            compiled.lower(COORDINATES)
    with pytest.raises(TypeError, match="float64"):
        xp.compile(lambda coordinates: rhf(coordinates)).lower(
            COORDINATES.astype(np.float32)
        )
    with pytest.raises(ValueError, match="shape"):
        xp.compile(lambda coordinates: rhf(coordinates)).lower(np.ones((3, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        xp.compile(lambda coordinates: rhf(coordinates))(COORDINATES * np.nan)
    with pytest.raises(ValueError, match="budget"):
        xp.value_and_grad(
            lambda coordinates: rhf(coordinates), max_work=0, cache=tmp_path
        )(COORDINATES)
    with pytest.raises(NotImplementedError, match="public compile"):
        xp.trace(
            lambda coordinates: rhf(coordinates),
            {"coordinates": xp.TensorSpec((), role="input")},
        )
    assert not polynomial_source


def test_molecular_source_count_nested_dependencies_and_invalid_geometry_are_bounded(
    tmp_path: Path, polynomial_source: list[Any]
) -> None:
    operators = tuple(
        xp.RHF(("H", "H"), max_iterations=count + 1) for count in range(17)
    )

    def excessive(coordinates: Any) -> Any:
        energy = operators[0](coordinates)
        for operator in operators[1:]:
            energy = energy + operator(coordinates)
        return energy

    with pytest.raises(ValueError, match="16-source budget"):
        xp.compile(excessive).lower(COORDINATES)

    def nested(coordinates: Any) -> Any:
        return operators[0](coordinates + operators[0](coordinates))

    with pytest.raises(NotImplementedError, match="nested molecular"):
        xp.compile(nested).lower(COORDINATES)
    invalid = xp.compile(
        lambda coordinates: operators[0](coordinates * coordinates), cache=tmp_path
    )
    with pytest.raises(ValueError, match="native CPU tensor evaluation failed"):
        invalid(np.full((2, 3), 1e200))
    invalid.close()
    assert not polynomial_source


def test_scientific_coordinate_domains_are_preserved_not_inferred_from_extents(
    tmp_path: Path, polynomial_source: list[Any]
) -> None:
    rhf = xp.RHF(("H", "H"))
    atoms = xp.IndexSpace("nuclei", "atom", 2)
    cartesian = xp.IndexSpace("xyz", "cartesian", 3)
    spec = xp.TensorSpec(
        (xp.Index("atom", atoms), xp.Index("component", cartesian)), role="input"
    )
    base = xp.compile(
        lambda coordinates: rhf(coordinates),
        specs={"coordinates": spec},
        cache=tmp_path,
    )
    differentiated = xp.value_and_grad(base)
    lowered = differentiated.lower(COORDINATES)
    assert lowered.outputs["bar_coordinates"].spec.indices == spec.indices
    value, gradient = differentiated(COORDINATES)
    np.testing.assert_allclose(value, np.sum(COORDINATES * COORDINATES))
    np.testing.assert_allclose(gradient, 2 * COORDINATES)
    differentiated.close()
    base.close()
    wrong_domain = xp.IndexSpace("ao", "ao", 2)
    bad_spec = xp.TensorSpec(
        (xp.Index("ao", wrong_domain), xp.Index("component", cartesian)), role="input"
    )
    with pytest.raises(ValueError, match="atom/cartesian domains"):
        xp.compile(
            lambda coordinates: rhf(coordinates), specs={"coordinates": bad_spec}
        ).lower(COORDINATES)
    assert len(polynomial_source) == 1


@pytest.mark.parametrize("compiled_call", [False, True])
@pytest.mark.parametrize(
    "changes, error_type",
    [
        ({"executed_backend": "cuda"}, RuntimeError),
        ({"converged": False}, RuntimeError),
        ({"energy": float("nan")}, ArithmeticError),
    ],
)
def test_molecular_publication_checks_apply_to_eager_and_compiled_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_call: bool,
    changes: Any,
    error_type: type[Exception],
) -> None:
    result = SimpleNamespace(
        energy=1.0,
        forces=np.ones((2, 3)),
        converged=True,
        executed_backend="cpu_reference",
    )
    for name, value in changes.items():
        setattr(result, name, value)
    owner = SimpleNamespace(
        singlepoint=lambda *args, **kwargs: result, clear_cache=lambda: None
    )
    monkeypatch.setattr(xp.RHF, "_calculator", lambda self: owner)
    rhf = xp.RHF(("H", "H"))
    compiled = xp.value_and_grad(lambda coordinates: rhf(coordinates), cache=tmp_path)
    with pytest.raises(error_type):
        (compiled if compiled_call else rhf)(COORDINATES)
    assert not compiled.compilation_info()["molecular_calls"]
    compiled.close()


@pytest.fixture
def native_library() -> None:
    """Require an actual CPU library; never replace absent science with a mock."""
    from generativeqc import _native

    if not any(candidate.exists() for candidate in _native._candidate_paths()):
        pytest.skip("complete native RHF acceptance requires a built/installed library")


@pytest.mark.parametrize(
    "elements, coordinates, charge, basis",
    [
        (("H", "H"), COORDINATES, 0, "sto-3g"),
        (("He", "H"), COORDINATES, 1, "def2-svp"),
        (
            ("O", "H", "H"),
            np.asarray([[0.0, 0.0, 0.0], [0.0, -1.43, 1.11], [0.0, 1.43, 1.11]]),
            0,
            "sto-3g",
        ),
    ],
)
def test_complete_rhf_energy_nuclear_gradient_and_independent_finite_difference(
    tmp_path: Path,
    native_library: None,
    elements: Any,
    coordinates: Any,
    charge: int,
    basis: str,
) -> None:
    from generativeqc import Calculator

    rhf = xp.RHF(elements, basis=basis, charge=charge)
    compiled = xp.value_and_grad(lambda coordinates: rhf(coordinates), cache=tmp_path)
    value, gradient = compiled(coordinates)
    calculator = Calculator(
        method="rhf", basis=basis, initial_guess=None, **dict(rhf._controls)
    )
    try:
        legacy = calculator.singlepoint(rhf._atoms(coordinates), charge=charge)
        np.testing.assert_allclose(value, legacy.energy, atol=1e-11)
        np.testing.assert_allclose(gradient, -legacy.forces, atol=1e-10)
        np.testing.assert_allclose(np.sum(gradient, axis=0), 0, atol=2e-9)
        if elements == ("H", "H"):
            np.testing.assert_allclose(value, -1.11671432506255, rtol=0, atol=2e-9)
        direction = np.random.default_rng(2268).normal(size=coordinates.shape)
        direction /= np.linalg.norm(direction)
        step = 1e-4
        energies = [
            calculator.singlepoint(
                rhf._atoms(coordinates + sign * step * direction),
                charge=charge,
                properties=("energy",),
            ).energy
            for sign in (-1, 1)
        ]
        difference = (energies[1] - energies[0]) / (2 * step)
        np.testing.assert_allclose(
            np.sum(gradient * direction), difference, rtol=0, atol=5e-7
        )
        assert compiled.compilation_info()["molecular_calls"][0]["calls"] == 1
        assert (
            compiled.compilation_info()["molecular_calls"][0]["iterations"]
            == legacy.iterations
        )
    finally:
        calculator.clear_cache()
        compiled.close()


def test_native_rhf_nonconvergence_publishes_no_source_snapshot(
    tmp_path: Path, native_library: None
) -> None:
    rhf = xp.RHF(("O", "H", "H"), max_iterations=1)
    compiled = xp.value_and_grad(lambda coordinates: rhf(coordinates), cache=tmp_path)
    coordinates = np.asarray([[0.0, 0.0, 0.0], [0.0, -1.43, 1.11], [0.0, 1.43, 1.11]])
    with pytest.raises(RuntimeError, match=r"converg|iteration"):
        compiled(coordinates)
    assert not compiled.compilation_info()["molecular_calls"]
    compiled.close()


def test_native_geometry_composition_and_snapshot_pullback(
    tmp_path: Path, native_library: None
) -> None:
    from generativeqc import Calculator

    rhf = xp.RHF(("H", "H"))

    def expression(coordinates: Any, scale: Any) -> Any:
        energy = rhf(coordinates * scale)
        return energy * energy + xp.sum(coordinates * coordinates) / 10

    coordinates = COORDINATES.copy()
    scale = np.asarray(1.1)
    compiled = xp.value_and_grad(expression, argnums=(0, 1), cache=tmp_path)
    calculator = Calculator(method="rhf", initial_guess=None, **dict(rhf._controls))
    try:
        reference = calculator.singlepoint(rhf._atoms(coordinates * scale))
        gradient = -reference.forces
        expected_coordinate = 2 * reference.energy * gradient * scale + coordinates / 5
        expected_scale = 2 * reference.energy * np.sum(gradient * coordinates)
        value, (coordinate_gradient, scale_gradient) = compiled(coordinates, scale)
        np.testing.assert_allclose(
            value,
            reference.energy * reference.energy
            + np.sum(coordinates * coordinates) / 10,
        )
        np.testing.assert_allclose(coordinate_gradient, expected_coordinate, atol=1e-10)
        np.testing.assert_allclose(scale_gradient, expected_scale, atol=1e-10)
        step = 1e-4
        energies = [
            calculator.singlepoint(
                rhf._atoms(coordinates * (scale + sign * step)), properties=("energy",)
            ).energy
            for sign in (-1, 1)
        ]
        difference = (energies[1] * energies[1] - energies[0] * energies[0]) / (
            2 * step
        )
        np.testing.assert_allclose(scale_gradient, difference, rtol=0, atol=5e-7)
        _, pullback = xp.vjp(expression, coordinates, scale, cache=tmp_path)
        try:
            coordinates[:] = 100
            scale[...] = 10
            for weight in (0.5, 2.0):
                coordinate_bar, scale_bar = pullback(xp.asarray(weight))
                np.testing.assert_allclose(
                    coordinate_bar, weight * expected_coordinate, atol=1e-10
                )
                np.testing.assert_allclose(
                    scale_bar, weight * expected_scale, atol=1e-10
                )
            assert (
                pullback.function.compilation_info()["molecular_calls"][0]["calls"] == 1
            )
        finally:
            pullback.close()
    finally:
        compiled.close()
        calculator.clear_cache()


def test_native_molecular_resource_budget_is_independent_of_tensor_budget(
    tmp_path: Path, native_library: None
) -> None:
    from generativeqc import ResourceBudget

    rhf = xp.RHF(("H", "H"), resource_budget=ResourceBudget(host_bytes=1))
    compiled = xp.value_and_grad(lambda coordinates: rhf(coordinates), cache=tmp_path)
    with pytest.raises(MemoryError, match="no supported plan fits"):
        compiled(COORDINATES)
    assert not compiled.compilation_info()["molecular_calls"]
    compiled.close()


def test_molecular_cuda_lowering_does_not_probe_native_or_device(
    polynomial_source: list[Any],
) -> None:
    rhf = xp.RHF(("H", "H"), backend="cuda")
    compiled = xp.value_and_grad(lambda coordinates: rhf(coordinates), backend="cuda")
    lowered = compiled.lower(COORDINATES)
    assert lowered.provenance["molecular_sources"][0]["backend"] == "cuda"
    assert not polynomial_source
    compiled.close()


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_ARRAY_API_CUDA_TEST") != "1",
    reason="real molecular CUDA acceptance requires explicit scheduled opt-in",
)
def test_real_cuda_molecular_endpoint(tmp_path: Path) -> None:
    assert os.environ.get("SLURM_JOB_ID")
    assert os.environ.get("SLURMD_NODENAME") in {"n1", "n2", "n4", "n5"}
    rhf = xp.RHF(("H", "H"), backend="cuda")
    compiled = xp.value_and_grad(
        lambda coordinates: rhf(coordinates), backend="cuda", cache=tmp_path
    )
    try:
        value, gradient = compiled(COORDINATES)
        np.testing.assert_allclose(value, -1.11671432506255, rtol=0, atol=2e-9)
        step = 1e-4
        plus, minus = COORDINATES.copy(), COORDINATES.copy()
        plus[1, 2] += step
        minus[1, 2] -= step
        difference = (rhf(plus) - rhf(minus)) / (2 * step)
        np.testing.assert_allclose(gradient[1, 2], difference, rtol=0, atol=5e-7)
        assert compiled.compilation_info()["molecular_calls"][0]["backend"] == "cuda"
    finally:
        compiled.close()
