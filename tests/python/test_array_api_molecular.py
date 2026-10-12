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


def test_molecular_public_api_has_one_entry_not_a_rhf_alias() -> None:
    """Only the unified method-neutral source is public."""
    assert xp.MolecularModel.__name__ == "MolecularModel"
    assert not hasattr(xp, "RHF")
    assert xp.capabilities()["molecular_operators"] == ("MolecularModel",)


def _require_scheduled_cuda() -> None:
    """Remote Slurm names differ from SSH aliases; node3 stays excluded."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    assert os.environ.get("SLURMD_NODENAME") in {
        "n1",
        "n2",
        "n4",
        "n5",
        "node1",
        "node2",
        "node4",
        "node5",
    }


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

    monkeypatch.setattr(xp.MolecularModel, "_calculator", factory)
    monkeypatch.setattr(interpreter, "execute", forbidden)
    return owners


def test_composed_xc_methodir_identity_and_compiled_chain_rule(
    tmp_path: Path, polynomial_source: list[Any]
) -> None:
    """Different user labels for the same XC science share one native source."""
    from generativeqc import GridSpec, KsOptions
    from generativeqc.extensions import method, xc

    def composed(label: str) -> Any:
        functional = xc.compose(
            label,
            {"GGA_X_PBE": "1/2", "GGA_C_PBE": 1},
            exact_exchange="1/2",
        )
        return method.compose(label, xc=functional)

    first_ir = composed("user-pbe50-a")
    alias_ir = composed("user-pbe50-b")
    assert first_ir.identity == alias_ir.identity == method.named("PBE50").identity
    grid = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
    options = KsOptions(grid=grid, tile_points=32)
    first = xp.MolecularModel(("H", "H"), method=first_ir, ks_options=options)
    alias = xp.MolecularModel(("H", "H"), method=alias_ir, ks_options=options)
    different_grid = xp.MolecularModel(
        ("H", "H"),
        method=alias_ir,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=25, angular_polar=8, angular_azimuth=16),
            tile_points=32,
        ),
    )
    assert first._payload() == alias._payload()
    assert first._payload() != different_grid._payload()
    different_xc = xp.MolecularModel(
        ("H", "H"),
        method=method.named("PBE0"),
        ks_options=options,
    )
    assert first._payload() != different_xc._payload()
    assert first._payload()["method_ir_identity"] == first_ir.identity
    assert first._payload()["native_method"] == f"methodir:{first_ir.identity}"

    compiled = xp.value_and_grad(
        lambda coordinates: first(coordinates) + alias(coordinates),
        cache=tmp_path,
    )
    try:
        lowered = compiled.lower(COORDINATES)
        assert len(lowered.provenance["molecular_sources"]) == 1
        assert not polynomial_source
        value, gradient = compiled(COORDINATES)
        np.testing.assert_allclose(value, 2 * np.sum(COORDINATES * COORDINATES))
        np.testing.assert_allclose(gradient, 4 * COORDINATES)
        assert polynomial_source[0].calls == 1
        diagnostics = compiled.compilation_info()["molecular_calls"]
        assert diagnostics[0]["method"] == first_ir.identifier
    finally:
        compiled.close()
    assert polynomial_source[0].closed


def test_molecular_method_ir_and_ks_options_forward_to_calculator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not re-resolve a user MethodIR to a descriptive name or discard the grid."""
    import generativeqc
    from generativeqc import GridSpec, KsOptions
    from generativeqc.extensions import method, xc

    graph = method.compose(
        "scaled-pbe",
        xc=xc.compose(
            "scaled-pbe",
            {"GGA_X_PBE": "1/2", "GGA_C_PBE": 1},
        ),
    )
    options = KsOptions(grid=GridSpec(radial_points=12), tile_points=48)
    source = xp.MolecularModel(
        ("H", "H"),
        method=graph,
        ks_options=options,
        backend="cpu",
    )
    captured: dict[str, Any] = {}

    def calculator_factory(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(generativeqc, "Calculator", calculator_factory)
    source._calculator()
    assert captured["method"] is graph
    assert captured["ks_options"] is options
    assert captured["initial_guess"] is None
    assert captured["device"] == "cpu"
    assert captured["precision"] == "fp64"


def test_molecular_composed_xc_configuration_fails_closed() -> None:
    from generativeqc import GridSpec, KsOptions
    from generativeqc.extensions import method, xc

    graph = method.compose(
        "user-pbe",
        xc=xc.compose(
            "user-pbe",
            {"GGA_X_PBE": 1, "GGA_C_PBE": 1},
        ),
    )
    with pytest.raises(TypeError, match="MethodIR"):
        xp.MolecularModel(("H", "H"), method=object())
    with pytest.raises(TypeError, match="KsOptions"):
        xp.MolecularModel(("H", "H"), method=graph, ks_options={"grid": "fake"})
    with pytest.raises(ValueError, match="XC belongs to method"):
        xp.MolecularModel(
            ("H", "H"),
            method=graph,
            ks_options=KsOptions(grid=GridSpec(), composition=graph),
        )


@pytest.mark.parametrize(
    ("exchange", "analytic_gradient"),
    [
        ("1/4", True),
        ("1/2", False),
    ],
)
def test_native_composed_xc_methodir_reuses_existing_scf(
    tmp_path: Path, native_library: None, exchange: str, analytic_gradient: bool
) -> None:
    """True CPU native PBE0 E+F and PBE50 E use the unchanged Calculator science."""
    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc.extensions import method, xc

    x_weight = "3/4" if exchange == "1/4" else "1/2"
    functional = xc.compose(
        "custom-pbe-hybrid",
        {"GGA_X_PBE": x_weight, "GGA_C_PBE": 1},
        exact_exchange=exchange,
    )
    graph = method.compose("custom-pbe-hybrid", xc=functional)
    standard = "PBE0" if analytic_gradient else "PBE50"
    assert graph.identity == method.named(standard).identity
    options = KsOptions(
        grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
    )
    source = xp.MolecularModel(
        ("H", "H"),
        method=graph,
        basis="sto-3g",
        ks_options=options,
        backend="cpu",
    )
    compute = (
        xp.value_and_grad(lambda coordinates: source(coordinates), cache=tmp_path)
        if analytic_gradient
        else xp.compile(lambda coordinates: source(coordinates), cache=tmp_path)
    )
    original = Calculator(
        method=graph,
        basis="sto-3g",
        device="cpu",
        ks_options=options,
        initial_guess=None,
        **dict(source._controls),
    )
    try:
        property_names = ("energy", "forces") if analytic_gradient else ("energy",)
        reference = original.singlepoint(
            source._atoms(COORDINATES), properties=property_names
        )
        if analytic_gradient:
            energy, nuclear_gradient = compute(COORDINATES)
            np.testing.assert_allclose(nuclear_gradient, -reference.forces, atol=1e-9)
        else:
            energy = compute(COORDINATES)
        np.testing.assert_allclose(energy, reference.energy, atol=2e-10, rtol=0)
        assert compute.compilation_info()["molecular_calls"][0]["calls"] == 1
    finally:
        compute.close()
        original.clear_cache()


def test_lowering_is_native_free_and_configuration_is_immutable(
    tmp_path: Path, polynomial_source: list[Any]
) -> None:
    rhf = xp.MolecularModel(("H", "H"), method="rhf")
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
    rhf = xp.MolecularModel(("H", "H"), method="rhf")

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
    rhf = xp.MolecularModel(("H", "H"), method="rhf")

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
    rhf = xp.MolecularModel(("H", "H"), method="rhf")
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
        xp.MolecularModel(method="rhf", **options)


def test_molecular_capture_gates_before_execution(
    tmp_path: Path, polynomial_source: list[Any]
) -> None:
    rhf = xp.MolecularModel(("H", "H"), method="rhf")
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
        xp.MolecularModel(("H", "H"), method="rhf", max_iterations=count + 1)
        for count in range(17)
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
    rhf = xp.MolecularModel(("H", "H"), method="rhf")
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
    monkeypatch.setattr(xp.MolecularModel, "_calculator", lambda self: owner)
    rhf = xp.MolecularModel(("H", "H"), method="rhf")
    compiled = xp.value_and_grad(lambda coordinates: rhf(coordinates), cache=tmp_path)
    with pytest.raises(error_type):
        (compiled if compiled_call else rhf)(COORDINATES)
    assert not compiled.compilation_info()["molecular_calls"]
    compiled.close()


@pytest.mark.parametrize(
    "elements, method, multiplicity",
    [
        (("H",), "uhf", 2),
        (("H", "H"), "pbe-rks", 1),
        (("H", "H"), "pbe0-rks", 1),
    ],
)
def test_method_neutral_source_dispatch_and_analytic_chain_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    elements: tuple[str, ...],
    method: str,
    multiplicity: int,
) -> None:
    """A method's native gradient composes without a per-functional AD rule."""
    calls: list[dict[str, Any]] = []
    owners: list[Any] = []

    class NativeEndpoint:
        def __init__(self) -> None:
            self.closed = False

        def singlepoint(self, atoms: Any, **kwargs: Any) -> Any:
            coordinates = np.asarray([position for _, position in atoms])
            calls.append(dict(kwargs))
            return SimpleNamespace(
                energy=float(np.sum(coordinates**2)),
                forces=-2 * coordinates,
                converged=True,
                executed_backend="cpu_reference",
                iterations=2,
                energy_change=0.0,
                density_rms=0.0,
                basis_metadata={"model_identity": "fake-method-owner"},
            )

        def clear_cache(self) -> None:
            self.closed = True

    def fake_calculator(**kwargs: Any) -> Any:
        assert kwargs["method"] == method
        assert kwargs["precision"] == "fp64"
        owner = NativeEndpoint()
        owners.append(owner)
        return owner

    monkeypatch.setattr("generativeqc.Calculator", fake_calculator)
    model = xp.MolecularModel(elements, method=method, multiplicity=multiplicity)
    coordinates = np.asarray(COORDINATES[: len(elements)])
    compiled = xp.value_and_grad(lambda geometry: model(geometry) * 2, cache=tmp_path)
    try:
        source = compiled.lower(coordinates).provenance["molecular_sources"][0]
        assert source["native_method"] == method
        assert source["multiplicity"] == multiplicity
        value, gradient = compiled(coordinates)
        np.testing.assert_allclose(value, 2 * np.sum(coordinates**2))
        np.testing.assert_allclose(gradient, 4 * coordinates)
        assert calls == [
            {
                "charge": 0,
                "multiplicity": multiplicity,
                "properties": ("energy", "forces"),
            }
        ]
        assert compiled.compilation_info()["molecular_calls"][0]["method"] == method
    finally:
        compiled.close()
    assert owners[0].closed


def test_method_neutral_source_fails_closed_on_missing_analytic_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No finite differences or frozen gradient substitution are allowed."""

    class MissingForce:
        def singlepoint(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(
                energy=1.0,
                forces=None,
                converged=True,
                executed_backend="cpu_reference",
            )

        def clear_cache(self) -> None:
            pass

    monkeypatch.setattr(xp.MolecularModel, "_calculator", lambda self: MissingForce())
    model = xp.MolecularModel(("H", "H"), method="pbe-rks")
    compiled = xp.value_and_grad(lambda geometry: model(geometry), cache=tmp_path)
    try:
        with pytest.raises(ArithmeticError, match="analytic nuclear gradient"):
            compiled(COORDINATES)
        assert not compiled.compilation_info()["molecular_calls"]
    finally:
        compiled.close()


def test_molecular_source_phase_profiling_is_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    polynomial_source: list[Any],
) -> None:
    monkeypatch.setenv("GENERATIVEQC_ARRAY_API_PROFILE", "1")
    model = xp.MolecularModel(("H", "H"), method="rhf")
    compiled = xp.value_and_grad(lambda geometry: model(geometry), cache=tmp_path)
    try:
        compiled(COORDINATES)
        info = compiled.compilation_info()
        assert info["molecular_profile_enabled"] is True
        for stage in (
            "tensor_snapshot_binding",
            "molecular_dispatch",
            "tensor_execute",
            "source_geometry",
            "source_native_singlepoint",
            "source_result_publication",
            "source_diagnostics",
        ):
            assert info["molecular_profile_ns"][stage] >= 0
            assert info["molecular_profile_counts"][stage] >= 1
        assert polynomial_source[0].calls == 1
    finally:
        compiled.close()


def test_method_neutral_identity_tracks_spin_and_method() -> None:
    singlet = xp.MolecularModel(("H", "H"), method="uhf", multiplicity=1)
    triplet = xp.MolecularModel(("H", "H"), method="uhf", multiplicity=3)
    another_method = xp.MolecularModel(("H", "H"), method="pbe-rks")
    assert singlet != triplet
    assert singlet != another_method
    assert singlet._payload()["multiplicity"] == 1
    assert triplet._payload()["multiplicity"] == 3
    with pytest.raises(FrozenInstanceError):
        singlet._multiplicity = 3


@pytest.mark.parametrize(
    "options, message",
    [
        ({"method": ""}, "nonempty"),
        ({"method": "uhf", "multiplicity": 2}, "multiplicity"),
        ({"method": "pbe-rks", "density_fitting": "invented"}, "density_fitting"),
        ({"method": "pbe-rks", "auxiliary_basis": "def2-svp"}, "auxiliary_basis"),
    ],
)
def test_method_neutral_static_admission_rejects_invalid_configuration(
    options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        xp.MolecularModel(("H", "H"), **options)


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

    rhf = xp.MolecularModel(elements, method="rhf", basis=basis, charge=charge)
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
    rhf = xp.MolecularModel(("O", "H", "H"), method="rhf", max_iterations=1)
    compiled = xp.value_and_grad(lambda coordinates: rhf(coordinates), cache=tmp_path)
    coordinates = np.asarray([[0.0, 0.0, 0.0], [0.0, -1.43, 1.11], [0.0, 1.43, 1.11]])
    with pytest.raises(RuntimeError, match=r"converg|iteration"):
        compiled(coordinates)
    assert not compiled.compilation_info()["molecular_calls"]
    compiled.close()


@pytest.mark.parametrize(
    "backend",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                os.environ.get("GENERATIVEQC_ARRAY_API_CUDA_TEST") != "1",
                reason="real composition/VJP requires an explicit Slurm GPU allocation",
            ),
        ),
    ],
)
def test_native_geometry_composition_and_snapshot_pullback(
    tmp_path: Path, native_library: None, backend: str
) -> None:
    from generativeqc import Calculator

    if backend == "cuda":
        _require_scheduled_cuda()
    rhf = xp.MolecularModel(("H", "H"), method="rhf", backend=backend)

    def expression(coordinates: Any, scale: Any) -> Any:
        energy = rhf(coordinates * scale)
        return energy * energy + xp.sum(coordinates * coordinates) / 10

    coordinates = COORDINATES.copy()
    scale = np.asarray(1.1)
    options = {
        "backend": backend,
        "cache": tmp_path,
        "cuda_arch": os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120"),
    }
    compiled = xp.value_and_grad(expression, argnums=(0, 1), **options)
    calculator = Calculator(
        method="rhf", device=backend, initial_guess=None, **dict(rhf._controls)
    )
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
        _, pullback = xp.vjp(expression, coordinates, scale, **options)
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

    rhf = xp.MolecularModel(
        ("H", "H"), method="rhf", resource_budget=ResourceBudget(host_bytes=1)
    )
    compiled = xp.value_and_grad(lambda coordinates: rhf(coordinates), cache=tmp_path)
    with pytest.raises(MemoryError, match="no supported plan fits"):
        compiled(COORDINATES)
    assert not compiled.compilation_info()["molecular_calls"]
    compiled.close()


def test_molecular_cuda_lowering_does_not_probe_native_or_device(
    polynomial_source: list[Any],
) -> None:
    rhf = xp.MolecularModel(("H", "H"), method="rhf", backend="cuda")
    compiled = xp.value_and_grad(lambda coordinates: rhf(coordinates), backend="cuda")
    lowered = compiled.lower(COORDINATES)
    assert lowered.provenance["molecular_sources"][0]["backend"] == "cuda"
    assert not polynomial_source
    compiled.close()


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_ARRAY_API_CUDA_TEST") != "1",
    reason="real molecular CUDA acceptance requires explicit scheduled opt-in",
)
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
def test_real_cuda_molecular_endpoint(
    tmp_path: Path, elements: Any, coordinates: Any, charge: int, basis: str
) -> None:
    """Check complete CUDA gradients against independent energy differences."""
    from generativeqc import Calculator

    _require_scheduled_cuda()
    rhf = xp.MolecularModel(
        elements, method="rhf", basis=basis, charge=charge, backend="cuda"
    )
    compiled = xp.value_and_grad(
        lambda coordinates: rhf(coordinates),
        backend="cuda",
        cache=tmp_path,
        cuda_arch=os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120"),
    )
    calculator = Calculator(
        method="rhf",
        basis=basis,
        device="cuda",
        initial_guess=None,
        density_fitting="none",
        precision="fp64",
        **dict(rhf._controls),
    )
    try:
        value, gradient = compiled(coordinates)
        reference = calculator.singlepoint(rhf._atoms(coordinates), charge=charge)
        assert reference.executed_backend == "cuda"
        np.testing.assert_allclose(value, reference.energy, rtol=0, atol=1e-11)
        np.testing.assert_allclose(gradient, -reference.forces, rtol=0, atol=1e-10)
        np.testing.assert_allclose(np.sum(gradient, axis=0), 0, rtol=0, atol=2e-9)
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
        assert compiled.compilation_info()["molecular_calls"][0]["backend"] == "cuda"
        assert compiled.compilation_info()["molecular_calls"][0]["calls"] == 1
    finally:
        compiled.close()
        calculator.clear_cache()
