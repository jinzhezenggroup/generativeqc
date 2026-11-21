"""Molecular adapter admission and ownership, not mocked chemistry qualification."""

from __future__ import annotations

import typing
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import Atom, Calculator
from generativeqc.saddle import RksColdFockDamping, SaddleOptions, optimize_saddle
from generativeqc_compiler.method import resolve_method

ATOMS = (("O", (0, 0, 0)), ("H", (1.4, 1.0, 0)), ("H", (-1.4, 1.0, 0)))
MASSES = (15.99491461957, 1.00782503223, 1.00782503223)


@pytest.mark.parametrize("packed_bytes", [0, 47, 48])
def test_bounded_cold_fock_damping(
    monkeypatch: pytest.MonkeyPatch, packed_bytes: int
) -> None:
    """Run the real named preparation path without claiming a molecular saddle."""
    calculator = Calculator(
        method=resolve_method("B3LYP-D3(BJ)"),
        device="cpu",
        basis="sto-3g",
        basis_representation="spherical",
        initial_guess=None,
        cpu_packed_eri_maximum_bytes=packed_bytes,
    )
    if not calculator._native_cpu_second_order or not hasattr(
        calculator._library, "generativeqc_batch_prepare_named_ks_cold_rks_v1"
    ):
        pytest.skip("native CPU saddle with cold Fock damping is unavailable")
    atoms = (("H", (0, 0, 0)), ("H", (0, 0, 1.5)))
    masses = (1.00782503223,) * 2
    damping = RksColdFockDamping(8, 0.5, 4096)
    # A conservative method summary cannot mask a real prepared force owner.
    monkeypatch.setattr(
        calculator,
        "_capabilities",
        replace(calculator.capabilities, supported_properties=frozenset({"energy"})),
    )
    with pytest.raises(ValueError, match="accepted-frame warm starts"):
        optimize_saddle(calculator, atoms, masses=masses, cold_fock_damping=damping)
    result = optimize_saddle(
        calculator,
        atoms,
        masses=masses,
        cold_fock_damping=damping,
        accepted_frame_warm_start=True,
    )
    if packed_bytes == 47:
        assert not result.optimizer.receipt["has_valid_state"]
        assert any(
            "packed spherical ERI capacity exceeded" in message
            for _, _, message in result.optimizer.failures
        )
        assert result.diagnostics["scf_iterations"] == 0
        return
    assert result.optimizer.receipt["has_valid_state"], result.optimizer.failures
    initial = next(
        event
        for event in result.diagnostics["evaluations"]
        if event["stage"] == "energy_gradient" and event["status"] == "complete"
    )
    assert initial["cold_fock_prefix_iterations"] == 8
    assert not initial["accepted_frame_warm_start"]


@pytest.fixture
def calculator(monkeypatch: pytest.MonkeyPatch) -> Calculator:
    try:
        result = Calculator(
            method=resolve_method("B3LYP"),
            device="cpu",
            basis="sto-3g",
            basis_representation="spherical",
        )
    except (OSError, RuntimeError) as error:
        pytest.skip(f"native library unavailable: {error}")

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("admission refusal started a production SCF")

    monkeypatch.setattr(result, "prepare_batch", forbidden)
    return result


@pytest.mark.parametrize(
    "changes",
    [
        {"multiplicity": 3},
        {"multiplicity": 0},
        {"multiplicity": -1},
        {"charge": True},
        {"masses": (1, 1)},
        {"masses": (16, 1, 0)},
        {"minimum_occupied_overlap": 0},
        {"minimum_occupied_overlap": float("nan")},
        {"root_overlap_budget_bytes": 0},
        {"output_budget_bytes": 1},
        {"grid_budget_bytes": -1},
        {"numeric_budget_bytes": False},
        {"numeric_budget_bytes": 1 << 63},
        {"response_budget_bytes": 0},
        {"numeric_budget_bytes": 128 << 20, "response_budget_bytes": 129 << 20},
        {"max_eri_primitives": 0},
        {"max_eri_primitives": True},
        {"max_eri_primitives": 1 << 63},
        {"max_map_terms": 0},
        {"max_map_terms": True},
        {"max_map_terms": 1 << 63},
        {"options": SaddleOptions()},
        {"evaluation_progress": 1},
    ],
)
def test_fail_closed_before_scf(
    calculator: Calculator, changes: dict[str, typing.Any]
) -> None:
    request = {"masses": MASSES, **changes}
    with pytest.raises((ValueError, TypeError, MemoryError, NotImplementedError)):
        optimize_saddle(calculator, ATOMS, **request)


@pytest.mark.parametrize(
    "limits",
    [{"numeric_budget_bytes": 129 << 20}, {"response_budget_bytes": 1}],
)
def test_native_hessian_budget_refusal_and_recovery(limits: dict[str, int]) -> None:
    """Exercise real retained-batch limits, not the saddle classification."""
    calculator = Calculator(
        method=resolve_method("B3LYP-D3(BJ)"), device="cpu", basis="sto-3g"
    )
    if not calculator._native_cpu_second_order:
        pytest.skip("native CPU saddle is unavailable")
    atoms = (("H", (0, 0, 0)), ("H", (0, 0, 1.5)))
    options = SaddleOptions(project_rigid_motions=True, max_attempts=1)
    refused = optimize_saddle(
        calculator, atoms, masses=(1.0, 1.0), options=options, **limits
    )
    assert not refused.converged
    assert not refused.optimizer.receipt["stationary_verified"]
    assert refused.harmonics is None
    refused_events = refused.diagnostics["evaluations"]
    assert any(
        event["stage"] == "energy_gradient" and event["status"] == "complete"
        for event in refused_events
    )
    assert any(
        event["stage"] == "hessian"
        and event["status"] == "failed"
        and event["error"].startswith("GENERATIVEQC error 5:")
        for event in refused_events
    ), refused.optimizer.failures
    assert not any(
        event["stage"] == "hessian" and event["status"] == "complete"
        for event in refused_events
    )
    recovered = optimize_saddle(
        calculator,
        atoms,
        masses=(1.0, 1.0),
        options=options,
        numeric_budget_bytes=512 << 20,
        response_budget_bytes=128 << 20,
    )
    assert recovered.optimizer.receipt["has_valid_state"], recovered.optimizer.failures
    assert any(
        event["stage"] == "hessian" and event["status"] == "complete"
        for event in recovered.diagnostics["evaluations"]
    )


@pytest.mark.parametrize(
    "coordinate",
    [
        1 + 2j,
        np.complex64(1 + 2j),
        np.complex128(1 + 2j),
        True,
        np.bool_(True),
        "1.0",
        np.str_("1.0"),
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_atom_coordinates_before_callbacks(
    calculator: Calculator, coordinate: typing.Any
) -> None:
    def forbidden(*args: typing.Any) -> typing.NoReturn:
        pytest.fail("invalid molecular coordinates reached a callback")

    atoms = (ATOMS[0], ("H", [coordinate, 1.0, 0.0]), ATOMS[2])
    with pytest.raises((TypeError, ValueError), match="coordinates"):
        optimize_saddle(
            calculator,
            atoms,
            masses=MASSES,
            progress=forbidden,
            evaluation_progress=forbidden,
            cancelled=forbidden,
        )


@pytest.mark.parametrize(
    "atoms",
    [
        ATOMS,
        [[element, list(position)] for element, position in ATOMS],
        [(element, np.asarray(position)) for element, position in ATOMS],
        tuple(Atom.from_value(atom) for atom in ATOMS),
    ],
)
def test_real_atom_input_forms(calculator: Calculator, atoms: typing.Any) -> None:
    result = optimize_saddle(calculator, atoms, masses=MASSES, cancelled=lambda: True)
    assert result.optimizer.receipt["status"] == "cancelled"
    np.testing.assert_array_equal(
        result.optimizer.coordinates, [atom[1] for atom in ATOMS]
    )


def test_unqualified_hessian_before_scf(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        calculator,
        "_capabilities",
        replace(calculator.capabilities, supported_second_order=frozenset()),
    )
    with pytest.raises(NotImplementedError, match="public hessian"):
        optimize_saddle(calculator, ATOMS, masses=MASSES)


def test_missing_prepared_force_owner_does_not_execute_scf(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A force bit at method level must not allow a compatibility fallback."""

    def forbidden(**kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("missing native force owner fell back to batch execution")

    batch = SimpleNamespace(
        _native_dft_force_eligible=lambda index: False, execute=forbidden
    )
    monkeypatch.setattr(calculator, "prepare_batch", lambda *a, **k: nullcontext(batch))
    result = optimize_saddle(calculator, ATOMS, masses=MASSES)
    assert not result.optimizer.receipt["has_valid_state"]
    assert any(
        "prepared native CPU force owner" in message
        for _, _, message in result.optimizer.failures
    )
    assert result.diagnostics["scf_iterations"] == 0


def test_root_refusal_identifies_the_failed_spin_and_gate() -> None:
    from generativeqc._molecular_saddle import _occupied_overlaps

    reference = np.array([[1.0], [0.0]])
    switched = np.array([[0.0], [1.0]])
    with pytest.raises(
        ValueError, match="spin channel 1: minimum overlap 0, required 1"
    ):
        _occupied_overlaps((reference, switched), (reference, reference), np.eye(2), 1)


def test_cancel_before_preparation(calculator: Calculator) -> None:
    result = optimize_saddle(calculator, ATOMS, masses=MASSES, cancelled=lambda: True)
    assert not result.converged
    assert result.optimizer.receipt["status"] == "cancelled"
    assert not result.optimizer.receipt["has_valid_state"]
    assert result.harmonics is None
    assert not result.diagnostics["evaluations"]
    assert result.diagnostics["scf_iterations"] == 0


@pytest.mark.parametrize("functional", ("B3LYP", "PBE0"))
def test_unrestricted_admission_preserves_public_hessian_boundary(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch, functional: str
) -> None:
    """Cancellation tests admission without pretending to qualify a molecule."""
    unrestricted = Calculator(
        method=resolve_method(functional, spin="polarized"),
        device="cpu",
        basis="sto-3g",
        basis_representation="spherical",
    )

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("cancelled or unqualified UKS request started an SCF")

    monkeypatch.setattr(unrestricted, "prepare_batch", forbidden)
    if functional == "PBE0":
        with pytest.raises(NotImplementedError, match="qualified|public hessian"):
            optimize_saddle(unrestricted, ATOMS, masses=MASSES, cancelled=lambda: True)
    else:
        result = optimize_saddle(
            unrestricted, ATOMS, masses=MASSES, cancelled=lambda: True
        )
        assert result.optimizer.receipt["status"] == "cancelled"
        assert result.diagnostics["spin"] == "polarized"
        assert not result.diagnostics["evaluations"]


def test_preparation_failure_is_a_retained_result(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise MemoryError("prepared model capacity")

    monkeypatch.setattr(calculator, "prepare_batch", failed)
    result = optimize_saddle(calculator, ATOMS, masses=MASSES)
    assert not result.converged
    assert result.optimizer.receipt["status"] == "evaluation_failed"
    assert result.optimizer.failures == (
        ("energy_gradient", "MemoryError", "prepared model capacity"),
    )
    assert not result.diagnostics["work_counters_complete"]
    assert result.diagnostics["evaluations"][-1]["status"] == "failed"
    np.testing.assert_array_equal(
        result.optimizer.coordinates, [atom[1] for atom in ATOMS]
    )


def test_model_mutation_in_stage_observer_fails_before_scf(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(event: typing.Mapping[str, typing.Any]) -> None:
        monkeypatch.setattr(calculator, "_model_signature", lambda: "changed model")

    result = optimize_saddle(
        calculator, ATOMS, masses=MASSES, evaluation_progress=mutate
    )
    assert not result.converged
    assert "model changed" in result.optimizer.failures[0][2]
    assert result.optimizer.receipt["energy_gradient_calls"] == 1


def test_harmonic_publication_failure_retains_native_geometry(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import _molecular_saddle
    from generativeqc.saddle import SaddleResult

    positions = np.array([atom[1] for atom in ATOMS])
    native = SaddleResult(
        positions,
        np.zeros_like(positions),
        np.eye(9),
        np.zeros_like(positions),
        np.ones(3),
        {
            "has_valid_state": True,
            "stationary_verified": True,
            "status": "first_order_saddle",
        },
        "synthetic publication control",
        (),
        0.1,
    )

    def failed(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise MemoryError("harmonic workspace")

    monkeypatch.setattr(
        _molecular_saddle, "correct_saddle", lambda *args, **kwargs: native
    )
    monkeypatch.setattr("generativeqc.vibrations.analyze_harmonic_modes", failed)
    result = optimize_saddle(calculator, ATOMS, masses=MASSES)
    assert not result.converged and result.optimizer.converged
    assert result.harmonics is None
    assert result.diagnostics["final_status"] == "harmonic_analysis_failed"
    assert result.diagnostics["harmonic_failure"] == (
        "MemoryError",
        "harmonic workspace",
    )
    np.testing.assert_array_equal(result.optimizer.coordinates, positions)


@pytest.mark.parametrize(
    "change", [None, "rigid_rank", "zero_negative", "two_negative", "soft"]
)
def test_harmonic_publication_requires_consistent_classification(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch, change: str | None
) -> None:
    """Synthetic native derivatives test publication policy, not molecular accuracy."""
    from dataclasses import replace

    from generativeqc import _molecular_saddle
    from generativeqc.saddle import correct_saddle
    from generativeqc.vibrations import analyze_harmonic_modes

    positions = np.array([atom[1] for atom in ATOMS])
    centered = positions - positions.mean(axis=0)
    rigid = np.column_stack(
        (
            np.tile(np.eye(3), (len(positions), 1)),
            *(np.cross(axis, centered).ravel() for axis in np.eye(3)),
        )
    )
    active = np.linalg.svd(rigid, full_matrices=True)[0][:, 6:]
    matrix = (active * np.array([-1.0, 1.0, 2.0])) @ active.T
    native = correct_saddle(
        positions,
        energy_gradient=lambda current: (0.0, np.zeros_like(current)),
        hessian=lambda current: matrix,
        options=SaddleOptions(project_rigid_motions=True),
    )
    assert native.converged and native.receipt["negative_curvatures"] == 1
    actual = analyze_harmonic_modes(
        matrix,
        coordinates=positions,
        masses=MASSES,
        hessian_units="hartree/bohr^2",
        coordinate_units="bohr",
        mass_units="dalton",
        symmetrize=True,
    )
    changes = {
        None: {},
        "rigid_rank": {"rigid_mode_count": 5},
        "zero_negative": {"negative_mode_count": 0},
        "two_negative": {"negative_mode_count": 2},
        "soft": {"significant_imaginary_mode_count": 0},
    }
    published = replace(actual, **changes[change])
    monkeypatch.setattr(
        _molecular_saddle, "correct_saddle", lambda *args, **kwargs: native
    )
    monkeypatch.setattr(
        "generativeqc.vibrations.analyze_harmonic_modes",
        lambda *args, **kwargs: published,
    )
    result = optimize_saddle(calculator, ATOMS, masses=MASSES)
    assert result.optimizer is native
    np.testing.assert_array_equal(result.optimizer.hessian, matrix)
    if change in (None, "soft"):
        assert result.converged and result.harmonics is published
        assert result.diagnostics["harmonic_failure"] is None
    else:
        assert not result.converged and result.harmonics is None
        assert result.diagnostics["final_status"] == "harmonic_analysis_failed"
        assert result.diagnostics["harmonic_failure"][0] == "ValueError"
        assert "differs" in result.diagnostics["harmonic_failure"][1]


def test_near_linear_native_harmonic_rank_mismatch_fails_closed(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real native projection with a synthetic matrix; no chemistry qualification."""
    from generativeqc import _molecular_saddle
    from generativeqc.saddle import correct_saddle

    positions = np.array([[-1.0, 0, 0], [0, 1.3e-8, 0], [1.0, 0, 0]])
    rotation = np.cross([1.0, 0, 0], positions - positions.mean(axis=0)).ravel()
    rotation /= np.linalg.norm(rotation)
    matrix = np.eye(9) - 2 * np.outer(rotation, rotation)
    native = correct_saddle(
        positions,
        energy_gradient=lambda current: (0.0, np.zeros_like(current)),
        hessian=lambda current: matrix,
        options=SaddleOptions(project_rigid_motions=True),
    )
    assert native.converged and native.receipt["rigid_modes"] == 5
    monkeypatch.setattr(
        _molecular_saddle, "correct_saddle", lambda *args, **kwargs: native
    )
    atoms = tuple(zip(("H", "O", "H"), positions, strict=True))
    result = optimize_saddle(calculator, atoms, masses=(1.0, 16.0, 1.0))
    assert result.optimizer is native
    assert not result.converged and result.harmonics is None
    assert result.diagnostics["final_status"] == "harmonic_analysis_failed"
    assert "rigid-mode rank" in result.diagnostics["harmonic_failure"][1]


@pytest.mark.parametrize("mode", ["cold", "warm", "fallback"])
def test_native_total_publication_closes_batch_and_tracks_only_accepted_roots(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """An executable lifecycle test; synthetic derivatives are not molecular evidence."""
    from contextlib import contextmanager
    from dataclasses import dataclass
    from types import MappingProxyType

    from generativeqc import _molecular_saddle
    from generativeqc.saddle import SaddleProgress, SaddleResult

    live = set()
    warm = mode != "cold"
    fallback = mode == "fallback"
    initial_occupied = np.eye(3)[:, :2]
    trial_occupied = initial_occupied.copy()
    trial_occupied[:, 0] = (0.95, 0, np.sqrt(1 - 0.95**2))
    later_occupied = trial_occupied.copy()
    later_occupied[2, 0] *= -1
    orbital_sources = iter((initial_occupied, trial_occupied, later_occupied))
    restores = []
    retired = []
    created = []

    class Snapshot:
        numeric_bytes = 72

        def __init__(self, index: int) -> None:
            self.index = index

        def require_geometry(self, index: int, positions: np.ndarray) -> None:
            assert live == {"batch"} and index == 0 and positions.shape == (3, 3)

        def restore(self, batch: typing.Any) -> None:
            assert live == {"batch"} and self.index not in retired
            restores.append(self.index)
            batch.primal.warm_start_used = True
            if fallback and len(restores) == 2:
                batch.primal.warm_start_fallback = True
                batch.primal.fock_builds = None

        def close(self) -> None:
            assert self.index not in retired
            retired.append(self.index)

    def capture(batch: typing.Any, maximum_bytes: int) -> Snapshot:
        assert live == {"batch"} and maximum_bytes >= Snapshot.numeric_bytes
        snapshot = Snapshot(len(created))
        created.append(snapshot)
        return snapshot

    @contextmanager
    def owned(name: str, value: typing.Any) -> typing.Iterator[typing.Any]:
        live.add(name)
        try:
            yield value
        finally:
            live.remove(name)

    def prepare(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        assert kwargs["warm_start"] is warm
        primal = SimpleNamespace(
            energy=-1.0,
            forces=np.zeros((3, 3)),
            iterations=7,
            fock_builds=8,
            converged=True,
            succeeded=True,
            physical_residual_rms=1e-12,
            warm_start_used=False,
            warm_start_fallback=False,
        )
        return owned(
            "batch",
            SimpleNamespace(
                primal=primal,
                execute=lambda **kwargs: SimpleNamespace(items=(primal,)),
                _native_dft_force_eligible=lambda index: True,
            ),
        )

    def orbital_frame(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        assert live == {"batch"}
        occupied = next(orbital_sources)
        coefficients = np.column_stack((occupied, np.eye(3)[:, 2]))
        state = SimpleNamespace(
            identity=SimpleNamespace(spin="unpolarized", geometry_identity="geometry"),
            coefficients=coefficients[None],
            occupations=np.array([[2, 2, 0]]),
            numeric_reservation_bytes=4096,
        )
        return state

    @dataclass(frozen=True)
    class Electronic:
        matrix: np.ndarray
        _diagnostics: typing.Mapping[str, typing.Any]

        @property
        def diagnostics(self) -> typing.Mapping[str, typing.Any]:
            return self._diagnostics

    def build(*args: typing.Any, **kwargs: typing.Any) -> Electronic:
        assert live == {"batch"}
        return Electronic(
            np.eye(9),
            {
                "nuclear_response_solves": 9,
                "response_residual_norm": 1e-12,
                "raw_symmetry_error": 0.0,
                "total_method_identity": calculator.method_ir.identity,
                "response_iterations": 9,
                "induced_fock_actions": 9,
                "first_eri_primitives": 100,
                "curvature_eri_primitives": 200,
                "first_one_electron_primitives": 30,
                "curvature_one_electron_primitives": 60,
                "source_xc_point_visits": 90,
                "curvature_xc_point_visits": 100,
                "response_xc_point_visits": 120,
                "ecp_kernel_blocks": 2,
                "ecp_radial_layers": 6,
                "ecp_ao_primitive_points": 33,
                "ecp_contraction_terms": 42,
                "d3_pair_visits": 18,
                "d3_reference_terms": 80,
            },
        )

    def drive(coordinates: typing.Any, **callbacks: typing.Any) -> SaddleResult:
        initial = np.array(coordinates)
        _energy, gradient = callbacks["energy_gradient"](initial)
        matrix = callbacks["hessian"](initial)
        assert not live
        receipt = MappingProxyType(
            {
                "stationary_verified": False,
                "has_valid_state": True,
                "status": "attempt_limit",
            }
        )
        callbacks["progress"](SaddleProgress(initial, receipt, 0.1))
        callbacks["energy_gradient"](initial - 0.01)
        callbacks["hessian"](initial - 0.01)
        displaced = initial + 0.01
        if fallback:
            with pytest.raises(RuntimeError, match="cold fallback result refused"):
                callbacks["energy_gradient"](displaced)
        else:
            callbacks["energy_gradient"](displaced)
            callbacks["hessian"](displaced)
        assert not live
        return SaddleResult(
            initial,
            gradient,
            matrix,
            np.zeros_like(initial),
            np.ones(3),
            receipt,
            "unit control",
            (),
            0.2,
        )

    monkeypatch.setattr(calculator, "prepare_batch", prepare)
    monkeypatch.setattr("generativeqc._native_cpu_hessian.orbital_frame", orbital_frame)
    monkeypatch.setattr("generativeqc._native_cpu_hessian.execute_batch", build)
    monkeypatch.setattr("generativeqc._warm_seed_native.capture_snapshot", capture)

    def cross_overlap(
        target: Calculator,
        source: Calculator,
        atoms: tuple[Atom, ...],
        *,
        source_atoms: tuple[Atom, ...],
        **kwargs: typing.Any,
    ) -> np.ndarray:
        assert target is source is calculator
        np.testing.assert_allclose(
            [atom.position for atom in atoms],
            [atom.position for atom in source_atoms],
            atol=1e-15,
        )
        return np.eye(3)

    monkeypatch.setattr("generativeqc.overlap.cross_overlap", cross_overlap)
    monkeypatch.setattr(_molecular_saddle, "correct_saddle", drive)
    result = optimize_saddle(
        calculator, ATOMS, masses=MASSES, accepted_frame_warm_start=warm
    )
    assert not result.converged
    assert not live
    frames = 2 if fallback else 3
    assert result.diagnostics["scf_iterations"] == 7 * frames
    assert result.diagnostics["nuclear_response_solves"] == 9 * frames
    assert result.diagnostics["d3_pair_visits"] == 18 * frames
    assert result.diagnostics["d3_reference_terms"] == 80 * frames
    assert result.diagnostics["curvature_one_electron_primitives"] == 60 * frames
    assert result.diagnostics["response_xc_point_visits"] == 120 * frames
    assert result.diagnostics["ecp_kernel_blocks"] == 2 * frames
    assert result.diagnostics["ecp_radial_layers"] == 6 * frames
    assert result.diagnostics["ecp_ao_primitive_points"] == 33 * frames
    assert result.diagnostics["ecp_contraction_terms"] == 42 * frames
    assert restores == ([0, 0] if warm else [])
    assert sorted(retired) == (list(range(frames)) if warm else [])
    overlaps = [
        event["minimum_occupied_overlap"]
        for event in result.diagnostics["evaluations"]
        if event["stage"] == "energy_gradient" and event["status"] == "complete"
    ]
    np.testing.assert_allclose(overlaps, [1, 0.95, 0.95][:frames])
    registrations = [
        event["root_registration_translation_bohr"]
        for event in result.diagnostics["evaluations"]
        if event["stage"] == "energy_gradient" and event["status"] == "complete"
    ]
    np.testing.assert_allclose(
        registrations, [[0, 0, 0], [-0.01] * 3, [0.01] * 3][:frames]
    )
