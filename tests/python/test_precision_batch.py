"""Prepared policy lifetimes, real seed ABI transport and bounded audit failures.

The fake native fleet below controls numerical observations, not policy decisions.
Real same-model qualification and frozen/moved SCF use the scheduled CUDA tests.
"""

from __future__ import annotations

import ctypes
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import PrecisionCalculator, PrecisionPolicy, ResourceBudget, _native
from generativeqc import _warm_seed_native as native_seeds
from generativeqc._precision_batch import _SeedLease
from generativeqc._warm_state import WarmStartState
from generativeqc.batch import BatchItemResult, BatchResult
from generativeqc.ks_diagnostics import KsTransportDiagnostic
from test_precision_policy import (
    ATOMS,
    ENERGY,
    strict_native_result,
)
from test_precision_policy import backend as _scalar_backend

backend = _scalar_backend

SEED_PROTOCOL = (
    "generativeqc_batch_query_warm_seed_capacity_v1",
    "generativeqc_batch_capture_warm_seeds_v1",
    "generativeqc_batch_restore_warm_seeds_v1",
    "generativeqc_warm_seed_snapshot_read_v1",
    "generativeqc_warm_seed_snapshot_destroy",
)


@pytest.mark.parametrize("symbol", SEED_PROTOCOL)
def test_missing_seed_protocol_rejects_before_fleet_preparation(
    fleet: Any, monkeypatch: pytest.MonkeyPatch, symbol: str
) -> None:
    facade = PrecisionCalculator(policy="balanced")
    monkeypatch.setattr(facade._candidate._library, symbol, None)
    with pytest.raises(RuntimeError, match="required warm-seed protocol"):
        facade.prepare_batch([ATOMS], warm_start=False)
    assert fleet.fleets == [] and fleet.exports == 0


@pytest.mark.parametrize("symbol", SEED_PROTOCOL)
@pytest.mark.parametrize("selected_owner", ("calculator", "fleet"))
def test_changed_seed_protocol_rejects_before_fleet_execution(
    fleet: Any, monkeypatch: pytest.MonkeyPatch, symbol: str, selected_owner: str
) -> None:
    facade = PrecisionCalculator(policy="balanced")
    with facade.prepare_batch([ATOMS]) as batch:
        library = (
            facade._candidate._library
            if selected_owner == "calculator"
            else fleet.live._library
        )
        monkeypatch.setattr(library, symbol, None)
        with pytest.raises(RuntimeError, match="required warm-seed protocol"):
            batch.execute()
        assert fleet.executions == [] and fleet.exports == 0


def test_missing_execution_protocol_rejects_before_candidate_fleet_work(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facade = PrecisionCalculator(policy="balanced")
    with facade.prepare_batch([ATOMS]) as batch:
        monkeypatch.setattr(
            facade._candidate._library, "generativeqc_precision_execution_run_v1", None
        )
        with pytest.raises(
            NotImplementedError, match="required precision execution controller"
        ):
            batch.execute()
        assert fleet.executions == []


def test_missing_observable_protocol_rejects_before_candidate_fleet_work(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facade = PrecisionCalculator(policy="balanced")
    with facade.prepare_batch([ATOMS]) as batch:
        monkeypatch.setattr(
            facade._candidate._library,
            "generativeqc_observable_comparison_query_v1",
            None,
        )
        with pytest.raises(NotImplementedError, match="required observable comparison"):
            batch.execute()
        assert fleet.executions == []


def test_missing_candidate_protocol_rejects_before_fleet_execution(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facade = PrecisionCalculator(policy="balanced")
    with facade.prepare_batch([ATOMS]) as batch:
        monkeypatch.setattr(
            facade._candidate._library,
            "generativeqc_precision_candidate_query_v1",
            None,
        )
        with pytest.raises(
            NotImplementedError, match="required precision-candidate query"
        ):
            batch.execute()
        assert fleet.executions == []


def test_changed_candidate_owner_rejects_before_preparing_a_fleet(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facade = PrecisionCalculator(policy="balanced")
    monkeypatch.setattr(
        facade._candidate._library, "generativeqc_precision_candidate_query_v1", None
    )

    def forbidden_preparation(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("missing classifier started native fleet preparation")

    monkeypatch.setattr(facade._candidate, "prepare_batch", forbidden_preparation)
    with pytest.raises(NotImplementedError, match="required precision-candidate query"):
        facade.prepare_batch([ATOMS])
    assert fleet.executions == []


def item(index: int = 0, energy: float = -1.0, **changes: Any) -> BatchItemResult:
    """Detach a converged result whose Fock total is not its iteration count."""
    return replace(
        BatchItemResult(
            index,
            0,
            "success",
            energy,
            np.zeros((2, 3)),
            True,
            4,
            1e-12,
            1e-10,
            "cpu_reference",
            0,
            False,
            False,
            fock_builds=7,
        ),
        **changes,
    )


@pytest.fixture
def fleet(backend: Any, monkeypatch: Any) -> Any:
    """Model the snapshot ABI, retaining detached seeds after source teardown."""
    from generativeqc.precision_policy import Calculator

    state = backend
    state.fleets = []
    state.fleet_outputs = []
    state.executions = []
    state.restored = []
    state.exports = 0
    state.on_execute = None
    state.on_prepare = None
    state.signature = "same-model"
    state.seed_failure = None
    state.snapshots = {}
    state.next_snapshot = 1

    class Library:
        @staticmethod
        def generativeqc_batch_query_warm_seed_capacity_v1(
            owner: Any, pointer: Any
        ) -> int:
            output = ctypes.cast(
                pointer, ctypes.POINTER(native_seeds.Capacity)
            ).contents
            output.density_bytes = 32 * owner.system_count
            output.coordinate_bytes = sum(24 * len(atoms) for atoms in owner._systems)
            output.total_bytes = output.density_bytes + output.coordinate_bytes
            output.largest_density_bytes = 32 if owner.system_count else 0
            return 0

        @staticmethod
        def generativeqc_batch_capture_warm_seeds_v1(
            owner: Any, maximum: int, pointer: Any, handle: Any
        ) -> int:
            if state.seed_failure is not None:
                raise state.seed_failure
            state.exports += 1
            seeds = [
                None if seed is None else tuple(array.copy() for array in seed)
                for seed in owner.seeds
            ]
            output = ctypes.cast(
                pointer, ctypes.POINTER(native_seeds._SnapshotInfo)
            ).contents
            output.system_count = owner.system_count
            output.numeric_bytes = sum(
                array.nbytes for seed in seeds if seed is not None for array in seed
            )
            output.largest_density_bytes = max(
                (seed[0].nbytes for seed in seeds if seed is not None), default=0
            )
            assert output.numeric_bytes <= maximum
            identifier = state.next_snapshot
            state.next_snapshot += 1
            state.snapshots[identifier] = seeds
            ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(
                identifier
            )
            return 0

        @staticmethod
        def generativeqc_batch_restore_warm_seeds_v1(owner: Any, handle: Any) -> int:
            owner.seeds = [
                None if seed is None else tuple(array.copy() for array in seed)
                for seed in state.snapshots[handle.value]
            ]
            state.restored.append(owner.seeds.copy())
            return 0

        @staticmethod
        def generativeqc_warm_seed_snapshot_read_v1(
            handle: Any, index: int, pointer: Any
        ) -> int:
            descriptor = ctypes.cast(
                pointer, ctypes.POINTER(native_seeds._SeedView)
            ).contents
            seed = state.snapshots[handle.value][index]
            descriptor.present = int(seed is not None)
            if seed is not None:
                density, coordinates = seed
                descriptor.density_count = density.size
                descriptor.coordinate_count = coordinates.size
                descriptor.density = density.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)
                )
                descriptor.coordinates = coordinates.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)
                )
            return 0

        @staticmethod
        def generativeqc_warm_seed_snapshot_destroy(handle: Any) -> None:
            del state.snapshots[handle.value]

    library = Library()
    original_init = Calculator.__init__

    def initialize(calculator: Any, **options: Any) -> None:
        """Preflight fake work symbols, never real native work on fake pointers."""
        original_init(calculator, **options)
        for symbol in SEED_PROTOCOL:
            setattr(calculator._library, symbol, getattr(library, symbol))

    monkeypatch.setattr(Calculator, "__init__", initialize)

    class Fleet:
        def __init__(self, calculator: Any, systems: Any, **options: Any) -> None:
            assert state.live is None, "candidate and reference device owners overlap"
            if state.on_prepare is not None:
                state.on_prepare(calculator, options)
            self._calculator = calculator
            self._systems = tuple(systems)
            self.system_count = len(systems)
            self._warm_state = WarmStartState(
                self.system_count, enabled=options.get("warm_start", True)
            )
            self._library = library
            self._context = None
            self._batch = self
            self.charges = tuple(options.get("charges") or [0] * self.system_count)
            self.multiplicities = tuple(
                options.get("multiplicities") or [1] * self.system_count
            )
            self._basis_metadata = tuple(
                {"orbital": {"representation": "spherical"}} for _ in systems
            )
            self.basis_metadata = self._basis_metadata
            self.resource_plan = options.get("resource_plan")
            self.resource_diagnostics = {}
            self.ks_transport_diagnostics = (None,) * self.system_count
            self.seeds = [None] * self.system_count
            self.closed = False
            state.live = self
            state.fleets.append(self)

        def execute(self, coordinates: Any = None, **options: Any) -> BatchResult:
            assert not self.closed and state.live is self
            state.executions.append((self, coordinates, options, self.seeds.copy()))
            if state.on_execute is not None:
                state.on_execute(self, coordinates)
            values = state.fleet_outputs.pop(0)
            if isinstance(values, Exception):
                raise values
            for observed in values:
                if (
                    self._warm_state.enabled
                    and self._warm_state.updates
                    and observed.succeeded
                ):
                    index = observed.index
                    positions = (
                        coordinates[index]
                        if coordinates is not None and coordinates[index] is not None
                        else [atom.position for atom in self._systems[index]]
                    )
                    self.seeds[index] = (
                        np.full(4, observed.energy),
                        np.asarray(positions).reshape(-1).copy(),
                    )
                    self._warm_state.record_success(
                        index, controls={}, backend="cpu_reference"
                    )
            return BatchResult(tuple(values))

        def set_warm_start_updates(self, enabled: bool) -> None:
            self._warm_state.updates = bool(enabled)

        def clear_warm_starts(self) -> None:
            self.seeds = [None] * self.system_count
            self._warm_state.clear()

        def close(self) -> None:
            if not self.closed:
                assert state.live is self
                state.live = None
                self.closed = True

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    monkeypatch.setattr(
        Calculator,
        "prepare_batch",
        lambda self, systems, **options: Fleet(self, systems, **options),
        raising=False,
    )
    monkeypatch.setattr(
        Calculator, "_model_signature", lambda self: state.signature, raising=False
    )
    monkeypatch.setattr(
        Calculator,
        "_resource_budget",
        property(lambda self: self.options.get("resource_budget")),
        raising=False,
    )
    monkeypatch.setattr(Calculator, "_method", _native.METHOD_RHF, raising=False)
    monkeypatch.setattr(
        Calculator,
        "_shells_for_atoms",
        lambda self, atoms: [SimpleNamespace(angular_momentum=0)] * 2,
        raising=False,
    )
    return state


@pytest.mark.parametrize("profile", ("standard", "fp64"))
def test_native_fleet_reuse_without_item_timing_or_seed_exports(
    fleet: Any, profile: str
) -> None:
    calculator = PrecisionCalculator(policy=profile)
    fleet.fleet_outputs = [[item(0), item(1)], [item(0), item(1)]]
    with calculator.prepare_batch([ATOMS, ATOMS]) as prepared:
        first = prepared.execute(strict=True)
        prepared.set_warm_start_updates(False)
        second = prepared.execute(strict=True)
        assert fleet.live is fleet.fleets[0]
        assert len(fleet.fleets) == 1 and fleet.exports == 0
        for result in (first, second):
            report = result.precision_validation
            assert report["candidate_calls"] == 1 and report["audit_calls"] == []
            assert report["candidate_fock_builds"] == [7, 7]
            assert report["endpoint_seconds"] >= report["candidate_seconds"] >= 0
            for observed in result.items:
                validation = observed.precision_validation
                assert validation.status == "strict_fp64"
                assert validation.solves[0].seconds is None
                assert validation.endpoint_seconds is None
                assert validation.schema_version == 2
                assert observed.fock_builds == 7
            json.dumps(report, allow_nan=False)
    assert fleet.live is None


def test_fast_standard_fallback_reuses_owner_and_frozen_density(fleet: Any) -> None:
    """A strict fallback never suspends the expensive native prepared owner."""
    fleet.block_auto = True
    evidence = strict_native_result().precision
    observed = item(precision=evidence, executed_backend="cuda")
    fleet.fleet_outputs = [[observed], [observed]]
    calculator = PrecisionCalculator(policy="fast", device="cuda")
    with calculator.prepare_batch([ATOMS]) as prepared:
        first = prepared.execute(strict=True)
        prepared.set_warm_start_updates(False)
        second = prepared.execute(strict=True)
        assert len(fleet.fleets) == 1 and fleet.exports == 0
        assert fleet.live is fleet.fleets[0]
        for result in (first, second):
            assert not result.precision_validation["audit_calls"]
            assert result.precision_validation["suspension_seconds"] == 0
            assert (
                result.items[0].precision_validation.status
                == "strict_standard_fallback"
            )


@pytest.mark.parametrize(
    "status",
    (
        _native.STATUS_NUMERICAL_FAILURE,
        _native.STATUS_OUT_OF_MEMORY,
        _native.STATUS_INVALID_ARGUMENT,
    ),
)
def test_fast_failed_proposal_retries_only_numerics_and_retains_work(
    fleet: Any,
    status: int,
) -> None:
    """The same bounded strict retry applies to scalar and prepared proposals."""
    failed = item(
        status=status,
        status_message="final-state check failed",
        converged=False,
        energy=float("nan"),
        forces=None,
        fock_builds=None,
        executed_backend="cuda",
    )
    strict = item(executed_backend="cuda", precision=strict_native_result().precision)
    fleet.block_auto = True
    fleet.fleet_outputs = [[failed], [strict]]
    calculator = PrecisionCalculator(policy="fast", device="cuda")
    calculator._fast_fock = True
    with calculator.prepare_batch([ATOMS]) as prepared:
        output = prepared.execute()
        observed = output.items[0]
        if status == _native.STATUS_NUMERICAL_FAILURE:
            assert observed.succeeded and observed.fock_builds is None
            validation = observed.precision_validation
            assert validation.status == "strict_standard_fallback"
            assert validation.solve_count == 2
            assert validation.solves[0].error == "final-state check failed"
            assert validation.solves[0].fock_builds is None
            assert "budget_fock_failed_fp64_fallback" in validation.fallback_reasons
            assert len(output.precision_validation["audit_calls"]) == 1
        else:
            assert observed.status == status
            assert output.precision_validation["audit_calls"] == []
    assert fleet.live is None


def test_prepared_identity_tracks_each_actual_geometry_and_failed_peer(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frozen provenance must not turn into a stale geometry or result cache."""
    from generativeqc import Atom, ResolvedModel
    from generativeqc.precision_policy import Calculator
    from generativeqc.profiles import canonical_hash

    def model(owner: Any, atoms: Any, **controls: Any) -> ResolvedModel:
        geometry = canonical_hash(
            [(atom.atomic_number, atom.position) for atom in atoms]
        )
        return ResolvedModel("rhf", geometry, "orbital", 2)

    monkeypatch.setattr(Calculator, "resolved_model", model)
    original = tuple(Atom.from_value(atom) for atom in ATOMS)
    moved = np.asarray([atom.position for atom in original])
    moved[0, 0] += 0.025
    moved_atoms = tuple(
        Atom(atom.atomic_number, tuple(position))
        for atom, position in zip(original, moved, strict=True)
    )
    identities = (model(None, original).identity, model(None, moved_atoms).identity)
    assert identities[0] != identities[1]
    failed = item(
        1, status=_native.STATUS_INVALID_ARGUMENT, forces=None, converged=False
    )
    fleet.fleet_outputs = [
        [item(0, energy=-1.0), item(1)],
        [item(0, energy=-2.0), failed],
        [item(0, energy=-3.0), item(1)],
        [item(0, energy=-4.0), item(1)],
    ]
    with PrecisionCalculator(policy="fp64").prepare_batch([ATOMS, ATOMS]) as prepared:
        for step, coordinates in enumerate(
            (None, [moved, None], [moved.copy(), None], None)
        ):
            result = prepared.execute(coordinates)
            assert result.items[0].energy == -1.0 - step
            assert (
                result.items[0].precision_validation.model_identity
                == identities[int(step in (1, 2))]
            )
            if step == 1:
                assert result.failure_indices == (1,)
                assert result.items[1].precision_validation is None
            else:
                assert (
                    result.items[1].precision_validation.model_identity == identities[0]
                )


def test_frozen_candidate_seeds_survive_audit_and_rejected_candidate(
    fleet: Any,
) -> None:
    calculator = PrecisionCalculator(policy=PrecisionPolicy("fp64", target=ENERGY))
    fleet.fleet_outputs = [
        [item(energy=-1)],
        [item(energy=-1 + 1e-8)],
        [item(energy=-2)],
        [item(energy=-3)],
        [item(energy=-3 + 1e-8)],
        [item(energy=-4)],
        [item(energy=-4)],
    ]
    moved = np.asarray([atom[1] for atom in ATOMS]) + 0.01
    with calculator.prepare_batch([ATOMS]) as prepared:
        prepared.execute(strict=True)
        assert fleet.live is None
        prepared.set_warm_start_updates(False)
        result = prepared.execute([moved], strict=True)
        observed = result.items[0]
        assert observed.energy == -3
        assert observed.fock_builds == 21
        assert observed.precision_validation.rejected_errors
        assert len(result.precision_validation["audit_calls"]) == 2
        prepared.execute([moved], strict=True)
        for restored in fleet.restored:
            np.testing.assert_array_equal(restored[0][0], -np.ones(4))
            np.testing.assert_array_equal(
                restored[0][1], np.asarray([atom[1] for atom in ATOMS]).reshape(-1)
            )
        for owner, _, _, seeds in fleet.executions:
            if owner._calculator._energy_tolerance < 1e-11:
                # Movement invalidates audit history; even the tighter fallback
                # must start independently, never with the candidate's density.
                assert seeds == [None]
    assert fleet.live is None


def test_auditor_replays_only_its_own_frozen_density(fleet: Any) -> None:
    """Every call still audits; candidate/auditor proposal histories stay disjoint."""
    candidate, audit = -1.0, -1.0 + 1e-8
    fleet.fleet_outputs = [
        [item(energy=candidate)],
        [item(energy=audit)],
        [item(energy=candidate + 2e-8)],
        [item(energy=audit + 2e-8)],
        [item(energy=candidate + 3e-8)],
        [item(energy=audit + 3e-8)],
    ]
    with PrecisionCalculator(
        policy=PrecisionPolicy("fp64", target=ENERGY)
    ).prepare_batch([ATOMS]) as prepared:
        for replay in range(3):
            result = prepared.execute(strict=True)
            assert result.items[0].precision_validation.solve_count == 2
            assert result.items[0].fock_builds == 14
            assert result.precision_validation["audit_calls"][0]["completed"]
            assert result.precision_validation["audit_seed_bytes_retained"] == 80
            if replay == 0:
                prepared.set_warm_start_updates(False)
        for replay in (1, 2):
            np.testing.assert_array_equal(
                fleet.executions[2 * replay][3][0][0], candidate
            )
            np.testing.assert_array_equal(
                fleet.executions[2 * replay + 1][3][0][0], audit
            )


def test_audit_seed_budget_falls_back_to_full_cold_audit(fleet: Any) -> None:
    """The candidate's required staging wins over optional auditor history."""
    fleet.fleet_outputs = [[item()] for _ in range(4)]
    with PrecisionCalculator(policy="balanced").prepare_batch(
        [ATOMS], maximum_seed_bytes=80
    ) as prepared:
        for _ in range(2):
            result = prepared.execute(strict=True)
            assert result.items[0].precision_validation.solve_count == 2
            assert result.precision_validation["seed_bytes_retained"] == 80
            assert result.precision_validation["audit_seed_bytes_retained"] == 0
        assert fleet.executions[3][3] == [None]


def test_tighter_verification_never_uses_regular_audit_seed(fleet: Any) -> None:
    """A failed comparison still triggers an independent tighter SCF."""
    fleet.fleet_outputs = [
        [item(energy=-1)],
        [item(energy=-1)],
        [item(energy=-2)],
        [item(energy=-1)],
        [item(energy=-1 + 1e-8)],
    ]
    with PrecisionCalculator(
        policy=PrecisionPolicy("fp64", target=ENERGY)
    ).prepare_batch([ATOMS]) as prepared:
        prepared.execute(strict=True)
        result = prepared.execute(strict=True)
        assert result.items[0].precision_validation.rejected_errors
        assert result.items[0].precision_validation.solve_count == 3
        np.testing.assert_array_equal(fleet.executions[3][3][0][0], -1)
        assert fleet.executions[4][3] == [None]


def test_snapshot_geometry_and_per_item_audit_failure(fleet: Any) -> None:
    calculator = PrecisionCalculator(policy=PrecisionPolicy("fp64", target=ENERGY))
    moved = np.asarray([atom[1] for atom in ATOMS]) + 0.01
    frozen = moved.copy()
    fleet.fleet_outputs = [[item(0), item(1)], RuntimeError("audit failed"), [item()]]

    def mutate(owner: Any, coordinates: Any) -> None:
        if coordinates is not None:
            moved[:] = 9

    fleet.on_execute = mutate
    with calculator.prepare_batch([ATOMS, ATOMS]) as prepared:
        output = prepared.execute([moved, moved])
        assert output.failure_indices == (0,)
        assert output.items[0].forces is None and np.isnan(output.items[0].energy)
        assert output.items[0].precision is None
        assert output.items[1].precision_validation.status == "observed_met"
        attempts = output.precision_validation["audit_calls"]
        assert not attempts[0]["completed"] and "audit failed" in attempts[0]["error"]
        assert attempts[1]["completed"]
        for owner in fleet.fleets[1:]:
            np.testing.assert_array_equal(
                [atom.position for atom in owner._systems[0]], frozen
            )
    assert fleet.live is None


@pytest.mark.parametrize("strict", [False, True])
def test_failed_candidate_peers_skip_audit_without_hiding_success(
    fleet: Any, strict: bool
) -> None:
    calculator = PrecisionCalculator(policy=PrecisionPolicy("fp64", target=ENERGY))
    fleet.fleet_outputs = [
        [
            item(status=_native.STATUS_INVALID_ARGUMENT, forces=None, converged=False),
            item(1),
        ],
        [item()],
    ]
    with calculator.prepare_batch([ATOMS, ATOMS]) as prepared:
        if strict:
            with pytest.raises(RuntimeError, match="batched item failures: 0"):
                prepared.execute(strict=True)
        else:
            output = prepared.execute()
            assert output.failure_indices == (0,)
            assert output.items[0].precision_validation is None
            assert output.precision_validation["audit_calls"][0]["index"] == 1
        assert len(fleet.executions) == 2
        assert fleet.live is None


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_audit_is_not_marked_completed(fleet: Any, bad: float) -> None:
    fleet.fleet_outputs = [[item()], [item(energy=bad)]]
    with PrecisionCalculator(policy="balanced").prepare_batch([ATOMS]) as prepared:
        output = prepared.execute()
        assert not output.succeeded
        assert not output.precision_validation["audit_calls"][0]["completed"]
        assert output.items[0].forces is None


@pytest.mark.parametrize(
    "failure", [MemoryError("staging"), NotImplementedError("seed ABI")]
)
def test_seed_failure_preserves_owner_and_is_auditable(
    fleet: Any, failure: Exception
) -> None:
    fleet.seed_failure = failure
    fleet.fleet_outputs = [[item(0), item(1)]]
    with PrecisionCalculator(policy="balanced").prepare_batch(
        [ATOMS, ATOMS]
    ) as prepared:
        output = prepared.execute()
        assert output.failure_indices == (0, 1)
        assert fleet.live is fleet.fleets[0]
        assert len(fleet.fleets) == 1
        assert all(
            str(failure) in attempt["error"]
            for attempt in output.precision_validation["audit_calls"]
        )
    assert fleet.live is None


@pytest.mark.parametrize("partial_failure", [False, True])
def test_audit_transport_includes_resident_seed_exports(
    fleet: Any, monkeypatch: Any, partial_failure: bool
) -> None:
    """Retained diagnostics include suspension D2H, even before a failed peer."""
    fleet.fleet_outputs = [[item(0), item(1)], [item()], [item()]]
    with PrecisionCalculator(policy="balanced").prepare_batch(
        [ATOMS, ATOMS]
    ) as prepared:
        owner = fleet.live
        owner.ks_transport_diagnostics = (
            KsTransportDiagnostic(0, 0, 0, 0, 0, 0, 0),
        ) * 2
        exported = set()
        capture = owner._library.generativeqc_batch_capture_warm_seeds_v1

        def export(batch: Any, maximum: int, pointer: Any, handle: Any) -> int:
            for index in range(batch.system_count):
                if partial_failure and index == 1:
                    raise RuntimeError("second resident export failed")
                if index not in exported:
                    rows = list(batch.ks_transport_diagnostics)
                    rows[index] = replace(
                        rows[index], matrix_d2h_bytes=32, synchronizations=1
                    )
                    batch.ks_transport_diagnostics = tuple(rows)
                    exported.add(index)
            return capture(batch, maximum, pointer, handle)

        monkeypatch.setattr(
            owner._library, "generativeqc_batch_capture_warm_seeds_v1", export
        )
        output = prepared.execute()
        transport = prepared.ks_transport_diagnostics
        assert transport[0].matrix_d2h_bytes == 32
        assert transport[0].synchronizations == 1
        assert transport[1].matrix_d2h_bytes == (0 if partial_failure else 32)
        assert transport[1].synchronizations == (0 if partial_failure else 1)
        if partial_failure:
            assert output.failure_indices == (0, 1)
            assert fleet.live is owner and not owner.closed
            assert all(
                "second resident export failed" in attempt["error"]
                for attempt in output.precision_validation["audit_calls"]
            )
        else:
            assert output.succeeded and owner.closed
            assert fleet.live is None


def test_failed_seed_export_transport_query_preserves_original_error(
    fleet: Any, monkeypatch: Any
) -> None:
    """A secondary diagnostic failure must not replace the staging cause."""
    fleet.fleet_outputs = [[item()]]
    with PrecisionCalculator(policy="balanced").prepare_batch([ATOMS]) as prepared:
        owner = fleet.live
        initial = (KsTransportDiagnostic(0, 0, 0, 64, 2, 0, 0),)
        export_failed = False

        def read_transport(batch: Any) -> tuple:
            if export_failed:
                raise RuntimeError("secondary transport query failed")
            return initial

        def fail_export(*args: Any) -> int:
            nonlocal export_failed
            export_failed = True
            raise MemoryError("original resident export failed")

        monkeypatch.setattr(
            type(owner),
            "ks_transport_diagnostics",
            property(read_transport),
            raising=False,
        )
        monkeypatch.setattr(
            owner._library, "generativeqc_batch_capture_warm_seeds_v1", fail_export
        )
        output = prepared.execute()
        assert output.items[0].status == _native.STATUS_OUT_OF_MEMORY
        assert "original resident export failed" in output.items[0].status_message
        attempt = output.precision_validation["audit_calls"][0]
        assert "original resident export failed" in attempt["error"]
        assert "secondary transport" not in attempt["error"]
        assert prepared.ks_transport_diagnostics == (None,)
        assert fleet.live is owner and not owner.closed


def test_clear_suspended_seeds_disables_restore_and_close_is_idempotent(
    fleet: Any,
) -> None:
    fleet.fleet_outputs = [[item()], [item()], [item()], [item()]]
    prepared = PrecisionCalculator(policy="balanced").prepare_batch([ATOMS])
    prepared.execute(strict=True)
    prepared.clear_warm_starts()
    prepared.set_warm_start_updates(False)
    prepared.execute(strict=True)
    assert fleet.restored == []
    assert fleet.executions[2][3] == [None]
    prepared.close()
    prepared.close()
    for action in (
        prepared.execute,
        prepared.clear_warm_starts,
        lambda: prepared.set_warm_start_updates(True),
        prepared.__enter__,
    ):
        with pytest.raises(RuntimeError, match="closed"):
            action()


def test_prepared_model_mutation_rejected_before_resume(fleet: Any) -> None:
    fleet.fleet_outputs = [[item()], [item()]]
    with PrecisionCalculator(policy="balanced").prepare_batch([ATOMS]) as prepared:
        prepared.execute(strict=True)
        fleet.signature = "changed-model"
        with pytest.raises(ValueError, match="model changed"):
            prepared.execute()
        assert len(fleet.fleets) == 2 and fleet.live is None


def test_seed_lease_limits_and_detached_metadata(fleet: Any) -> None:
    calculator = PrecisionCalculator(policy="fp64")
    fleet.fleet_outputs = [[item()]]
    with calculator.prepare_batch([ATOMS]) as prepared:
        prepared.execute(strict=True)
        owner = fleet.live
        with pytest.raises(MemoryError, match="maximum_seed_bytes"):
            _SeedLease(owner, 79)
        assert fleet.exports == 0
        lease = _SeedLease(owner, 80)
        assert lease.numeric_bytes == 80
        owner._warm_state.metadata[0]["backend"] = "changed"
        assert lease.metadata.metadata[0]["backend"] == "cpu_reference"
        assert all(not array.flags.writeable for array in lease.storage)
        owner.resource_plan = SimpleNamespace(
            budget=ResourceBudget(host_bytes=1), peak_bytes={"host": 0}
        )
        with pytest.raises(MemoryError, match="headroom"):
            _SeedLease(owner, 80)


def test_explicit_resource_plan_is_charged_to_auditors(fleet: Any) -> None:
    budget = ResourceBudget(host_bytes=64 << 20, host_reserve_bytes=1024)
    plan = SimpleNamespace(budget=budget, peak_bytes={"host": 0})
    fleet.fleet_outputs = [[item()], [item()]]
    with PrecisionCalculator(policy="balanced").prepare_batch(
        [ATOMS], resource_plan=plan
    ) as prepared:
        prepared.execute(strict=True)
        auditor = fleet.fleets[1]._calculator
        assert (
            auditor._resource_budget.host_reserve_bytes > budget.host_reserve_bytes + 80
        )
        assert auditor._resource_budget.host_bytes == budget.host_bytes


def test_disabled_warm_start_materializes_no_seeds(fleet: Any) -> None:
    fleet.seed_failure = RuntimeError("not available")
    fleet.fleet_outputs = [[item()], [item()]]
    with PrecisionCalculator(policy="balanced").prepare_batch(
        [ATOMS], warm_start=False, maximum_seed_bytes=0
    ) as prepared:
        output = prepared.execute(strict=True)
        assert output.precision_validation["seed_bytes_retained"] == 0
        assert fleet.exports == 0


def test_incomplete_auditor_protocol_is_not_an_optional_cold_fallback(
    fleet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fleet.fleet_outputs = [[item()], [item()]]

    def remove_protocol(owner: Any, coordinates: Any) -> None:
        if owner is not fleet.fleets[0]:
            monkeypatch.setattr(
                owner._library, "generativeqc_warm_seed_snapshot_read_v1", None
            )

    with PrecisionCalculator(policy="balanced").prepare_batch([ATOMS]) as prepared:
        fleet.on_execute = remove_protocol
        output = prepared.execute()
        assert output.failure_indices == (0,)
        attempt = output.precision_validation["audit_calls"][0]
        assert not attempt["completed"]
        assert "required warm-seed protocol" in attempt["error"]


@pytest.mark.parametrize(
    "failure",
    [MemoryError("optional seed headroom"), NotImplementedError("seed domain")],
)
def test_supported_auditor_seed_refusal_keeps_complete_cold_audits(
    fleet: Any, failure: Exception
) -> None:
    facade = PrecisionCalculator(policy="balanced")
    fleet.fleet_outputs = [[item()], [item()], [item()], [item()]]

    def refuse_optional_seed(owner: Any, coordinates: Any) -> None:
        fleet.seed_failure = None if owner._calculator is facade._candidate else failure

    fleet.on_execute = refuse_optional_seed
    with facade.prepare_batch([ATOMS]) as prepared:
        for _ in range(2):
            output = prepared.execute(strict=True)
            assert output.items[0].precision_validation.solve_count == 2
            assert output.precision_validation["audit_calls"][0]["completed"]
            assert output.precision_validation["audit_seed_bytes_retained"] == 0


def test_audit_setup_failure_releases_candidate_and_is_recorded(fleet: Any) -> None:
    fleet.fleet_outputs = [[item()]]
    with PrecisionCalculator(policy="balanced").prepare_batch([ATOMS]) as prepared:

        def reject(calculator: Any, options: Any) -> None:
            raise MemoryError("reference allocation")

        fleet.on_prepare = reject
        output = prepared.execute()
        assert output.items[0].status == _native.STATUS_OUT_OF_MEMORY
        assert (
            "reference allocation"
            in output.precision_validation["audit_calls"][0]["error"]
        )
        assert fleet.live is None


def test_qualified_fast_retains_native_fleet_without_audit_or_staging(
    fleet: Any,
) -> None:
    """Policy admission requires actual mixed work; preparing is not scalar replay."""
    fleet.transport_v2 = True
    work = {
        "gradient": {
            "schema_version": 2,
            "qualification": "pbe0-rks-budget-fp32-v1",
            "complete": True,
            "attempted_fp32_geometry_points": 80,
            "attempted_fp32_grid_points": 80,
            "returned_mode": "mixed",
        },
    }
    observed = item(executed_backend="cuda", precision=work)
    fleet.fleet_outputs = [[observed], [observed]]
    with PrecisionCalculator(
        method="pbe0-rks", device="cuda", policy="fast"
    ).prepare_batch([ATOMS]) as prepared:
        for _ in range(2):
            output = prepared.execute(strict=True)
            assert (
                output.items[0].precision_validation.status
                == "qualified_mixed_gradient"
            )
            assert output.precision_validation["audit_calls"] == []
            assert output.precision_validation["candidate_precision"] == [work]
        assert fleet.live is fleet.fleets[0] and len(fleet.fleets) == 1
        assert fleet.exports == 0 and fleet.calls == []


def test_prepared_auto_property_fallback_does_not_change_scalar_selection(
    fleet: Any,
) -> None:
    fleet.transport_v2 = True
    fleet.auto_properties = frozenset(("energy",))
    calculator = PrecisionCalculator(method="pbe0-rks", device="cuda", policy="fast")
    with calculator.prepare_batch([ATOMS]) as prepared:
        assert prepared.capabilities.supported_properties == frozenset(
            ("energy", "forces")
        )
        assert prepared._calculator.options["precision"] == "fp64"
        assert (
            prepared._calculator._application_precision_budget
            == calculator.policy.budget
        )
        assert calculator._candidate_for(frozenset(("energy",)))[1] == "auto"


@pytest.mark.parametrize("limit", [True, -1, 1.5, 2**63])
def test_invalid_seed_bound_allocates_no_owner(fleet: Any, limit: Any) -> None:
    with pytest.raises(ValueError, match="maximum_seed_bytes"):
        PrecisionCalculator(policy="fp64").prepare_batch(
            [ATOMS], maximum_seed_bytes=limit
        )
    assert fleet.live is None


def test_native_seed_without_python_success_metadata_is_not_dropped(fleet: Any) -> None:
    calculator = PrecisionCalculator(policy="fp64")
    with calculator.prepare_batch([ATOMS]):
        owner = fleet.live
        owner.seeds[0] = (np.ones(4), np.zeros(6))
        assert owner._warm_state.metadata == [None]
        lease = _SeedLease(owner, 80)
    with calculator.prepare_batch([ATOMS]):
        lease.restore(fleet.live, updates=False)
        assert fleet.live._warm_state.metadata == [None]
        np.testing.assert_array_equal(fleet.live.seeds[0][0], np.ones(4))


def test_audit_cleanup_failure_cannot_be_reported_as_completed(
    fleet: Any, monkeypatch: Any
) -> None:
    """The complete audit includes resource release, not only finite observables."""
    from generativeqc.precision_policy import Calculator

    clear = Calculator.clear_cache

    def failing_clear(calculator: Any) -> None:
        clear(calculator)
        if calculator._energy_tolerance < 1e-11:
            raise RuntimeError("audit release failed")

    monkeypatch.setattr(Calculator, "clear_cache", failing_clear)
    fleet.fleet_outputs = [[item()], [item()]]
    with PrecisionCalculator(policy="balanced").prepare_batch([ATOMS]) as prepared:
        output = prepared.execute()
        assert output.failure_indices == (0,) and fleet.live is None
        attempt = output.precision_validation["audit_calls"][0]
        assert not attempt["completed"] and "release failed" in attempt["error"]
        assert attempt["seconds"] >= 0
