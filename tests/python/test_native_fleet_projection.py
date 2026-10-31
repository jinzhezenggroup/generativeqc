"""Prepared-owner projection, per-item admission and atomic seed publication."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import Calculator, _native, progressive
from generativeqc import _fleet_projection_native as native
from generativeqc import _system_projection_native as system_native
from generativeqc.checkpoint import _descriptor, _pointer
from generativeqc.projection import ProjectionPolicy, ProjectionRejected

if TYPE_CHECKING:
    from collections.abc import Iterator

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))]
HE = [("He", (0.0, 0.0, 0.0))]


@contextmanager
def fleets(
    *, method: str = "rhf", fitted: bool = False, systems: list | None = None
) -> Iterator[tuple]:
    """Retain real native owners throughout each transport/lifetime check."""
    geometries = [H2] if systems is None else systems
    with (
        Calculator(method=method).prepare_batch(geometries) as source,
        Calculator(
            basis="def2-svp", method=method, density_fitting="cpu" if fitted else "none"
        ).prepare_batch(geometries) as target,
    ):
        assert native.bind(target._library), (
            "build the current native fleet projection protocol"
        )
        source.execute(strict=True)
        yield source, target


def controls(*, strict: bool = True) -> native.Controls:
    """Use the public default policy with an explicit numeric budget."""
    return native.Controls(
        ct.sizeof(native.Controls),
        _native.ABI_VERSION,
        int(strict),
        0,
        4096,
        1 << 28,
        1e-10,
        1e-7,
        0.5,
    )


def query(source: Any, target: Any, policy: native.Controls) -> native.Capacity:
    result = native.Capacity(ct.sizeof(native.Capacity), _native.ABI_VERSION)
    assert (
        target._library.generativeqc_hf_fleet_projection_query_v1(
            target._batch, source._batch, ct.byref(policy), None, 0, ct.byref(result)
        )
        == 0
    )
    return result


def records(count: int) -> Any:
    return (native.Item * count)(
        *(
            native.Item(ct.sizeof(native.Item), _native.ABI_VERSION, 99)
            for _ in range(count)
        )
    )


def execute(source: Any, target: Any, policy: native.Controls, outputs: Any) -> int:
    return target._library.generativeqc_hf_fleet_initialize_cpu_v1(
        target._batch, source._batch, ct.byref(policy), None, 0, outputs, len(outputs)
    )


def snapshot(batch: Any, index: int = 0) -> tuple | None:
    """Read state only in the test oracle; production transfer never exports it."""
    state = _descriptor()
    _native.check(
        batch._library,
        batch._library.generativeqc_batch_get_hf_warm_state(
            batch._batch, index, ct.byref(state)
        ),
    )
    if not state.present:
        return None
    density = np.empty(state.density_count)
    coordinates = np.empty(state.coordinate_count)
    state.density, state.coordinates = _pointer(density), _pointer(coordinates)
    _native.check(
        batch._library,
        batch._library.generativeqc_batch_get_hf_warm_state(
            batch._batch, index, ct.byref(state)
        ),
    )
    return density.tobytes(), coordinates.tobytes(), state.energy, state.iterations


def forbid_frontend_science(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "native fleet transfer entered Python scientific orchestration"
        )

    monkeypatch.setattr(progressive, "_initialize_from_compatibility", forbidden)
    monkeypatch.setattr(system_native, "project", forbidden)
    monkeypatch.setattr(progressive, "cross_overlap", forbidden)
    monkeypatch.setattr(progressive, "project_density", forbidden)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize("fitted", [False, True])
def test_prepared_owner_keeps_density_native_and_reaches_target(
    monkeypatch: pytest.MonkeyPatch, method: str, fitted: bool
) -> None:
    forbid_frontend_science(monkeypatch)
    with fleets(method=method, fitted=fitted, systems=[H2, HE]) as (source, target):
        before = [snapshot(source, index) for index in range(2)]
        with monkeypatch.context() as guarded:

            def forbidden_export(*args: Any) -> None:
                raise AssertionError("production projection exported source density")

            guarded.setattr(
                source._library,
                "generativeqc_batch_get_hf_warm_state",
                forbidden_export,
            )
            # These are presentation fields, not native source eligibility.
            source._last_statuses = None
            source._warm_state.projection_indices = {0, 1}
            report = target.initialize_from(source)
        assert report["execution_owner"] == "native_cpu_fleet"
        assert all(item["accepted"] for item in report["items"])
        assert [snapshot(source, index) for index in range(2)] == before
        result = target.execute(strict=True)
        assert all(item.restart_origin == "basis_projection" for item in result.items)
        cold = target._calculator.singlepoint(H2)
        assert result.items[0].energy == pytest.approx(cold.energy, abs=1e-9)
        np.testing.assert_allclose(
            result.items[0].forces, cold.forces, atol=2e-8, rtol=0
        )


@pytest.mark.parametrize("count", [1, 2])
def test_exact_numeric_capacity_and_below_limit_rollback(count: int) -> None:
    with fleets(systems=[H2] * count) as (source, target):
        policy = controls()
        capacity = query(source, target, policy)
        original_source = [snapshot(source, index) for index in range(count)]
        outputs = records(count)
        before = bytes(outputs)
        policy.maximum_host_bytes = capacity.numeric_peak_bytes - 1
        assert execute(source, target, policy, outputs) == _native.STATUS_OUT_OF_MEMORY
        assert bytes(outputs) == before
        assert all(snapshot(target, index) is None for index in range(count))
        assert [snapshot(source, index) for index in range(count)] == original_source
        policy.maximum_host_bytes += 1
        assert execute(source, target, policy, outputs) == 0
        assert all(row.accepted for row in outputs)
        assert all(snapshot(target, index) is not None for index in range(count))


def test_malformed_replay_revokes_native_source_eligibility() -> None:
    with fleets() as (source, target):
        original = snapshot(source)
        assert (
            source._library.generativeqc_batch_execute(source._batch, None, 0, None, 0)
            == _native.STATUS_INVALID_ARGUMENT
        )
        assert snapshot(source) == original, (
            "last-good seed must survive a rejected replay"
        )
        source._last_statuses = (_native.STATUS_SUCCESS,)
        report = target.initialize_from(source, strict=False)
        assert not report["items"][0]["accepted"]
        assert "successful current SCF" in report["items"][0]["reason"]
        assert target.execute(strict=True).items[0].restart_origin == "cold"


def test_imported_proposal_cannot_be_promoted_by_frontend_metadata() -> None:
    with fleets() as (source, target):
        target.initialize_from(source)
        with Calculator(basis="def2-svp").prepare_batch([H2]) as third:
            target._last_statuses = (_native.STATUS_SUCCESS,)
            target._warm_state.projection_indices.clear()
            report = third.initialize_from(target, strict=False)
            assert not report["items"][0]["accepted"]
            target.execute(strict=True)
            assert third.initialize_from(target)["items"][0]["accepted"]


def test_frozen_proposal_does_not_inherit_an_unretained_success() -> None:
    with fleets() as (source, target):
        target.initialize_from(source)
        original = snapshot(target)
        target.set_warm_start_updates(False)
        target.execute(strict=True)
        assert snapshot(target) == original
        with Calculator(basis="def2-svp").prepare_batch([H2]) as third:
            report = third.initialize_from(target, strict=False)
            assert not report["items"][0]["accepted"]
            target.set_warm_start_updates(True)
            target.execute(strict=True)
            assert third.initialize_from(target)["items"][0]["accepted"]


def test_later_numerical_rejection_is_atomic_or_keeps_neighbor_cold() -> None:
    # H2 loses about 0.028 of the occupied norm; He loses about 0.040.
    # This bound accepts the first projection before the second one rejects.
    with fleets(systems=[H2, HE]) as (source, target):
        before = [snapshot(source, index) for index in range(2)]
        policy = ProjectionPolicy(maximum_residual=0.03)
        with pytest.raises(ProjectionRejected, match="item 1.*projection residual"):
            target.initialize_from(source, policy=policy)
        assert all(snapshot(target, index) is None for index in range(2))
        report = target.initialize_from(source, policy=policy, strict=False)
        assert [item["accepted"] for item in report["items"]] == [True, False]
        assert [snapshot(source, index) for index in range(2)] == before
        result = target.execute(strict=True)
        assert [item.restart_origin for item in result.items] == [
            "basis_projection",
            "cold",
        ]


@pytest.mark.parametrize("bad", ["count", "null", "nonfinite"])
def test_coordinate_spans_reject_bad_neighbor_without_reading_payload(bad: str) -> None:
    with fleets(systems=[H2, HE]) as (source, target):
        xyz = np.asarray([atom[1] for atom in HE], dtype=np.float64).reshape(-1)
        if bad == "nonfinite":
            xyz[0] = np.nan
        inputs = (_native.BatchInputDescriptor * 2)(
            _native.BatchInputDescriptor(
                ct.sizeof(_native.BatchInputDescriptor), _native.ABI_VERSION
            ),
            _native.BatchInputDescriptor(
                ct.sizeof(_native.BatchInputDescriptor),
                _native.ABI_VERSION,
                None if bad == "null" else _pointer(xyz),
                2**32 - 1 if bad == "count" else xyz.size,
            ),
        )
        policy, outputs = controls(), records(2)
        before = bytes(outputs)
        operation = target._library.generativeqc_hf_fleet_initialize_cpu_v1
        assert (
            operation(
                target._batch, source._batch, ct.byref(policy), inputs, 2, outputs, 2
            )
            == _native.STATUS_INVALID_ARGUMENT
        )
        assert bytes(outputs) == before
        assert all(snapshot(target, index) is None for index in range(2))
        policy.strict = 0
        assert (
            operation(
                target._batch, source._batch, ct.byref(policy), inputs, 2, outputs, 2
            )
            == 0
        )
        assert [row.accepted for row in outputs] == [1, 0]


def test_explicit_coordinates_use_successful_replay_geometry() -> None:
    with fleets() as (source, target):
        xyz = np.array([atom[1] for atom in H2])
        xyz[1, 2] += 0.08
        source.execute(coordinates=[xyz], strict=True)
        with pytest.raises(ProjectionRejected, match="geometry"):
            target.initialize_from(source)
        inputs = (_native.BatchInputDescriptor * 1)(
            _native.BatchInputDescriptor(
                ct.sizeof(_native.BatchInputDescriptor),
                _native.ABI_VERSION,
                _pointer(xyz),
                xyz.size,
            )
        )
        policy, outputs = controls(), records(1)
        assert (
            target._library.generativeqc_hf_fleet_initialize_cpu_v1(
                target._batch, source._batch, ct.byref(policy), inputs, 1, outputs, 1
            )
            == 0
        )
        seeded = snapshot(target)
        assert seeded is not None and seeded[1] == xyz.tobytes()
        result = target.execute(coordinates=[xyz], strict=True).items[0]
        cold = target._calculator.singlepoint(
            [(atom[0], tuple(position)) for atom, position in zip(H2, xyz, strict=True)]
        )
        assert result.warm_start_used
        assert result.energy == pytest.approx(cold.energy, abs=1e-9)
        np.testing.assert_allclose(result.forces, cold.forces, atol=2e-8, rtol=0)


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_moved_water_and_radical_against_independent_pyscf(
    monkeypatch: pytest.MonkeyPatch, fitted: bool, method: str
) -> None:
    """Exercise O p/d shells, multiple occupied orbitals and open-shell spin."""
    pyscf = pytest.importorskip("pyscf")
    forbid_frontend_science(monkeypatch)
    original = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 1.43, 1.11)),
        ("H", (0.0, -1.43, 1.11)),
    ]
    if method == "uhf":
        original = original[:2]
    moved = [
        (z, (r[0], r[1], r[2] + (0.035 if i == 1 else 0.0)))
        for i, (z, r) in enumerate(original)
    ]
    multiplicity = 2 if method == "uhf" else 1
    options = {
        "method": method,
        "max_iterations": 100,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-14,
    }
    with (
        Calculator(**options).prepare_batch(
            [original], multiplicities=[multiplicity]
        ) as source,
        Calculator(
            basis="def2-svp", density_fitting="cpu" if fitted else "none", **options
        ).prepare_batch([moved], multiplicities=[multiplicity]) as target,
    ):
        source.execute(strict=True)
        source.execute(coordinates=[[r for _, r in moved]], strict=True)
        report = target.initialize_from(source)
        assert report["execution_owner"] == "native_cpu_fleet"
        result = target.execute(strict=True).items[0]
        assert result.restart_origin == "basis_projection"
    molecule = pyscf.gto.M(
        atom=moved,
        basis="def2-svp",
        unit="Bohr",
        cart=True,
        spin=multiplicity - 1,
        verbose=0,
    )
    reference = pyscf.scf.UHF(molecule) if method == "uhf" else pyscf.scf.RHF(molecule)
    if fitted:
        reference = reference.density_fit(auxbasis="def2-svp")
    reference.conv_tol, reference.conv_tol_grad = 1e-12, 1e-10
    reference.max_cycle = 100
    reference.kernel()
    assert reference.converged
    assert result.energy == pytest.approx(reference.e_tot, abs=1e-8)
    np.testing.assert_allclose(
        result.forces, -reference.nuc_grad_method().kernel(), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("failure", ["native", "receipt", "presentation"])
def test_failure_never_retries_and_frontend_publication_rolls_back(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    forbid_frontend_science(monkeypatch)
    with fleets() as (source, target):
        operation = target._library.generativeqc_hf_fleet_initialize_cpu_v1

        def fail(*args: Any) -> int:
            if failure == "native":
                return _native.STATUS_INTERNAL_ERROR
            status = operation(*args)
            assert status == 0
            ct.cast(args[-2], ct.POINTER(native.Item)).contents.spin_count = 9
            return status

        def reject_presentation(*args: Any) -> None:
            raise MemoryError("injected detached report allocation failure")

        if failure == "presentation":
            monkeypatch.setattr(native, "_publish", reject_presentation)
        else:
            monkeypatch.setattr(
                target._library, "generativeqc_hf_fleet_initialize_cpu_v1", fail
            )
        with pytest.raises((RuntimeError, MemoryError)):
            target.initialize_from(source)
        assert snapshot(target) is None
        assert target.execute(strict=True).items[0].restart_origin == "cold"


@pytest.mark.parametrize("bad", ["header", "flags", "count", "output"])
def test_invalid_abi_controls_and_counts_preserve_outputs(bad: str) -> None:
    with fleets() as (source, target):
        policy = controls()
        output = records(1)
        if bad == "header":
            policy.abi_version += 1
        if bad == "flags":
            policy.reserved = 1
        if bad == "output":
            output[0].struct_size -= 1
        before = bytes(output)
        status = target._library.generativeqc_hf_fleet_initialize_cpu_v1(
            target._batch,
            source._batch,
            ct.byref(policy),
            None,
            0,
            output,
            int(bad != "count"),
        )
        assert status == (
            _native.STATUS_ABI_MISMATCH
            if bad in {"header", "output"}
            else _native.STATUS_INVALID_ARGUMENT
        )
        assert bytes(output) == before and snapshot(target) is None


def test_concurrent_targets_borrow_one_source_without_mutation() -> None:
    with (
        fleets() as (source, target),
        Calculator(basis="def2-svp").prepare_batch([H2]) as other,
    ):
        original = snapshot(source)
        with ThreadPoolExecutor(max_workers=2) as pool:
            reports = list(
                pool.map(lambda owner: owner.initialize_from(source), [target, other])
            )
        assert all(report["items"][0]["accepted"] for report in reports)
        assert snapshot(target) == snapshot(other)
        assert snapshot(source) == original


def test_old_protocol_and_custom_policy_remain_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CustomPolicy(ProjectionPolicy):
        """A caller extension is outside the exact built-in native transport."""

    with fleets() as (source, target):
        assert "execution_owner" not in target.initialize_from(
            source, policy=CustomPolicy()
        )
    with fleets() as (source, target):
        monkeypatch.setattr(native, "bind", lambda library: False)
        assert "execution_owner" not in target.initialize_from(source)
