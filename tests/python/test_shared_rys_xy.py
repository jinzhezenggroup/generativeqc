"""Fixture-only shared-axis admission and artifact identity contracts."""

from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc_compiler.integral import (
    KernelConsumer,
    KernelSelection,
    ScheduleKind,
    build_fused_shell_plan,
    emit_shell_class_fused_cuda,
)
from generativeqc_compiler.integral.capabilities import (
    CAPABILITY_K_BLOCK_FOCK,
    CAPABILITY_MIXED_FOCK,
    CAPABILITY_STREAMING_FOCK,
)
from generativeqc_compiler.integral.cuda_schedule import PairOrientation, PairStorage
from generativeqc_compiler.integral.ir import DensityModel
from generativeqc_compiler.integral.production_profile import _schedule_from_payload
from generativeqc_compiler.integral.tuning.emission import (
    schedule_execution_source_identity,
)
from generativeqc_compiler.integral.tuning.policy import ScheduleTrial, schedule_payload
from test_direct_rys_values import TARGET, value_plan


@pytest.mark.parametrize("value", (None, 0, 1, "false", "true", [], {}))
def test_shared_xy_is_a_strict_boolean(value: object) -> None:
    plan = value_plan("dppp")
    with pytest.raises(TypeError, match="must be boolean"):
        replace(plan.schedule, fock_shared_rys_xy=value)
    payload = {**schedule_payload(plan.schedule), "fock_shared_rys_xy": value}
    with pytest.raises(ValueError, match="invalid production kernel schedule"):
        _schedule_from_payload(payload)


@pytest.mark.parametrize("name", ("dppp", "pppp"))
def test_shared_xy_round_trip_and_emitted_identity(name: str) -> None:
    plan = value_plan(name)
    assert not plan.schedule.fock_shared_rys_xy
    schedule = replace(plan.schedule, fock_shared_rys_xy=True)
    assert _schedule_from_payload(schedule_payload(schedule)) == schedule
    shared = build_fused_shell_plan(
        plan.spec,
        integral=plan.kernel.integral,
        schedule=schedule,
        target=TARGET,
    )
    private_trial = ScheduleTrial(
        plan.spec,
        plan.schedule,
        TARGET,
        KernelConsumer.FOCK,
        integral=plan.kernel.integral,
    )
    shared_trial = replace(private_trial, schedule=schedule)
    assert private_trial.schedule_id != shared_trial.schedule_id
    assert private_trial.entry_point != shared_trial.entry_point
    assert schedule_execution_source_identity(
        private_trial
    ) != schedule_execution_source_identity(shared_trial)
    private_source = emit_shell_class_fused_cuda(plan.spec, plan)
    shared_source = emit_shell_class_fused_cuda(plan.spec, shared)
    assert private_source != shared_source
    assert shared_source == emit_shell_class_fused_cuda(plan.spec, shared)
    # A parsed capability may be represented in a fixture, but cannot silently
    # promote either the primary or independent Fock production schedule.
    with pytest.raises(ValueError, match="fixture-only"):
        KernelSelection(
            architecture=TARGET.architecture,
            spec=plan.spec,
            consumers=(KernelConsumer.FOCK,),
            recurrence="rys3",
            integral=plan.kernel.integral,
            schedule=schedule,
        )
    with pytest.raises(ValueError, match="fixture-only"):
        KernelSelection(
            architecture=TARGET.architecture,
            spec=plan.spec,
            consumers=(KernelConsumer.FOCK,),
            recurrence="rys3",
            integral=plan.kernel.integral,
            schedule=plan.schedule,
            fock_schedule=schedule,
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"kind": ScheduleKind.SHELL_TASK},
        {"block_threads": 224},
        {"component_tile": 163},
        {"shared_coulomb": False},
        {"mixed_pair_products_fp64": True},
        {"pair_orientation": PairOrientation.SWAPPED},
        {"pair_storage": PairStorage.RECOMPUTED},
    ),
)
def test_shared_xy_rejects_other_mappings_and_precision(changes: dict) -> None:
    plan = value_plan("dppp")
    with pytest.raises(ValueError, match="shared Rys x/y"):
        schedule = replace(plan.schedule, fock_shared_rys_xy=True, **changes)
        build_fused_shell_plan(
            plan.spec,
            integral=plan.kernel.integral,
            schedule=schedule,
            target=TARGET,
        )


@pytest.mark.parametrize("name", ("dpss", "ppps", "ddss", "ddpp"))
def test_shared_xy_rejects_other_classes_and_roots(name: str) -> None:
    plan = value_plan(name)
    with pytest.raises(ValueError, match="shared Rys x/y"):
        build_fused_shell_plan(
            plan.spec,
            integral=plan.kernel.integral,
            target=TARGET,
            schedule=replace(plan.schedule, fock_shared_rys_xy=True),
        )


def test_shared_xy_rejects_derivatives() -> None:
    plan = value_plan("dppp")
    with pytest.raises(ValueError, match="shared Rys x/y"):
        build_fused_shell_plan(
            plan.spec,
            consumers=(KernelConsumer.FORCE,),
            recurrence="rys4",
            schedule=replace(plan.schedule, fock_shared_rys_xy=True),
            target=TARGET,
        )


@pytest.mark.parametrize(
    "capability",
    (
        CAPABILITY_MIXED_FOCK,
        CAPABILITY_STREAMING_FOCK,
        CAPABILITY_K_BLOCK_FOCK,
    ),
)
def test_shared_xy_rejects_non_fp64_or_streaming_consumers(capability: str) -> None:
    plan = value_plan("pppp")
    shared = build_fused_shell_plan(
        plan.spec,
        integral=plan.kernel.integral,
        target=TARGET,
        schedule=replace(plan.schedule, fock_shared_rys_xy=True),
    )
    with pytest.raises(ValueError, match="does not support"):
        emit_shell_class_fused_cuda(plan.spec, shared, capabilities=(capability,))


@pytest.mark.parametrize("name", ("dppp", "pppp"))
def test_synthetic_fixture_masks_have_exact_live_counts(name: str) -> None:
    pytest.importorskip("pyscf")
    from tools.generativeqc_validation.shared_rys_xy_fixture import fixture_inputs

    metadata, _ = fixture_inputs(name, "cartesian")
    assert metadata["retained_components"] == [
        value_plan(name).spec.component_count,
        27,
        1,
        2,
        0,
    ]
    assert metadata["primitive_quartets_per_task"] == 4
    assert metadata["retained_producer_lanes"][1] == []


def test_synthetic_source_receipt_binds_both_variants(tmp_path: Path) -> None:
    import hashlib
    import json

    pytest.importorskip("pyscf")
    from tools.generativeqc_validation.shared_rys_xy_fixture import prepare_fixture

    receipt = prepare_fixture(tmp_path)
    source = (tmp_path / "shared_rys_xy_fixture.cu").read_bytes()
    assert receipt["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert json.loads((tmp_path / "receipt.json").read_text()) == json.loads(
        json.dumps(receipt)
    )
    assert len(receipt["cases"]) == 6
    for name, extra in (("dppp", 576), ("pppp", 432)):
        assert receipt["expected_extra_shared_bytes"][name] == extra
        assert receipt["schedules"][f"{name}_shared"]["fock_shared_rys_xy"] is True
        assert not receipt["schedules"][f"{name}_private"].get(
            "fock_shared_rys_xy", False
        )


def test_shared_xy_rejects_explicit_unrestricted_only_ir() -> None:
    plan = value_plan("dppp")
    contraction = replace(
        plan.kernel.integral.contractions[0], density=frozenset((DensityModel.UHF,))
    )
    integral = replace(plan.kernel.integral, contractions=(contraction,))
    with pytest.raises(ValueError, match="shared Rys x/y"):
        build_fused_shell_plan(
            plan.spec,
            integral=integral,
            target=TARGET,
            schedule=replace(plan.schedule, fock_shared_rys_xy=True),
        )
