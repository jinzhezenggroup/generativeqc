"""Deterministic pass-pipeline execution and bisection contracts."""

import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.pass_manager import PassManager, PassStage
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    constant,
    execute,
    input_tensor,
    multiply,
    optimize,
    prepare_for_backend,
)
from generativeqc_compiler.tensor.cpu import emit_cpu
from generativeqc_compiler.tensor.cuda_plan import plan_cuda
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp


def fingerprint(value: int) -> str:
    return str(value)


def test_pass_manager_runs_versioned_pipeline_and_records_changes() -> None:
    manager = PassManager(
        name="test.pipeline",
        version=2,
        stages=(
            PassStage("plus_one", 1, lambda value: value + 1, invalidates=("range",)),
            PassStage("identity", 3, lambda value: value),
            PassStage("double", 1, lambda value: value * 2, requires=("plus_one",)),
        ),
        fingerprint=fingerprint,
    )
    first = manager.run(4)
    second = manager.run(4)
    assert first == second
    assert first.value == 10
    assert len(first.pipeline_identity) == 64
    assert [record.name for record in first.records] == [
        "plus_one",
        "identity",
        "double",
    ]
    assert [record.changed for record in first.records] == [True, False, True]
    assert first.records[0].invalidated_analyses == ("range",)
    assert first.records[1].invalidated_analyses == ()


def test_pass_manager_supports_prefix_bisection_and_explicit_disables() -> None:
    manager = PassManager(
        name="test.bisect",
        version=1,
        stages=(
            PassStage("a", 1, lambda value: value + "a"),
            PassStage("b", 1, lambda value: value + "b"),
            PassStage("c", 1, lambda value: value + "c"),
        ),
        fingerprint=lambda value: value or "empty",
    )
    prefix = manager.run("", stop_after="b")
    skipped = manager.run("", disabled=("b",))
    assert prefix.value == "ab"
    assert prefix.stopped_after == "b"
    assert skipped.value == "ac"
    assert skipped.disabled == ("b",)
    assert prefix.pipeline_identity != skipped.pipeline_identity


def test_pass_manager_fails_closed_for_invalid_pipeline_or_bisection() -> None:
    with pytest.raises(ValueError, match="unique"):
        PassManager(
            name="duplicate",
            version=1,
            stages=(
                PassStage("same", 1, lambda value: value),
                PassStage("same", 1, lambda value: value),
            ),
            fingerprint=fingerprint,
        )
    with pytest.raises(ValueError, match="non-prior"):
        PassManager(
            name="dependency",
            version=1,
            stages=(PassStage("later", 1, lambda value: value, requires=("missing",)),),
            fingerprint=fingerprint,
        )

    manager = PassManager(
        name="required",
        version=1,
        stages=(
            PassStage("prepare", 1, lambda value: value + 1),
            PassStage("consume", 1, lambda value: value * 2, requires=("prepare",)),
        ),
        fingerprint=fingerprint,
    )
    with pytest.raises(ValueError, match="requires disabled"):
        manager.run(2, disabled=("prepare",))
    with pytest.raises(ValueError, match="unknown disabled"):
        manager.run(2, disabled=("absent",))


def test_tensor_optimizer_exposes_bisection_controls_and_fingerprints() -> None:
    live = multiply(constant(2), constant(3))
    dead = multiply(constant(4), constant(5))
    program = Program({"value": live}, definitions=(dead,))

    diagnostic = optimize(
        program,
        disabled_passes=("dead_nodes",),
        stop_after="identity_transposes",
    )
    provenance = diagnostic.provenance
    diagnostics = provenance["optimizer_diagnostics"]
    assert diagnostics["disabled_passes"] == ["dead_nodes"]
    assert diagnostics["stopped_after"] == "identity_transposes"
    assert [item["name"] for item in provenance["optimizer_passes"]] == [
        "identity_transposes"
    ]
    record = provenance["optimizer_passes"][0]
    assert len(record["before"]) == 64
    assert len(record["after"]) == 64
    assert record["invalidated_analyses"] == []
    assert provenance["pruning_diagnostics"]["minimal_before_lowering"] is False
    assert diagnostic.definitions == (dead,)

    full = optimize(program)
    assert full.provenance["optimizer_identity"] != provenance["optimizer_identity"]
    assert full.provenance["pruning_diagnostics"]["minimal_before_lowering"] is True

    with pytest.raises(ValueError, match="unknown disabled"):
        optimize(program, disabled_passes=("missing_pass",))


def test_backend_preparation_preserves_noncanonical_bisection_provenance() -> None:
    dead = multiply(constant(4), constant(5))
    program = Program(
        {"value": multiply(constant(2), constant(3))},
        definitions=(dead,),
    )
    prepared = prepare_for_backend(
        program,
        "scalar",
        disabled_passes=("dead_nodes",),
        stop_after="identity_transposes",
    )
    production = prepared.provenance["production_preparation"]
    assert production["disabled_passes"] == ["dead_nodes"]
    assert production["stopped_after"] == "identity_transposes"
    assert prepared.definitions == (dead,)
    assert prepared.logical_hash == program.logical_hash
    assert prepared is not program


@pytest.mark.parametrize("backend", ["cpu", "cuda", "portable", "scalar"])
def test_diagnostic_selection_survives_repeated_backend_preparation(
    backend: str,
) -> None:
    program = Program({"value": multiply(constant(2), constant(3))})
    diagnostic = prepare_for_backend(program, backend, stop_after="identity_transposes")
    repeated = prepare_for_backend(diagnostic, backend)
    assert repeated.outputs["value"].op == "multiply"
    assert repeated.provenance["optimizer_diagnostics"]["stopped_after"] == (
        "identity_transposes"
    )
    assert (
        repeated.provenance["optimizer_identity"]
        == diagnostic.provenance["optimizer_identity"]
    )
    # Explicitly running optimize without controls restores canonical lowering.
    canonical = prepare_for_backend(optimize(diagnostic), backend)
    assert canonical.outputs["value"].op == "constant"
    assert canonical.provenance["optimizer_diagnostics"]["stopped_after"] is None


def test_diagnostic_selection_survives_actual_lowering_entrypoints() -> None:
    program = Program({"value": multiply(constant(2), constant(3))})
    diagnostic = prepare_for_backend(
        program, "cuda", disabled_passes=("scalar_constants",)
    )
    plan = plan_cuda(
        diagnostic, cuda_target_info("sm_80"), provider_bytes=0, library_bytes=0
    )
    assert plan.program.outputs["value"].op == "multiply"
    assert plan.program.provenance["optimizer_diagnostics"]["disabled_passes"] == [
        "scalar_constants"
    ]
    assert " * " in emit_scalar_cpp(diagnostic, function_name="diagnostic")
    assert " * " in emit_cpu(diagnostic)[0]


def test_backend_preparation_inherits_combined_controls_and_allows_override() -> None:
    program = Program({"value": multiply(constant(2), constant(3))})
    diagnostic = prepare_for_backend(
        program,
        "scalar",
        disabled_passes=("dead_nodes",),
        stop_after="identity_transposes",
    )
    inherited = prepare_for_backend(diagnostic, "scalar")
    assert inherited.provenance["optimizer_diagnostics"]["disabled_passes"] == [
        "dead_nodes"
    ]
    assert inherited.provenance["optimizer_diagnostics"]["stopped_after"] == (
        "identity_transposes"
    )
    overridden = prepare_for_backend(
        diagnostic, "scalar", disabled_passes=("scalar_constants",)
    )
    controls = overridden.provenance["optimizer_diagnostics"]
    assert controls["disabled_passes"] == ["scalar_constants"]
    assert controls["stopped_after"] is None


def test_tensor_optimizer_records_shared_pipeline_without_changing_equation() -> None:
    live = multiply(constant(2), constant(3))
    dead = multiply(constant(4), constant(5))
    program = Program({"value": live}, definitions=(dead,))
    optimized = optimize(program)
    provenance = optimized.provenance
    assert provenance["original_logical_hash"] == program.logical_hash
    assert (
        execute(optimized, {}).outputs["value"] == execute(program, {}).outputs["value"]
    )
    assert len(provenance["optimizer_identity"]) == 64
    assert [item["name"] for item in provenance["optimizer_passes"]] == [
        "dead_nodes",
        "identity_transposes",
        "view_canonicalization",
        "algebraic_canonicalization",
        "exact_cse",
        "scalar_constants",
        "post_fold_exact_cse",
        "post_fold_dead_nodes",
    ]
    assert provenance["rewrites"] == [
        "dead_nodes",
        "identity_transposes",
        "view_canonicalization",
        "algebraic_canonicalization",
        "exact_cse",
        "scalar_constants",
        "exact_cse",
        "dead_nodes",
    ]
    pruning = provenance["pruning_diagnostics"]
    assert pruning["schema"] == "generativeqc.compiler.pruning.v1"
    assert pruning["requested_outputs"] == ["value"]
    assert pruning["retained_outputs"] == ["value"]
    assert pruning["nodes_before"] > pruning["nodes_after"]
    assert pruning["nodes_removed"] == (
        pruning["nodes_before"] - pruning["nodes_after"]
    )
    assert pruning["definitions_before"] == 1
    assert pruning["definitions_after"] == 0
    assert pruning["definitions_removed"] == 1
    assert pruning["minimal_before_lowering"] is True
    assert optimized.nodes == optimized.live_nodes


def test_tensor_optimizer_projects_optional_diagnostics_before_dce() -> None:
    """Production roots keep final outputs; diagnostics survive only on demand."""

    i = Index("i", IndexSpace("ao", "batch", 4))
    spec = TensorSpec((i,), role="input")
    density = input_tensor("density", spec)
    diagnostic_source = input_tensor("stationary_weight_source", spec)
    force = multiply(density, density)
    diagnostic = multiply(density, diagnostic_source)
    program = Program(
        {
            "force": force,
            "stationary_weight_diagnostic": diagnostic,
        }
    )

    production = optimize(program, requested_outputs=("force",))
    pruning = production.provenance["pruning_diagnostics"]
    assert tuple(production.outputs) == ("force",)
    assert pruning["available_outputs"] == [
        "force",
        "stationary_weight_diagnostic",
    ]
    assert pruning["requested_outputs"] == ["force"]
    assert pruning["retained_outputs"] == ["force"]
    assert pruning["removed_outputs"] == ["stationary_weight_diagnostic"]
    assert pruning["inputs_before"] == ["density", "stationary_weight_source"]
    assert pruning["inputs_after"] == ["density"]
    assert pruning["removed_inputs"] == ["stationary_weight_source"]
    assert all(
        not (node.op == "input" and node.attrs["name"] == "stationary_weight_source")
        for node in production.nodes
    )

    diagnostic_build = optimize(
        program,
        requested_outputs=("force", "stationary_weight_diagnostic"),
    )
    diagnostic_pruning = diagnostic_build.provenance["pruning_diagnostics"]
    assert tuple(diagnostic_build.outputs) == (
        "force",
        "stationary_weight_diagnostic",
    )
    assert diagnostic_pruning["removed_outputs"] == []
    assert diagnostic_pruning["removed_inputs"] == []
    assert (
        diagnostic_build.provenance["optimizer_identity"]
        == (production.provenance["optimizer_identity"])
    )
    assert (
        diagnostic_build.provenance["specialized_logical_hash"]
        != (production.provenance["specialized_logical_hash"])
    )
    assert diagnostic_build.logical_hash != production.logical_hash


def test_tensor_output_specialization_fails_closed() -> None:
    program = Program({"value": multiply(constant(2), constant(3))})

    with pytest.raises(ValueError, match="at least one output"):
        optimize(program, requested_outputs=())
    with pytest.raises(ValueError, match="duplicate outputs"):
        optimize(program, requested_outputs=("value", "value"))
    with pytest.raises(ValueError, match="unknown outputs"):
        optimize(program, requested_outputs=("missing",))
    with pytest.raises(TypeError, match="sequence"):
        optimize(program, requested_outputs="value")
