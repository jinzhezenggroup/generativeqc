"""Shared arithmetic admission, retained schedules and frozen production TUs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc_compiler.tensor.ir import Node, add, divide, input_tensor, multiply
from generativeqc_compiler.tensor.ordered_history import (
    chronological_slots,
    ordered_history_program,
    recognize_ordered_history,
    require_history_schedule,
    retained_history_schedule,
)
from generativeqc_compiler.tensor.ordered_history_emit import (
    CorrectionBindings,
    CudaFlagFailure,
    ReturnFalseFailure,
    ReturnStatusFailure,
    emit_cholesky,
    emit_correction,
    emit_dot_loop,
)
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp_statement
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "src/xtb/native/src"


def _scalar(name: str) -> Node:
    return input_tensor(name, TensorSpec((), dtype="float64", role="input"))


def test_frozen_cpu_body_and_cuda_diagnostic_source_identity() -> None:
    # Keep the pre-extraction CPU source and its original frozen hash as an
    # independent reference. Only names and caller-status encoding may differ
    # in the relocated numerical body; algebra and operation order cannot.
    from generativeqc_compiler.method.gfn2_history_lowering import (
        emit_gfn2_history_artifacts,
        gfn2_history_correction_bindings,
        gfn2_history_gram_bindings,
    )
    from generativeqc_compiler.tensor.broyden_cpu_lowering import (
        emit_broyden_cpu_artifacts,
    )
    from generativeqc_compiler.tensor.ordered_history_artifacts import (
        emit_ordered_history_artifacts,
    )

    frozen = ROOT / "tests/native/fixtures/johnson_prechange_3b97c234"
    old = (frozen / "model/common/scc_mixer.cpp").read_text()
    for path in sorted(frozen.glob("generated_gfn2_history_cpu_*.inc")):
        old = old.replace(f'#include "{path.name}"', path.read_text())
    assert hashlib.sha256(old.encode()).hexdigest() == (
        "844ad997749c33b55bdc8d568c7dd985bfa262f661b1284c69cec8d86c88983d"
    )
    source = (ROOT / "src/solver/cpu/johnson_broyden.cpp").read_text()
    for name, body in emit_broyden_cpu_artifacts().items():
        if name.endswith(".inc"):
            include = f'#include "{name}"'
            assert source.count(include) == 1
            source = source.replace(include, body)

    def numerical_body(text: str) -> str:
        begin = text.index("std::size_t system_index(")
        end = text.index("\n}  // namespace", begin)
        return text[begin:end].replace("// clang-format on\n", "")

    old_body = numerical_body(old)
    for before, after in (
        ("SccMixer", "Broyden"),
        ("generativeqc_xtb_status_t", "BroydenResult"),
        ("GENERATIVEQC_XTB_STATUS_SUCCESS", "BroydenResult::success"),
        ("GENERATIVEQC_XTB_STATUS_INTERNAL_ERROR", "BroydenResult::numerical_failure"),
        (
            "state.system_statuses[system] = BroydenResult::success;",
            "state.system_statuses[system] = state.status_encoding.success;",
        ),
        (
            "state.system_statuses[system] = BroydenResult::numerical_failure;",
            "state.system_statuses[system] = state.status_encoding.numerical_failure;",
        ),
    ):
        old_body = old_body.replace(before, after)
    assert numerical_body(source) == old_body

    baseline = emit_ordered_history_artifacts(
        "cuda",
        source_prefix="generated_gfn2_history_cuda",
        gram=replace(
            gfn2_history_gram_bindings("cuda"),
            coefficient_visit_counter=None,
            overlap_visit_counter=None,
        ),
        correction=replace(
            gfn2_history_correction_bindings("cuda"), visit_counter=None
        ),
    )
    assert (
        json.loads(baseline["generated_gfn2_history_cuda_identity.json"])[
            "source_identity"
        ]
        == "f244c11c54c4124254f69177237a3a1dcccc09580f6a55c7e3f24b707092cf09"
    )

    cuda = (NATIVE / "backends/cuda/gfn2_scc_mixer.cu").read_text()
    for name, body in emit_gfn2_history_artifacts("cuda").items():
        if name.endswith(".inc"):
            include = f'#include "{name}"'
            assert cuda.count(include) == 1
            cuda = cuda.replace(include, body)
    assert hashlib.sha256(cuda.encode()).hexdigest() == (
        "b940a59232a672d5b3a81857e5b491a23761e556f4d8fa4311ef536cb0cdb36c"
    )


@pytest.mark.parametrize(
    "stage", [name for name, _ in ordered_history_program().stages]
)
def test_every_stage_mutation_rejects_retained_emission(stage: str) -> None:
    p = ordered_history_program()
    changed = replace(
        p,
        stages=tuple(
            (name, Program({name: _scalar("wrong")}))
            if name == stage
            else (name, graph)
            for name, graph in p.stages
        ),
    )
    assert changed.identity != p.identity
    with pytest.raises(ValueError, match="canonical"):
        recognize_ordered_history(changed)
    with pytest.raises(ValueError, match="canonical"):
        emit_cholesky(changed, retained_history_schedule(p, "compact-cpu"))


@pytest.mark.parametrize(
    "field",
    ("ring", "dot", "gram", "solve", "solve_checks", "correction", "publication"),
)
def test_recurrence_and_finite_contract_mutations_are_not_diagnostics_only(
    field: str,
) -> None:
    p = ordered_history_program()
    changed = replace(p, **{field: "different"})
    with pytest.raises(ValueError, match="canonical"):
        retained_history_schedule(changed, "capacity-cuda")


@pytest.mark.parametrize(
    "field,value",
    (
        ("leading_dimension", "dimension"),
        ("dot_failure", "return"),
        ("weight_failure", "short-circuit"),
        ("arithmetic", "forced-unfused"),
        ("final_store", "check-before-store"),
        ("program_identity", "0" * 64),
    ),
)
def test_forged_cuda_schedule_is_rejected(field: str, value: str) -> None:
    p = ordered_history_program()
    wrong = replace(retained_history_schedule(p, "capacity-cuda"), **{field: value})
    with pytest.raises(ValueError, match="admitted"):
        require_history_schedule(p, wrong)


def test_unsupported_provider_does_not_silently_select_incumbent() -> None:
    with pytest.raises(ValueError, match="unqualified"):
        retained_history_schedule(ordered_history_program(), "cublas")


def test_parenthesization_seed_sign_and_compound_assignment_proofs() -> None:
    a, b, c = (_scalar(n) for n in ("a", "b", "c"))
    bindings = {"a": "value", "b": "left[i]", "c": "right[i]"}
    emit = lambda root, form="=": emit_scalar_cpp_statement(
        Program({"out": root}), bindings=bindings, target="value", form=form
    )
    assert emit(multiply(multiply(a, b), c)) == "value = value * left[i] * right[i];"
    assert emit(multiply(a, multiply(b, c))) == "value = value * (left[i] * right[i]);"
    assert (
        emit(add(a, multiply(b, c), coefficients=(1, -1)), "-=")
        == "value -= left[i] * right[i];"
    )
    assert emit(add(a, multiply(b, c)), "+=") == "value += left[i] * right[i];"
    assert emit(divide(a, multiply(b, c)), "/=") == "value /= left[i] * right[i];"
    assert emit(add(a, add(b, c))) == "value = value + (left[i] + right[i]);"
    with pytest.raises(ValueError, match="target seed"):
        emit(add(multiply(b, c), a), "+=")
    with pytest.raises(ValueError, match="target seed"):
        emit(add(a, multiply(b, c)), "-=")


@pytest.mark.parametrize(
    "bad",
    (
        "value + other",
        "value++",
        "value;evil()",
        "f(i=2)",
        "f(++i)",
        "a[broken",
        "std::sqrt(left * right)",
        "left[(side_effect(), i)]",
        "read(throw 42)",
        "side_effect()",
        "a[0]b",
        "a[f(i)]",
        "a[i, j]",
        "a[(i + j)]",
    ),
)
def test_physical_binding_cannot_smuggle_arithmetic_or_effects(bad: str) -> None:
    with pytest.raises(ValueError):
        emit_scalar_cpp_statement(
            Program({"out": _scalar("a")}), bindings={"a": bad}, target="value"
        )


@pytest.mark.parametrize(
    "read",
    (
        "value",
        "state.omega",
        "matrix[row * leading_dimension + column]",
        "workspace.coefficients[coefficient_begin + history]",
        "matrix[row + 1u]",
    ),
)
def test_closed_physical_read_grammar_preserves_supported_spellings(read: str) -> None:
    assert (
        emit_scalar_cpp_statement(
            Program({"out": _scalar("a")}), bindings={"a": read}, target="out"
        )
        == f"out = {read};"
    )


def test_only_explicit_one_index_accessors_can_be_scalar_leaves() -> None:
    p = Program({"out": _scalar("a")})
    for read, accessor in (
        ("u_vector(slot)[component]", "u_vector"),
        ("slot_omega(slot)", "slot_omega"),
    ):
        with pytest.raises(ValueError, match="declared"):
            emit_scalar_cpp_statement(p, bindings={"a": read}, target="out")
        assert (
            emit_scalar_cpp_statement(
                p, bindings={"a": read}, target="out", read_accessors=(accessor,)
            )
            == f"out = {read};"
        )
    for bad in (
        "u_vector(slot + 1)[component]",
        "u_vector(side_effect())[component]",
        "u_vector(slot, extra)[component]",
    ):
        with pytest.raises(ValueError, match="declared"):
            emit_scalar_cpp_statement(
                p, bindings={"a": bad}, target="out", read_accessors=("u_vector",)
            )
    with pytest.raises(ValueError):
        emit_scalar_cpp_statement(
            p,
            bindings={"a": "value"},
            target="u_vector(slot)[component]",
            read_accessors=("u_vector",),
        )
    with pytest.raises(ValueError, match="identifier"):
        emit_scalar_cpp_statement(
            p,
            bindings={"a": "std::sqrt(left)"},
            target="out",
            read_accessors=("std::sqrt",),
        )


def _cuda_correction_bindings() -> CorrectionBindings:
    return CorrectionBindings(
        current="x[index]",
        damping="alpha",
        residual="f[index]",
        coefficient="coefficients[coefficient_begin + history]",
        mixed="mixed[index]",
        capacity="capacity",
        tentative_weight="tentative_weight",
        weights="weights",
        tentative_vectors="tentative_vectors",
        vectors="vectors",
        failure=CudaFlagFailure(
            "record_error", "device_error", "Error::nonfinite", "valid"
        ),
    )


@pytest.mark.parametrize(
    "field",
    (
        "current",
        "damping",
        "residual",
        "coefficient",
        "mixed",
        "capacity",
        "tentative_weight",
        "weights",
        "tentative_vectors",
        "vectors",
    ),
)
@pytest.mark.parametrize(
    "bad", ("std::sqrt(left * right)", "a[(side_effect(), i)]", "a[0]b")
)
def test_all_cuda_physical_fields_use_the_same_admission(field: str, bad: str) -> None:
    p = ordered_history_program()
    bad_bindings = replace(_cuda_correction_bindings(), **{field: bad})
    with pytest.raises(ValueError, match="declared"):
        emit_correction(p, retained_history_schedule(p, "capacity-cuda"), bad_bindings)


@pytest.mark.parametrize(
    "field", ("history_slots", "weight_accessor", "vector_accessor")
)
@pytest.mark.parametrize(
    "bad", ("std::sqrt(left * right)", "a[(side_effect(), i)]", "a[0]b")
)
def test_cpu_slot_and_accessor_bindings_are_closed_reads(field: str, bad: str) -> None:
    p = ordered_history_program()
    bindings = CorrectionBindings(
        current="x[component]",
        damping="alpha",
        residual="f[component]",
        coefficient="coefficients[history]",
        mixed="mixed[component]",
        history_slots="slots",
        weight_accessor="slot_omega",
        vector_accessor="u_vector",
        failure=ReturnStatusFailure("fail", "state", "system", "failed", "error"),
    )
    with pytest.raises(ValueError):
        emit_correction(
            p,
            retained_history_schedule(p, "compact-cpu"),
            replace(bindings, **{field: bad}),
        )


@pytest.mark.parametrize(
    "bad",
    (
        (),
        ("break;",),
        ("return false;",),
        ReturnFalseFailure(),
        ReturnStatusFailure("fail", "state", "system", "failed", "error"),
    ),
)
def test_cuda_dot_and_correction_cannot_omit_error_or_invalidation(bad: object) -> None:
    p = ordered_history_program()
    schedule = retained_history_schedule(p, "capacity-cuda")
    with pytest.raises(ValueError, match="failure effect"):
        emit_dot_loop(
            p,
            schedule,
            accumulator="sum",
            left="a[i]",
            right="b[i]",
            index="i",
            extent="n",
            failure=bad,
            indent=0,
        )
    with pytest.raises(ValueError, match="failure effect"):
        emit_correction(p, schedule, replace(_cuda_correction_bindings(), failure=bad))


@pytest.mark.parametrize("field", ("function", "error_buffer", "code", "valid"))
def test_cuda_failure_binding_rejects_missing_or_executable_fields(field: str) -> None:
    effect = CudaFlagFailure(
        "record_error", "device_error", "Error::nonfinite", "valid"
    )
    for bad in ("", "side_effect()", "valid; return;"):
        with pytest.raises(ValueError):
            replace(effect, **{field: bad})


def test_failure_control_flow_is_owned_by_the_shared_phase() -> None:
    p = ordered_history_program()
    cuda = retained_history_schedule(p, "capacity-cuda")
    bindings = _cuda_correction_bindings()
    dot = emit_dot_loop(
        p,
        cuda,
        accumulator="sum",
        left="a[i]",
        right="b[i]",
        index="i",
        extent="n",
        failure=bindings.failure,
        indent=0,
    )
    correction = emit_correction(p, cuda, bindings)
    for source in (dot, correction):
        assert source.index(
            "record_error(device_error, Error::nonfinite);"
        ) < source.index("atomicExch(&valid, 0);")
        assert "return" not in source
    assert dot.index("atomicExch(&valid, 0);") < dot.index("break;")
    assert "break;" not in correction
    cpu = retained_history_schedule(p, "compact-cpu")
    with pytest.raises(ValueError, match="failure effect"):
        emit_dot_loop(
            p,
            cpu,
            accumulator="sum",
            left="a[i]",
            right="b[i]",
            index="i",
            extent="n",
            failure=bindings.failure,
            indent=0,
        )
    with pytest.raises(ValueError, match="failure effect"):
        emit_correction(p, cpu, replace(bindings, failure=ReturnFalseFailure()))


def test_unknown_bindings_precision_and_shape_are_rejected() -> None:
    for spec in (
        TensorSpec((), dtype="float32", role="input"),
        TensorSpec(
            (Index("i", IndexSpace("components", "component", 2)),), role="input"
        ),
    ):
        with pytest.raises(ValueError, match="float64"):
            emit_scalar_cpp_statement(
                Program({"out": input_tensor("a", spec)}),
                bindings={"a": "a"},
                target="out",
            )
    with pytest.raises(ValueError, match="every input"):
        emit_scalar_cpp_statement(
            Program({"out": _scalar("a")}), bindings={"a": "a", "b": "b"}, target="out"
        )


def test_builder_consumes_scalar_lowering_not_cached_backend_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.tensor import ordered_history_emit as emitter

    p = ordered_history_program()
    seen = []
    original = emitter.emit_scalar_cpp_statement

    def marked(program: Program, **kwargs: object) -> str:
        seen.append(program.logical_hash)
        return "/* scalar-stage */ " + original(program, **kwargs)

    monkeypatch.setattr(emitter, "emit_scalar_cpp_statement", marked)
    for backend in ("compact-cpu", "capacity-cuda"):
        source = emit_cholesky(p, retained_history_schedule(p, backend))
        assert source.count("/* scalar-stage */") == 7
    assert {
        p.stage(n).logical_hash for n in ("subtract_product", "divide", "sqrt")
    } <= set(seen)


@pytest.mark.parametrize("capacity", (1, 2, 4, 64, 65))
def test_chronology_uses_the_latest_tentative_slot_without_method_limit(
    capacity: int,
) -> None:
    p = ordered_history_program()
    assert p.identity == ordered_history_program().identity
    for iteration in (1, capacity, capacity + 1, 3 * capacity + 2, 2**64 - 2):
        new, slots = chronological_slots(iteration, capacity)
        expected_iterations = range(max(1, iteration - capacity + 1), iteration + 1)
        assert slots == tuple((j - 1) % capacity for j in expected_iterations)
        assert slots[-1] == new and len(set(slots)) == len(slots)


@pytest.mark.parametrize(
    "iteration,capacity", ((0, 4), (1, 0), (2**64 - 1, 4), (True, 4), (2, -1))
)
def test_invalid_ring_admission(iteration: int, capacity: int) -> None:
    with pytest.raises(ValueError):
        chronological_slots(iteration, capacity)
