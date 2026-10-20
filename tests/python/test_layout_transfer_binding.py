"""Executable layout-transfer obligations for joint provider selection."""

from __future__ import annotations

import shutil
import subprocess
import typing
from dataclasses import replace

import numpy as np
import pytest
from _cc_owner_test_support import compile_owner
from generativeqc_compiler.common.lowering_contract import (
    LayoutTransfer,
    LoweringConstraints,
)
from generativeqc_compiler.common.lowering_provider import (
    candidate_constraint_rejection,
    collect_lowering_candidates,
)
from generativeqc_compiler.common.lowering_selection import select_lowering_binding
from generativeqc_compiler.common.native_lowering import native_lowering_portfolio
from test_joint_lowering import (
    COMPILATION,
    GENERATED,
    TARGET,
    _offer,
    _Provider,
    _request,
)

if typing.TYPE_CHECKING:
    from pathlib import Path

    from generativeqc_compiler.common.lowering_provider import (
        LoweringCandidate,
        LoweringRequest,
    )
    from generativeqc_compiler.common.lowering_selection import LoweringBinding


def _packed(
    request: LoweringRequest | None = None, *, strides: tuple[int, int] = (1, 2)
) -> LoweringCandidate:
    request = request or _request()
    original = request.operands[0]
    physical = replace(original, strides=strides)
    transfer = LayoutTransfer(
        original,
        physical,
        "float64",
        "before",
        "test-affine-copy",
    )
    base = _offer(request)
    return replace(
        base,
        implementation="explicit-affine-copy",
        execution=replace(
            base.execution,
            layouts=(physical, *request.operands[1:]),
            layout_transfers=(transfer,),
            temporary_bytes=transfer.buffer_bytes,
        ),
        cost=replace(
            base.cost,
            kernel_ns=10,
            pack_ns=20,
            pack_bytes=transfer.read_bytes + transfer.write_bytes,
        ),
    )


def _select(candidate: LoweringCandidate) -> LoweringBinding:
    request = candidate.request
    return select_lowering_binding(
        request, TARGET, COMPILATION, (_offer(request), candidate)
    )


@pytest.mark.parametrize("constraints", [None, LoweringConstraints()])
def test_undeclared_physical_reinterpretation_is_rejected_and_retained(
    constraints: LoweringConstraints | None,
) -> None:
    request = _request(constraints=constraints)
    packed = _packed(request)
    bad = replace(
        packed,
        execution=replace(packed.execution, layout_transfers=(), temporary_bytes=0),
        cost=replace(packed.cost, pack_bytes=0, pack_ns=0),
    )
    binding = _select(bad)
    assert binding.selected == _offer(request)
    assert "physical layout change" in dict(binding.rejections)[bad.identity]
    collected = collect_lowering_candidates(
        request, TARGET, (_Provider(GENERATED, (bad,)),)
    )
    assert collected[0].status == "unsupported"
    assert collected[0].execution == bad.execution


def test_affine_copy_matches_independent_scalar_product_and_excludes_padding() -> None:
    packed = _packed(strides=(1, 4))
    transfer = packed.execution.layout_transfers[0]
    original = np.arange(6, dtype=np.float64).reshape(2, 3) + 0.25
    right = np.arange(6, dtype=np.float64).reshape(3, 2) / 7.0
    backing = np.full(transfer.buffer_bytes // 8, np.nan)
    physical = np.ndarray((2, 3), buffer=backing, dtype=np.float64, strides=(8, 32))
    np.copyto(physical, original)
    expected = np.array(
        [
            [
                sum(float(original[i, k]) * float(right[k, j]) for k in range(3))
                for j in range(2)
            ]
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(physical @ right, expected, rtol=1e-15, atol=1e-15)
    assert transfer.read_bytes == transfer.write_bytes == original.nbytes == 48
    assert transfer.buffer_bytes == 80
    assert np.isnan(backing[[2, 3, 6, 7]]).all()
    assert _select(packed).selected == packed


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("temporary_bytes", 79, "temporary bytes"),
        ("pack_bytes", 95, "transfer traffic"),
    ],
)
def test_layout_cost_cannot_omit_copy_traffic_or_padded_capacity(
    field: str, value: int, reason: str
) -> None:
    packed = _packed(strides=(1, 4))
    if field == "temporary_bytes":
        bad = replace(
            packed, execution=replace(packed.execution, temporary_bytes=value)
        )
    else:
        bad = replace(packed, cost=replace(packed.cost, pack_bytes=value))
    assert reason in candidate_constraint_rejection(bad)
    assert _select(bad).selected == _offer(bad.request)


def test_complete_copy_cost_and_exact_resource_limit_control_selection() -> None:
    packed = _packed(strides=(1, 4))
    assert _select(packed).selected == packed
    costly = replace(packed, cost=replace(packed.cost, pack_ns=100))
    assert _select(costly).selected == _offer(packed.request)
    unknown = replace(packed, cost=replace(packed.cost, pack_ns=None))
    assert _select(unknown).selected == _offer(packed.request)
    for allowance, selected in ((79, False), (80, True)):
        request = replace(
            packed.request,
            constraints=LoweringConstraints(additional_device_bytes=allowance),
        )
        candidate = replace(packed, request=request)
        assert (_select(candidate).selected == candidate) is selected


def test_read_write_requires_both_transfers_but_one_simultaneous_buffer() -> None:
    base = _request()
    request = replace(
        base,
        operands=(replace(base.operands[0], access="read-write"), *base.operands[1:]),
    )
    packed = _packed(request)
    before = packed.execution.layout_transfers[0]
    after = LayoutTransfer(
        before.destination, before.source, "float64", "after", "test-publication-copy"
    )
    assert "exact explicit" in candidate_constraint_rejection(packed)
    complete = replace(
        packed,
        execution=replace(packed.execution, layout_transfers=(before, after)),
        cost=replace(packed.cost, pack_bytes=192),
    )
    assert complete.execution.temporary_bytes == 48
    assert candidate_constraint_rejection(complete) is None
    assert _select(complete).selected == complete


def test_changed_public_output_needs_reverse_publication() -> None:
    request = _request()
    base = _offer(request)
    output = request.operands[-1]
    physical = replace(output, strides=(1, 2))
    after = LayoutTransfer(physical, output, "float64", "after", "test-publication")
    candidate = replace(
        base,
        implementation="packed-output",
        execution=replace(
            base.execution,
            layouts=(*request.operands[:-1], physical),
            layout_transfers=(after,),
            temporary_bytes=32,
        ),
        cost=replace(base.cost, kernel_ns=10, pack_ns=20, pack_bytes=64),
    )
    assert _select(candidate).selected == candidate
    wrong = LayoutTransfer(output, physical, "float64", "before", "wrong-direction")
    rejected = replace(
        candidate, execution=replace(candidate.execution, layout_transfers=(wrong,))
    )
    assert "exact explicit" in candidate_constraint_rejection(rejected)


@pytest.mark.parametrize("mutation", ["endpoints", "dtype", "cast", "unexpected"])
def test_transfer_cannot_invent_endpoint_dtype_or_cast_authority(mutation: str) -> None:
    packed = _packed()
    transfer = packed.execution.layout_transfers[0]
    if mutation == "endpoints":
        transfer = replace(
            transfer, destination=replace(transfer.destination, strides=(1, 4))
        )
    elif mutation == "dtype":
        transfer = replace(transfer, dtype="float32")
    elif mutation == "cast":
        mixed = _offer(packed.request, dtype="float32")
        packed = replace(
            packed,
            execution=replace(packed.execution, precision=mixed.execution.precision),
            cost=mixed.cost,
        )
    else:
        packed = replace(
            packed, execution=replace(packed.execution, layouts=packed.request.operands)
        )
    rejected = replace(
        packed, execution=replace(packed.execution, layout_transfers=(transfer,))
    )
    assert candidate_constraint_rejection(rejected) is not None


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"strides": None}, "concrete affine"),
        ({"alias_group": "shared"}, "disjoint"),
        ({"triangle": "upper"}, "full ownership"),
    ],
)
def test_transfer_refuses_unproven_ownership_and_virtual_values(
    change: dict[str, typing.Any], reason: str
) -> None:
    source = replace(_request().operands[0], **change)
    destination = replace(source, strides=(1, 2))
    with pytest.raises(ValueError, match=reason):
        LayoutTransfer(source, destination, "float64", "before", "copy")


def test_overlapping_destination_duplicate_boundary_and_unbound_stage_fail_closed() -> (
    None
):
    packed = _packed()
    transfer = packed.execution.layout_transfers[0]
    with pytest.raises(ValueError, match="overlap"):
        replace(transfer, destination=replace(transfer.destination, strides=(0, 1)))
    with pytest.raises(ValueError, match="duplicate"):
        replace(packed.execution, layout_transfers=(transfer, transfer))
    with pytest.raises(ValueError, match="stage"):
        replace(transfer, stage="iterate")


def test_binding_identity_tracks_copy_implementation_but_not_latency() -> None:
    packed = _packed()
    binding = _select(packed)
    measured = replace(packed, cost=replace(packed.cost, pack_ns=21))
    assert _select(measured).cache_identity == binding.cache_identity
    transfer = replace(packed.execution.layout_transfers[0], implementation="copy-v2")
    changed = replace(
        packed, execution=replace(packed.execution, layout_transfers=(transfer,))
    )
    assert _select(changed).cache_identity != binding.cache_identity
    assert changed.request.semantic_identity == packed.request.semantic_identity


def test_native_owner_without_copy_executor_keeps_strict_fallback(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    packed = _packed()
    strict = _offer(packed.request)
    undeclared = replace(
        packed,
        implementation="missing-copy",
        execution=replace(packed.execution, layout_transfers=(), temporary_bytes=0),
    )
    declarations = native_lowering_portfolio(
        packed.request, (strict, packed, undeclared), TARGET, COMPILATION, name="copies"
    )
    source = tmp_path / "copies.cpp"
    source.write_text(
        '#include "runtime/lowering_binding.hpp"\n'
        "using namespace generativeqc::runtime;\n" + declarations + "\nint main(){"
        "auto result=select_native_lowering(copies_request,copies_candidates,"
        "copies_target,copies_compilation);"
        "if(result.selected!=0 || result.rejections.size()!=2) return 1;"
        'if(copies_candidates[1].rejection.find("no layout-transfer executor")==std::string_view::npos) return 2;'
        "return 0;}\n"
    )
    binary = tmp_path / "copies"
    compile_owner(compiler, tmp_path, [source], binary)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_independent_input_copies_charge_simultaneous_storage_and_traffic() -> None:
    first = _packed()
    request = first.request
    old = request.operands[1]
    new = replace(old, strides=(1, 3))
    second = LayoutTransfer(old, new, "float64", "before", "copy-second")
    candidate = replace(
        first,
        execution=replace(
            first.execution,
            layouts=(first.execution.layouts[0], new, request.operands[2]),
            layout_transfers=(*first.execution.layout_transfers, second),
            temporary_bytes=96,
        ),
        cost=replace(first.cost, pack_bytes=192),
    )
    assert candidate_constraint_rejection(candidate) is None
    assert "temporary bytes" in candidate_constraint_rejection(
        replace(candidate, execution=replace(candidate.execution, temporary_bytes=95))
    )


def test_overlapping_execution_output_cannot_be_a_publication_source() -> None:
    output = _request().operands[2]
    with pytest.raises(ValueError, match="overlap"):
        LayoutTransfer(
            replace(output, strides=(0, 1)), output, "float64", "after", "copy-out"
        )
